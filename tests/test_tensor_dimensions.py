import unittest
import torch
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.mind.mind import Mind
from utils.config import NeuralChildConfig, DEFAULT_NEURAL_CHILD_CONFIG


# Mock embedding response
MOCK_EMBEDDING = np.random.randn(384).astype(np.float32).tolist()
MOCK_EMBEDDING_RESPONSE = {
    "data": [
        {
            "embedding": MOCK_EMBEDDING,
            "index": 0,
            "object": "embedding"
        }
    ],
    "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
    "object": "list",
    "usage": {"prompt_tokens": 5, "total_tokens": 5}
}

# Mock mother response
MOCK_MOTHER_RESPONSE = {
    "verbal_response": "Hello my sweet child!",
    "emotional_state": {
        "joy": 0.8,
        "trust": 0.9,
        "anticipation": 0.4
    },
    "teaching_elements": [
        {
            "concept": "greeting",
            "examples": ["hello", "hi"],
            "explanation": "We say hello to greet people"
        }
    ]
}


class TestTensorDimensions(unittest.TestCase):
    """Tests to ensure tensor dimensions remain compatible across components."""
    
    def setUp(self):
        # Use CPU for testing
        self.device = "cpu"
        torch.set_default_device(self.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # Setup LLM client mock patcher
        self.llm_client_patcher = patch('llm_module.LLMClient')
        self.mock_llm_client = self.llm_client_patcher.start()
        
        # Configure the mock client
        mock_instance = self.mock_llm_client.return_value
        mock_instance.chat_completion.return_value = "Mock child response."
        mock_instance.structured_completion.return_value = MOCK_MOTHER_RESPONSE
        
        # Setup embedding API mock
        self.embedding_patcher = patch('requests.post')
        self.mock_post = self.embedding_patcher.start()
        
        # Configure the mock response
        self.mock_post.return_value = MagicMock(
            json=lambda: MOCK_EMBEDDING_RESPONSE,
            status_code=200,
            raise_for_status=lambda: None
        )
        
        # Initialize mind
        self.mind = Mind(
            config=DEFAULT_NEURAL_CHILD_CONFIG,
            device=self.device,
            base_path=self.base_path,
            load_existing=False
        )
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Stop all patchers
        self.llm_client_patcher.stop()
        self.embedding_patcher.stop()
    
    def test_cognitive_network_dimensions(self):
        """Test cognitive network input/output dimensions."""
        # Get input/output dimensions
        input_size = self.mind.cognitive_component.input_size
        output_size = self.mind.cognitive_component.output_size
        
        # Create random input tensor of correct shape
        input_tensor = torch.randn(1, input_size, device=self.device)
        
        # Forward pass through the network
        with torch.no_grad():
            output_tensor = self.mind.cognitive_component.forward(input_tensor)
        
        # Check output dimensions
        self.assertEqual(output_tensor.shape, (1, output_size))
    
    def test_emotional_network_dimensions(self):
        """Test emotional network input/output dimensions."""
        # Get input/output dimensions
        input_size = self.mind.emotional_component.input_size
        output_size = self.mind.emotional_component.output_size
        
        # Create random input tensor of correct shape
        input_tensor = torch.randn(1, input_size, device=self.device)
        
        # Forward pass through the network
        with torch.no_grad():
            output_tensor = self.mind.emotional_component.forward(input_tensor)
        
        # Check output dimensions
        self.assertEqual(output_tensor.shape, (1, output_size))
    
    def test_memory_embedding_dimensions(self):
        """Test memory embedding dimensions."""
        # Create a test memory
        test_memory = {
            "type": "episodic",
            "description": "Test memory",
            "context": {"location": "test"}
        }
        
        # Generate embedding
        embedding = self.mind.memory_component._create_memory_embedding(test_memory)
        
        # Check dimension
        self.assertEqual(len(embedding), self.mind.memory_component.embedding_dimension)
    
    def test_cross_component_forward_compatibility(self):
        """Test that outputs from one component can be used as inputs to another."""
        # Get cognitive component output
        cognitive_input = torch.randn(1, self.mind.cognitive_component.input_size, device=self.device)
        with torch.no_grad():
            cognitive_output = self.mind.cognitive_component.forward(cognitive_input)
        
        # Check if cognitive output matches emotional input dimensions
        self.assertEqual(cognitive_output.shape[1], self.mind.emotional_component.input_size)
        
        # Forward pass through emotional network with cognitive output
        with torch.no_grad():
            emotional_output = self.mind.emotional_component.forward(cognitive_output)
        
        # Check output dimensions
        self.assertEqual(emotional_output.shape, (1, self.mind.emotional_component.output_size))
    
    def test_training_dimensions(self):
        """Test that training handles dimensions correctly."""
        # Create random input/target tensors
        input_tensor = torch.randn(1, self.mind.cognitive_component.input_size, device=self.device)
        target_tensor = torch.randn(1, self.mind.cognitive_component.output_size, device=self.device)
        
        # Train component
        loss = self.mind.cognitive_component.train_component(
            input_tensor, target_tensor, learning_rate=0.01
        )
        
        # Verify training completed without dimension errors
        self.assertIsNotNone(loss)
    
    def test_mind_parameter_dimensions(self):
        """Test the dimensions of all parameters in the Mind's neural components."""
        # Loop through all components
        for name, component in self.mind.components.items():
            if hasattr(component, 'parameters'):
                # Check each parameter's dimensions
                for param_name, param in component.named_parameters():
                    # Verify parameter is on the expected device
                    self.assertEqual(param.device.type, self.device)
                    
                    # For specific known parameters, verify their dimensions
                    if name == "cognitive" and param_name == "network.0.weight":
                        # First layer weight matrix in cognitive component
                        self.assertEqual(
                            param.shape, 
                            (self.mind.cognitive_component.hidden_size, 
                             self.mind.cognitive_component.input_size)
                        )
                    
                    # Just ensure all parameters have some valid dimensions
                    self.assertGreater(param.dim(), 0)


if __name__ == "__main__":
    unittest.main()