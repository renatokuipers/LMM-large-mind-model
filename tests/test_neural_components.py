import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.cognition.cognitive_component import CognitiveComponent
from neural_child.emotion.emotional_component import EmotionalComponent
from neural_child.language.language_component import LanguageComponent
from neural_child.memory.memory_component import MemoryComponent
from neural_child.social.social_component import SocialComponent


# Mock data for language component
MOCK_LANGUAGE_OUTPUT = {
    "child_utterance": "ma-ma",
    "comprehension": 0.35,
    "vocabulary_size": 10,
    "language_development": {
        "verbal_production": 0.2,
        "verbal_comprehension": 0.4,
        "phonological_awareness": 0.15
    }
}


class TestNeuralComponents(unittest.TestCase):
    
    def setUp(self):
        # Set device to CPU for consistent testing
        self.device = "cpu"
        torch.set_default_device(self.device)
        
        # Initialize components with consistent random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Setup LLM client mock patcher
        self.llm_client_patcher = patch('llm_module.LLMClient')
        self.mock_llm_client = self.llm_client_patcher.start()
        
        # Configure the mock client
        mock_instance = self.mock_llm_client.return_value
        mock_instance.chat_completion.return_value = "Mock language response for child."
        mock_instance.structured_completion.return_value = MOCK_LANGUAGE_OUTPUT
        
        # Setup embedding API mocks
        self.embedding_patcher = patch('requests.post')
        self.mock_post = self.embedding_patcher.start()
        
        # Configure embedding API mock to return a consistent embedding
        mock_embedding = np.random.randn(384).astype(np.float32).tolist()
        mock_response = {
            "data": [
                {
                    "embedding": mock_embedding,
                    "index": 0,
                    "object": "embedding"
                }
            ],
            "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        
        self.mock_post.return_value = MagicMock(
            json=lambda: mock_response,
            status_code=200,
            raise_for_status=lambda: None
        )
        
        self.cognitive = CognitiveComponent(device=self.device)
        self.emotional = EmotionalComponent()
        self.language = LanguageComponent()
        self.memory = MemoryComponent()
        self.social = SocialComponent(device=self.device)
    
    def tearDown(self):
        # Stop all patchers
        self.llm_client_patcher.stop()
        self.embedding_patcher.stop()
    
    def test_component_initialization(self):
        """Test that all components initialize correctly with expected activation levels."""
        self.assertIsNotNone(self.cognitive)
        self.assertIsNotNone(self.emotional)
        self.assertIsNotNone(self.language)
        self.assertIsNotNone(self.memory)
        self.assertIsNotNone(self.social)
        
        # Check initial activation levels
        self.assertEqual(self.cognitive.activation_level, 0.0)
        self.assertEqual(self.emotional.activation_level, 0.0)
        self.assertEqual(self.language.activation_level, 0.0)
        self.assertEqual(self.memory.activation_level, 0.0)
        self.assertEqual(self.social.activation_level, 0.0)
    
    def test_cognitive_process(self):
        """Test cognitive component processing."""
        input_data = {
            "mother_utterance": "Hello little one",
            "emotional_state": {"joy": 0.7, "trust": 0.6},
            "complexity": 0.3
        }
        
        output = self.cognitive.process(input_data)
        
        # Verify output structure without being too rigid about exact values
        self.assertIn("understanding_level", output)
        self.assertIn("cognitive_response", output)
        self.assertIn("attention_level", output)
        self.assertGreaterEqual(output["understanding_level"], 0.0)
        self.assertLessEqual(output["understanding_level"], 1.0)
    
    def test_emotional_process(self):
        """Test emotional component processing."""
        input_data = {
            "mother_emotional_state": {"joy": 0.7, "trust": 0.6},
            "interaction_context": {"type": "nurturing", "valence": 0.8},
            "developmental_stage": "Infancy",
            "age_months": 6.0
        }
        
        output = self.emotional.process(input_data)
        
        # Verify output structure
        self.assertIn("emotional_state", output)
        self.assertIn("dominant_emotion", output)
        
        # Check that emotion values are normalized
        for emotion, value in output["emotional_state"].items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_language_process(self):
        """Test language component processing."""
        input_data = {
            "mother_utterance": "Say hello to mommy",
            "teaching_elements": [{"type": "vocabulary", "content": "hello", "importance": 0.8}],
            "developmental_stage": "Infancy",
            "age_months": 10.0,
            "emotional_state": {"joy": 0.7, "trust": 0.6}
        }
        
        output = self.language.process(input_data)
        
        # Verify output structure
        self.assertIn("child_utterance", output)
        self.assertIn("comprehension", output)
        self.assertIn("vocabulary_size", output)
        self.assertIn("language_development", output)
    
    def test_memory_process(self):
        """Test memory component processing."""
        input_data = {
            "current_experience": {
                "type": "interaction",
                "description": "Mother said hello",
                "context": {"location": "home"}
            },
            "emotional_state": {"joy": 0.7},
            "developmental_stage": "Infancy",
            "age_months": 8.0
        }
        
        output = self.memory.process(input_data)
        
        # Verify output structure
        self.assertIn("working_memory", output)
        self.assertIn("memory_development", output)
    
    def test_social_process(self):
        """Test social component processing."""
        input_data = {
            "agent": "mother",
            "content": "Hello sweetie",
            "emotional_tone": {"joy": 0.7, "trust": 0.6},
            "context": {"social_complexity": 0.3},
            "age_months": 7.0
        }
        
        output = self.social.process(input_data)
        
        # Verify output structure
        self.assertIn("social_response", output)
        self.assertIn("attachment_status", output)
        self.assertIn("social_development", output)
    
    def test_component_save_load(self):
        """Test that components can save and load their state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            
            # Process some data to change component state
            self.cognitive.process({
                "mother_utterance": "Hello little one",
                "emotional_state": {"joy": 0.7},
                "complexity": 0.3
            })
            
            # Save state
            self.cognitive.save(save_path)
            
            # Create a new component
            new_component = CognitiveComponent(device=self.device)
            
            # Verify different initial state
            self.assertNotEqual(
                self.cognitive.activation_level,
                new_component.activation_level
            )
            
            # Load state
            new_component.load(save_path)
            
            # Verify state was restored
            self.assertEqual(
                self.cognitive.activation_level,
                new_component.activation_level
            )
    
    def test_cross_component_compatibility(self):
        """Test compatibility between outputs and inputs of different components."""
        # Process through cognitive first
        cognitive_input = {
            "mother_utterance": "Hello little one",
            "emotional_state": {"joy": 0.7},
            "complexity": 0.3
        }
        cognitive_output = self.cognitive.process(cognitive_input)
        
        # Feed cognitive output to emotional component
        emotional_input = {
            "mother_emotional_state": {"joy": 0.7},
            "interaction_context": {"type": "nurturing"},
            "developmental_stage": "Infancy",
            "age_months": 6.0,
            "cognitive_output": cognitive_output  # Pass cognitive output to emotional
        }
        emotional_output = self.emotional.process(emotional_input)
        
        # Verify emotional component handled cognitive output
        self.assertIsNotNone(emotional_output)
        self.assertIn("emotional_state", emotional_output)
        
        # Now chain emotional output to language component
        language_input = {
            "mother_utterance": "Hello little one",
            "developmental_stage": "Infancy",
            "age_months": 6.0,
            "emotional_state": emotional_output["emotional_state"],
            "cognitive_output": cognitive_output
        }
        language_output = self.language.process(language_input)
        
        # Verify language component handled previous outputs
        self.assertIsNotNone(language_output)
        self.assertIn("child_utterance", language_output)


if __name__ == "__main__":
    unittest.main()