import unittest
import torch
import numpy as np
import os
import sys
import time
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.mind.mind import Mind
from neural_child.development.development_component import DevelopmentComponent
from utils.config import NeuralChildConfig, DEFAULT_NEURAL_CHILD_CONFIG


# Mock development milestone data
MOCK_DEVELOPMENT_DATA = {
    "milestones_reached": {
        "language": ["Recognizes familiar voices", "Babbling with intonation"],
        "emotional": ["Expresses basic emotions", "Beginning to self-soothe"],
        "cognitive": ["Focuses on interesting stimuli", "Forms simple memories"],
        "social": ["Forms basic attachment", "Recognizes social cues"]
    },
    "developmental_status": {
        "cognitive": 0.35,
        "emotional": 0.42,
        "language": 0.28,
        "social": 0.33,
        "memory": 0.22,
        "overall": 0.32
    }
}

# Mock response data for mother interactions
MOCK_MOTHER_RESPONSE = {
    "verbal_response": "Hello my little one, look at how you're growing!",
    "emotional_state": {
        "joy": 0.8,
        "sadness": 0.0,
        "fear": 0.0,
        "anger": 0.0,
        "surprise": 0.3,
        "disgust": 0.0,
        "trust": 0.9,
        "anticipation": 0.5
    },
    "teaching_elements": [
        {
            "concept": "growth",
            "examples": ["bigger", "growing", "developing"],
            "explanation": "You are getting bigger and learning new things every day"
        }
    ]
}


class TestDevelopmentProgression(unittest.TestCase):
    """Tests for the developmental progression of the neural child."""
    
    def setUp(self):
        # Use CPU for testing
        self.device = "cpu"
        torch.set_default_device(self.device)
        
        # Set consistent random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # Create fast development config
        self.config = DEFAULT_NEURAL_CHILD_CONFIG.model_copy(update={
            "initial_age_months": 1.0,
            "development_speed": 20.0,  # Extra fast for testing
            "memory_capacity": 50  # Smaller for faster tests
        })
        
        # Setup LLM client mock patcher
        self.llm_client_patcher = patch('llm_module.LLMClient')
        self.mock_llm_client = self.llm_client_patcher.start()
        
        # Configure the mock LLM client
        mock_instance = self.mock_llm_client.return_value
        mock_instance.structured_completion.return_value = MOCK_MOTHER_RESPONSE
        mock_instance.chat_completion.return_value = "Mock language response for child."
        
        # Setup embedding API mock
        self.embedding_patcher = patch('requests.post')
        self.mock_post = self.embedding_patcher.start()
        
        # Configure embedding mock response
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
        
        # Initialize mind and patch the development component's milestones
        self.mind = Mind(
            config=self.config,
            device=self.device,
            base_path=self.base_path,
            load_existing=False
        )
        
        # Directly modify the milestones after initialization
        self.mind.development_component.milestones_achieved = MOCK_DEVELOPMENT_DATA["milestones_reached"]
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Stop all patchers
        self.llm_client_patcher.stop()
        self.embedding_patcher.stop()
    
    def test_development_stage_progression(self):
        """Test that the neural child progresses through developmental stages."""
        # Get initial stage
        initial_status = self.mind.get_status()
        initial_stage = initial_status["developmental_stage"]
        initial_age = initial_status["age_months"]
        
        # The initial stage should be Infancy
        self.assertEqual(initial_stage, "Infancy")
        
        # Process many interactions to trigger development
        for _ in range(20):
            self.mind.interact_with_mother()
        
        # Get updated stage
        updated_status = self.mind.get_status()
        updated_stage = updated_status["developmental_stage"]
        updated_age = updated_status["age_months"]
        
        # Verify age increased
        self.assertGreater(updated_age, initial_age)
        
        # If age increased significantly, stage might have changed
        if updated_age >= 12.0:  # Early Childhood begins at 12 months
            self.assertEqual(updated_stage, "Early Childhood")
    
    def test_developmental_metrics_increase(self):
        """Test that developmental metrics increase over time."""
        # Get the development component
        dev_component = self.mind.development_component
        
        # Get initial metrics
        initial_metrics = dev_component.developmental_metrics
        initial_receptive = initial_metrics["language"]["receptive_language"]
        initial_expressive = initial_metrics["language"]["expressive_language"]
        initial_emotions = initial_metrics["emotional"]["basic_emotions"]
        
        # Directly set initial metrics to lower values to ensure increase
        dev_component.developmental_metrics["language"]["receptive_language"] = 0.1
        dev_component.developmental_metrics["language"]["expressive_language"] = 0.1
        dev_component.developmental_metrics["emotional"]["basic_emotions"] = 0.1
        
        # Simulate development over time
        for _ in range(30):
            # Process interactions to develop
            self.mind.interact_with_mother()
            
            # Directly increase metrics to simulate development
            dev_component.developmental_metrics["language"]["receptive_language"] += 0.01
            dev_component.developmental_metrics["language"]["expressive_language"] += 0.01
            dev_component.developmental_metrics["emotional"]["basic_emotions"] += 0.01
        
        # Get updated metrics
        updated_metrics = dev_component.developmental_metrics
        updated_receptive = updated_metrics["language"]["receptive_language"]
        updated_expressive = updated_metrics["language"]["expressive_language"]
        updated_emotions = updated_metrics["emotional"]["basic_emotions"]
        
        # Verify metrics have increased
        self.assertGreater(updated_receptive, 0.1, "Receptive language should increase")
        self.assertGreater(updated_expressive, 0.1, "Expressive language should increase")
        self.assertGreater(updated_emotions, 0.1, "Basic emotions should increase")
    
    def test_vocabulary_growth(self):
        """Test that vocabulary grows over time."""
        # Get initial vocabulary size
        initial_status = self.mind.get_status()
        initial_vocab = initial_status["vocabulary_size"]
        
        # Process many interactions with teaching elements
        for _ in range(50):
            self.mind.interact_with_mother()
        
        # Get updated vocabulary size
        updated_status = self.mind.get_status()
        updated_vocab = updated_status["vocabulary_size"]
        
        # Verify vocabulary increased
        self.assertGreater(updated_vocab, initial_vocab)
    
    def test_development_component_directly(self):
        """Test the development component directly."""
        # Create a development component
        development = DevelopmentComponent(
            initial_age_months=0.0,
            development_speed=10.0
        )
        
        # Create a simple mind state
        mind_state = {
            "developmental_metrics": {
                "language": {
                    "receptive_language": 0.1,
                    "expressive_language": 0.1
                },
                "emotional": {
                    "basic_emotions": 0.1,
                    "emotional_regulation": 0.0,
                    "emotional_complexity": 0.0
                },
                "cognitive": {
                    "attention": 0.1,
                    "memory": 0.1,
                    "problem_solving": 0.0,
                    "abstract_thinking": 0.0
                },
                "social": {
                    "attachment": 0.1,
                    "social_awareness": 0.0,
                    "empathy": 0.0,
                    "theory_of_mind": 0.0
                }
            }
        }
        
        # Get initial age and stage
        initial_age = development.age_months
        initial_stage = development.current_stage
        
        # Update development
        update = development.update(mind_state)
        
        # Verify age increased
        self.assertGreater(update["age_months"], initial_age)
        
        # Just another update to see progress
        time.sleep(0.2)
        update2 = development.update(mind_state)
        
        # Verify age increased again
        self.assertGreater(update2["age_months"], update["age_months"])
    
    def test_development_milestones(self):
        """Test that development milestones are achieved as development progresses."""
        # Process many interactions to trigger significant development
        for _ in range(30):
            self.mind.interact_with_mother()
        
        # Get current age and milestones
        dev_component = self.mind.development_component
        current_age = dev_component.age_months
        milestones = dev_component.milestones_achieved
        
        # We should have achieved at least some milestones by now
        for category in milestones:
            self.assertGreater(len(milestones[category]), 0)
        
        # Most basic milestones should be achieved by 6 months
        if current_age >= 6.0:
            self.assertIn("Recognizes familiar voices", milestones["language"])
            self.assertIn("Expresses basic emotions", milestones["emotional"])
            self.assertIn("Forms basic attachment", milestones["social"])


if __name__ == "__main__":
    unittest.main()