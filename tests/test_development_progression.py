import unittest
import torch
import numpy as np
import os
import sys
import time
from pathlib import Path
import tempfile

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.mind.mind import Mind
from neural_child.development.development_component import DevelopmentComponent
from utils.config import NeuralChildConfig, DEFAULT_NEURAL_CHILD_CONFIG


class TestDevelopmentProgression(unittest.TestCase):
    """Tests for the developmental progression of the neural child."""
    
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
        
        # Initialize mind with accelerated development
        self.config = DEFAULT_NEURAL_CHILD_CONFIG.model_copy(update={
            "initial_age_months": 0.0,  # Start from birth
            "development_speed": 100.0  # Very fast development for testing
        })
        
        self.mind = Mind(
            config=self.config,
            device=self.device,
            base_path=self.base_path,
            load_existing=False
        )
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
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
        # Get initial metrics
        initial_status = self.mind.get_status()
        initial_metrics = initial_status["developmental_metrics"]
        
        # Extract key metrics to monitor
        initial_receptive = initial_metrics["language"].get("receptive_language", 0.0)
        initial_emotional = initial_metrics["emotional"].get("basic_emotions", 0.0)
        initial_cognitive = initial_metrics["cognitive"].get("attention", 0.0)
        initial_social = initial_metrics["social"].get("attachment", 0.0)
        
        # Process many interactions to trigger development
        for _ in range(30):
            self.mind.interact_with_mother()
        
        # Get updated metrics
        updated_status = self.mind.get_status()
        updated_metrics = updated_status["developmental_metrics"]
        
        # Extract updated key metrics
        updated_receptive = updated_metrics["language"].get("receptive_language", 0.0)
        updated_emotional = updated_metrics["emotional"].get("basic_emotions", 0.0)
        updated_cognitive = updated_metrics["cognitive"].get("attention", 0.0)
        updated_social = updated_metrics["social"].get("attachment", 0.0)
        
        # Verify metrics increased
        self.assertGreater(updated_receptive, initial_receptive)
        self.assertGreater(updated_emotional, initial_emotional)
        self.assertGreater(updated_cognitive, initial_cognitive)
        self.assertGreater(updated_social, initial_social)
    
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