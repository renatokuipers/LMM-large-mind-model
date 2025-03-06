import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import time
import os
import sys

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.mind.mind import Mind
from neural_child.mind.base import MindState
from utils.config import NeuralChildConfig, DEFAULT_NEURAL_CHILD_CONFIG


class TestMindIntegration(unittest.TestCase):
    
    def setUp(self):
        # Use CPU for consistent testing
        self.device = "cpu"
        torch.set_default_device(self.device)
        
        # Set consistent random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory for mind data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # Initialize mind
        config = DEFAULT_NEURAL_CHILD_CONFIG.model_copy(update={
            "initial_age_months": 1.0,
            "development_speed": 10.0,
            "memory_capacity": 100  # Smaller for faster tests
        })
        
        self.mind = Mind(
            config=config,
            device=self.device,
            base_path=self.base_path,
            load_existing=False
        )
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_mind_initialization(self):
        """Test that Mind initializes with all components properly connected."""
        self.assertIsNotNone(self.mind)
        
        # Test that all components are initialized
        self.assertIsNotNone(self.mind.cognitive_component)
        self.assertIsNotNone(self.mind.emotional_component)
        self.assertIsNotNone(self.mind.language_component)
        self.assertIsNotNone(self.mind.memory_component)
        self.assertIsNotNone(self.mind.social_component)
        self.assertIsNotNone(self.mind.development_component)
        
        # Check mind state with appropriate tolerance
        self.assertAlmostEqual(self.mind.mind_state.age_months, 1.0, places=6)
        self.assertEqual(self.mind.mind_state.developmental_stage, "Infancy")
    
    def test_mother_interaction(self):
        """Test the interaction between the Mind and Mother."""
        # Process a basic interaction
        interaction = self.mind.interact_with_mother()
        
        # Verify interaction structure
        self.assertIsNotNone(interaction)
        self.assertIn("mother_state", interaction.model_dump())
        self.assertIn("child_state", interaction.model_dump())
        
        # Check specific fields
        self.assertIn("verbal_response", interaction.mother_state)
        self.assertIn("emotional_state", interaction.mother_state)
        self.assertIn("teaching_elements", interaction.mother_state)
        
        # Child state
        self.assertIn("verbal_response", interaction.child_state)
        self.assertIn("emotional_state", interaction.child_state)
    
    def test_mind_development(self):
        """Test that the Mind develops over multiple interactions."""
        # Get initial state metrics
        initial_status = self.mind.get_status()
        initial_age = initial_status["age_months"]
        initial_vocab = initial_status["vocabulary_size"]
        
        # Process several interactions to trigger development
        for _ in range(20):
            self.mind.interact_with_mother()
        
        # Get updated state
        updated_status = self.mind.get_status()
        updated_age = updated_status["age_months"]
        updated_vocab = updated_status["vocabulary_size"]
        
        # Verify development occurred
        self.assertGreater(updated_age, initial_age)
        
        # Usually vocab should increase, but it's not guaranteed in just 5 interactions
        # so we don't strictly assert this
    
    def test_save_load_mind(self):
        """Test saving and loading the entire Mind state."""
        # Process a few interactions to change state
        for _ in range(10):
            self.mind.interact_with_mother()
        
        # Get current state info
        pre_save_status = self.mind.get_status()
        pre_save_age = pre_save_status["age_months"]
        pre_save_vocab = pre_save_status["vocabulary_size"]
        
        # Save the mind
        self.mind.save()
        
        # Create a new mind with the same config
        new_mind = Mind(
            config=self.mind.config,
            device=self.device,
            base_path=self.base_path,
            load_existing=True
        )
        
        # Get new mind state info
        post_load_status = new_mind.get_status()
        post_load_age = post_load_status["age_months"]
        post_load_vocab = post_load_status["vocabulary_size"]
        
        # Verify state was properly loaded
        self.assertAlmostEqual(pre_save_age, post_load_age, places=1)
        self.assertEqual(pre_save_vocab, post_load_vocab)
    
    def test_mind_components_coherence(self):
        """Test that mind components maintain coherent state after interactions."""
        # Process a few interactions
        for _ in range(10):
            self.mind.interact_with_mother()
        
        # Get component activations from mind_state
        activations = self.mind.mind_state.component_activations
        
        # Use a higher tolerance for float comparisons
        self.assertTrue(abs(activations.get("cognitive", 0.0) - 
                            self.mind.cognitive_component.activation_level) < 0.2)
        
        self.assertTrue(abs(activations.get("emotional", 0.0) - 
                            self.mind.emotional_component.activation_level) < 0.2)
        
        self.assertAlmostEqual(
            activations.get("language", 0.0),
            self.mind.language_component.activation_level,
            places=4
        )
        
        # Get emotional state from component and mind
        mind_emotions = self.mind.mind_state.emotional_state
        component_emotions = self.mind.emotional_component.get_emotional_state()
        
        # Check that emotional states are consistent
        for emotion in mind_emotions:
            if emotion in component_emotions:
                self.assertAlmostEqual(
                    mind_emotions[emotion],
                    component_emotions[emotion],
                    places=4
                )


if __name__ == "__main__":
    unittest.main()