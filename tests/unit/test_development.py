"""
Unit tests for the Development class.

This module contains tests for the Development class, which is responsible for
managing the child's progression through developmental stages.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
from datetime import datetime, timedelta

from neuralchild.core.development import Development
from neuralchild.core.child import Child
from neuralchild.core.mother import Mother
from neuralchild.utils.data_types import (
    DevelopmentalStage, DevelopmentalSubstage, MotherPersonality,
    DevelopmentConfig, SystemState, ChildState, InteractionLog,
    MotherResponse, ChildResponse, Emotion, EmotionType
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=-1, stream=False):
        """Mock chat completion that returns a simple response."""
        return json.dumps({
            "response": "This is a mock response",
            "emotions": [{"type": "joy", "intensity": 0.7}],
            "teaching_elements": {"concept": "testing"}
        })


class TestDevelopment(unittest.TestCase):
    """Test case for the Development class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Mother with a mock LLM client
        self.mother = Mother(personality=MotherPersonality.BALANCED)
        self.mother.llm_client = MockLLMClient()
        
        # Create a Child
        self.child = Child()
        
        # Create a DevelopmentConfig
        self.config = DevelopmentConfig(
            time_acceleration_factor=100,
            random_seed=42,
            mother_personality=MotherPersonality.BALANCED,
            start_age_months=0,
            enable_random_factors=True
        )
        
        # Create Development instance
        self.development = Development(
            child=self.child,
            mother=self.mother,
            config=self.config
        )
        
        # Register components
        self._register_mock_components()
    
    def _register_mock_components(self):
        """Register mock components to the child's mind."""
        # Create mock components
        for component_id, component_type in [
            ("memory_system", "memory"),
            ("language_component", "language"),
            ("emotional_component", "emotional"),
            ("consciousness_component", "consciousness"),
            ("social_component", "social"),
            ("cognitive_component", "cognitive")
        ]:
            mock_component = MagicMock()
            self.child.register_component(component_id, component_type, mock_component)
    
    def test_init(self):
        """Test initialization of Development instance."""
        self.assertEqual(self.development.child, self.child)
        self.assertEqual(self.development.mother, self.mother)
        self.assertEqual(self.development.config, self.config)
        self.assertEqual(self.development.child.state.simulated_age_months, 0)
        self.assertEqual(self.development.child.state.developmental_stage, DevelopmentalStage.INFANCY)
        self.assertEqual(self.development.child.state.developmental_substage, DevelopmentalSubstage.EARLY_INFANCY)
    
    def test_update_simulated_time(self):
        """Test updating the simulated time."""
        initial_time = self.development.simulated_time
        
        # Mock datetime.now() to return a consistent time delta
        with patch('neuralchild.core.development.datetime') as mock_datetime:
            # Mock now() to return a time 10 seconds later than the initial time
            mock_now = initial_time + timedelta(seconds=10)
            mock_datetime.now.return_value = mock_now
            
            # Also update the class variable for simulation
            self.development.last_interaction_time = initial_time
            
            # Run the update method
            self.development.update_simulated_time()
            
            # With acceleration factor of 100, 10 seconds real time = 1000 seconds simulated
            # 1000 seconds is about 0.01 months (assuming 30 days/month)
            self.assertGreater(self.development.child.state.simulated_age_months, 0)
    
    def test_simulate_interaction(self):
        """Test simulating an interaction between Mother and Child."""
        # Test with infant vocalization
        child_response, mother_response = self.development.simulate_interaction(
            initial_vocalization="goo"
        )
        
        self.assertIsNotNone(child_response)
        self.assertIsNotNone(mother_response)
        self.assertEqual(mother_response.text, "This is a mock response")
        
        # Test with text for older child
        child_response, mother_response = self.development.simulate_interaction(
            initial_text="Hello, mother!"
        )
        
        self.assertIsNotNone(child_response)
        self.assertIsNotNone(mother_response)
        self.assertEqual(mother_response.text, "This is a mock response")
    
    def test_accelerate_development_short(self):
        """Test accelerating development for a short period."""
        # Accelerate development by 1 month
        stages_progressed = self.development.accelerate_development(1)
        
        # Check that development occurred
        self.assertEqual(self.child.state.simulated_age_months, 1)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.INFANCY)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.EARLY_INFANCY)
        
        # Verify stages_progressed contains the stages that were passed through
        self.assertEqual(len(stages_progressed), 1)
        self.assertEqual(stages_progressed[0], DevelopmentalStage.INFANCY)
    
    def test_accelerate_development_longer(self):
        """Test accelerating development for a longer period (crossing stages)."""
        # Accelerate development by 24 months (2 years)
        # This should cross from INFANCY to EARLY_CHILDHOOD
        stages_progressed = self.development.accelerate_development(24)
        
        # Check that development occurred and stage changed
        self.assertEqual(self.child.state.simulated_age_months, 24)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.EARLY_CHILDHOOD)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.EARLY_TODDLER)
        
        # Verify stages_progressed contains the stages that were passed through
        self.assertEqual(len(stages_progressed), 2)
        self.assertEqual(stages_progressed[0], DevelopmentalStage.INFANCY)
        self.assertEqual(stages_progressed[1], DevelopmentalStage.EARLY_CHILDHOOD)
    
    def test_save_and_load_system_state(self):
        """Test saving and loading the system state."""
        # Accelerate development slightly to have some data
        self.development.accelerate_development(1)
        
        # Save state to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save state
            self.development.save_system_state(temp_path)
            
            # Verify file exists and has content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
            
            # Load state into a new Development instance
            loaded_development = Development.load_system_state(temp_path)
            
            # Verify state was loaded correctly
            self.assertEqual(loaded_development.child.state.simulated_age_months, 1)
            self.assertEqual(loaded_development.child.state.developmental_stage, DevelopmentalStage.INFANCY)
            self.assertEqual(loaded_development.child.state.developmental_substage, DevelopmentalSubstage.EARLY_INFANCY)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_interaction_history(self):
        """Test that interaction history is properly maintained."""
        # Initial history should be empty
        self.assertEqual(len(self.development.system_state.interaction_history), 0)
        
        # Simulate an interaction
        self.development.simulate_interaction(initial_text="Hello")
        
        # Check that history was updated
        self.assertEqual(len(self.development.system_state.interaction_history), 1)
        
        # Simulate more interactions
        for _ in range(5):
            self.development.simulate_interaction(initial_text="Hello again")
        
        # Check that history was updated
        self.assertEqual(len(self.development.system_state.interaction_history), 6)
        
        # Check history limit
        for _ in range(200):  # More than the max history size
            self.development.simulate_interaction(initial_text="Hello again")
        
        # History should be capped
        self.assertLessEqual(len(self.development.system_state.interaction_history), 100)  # Max history size


if __name__ == '__main__':
    unittest.main() 