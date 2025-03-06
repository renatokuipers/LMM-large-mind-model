"""
Unit tests for component integration and system interactions.

This module contains tests that verify the interactions between
different components of the NeuralChild framework.
"""

import unittest
from unittest.mock import MagicMock, patch

from neuralchild.utils.component_integration import ComponentIntegration
from neuralchild.utils.data_types import DevelopmentalStage
from neuralchild.core.child import Child
from neuralchild.core.mother import Mother
from neuralchild.core.development import Development
from neuralchild.utils.data_types import DevelopmentConfig, MotherPersonality


class TestComponentIntegration(unittest.TestCase):
    """Test case for the ComponentIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = ComponentIntegration()
    
    def test_init(self):
        """Test initialization of ComponentIntegration."""
        self.assertEqual(self.integration.integration_level, 0.1)
        self.assertEqual(len(self.integration.component_states), 0)
        self.assertEqual(len(self.integration.integration_history), 0)
    
    def test_register_component_state(self):
        """Test registering component states."""
        # Register some components
        self.integration.register_component_state("memory_system", "memory")
        self.integration.register_component_state("language_component", "language")
        
        # Verify components were registered
        self.assertIn("memory_system", self.integration.component_states)
        self.assertIn("language_component", self.integration.component_states)
        self.assertEqual(self.integration.component_states["memory_system"]["type"], "memory")
        self.assertEqual(self.integration.component_states["language_component"]["type"], "language")
    
    def test_get_component_influence(self):
        """Test getting component influence."""
        # Register components
        self.integration.register_component_state("memory_system", "memory")
        self.integration.register_component_state("language_component", "language")
        
        # Get influence of memory on language
        influence = self.integration.get_component_influence(
            source_component="memory",
            target_component="language",
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD
        )
        
        # Influence should be a float between 0 and 1
        self.assertIsInstance(influence, float)
        self.assertGreaterEqual(influence, 0)
        self.assertLessEqual(influence, 1)
    
    def test_apply_cross_component_effects(self):
        """Test applying cross-component effects."""
        # Register components
        self.integration.register_component_state("memory_system", "memory")
        self.integration.register_component_state("language_component", "language")
        self.integration.register_component_state("emotional_component", "emotional")
        
        # Apply effects
        effects = self.integration.apply_cross_component_effects(
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            active_component_id="language_component"
        )
        
        # Should have effects for each component
        self.assertIn("memory_system", effects)
        self.assertIn("emotional_component", effects)
        
        # Active component should not have effects on itself
        self.assertNotIn("language_component", effects)
        
        # History should be updated
        self.assertEqual(len(self.integration.integration_history), 1)
    
    def test_synchronize_development(self):
        """Test synchronizing integration level with developmental stage."""
        # Initial integration level
        initial_level = self.integration.integration_level
        
        # Synchronize to higher stage
        self.integration.synchronize_development(DevelopmentalStage.ADOLESCENCE)
        
        # Integration level should increase
        self.assertGreater(self.integration.integration_level, initial_level)


class TestSystemInteractions(unittest.TestCase):
    """Test case for interactions between core system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create Child, Mother, and Development instances with mock components
        self.child = Child()
        self.mother = Mother()
        
        # Mock the LLM client
        self.mother.llm_client = MagicMock()
        self.mother.llm_client.chat_completion.return_value = '{"response": "Test response", "emotions": [{"type": "joy", "intensity": 0.7}]}'
        
        # Create Development instance
        self.development = Development(
            child=self.child,
            mother=self.mother,
            config=DevelopmentConfig(
                time_acceleration_factor=100,
                random_seed=42,
                mother_personality=MotherPersonality.BALANCED
            )
        )
        
        # Register mock components
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
            
            # Configure component mocks based on type
            if component_type == "language":
                mock_component.process_language_input.return_value = {
                    "understood_words": ["test"],
                    "detected_sentiment": "positive",
                    "response_type": "simple_phrase"
                }
                mock_component.get_vocabulary_size.return_value = 50
                mock_component.get_grammar_complexity.return_value = 0.4
            elif component_type == "emotional":
                mock_component.process_emotional_input.return_value = []
                mock_component.get_emotional_regulation.return_value = 0.5
            elif component_type == "social":
                mock_component.get_social_awareness.return_value = 0.6
            elif component_type == "cognitive":
                mock_component.get_object_permanence.return_value = 0.7
                mock_component.get_abstract_thinking.return_value = 0.3
            
            self.child.register_component(component_id, component_type, mock_component)
    
    def test_end_to_end_interaction(self):
        """Test an end-to-end interaction between Mother and Child."""
        # Simulate an interaction
        child_response, mother_response = self.development.simulate_interaction(
            initial_text="Hello"
        )
        
        # Verify responses
        self.assertIsNotNone(child_response)
        self.assertIsNotNone(mother_response)
        self.assertEqual(mother_response.text, "Test response")
        
        # System state should be updated
        self.assertEqual(len(self.development.system_state.interaction_history), 1)
        
        # Child's language component should have been called
        language_component = self.child.components["language_component"]
        language_component.process_language_input.assert_called()
    
    def test_component_influence_during_interaction(self):
        """Test that components influence each other during interactions."""
        # Create a more sophisticated response
        self.mother.llm_client.chat_completion.return_value = json.dumps({
            "response": "That's a good observation! How do you feel about it?",
            "emotions": [{"type": "joy", "intensity": 0.7}],
            "teaching_elements": {"concept": "reflection", "strategy": "open_question"}
        })
        
        # Patch the integration's apply_cross_component_effects method to track calls
        with patch.object(self.child.integration, 'apply_cross_component_effects', wraps=self.child.integration.apply_cross_component_effects) as mock_apply:
            # Simulate an interaction with consciousness-related content
            self.development.simulate_interaction(
                initial_text="I think I understand how this works"
            )
            
            # Verify that cross-component effects were applied
            mock_apply.assert_called()
            
            # The consciousness component should have been detected as active
            # based on the message content about thinking and understanding
            args, kwargs = mock_apply.call_args
            self.assertEqual(kwargs["developmental_stage"], self.child.state.developmental_stage)
    
    def test_development_acceleration_integration(self):
        """Test that accelerated development properly integrates components."""
        # Set initial metrics
        mock_emotional = self.child.components["emotional_component"]
        mock_emotional.get_emotional_regulation.return_value = 0.3
        
        mock_language = self.child.components["language_component"]
        mock_language.get_vocabulary_size.return_value = 30
        mock_language.get_grammar_complexity.return_value = 0.3
        
        mock_social = self.child.components["social_component"]
        mock_social.get_social_awareness.return_value = 0.3
        
        mock_cognitive = self.child.components["cognitive_component"]
        mock_cognitive.get_object_permanence.return_value = 0.4
        mock_cognitive.get_abstract_thinking.return_value = 0.2
        
        # Stub simulate_interaction to prevent actual API calls
        with patch.object(self.development, 'simulate_interaction') as mock_simulate:
            mock_simulate.return_value = (MagicMock(), MagicMock())
            
            # Run accelerated development for 1 month
            self.development.accelerate_development(1)
            
            # Verify that simulate_interaction was called multiple times
            self.assertGreater(mock_simulate.call_count, 1)
            
            # Verify that developmental metrics were updated
            self.child.update_developmental_metrics.assert_called()


import json


if __name__ == '__main__':
    unittest.main() 