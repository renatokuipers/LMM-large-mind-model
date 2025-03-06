"""
Unit tests for the Child class.

This module contains tests for the Child class, which is the central component
of the NeuralChild framework.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from neuralchild.core.child import Child
from neuralchild.utils.data_types import (
    ChildState, ChildResponse, MotherResponse, Emotion, EmotionType,
    DevelopmentalStage, DevelopmentalSubstage, ComponentState
)


class TestChild(unittest.TestCase):
    """Test case for the Child class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a Child with default state
        self.child = Child()
        
        # Create sample mother responses for testing
        self.mother_response = MotherResponse(
            text="Hello, little one!",
            emotional_state=[Emotion(type=EmotionType.JOY, intensity=0.7)],
            teaching_elements={"concept": "greeting"},
            non_verbal_cues="Smiles warmly"
        )
    
    def test_init(self):
        """Test initialization of Child instance with default parameters."""
        self.assertIsNotNone(self.child.state)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.INFANCY)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.EARLY_INFANCY)
        self.assertEqual(self.child.state.simulated_age_months, 0)
        self.assertEqual(len(self.child.components), 0)
        self.assertIsNotNone(self.child.integration)
    
    def test_init_with_custom_state(self):
        """Test initialization with a custom state."""
        # Create a custom state
        custom_state = ChildState(
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            developmental_substage=DevelopmentalSubstage.EARLY_TODDLER,
            simulated_age_months=24
        )
        
        # Create child with custom state
        child = Child(initial_state=custom_state)
        
        # Verify state was set correctly
        self.assertEqual(child.state.developmental_stage, DevelopmentalStage.EARLY_CHILDHOOD)
        self.assertEqual(child.state.developmental_substage, DevelopmentalSubstage.EARLY_TODDLER)
        self.assertEqual(child.state.simulated_age_months, 24)
    
    def test_register_component(self):
        """Test registering a component with the child's mind."""
        # Create a mock component
        mock_component = MagicMock()
        
        # Register the component
        self.child.register_component("test_component", "test", mock_component)
        
        # Verify component was registered
        self.assertIn("test_component", self.child.components)
        self.assertEqual(self.child.components["test_component"], mock_component)
        
        # Verify component state was created
        self.assertIn("test_component", self.child.state.component_states)
        self.assertEqual(self.child.state.component_states["test_component"].component_id, "test_component")
        self.assertEqual(self.child.state.component_states["test_component"].component_type, "test")
    
    def test_process_mother_response_infancy(self):
        """Test processing a mother's response during infancy."""
        # Set up the child in infancy stage
        self.child.state.developmental_stage = DevelopmentalStage.INFANCY
        self.child.state.developmental_substage = DevelopmentalSubstage.EARLY_INFANCY
        self.child.state.simulated_age_months = 3
        
        # Register mock components for processing
        for component_id, component_type in [
            ("memory_system", "memory"),
            ("language_component", "language"),
            ("emotional_component", "emotional")
        ]:
            mock_component = MagicMock()
            self.child.register_component(component_id, component_type, mock_component)
        
        # Process the mother's response
        child_response = self.child.process_mother_response(self.mother_response)
        
        # Verify response was generated
        self.assertIsInstance(child_response, ChildResponse)
        self.assertIsNotNone(child_response.vocalization or child_response.text)
        self.assertGreater(len(child_response.emotional_state), 0)
    
    def test_process_mother_response_early_childhood(self):
        """Test processing a mother's response during early childhood."""
        # Set up the child in early childhood stage
        self.child.state.developmental_stage = DevelopmentalStage.EARLY_CHILDHOOD
        self.child.state.developmental_substage = DevelopmentalSubstage.EARLY_TODDLER
        self.child.state.simulated_age_months = 30
        
        # Register mock components for processing
        for component_id, component_type in [
            ("memory_system", "memory"),
            ("language_component", "language"),
            ("emotional_component", "emotional"),
            ("consciousness_component", "consciousness")
        ]:
            mock_component = MagicMock()
            self.child.register_component(component_id, component_type, mock_component)
        
        # Process the mother's response
        child_response = self.child.process_mother_response(self.mother_response)
        
        # Verify response was generated
        self.assertIsInstance(child_response, ChildResponse)
        self.assertIsNotNone(child_response.text)  # Should have text at this stage
        self.assertGreater(len(child_response.emotional_state), 0)
    
    def test_check_stage_progression_no_change(self):
        """Test checking for stage progression with no change."""
        # Set initial stage
        self.child.state.developmental_stage = DevelopmentalStage.INFANCY
        self.child.state.developmental_substage = DevelopmentalSubstage.EARLY_INFANCY
        self.child.state.simulated_age_months = 3
        
        # Mock metrics with insufficient values for progression
        self.child.state.metrics.vocabulary_size = 10
        self.child.state.metrics.grammatical_complexity = 0.1
        self.child.state.metrics.emotional_regulation = 0.1
        self.child.state.metrics.social_awareness = 0.1
        self.child.state.metrics.object_permanence = 0.1
        self.child.state.metrics.abstract_thinking = 0.1
        self.child.state.metrics.self_awareness = 0.1
        
        # Check for progression
        progression_occurred = self.child.check_stage_progression()
        
        # Verify no progression occurred
        self.assertFalse(progression_occurred)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.INFANCY)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.EARLY_INFANCY)
    
    def test_check_stage_progression_substage_change(self):
        """Test checking for stage progression with a substage change."""
        # Set initial stage at the end of EARLY_INFANCY
        self.child.state.developmental_stage = DevelopmentalStage.INFANCY
        self.child.state.developmental_substage = DevelopmentalSubstage.EARLY_INFANCY
        self.child.state.simulated_age_months = 7  # Close to transition
        
        # Set metrics high enough for progression to next substage
        self.child.state.metrics.vocabulary_size = 20
        self.child.state.metrics.grammatical_complexity = 0.3
        self.child.state.metrics.emotional_regulation = 0.3
        self.child.state.metrics.social_awareness = 0.3
        self.child.state.metrics.object_permanence = 0.4
        self.child.state.metrics.abstract_thinking = 0.2
        self.child.state.metrics.self_awareness = 0.3
        
        # Check for progression
        progression_occurred = self.child.check_stage_progression()
        
        # Verify substage progression occurred
        self.assertTrue(progression_occurred)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.INFANCY)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.MIDDLE_INFANCY)
    
    def test_check_stage_progression_stage_change(self):
        """Test checking for stage progression with a stage change."""
        # Set initial stage at the end of LATE_INFANCY
        self.child.state.developmental_stage = DevelopmentalStage.INFANCY
        self.child.state.developmental_substage = DevelopmentalSubstage.LATE_INFANCY
        self.child.state.simulated_age_months = 23  # Close to transition
        
        # Set metrics high enough for progression to next stage
        self.child.state.metrics.vocabulary_size = 100
        self.child.state.metrics.grammatical_complexity = 0.6
        self.child.state.metrics.emotional_regulation = 0.5
        self.child.state.metrics.social_awareness = 0.5
        self.child.state.metrics.object_permanence = 0.8
        self.child.state.metrics.abstract_thinking = 0.4
        self.child.state.metrics.self_awareness = 0.5
        
        # Check for progression
        progression_occurred = self.child.check_stage_progression()
        
        # Verify stage progression occurred
        self.assertTrue(progression_occurred)
        self.assertEqual(self.child.state.developmental_stage, DevelopmentalStage.EARLY_CHILDHOOD)
        self.assertEqual(self.child.state.developmental_substage, DevelopmentalSubstage.EARLY_TODDLER)
    
    def test_update_developmental_metrics(self):
        """Test updating developmental metrics based on component states."""
        # Register mock components
        for component_id, component_type in [
            ("memory_system", "memory"),
            ("language_component", "language"),
            ("emotional_component", "emotional"),
            ("social_component", "social"),
            ("consciousness_component", "consciousness")
        ]:
            mock_component = MagicMock()
            
            # Configure mock methods based on component type
            if component_type == "language":
                mock_component.get_vocabulary_size.return_value = 50
                mock_component.get_grammar_complexity.return_value = 0.4
            elif component_type == "emotional":
                mock_component.get_emotional_regulation.return_value = 0.5
            elif component_type == "social":
                mock_component.get_social_awareness.return_value = 0.6
            
            self.child.register_component(component_id, component_type, mock_component)
        
        # Update metrics
        self.child.update_developmental_metrics()
        
        # Verify metrics were updated
        self.assertEqual(self.child.state.metrics.vocabulary_size, 50)
        self.assertEqual(self.child.state.metrics.grammatical_complexity, 0.4)
        self.assertEqual(self.child.state.metrics.emotional_regulation, 0.5)
        self.assertEqual(self.child.state.metrics.social_awareness, 0.6)
        
        # Verify history was updated
        self.assertGreater(len(self.child.state.metrics.history["vocabulary_size"]), 0)
        self.assertGreater(len(self.child.state.metrics.history["grammatical_complexity"]), 0)
        self.assertGreater(len(self.child.state.metrics.history["emotional_regulation"]), 0)
        self.assertGreater(len(self.child.state.metrics.history["social_awareness"]), 0)


if __name__ == '__main__':
    unittest.main() 