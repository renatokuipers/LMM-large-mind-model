"""
Unit tests for the Mother class.

This module contains tests for the Mother class, which is responsible for
generating nurturing responses to the Child's communications.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile

from neuralchild.core.mother import Mother
from neuralchild.utils.data_types import (
    MotherResponse, ChildResponse, Emotion, EmotionType,
    DevelopmentalStage, DevelopmentalSubstage, MotherPersonality
)

from llm_module import Message


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, response_type="json"):
        self.response_type = response_type
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=-1, stream=False):
        """Mock chat completion that returns a response."""
        if self.response_type == "json":
            return json.dumps({
                "response": "This is a mock response",
                "emotions": [{"type": "joy", "intensity": 0.7}],
                "teaching_elements": {"concept": "testing"},
                "non_verbal_cues": "Smiles warmly"
            })
        else:
            return "Plain text response without JSON"


class TestMother(unittest.TestCase):
    """Test case for the Mother class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a Mother with a mock LLM client
        self.mother = Mother(personality=MotherPersonality.BALANCED)
        self.mother.llm_client = MockLLMClient()
        
        # Create sample child responses for different developmental stages
        self.infant_response = ChildResponse(
            vocalization="goo",
            emotional_state=[Emotion(type=EmotionType.JOY, intensity=0.5)],
            attention_focus="mother"
        )
        
        self.child_response = ChildResponse(
            text="Hello, mother!",
            emotional_state=[Emotion(type=EmotionType.JOY, intensity=0.7)],
            attention_focus="mother"
        )
    
    def test_init(self):
        """Test initialization of Mother instance."""
        self.assertEqual(self.mother.personality, MotherPersonality.BALANCED)
        self.assertEqual(self.mother.temperature, 0.7)
        self.assertIsNotNone(self.mother.llm_client)
        self.assertEqual(len(self.mother.history), 0)
    
    def test_respond_to_child_infancy(self):
        """Test responding to a child in infancy stage."""
        response = self.mother.respond_to_child(
            child_response=self.infant_response,
            child_developmental_stage=DevelopmentalStage.INFANCY,
            child_age_months=3,
            child_developmental_substage=DevelopmentalSubstage.EARLY_INFANCY
        )
        
        # Check response properties
        self.assertEqual(response.text, "This is a mock response")
        self.assertEqual(len(response.emotional_state), 1)
        self.assertEqual(response.emotional_state[0].type, EmotionType.JOY)
        self.assertEqual(response.emotional_state[0].intensity, 0.7)
        self.assertEqual(response.non_verbal_cues, "Smiles warmly")
    
    def test_respond_to_child_early_childhood(self):
        """Test responding to a child in early childhood stage."""
        response = self.mother.respond_to_child(
            child_response=self.child_response,
            child_developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            child_age_months=30,
            child_developmental_substage=DevelopmentalSubstage.EARLY_TODDLER
        )
        
        # Check response properties
        self.assertEqual(response.text, "This is a mock response")
        self.assertEqual(len(response.emotional_state), 1)
        self.assertEqual(response.emotional_state[0].type, EmotionType.JOY)
        self.assertEqual(response.emotional_state[0].intensity, 0.7)
    
    def test_respond_to_child_json_parsing_error(self):
        """Test handling JSON parsing errors in LLM responses."""
        # Set up client to return non-JSON response
        self.mother.llm_client = MockLLMClient(response_type="text")
        
        response = self.mother.respond_to_child(
            child_response=self.child_response,
            child_developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            child_age_months=30,
            child_developmental_substage=DevelopmentalSubstage.EARLY_TODDLER
        )
        
        # Should still get a valid response with default values
        self.assertEqual(response.text, "Plain text response without JSON"[:500])
        self.assertEqual(len(response.emotional_state), 1)
        self.assertEqual(response.emotional_state[0].type, EmotionType.SURPRISE)
    
    def test_create_prompt(self):
        """Test prompt creation for different developmental stages."""
        # Test for infancy
        prompt_infancy = self.mother._create_prompt(
            child_text="goo",
            child_emotions=[Emotion(type=EmotionType.JOY, intensity=0.5)],
            developmental_stage=DevelopmentalStage.INFANCY,
            developmental_substage=DevelopmentalSubstage.EARLY_INFANCY,
            child_age_months=3
        )
        
        self.assertIn("infancy", prompt_infancy.lower())
        self.assertIn("3 months", prompt_infancy.lower())
        self.assertIn("goo", prompt_infancy)
        
        # Test for early childhood
        prompt_childhood = self.mother._create_prompt(
            child_text="Hello, mother!",
            child_emotions=[Emotion(type=EmotionType.JOY, intensity=0.7)],
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            developmental_substage=DevelopmentalSubstage.EARLY_TODDLER,
            child_age_months=30
        )
        
        self.assertIn("early_childhood", prompt_childhood.lower())
        self.assertIn("30 months", prompt_childhood.lower())
        self.assertIn("Hello, mother!", prompt_childhood)
    
    def test_personality_affect(self):
        """Test that different personalities affect responses."""
        # Create mothers with different personalities
        nurturing_mother = Mother(personality=MotherPersonality.NURTURING)
        nurturing_mother.llm_client = self.mother.llm_client
        
        authoritarian_mother = Mother(personality=MotherPersonality.AUTHORITARIAN)
        authoritarian_mother.llm_client = self.mother.llm_client
        
        # Test with the same child message
        prompt_nurturing = nurturing_mother._create_prompt(
            child_text="Hello",
            child_emotions=[Emotion(type=EmotionType.JOY, intensity=0.5)],
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            developmental_substage=DevelopmentalSubstage.EARLY_TODDLER,
            child_age_months=30
        )
        
        prompt_authoritarian = authoritarian_mother._create_prompt(
            child_text="Hello",
            child_emotions=[Emotion(type=EmotionType.JOY, intensity=0.5)],
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            developmental_substage=DevelopmentalSubstage.EARLY_TODDLER,
            child_age_months=30
        )
        
        # Prompts should be different based on personality
        self.assertNotEqual(prompt_nurturing, prompt_authoritarian)
        self.assertIn("warm", prompt_nurturing.lower())
        self.assertIn("structured", prompt_authoritarian.lower())
    
    def test_generate_fallback_response(self):
        """Test fallback response generation when LLM is unavailable."""
        # Test for different developmental stages
        for stage in [
            DevelopmentalStage.INFANCY,
            DevelopmentalStage.EARLY_CHILDHOOD,
            DevelopmentalStage.MIDDLE_CHILDHOOD,
            DevelopmentalStage.ADOLESCENCE,
            DevelopmentalStage.EARLY_ADULTHOOD
        ]:
            response = self.mother._generate_fallback_response(stage)
            
            # Check that we got a valid response
            self.assertIsInstance(response, MotherResponse)
            self.assertIsNotNone(response.text)
            self.assertGreater(len(response.text), 0)
            self.assertGreater(len(response.emotional_state), 0)
    
    def test_set_personality(self):
        """Test changing the mother's personality."""
        # Initial personality
        self.assertEqual(self.mother.personality, MotherPersonality.BALANCED)
        
        # Change personality
        self.mother.set_personality(MotherPersonality.NURTURING)
        self.assertEqual(self.mother.personality, MotherPersonality.NURTURING)
        
        # Test a different personality
        self.mother.set_personality(MotherPersonality.AUTHORITARIAN)
        self.assertEqual(self.mother.personality, MotherPersonality.AUTHORITARIAN)
    
    def test_history_management(self):
        """Test history management functions."""
        # Initially empty
        self.assertEqual(len(self.mother.history), 0)
        
        # Add items to history
        self.mother.history.append({"message": "test1"})
        self.mother.history.append({"message": "test2"})
        self.assertEqual(len(self.mother.history), 2)
        
        # Clear history
        self.mother.clear_history()
        self.assertEqual(len(self.mother.history), 0)
    
    def test_save_and_load_history(self):
        """Test saving and loading interaction history."""
        # Add items to history
        self.mother.history.append({"message": "test1"})
        self.mother.history.append({"message": "test2"})
        
        # Save history to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save history
            self.mother.save_history(temp_path)
            
            # Verify file exists and has content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
            
            # Clear history
            self.mother.clear_history()
            self.assertEqual(len(self.mother.history), 0)
            
            # Load history
            self.mother.load_history(temp_path)
            
            # Verify history was loaded
            self.assertEqual(len(self.mother.history), 2)
            self.assertEqual(self.mother.history[0]["message"], "test1")
            self.assertEqual(self.mother.history[1]["message"], "test2")
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main() 