"""
Tests for the Mother LLM module of the Large Mind Model.
"""

import pytest
import json
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from lmm.mother.mother_llm import MotherLLM
from lmm.config import MotherConfig


class TestMotherLLM:
    """Test suite for the Mother LLM."""
    
    def test_mother_initialization(self, mother_llm, test_config):
        """Test that Mother LLM initializes correctly."""
        assert isinstance(mother_llm.config, dict)
        assert "personality_traits" in mother_llm.config
        assert "parenting_style" in mother_llm.config
        
        # Test with specific configuration
        config = MotherConfig(
            personality_traits={
                "patience": 0.9,
                "kindness": 0.8,
                "strictness": 0.3,
                "expressiveness": 0.7,
            },
            parenting_style="permissive",
            teaching_approach="direct_instruction"
        )
        
        mother = MotherLLM(config=config.dict())
        assert mother.config["parenting_style"] == "permissive"
        assert mother.config["personality_traits"]["patience"] == 0.9
    
    def test_personality_influence(self, mother_llm):
        """Test that personality traits influence interactions."""
        # Get baseline response with default personality
        response1 = mother_llm.process({
            "action": "respond",
            "to_text": "I made a mistake",
            "context": {"child_development_level": 0.5}
        })
        
        # Create new mother with different personality
        strict_config = MotherConfig(
            personality_traits={
                "patience": 0.3,
                "kindness": 0.5,
                "strictness": 0.9,
                "expressiveness": 0.4,
            },
            parenting_style="authoritarian"
        )
        
        strict_mother = MotherLLM(config=strict_config.dict())
        
        # Get response with stricter personality
        response2 = strict_mother.process({
            "action": "respond",
            "to_text": "I made a mistake",
            "context": {"child_development_level": 0.5}
        })
        
        # Responses should be different
        assert response1["response"] != response2["response"]
        
        # Analyze sentiment
        assert "sentiment" in response1
        assert "sentiment" in response2
        # Strict mother should be more critical
        assert response2["sentiment"]["critical"] > response1["sentiment"]["critical"]
    
    @patch('lmm.mother.mother_llm.requests.post')
    def test_llm_interaction(self, mock_post, mother_llm):
        """Test LLM API interaction."""
        # Mock the LLM API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I understand you're frustrated. Let's work through this together."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test interaction
        result = mother_llm.process({
            "action": "respond",
            "to_text": "I can't solve this problem!",
            "context": {
                "child_development_level": 0.6,
                "child_emotional_state": "frustrated",
                "interaction_history": []
            }
        })
        
        # Verify result
        assert result["success"] is True
        assert "response" in result
        assert result["response"] == "I understand you're frustrated. Let's work through this together."
        
        # Verify correct API call was made
        mock_post.assert_called_once()
        # Verify personality was included in prompt
        personality_str = str(mock_post.call_args[1]["json"]["messages"])
        assert "personality" in personality_str.lower()
    
    def test_developmental_adaptation(self, mother_llm):
        """Test that responses adapt to different developmental levels."""
        # Response to infant level
        infant_response = mother_llm.process({
            "action": "respond",
            "to_text": "ba ba",
            "context": {
                "child_development_level": 0.1,
                "child_age_months": 9,
                "child_linguistic_stage": "pre_linguistic"
            }
        })
        
        # Response to child level
        child_response = mother_llm.process({
            "action": "respond",
            "to_text": "Why sky blue?",
            "context": {
                "child_development_level": 0.4,
                "child_age_months": 36,
                "child_linguistic_stage": "simple_sentences"
            }
        })
        
        # Response to adolescent level
        adolescent_response = mother_llm.process({
            "action": "respond",
            "to_text": "I'm wondering about the philosophical implications of consciousness.",
            "context": {
                "child_development_level": 0.7,
                "child_age_months": 156,
                "child_linguistic_stage": "complex_language"
            }
        })
        
        # Verify differences in complexity
        assert len(infant_response["response"]) < len(child_response["response"])
        assert len(child_response["response"]) < len(adolescent_response["response"])
        
        # Check linguistic adaptation
        assert "linguistic_complexity" in infant_response
        assert "linguistic_complexity" in child_response
        assert "linguistic_complexity" in adolescent_response
        
        assert infant_response["linguistic_complexity"] < child_response["linguistic_complexity"]
        assert child_response["linguistic_complexity"] < adolescent_response["linguistic_complexity"]
    
    def test_teaching_behavior(self, mother_llm):
        """Test teaching behaviors with different approaches."""
        # Default Mother (guided discovery approach)
        guided_response = mother_llm.process({
            "action": "teach",
            "topic": "addition",
            "context": {
                "child_development_level": 0.4,
                "child_knowledge": {"math": 0.2},
                "learning_style": "visual"
            }
        })
        
        # Create direct instruction Mother
        direct_config = MotherConfig(
            teaching_approach="direct_instruction",
            personality_traits={"patience": 0.7, "expressiveness": 0.8}
        )
        direct_mother = MotherLLM(config=direct_config.dict())
        
        direct_response = direct_mother.process({
            "action": "teach",
            "topic": "addition",
            "context": {
                "child_development_level": 0.4,
                "child_knowledge": {"math": 0.2},
                "learning_style": "visual"
            }
        })
        
        # Responses should be different
        assert guided_response["response"] != direct_response["response"]
        
        # Direct instruction should contain more explicit instructions
        assert "teaching_style" in guided_response
        assert "teaching_style" in direct_response
        assert guided_response["teaching_style"] == "guided_discovery"
        assert direct_response["teaching_style"] == "direct_instruction"
    
    def test_emotional_support(self, mother_llm):
        """Test emotional support capabilities."""
        # Test comforting response
        comfort_response = mother_llm.process({
            "action": "respond",
            "to_text": "I'm feeling sad",
            "context": {
                "child_development_level": 0.5,
                "child_emotional_state": "sadness",
                "emotional_intensity": 0.8
            }
        })
        
        # Test celebration response
        celebration_response = mother_llm.process({
            "action": "respond",
            "to_text": "I did it! I solved the puzzle!",
            "context": {
                "child_development_level": 0.5,
                "child_emotional_state": "joy",
                "emotional_intensity": 0.9
            }
        })
        
        # Verify appropriate emotional responses
        assert "emotional_support" in comfort_response
        assert "validation" in comfort_response
        
        assert "reinforcement" in celebration_response
        assert "emotional_mirroring" in celebration_response
        
        # Different emotional contexts should yield different responses
        assert comfort_response["response"] != celebration_response["response"]
        assert comfort_response["emotional_support"] != celebration_response["emotional_support"]
    
    def test_conversation_memory(self, mother_llm):
        """Test conversational memory and context retention."""
        # Start a conversation
        conversation = []
        
        # First interaction
        response1 = mother_llm.process({
            "action": "respond",
            "to_text": "My name is Alex",
            "context": {
                "child_development_level": 0.6,
                "conversation_history": conversation
            }
        })
        conversation.append({"role": "child", "text": "My name is Alex"})
        conversation.append({"role": "mother", "text": response1["response"]})
        
        # Second interaction referring to previous info
        response2 = mother_llm.process({
            "action": "respond",
            "to_text": "What's my name?",
            "context": {
                "child_development_level": 0.6,
                "conversation_history": conversation
            }
        })
        
        # Verify memory retention
        assert "Alex" in response2["response"]
    
    def test_parenting_style_influence(self, mother_llm):
        """Test different parenting styles influence interactions."""
        styles = ["authoritative", "permissive", "authoritarian"]
        responses = {}
        
        for style in styles:
            config = MotherConfig(parenting_style=style)
            parent = MotherLLM(config=config.dict())
            
            # Test response to a limit-testing scenario
            responses[style] = parent.process({
                "action": "respond",
                "to_text": "I don't want to clean up my toys!",
                "context": {"child_development_level": 0.4}
            })
        
        # Verify different styles produce different responses
        assert responses["authoritative"]["response"] != responses["permissive"]["response"]
        assert responses["authoritative"]["response"] != responses["authoritarian"]["response"]
        assert responses["permissive"]["response"] != responses["authoritarian"]["response"]
        
        # Verify style-specific characteristics
        assert responses["authoritative"]["boundary_setting"] is True
        assert responses["permissive"]["boundary_setting"] is False
        assert responses["authoritarian"]["directive"] is True
    
    def test_developmental_guidance(self, mother_llm):
        """Test developmental guidance capabilities."""
        # Test identifying learning opportunity
        result = mother_llm.process({
            "action": "analyze_development",
            "child_state": {
                "development_level": 0.4,
                "modules": {
                    "language": 0.5,
                    "emotion": 0.3,
                    "memory": 0.4,
                    "social": 0.2
                }
            }
        })
        
        assert result["success"] is True
        assert "recommendations" in result
        assert "priority_areas" in result
        assert len(result["priority_areas"]) > 0
        
        # Test generating developmentally appropriate activity
        result = mother_llm.process({
            "action": "suggest_activity",
            "target_module": "social",
            "context": {
                "child_development_level": 0.4,
                "module_levels": {
                    "social": 0.2,
                    "language": 0.5
                }
            }
        })
        
        assert result["success"] is True
        assert "activity" in result
        assert "expected_benefit" in result
        assert "social" in result["target_modules"] 