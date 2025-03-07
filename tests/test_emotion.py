"""
Tests for the Emotion module of the Large Mind Model.
"""

import pytest
import numpy as np
import time
from pathlib import Path

from lmm.core.emotion import EmotionModule


class TestEmotionModule:
    """Test suite for the Emotion module."""
    
    def test_emotion_initialization(self, emotion_module):
        """Test that emotion module initializes correctly."""
        assert emotion_module.name == "emotion"
        assert not emotion_module.initialized
        
        # Initialize the module
        result = emotion_module.initialize()
        assert result is True
        assert emotion_module.initialized
        assert emotion_module.development_level == 0.0
    
    def test_basic_emotions(self, emotion_module):
        """Test basic emotion recognition and processing."""
        emotion_module.initialize()
        
        # Test recognizing a basic emotion
        result = emotion_module.process({
            "action": "recognize_emotion",
            "text": "I am so happy today!",
            "context": {"source": "external", "speaker": "mother"}
        })
        assert result["success"] is True
        assert result["emotion"] == "joy" or result["emotion"] == "happiness"
        assert 0.0 <= result["intensity"] <= 1.0
        
        # Test recognizing another emotion
        result = emotion_module.process({
            "action": "recognize_emotion",
            "text": "I'm feeling very sad about what happened.",
            "context": {"source": "external", "speaker": "mother"}
        })
        assert result["success"] is True
        assert result["emotion"] == "sadness"
        assert 0.0 <= result["intensity"] <= 1.0
    
    def test_emotional_state(self, emotion_module):
        """Test tracking and updating emotional state."""
        emotion_module.initialize()
        
        # Get initial emotional state
        result = emotion_module.process({"action": "get_emotional_state"})
        assert result["success"] is True
        assert "current_emotion" in result
        assert "intensity" in result
        assert "emotional_history" in result
        
        # Process an emotion-inducing event
        result = emotion_module.process({
            "action": "process_event",
            "event": "Mother smiled and gave a compliment",
            "context": {"source": "external", "valence": "positive"}
        })
        assert result["success"] is True
        assert "emotion_change" in result
        
        # Check emotional state after event
        result = emotion_module.process({"action": "get_emotional_state"})
        assert result["current_emotion"] != "neutral"  # Should have changed from neutral
        assert result["intensity"] > 0.0
    
    def test_emotion_development(self, emotion_module):
        """Test emotional development through stages."""
        emotion_module.initialize()
        
        # Check initial development stage
        result = emotion_module.process({"action": "get_emotional_development"})
        assert result["success"] is True
        assert result["stage"] == "basic"
        assert result["available_emotions"] == ["joy", "sadness", "fear", "anger"]
        
        # Develop to intermediate stage
        emotion_module.development_level = 0.4
        emotion_module.update({"emotional_experiences": 100})
        
        result = emotion_module.process({"action": "get_emotional_development"})
        assert result["stage"] == "intermediate"
        assert len(result["available_emotions"]) > 4  # More emotions available
        
        # Develop to advanced stage
        emotion_module.development_level = 0.8
        emotion_module.update({"emotional_experiences": 500})
        
        result = emotion_module.process({"action": "get_emotional_development"})
        assert result["stage"] == "advanced"
        assert "complex_emotions" in result
        assert "emotional_intelligence" in result
    
    def test_emotion_responses(self, emotion_module):
        """Test emotional responses to stimuli."""
        emotion_module.initialize()
        
        # Set up some emotional associations
        associations = [
            {"stimulus": "mother's face", "emotion": "joy", "intensity": 0.8},
            {"stimulus": "loud noise", "emotion": "fear", "intensity": 0.7},
            {"stimulus": "hunger", "emotion": "distress", "intensity": 0.6}
        ]
        
        for assoc in associations:
            emotion_module.process({
                "action": "learn_association",
                **assoc
            })
        
        # Test response to a known stimulus
        result = emotion_module.process({
            "action": "respond_to_stimulus",
            "stimulus": "mother's face"
        })
        assert result["success"] is True
        assert result["response"]["emotion"] == "joy"
        assert result["response"]["intensity"] >= 0.5
        
        # Test response to another stimulus
        result = emotion_module.process({
            "action": "respond_to_stimulus",
            "stimulus": "loud noise"
        })
        assert result["success"] is True
        assert result["response"]["emotion"] == "fear"
    
    def test_emotional_memory_integration(self, emotion_module, memory_module, mock_embedding):
        """Test integration with memory module for emotional memories."""
        emotion_module.initialize()
        memory_module.initialize()
        
        # Connect modules
        emotion_module.connect("memory", memory_module)
        
        # Create an emotional memory
        memory = {
            "text": "Mother hugged me when I was crying",
            "embedding": mock_embedding(1, 512)[0],
            "timestamp": int(time.time()),
            "memory_type": "episodic",
            "emotion": "comfort",
            "emotion_intensity": 0.9,
            "importance": 0.8
        }
        
        # Store via memory module
        memory_id = memory_module.process({
            "action": "store",
            "memory": memory
        })["memory_id"]
        
        # Tag memory with emotion
        result = emotion_module.process({
            "action": "tag_memory",
            "memory_id": memory_id,
            "emotion": "comfort",
            "intensity": 0.9
        })
        assert result["success"] is True
        
        # Retrieve emotional memories
        result = emotion_module.process({
            "action": "retrieve_emotional_memories",
            "emotion": "comfort",
            "limit": 5
        })
        assert result["success"] is True
        assert len(result["memories"]) > 0
        assert any(m["emotion"] == "comfort" for m in result["memories"])
    
    def test_emotional_regulation(self, emotion_module):
        """Test emotional regulation capabilities."""
        emotion_module.initialize()
        
        # Set a strong negative emotion
        emotion_module.process({
            "action": "set_emotion",
            "emotion": "anger",
            "intensity": 0.9,
            "reason": "For testing regulation"
        })
        
        # Check emotional state
        state = emotion_module.process({"action": "get_emotional_state"})
        assert state["current_emotion"] == "anger"
        assert state["intensity"] >= 0.8
        
        # Regulate emotion down
        result = emotion_module.process({
            "action": "regulate_emotion",
            "strategy": "cognitive_reappraisal",
            "target_intensity": 0.3
        })
        assert result["success"] is True
        assert "regulation_effect" in result
        
        # Check state after regulation
        state = emotion_module.process({"action": "get_emotional_state"})
        assert state["intensity"] < 0.8  # Should be reduced
    
    def test_empathy(self, emotion_module):
        """Test empathy capabilities."""
        emotion_module.initialize()
        
        # Set higher development for empathy
        emotion_module.development_level = 0.7
        
        # Test empathizing with another's emotion
        result = emotion_module.process({
            "action": "empathize",
            "target": "mother",
            "observed_emotion": "sadness",
            "context": "Mother looks sad after breaking her favorite cup"
        })
        assert result["success"] is True
        assert "empathetic_response" in result
        assert "mirrored_emotion" in result
        assert result["mirrored_emotion"]["emotion"] == "sadness"
        
        # Check if empathy influenced own emotional state
        state = emotion_module.process({"action": "get_emotional_state"})
        assert "empathy" in str(state).lower()
    
    def test_complex_emotions(self, emotion_module):
        """Test complex and mixed emotions."""
        emotion_module.initialize()
        
        # Set high development level for complex emotions
        emotion_module.development_level = 0.9
        
        # Process complex emotion scenario
        result = emotion_module.process({
            "action": "process_complex_scenario",
            "scenario": "Receiving a gift I wanted but from someone I'm upset with",
            "context": {"relationship": "conflicted", "event_valence": "mixed"}
        })
        assert result["success"] is True
        assert "emotions" in result
        assert len(result["emotions"]) > 1  # Multiple emotions
        
        # Verify presence of complex emotions
        emotions = [e["emotion"] for e in result["emotions"]]
        assert any(e in emotions for e in ["gratitude", "resentment", "confusion", "ambivalence"])
    
    def test_emotional_learning(self, emotion_module):
        """Test emotional learning from experiences."""
        emotion_module.initialize()
        
        # Initial emotional association
        result = emotion_module.process({
            "action": "learn_association",
            "stimulus": "dog",
            "emotion": "neutral",
            "intensity": 0.1
        })
        assert result["success"] is True
        
        # Create a negative experience
        emotion_module.process({
            "action": "process_event",
            "event": "A dog barked loudly and scared me",
            "context": {"stimulus": "dog", "valence": "negative", "intensity": 0.8}
        })
        
        # Check if association changed
        result = emotion_module.process({
            "action": "get_association",
            "stimulus": "dog"
        })
        assert result["success"] is True
        assert result["emotion"] != "neutral"  # Should have changed
        assert result["intensity"] > 0.1  # Should be stronger
        
        # Create positive experiences to change association
        for _ in range(3):
            emotion_module.process({
                "action": "process_event",
                "event": "A friendly dog played with me",
                "context": {"stimulus": "dog", "valence": "positive", "intensity": 0.7}
            })
        
        # Check if association changed again
        result = emotion_module.process({
            "action": "get_association",
            "stimulus": "dog"
        })
        assert result["emotion"] in ["joy", "happiness", "positive"]  # Should be positive now 