"""
Tests for the Consciousness module of the Large Mind Model.
"""

import pytest
import numpy as np
import time
from pathlib import Path

from lmm.core.consciousness import ConsciousnessModule


class TestConsciousnessModule:
    """Test suite for the Consciousness module."""
    
    def test_consciousness_initialization(self, consciousness_module):
        """Test that consciousness module initializes correctly."""
        assert consciousness_module.name == "consciousness"
        assert not consciousness_module.initialized
        
        # Initialize the module
        result = consciousness_module.initialize()
        assert result is True
        assert consciousness_module.initialized
        assert consciousness_module.development_level == 0.0
    
    def test_self_awareness(self, consciousness_module):
        """Test self-awareness functionality."""
        consciousness_module.initialize()
        
        # Test self-inquiry
        result = consciousness_module.process({
            "action": "self_inquiry",
            "query": "What is my current state?"
        })
        assert result["success"] is True
        assert "self_reflection" in result
        assert "current_state" in result
        
        # Test identity
        result = consciousness_module.process({
            "action": "get_identity"
        })
        assert result["success"] is True
        assert "identity" in result
        assert "name" in result["identity"]
        assert "self_concept" in result["identity"]
    
    def test_reflection(self, consciousness_module, memory_module, mock_memory_entries, mock_embedding):
        """Test reflection on memories."""
        consciousness_module.initialize()
        memory_module.initialize()
        
        # Connect the modules
        consciousness_module.connect("memory", memory_module)
        
        # Store memories
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            result = memory_module.process({"action": "store", "memory": entry})
            memory_ids.append(result["memory_id"])
        
        # Reflect on a specific memory
        result = consciousness_module.process({
            "action": "reflect_on_memory",
            "memory_id": memory_ids[0]
        })
        assert result["success"] is True
        assert "reflection" in result
        assert "insights" in result
        
        # Reflect on multiple memories
        result = consciousness_module.process({
            "action": "reflect_on_memories",
            "memory_ids": memory_ids[:3]
        })
        assert result["success"] is True
        assert "reflections" in result
        assert len(result["reflections"]) == 3
    
    def test_introspection(self, consciousness_module, full_mind_model):
        """Test introspection on internal state."""
        consciousness_module.initialize()
        
        # Test introspection on current thoughts
        result = consciousness_module.process({
            "action": "introspect",
            "focus": "thoughts"
        })
        assert result["success"] is True
        assert "introspection" in result
        assert "thoughts" in result
        
        # Test introspection on emotions
        result = consciousness_module.process({
            "action": "introspect",
            "focus": "emotions"
        })
        assert result["success"] is True
        assert "introspection" in result
        assert "emotions" in result
        
        # Test introspection on entire self
        result = consciousness_module.process({
            "action": "introspect",
            "focus": "self"
        })
        assert result["success"] is True
        assert "introspection" in result
        assert "self_analysis" in result
    
    def test_awareness_levels(self, consciousness_module):
        """Test different levels of consciousness awareness."""
        consciousness_module.initialize()
        
        # Start at base level
        state = consciousness_module.get_state()
        assert state["awareness_level"] == "basic"
        
        # Increase development level
        consciousness_module.development_level = 0.3
        result = consciousness_module.process({"action": "update_awareness"})
        assert result["success"] is True
        assert consciousness_module.get_state()["awareness_level"] == "developing"
        
        # Further increase
        consciousness_module.development_level = 0.7
        result = consciousness_module.process({"action": "update_awareness"})
        assert result["success"] is True
        assert consciousness_module.get_state()["awareness_level"] == "advanced"
        
        # Full development
        consciousness_module.development_level = 1.0
        result = consciousness_module.process({"action": "update_awareness"})
        assert result["success"] is True
        assert consciousness_module.get_state()["awareness_level"] == "complete"
    
    def test_mental_models(self, consciousness_module):
        """Test mental model building and updating."""
        consciousness_module.initialize()
        
        # Create a mental model
        result = consciousness_module.process({
            "action": "create_mental_model",
            "domain": "physics",
            "concepts": ["gravity", "mass", "acceleration"],
            "relationships": {
                "gravity": ["mass", "acceleration"],
                "mass": ["gravity"],
                "acceleration": ["gravity"]
            }
        })
        assert result["success"] is True
        assert "model_id" in result
        
        model_id = result["model_id"]
        
        # Retrieve the model
        result = consciousness_module.process({
            "action": "get_mental_model",
            "model_id": model_id
        })
        assert result["success"] is True
        assert result["model"]["domain"] == "physics"
        
        # Update the model
        result = consciousness_module.process({
            "action": "update_mental_model",
            "model_id": model_id,
            "add_concepts": ["force"],
            "add_relationships": {
                "force": ["mass", "acceleration"],
                "mass": ["force"],
                "acceleration": ["force"]
            }
        })
        assert result["success"] is True
        
        # Verify update
        result = consciousness_module.process({
            "action": "get_mental_model",
            "model_id": model_id
        })
        assert "force" in result["model"]["concepts"]
    
    def test_metacognition(self, consciousness_module, full_mind_model):
        """Test metacognitive processes."""
        consciousness_module.initialize()
        
        # Test thinking about thinking
        result = consciousness_module.process({
            "action": "metacognition",
            "focus": "learning_process"
        })
        assert result["success"] is True
        assert "metacognitive_analysis" in result
        
        # Test cognitive monitoring
        result = consciousness_module.process({
            "action": "monitor_cognition",
            "duration": 5  # seconds
        })
        assert result["success"] is True
        assert "cognitive_patterns" in result
        assert "awareness_report" in result
    
    def test_self_improvement(self, consciousness_module):
        """Test self-improvement mechanisms."""
        consciousness_module.initialize()
        
        # Create initial self-assessment
        result = consciousness_module.process({
            "action": "self_assessment"
        })
        assert result["success"] is True
        assert "strengths" in result
        assert "weaknesses" in result
        
        initial_assessment = result
        
        # Generate improvement plan
        result = consciousness_module.process({
            "action": "create_improvement_plan",
            "based_on": initial_assessment
        })
        assert result["success"] is True
        assert "plan" in result
        assert "goals" in result["plan"]
        assert "steps" in result["plan"]
        
        # Implement improvement
        result = consciousness_module.process({
            "action": "implement_improvement",
            "area": result["plan"]["goals"][0]
        })
        assert result["success"] is True
        
        # Verify improvement
        new_assessment = consciousness_module.process({
            "action": "self_assessment"
        })
        # Should show some difference from initial assessment
        assert new_assessment["strengths"] != initial_assessment["strengths"] or \
               new_assessment["weaknesses"] != initial_assessment["weaknesses"]
    
    def test_attention_focus(self, consciousness_module, full_mind_model):
        """Test attention focus mechanisms."""
        consciousness_module.initialize()
        
        # Set initial attention on emotion module
        result = consciousness_module.process({
            "action": "focus_attention",
            "target_module": "emotion",
            "intensity": 0.8
        })
        assert result["success"] is True
        
        # Get current focus
        result = consciousness_module.process({
            "action": "get_attention_focus"
        })
        assert result["success"] is True
        assert result["focus"]["target_module"] == "emotion"
        assert result["focus"]["intensity"] == 0.8
        
        # Shift attention to another module
        result = consciousness_module.process({
            "action": "focus_attention",
            "target_module": "language",
            "intensity": 0.6
        })
        assert result["success"] is True
        
        # Verify focus shift
        result = consciousness_module.process({
            "action": "get_attention_focus"
        })
        assert result["focus"]["target_module"] == "language"
        assert result["focus"]["intensity"] == 0.6 