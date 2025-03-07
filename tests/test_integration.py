"""
Integration tests for the Large Mind Model.

These tests verify how different modules work together as an integrated system.
"""

import pytest
import time
import json
from pathlib import Path

from lmm.development.stages import DevelopmentManager


class TestSystemIntegration:
    """Integration test suite for the Large Mind Model system."""
    
    def test_module_communication(self, full_mind_model):
        """Test communication between modules."""
        # Initialize all modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
        
        # Test communication from language to memory
        language_module = full_mind_model["language"]
        memory_module = full_mind_model["memory"]
        
        # Process a language input that should create a memory
        result = language_module.process({
            "action": "process_input",
            "text": "The sky is blue",
            "source": "mother",
            "store_in_memory": True
        })
        
        assert result["success"] is True
        assert "memory_id" in result
        
        # Verify the memory was stored
        memory_result = memory_module.process({
            "action": "retrieve",
            "memory_id": result["memory_id"]
        })
        
        assert memory_result["success"] is True
        assert "sky" in memory_result["memory"]["text"].lower()
        assert "blue" in memory_result["memory"]["text"].lower()
    
    def test_emotional_language_integration(self, full_mind_model):
        """Test integration between emotion and language modules."""
        # Initialize modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
        
        language_module = full_mind_model["language"]
        emotion_module = full_mind_model["emotion"]
        
        # Set a development level to enable processing
        language_module.development_level = 0.5
        emotion_module.development_level = 0.5
        
        # Process an emotional statement
        result = language_module.process({
            "action": "process_input",
            "text": "I am very happy today!",
            "source": "mother",
            "analyze_emotion": True
        })
        
        assert result["success"] is True
        assert "emotion_analysis" in result
        assert result["emotion_analysis"]["emotion"] in ["joy", "happiness"]
        
        # Verify emotion module state was updated
        emotion_state = emotion_module.process({
            "action": "get_emotional_state"
        })
        
        assert emotion_state["success"] is True
        # The emotion might be influenced but not identical
        assert emotion_state["current_emotion"] != "neutral"
    
    def test_consciousness_reflection(self, full_mind_model, mock_memory_entries, mock_embedding):
        """Test consciousness module's reflection on memories."""
        # Initialize modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
                
        # Set development levels
        for name, module in full_mind_model.items():
            if hasattr(module, "development_level"):
                module.development_level = 0.6
        
        consciousness_module = full_mind_model["consciousness"]
        memory_module = full_mind_model["memory"]
        
        # Store memories
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            result = memory_module.process({
                "action": "store",
                "memory": entry
            })
            memory_ids.append(result["memory_id"])
        
        # Reflect on memories
        result = consciousness_module.process({
            "action": "reflect_on_memories",
            "memory_ids": memory_ids
        })
        
        assert result["success"] is True
        assert "reflections" in result
        assert len(result["reflections"]) == len(memory_ids)
    
    def test_thought_imagination_integration(self, full_mind_model):
        """Test integration between thought and imagination modules."""
        # Initialize modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
                
        # Set development levels
        for name, module in full_mind_model.items():
            if hasattr(module, "development_level"):
                module.development_level = 0.7
        
        thought_module = full_mind_model["thought"]
        imagination_module = full_mind_model["imagination"]
        
        # Generate a thought
        thought_result = thought_module.process({
            "action": "generate_thought",
            "focus": "creative",
            "context": {"recent_experiences": ["Saw a bird flying"]}
        })
        
        assert thought_result["success"] is True
        assert "thought" in thought_result
        
        # Use thought to imagine something
        imagination_result = imagination_module.process({
            "action": "imagine",
            "based_on_thought": thought_result["thought"],
            "elaboration_level": 0.8
        })
        
        assert imagination_result["success"] is True
        assert "imagination" in imagination_result
        assert imagination_result["imagination"] != thought_result["thought"]
        assert len(imagination_result["imagination"]) > len(thought_result["thought"])
    
    def test_developmental_progression(self, full_mind_model, developmental_stages, test_config):
        """Test a complete developmental progression with all modules."""
        # Create development manager
        dev_manager = DevelopmentManager(
            config=test_config.development.dict(),
            stages=developmental_stages,
            mind_model=full_mind_model
        )
        dev_manager.initialize()
        
        # Initialize all modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
        
        # Simulate development through stages
        stages = ["prenatal", "infancy", "childhood"]
        
        for stage in stages:
            # Define development levels based on stage
            if stage == "prenatal":
                levels = {
                    "memory": 0.05,
                    "language": 0.03,
                    "emotion": 0.04,
                    "consciousness": 0.02,
                    "social": 0.01,
                    "thought": 0.02,
                    "imagination": 0.01
                }
            elif stage == "infancy":
                levels = {
                    "memory": 0.25,
                    "language": 0.20,
                    "emotion": 0.22,
                    "consciousness": 0.18,
                    "social": 0.15,
                    "thought": 0.17,
                    "imagination": 0.16
                }
            elif stage == "childhood":
                levels = {
                    "memory": 0.55,
                    "language": 0.50,
                    "emotion": 0.48,
                    "consciousness": 0.45,
                    "social": 0.42,
                    "thought": 0.44,
                    "imagination": 0.40
                }
            
            # Update development levels
            dev_manager.update_development_levels(levels)
            
            # Progress to next stage
            if dev_manager.current_stage != stage:
                result = dev_manager.set_stage(stage)
                assert result["success"] is True
            
            # Apply development to modules
            result = dev_manager.apply_development_to_modules()
            assert result["success"] is True
            
            # Check abilities at each stage
            stage_capabilities = dev_manager.get_stage_capabilities(stage)
            assert stage_capabilities["success"] is True
            
            # Test some basic functionality appropriate to this stage
            if stage == "prenatal":
                # Basic memory storage without linguistic processing
                memory_module = full_mind_model["memory"]
                emotion_module = full_mind_model["emotion"]
                
                # Store a simple memory
                result = memory_module.process({
                    "action": "store",
                    "memory": {
                        "text": "First sensory experience",
                        "embedding": None,
                        "timestamp": int(time.time()),
                        "memory_type": "sensory",
                        "emotion": "neutral",
                        "importance": 0.7
                    }
                })
                assert result["success"] is True
                
                # React to a stimulus
                result = emotion_module.process({
                    "action": "respond_to_stimulus",
                    "stimulus": "warm touch"
                })
                assert result["success"] is True
                
            elif stage == "infancy":
                # Test basic language processing
                language_module = full_mind_model["language"]
                
                # Process simple words
                result = language_module.process({
                    "action": "process_input",
                    "text": "mama",
                    "source": "self"
                })
                assert result["success"] is True
                
                # Test simple emotional response
                emotion_module = full_mind_model["emotion"]
                result = emotion_module.process({
                    "action": "recognize_emotion",
                    "text": "Good job!",
                    "context": {"source": "mother", "tone": "excited"}
                })
                assert result["success"] is True
                assert result["emotion"] in ["joy", "happiness"]
                
            elif stage == "childhood":
                # Test more complex language and consciousness
                language_module = full_mind_model["language"]
                consciousness_module = full_mind_model["consciousness"]
                
                # Process a sentence
                result = language_module.process({
                    "action": "process_input",
                    "text": "I want to play with the red ball",
                    "source": "self"
                })
                assert result["success"] is True
                
                # Test self-awareness
                result = consciousness_module.process({
                    "action": "self_inquiry",
                    "query": "What do I want?"
                })
                assert result["success"] is True
                assert "self_reflection" in result
    
    def test_mother_child_interaction(self, full_mind_model):
        """Test interaction between mother LLM and the mind model."""
        # Initialize modules
        for name, module in full_mind_model.items():
            if hasattr(module, "initialize"):
                module.initialize()
                
        # Set development levels for early childhood
        for name, module in full_mind_model.items():
            if hasattr(module, "development_level"):
                module.development_level = 0.4
        
        mother_llm = full_mind_model["mother"]
        language_module = full_mind_model["language"]
        memory_module = full_mind_model["memory"]
        emotion_module = full_mind_model["emotion"]
        
        # Simulate a mother-child interaction
        # 1. Mother asks a question
        mother_utterance = mother_llm.process({
            "action": "initiate_interaction",
            "interaction_type": "question",
            "topic": "colors",
            "context": {
                "child_development_level": 0.4,
                "child_linguistic_stage": "simple_sentences"
            }
        })
        
        assert mother_utterance["success"] is True
        assert "response" in mother_utterance
        
        # 2. Child processes the question through language module
        language_result = language_module.process({
            "action": "process_input",
            "text": mother_utterance["response"],
            "source": "mother",
            "analyze_emotion": True
        })
        
        assert language_result["success"] is True
        assert "understanding" in language_result
        
        # 3. Child generates a response
        child_response = language_module.process({
            "action": "generate_response",
            "to_utterance": mother_utterance["response"],
            "context": {
                "emotional_state": emotion_module.get_state(),
                "memories": memory_module.process({"action": "list_recent", "limit": 5})["memories"]
            }
        })
        
        assert child_response["success"] is True
        assert "response" in child_response
        
        # 4. Mother responds to the child
        mother_reply = mother_llm.process({
            "action": "respond",
            "to_text": child_response["response"],
            "context": {
                "child_development_level": 0.4,
                "previous_utterance": mother_utterance["response"],
                "interaction_history": [
                    {"speaker": "mother", "text": mother_utterance["response"]},
                    {"speaker": "child", "text": child_response["response"]}
                ]
            }
        })
        
        assert mother_reply["success"] is True
        assert "response" in mother_reply
        
        # 5. Verify the interaction created memories
        memory_result = memory_module.process({
            "action": "filter",
            "memory_type": "conversation",
            "limit": 5
        })
        
        assert memory_result["success"] is True
        assert len(memory_result["memories"]) > 0 