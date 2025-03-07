"""
Tests for the Memory module of the Large Mind Model.
"""

import pytest
import numpy as np
import faiss
from pathlib import Path
import time

from lmm.core.memory import MemoryModule


class TestMemoryModule:
    """Test suite for the Memory module."""
    
    def test_memory_initialization(self, memory_module):
        """Test that memory module initializes correctly."""
        assert memory_module.name == "memory"
        assert not memory_module.initialized
        
        # Initialize the module
        result = memory_module.initialize()
        assert result is True
        assert memory_module.initialized
        assert memory_module.development_level == 0.0
    
    def test_memory_storage(self, memory_module, mock_embedding):
        """Test storing memories."""
        memory_module.initialize()
        
        # Create a test memory
        memory = {
            "text": "This is a test memory",
            "embedding": mock_embedding(1, 512)[0],
            "timestamp": int(time.time()),
            "memory_type": "episodic",
            "emotion": "neutral",
            "importance": 0.7
        }
        
        # Store the memory
        result = memory_module.process({"action": "store", "memory": memory})
        assert result["success"] is True
        assert result["memory_id"] is not None
        
        # Verify memory was stored
        memory_id = result["memory_id"]
        stored_memories = memory_module.get_state().get("stored_memories", 0)
        assert stored_memories > 0
    
    def test_memory_retrieval(self, memory_module, mock_embedding, mock_memory_entries):
        """Test retrieving memories."""
        memory_module.initialize()
        
        # Store multiple test memories
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            result = memory_module.process({"action": "store", "memory": entry})
            memory_ids.append(result["memory_id"])
        
        # Test direct retrieval by ID
        result = memory_module.process({"action": "retrieve", "memory_id": memory_ids[0]})
        assert result["success"] is True
        assert result["memory"]["text"] == mock_memory_entries[0]["text"]
        
        # Test semantic search retrieval
        query_embedding = mock_embedding(1, 512)[0]
        result = memory_module.process({
            "action": "search", 
            "embedding": query_embedding,
            "limit": 3
        })
        assert result["success"] is True
        assert len(result["memories"]) <= 3
        
        # Test filtering by memory type
        result = memory_module.process({
            "action": "filter",
            "memory_type": "semantic",
            "limit": 5
        })
        assert result["success"] is True
        assert all(m["memory_type"] == "semantic" for m in result["memories"])
        
        # Test time-based retrieval
        result = memory_module.process({
            "action": "timerange",
            "start_time": 1646000000,
            "end_time": 1647000000,
            "limit": 10
        })
        assert result["success"] is True
        assert all(1646000000 <= m["timestamp"] <= 1647000000 for m in result["memories"])
    
    def test_memory_forgetting(self, memory_module, mock_embedding, mock_memory_entries):
        """Test memory forgetting mechanism."""
        memory_module.initialize()
        
        # Store memories
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            result = memory_module.process({"action": "store", "memory": entry})
            memory_ids.append(result["memory_id"])
        
        # Test forgetting a specific memory
        result = memory_module.process({"action": "forget", "memory_id": memory_ids[0]})
        assert result["success"] is True
        
        # Verify memory was forgotten
        result = memory_module.process({"action": "retrieve", "memory_id": memory_ids[0]})
        assert result["success"] is False
        
        # Test importance-based forgetting
        result = memory_module.process({
            "action": "forget_by_importance",
            "threshold": 0.6  # Forget memories with importance < 0.6
        })
        assert result["success"] is True
        
        # Verify low-importance memories were forgotten
        all_memories = memory_module.process({"action": "list_all"})["memories"]
        assert all(m["importance"] >= 0.6 for m in all_memories)
    
    def test_memory_persistence(self, memory_module, mock_embedding, temp_memory_dir):
        """Test saving and loading memory state."""
        memory_module.initialize()
        
        # Create and store a memory
        memory = {
            "text": "This should persist after saving and loading",
            "embedding": mock_embedding(1, 512)[0],
            "timestamp": int(time.time()),
            "memory_type": "semantic",
            "emotion": "curiosity",
            "importance": 0.9
        }
        memory_module.process({"action": "store", "memory": memory})
        
        # Save state
        save_path = temp_memory_dir / "memory_test"
        save_path.mkdir(exist_ok=True)
        result = memory_module.save_state(str(save_path))
        assert result is True
        
        # Create a new memory module and load the state
        new_module = MemoryModule(name="memory", config=memory_module.config)
        new_module.initialize()
        result = new_module.load_state(str(save_path))
        assert result is True
        
        # Verify memories were restored
        result = new_module.process({"action": "list_all"})
        assert any(m["text"] == memory["text"] for m in result["memories"])
    
    def test_memory_consolidation(self, memory_module, mock_embedding, mock_memory_entries):
        """Test memory consolidation process (converting short-term to long-term)."""
        memory_module.initialize()
        
        # Store memories as short-term
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            entry["memory_type"] = "short_term"
            result = memory_module.process({"action": "store", "memory": entry})
            memory_ids.append(result["memory_id"])
        
        # Run consolidation process
        result = memory_module.process({"action": "consolidate"})
        assert result["success"] is True
        assert result["consolidated_count"] > 0
        
        # Verify some memories were converted to long-term
        result = memory_module.process({
            "action": "filter",
            "memory_type": "long_term"
        })
        assert len(result["memories"]) > 0
    
    def test_memory_associations(self, memory_module, mock_embedding, mock_memory_entries):
        """Test creating and retrieving memory associations."""
        memory_module.initialize()
        
        # Store memories
        memory_ids = []
        for entry in mock_memory_entries:
            entry["embedding"] = mock_embedding(1, 512)[0]
            result = memory_module.process({"action": "store", "memory": entry})
            memory_ids.append(result["memory_id"])
        
        # Create associations between memories
        result = memory_module.process({
            "action": "associate",
            "source_id": memory_ids[0],
            "target_id": memory_ids[1],
            "association_type": "related_to",
            "strength": 0.8
        })
        assert result["success"] is True
        
        # Create another association
        result = memory_module.process({
            "action": "associate",
            "source_id": memory_ids[0],
            "target_id": memory_ids[2],
            "association_type": "leads_to",
            "strength": 0.7
        })
        assert result["success"] is True
        
        # Retrieve associations for a memory
        result = memory_module.process({
            "action": "get_associations",
            "memory_id": memory_ids[0]
        })
        assert result["success"] is True
        assert len(result["associations"]) == 2
        
        # Verify association properties
        associations = result["associations"]
        assert any(a["target_id"] == memory_ids[1] and a["association_type"] == "related_to" for a in associations)
        assert any(a["target_id"] == memory_ids[2] and a["association_type"] == "leads_to" for a in associations)
    
    def test_memory_emotional_weighting(self, memory_module, mock_embedding):
        """Test that emotional memories get properly weighted."""
        memory_module.initialize()
        
        # Store memories with varying emotional intensities
        memories = [
            {
                "text": "A neutral fact about the world",
                "embedding": mock_embedding(1, 512)[0],
                "timestamp": int(time.time()),
                "memory_type": "semantic",
                "emotion": "neutral",
                "emotion_intensity": 0.1,
                "importance": 0.5
            },
            {
                "text": "A slightly happy memory",
                "embedding": mock_embedding(1, 512)[0],
                "timestamp": int(time.time()),
                "memory_type": "episodic",
                "emotion": "joy",
                "emotion_intensity": 0.4,
                "importance": 0.5
            },
            {
                "text": "An intensely scary experience",
                "embedding": mock_embedding(1, 512)[0],
                "timestamp": int(time.time()),
                "memory_type": "episodic",
                "emotion": "fear",
                "emotion_intensity": 0.9,
                "importance": 0.5
            }
        ]
        
        for memory in memories:
            memory_module.process({"action": "store", "memory": memory})
        
        # Retrieve memories by emotional salience
        result = memory_module.process({
            "action": "retrieve_by_emotion",
            "emotion": "any",
            "sort_by": "intensity",
            "limit": 3
        })
        
        # The intensely emotional memory should be ranked first
        assert result["memories"][0]["emotion"] == "fear"
        assert result["memories"][0]["emotion_intensity"] == 0.9
        
        # Filter by specific emotion
        result = memory_module.process({
            "action": "retrieve_by_emotion",
            "emotion": "joy",
            "limit": 1
        })
        assert result["memories"][0]["emotion"] == "joy" 