"""
Memory Persistence module for the Large Mind Model (LMM).

This module implements the memory system for the LMM, including semantic
and episodic memory, with support for storage, retrieval, and forgetting.
"""
import os
import json
import time
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
from pydantic import BaseModel, Field

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.memory.vector_store import VectorStore

logger = get_logger("lmm.memory.persistence")

class MemoryType(str, Enum):
    """Types of memories in the LMM memory system."""
    EPISODIC = "episodic"  # Personal experiences and events
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and how to do things
    EMOTIONAL = "emotional"  # Emotional experiences and associations

class MemoryImportance(str, Enum):
    """Importance levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Memory(BaseModel):
    """Model for a memory in the LMM memory system."""
    content: str = Field(..., description="Content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    importance: MemoryImportance = Field(MemoryImportance.MEDIUM, description="Importance of the memory")
    created_at: datetime = Field(default_factory=datetime.now, description="When the memory was created")
    last_accessed: Optional[datetime] = Field(None, description="When the memory was last accessed")
    access_count: int = Field(0, description="Number of times the memory has been accessed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    vector_store_id: Optional[int] = Field(None, description="ID in the vector store")

class MemoryManager:
    """
    Manages the memory system for the LMM.
    
    This class provides methods for storing, retrieving, and managing
    different types of memories, with support for importance-based
    retention and forgetting.
    """
    
    def __init__(self, vector_store_dimension: int = 1024):
        """
        Initialize the Memory Manager.
        
        Args:
            vector_store_dimension: Dimension of the embedding vectors
        """
        config = get_config()
        self.vector_store = VectorStore(dimension=vector_store_dimension)
        self.vector_store_path = os.path.normpath(config.memory.vector_db_path)
        
        # Try to load existing vector store
        if os.path.exists(self.vector_store_path):
            success = self.vector_store.load(self.vector_store_path)
            if success:
                logger.info(f"Loaded existing vector store from {self.vector_store_path}")
            else:
                logger.warning(f"Failed to load vector store from {self.vector_store_path}")
        
        # Initialize memory collections
        self.memories: Dict[int, Memory] = {}
        self.memory_count_by_type: Dict[MemoryType, int] = {
            memory_type: 0 for memory_type in MemoryType
        }
        
        logger.info("Initialized Memory Manager")
    
    def add_memory(
        self, 
        content: str, 
        memory_type: Union[MemoryType, str],
        importance: Union[MemoryImportance, str] = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a memory to the memory system.
        
        Args:
            content: Content of the memory
            memory_type: Type of memory
            importance: Importance of the memory
            metadata: Additional metadata
            
        Returns:
            ID of the added memory
        """
        # Convert string types to enum values if needed
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        
        if isinstance(importance, str):
            importance = MemoryImportance(importance)
        
        # Create memory object
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {}
        )
        
        # Add to vector store
        vector_metadata = {
            "memory_type": memory_type,
            "importance": importance,
            "created_at": memory.created_at.isoformat()
        }
        if metadata:
            vector_metadata.update(metadata)
        
        vector_store_id = self.vector_store.add(content, vector_metadata)
        memory.vector_store_id = vector_store_id
        
        # Add to memory collection
        self.memories[vector_store_id] = memory
        self.memory_count_by_type[memory_type] += 1
        
        # Save vector store periodically
        if self.vector_store.count() % 10 == 0:
            self._save_vector_store()
        
        logger.debug(f"Added {memory_type} memory: {content[:50]}...")
        return vector_store_id
    
    def retrieve_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory object, or None if not found
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Update access information
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            return memory
        return None
    
    def search_memories(
        self, 
        query: str, 
        memory_type: Optional[Union[MemoryType, str]] = None,
        min_importance: Optional[Union[MemoryImportance, str]] = None,
        limit: int = 5
    ) -> List[Memory]:
        """
        Search for memories similar to the query.
        
        Args:
            query: Query text
            memory_type: Optional filter by memory type
            min_importance: Optional minimum importance level
            limit: Maximum number of results
            
        Returns:
            List of matching Memory objects
        """
        # Convert string types to enum values if needed
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        
        if isinstance(min_importance, str):
            min_importance = MemoryImportance(min_importance)
        
        # Define filter function
        def filter_memory(result):
            if memory_type and result.get("memory_type") != memory_type:
                return False
            
            if min_importance:
                importance_levels = [imp.value for imp in MemoryImportance]
                result_importance = result.get("importance", MemoryImportance.MEDIUM.value)
                if importance_levels.index(result_importance) < importance_levels.index(min_importance.value):
                    return False
            
            return True
        
        # Search vector store
        results = self.vector_store.search(query, k=limit * 2, filter_fn=filter_memory)
        
        # Convert to Memory objects and update access information
        memories = []
        for result in results[:limit]:
            memory_id = result["index_id"]
            memory = self.memories.get(memory_id)
            
            if memory:
                # Update access information
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                
                # Add similarity score to metadata
                memory.metadata["similarity"] = result["similarity"]
                
                memories.append(memory)
        
        logger.debug(f"Found {len(memories)} memories for query: {query[:50]}...")
        return memories
    
    def forget_memories(
        self,
        older_than_days: Optional[int] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        max_importance: Optional[Union[MemoryImportance, str]] = MemoryImportance.LOW
    ) -> int:
        """
        Forget (remove) memories based on criteria.
        
        Args:
            older_than_days: Only forget memories older than this many days
            memory_type: Only forget memories of this type
            max_importance: Only forget memories with importance up to this level
            
        Returns:
            Number of memories forgotten
        """
        # Convert string types to enum values if needed
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        
        if isinstance(max_importance, str):
            max_importance = MemoryImportance(max_importance)
        
        # Calculate cutoff date if specified
        cutoff_date = None
        if older_than_days is not None:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        # Get importance level index for comparison
        importance_levels = [imp.value for imp in MemoryImportance]
        max_importance_index = importance_levels.index(max_importance.value)
        
        # Identify memories to forget
        memories_to_forget = []
        for memory_id, memory in self.memories.items():
            # Check memory type
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Check importance
            memory_importance_index = importance_levels.index(memory.importance.value)
            if memory_importance_index > max_importance_index:
                continue
            
            # Check age
            if cutoff_date and memory.created_at > cutoff_date:
                continue
            
            memories_to_forget.append(memory_id)
        
        # Forget memories
        for memory_id in memories_to_forget:
            memory = self.memories.pop(memory_id, None)
            if memory:
                self.memory_count_by_type[memory.memory_type] -= 1
        
        # Note: We don't actually remove items from the vector store,
        # as FAISS doesn't support removal. Instead, we just remove them
        # from our memory collection, effectively "forgetting" them.
        
        logger.info(f"Forgot {len(memories_to_forget)} memories")
        return len(memories_to_forget)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        # Count memories by type
        memory_types = {
            memory_type.value: 0 for memory_type in MemoryType
        }
        for memory in self.memories.values():
            memory_types[memory.memory_type.value] += 1
        
        # Count memories by importance
        memory_importance = {
            importance.value: 0 for importance in MemoryImportance
        }
        for memory in self.memories.values():
            memory_importance[memory.importance.value] += 1
        
        # Calculate access statistics
        access_stats = {
            "never_accessed": 0,
            "accessed_today": 0,
            "accessed_this_week": 0,
            "accessed_this_month": 0
        }
        
        now = datetime.now()
        for memory in self.memories.values():
            if memory.last_accessed is None:
                access_stats["never_accessed"] += 1
            else:
                days_since_access = (now - memory.last_accessed).days
                if days_since_access < 1:
                    access_stats["accessed_today"] += 1
                if days_since_access < 7:
                    access_stats["accessed_this_week"] += 1
                if days_since_access < 30:
                    access_stats["accessed_this_month"] += 1
        
        # Compute average access count
        total_accesses = sum(memory.access_count for memory in self.memories.values())
        avg_access_count = total_accesses / len(self.memories) if self.memories else 0
        
        # Create comprehensive stats dictionary
        stats = {
            "total_memories": len(self.memories),
            "memory_types": memory_types,
            "memory_importance": memory_importance,
            "access_stats": access_stats,
            "avg_access_count": avg_access_count,
            "vector_store_size": self.vector_store.count(),
            "oldest_memory": min((memory.created_at for memory in self.memories.values()), default=None),
            "newest_memory": max((memory.created_at for memory in self.memories.values()), default=None)
        }
        
        # Convert datetime objects to strings for JSON serialization
        if stats["oldest_memory"]:
            stats["oldest_memory"] = stats["oldest_memory"].isoformat()
        if stats["newest_memory"]:
            stats["newest_memory"] = stats["newest_memory"].isoformat()
        
        return stats
    
    def _save_vector_store(self) -> None:
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save(self.vector_store_path)
    
    def save(self) -> None:
        """Save the memory system to disk."""
        self._save_vector_store()
        logger.info("Saved memory system")
    
    def clear(self) -> None:
        """Clear the memory system."""
        self.vector_store.clear()
        self.memories = {}
        self.memory_count_by_type = {
            memory_type: 0 for memory_type in MemoryType
        }
        logger.warning("Cleared memory system") 