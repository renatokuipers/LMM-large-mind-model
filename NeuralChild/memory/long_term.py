"""
Long-term memory implementation for the NeuralChild system.
Handles persistent storage of consolidated memories across domains.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Generic, TypeVar, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import numpy as np
from pydantic import ValidationError

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("Faiss not available; vector similarity search will be disabled")

from ..models.memory_models import (
    MemoryItem, LongTermMemory, LongTermMemoryConfig, LongTermMemoryDomain,
    Episode, MemoryType, MemoryAttributes, MemoryAccessibility, MemoryStage
)
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic memory content
T = TypeVar('T')

class LongTermMemorySystem:
    """
    Long-term memory system for persistent storage across memory domains.
    Implements forgetting curves, retrieval mechanics, and domain organization.
    """
    
    def __init__(
        self,
        model: Optional[LongTermMemory] = None,
        config_override: Optional[LongTermMemoryConfig] = None,
        enable_vector_search: bool = None
    ):
        """
        Initialize the long-term memory system.
        
        Args:
            model: Optional existing LongTermMemory model
            config_override: Optional configuration override
            enable_vector_search: Whether to enable vector-based similarity search
        """
        if model is None:
            config_params = config.MEMORY["long_term_memory"]
            memory_config = config_override or LongTermMemoryConfig(
                consolidation_time=config_params["consolidation_time_seconds"],
                retrieval_difficulty_factor=config_params["retrieval_difficulty_factor"],
                forgetting_curve_factor=config_params["forgetting_curve_factor"],
                semantic_link_strength=config_params["semantic_link_strength"],
                emotional_persistence_factor=config_params["emotional_persistence_factor"]
            )
            self.model = LongTermMemory(config=memory_config)
        else:
            self.model = model
            
        self._last_update = time.time()
        
        # Vector search configuration
        if enable_vector_search is None:
            enable_vector_search = config.MEMORY.get("enable_vector_storage", False)
        
        self._vector_search_enabled = enable_vector_search and FAISS_AVAILABLE
        self._vector_indices = {}
        self._item_to_vector_idx = {}
        
        if self._vector_search_enabled:
            self._init_vector_indices()
            
        logger.debug("Initialized long-term memory system")
    
    def _init_vector_indices(self) -> None:
        """Initialize FAISS indices for vector similarity search"""
        if not self._vector_search_enabled:
            return
            
        embedding_dim = config.MEMORY.get("embedding_dimensions", 384)
        
        # Create one index per domain
        for domain in LongTermMemoryDomain:
            self._vector_indices[domain] = faiss.IndexFlatL2(embedding_dim)
            self._item_to_vector_idx[domain] = {}
            
        logger.debug(f"Initialized vector indices with dimension {embedding_dim}")
    
    def add_item(self, item: MemoryItem, domain: LongTermMemoryDomain) -> bool:
        """
        Add a memory item to long-term memory.
        
        Args:
            item: The memory item to add
            domain: The domain to add the item to
            
        Returns:
            True if the item was added, False otherwise
        """
        # Update attributes for long-term storage
        item.attributes.stage = MemoryStage.STABLE
        
        # Add to long-term memory
        result = self.model.add_item(item, domain)
        
        # Add to vector index if enabled and embedding exists
        if result and self._vector_search_enabled and item.embedding is not None:
            self._add_to_vector_index(item.id, domain, item.embedding)
        
        logger.debug(f"Added item to long-term memory domain {domain}: {item.id}")
        return result
    
    def add_episode(self, episode: Episode) -> bool:
        """
        Add a consolidated episode to long-term memory.
        
        Args:
            episode: The episode to add
            
        Returns:
            True if the episode was added, False otherwise
        """
        # Mark as consolidated
        episode.is_consolidated = True
        
        # Add to long-term memory
        result = self.model.add_episode(episode)
        
        # Add to vector index if enabled and embedding exists
        if result and self._vector_search_enabled and episode.embedding is not None:
            # Episodes don't belong to a specific domain, so we'll use a special key
            if "EPISODES" not in self._vector_indices:
                embedding_dim = len(episode.embedding)
                self._vector_indices["EPISODES"] = faiss.IndexFlatL2(embedding_dim)
                self._item_to_vector_idx["EPISODES"] = {}
            
            vector_idx = len(self._item_to_vector_idx["EPISODES"])
            self._item_to_vector_idx["EPISODES"][episode.id] = vector_idx
            
            # Convert embedding to numpy array and add to index
            embedding_np = np.array([episode.embedding], dtype=np.float32)
            self._vector_indices["EPISODES"].add(embedding_np)
        
        logger.debug(f"Added episode to long-term memory: {episode.id}")
        return result
    
    def _add_to_vector_index(self, item_id: UUID, domain: LongTermMemoryDomain, 
                            embedding: List[float]) -> None:
        """
        Add an item's embedding to the vector index.
        
        Args:
            item_id: ID of the item
            domain: Memory domain
            embedding: Vector embedding
        """
        if not self._vector_search_enabled:
            return
            
        # Get the next index
        vector_idx = len(self._item_to_vector_idx[domain])
        self._item_to_vector_idx[domain][item_id] = vector_idx
        
        # Convert embedding to numpy array and add to index
        embedding_np = np.array([embedding], dtype=np.float32)
        self._vector_indices[domain].add(embedding_np)
    
    def get_item(self, item_id: UUID) -> Optional[MemoryItem]:
        """
        Retrieve an item from long-term memory.
        Updates the item's access time and rehearsal count.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        return self.model.get_item(item_id)
    
    def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """
        Retrieve an episode from long-term memory.
        Updates the episode's access time.
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            The episode if found, None otherwise
        """
        return self.model.get_episode(episode_id)
    
    def get_domain_items(self, domain: LongTermMemoryDomain) -> List[MemoryItem]:
        """
        Get all items in a specific memory domain.
        
        Args:
            domain: The memory domain to retrieve items from
            
        Returns:
            List of memory items in the domain
        """
        return self.model.get_domain_items(domain)
    
    def update_memory_strength(self, item_id: UUID, strength_delta: float) -> bool:
        """
        Update the strength of a memory (e.g., after retrieval).
        
        Args:
            item_id: ID of the memory item
            strength_delta: Change in strength (positive or negative)
            
        Returns:
            True if the memory was updated, False otherwise
        """
        return self.model.update_memory_strength(item_id, strength_delta)
    
    def apply_forgetting_curve(self, elapsed_seconds: float) -> None:
        """
        Apply forgetting curve to all memories based on elapsed time.
        
        Args:
            elapsed_seconds: Time elapsed since last update
        """
        forgetting_rate = self.model.config.forgetting_curve_factor * elapsed_seconds
        
        # Apply to memory items
        for item_id, item in self.model.items.items():
            # Calculate time since last access
            time_since_access = (datetime.now() - item.attributes.last_accessed).total_seconds()
            
            # Apply Ebbinghaus forgetting curve with rehearsal and emotional modulation
            decay_factor = forgetting_rate * np.exp(-item.attributes.rehearsal_count * 0.1)
            
            # Emotional memories decay more slowly
            if item.attributes.emotional_intensity > 0.5:
                emotional_factor = 1.0 - (item.attributes.emotional_intensity * 
                                         self.model.config.emotional_persistence_factor)
                decay_factor *= emotional_factor
            
            # Calculate new strength
            new_strength = item.attributes.strength * (1.0 - decay_factor)
            item.attributes.strength = max(0.1, new_strength)  # Never go below 0.1
            
            # Update accessibility based on strength
            item.attributes.accessibility = max(0.1, item.attributes.strength * 0.8)
            
            # Update memory stage if needed
            if item.attributes.strength < 0.3 and item.attributes.stage != MemoryStage.DECLINING:
                item.attributes.stage = MemoryStage.DECLINING
            elif item.attributes.strength < 0.15 and item.attributes.stage != MemoryStage.FORGOTTEN:
                item.attributes.stage = MemoryStage.FORGOTTEN
    
    def find_similar_items(self, embedding: List[float], domain: LongTermMemoryDomain,
                         top_k: int = 5) -> List[Tuple[UUID, float]]:
        """
        Find items similar to the given embedding using vector search.
        
        Args:
            embedding: Vector embedding to search for
            domain: Memory domain to search in
            top_k: Number of results to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self._vector_search_enabled:
            logger.warning("Vector search is not enabled")
            return []
            
        if domain not in self._vector_indices:
            logger.warning(f"No vector index for domain {domain}")
            return []
            
        # Convert embedding to numpy array
        query_vector = np.array([embedding], dtype=np.float32)
        
        # Search the index
        distances, indices = self._vector_indices[domain].search(query_vector, top_k)
        
        # Map back to item IDs
        results = []
        reverse_mapping = {v: k for k, v in self._item_to_vector_idx[domain].items()}
        
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS returns -1 for padding if fewer than top_k results
                continue
                
            if idx in reverse_mapping:
                item_id = reverse_mapping[idx]
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append((item_id, similarity))
        
        return results
    
    def associate_items(self, item_id1: UUID, item_id2: UUID, strength: float = 0.5) -> bool:
        """
        Create an association between two memory items.
        
        Args:
            item_id1: ID of the first item
            item_id2: ID of the second item
            strength: Strength of the association
            
        Returns:
            True if the association was created, False otherwise
        """
        # Check if both items exist
        if item_id1 not in self.model.items or item_id2 not in self.model.items:
            return False
            
        # Add association to first item
        if "associations" not in self.model.items[item_id1].dict():
            self.model.items[item_id1].associations = {}
        self.model.items[item_id1].associations[str(item_id2)] = strength
        
        # Add reciprocal association to second item
        if "associations" not in self.model.items[item_id2].dict():
            self.model.items[item_id2].associations = {}
        self.model.items[item_id2].associations[str(item_id1)] = strength
        
        return True
    
    def get_associated_items(self, item_id: UUID, min_strength: float = 0.0) -> List[Tuple[UUID, float]]:
        """
        Get items associated with the given item.
        
        Args:
            item_id: ID of the item to get associations for
            min_strength: Minimum association strength
            
        Returns:
            List of (item_id, association_strength) tuples
        """
        if item_id not in self.model.items:
            return []
            
        item = self.model.items[item_id]
        if not item.associations:
            return []
            
        results = []
        for associated_id_str, strength in item.associations.items():
            if strength >= min_strength:
                try:
                    associated_id = UUID(associated_id_str)
                    if associated_id in self.model.items:
                        results.append((associated_id, strength))
                except ValueError:
                    # Invalid UUID string, skip
                    continue
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def update(self, elapsed_seconds: Optional[float] = None) -> None:
        """
        Update long-term memory, applying forgetting curves and other processes.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds = current_time - self._last_update
            self._last_update = current_time
        
        # Apply forgetting curve
        self.apply_forgetting_curve(elapsed_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the long-term memory to a dictionary for serialization"""
        return self.model.model_dump()