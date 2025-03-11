"""
Working memory module for the LMM project.

This module implements a working memory system with limited capacity,
activation decay, and consolidation to long-term memory.
"""
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import time
import numpy as np
from collections import deque

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.vector_store import get_embeddings, VectorStore
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.neural_substrate.neural_network import NeuralNetwork, NetworkType
from lmm_project.neural_substrate.neural_cluster import ClusterType
from lmm_project.storage.vector_db import VectorDB

from .models import (
    MemoryType,
    MemoryStatus,
    WorkingMemoryItem,
    MemoryEvent,
    MemoryConfig,
    ConsolidationCandidate
)

# Initialize logger
logger = get_module_logger("modules.memory.working_memory")

class WorkingMemory:
    """
    Manages short-term active memory with limited capacity.
    Implements activation-based decay, consolidation to long-term memory,
    and developmental changes in capacity and retention.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the working memory system.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the memory system
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Working memory storage
        self._memory_items: Dict[UUID, WorkingMemoryItem] = {}
        
        # Recently decayed items (for potential recovery)
        self._recently_decayed = deque(maxlen=5)
        
        # Last consolidation time
        self._last_consolidation_check = time.time()
        
        # Last decay update time
        self._last_decay_update = time.time()
        
        # Neural network for encoding and activation
        self._neural_network = None
        self._initialize_neural_network()
        
        # Local vector store for similarity search
        self._vector_store = self._initialize_vector_store()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Working memory initialized with age {developmental_age}, " 
                   f"capacity {self._calculate_capacity()}")
    
    def _initialize_neural_network(self) -> None:
        """Initialize neural network for memory encoding if enabled"""
        if not self._config.use_neural_networks:
            logger.info("Neural networks disabled for working memory")
            return
            
        try:
            # Create a neural network for working memory
            self._neural_network = NeuralNetwork(
                network_id="working_memory_network",
                config={
                    "network_type": NetworkType.MODULAR,
                    "input_size": self._config.embedding_dimension,
                    "output_size": self._config.embedding_dimension // 2,
                    "hidden_layers": [self._config.embedding_dimension // 2],
                    "learning_rate": self._config.network_learn_rate,
                    "plasticity_enabled": True,
                    "cluster_sizes": [
                        self._config.embedding_dimension, 
                        self._config.embedding_dimension // 2
                    ],
                    "cluster_types": [
                        ClusterType.FEED_FORWARD, 
                        ClusterType.RECURRENT
                    ]
                }
            )
            logger.info("Neural network initialized for working memory encoding")
        except Exception as e:
            logger.error(f"Failed to initialize neural network: {e}")
            self._neural_network = None
    
    def _initialize_vector_store(self) -> Optional[VectorStore]:
        """Initialize vector store for similarity search"""
        try:
            vector_store = VectorStore(
                dimension=self._config.embedding_dimension,
                index_type="Flat",  # For working memory, small enough for exact search
                use_gpu=self._config.vector_store_gpu_enabled,
                storage_dir=self._config.vector_store_dir
            )
            logger.info("Vector store initialized for working memory")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return None
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("memory_accessed", self._handle_memory_access)
        self._event_bus.subscribe("focus_shift", self._handle_focus_shift)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        self._event_bus.subscribe("timer_tick", self._handle_timer_tick)
    
    def _handle_memory_access(self, event: Message) -> None:
        """
        Handle a memory access event.
        
        Args:
            event: The event containing memory access data
        """
        try:
            memory_id = event.data.get("memory_id")
            if not memory_id:
                return
                
            # Convert string ID to UUID if needed
            if isinstance(memory_id, str):
                memory_id = UUID(memory_id)
                
            # If it's in working memory, boost activation
            if memory_id in self._memory_items:
                boost_amount = event.data.get("boost_amount", 0.2)
                self._update_item_activation(memory_id, boost_amount)
                
                logger.debug(f"Memory item {memory_id} activation boosted by {boost_amount}")
                
            # Publish memory accessed event
            self._publish_memory_event("memory_item_activated", {
                "memory_id": str(memory_id),
                "activation_level": self._memory_items[memory_id].activation_level if memory_id in self._memory_items else None
            })
        except Exception as e:
            logger.error(f"Error handling memory access event: {e}")
    
    def _handle_focus_shift(self, event: Message) -> None:
        """
        Handle a focus shift event, which may trigger memory updates.
        
        Args:
            event: The event containing focus shift data
        """
        try:
            # Extract focus information
            focus_target = event.data.get("focus_target")
            importance = event.data.get("importance", 0.5)
            
            if not focus_target:
                logger.warning("Focus shift event missing focus_target")
                return
                
            # Convert to content dictionary if it's not already
            if not isinstance(focus_target, dict):
                focus_target = {"text": str(focus_target)}
                
            # Create working memory item for the focus target
            # This may replace existing items if capacity is reached
            memory_item = self.store_item(
                content=focus_target,
                source="attention",
                importance=importance
            )
            
            if memory_item:
                # Publish response indicating the new memory item
                self._publish_memory_event("focus_memorized", {
                    "memory_id": str(memory_item.id),
                    "importance": importance
                })
                
                # Check if we need to consolidate or forget items
                current_time = time.time()
                if current_time - self._last_consolidation_check > 5.0:  # Check every 5 seconds
                    self._check_for_consolidation()
                    self._last_consolidation_check = current_time
        except Exception as e:
            logger.error(f"Error handling focus shift event: {e}")
    
    def _handle_age_update(self, event: Message) -> None:
        """
        Handle a developmental age update.
        
        Args:
            event: The event containing the new age
        """
        try:
            new_age = event.data.get("age")
            if new_age is not None:
                self.update_developmental_age(float(new_age))
        except Exception as e:
            logger.error(f"Error handling age update event: {e}")
    
    def _handle_timer_tick(self, event: Message) -> None:
        """
        Handle a timer tick event, which triggers memory decay.
        
        Args:
            event: The timer event
        """
        try:
            current_time = time.time()
            elapsed_ms = event.data.get("elapsed_ms", 1000)  # Default to 1 second if not specified
            
            # Apply decay to working memory based on elapsed time
            if current_time - self._last_decay_update > 0.5:  # Limit update frequency to 2Hz
                self._apply_decay()
                self._last_decay_update = current_time
        except Exception as e:
            logger.error(f"Error handling timer tick event: {e}")
    
    def store_item(
        self,
        content: Dict[str, Any],
        source: Optional[str] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        tags: Set[str] = None
    ) -> Optional[WorkingMemoryItem]:
        """
        Store an item in working memory.
        
        Args:
            content: Content of the memory
            source: Source of the memory
            importance: Importance of the memory (0-1)
            emotional_valence: Emotional valence (-1 to 1)
            tags: Set of tags for the memory
            
        Returns:
            The created memory item, or None if creation failed
        """
        # Calculate current capacity
        capacity = self._calculate_capacity()
        
        # Check if we need to free space
        if len(self._memory_items) >= capacity:
            success = self._free_space()
            if not success:
                logger.warning("Failed to free space in working memory")
                return None
        
        try:
            # Extract text for embedding generation
            text = self._extract_text_from_content(content)
            
            # Generate embedding
            embedding = self._get_embedding(text)
            
            # Create activation pattern using neural network if available
            neural_pattern = None
            if self._neural_network and embedding:
                neural_pattern = self._neural_network.process(embedding)
            
            # Calculate retention time based on importance and developmental age
            retention_time_ms = int(30000 * (1.0 + importance) * (1.0 + self._developmental_age / 2.0))
            
            # Calculate dynamic decay rate based on importance and age
            decay_rate = self._config.get_decay_rate(self._developmental_age)
            adjusted_decay_rate = decay_rate * (1.0 - (importance * 0.5))
            
            # Process developmental factors
            developmental_factors = {
                "age": self._developmental_age,
                "capacity": capacity,
                "retention_multiplier": 1.0 + self._developmental_age / 2.0
            }
            
            # Create the memory item
            memory_item = WorkingMemoryItem(
                memory_type=MemoryType.WORKING,
                content=content,
                source=source,
                importance=importance,
                emotional_valence=emotional_valence,
                tags=set(tags) if tags else set(),
                vector_embedding=embedding,
                activation_level=0.8 + (importance * 0.2),  # Higher initial activation for important items
                retention_time_ms=retention_time_ms,
                decay_rate=adjusted_decay_rate,
                neural_activation_pattern=neural_pattern,
                developmental_factors=developmental_factors
            )
            
            # Store the item
            self._memory_items[memory_item.id] = memory_item
            
            # Add to vector store if available
            if self._vector_store and embedding:
                try:
                    self._vector_store.add_vectors(
                        vectors=[embedding],
                        metadata_list=[{"memory_id": str(memory_item.id)}]
                    )
                except Exception as e:
                    logger.error(f"Failed to add vector to store: {e}")
            
            # Publish memory created event
            self._publish_memory_event("working_memory_item_created", {
                "memory_id": str(memory_item.id),
                "importance": importance
            })
            
            logger.debug(f"Created working memory item {memory_item.id} with importance {importance}")
            return memory_item
            
        except Exception as e:
            logger.error(f"Error creating working memory item: {e}")
            return None
    
    def get_items(self) -> List[WorkingMemoryItem]:
        """
        Get all items in working memory.
        
        Returns:
            List of all memory items
        """
        return list(self._memory_items.values())
    
    def get_item(self, item_id: UUID) -> Optional[WorkingMemoryItem]:
        """
        Get a specific memory item by ID.
        
        Args:
            item_id: ID of the memory item
            
        Returns:
            The memory item, or None if not found
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(item_id, str):
                item_id = UUID(item_id)
                
            if item_id in self._memory_items:
                # Update access stats
                self._memory_items[item_id].update_access()
                
                # Boost activation
                self._update_item_activation(item_id, 0.1)
                
                # Return the item
                return self._memory_items[item_id]
            
            # Check if it's in recently decayed items
            for item in self._recently_decayed:
                if item.id == item_id:
                    logger.info(f"Recovered recently decayed item {item_id}")
                    # Return to working memory with boosted activation
                    item.activation_level = 0.7
                    self._memory_items[item_id] = item
                    self._recently_decayed.remove(item)
                    return item
            
            return None
        except Exception as e:
            logger.error(f"Error getting memory item {item_id}: {e}")
            return None
    
    def remove_item(self, item_id: UUID) -> bool:
        """
        Remove a memory item from working memory.
        
        Args:
            item_id: ID of the memory item to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(item_id, str):
                item_id = UUID(item_id)
                
            if item_id in self._memory_items:
                # Remove from vector store if available
                if self._vector_store:
                    embedding = self._memory_items[item_id].vector_embedding
                    if embedding:
                        try:
                            # This is simplified; in practice you'd need to track vector IDs
                            # Here we search and delete based on metadata
                            search_results = self._vector_store.search(
                                query_vector=np.array(embedding, dtype=np.float32),
                                k=1,
                                return_metadata=True
                            )
                            
                            for result_id, _, metadata in search_results:
                                if metadata.get("memory_id") == str(item_id):
                                    # In practice, you'd delete the vector by its FAISS ID
                                    # This is a placeholder for that functionality
                                    pass
                        except Exception as search_error:
                            logger.error(f"Error searching vector store: {search_error}")
                
                # Remove the item
                item = self._memory_items.pop(item_id)
                
                # Add to recently decayed
                self._recently_decayed.append(item)
                
                # Publish memory removed event
                self._publish_memory_event("working_memory_item_removed", {
                    "memory_id": str(item_id)
                })
                
                logger.debug(f"Removed working memory item {item_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error removing memory item {item_id}: {e}")
            return False
    
    def get_consolidation_candidates(self) -> List[ConsolidationCandidate]:
        """
        Get memory items that are candidates for consolidation to long-term memory.
        
        Returns:
            List of consolidation candidates
        """
        try:
            candidates = []
            current_time = time.time()
            
            for memory_id, memory_item in self._memory_items.items():
                # Skip items that were just created
                if (current_time - memory_item.created_at.timestamp()) < 5.0:
                    continue
                
                # Calculate recency factor (1.0 = most recent)
                time_since_access = (current_time - memory_item.last_accessed.timestamp())
                recency = max(0.0, 1.0 - (time_since_access / 300.0))  # 5 minutes to 0.0
                
                # Calculate dynamic consolidation threshold based on development
                consolidation_threshold = max(
                    0.3,  # Minimum threshold
                    self._config.consolidation_threshold - (self._developmental_age * 0.05)  # Easier to consolidate as development progresses
                )
                
                # Calculate decay status
                decay_status = 1.0 - memory_item.activation_level
                
                # Calculate emotional impact
                emotional_impact = abs(memory_item.emotional_valence) * 0.3
                
                # Calculate overall consolidation score
                consolidation_score = (
                    (memory_item.importance * 0.4) +
                    (recency * 0.2) +
                    (memory_item.activation_level * 0.2) +
                    (emotional_impact * 0.2)
                )
                
                # Add as candidate if score exceeds threshold or activation is low
                if consolidation_score >= consolidation_threshold or memory_item.activation_level < 0.3:
                    candidates.append(ConsolidationCandidate(
                        item_id=memory_id,
                        importance=memory_item.importance,
                        recency=recency,
                        activation=memory_item.activation_level,
                        consolidation_score=consolidation_score,
                        decay_status=decay_status,
                        emotional_impact=emotional_impact,
                        consolidation_threshold=consolidation_threshold
                    ))
            
            # Sort by consolidation score, highest first
            candidates.sort(key=lambda c: c.consolidation_score, reverse=True)
            
            return candidates
        except Exception as e:
            logger.error(f"Error getting consolidation candidates: {e}")
            return []
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the memory system.
        
        Args:
            new_age: New developmental age
        """
        if new_age < 0:
            logger.warning(f"Invalid developmental age: {new_age}")
            return
            
        old_age = self._developmental_age
        self._developmental_age = new_age
        
        old_capacity = self._calculate_capacity(old_age)
        new_capacity = self._calculate_capacity(new_age)
        
        logger.info(f"Working memory developmental age updated from {old_age:.2f} to {new_age:.2f}, "
                   f"capacity changed from {old_capacity} to {new_capacity}")
    
    def _calculate_capacity(self, age: Optional[float] = None) -> int:
        """
        Calculate the working memory capacity based on developmental age.
        
        Args:
            age: Age to calculate capacity for, or current age if None
            
        Returns:
            Working memory capacity
        """
        age_to_use = age if age is not None else self._developmental_age
        return self._config.get_working_memory_capacity(age_to_use)
    
    def _update_item_activation(self, item_id: UUID, boost: float) -> None:
        """
        Update activation level of a memory item.
        
        Args:
            item_id: ID of the memory item
            boost: Amount to boost activation by
        """
        try:
            if item_id in self._memory_items:
                # Get current activation
                item = self._memory_items[item_id]
                
                # Update access timestamp
                item.update_access()
                
                # Boost activation level
                item.activation_level = min(1.0, item.activation_level + boost)
                
                # Adjust related items if available
                for related_id in item.related_items:
                    if related_id in self._memory_items:
                        # Smaller boost for related items
                        related_boost = boost * 0.3
                        self._memory_items[related_id].activation_level = min(
                            1.0, 
                            self._memory_items[related_id].activation_level + related_boost
                        )
                        self._memory_items[related_id].update_access()
        except Exception as e:
            logger.error(f"Error updating item activation for {item_id}: {e}")
    
    def _apply_decay(self) -> None:
        """
        Apply activation decay to all memory items.
        """
        try:
            current_time = time.time()
            items_to_remove = []
            
            for memory_id, memory_item in self._memory_items.items():
                # Apply decay based on individual item's decay rate
                memory_item.activation_level = max(
                    0.0, 
                    memory_item.activation_level - memory_item.decay_rate
                )
                
                # Check if item has expired (activation too low)
                if memory_item.activation_level < 0.1:
                    # Check if it should be consolidated before removal
                    time_since_creation = (current_time - memory_item.created_at.timestamp()) / 1000.0
                    
                    # Items that have been around a while and are important should be consolidated
                    if time_since_creation > 30.0 and memory_item.importance > 0.5:
                        # Don't remove yet, let consolidation handle it
                        # But mark it for consolidation
                        self._publish_memory_event("memory_consolidation_needed", {
                            "memory_id": str(memory_id),
                            "importance": memory_item.importance,
                            "urgency": "high"
                        })
                    else:
                        # Track for removal
                        items_to_remove.append(memory_id)
            
            # Remove expired items
            for memory_id in items_to_remove:
                item = self._memory_items[memory_id]
                
                # Add to recently decayed queue before removing
                self._recently_decayed.append(item)
                
                # Remove from working memory
                del self._memory_items[memory_id]
                
                # Publish memory decayed event
                self._publish_memory_event("working_memory_item_decayed", {
                    "memory_id": str(memory_id),
                    "importance": item.importance
                })
                
                logger.debug(f"Memory item {memory_id} decayed and removed from working memory")
                
            if items_to_remove:
                logger.info(f"Removed {len(items_to_remove)} decayed memory items")
        except Exception as e:
            logger.error(f"Error applying decay: {e}")
    
    def _free_space(self, count: int = 1) -> bool:
        """
        Free space in working memory by removing low-priority items.
        
        Args:
            count: Number of spaces to free
            
        Returns:
            True if space was freed, False otherwise
        """
        try:
            if not self._memory_items:
                return True  # Already empty
                
            # Calculate how many items we need to remove
            current_count = len(self._memory_items)
            capacity = self._calculate_capacity()
            
            items_to_remove = min(count, max(0, current_count - capacity + count))
            
            if items_to_remove <= 0:
                return True  # No need to remove anything
                
            # Get candidates for removal
            candidates = []
            for memory_id, memory_item in self._memory_items.items():
                # Calculate removal priority (lower = more likely to be removed)
                priority = (
                    (memory_item.importance * 0.4) +
                    (memory_item.activation_level * 0.6)
                )
                
                candidates.append((memory_id, priority))
                
            # Sort by priority, lowest first
            candidates.sort(key=lambda x: x[1])
            
            # Remove the lowest priority items
            removed = 0
            for memory_id, _ in candidates:
                if removed >= items_to_remove:
                    break
                    
                # Check if it should be consolidated before removal
                if self._memory_items[memory_id].importance > 0.5:
                    # Try to consolidate important memories
                    self._publish_memory_event("memory_consolidation_needed", {
                        "memory_id": str(memory_id),
                        "importance": self._memory_items[memory_id].importance,
                        "urgency": "medium"
                    })
                
                # Add to recently decayed
                self._recently_decayed.append(self._memory_items[memory_id])
                
                # Remove from working memory
                del self._memory_items[memory_id]
                
                # Publish memory removed event
                self._publish_memory_event("working_memory_item_removed", {
                    "memory_id": str(memory_id),
                    "reason": "capacity_limit"
                })
                
                removed += 1
                
            logger.info(f"Freed space by removing {removed} low-priority items from working memory")
            return removed > 0
        except Exception as e:
            logger.error(f"Error freeing space: {e}")
            return False
    
    def _check_for_consolidation(self) -> None:
        """
        Check if any memory items should be consolidated to long-term memory.
        """
        try:
            # Get consolidation candidates
            candidates = self.get_consolidation_candidates()
            
            if candidates:
                # Publish consolidation request event
                self._publish_memory_event("consolidation_candidates_ready", {
                    "candidate_count": len(candidates),
                    "candidates": [c.dict() for c in candidates]
                })
                
                logger.debug(f"Published consolidation request with {len(candidates)} candidates")
        except Exception as e:
            logger.error(f"Error checking for consolidation: {e}")
    
    def _extract_text_from_content(self, content: Dict[str, Any]) -> str:
        """
        Extract text from content for embedding generation.
        
        Args:
            content: Content dictionary
            
        Returns:
            Extracted text
        """
        result = []
        
        def extract_text(data):
            if isinstance(data, str):
                result.append(data)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key == "text" or key == "content":
                        result.append(str(value))
                    else:
                        extract_text(value)
            elif isinstance(data, list):
                for item in data:
                    extract_text(item)
        
        extract_text(content)
        return " ".join(result)
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get vector embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Vector embedding, or None if generation failed
        """
        if not text:
            return None
            
        try:
            # Get embedding from vector store utility
            embedding = get_embeddings(text)
            
            # Process through neural network if available
            if self._neural_network and embedding is not None:
                # Create activation pattern
                neural_pattern = self._neural_network.process(embedding)
                
                # Use as features for learning
                if self._config.use_neural_networks:
                    # Train the network with this pattern (simplified)
                    try:
                        self._neural_network.learn(
                            inputs=embedding,
                            targets=embedding,  # Simplified autoencoder-like learning
                            learning_rate=self._config.network_learn_rate
                        )
                    except Exception as e:
                        logger.error(f"Error training neural network: {e}")
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _publish_memory_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Publish a memory event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        event = MemoryEvent(
            event_type=event_type,
            source_module="memory.working_memory",
            payload=payload
        )
        
        # Publish to event bus
        self._event_bus.publish(event_type, event.dict())
        
        # Also publish as message
        message = Message(
            type=MessageType.MEMORY,
            source="memory.working_memory",
            recipient=Recipient.BROADCAST,
            content={
                "event_type": event_type,
                **payload
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
