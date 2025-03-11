import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime
import time
from collections import deque
import numpy as np

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.storage.vector_db import VectorDB
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType

from .models import (
    MemoryType,
    MemoryStatus,
    MemoryItem,
    MemoryQuery,
    MemoryRetrievalResult,
    ConsolidationCandidate,
    MemoryEvent,
    MemoryConfig
)

# Initialize logger
logger = get_module_logger("modules.memory.long_term_memory")

class LongTermMemory:
    """
    Manages persistent storage of memories that have been
    consolidated from working memory. Handles retrieval,
    forgetting, and reinforcement of memories.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the long-term memory system.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the memory system
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Memory storage by ID
        self._memories: Dict[UUID, MemoryItem] = {}
        
        # Memory strength tracking
        self._memory_strengths: Dict[UUID, float] = {}
        
        # Recent retrievals for tracking/state
        self._recent_retrievals = deque(maxlen=10)
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Long-term memory initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("working_memory_consolidation_needed", self._handle_consolidation_request)
        self._event_bus.subscribe("memory_reinforcement_requested", self._handle_reinforcement_request)
        self._event_bus.subscribe("memory_retrieval_requested", self._handle_retrieval_request)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        self._event_bus.subscribe("timer_tick", self._handle_timer_tick)
    
    def _handle_consolidation_request(self, event: Message) -> None:
        """
        Handle a working memory consolidation request.
        
        Args:
            event: The event containing consolidation data
        """
        try:
            item_id = event.data.get("item_id")
            if not item_id:
                logger.warning("Consolidation request missing item_id")
                return
                
            # Convert to UUID if string
            if isinstance(item_id, str):
                item_id = UUID(item_id)
                
            # Forward to other components to get the actual memory item
            self._event_bus.publish("memory_item_requested", {
                "memory_id": str(item_id),
                "requester": "long_term_memory",
                "request_id": str(uuid4()),
                "purpose": "consolidation"
            })
            
            logger.debug(f"Requested memory item {item_id} for consolidation")
        except Exception as e:
            logger.error(f"Error handling consolidation request: {e}")
    
    def _handle_reinforcement_request(self, event: Message) -> None:
        """
        Handle a memory reinforcement request.
        
        Args:
            event: The event containing reinforcement data
        """
        try:
            memory_id = event.data.get("memory_id")
            amount = event.data.get("amount", 0.1)
            
            if memory_id:
                # Convert to UUID if string
                if isinstance(memory_id, str):
                    memory_id = UUID(memory_id)
                    
                success = self.reinforce_memory(memory_id, amount)
                
                # Publish response
                self._event_bus.publish("memory_reinforcement_result", {
                    "memory_id": str(memory_id),
                    "success": success,
                    "request_id": event.data.get("request_id")
                })
        except Exception as e:
            logger.error(f"Error handling reinforcement request: {e}")
    
    def _handle_retrieval_request(self, event: Message) -> None:
        """
        Handle a memory retrieval request.
        
        Args:
            event: The event containing retrieval request data
        """
        try:
            query_data = event.data.get("query")
            if not query_data:
                logger.warning("Retrieval request missing query")
                return
                
            # Create query object
            query = MemoryQuery.parse_obj(query_data)
            
            # Perform retrieval
            result = self.retrieve_memories(query)
            
            # Publish result
            self._event_bus.publish("memory_retrieval_result", {
                "result": result.dict(),
                "request_id": event.data.get("request_id")
            })
            
            logger.debug(f"Retrieved {len(result.memory_items)} memories for query")
        except Exception as e:
            logger.error(f"Error handling retrieval request: {e}")
    
    def _handle_age_update(self, event: Message) -> None:
        """
        Handle development age update events.
        
        Args:
            event: Event containing the new age
        """
        try:
            new_age = event.data.get("new_age")
            if new_age is not None:
                self.update_developmental_age(new_age)
        except Exception as e:
            logger.error(f"Error handling age update: {e}")
    
    def _handle_timer_tick(self, event: Message) -> None:
        """
        Handle timer tick events for maintenance.
        
        Args:
            event: Timer event
        """
        # For long-term memory, we don't need to do much on timer ticks
        # as forgetting is more gradual and based on access patterns
        pass
    
    def store_memory(self, memory: MemoryItem) -> UUID:
        """
        Store a memory in long-term storage.
        
        Args:
            memory: The memory item to store
            
        Returns:
            ID of the stored memory
        """
        # Update status to consolidated
        memory.status = MemoryStatus.CONSOLIDATED
        
        # Store in local memory
        self._memories[memory.id] = memory
        self._memory_strengths[memory.id] = memory.strength
        
        # Store in vector database if it has an embedding
        if memory.vector_embedding:
            metadata = {
                "id": str(memory.id),
                "memory_type": memory.memory_type,
                "created_at": memory.created_at.isoformat(),
                "tags": list(memory.tags),
                "source": memory.source or "",
                "strength": memory.strength,
                "status": memory.status
            }
            
            save_vector(
                vector=memory.vector_embedding,
                metadata=metadata,
                collection=f"memory_{memory.memory_type.value}"
            )
        
        # Publish event
        self._publish_memory_event("memory_stored", {
            "memory_id": str(memory.id),
            "memory_type": memory.memory_type,
            "strength": memory.strength
        })
        
        logger.info(f"Stored memory in long-term storage: {memory.id}")
        return memory.id
    
    def consolidate_memories(self, candidates: List[ConsolidationCandidate]) -> int:
        """
        Consolidate working memory items to long-term memory.
        
        Args:
            candidates: List of consolidation candidates
            
        Returns:
            Number of memories successfully consolidated
        """
        consolidated_count = 0
        
        for candidate in candidates:
            # Request the working memory item
            self._event_bus.publish("memory_item_requested", {
                "memory_id": str(candidate.item_id),
                "requester": "long_term_memory",
                "request_id": str(uuid4()),
                "purpose": "consolidation"
            })
            
            consolidated_count += 1
            
            # In a real system, this would be async with a callback,
            # but for this implementation we'll assume it was successfully
            # requested and will be processed elsewhere
            
        logger.info(f"Requested consolidation for {consolidated_count} memories")
        return consolidated_count
    
    def process_memory_item(self, memory_item: MemoryItem) -> bool:
        """
        Process a memory item for consolidation.
        
        Args:
            memory_item: The memory item to process
            
        Returns:
            Whether processing was successful
        """
        try:
            # Don't process already consolidated items
            if memory_item.status == MemoryStatus.CONSOLIDATED:
                return False
                
            # Adjust strength based on importance and emotional content
            strength_boost = (memory_item.emotional_valence ** 2) * 0.2  # Stronger for strong emotions
            adjusted_strength = min(1.0, memory_item.strength + strength_boost)
            
            # Create a copy with updated strength and status
            consolidated_item = memory_item.copy(
                update={
                    "strength": adjusted_strength,
                    "status": MemoryStatus.CONSOLIDATED
                }
            )
            
            # Store the consolidated memory
            self.store_memory(consolidated_item)
            
            # Publish event
            self._publish_memory_event("memory_consolidated", {
                "memory_id": str(consolidated_item.id),
                "memory_type": consolidated_item.memory_type,
                "original_strength": memory_item.strength,
                "new_strength": adjusted_strength
            })
            
            return True
        except Exception as e:
            logger.error(f"Error processing memory item: {e}")
            return False
    
    def retrieve_memories(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """
        Retrieve memories based on query.
        
        Args:
            query: The query for memory retrieval
            
        Returns:
            Results of the memory retrieval
        """
        results = []
        relevance_scores: Dict[UUID, float] = {}
        
        try:
            # If we have a vector query, use vector search
            if query.query_vector:
                memories = self._retrieve_by_vector(query)
                
                for memory, score in memories:
                    results.append(memory)
                    relevance_scores[memory.id] = score
            
            # If we have a text query, search by text
            elif query.query_text:
                memories = self._retrieve_by_text(query)
                
                for memory, score in memories:
                    if memory.id not in relevance_scores:  # Avoid duplicates
                        results.append(memory)
                        relevance_scores[memory.id] = score
            
            # For other criteria, filter in-memory
            # (In a real system, this would be a database query)
            filter_results = self._filter_memories(query)
            
            for memory in filter_results:
                if memory.id not in relevance_scores:  # Avoid duplicates
                    results.append(memory)
                    relevance_scores[memory.id] = 0.7  # Default relevance for filter matches
            
            # Create and return result
            result = MemoryRetrievalResult(
                query=query,
                memory_items=results[:query.limit],
                total_results=len(results),
                relevance_scores=relevance_scores
            )
            
            # Add to recent retrievals
            self._recent_retrievals.append({
                "timestamp": datetime.now().isoformat(),
                "query_text": query.query_text,
                "memory_type": query.memory_type.value if query.memory_type else "any",
                "result_count": len(results)
            })
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            
            # Return empty result on error
            return MemoryRetrievalResult(
                query=query,
                memory_items=[],
                total_results=0,
                relevance_scores={}
            )
    
    def _retrieve_by_vector(self, query: MemoryQuery) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve memories by vector similarity.
        
        Args:
            query: The query for memory retrieval
            
        Returns:
            List of memory items with their relevance scores
        """
        results = []
        
        # Determine collection based on memory type
        collection = None
        if query.memory_type:
            collection = f"memory_{query.memory_type.value}"
        
        # Search vector database
        search_results = search_vectors(
            query_vector=query.query_vector,
            collection=collection,
            limit=query.limit * 2  # Get more results for filtering
        )
        
        # Process results
        for result in search_results:
            memory_id = result.get("metadata", {}).get("id")
            if not memory_id:
                continue
                
            # Get the memory from local storage
            memory = self._memories.get(UUID(memory_id))
            if not memory:
                continue
                
            # Apply filters
            if self._passes_filters(memory, query):
                results.append((memory, result.get("score", 0.7)))
                
                # Update access information
                memory.update_access()
        
        return results
    
    def _retrieve_by_text(self, query: MemoryQuery) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve memories by text query.
        
        Args:
            query: The query for memory retrieval
            
        Returns:
            List of memory items with their relevance scores
        """
        # In a real implementation, this would convert text to embedding
        # and then use vector search. For this implementation, we'll
        # do a simple in-memory text search.
        results = []
        
        query_text = query.query_text.lower()
        
        for memory in self._memories.values():
            score = 0.0
            
            # Check content for matches
            content_text = self._extract_text_from_content(memory.content).lower()
            if query_text in content_text:
                # Calculate match quality
                score = 0.6 + (0.2 * (len(query_text) / len(content_text)))
                score = min(0.95, score)  # Cap at 0.95
                
                # Apply filters
                if self._passes_filters(memory, query):
                    results.append((memory, score))
                    
                    # Update access information
                    memory.update_access()
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:query.limit]
    
    def _filter_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Filter memories based on query criteria.
        
        Args:
            query: The query for memory retrieval
            
        Returns:
            List of filtered memory items
        """
        results = []
        
        for memory in self._memories.values():
            if self._passes_filters(memory, query):
                results.append(memory)
                
                # Update access information
                memory.update_access()
        
        return results[:query.limit]
    
    def _passes_filters(self, memory: MemoryItem, query: MemoryQuery) -> bool:
        """
        Check if a memory passes the query filters.
        
        Args:
            memory: The memory to check
            query: The query with filters
            
        Returns:
            Whether the memory passes all filters
        """
        # Check memory type
        if query.memory_type and memory.memory_type != query.memory_type:
            return False
            
        # Check minimum strength
        if memory.strength < query.min_strength:
            return False
            
        # Check tags
        if query.tags and not any(tag in memory.tags for tag in query.tags):
            return False
            
        # Check source
        if query.source and memory.source != query.source:
            return False
            
        # Check time range
        if query.time_range:
            start = query.time_range.get("start")
            end = query.time_range.get("end")
            
            if start and memory.created_at < start:
                return False
                
            if end and memory.created_at > end:
                return False
        
        return True
    
    def reinforce_memory(self, memory_id: UUID, amount: float = 0.1) -> bool:
        """
        Reinforce a memory to increase its strength.
        
        Args:
            memory_id: ID of the memory to reinforce
            amount: Amount to increase strength by
            
        Returns:
            Whether reinforcement was successful
        """
        if memory_id not in self._memories:
            return False
            
        memory = self._memories[memory_id]
        
        # Update strength (capped at 1.0)
        old_strength = memory.strength
        memory.strength = min(1.0, memory.strength + amount)
        self._memory_strengths[memory_id] = memory.strength
        
        # Update access information
        memory.update_access()
        
        # Update vector database
        if memory.vector_embedding:
            metadata = {
                "id": str(memory.id),
                "memory_type": memory.memory_type,
                "created_at": memory.created_at.isoformat(),
                "tags": list(memory.tags),
                "source": memory.source or "",
                "strength": memory.strength,
                "status": memory.status
            }
            
            save_vector(
                vector=memory.vector_embedding,
                metadata=metadata,
                collection=f"memory_{memory.memory_type.value}",
                update=True
            )
        
        # Publish event
        self._publish_memory_event("memory_reinforced", {
            "memory_id": str(memory_id),
            "old_strength": old_strength,
            "new_strength": memory.strength,
            "reinforcement_amount": amount
        })
        
        logger.debug(f"Reinforced memory {memory_id}: {old_strength} → {memory.strength}")
        return True
    
    def forget_memory(self, memory_id: UUID) -> bool:
        """
        Explicitly forget a memory.
        
        Args:
            memory_id: ID of the memory to forget
            
        Returns:
            Whether forgetting was successful
        """
        if memory_id not in self._memories:
            return False
            
        memory = self._memories[memory_id]
        
        # Update status
        memory.status = MemoryStatus.FORGOTTEN
        
        # Remove from vector database
        if memory.vector_embedding:
            delete_vector(
                vector_id=str(memory_id),
                collection=f"memory_{memory.memory_type.value}"
            )
        
        # Publish event
        self._publish_memory_event("memory_forgotten", {
            "memory_id": str(memory_id),
            "memory_type": memory.memory_type
        })
        
        logger.info(f"Forgot memory: {memory_id}")
        return True
    
    def get_memory(self, memory_id: UUID) -> Optional[MemoryItem]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            The memory item or None if not found
        """
        memory = self._memories.get(memory_id)
        
        if memory:
            # Update access information
            memory.update_access()
            
            # Publish access event
            self._publish_memory_event("memory_accessed", {
                "memory_id": str(memory_id),
                "memory_type": memory.memory_type
            })
        
        return memory
    
    def get_recent_retrievals(self) -> List[Dict[str, Any]]:
        """
        Get recent memory retrievals.
        
        Returns:
            List of recent retrievals
        """
        return list(self._recent_retrievals)
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age and adjust behavior.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        logger.info(f"Long-term memory age updated to {new_age}")
    
    def _extract_text_from_content(self, content: Dict[str, Any]) -> str:
        """
        Extract textual content from a memory content dictionary.
        
        Args:
            content: Memory content
            
        Returns:
            Extracted text
        """
        # Extract text fields recursively
        def extract_text(data):
            if isinstance(data, str):
                return data
            elif isinstance(data, dict):
                return " ".join(str(extract_text(v)) for v in data.values() if v)
            elif isinstance(data, (list, tuple)):
                return " ".join(str(extract_text(item)) for item in data if item)
            else:
                return str(data) if data is not None else ""
        
        return extract_text(content)
    
    def _publish_memory_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Publish a memory event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        event = MemoryEvent(
            event_type=event_type,
            payload=payload
        )
        
        # Publish to event bus
        self._event_bus.publish(event_type, event.dict())
        
        # Also publish as message
        message = Message(
            type=MessageType.MEMORY,
            source="memory.long_term_memory",
            recipient=Recipient.BROADCAST,
            content={
                "event_type": event_type,
                **payload
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
