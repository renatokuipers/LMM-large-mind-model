import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime
import time
from collections import deque
import numpy as np

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.vector_store import get_embeddings
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType

from .models import (
    MemoryType,
    MemoryStatus,
    MemoryStrength,
    EpisodicMemoryItem,
    MemoryEvent,
    MemoryConfig
)

# Initialize logger
logger = get_module_logger("modules.memory.episodic_memory")

class EpisodicMemory:
    """
    Manages episodic memory (event-based experiences).
    Stores and retrieves memories of events and experiences,
    maintaining temporal relationships and emotional context.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the episodic memory system.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the memory system
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Episodic memory storage
        self._memories: Dict[UUID, EpisodicMemoryItem] = {}
        
        # Temporal index (mapping timestamps to memory IDs)
        self._temporal_index: Dict[datetime, List[UUID]] = {}
        
        # Actor index (mapping actors to memory IDs)
        self._actor_index: Dict[str, List[UUID]] = {}
        
        # Location index (mapping locations to memory IDs)
        self._location_index: Dict[str, List[UUID]] = {}
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Episodic memory initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("create_episodic_memory", self._handle_create_memory)
        self._event_bus.subscribe("episodic_memory_requested", self._handle_memory_request)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _handle_create_memory(self, event: Message) -> None:
        """
        Handle a request to create a new episodic memory.
        
        Args:
            event: The event containing memory creation data
        """
        try:
            memory_data = event.data.get("memory_data")
            if not memory_data:
                logger.warning("Create memory event missing memory_data")
                return
                
            # Extract memory details
            content = memory_data.get("content")
            actors = memory_data.get("actors", [])
            location = memory_data.get("location")
            narrative = memory_data.get("narrative")
            emotional_valence = memory_data.get("emotional_valence", 0.0)
            tags = set(memory_data.get("tags", []))
            
            if not content:
                logger.warning("Create memory event missing content")
                return
                
            # Create the memory
            memory = self.store_memory(
                content=content,
                actors=actors,
                location=location,
                narrative=narrative,
                emotional_valence=emotional_valence,
                tags=tags
            )
            
            if memory:
                # Publish response
                self._event_bus.publish("episodic_memory_created", {
                    "memory_id": str(memory.id),
                    "request_id": event.data.get("request_id")
                })
        except Exception as e:
            logger.error(f"Error handling create memory event: {e}")
    
    def _handle_memory_request(self, event: Message) -> None:
        """
        Handle a request to retrieve episodic memories.
        
        Args:
            event: The event containing memory retrieval parameters
        """
        try:
            memory_id = event.data.get("memory_id")
            if not memory_id:
                logger.warning("Memory request missing memory_id")
                return
                
            # Convert to UUID if string
            if isinstance(memory_id, str):
                memory_id = UUID(memory_id)
                
            # Get the memory
            memory = self._memories.get(memory_id)
            
            # Publish response
            self._event_bus.publish("episodic_memory_response", {
                "memory": memory.dict() if memory else None,
                "found": memory is not None,
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling memory request: {e}")
    
    def _handle_age_update(self, event: Message) -> None:
        """
        Handle development age update events.
        
        Args:
            event: The age update event
        """
        try:
            new_age = event.data.get("new_age")
            if new_age is not None:
                self.update_developmental_age(new_age)
        except Exception as e:
            logger.error(f"Error handling age update: {e}")
    
    def store_memory(
        self,
        content: Dict[str, Any],
        actors: List[str] = None,
        location: Optional[str] = None,
        narrative: Optional[str] = None,
        emotional_valence: float = 0.0,
        tags: Set[str] = None
    ) -> Optional[EpisodicMemoryItem]:
        """
        Store an episodic memory.
        
        Args:
            content: The content of the experience
            actors: People/entities involved in the experience
            location: Where the experience occurred
            narrative: Text description of the experience
            emotional_valence: Emotional tone of the memory
            tags: Optional tags for categorization
            
        Returns:
            The created episodic memory item or None if failed
        """
        try:
            # Determine detail level based on developmental age
            detail_level = self._get_detail_level()
            
            # Create embedding for the memory
            embedding_text = self._create_embedding_text(
                content, actors, location, narrative
            )
            vector_embedding = self._get_embedding(embedding_text) if embedding_text else None
            
            # Create the memory item
            memory = EpisodicMemoryItem(
                memory_type=MemoryType.EPISODIC,
                content=content,
                actors=actors or [],
                location=location,
                narrative=narrative,
                emotional_valence=emotional_valence,
                tags=tags or set(),
                vector_embedding=vector_embedding,
                strength=self._calculate_initial_strength(emotional_valence)
            )
            
            # Store the memory
            self._memories[memory.id] = memory
            
            # Update indexes
            self._update_indexes(memory)
            
            # Publish event
            self._publish_memory_event("episodic_memory_stored", {
                "memory_id": str(memory.id),
                "emotional_valence": emotional_valence,
                "timestamp": memory.timestamp.isoformat(),
                "actors": actors,
                "location": location
            })
            
            # Forward to long-term memory for storage
            self._event_bus.publish("memory_consolidation", {
                "memory": memory.dict(),
                "memory_type": MemoryType.EPISODIC
            })
            
            logger.info(f"Stored episodic memory: {memory.id}")
            return memory
        except Exception as e:
            logger.error(f"Error storing episodic memory: {e}")
            return None
    
    def get_memory(self, memory_id: UUID) -> Optional[EpisodicMemoryItem]:
        """
        Get a specific episodic memory.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            The episodic memory or None if not found
        """
        memory = self._memories.get(memory_id)
        
        if memory:
            # Update access information
            memory.update_access()
            
        return memory
    
    def get_memories_by_actor(self, actor: str) -> List[EpisodicMemoryItem]:
        """
        Get memories involving a specific actor.
        
        Args:
            actor: The actor to search for
            
        Returns:
            List of memories involving the actor
        """
        memory_ids = self._actor_index.get(actor.lower(), [])
        return [self._memories[memory_id] for memory_id in memory_ids if memory_id in self._memories]
    
    def get_memories_by_location(self, location: str) -> List[EpisodicMemoryItem]:
        """
        Get memories that occurred at a specific location.
        
        Args:
            location: The location to search for
            
        Returns:
            List of memories at the location
        """
        memory_ids = self._location_index.get(location.lower(), [])
        return [self._memories[memory_id] for memory_id in memory_ids if memory_id in self._memories]
    
    def get_memories_by_timeframe(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EpisodicMemoryItem]:
        """
        Get memories from a specific timeframe.
        
        Args:
            start_time: Start of the timeframe (inclusive)
            end_time: End of the timeframe (inclusive)
            
        Returns:
            List of memories in the timeframe
        """
        results = []
        
        # If no timeframe specified, return recent memories
        if not start_time and not end_time:
            # Sort memories by timestamp (newest first)
            sorted_memories = sorted(
                self._memories.values(),
                key=lambda m: m.timestamp,
                reverse=True
            )
            return sorted_memories[:10]  # Return most recent 10
        
        # Default times if not specified
        start = start_time or datetime.min
        end = end_time or datetime.max
        
        # Search temporal index
        for timestamp, memory_ids in self._temporal_index.items():
            if start <= timestamp <= end:
                for memory_id in memory_ids:
                    if memory_id in self._memories:
                        results.append(self._memories[memory_id])
        
        return results
    
    def get_count(self) -> int:
        """
        Get the count of episodic memories.
        
        Returns:
            Number of episodic memories
        """
        return len(self._memories)
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age and adjust behavior.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        logger.info(f"Episodic memory age updated to {new_age}")
    
    def _get_detail_level(self) -> float:
        """
        Get the current detail level based on developmental age.
        
        Returns:
            Detail level (0.0-1.0)
        """
        # Young minds store less detail, mature minds store more
        base_level = self._config.episodic_memory_detail_level
        age_factor = min(1.0, max(0.1, self._developmental_age))
        
        return base_level * age_factor
    
    def _calculate_initial_strength(self, emotional_valence: float) -> float:
        """
        Calculate initial memory strength based on emotional valence.
        
        Args:
            emotional_valence: Emotional valence of the memory
            
        Returns:
            Initial memory strength
        """
        # Emotional memories (positive or negative) are stronger
        # Neutral memories are weaker
        emotional_intensity = abs(emotional_valence)
        
        # Base strength (moderate)
        base_strength = MemoryStrength.MODERATE
        
        # Adjust based on emotional intensity
        strength_adjustment = emotional_intensity * 0.3
        
        # Return adjusted strength (capped at 0.0-1.0)
        return max(0.1, min(0.9, base_strength + strength_adjustment))
    
    def _create_embedding_text(
        self,
        content: Dict[str, Any],
        actors: List[str],
        location: Optional[str],
        narrative: Optional[str]
    ) -> str:
        """
        Create text for embedding from memory components.
        
        Args:
            content: Memory content
            actors: People/entities involved
            location: Where it occurred
            narrative: Text description
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Add narrative if available
        if narrative:
            parts.append(narrative)
        
        # Add location
        if location:
            parts.append(f"Location: {location}")
        
        # Add actors
        if actors:
            parts.append(f"Involving: {', '.join(actors)}")
        
        # Add content
        content_text = self._extract_text_from_content(content)
        if content_text:
            parts.append(f"Content: {content_text}")
        
        return " ".join(parts)
    
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
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get vector embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding or None if failed
        """
        try:
            if not text:
                return None
                
            # Get embeddings using utility function
            embedding = get_embeddings(text)
            if embedding is not None:
                return embedding
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            
        return None
    
    def _update_indexes(self, memory: EpisodicMemoryItem) -> None:
        """
        Update memory indexes with a new memory.
        
        Args:
            memory: The memory to index
        """
        # Update temporal index
        date_key = memory.timestamp.replace(microsecond=0)  # Remove microseconds for binning
        if date_key not in self._temporal_index:
            self._temporal_index[date_key] = []
        self._temporal_index[date_key].append(memory.id)
        
        # Update actor index
        for actor in memory.actors:
            actor_key = actor.lower()
            if actor_key not in self._actor_index:
                self._actor_index[actor_key] = []
            self._actor_index[actor_key].append(memory.id)
        
        # Update location index
        if memory.location:
            location_key = memory.location.lower()
            if location_key not in self._location_index:
                self._location_index[location_key] = []
            self._location_index[location_key].append(memory.id)
    
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
            source="memory.episodic_memory",
            recipient=Recipient.BROADCAST,
            content={
                "event_type": event_type,
                **payload
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
