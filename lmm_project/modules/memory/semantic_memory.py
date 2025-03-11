import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime
import time
import numpy as np
from collections import defaultdict

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.vector_store import get_embeddings
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType

from .models import (
    MemoryType,
    MemoryStatus,
    MemoryStrength,
    SemanticMemoryItem,
    MemoryEvent,
    MemoryConfig
)

# Initialize logger
logger = get_module_logger("modules.memory.semantic_memory")

class SemanticMemory:
    """
    Manages semantic memory (factual knowledge and concepts).
    Stores, organizes, and enables retrieval of factual information,
    concepts, and their relationships.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the semantic memory system.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the memory system
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Semantic memory storage
        self._memories: Dict[UUID, SemanticMemoryItem] = {}
        
        # Concept index (mapping concept names to memory IDs)
        self._concept_index: Dict[str, List[UUID]] = {}
        
        # Related concept index (mapping concepts to related concepts)
        self._related_concepts: Dict[str, Set[str]] = defaultdict(set)
        
        # Property index (mapping properties to memory IDs)
        self._property_index: Dict[str, List[UUID]] = defaultdict(list)
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Semantic memory initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("create_semantic_memory", self._handle_create_memory)
        self._event_bus.subscribe("semantic_memory_requested", self._handle_memory_request)
        self._event_bus.subscribe("semantic_query", self._handle_semantic_query)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _handle_create_memory(self, event: Message) -> None:
        """
        Handle a create semantic memory event.
        
        Args:
            event: The event containing memory data
        """
        try:
            memory_data = event.data.get("memory_data")
            if not memory_data:
                logger.warning("Create memory event missing memory_data")
                return
                
            # Extract memory details
            concept = memory_data.get("concept")
            definition = memory_data.get("definition")
            properties = memory_data.get("properties", {})
            examples = memory_data.get("examples", [])
            related_concepts = memory_data.get("related_concepts", [])
            confidence = memory_data.get("confidence", 0.7)
            tags = set(memory_data.get("tags", []))
            
            if not (concept and definition):
                logger.warning("Create memory event missing required fields")
                return
                
            # Create the memory
            memory = self.store_memory(
                concept=concept,
                definition=definition,
                properties=properties,
                examples=examples,
                related_concepts=related_concepts,
                confidence=confidence,
                tags=tags
            )
            
            if memory:
                # Publish response
                self._event_bus.publish("semantic_memory_created", {
                    "memory_id": str(memory.id),
                    "concept": concept,
                    "request_id": event.data.get("request_id")
                })
        except Exception as e:
            logger.error(f"Error handling create memory event: {e}")
    
    def _handle_memory_request(self, event: Message) -> None:
        """
        Handle a semantic memory request.
        
        Args:
            event: The event containing the request
        """
        try:
            memory_id = event.data.get("memory_id")
            concept = event.data.get("concept")
            
            if not (memory_id or concept):
                logger.warning("Memory request missing identifiers")
                return
                
            # Get memory by ID if provided
            memory = None
            if memory_id:
                # Convert to UUID if string
                if isinstance(memory_id, str):
                    memory_id = UUID(memory_id)
                memory = self._memories.get(memory_id)
            
            # Otherwise try by concept
            elif concept:
                memory = self.get_memory_by_concept(concept)
            
            # Publish response
            self._event_bus.publish("semantic_memory_response", {
                "memory": memory.dict() if memory else None,
                "found": memory is not None,
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling memory request: {e}")
    
    def _handle_semantic_query(self, event: Message) -> None:
        """
        Handle a semantic memory query.
        
        Args:
            event: The event containing the query
        """
        try:
            query = event.data.get("query")
            if not query:
                logger.warning("Semantic query missing query text")
                return
                
            # Get the query type
            query_type = event.data.get("query_type", "concept")
            
            # Perform query based on type
            results = []
            if query_type == "concept":
                memory = self.get_memory_by_concept(query)
                if memory:
                    results = [memory.dict()]
            elif query_type == "related":
                related = self.get_related_concepts(query)
                results = [mem.dict() for mem in related]
            elif query_type == "property":
                property_matches = self.get_memories_by_property(query)
                results = [mem.dict() for mem in property_matches]
            
            # Publish response
            self._event_bus.publish("semantic_query_result", {
                "results": results,
                "query": query,
                "query_type": query_type,
                "count": len(results),
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling semantic query: {e}")
    
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
    
    def store_memory(
        self,
        concept: str,
        definition: str,
        properties: Dict[str, Any] = None,
        examples: List[str] = None,
        related_concepts: List[str] = None,
        confidence: float = 0.7,
        tags: Set[str] = None
    ) -> Optional[SemanticMemoryItem]:
        """
        Store a semantic memory (factual knowledge or concept).
        
        Args:
            concept: The concept name
            definition: Definition of the concept
            properties: Properties of the concept
            examples: Examples of the concept
            related_concepts: Related concepts
            confidence: Confidence in this knowledge
            tags: Optional tags for categorization
            
        Returns:
            The created semantic memory item or None if failed
        """
        try:
            # Check if concept already exists
            existing = self.get_memory_by_concept(concept)
            if existing:
                # Update existing memory instead of creating new
                return self._update_existing_memory(
                    existing,
                    definition,
                    properties or {},
                    examples or [],
                    related_concepts or [],
                    confidence,
                    tags or set()
                )
            
            # Create embedding for the memory
            embedding_text = self._create_embedding_text(
                concept, definition, properties, examples, related_concepts
            )
            vector_embedding = self._get_embedding(embedding_text) if embedding_text else None
            
            # Create the memory item
            memory = SemanticMemoryItem(
                memory_type=MemoryType.SEMANTIC,
                content={"definition": definition},
                concept=concept,
                definition=definition,
                properties=properties or {},
                examples=examples or [],
                related_concepts=related_concepts or [],
                confidence=confidence,
                tags=tags or set(),
                vector_embedding=vector_embedding,
                strength=self._calculate_initial_strength(confidence, len(properties or {}))
            )
            
            # Store the memory
            self._memories[memory.id] = memory
            
            # Update indexes
            self._update_indexes(memory)
            
            # Publish event
            self._publish_memory_event("semantic_memory_stored", {
                "memory_id": str(memory.id),
                "concept": concept,
                "confidence": confidence
            })
            
            # Forward to long-term memory for storage
            self._event_bus.publish("memory_consolidation", {
                "memory": memory.dict(),
                "memory_type": MemoryType.SEMANTIC
            })
            
            logger.info(f"Stored semantic memory for concept '{concept}': {memory.id}")
            return memory
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}")
            return None
    
    def _update_existing_memory(
        self,
        existing: SemanticMemoryItem,
        definition: str,
        properties: Dict[str, Any],
        examples: List[str],
        related_concepts: List[str],
        confidence: float,
        tags: Set[str]
    ) -> SemanticMemoryItem:
        """
        Update an existing semantic memory.
        
        Args:
            existing: The existing memory item
            definition: Updated definition
            properties: Updated properties
            examples: Updated examples
            related_concepts: Updated related concepts
            confidence: Updated confidence
            tags: Updated tags
            
        Returns:
            The updated memory item
        """
        # Calculate new confidence (weighted average)
        new_confidence = (existing.confidence + confidence) / 2
        
        # Merge properties
        merged_properties = {**existing.properties, **properties}
        
        # Merge examples and related concepts
        merged_examples = list(set(existing.examples + examples))
        merged_related = list(set(existing.related_concepts + related_concepts))
        
        # Merge tags
        merged_tags = existing.tags.union(tags)
        
        # Create embedding for the updated memory
        embedding_text = self._create_embedding_text(
            existing.concept, definition, merged_properties, 
            merged_examples, merged_related
        )
        vector_embedding = self._get_embedding(embedding_text) if embedding_text else None
        
        # Update the memory
        existing.definition = definition
        existing.properties = merged_properties
        existing.examples = merged_examples
        existing.related_concepts = merged_related
        existing.confidence = new_confidence
        existing.tags = merged_tags
        existing.vector_embedding = vector_embedding or existing.vector_embedding
        existing.strength = self._calculate_initial_strength(new_confidence, len(merged_properties))
        existing.last_accessed = datetime.now()  # Update access time
        
        # Update indexes
        self._update_indexes(existing)
        
        # Publish event
        self._publish_memory_event("semantic_memory_updated", {
            "memory_id": str(existing.id),
            "concept": existing.concept,
            "confidence": new_confidence
        })
        
        logger.info(f"Updated semantic memory for concept '{existing.concept}': {existing.id}")
        return existing
    
    def get_memory_by_concept(self, concept: str) -> Optional[SemanticMemoryItem]:
        """
        Get a semantic memory by concept name.
        
        Args:
            concept: The concept name to search for
            
        Returns:
            The semantic memory or None if not found
        """
        # Normalize concept name
        concept_key = concept.lower()
        
        # Get memory IDs for the concept
        memory_ids = self._concept_index.get(concept_key, [])
        
        if memory_ids:
            # Get the memory
            memory = self._memories.get(memory_ids[0])
            
            if memory:
                # Update access information
                memory.update_access()
                
            return memory
        
        return None
    
    def get_memory(self, memory_id: UUID) -> Optional[SemanticMemoryItem]:
        """
        Get a specific semantic memory.
        
        Args:
            memory_id: ID of the memory to get
            
        Returns:
            The semantic memory or None if not found
        """
        memory = self._memories.get(memory_id)
        
        if memory:
            # Update access information
            memory.update_access()
            
        return memory
    
    def get_related_concepts(self, concept: str) -> List[SemanticMemoryItem]:
        """
        Get memories for concepts related to a given concept.
        
        Args:
            concept: The concept to find related concepts for
            
        Returns:
            List of memories for related concepts
        """
        # Normalize concept name
        concept_key = concept.lower()
        
        # Get related concept names
        related_concepts = self._related_concepts.get(concept_key, set())
        
        # Get memories for related concepts
        result = []
        for related in related_concepts:
            memory = self.get_memory_by_concept(related)
            if memory:
                result.append(memory)
        
        return result
    
    def get_memories_by_property(self, property_name: str) -> List[SemanticMemoryItem]:
        """
        Get memories that have a specific property.
        
        Args:
            property_name: The property name to search for
            
        Returns:
            List of memories with the property
        """
        # Normalize property name
        property_key = property_name.lower()
        
        # Get memory IDs for the property
        memory_ids = self._property_index.get(property_key, [])
        
        # Get the memories
        result = []
        for memory_id in memory_ids:
            memory = self._memories.get(memory_id)
            if memory:
                memory.update_access()
                result.append(memory)
        
        return result
    
    def get_count(self) -> int:
        """
        Get the count of semantic memories.
        
        Returns:
            Number of semantic memories
        """
        return len(self._memories)
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age and adjust behavior.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        logger.info(f"Semantic memory age updated to {new_age}")
    
    def _calculate_initial_strength(self, confidence: float, property_count: int) -> float:
        """
        Calculate initial memory strength based on confidence and richness.
        
        Args:
            confidence: Confidence in the knowledge
            property_count: Number of properties (richness of the concept)
            
        Returns:
            Initial memory strength
        """
        # Base strength from confidence
        base_strength = confidence
        
        # Boost for richness (more properties = stronger memory)
        richness_factor = min(0.3, property_count * 0.03)  # Up to 0.3 boost
        
        # Adjust by developmental age (younger minds have weaker semantic memories)
        age_factor = min(1.0, max(0.5, self._developmental_age))
        
        # Calculate final strength
        strength = base_strength + richness_factor
        strength = strength * age_factor
        
        # Return capped strength
        return max(0.1, min(0.9, strength))
    
    def _create_embedding_text(
        self,
        concept: str,
        definition: str,
        properties: Dict[str, Any],
        examples: List[str],
        related_concepts: List[str]
    ) -> str:
        """
        Create text for embedding from memory components.
        
        Args:
            concept: The concept name
            definition: Definition of the concept
            properties: Properties of the concept
            examples: Examples of the concept
            related_concepts: Related concepts
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Add concept
        parts.append(f"Concept: {concept}")
        
        # Add definition
        parts.append(f"Definition: {definition}")
        
        # Add properties
        if properties:
            property_strs = [f"{key}: {value}" for key, value in properties.items()]
            parts.append(f"Properties: {'; '.join(property_strs)}")
        
        # Add examples
        if examples:
            parts.append(f"Examples: {'; '.join(examples)}")
        
        # Add related concepts
        if related_concepts:
            parts.append(f"Related concepts: {', '.join(related_concepts)}")
        
        return " ".join(parts)
    
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
    
    def _update_indexes(self, memory: SemanticMemoryItem) -> None:
        """
        Update memory indexes with a new memory.
        
        Args:
            memory: The memory to index
        """
        # Update concept index
        concept_key = memory.concept.lower()
        if concept_key not in self._concept_index:
            self._concept_index[concept_key] = []
        if memory.id not in self._concept_index[concept_key]:
            self._concept_index[concept_key].append(memory.id)
        
        # Update related concepts index
        for related in memory.related_concepts:
            related_key = related.lower()
            
            # Add concept -> related
            self._related_concepts[concept_key].add(related_key)
            
            # Add related -> concept (bidirectional)
            self._related_concepts[related_key].add(concept_key)
        
        # Update property index
        for prop_name in memory.properties.keys():
            prop_key = prop_name.lower()
            if memory.id not in self._property_index[prop_key]:
                self._property_index[prop_key].append(memory.id)
    
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
            source="memory.semantic_memory",
            recipient=Recipient.BROADCAST,
            content={
                "event_type": event_type,
                **payload
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
