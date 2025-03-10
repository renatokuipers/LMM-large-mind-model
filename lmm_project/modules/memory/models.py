from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Set, Union, Literal
from datetime import datetime
import numpy as np
import uuid

class Memory(BaseModel):
    """Base class for all memory types in the system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    activation_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # How quickly this memory decays over time (lower = more stable)
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    # How many times this memory has been accessed
    access_count: int = Field(default=0, ge=0)
    # Last time this memory was accessed
    last_accessed: Optional[datetime] = None
    # Tags for this memory
    tags: Set[str] = Field(default_factory=set)
    # Emotional valence (-1.0 to 1.0)
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Emotional arousal (0.0 to 1.0)
    emotional_arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def update_activation(self, amount: float) -> None:
        """Update the activation level of this memory"""
        self.activation_level = max(0.0, min(1.0, self.activation_level + amount))
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def decay_activation(self, time_delta: float) -> None:
        """Decay the activation level over time"""
        decay_amount = self.decay_rate * time_delta
        self.activation_level = max(0.0, self.activation_level - decay_amount)


class WorkingMemoryItem(Memory):
    """An item in working memory - these are temporary and have limited capacity"""
    # Position in working memory buffer (lower = more attended to)
    buffer_position: int = Field(default=0, ge=0)
    # Time remaining until this item expires from working memory
    time_remaining: float = Field(default=30.0, ge=0.0)  # in seconds
    # Whether this item is being actively maintained through rehearsal
    is_rehearsed: bool = Field(default=False)
    # Reference to source in long-term memory, if any
    source_memory_id: Optional[str] = None


class SemanticMemory(Memory):
    """Semantic memory represents factual knowledge and concepts"""
    # Concept/knowledge represented
    concept: str
    # Confidence in this knowledge (0.0 to 1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Related concepts
    related_concepts: Dict[str, float] = Field(default_factory=dict)
    # Whether this is derived from experience or taught directly
    source_type: Literal["experience", "instruction"] = "experience"
    # Knowledge domain this belongs to
    domain: Optional[str] = None
    # Hierarchical position (if any)
    is_subconcept_of: Optional[str] = None
    has_subconcepts: List[str] = Field(default_factory=list)


class EpisodicMemory(Memory):
    """Episodic memory represents specific experiences and events"""
    # Where this memory took place
    context: str
    # When this memory took place (may be different than storage timestamp)
    event_time: datetime = Field(default_factory=datetime.now)
    # Other entities involved in this memory
    involved_entities: List[str] = Field(default_factory=list)
    # First-person perspective flag
    is_first_person: bool = Field(default=True)
    # How vivid/detailed this memory is (0.0 to 1.0)
    vividness: float = Field(default=0.8, ge=0.0, le=1.0)
    # Narrative sequence (if part of larger narrative)
    sequence_position: Optional[int] = None
    narrative_id: Optional[str] = None
    # Emotional impact at time of event (-1.0 to 1.0 for valence, 0.0 to 1.0 for intensity)
    emotional_impact: Dict[str, float] = Field(default_factory=dict)


class AssociativeLink(BaseModel):
    """A link between two memories"""
    source_id: str
    target_id: str
    # Strength of association (0.0 to 1.0)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    # Type of association
    link_type: str = "general"
    # When this association was formed
    formed_at: datetime = Field(default_factory=datetime.now)
    # How many times this association has been activated
    activation_count: int = Field(default=0, ge=0)
    
    def update_strength(self, amount: float) -> None:
        """Update the strength of this association"""
        self.strength = max(0.0, min(1.0, self.strength + amount))
        self.activation_count += 1


class MemoryConsolidationEvent(BaseModel):
    """Represents a consolidation event where memories are strengthened or weakened"""
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_ids: List[str]
    # How much each memory was strengthened (positive) or weakened (negative)
    strength_changes: Dict[str, float]
    # Why this consolidation happened
    reason: Literal["sleep", "rehearsal", "emotional_salience", "relevance"] = "rehearsal"
    # Any pattern discovered during consolidation
    discovered_pattern: Optional[str] = None


class MemoryRetrievalRequest(BaseModel):
    """Request model for memory retrieval operations"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: Literal["working", "episodic", "semantic", "associative", "all"] = "all"
    # Query parameters
    query: Optional[str] = None
    # Specific memory ID to retrieve
    memory_id: Optional[str] = None
    # Time frame for episodic memories (if applicable)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Context for context-based retrieval
    context: Optional[str] = None
    # Domain for semantic memory queries
    domain: Optional[str] = None
    # Tags to search for
    tags: Set[str] = Field(default_factory=set)
    # Whether to include memory content in results
    include_content: bool = Field(default=True)
    # Maximum number of results to return
    limit: int = Field(default=10, ge=1)
    # Additional parameters based on memory type
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalResult(BaseModel):
    """Result model for memory retrieval operations"""
    request_id: str
    status: Literal["success", "partial", "error"] = "success"
    memory_type: Literal["working", "episodic", "semantic", "associative", "all"]
    # Retrieved memory items
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    # Number of memories found
    count: int = Field(default=0)
    # Error message if status is error
    error: Optional[str] = None
    # Time taken for retrieval (ms)
    retrieval_time: float = Field(default=0.0)
    # Memory IDs that matched the query
    memory_ids: List[str] = Field(default_factory=list)
    # Additional result metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryStoreRequest(BaseModel):
    """Request model for memory storage operations"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: Literal["working", "episodic", "semantic", "associative"]
    # Memory content to store
    content: str
    # Memory metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Importance of the memory (0.0 to 1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # Tags for categorizing the memory
    tags: Set[str] = Field(default_factory=set)
    # Emotional factors
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotional_arousal: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryStoreResult(BaseModel):
    """Result model for memory storage operations"""
    request_id: str
    status: Literal["success", "error"] = "success"
    memory_type: Literal["working", "episodic", "semantic", "associative"]
    # ID of the stored memory
    memory_id: Optional[str] = None
    # Error message if status is error
    error: Optional[str] = None
    # Storage time (ms)
    storage_time: float = Field(default=0.0)
    # Additional result metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryIndex(BaseModel):
    """Index for efficient memory retrieval"""
    # Type of index
    index_type: Literal["temporal", "semantic", "tag", "context", "association"] = "temporal"
    # Mapping of keys to memory IDs
    keys_to_ids: Dict[str, List[str]] = Field(default_factory=dict)
    # Whether this index is enabled (for development-based enablement)
    enabled: bool = Field(default=True)
    # Last updated timestamp
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_entry(self, key: str, memory_id: str) -> None:
        """Add an entry to the index"""
        if key not in self.keys_to_ids:
            self.keys_to_ids[key] = []
        if memory_id not in self.keys_to_ids[key]:
            self.keys_to_ids[key].append(memory_id)
        self.last_updated = datetime.now()
    
    def remove_entry(self, key: str, memory_id: str) -> None:
        """Remove an entry from the index"""
        if key in self.keys_to_ids and memory_id in self.keys_to_ids[key]:
            self.keys_to_ids[key].remove(memory_id)
            if not self.keys_to_ids[key]:
                del self.keys_to_ids[key]
        self.last_updated = datetime.now()
    
    def get_memory_ids(self, key: str) -> List[str]:
        """Get memory IDs associated with a key"""
        return self.keys_to_ids.get(key, [])
    
    def get_all_memory_ids(self) -> List[str]:
        """Get all memory IDs in the index"""
        all_ids = []
        for ids in self.keys_to_ids.values():
            all_ids.extend(ids)
        return list(set(all_ids))  # Remove duplicates


class MemoryNeuralState(BaseModel):
    """Model for tracking neural states related to memory processing"""
    # Current activation patterns for different memory components
    activations: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Development levels for different memory components
    working_memory_development: float = Field(default=0.0, ge=0.0, le=1.0)
    episodic_memory_development: float = Field(default=0.0, ge=0.0, le=1.0) 
    semantic_memory_development: float = Field(default=0.0, ge=0.0, le=1.0)
    associative_memory_development: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Memory efficacy metrics
    working_memory_capacity: int = Field(default=3, ge=1)
    episodic_recall_accuracy: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_organization: float = Field(default=0.3, ge=0.0, le=1.0)
    associative_strength: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # Maximum activation storage (more recent activations only)
    max_activations_per_type: int = Field(default=20, ge=5)
    
    def add_activation(self, activation_type: str, data: Dict[str, Any]) -> None:
        """Add a new activation pattern for a memory component"""
        if activation_type not in self.activations:
            self.activations[activation_type] = []
            
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
            
        self.activations[activation_type].append(data)
        
        # Trim to max size
        if len(self.activations[activation_type]) > self.max_activations_per_type:
            self.activations[activation_type] = self.activations[activation_type][-self.max_activations_per_type:]
    
    def get_recent_activations(self, activation_type: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent activations for a component"""
        if activation_type not in self.activations:
            return []
            
        return self.activations[activation_type][-min(count, len(self.activations[activation_type])):]
    
    def clear_activations(self, activation_type: Optional[str] = None) -> None:
        """Clear activations for a type or all activations"""
        if activation_type:
            if activation_type in self.activations:
                self.activations[activation_type] = []
        else:
            self.activations = {}
