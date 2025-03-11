from enum import Enum
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
import numpy as np
import os
from pathlib import Path

# Constants for memory system
EMBEDDING_DIM = 768  # Standard embedding dimension
DEFAULT_ASSOCIATION_THRESHOLD = 0.2
DEFAULT_WORKING_MEMORY_BASE_CAPACITY = 5  # Miller's number (7±2), starting lower for development

class MemoryType(str, Enum):
    """Types of memory available in the memory system"""
    WORKING = "working"       # Short-term active memory
    EPISODIC = "episodic"     # Event-based memories (experiences)
    SEMANTIC = "semantic"     # Factual knowledge and concepts
    PROCEDURAL = "procedural" # How-to knowledge and skills
    ASSOCIATIVE = "associative" # Associations between memories

class MemoryStatus(str, Enum):
    """Status of a memory item"""
    ACTIVE = "active"         # Currently active in working memory
    CONSOLIDATED = "consolidated" # Stored in long-term memory
    DECAYING = "decaying"     # In the process of being forgotten
    FORGOTTEN = "forgotten"   # No longer directly accessible

class MemoryStrength(float, Enum):
    """Standard memory strength levels"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class EmotionalValence(float, Enum):
    """Emotional valence of a memory"""
    VERY_NEGATIVE = -0.8
    NEGATIVE = -0.4
    NEUTRAL = 0.0
    POSITIVE = 0.4
    VERY_POSITIVE = 0.8

class MemoryItem(BaseModel):
    """Base model for any memory item"""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    memory_type: MemoryType
    content: Dict[str, Any]
    source: Optional[str] = None
    strength: float = Field(default=MemoryStrength.MODERATE, ge=0.0, le=1.0)
    emotional_valence: float = Field(default=EmotionalValence.NEUTRAL, ge=-1.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    tags: Set[str] = Field(default_factory=set)
    vector_embedding: Optional[List[float]] = None
    access_count: int = Field(default=1, ge=0)
    developmental_factors: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}
    
    def update_access(self) -> None:
        """Update access timestamp and count when memory is accessed"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class WorkingMemoryItem(MemoryItem):
    """Item held in working memory"""
    activation_level: float = Field(default=0.8, ge=0.0, le=1.0)
    retention_time_ms: int = Field(default=30000)  # Default 30 seconds
    related_items: List[UUID] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.02, ge=0.0, le=1.0)  # Rate of activation decay per update
    neural_activation_pattern: Optional[List[float]] = None  # Neural network activation pattern
    
    model_config = {"extra": "forbid"}

class EpisodicMemoryItem(MemoryItem):
    """Event-based memory (experience)"""
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    location: Optional[str] = None
    actors: List[str] = Field(default_factory=list)
    sequence: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: Optional[str] = None
    context_vector: Optional[List[float]] = None  # Context embedding for retrieval by similarity
    temporal_index: Optional[float] = None  # Temporal position for sequential retrieval
    
    model_config = {"extra": "forbid"}

class SemanticMemoryItem(MemoryItem):
    """Factual knowledge or concept"""
    concept: str
    definition: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    examples: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    concept_vector: Optional[List[float]] = None  # Concept-specific embedding
    ontology_path: Optional[List[str]] = None  # Hierarchical path in knowledge structure
    
    model_config = {"extra": "forbid"}

class AssociativeLink(BaseModel):
    """Association between two memory items"""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    association_type: str
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_reinforced: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = Field(default=1, ge=1)
    bidirectional: bool = Field(default=False)
    features: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}
    
    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce the association by increasing strength"""
        self.strength = min(1.0, self.strength + amount)
        self.last_reinforced = datetime.now()
        self.reinforcement_count += 1

class MemoryQuery(BaseModel):
    """Query for memory retrieval"""
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    memory_type: Optional[MemoryType] = None
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    time_range: Optional[Dict[str, datetime]] = None
    min_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1)
    include_content: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    sort_by: Optional[str] = None  # Field to sort results by (e.g., "created_at", "strength")
    sort_direction: Optional[str] = None  # "asc" or "desc"
    
    model_config = {"extra": "forbid"}
    
    @field_validator('time_range')
    @classmethod
    def validate_time_range(cls, v: Optional[Dict[str, datetime]]) -> Optional[Dict[str, datetime]]:
        """Validate that time range has start and/or end"""
        if v is not None:
            valid_keys = {'start', 'end'}
            if not any(key in v for key in valid_keys):
                raise ValueError("Time range must contain 'start' and/or 'end'")
        return v

class MemoryRetrievalResult(BaseModel):
    """Result of a memory retrieval operation"""
    id: UUID = Field(default_factory=uuid4)
    query: MemoryQuery
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_items: List[MemoryItem] = Field(default_factory=list)
    total_results: int = 0
    relevance_scores: Dict[UUID, float] = Field(default_factory=dict)
    execution_time_ms: Optional[int] = None
    relevance_histogram: Optional[Dict[str, int]] = None
    
    model_config = {"extra": "forbid"}

class ConsolidationCandidate(BaseModel):
    """A working memory item that is a candidate for consolidation"""
    item_id: UUID
    importance: float
    recency: float  # Normalized recency (1.0 = most recent)
    activation: float
    consolidation_score: float  # Combined score for consolidation priority
    decay_status: float  # Current decay status (0-1 where 1 = fully decayed)
    emotional_impact: float  # Impact of emotional state on consolidation
    consolidation_threshold: float  # Dynamic threshold based on development
    
    model_config = {"extra": "forbid"}

class MemoryEvent(BaseModel):
    """Event generated by the memory system"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    source_module: str = "memory"
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class MemoryState(BaseModel):
    """Current state of the memory module"""
    working_memory_capacity: int
    working_memory_used: int
    working_memory_items: Dict[UUID, Dict[str, Any]] = Field(default_factory=dict)
    episodic_memory_count: int = 0
    semantic_memory_count: int = 0
    associative_link_count: int = 0
    recent_retrievals: List[Dict[str, Any]] = Field(default_factory=list)
    developmental_age: float
    active_consolidation_count: int = 0
    vector_indexes_status: Dict[str, bool] = Field(default_factory=dict)
    neural_network_activation: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class MemoryConfig(BaseModel):
    """Configuration for the memory module"""
    # Core capacity settings
    base_working_memory_capacity: int = Field(default=DEFAULT_WORKING_MEMORY_BASE_CAPACITY, ge=1)
    max_retrieval_results: int = Field(default=20, ge=1)
    
    # Thresholds and rates
    consolidation_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    base_decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # Vector embeddings
    embedding_dimension: int = Field(default=EMBEDDING_DIM)
    min_association_strength: float = Field(default=DEFAULT_ASSOCIATION_THRESHOLD, ge=0.0, le=1.0)
    semantic_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    
    # Memory operation configurations
    episodic_memory_detail_level: float = Field(default=0.7, ge=0.0, le=1.0)
    memory_strength_reinforcement: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Neural network integrations
    use_neural_networks: bool = Field(default=True)
    network_learn_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    
    # Storage configurations
    vector_store_index_name: str = Field(default="memory_vectors")
    vector_store_gpu_enabled: bool = Field(default=True)
    vector_store_dir: str = Field(default_factory=lambda: str(Path(os.path.join("storage", "vectors"))))
    
    # Developmental factors
    age_capacity_multiplier: float = Field(default=1.5, ge=0.0)  # How much capacity increases with age
    age_decay_divisor: float = Field(default=3.0, ge=0.1)  # How much decay reduces with age
    age_detail_multiplier: float = Field(default=2.0, ge=0.0)  # How much detail increases with age
    
    model_config = {"extra": "forbid"}
    
    def get_working_memory_capacity(self, developmental_age: float) -> int:
        """Calculate working memory capacity based on developmental age"""
        base = self.base_working_memory_capacity
        age_factor = min(1.0, developmental_age / 5.0)  # Full capacity at age 5.0
        capacity_growth = self.age_capacity_multiplier * age_factor
        return max(base, int(base + capacity_growth * 5))  # Max +5 slots at full development
    
    def get_decay_rate(self, developmental_age: float) -> float:
        """Calculate decay rate based on developmental age"""
        base = self.base_decay_rate
        age_factor = min(1.0, developmental_age / 6.0)  # Full reduction at age 6.0
        decay_reduction = base * (age_factor / self.age_decay_divisor)
        return max(0.001, base - decay_reduction)  # Ensure some minimal decay remains
    
    def get_detail_level(self, developmental_age: float) -> float:
        """Calculate memory detail level based on developmental age"""
        base = self.episodic_memory_detail_level
        age_factor = min(1.0, developmental_age / 7.0)  # Full detail at age 7.0
        detail_growth = (1.0 - base) * (age_factor * self.age_detail_multiplier)
        return min(0.99, base + detail_growth)  # Cap at 0.99
