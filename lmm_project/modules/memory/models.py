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
