from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator

class SensoryModality(str, Enum):
    """Sensory modalities available to the perception system"""
    TEXT = "text"                # Textual input (primary modality from Mother)
    AUDIO = "audio"              # Audio input (voice from Mother)
    VISUAL = "visual"            # Visual input (for future use)
    EMOTIONAL = "emotional"      # Emotional signals from interaction
    ABSTRACT = "abstract"        # Abstract concepts or thoughts
    INTERNAL = "internal"        # Internal sensations and states

class SalienceLevel(float, Enum):
    """Standard salience levels for sensory input"""
    NONE = 0.0
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0

class SensoryFeature(BaseModel):
    """A single extracted feature from sensory input"""
    name: str
    value: float
    modality: SensoryModality
    
    model_config = {"extra": "forbid"}

class SensoryInput(BaseModel):
    """Raw sensory input to the perception system"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    modality: SensoryModality
    content: Union[str, Dict[str, Any], List[float]]
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    salience: float = Field(default=SalienceLevel.MEDIUM, ge=0.0, le=1.0)
    
    model_config = {"extra": "forbid"}

class ProcessedInput(BaseModel):
    """Processed sensory input with extracted features"""
    id: UUID
    raw_input_id: UUID
    timestamp: datetime = Field(default_factory=datetime.now)
    modality: SensoryModality
    features: List[SensoryFeature] = Field(default_factory=list)
    context_vector: List[float] = Field(default_factory=list)
    salience: float = Field(default=SalienceLevel.MEDIUM, ge=0.0, le=1.0)
    
    model_config = {"extra": "forbid"}

class PatternType(str, Enum):
    """Types of patterns that can be recognized"""
    TEMPORAL = "temporal"        # Patterns over time
    SPATIAL = "spatial"          # Patterns in space
    SEMANTIC = "semantic"        # Patterns of meaning
    EMOTIONAL = "emotional"      # Patterns of emotion
    CAUSAL = "causal"            # Cause-effect patterns
    ASSOCIATIVE = "associative"  # Co-occurrence patterns
    CATEGORICAL = "categorical"  # Category patterns

class RecognizedPattern(BaseModel):
    """A pattern recognized in sensory input"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    pattern_type: PatternType
    pattern_key: str
    confidence: float = Field(ge=0.0, le=1.0)
    salience: float = Field(default=SalienceLevel.MEDIUM, ge=0.0, le=1.0)
    input_ids: List[UUID] = Field(default_factory=list)
    features: List[SensoryFeature] = Field(default_factory=list)
    vector_representation: Optional[List[float]] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class PerceptionEvent(BaseModel):
    """Event generated by the perception system"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    source_module: str = "perception"
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class PerceptionState(BaseModel):
    """Current state of the perception module"""
    active_modalities: List[SensoryModality] = Field(default_factory=list)
    current_salience_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    active_inputs: Dict[UUID, SensoryInput] = Field(default_factory=dict)
    recent_patterns: List[RecognizedPattern] = Field(default_factory=list)
    buffer_capacity: int = Field(default=10, ge=1)
    developmental_age: float = Field(default=0.0, ge=0.0)
    
    model_config = {"extra": "forbid"}

class PerceptionConfig(BaseModel):
    """Configuration for the perception module"""
    default_modalities: List[SensoryModality] = Field(
        default=[SensoryModality.TEXT, SensoryModality.EMOTIONAL]
    )
    base_salience_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    buffer_capacity: int = Field(default=10, ge=1)
    feature_extraction_resolution: float = Field(default=0.5, ge=0.1, le=1.0)
    pattern_recognition_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    vector_dimension: int = Field(default=768)
    enable_emotional_processing: bool = Field(default=True)
    contextual_memory_length: int = Field(default=5, ge=1)
    
    model_config = {"extra": "forbid"}
