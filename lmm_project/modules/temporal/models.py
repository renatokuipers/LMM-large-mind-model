from pydantic import BaseModel, Field 
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from uuid import uuid4
import numpy as np

# Sequence Learning Models
class SequencePattern(BaseModel):
    """Represents a learned temporal pattern of elements"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    elements: List[Any] = Field(default_factory=list)  # The sequence elements
    transitions: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # Element -> next element -> probability
    frequency: int = Field(default=1)  # How many times this sequence has been observed
    last_observed: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)  # Context in which this pattern occurs
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Confidence in this pattern
    
    model_config = {
        "extra": "forbid"
    }

class HierarchicalSequence(BaseModel):
    """Represents a hierarchical organization of sequence patterns"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    sub_sequences: List[str] = Field(default_factory=list)  # IDs of constituent sequences
    transitions: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # Sub-sequence -> next sub-sequence -> probability
    abstraction_level: int = Field(default=1, ge=1)  # Higher values indicate higher abstraction
    
    model_config = {
        "extra": "forbid"
    }

# Causality Models
class CausalRelationship(BaseModel):
    """Represents a causal relationship between cause and effect"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    cause: str  # Identifier for the cause
    effect: str  # Identifier for the effect
    strength: float = Field(default=0.5, ge=0.0, le=1.0)  # Causal strength
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Confidence in the relationship
    mechanism: Optional[str] = None  # Description of causal mechanism
    temporal_delay: Optional[float] = None  # Typical delay between cause and effect
    bidirectional: bool = False  # Whether the relationship works both ways
    context_dependencies: Dict[str, float] = Field(default_factory=dict)  # How context affects causal strength
    observed_count: int = Field(default=1)  # How many times this relationship has been observed
    
    model_config = {
        "extra": "forbid"
    }

class CausalModel(BaseModel):
    """Represents a network of causal relationships"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    relationships: Dict[str, CausalRelationship] = Field(default_factory=dict)  # ID -> relationship
    variables: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Variable properties
    context: Dict[str, Any] = Field(default_factory=dict)  # Context for this causal model
    
    model_config = {
        "extra": "forbid"
    }

# Prediction Models
class Prediction(BaseModel):
    """Represents a prediction about a future state or event"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    target: str  # What is being predicted
    predicted_value: Any  # The predicted value or state
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Prediction confidence
    time_horizon: float  # How far in the future (in time units)
    creation_time: datetime = Field(default_factory=datetime.now)
    basis: Dict[str, Any] = Field(default_factory=dict)  # Basis for the prediction
    predictive_model_id: Optional[str] = None  # ID of the model that generated this
    probability_distribution: Optional[Dict[Any, float]] = None  # For probabilistic predictions
    
    model_config = {
        "extra": "forbid"
    }

class PredictiveModel(BaseModel):
    """Represents a model used to generate predictions"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    model_type: str  # Type of predictive model (e.g., "statistical", "causal", "sequence-based")
    target_domain: str  # Domain this model predicts (e.g., "weather", "agent behavior", "system state")
    inputs: List[str] = Field(default_factory=list)  # Input variables
    output: str  # Output variable
    accuracy_history: List[Tuple[datetime, float]] = Field(default_factory=list)  # Historical accuracy
    current_accuracy: float = Field(default=0.5, ge=0.0, le=1.0)  # Current accuracy estimate
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Model parameters
    
    model_config = {
        "extra": "forbid"
    }

# Time Perception Models
class TimeInterval(BaseModel):
    """Represents a perceived time interval"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: datetime
    end_time: Optional[datetime] = None  # None if interval is ongoing
    duration_estimate: Optional[float] = None  # Estimated duration in seconds
    actual_duration: Optional[float] = None  # Actual duration if known
    context: Dict[str, Any] = Field(default_factory=dict)  # Context during interval
    events: List[str] = Field(default_factory=list)  # Events within this interval
    subjective_duration: Optional[float] = None  # Subjective duration perception (e.g., 1.0 = accurate, >1 = felt longer)
    
    model_config = {
        "extra": "forbid"
    }

class TemporalRhythm(BaseModel):
    """Represents a detected temporal rhythm or cycle"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    period: float  # Duration of one cycle in seconds
    phase: float = Field(default=0.0)  # Current phase (0.0 to 1.0)
    stability: float = Field(default=0.5, ge=0.0, le=1.0)  # How stable the rhythm is
    domain: str  # Domain of the rhythm (e.g., "circadian", "speech", "music")
    detected_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Confidence in detected rhythm
    
    model_config = {
        "extra": "forbid"
    }

class TemporalContext(BaseModel):
    """Represents the current temporal context"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    current_time: datetime = Field(default_factory=datetime.now)
    active_intervals: List[str] = Field(default_factory=list)  # IDs of active time intervals
    active_rhythms: List[str] = Field(default_factory=list)  # IDs of active rhythms
    subjective_time_rate: float = Field(default=1.0)  # Rate of subjective time perception (1.0 = normal)
    temporal_focus: str = Field(default="present")  # "past", "present", or "future"
    time_horizon: Dict[str, float] = Field(default_factory=lambda: {"past": 3600, "future": 3600})  # Time horizons in seconds
    
    model_config = {
        "extra": "forbid"
    }
