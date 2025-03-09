from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, model_validator
import numpy as np
from datetime import datetime

class SensoryInput(BaseModel):
    """
    Represents raw sensory input data received from the environment.
    For this LMM implementation, all sensory input is text-based.
    """
    input_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    text: str
    source: str = "mother"  # e.g., 'mother', 'environment', 'internal'
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class Pattern(BaseModel):
    """
    Represents a detected pattern in sensory input
    """
    pattern_id: str
    pattern_type: Literal["token", "n_gram", "semantic", "syntactic", "temporal"] 
    content: Any  # Could be a string, embedding, or other representation
    confidence: float = Field(ge=0.0, le=1.0)
    activation: float = Field(ge=0.0, le=1.0)
    
    # Relationship to other patterns (for hierarchical pattern building)
    parent_patterns: List[str] = Field(default_factory=list)
    child_patterns: List[str] = Field(default_factory=list)
    
    # Metadata for pattern analysis
    frequency: int = 0  # How often this pattern has been encountered
    last_seen: Optional[datetime] = None
    first_seen: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class PerceptionResult(BaseModel):
    """
    The processed result of perception, to be passed to other modules
    """
    input_id: str  # Reference to the original input
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Detected patterns with their activations
    detected_patterns: List[Pattern] = Field(default_factory=list)
    
    # Novelty measure (how unfamiliar is this input)
    novelty_score: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Intensity measure (how strong is this input)
    intensity_score: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Feature vector representation of the input (for neural processing)
    feature_vector: Optional[List[float]] = None
    
    # Simplified semantic content (for debugging and introspection)
    semantic_content: Dict[str, Any] = Field(default_factory=dict)
    
    # Development-specific properties
    developmental_level: float = Field(ge=0.0, le=1.0, default=0.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class PerceptionMemory(BaseModel):
    """
    Short-term memory structure specific to the perception module
    """
    recent_inputs: List[SensoryInput] = Field(default_factory=list, max_items=10)
    known_patterns: Dict[str, Pattern] = Field(default_factory=dict)
    pattern_frequency: Dict[str, int] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class PerceptionParameters(BaseModel):
    """
    Configurable parameters for the perception module
    """
    # Sensitivity to different types of patterns
    token_sensitivity: float = Field(ge=0.0, le=1.0, default=0.5)
    ngram_sensitivity: float = Field(ge=0.0, le=1.0, default=0.3)
    semantic_sensitivity: float = Field(ge=0.0, le=1.0, default=0.2)
    
    # Novelty detection parameters
    novelty_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    
    # Pattern recognition thresholds
    pattern_activation_threshold: float = Field(ge=0.0, le=1.0, default=0.3)
    pattern_confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.2)
    
    # Developmental adaptation
    developmental_scaling: bool = True  # Whether to scale parameters based on development
    
    model_config = {
        "arbitrary_types_allowed": True
    }
