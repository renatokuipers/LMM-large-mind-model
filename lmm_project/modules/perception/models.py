from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from datetime import datetime
import re

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
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "input_id": "sense_1234",
                    "text": "Hello, I'm here to help you learn.",
                    "source": "mother",
                    "context": {"interaction_type": "greeting"}
                }
            ]
        }
    }
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty"""
        if not v.strip():
            raise ValueError("Sensory input text cannot be empty")
        return v

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
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "pattern_id": "pat_5678",
                    "pattern_type": "token",
                    "content": "hello",
                    "confidence": 0.95,
                    "activation": 0.8,
                    "frequency": 3
                },
                {
                    "pattern_id": "pat_9012",
                    "pattern_type": "semantic",
                    "content": {"category": "greeting", "embedding": [0.1, 0.2, 0.3]},
                    "confidence": 0.85,
                    "activation": 0.7
                }
            ]
        }
    }
    
    @model_validator(mode='after')
    def validate_content_matches_type(self) -> 'Pattern':
        """Validate that content format matches pattern_type"""
        if self.pattern_type == "token" and not isinstance(self.content, str):
            raise ValueError("Token pattern content must be a string")
        
        if self.pattern_type == "n_gram" and not isinstance(self.content, str):
            raise ValueError("N-gram pattern content must be a string")
            
        if self.pattern_type == "semantic" and not isinstance(self.content, dict):
            raise ValueError("Semantic pattern content must be a dictionary")
            
        if self.pattern_type == "temporal" and not isinstance(self.content, list):
            raise ValueError("Temporal pattern content must be a list")
            
        return self

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
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "input_id": "sense_1234",
                    "novelty_score": 0.2,
                    "intensity_score": 0.7,
                    "semantic_content": {
                        "intent": "greeting",
                        "sentiment": "positive",
                        "entities": ["mother"]
                    },
                    "developmental_level": 0.3
                }
            ]
        }
    }
    
    @field_validator('feature_vector')
    @classmethod
    def validate_feature_vector(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate feature vector dimensions if present"""
        if v is not None and len(v) == 0:
            raise ValueError("Feature vector cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_patterns_consistency(self) -> 'PerceptionResult':
        """Check that detected patterns have consistent IDs"""
        pattern_ids = set()
        for pattern in self.detected_patterns:
            if pattern.pattern_id in pattern_ids:
                raise ValueError(f"Duplicate pattern ID: {pattern.pattern_id}")
            pattern_ids.add(pattern.pattern_id)
        return self

class PerceptionMemory(BaseModel):
    """
    Short-term memory structure specific to the perception module
    """
    recent_inputs: List[SensoryInput] = Field(default_factory=list, max_items=10)
    known_patterns: Dict[str, Pattern] = Field(default_factory=dict)
    pattern_frequency: Dict[str, int] = Field(default_factory=dict)
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "recent_inputs": [
                        {"input_id": "sense_1234", "text": "Hello there", "source": "mother"}
                    ],
                    "pattern_frequency": {"hello": 4, "there": 2}
                }
            ]
        }
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
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "token_sensitivity": 0.6,
                    "ngram_sensitivity": 0.4,
                    "semantic_sensitivity": 0.3,
                    "novelty_threshold": 0.8,
                    "pattern_activation_threshold": 0.4
                }
            ]
        }
    }
    
    @model_validator(mode='after')
    def validate_sensitivity_balance(self) -> 'PerceptionParameters':
        """Check that sensitivities are balanced appropriately"""
        total = self.token_sensitivity + self.ngram_sensitivity + self.semantic_sensitivity
        if total > 1.5:  # Allow some flexibility but prevent extreme values
            raise ValueError(f"Total sensitivity ({total}) is too high (should be ≤ 1.5)")
        return self
