"""
Core type definitions for the Large Mind Model (LMM) project.

This module provides common type definitions, enums, and type aliases
used throughout the LMM system.
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict, Literal

# Development stages
class DevelopmentalStage(str, Enum):
    """
    Developmental stages of the cognitive system.
    
    Each stage represents a distinct period of cognitive development with
    different capabilities, learning rates, and homeostatic setpoints.
    """
    PRENATAL = "prenatal"     # Initial formation (0.0-0.1)
    INFANT = "infant"         # Early development (0.1-0.3)
    CHILD = "child"           # Expanding capabilities (0.3-0.6)
    ADOLESCENT = "adolescent" # Advanced reasoning (0.6-0.8)
    ADULT = "adult"           # Full development (0.8-1.0)
    
    @classmethod
    def from_level(cls, level: float) -> 'DevelopmentalStage':
        """Convert a development level (0.0-1.0) to a developmental stage."""
        if level < 0.1:
            return cls.PRENATAL
        elif level < 0.3:
            return cls.INFANT
        elif level < 0.6:
            return cls.CHILD
        elif level < 0.8:
            return cls.ADOLESCENT
        else:
            return cls.ADULT
    
    @property
    def min_level(self) -> float:
        """Get the minimum development level for this stage."""
        if self == self.PRENATAL:
            return 0.0
        elif self == self.INFANT:
            return 0.1
        elif self == self.CHILD:
            return 0.3
        elif self == self.ADOLESCENT:
            return 0.6
        else:  # ADULT
            return 0.8
    
    @property
    def max_level(self) -> float:
        """Get the maximum development level for this stage."""
        if self == self.PRENATAL:
            return 0.1
        elif self == self.INFANT:
            return 0.3
        elif self == self.CHILD:
            return 0.6
        elif self == self.ADOLESCENT:
            return 0.8
        else:  # ADULT
            return 1.0
    
    def contains_level(self, level: float) -> bool:
        """Check if the given development level is within this stage."""
        return self.min_level <= level < self.max_level or \
               (self == self.ADULT and level >= self.min_level)

# Module types
class ModuleType(str, Enum):
    """
    Types of cognitive modules in the LMM system.
    
    Each module type represents a specialized component handling
    specific aspects of cognition.
    """
    # Core modules
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    LANGUAGE = "language"
    EMOTION = "emotion"
    CONSCIOUSNESS = "consciousness"
    EXECUTIVE = "executive"
    
    # Social and higher-order modules
    SOCIAL = "social"
    MOTIVATION = "motivation"
    TEMPORAL = "temporal"
    CREATIVITY = "creativity"
    SELF_REGULATION = "self_regulation"
    LEARNING = "learning"
    IDENTITY = "identity"
    BELIEF = "belief"
    
    # Homeostasis modules
    ENERGY = "energy"
    AROUSAL = "arousal"
    COGNITIVE_LOAD = "cognitive_load"
    SOCIAL_NEED = "social_need"
    COHERENCE = "coherence"
    
    # Meta modules
    DEVELOPMENT = "development"
    INTEGRATION = "integration"
    
    @property
    def is_homeostatic(self) -> bool:
        """Whether this is a homeostatic regulation module."""
        return self in [
            self.ENERGY, self.AROUSAL, self.COGNITIVE_LOAD, 
            self.SOCIAL_NEED, self.COHERENCE
        ]
    
    @property
    def is_core_cognitive(self) -> bool:
        """Whether this is a core cognitive module."""
        return self in [
            self.PERCEPTION, self.ATTENTION, self.MEMORY, 
            self.LANGUAGE, self.EMOTION, self.CONSCIOUSNESS, 
            self.EXECUTIVE
        ]

# Neural activation types
class ActivationType(str, Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"

# Learning types
class LearningType(str, Enum):
    HEBBIAN = "hebbian"
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    META = "meta"

# Neural connection types
class ConnectionType(str, Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"

# Emotion types
class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    DISGUST = "disgust"

# Memory types
class MemoryType(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ASSOCIATIVE = "associative"

# Social relationship types
class RelationshipType(str, Enum):
    """
    Types of social relationships that can be formed.
    
    Relationship types influence social interaction patterns, attachment,
    and social learning opportunities.
    """
    CAREGIVER = "caregiver"   # Primary caregivers (mother, father, etc.)
    PEER = "peer"             # Same-level individuals (friends, classmates)
    AUTHORITY = "authority"   # Non-caregiver authorities (teachers, leaders)
    ROMANTIC = "romantic"     # Emotional/intimate connections
    ACQUAINTANCE = "acquaintance"  # Casual/limited interactions
    MENTOR = "mentor"         # Guidance-focused relationships
    DEPENDENT = "dependent"   # Those being cared for by the system
    
    @property
    def is_attachment_figure(self) -> bool:
        """Return whether this relationship type can be an attachment figure."""
        return self in [self.CAREGIVER, self.MENTOR, self.ROMANTIC]
    
    @property
    def social_distance(self) -> float:
        """Return a measure of social distance (0.0-1.0, lower is closer)."""
        distances = {
            self.CAREGIVER: 0.1,
            self.ROMANTIC: 0.2,
            self.PEER: 0.3,
            self.MENTOR: 0.4,
            self.DEPENDENT: 0.5,
            self.AUTHORITY: 0.7,
            self.ACQUAINTANCE: 0.9
        }
        return distances.get(self, 0.5)

# Homeostatic signal types
class HomeostaticSignalType(str, Enum):
    """
    Types of homeostatic signals used for internal regulation.
    
    These signals indicate different regulatory needs and responses
    within the cognitive system.
    """
    NEED_INCREASE = "need_increase"       # Signal to increase a homeostatic need
    NEED_DECREASE = "need_decrease"       # Signal to decrease a homeostatic need
    URGENT_DEFICIT = "urgent_deficit"     # Critical deficit requiring immediate attention
    EXCESS_REGULATION = "excess_regulation"  # Need to reduce excessive levels
    RETURN_TO_SETPOINT = "return_to_setpoint"  # Restore to optimal level
    ADAPTATION_CHANGE = "adaptation_change"  # Update to homeostatic parameters
    COMPENSATORY_RESPONSE = "compensatory_response"  # Response to address imbalance

# Common data structure for neural activations
@dataclass
class Activation:
    value: float
    source: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# Type aliases for common complex types
ActivationMap = Dict[str, Activation]
EmbeddingVector = List[float]
NeuralWeights = Dict[str, Dict[str, float]]
StateDict = Dict[str, Any]
ModuleMap = Dict[str, Any]

# Type for representing cognitive embedding vectors
EmbeddingVector = List[float]

# Structured types for state serialization
class ModuleStateDict(TypedDict):
    """Dictionary containing a module's complete state."""
    module_type: ModuleType
    development_level: float
    parameters: Dict[str, Any]
    data: Dict[str, Any]

class HomeostasisStateDict(TypedDict):
    """Dictionary containing homeostasis system state."""
    energy_level: float
    arousal_level: float
    cognitive_load: float
    social_need: float
    coherence_level: float
    setpoints: Dict[str, float]
    modifiers: Dict[str, Any]

class SystemStateDict(TypedDict):
    """Dictionary containing complete system state."""
    version: str
    timestamp: str
    development_level: float
    developmental_stage: DevelopmentalStage
    modules: Dict[str, ModuleStateDict]
    homeostasis: HomeostasisStateDict
    relationships: Dict[str, Dict[str, Any]]
