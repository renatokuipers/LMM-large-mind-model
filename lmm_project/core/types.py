from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

# Development stages
class DevelopmentalStage(str, Enum):
    PRENATAL = "prenatal"
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"

# Module types
class ModuleType(str, Enum):
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    LANGUAGE = "language"
    EMOTION = "emotion"
    CONSCIOUSNESS = "consciousness"
    EXECUTIVE = "executive"
    SOCIAL = "social"
    MOTIVATION = "motivation"
    TEMPORAL = "temporal"
    CREATIVITY = "creativity"
    SELF_REGULATION = "self_regulation"
    LEARNING = "learning"
    IDENTITY = "identity"
    BELIEF = "belief"

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
    CAREGIVER = "caregiver"
    PEER = "peer"
    AUTHORITY = "authority"
    STRANGER = "stranger"

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
