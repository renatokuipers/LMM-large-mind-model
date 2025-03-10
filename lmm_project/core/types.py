from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypeVar, Generic
import uuid
from datetime import datetime


class ModuleType(Enum):
    """Types of cognitive modules available in the system."""
    PERCEPTION = auto()
    ATTENTION = auto()
    MEMORY = auto()
    LANGUAGE = auto()
    EMOTION = auto()
    CONSCIOUSNESS = auto()
    EXECUTIVE = auto()
    SOCIAL = auto()
    MOTIVATION = auto()
    TEMPORAL = auto()
    CREATIVITY = auto()
    SELF_REGULATION = auto()
    LEARNING = auto()
    IDENTITY = auto()
    BELIEF = auto()


class MessageType(Enum):
    """Types of messages that can be exchanged between modules."""
    PERCEPTION_INPUT = auto()
    ATTENTION_FOCUS = auto()
    MEMORY_STORAGE = auto()
    MEMORY_RETRIEVAL = auto()
    EMOTION_UPDATE = auto()
    CONSCIOUSNESS_BROADCAST = auto()
    LANGUAGE_PROCESSING = auto()
    EXECUTIVE_CONTROL = auto()
    SOCIAL_INTERACTION = auto()
    MOTIVATION_SIGNAL = auto()
    TEMPORAL_PROCESSING = auto()
    CREATIVITY_OUTPUT = auto()
    SELF_REGULATION_SIGNAL = auto()
    LEARNING_UPDATE = auto()
    IDENTITY_UPDATE = auto()
    BELIEF_UPDATE = auto()
    DEVELOPMENT_MILESTONE = auto()
    SYSTEM_STATUS = auto()


class DevelopmentalStage(Enum):
    """Developmental stages of the mind."""
    PRENATAL = auto()  # 0.0-0.1 age units
    INFANT = auto()    # 0.1-1.0 age units
    CHILD = auto()     # 1.0-3.0 age units
    ADOLESCENT = auto() # 3.0-6.0 age units
    ADULT = auto()     # 6.0+ age units


# Type aliases for improved readability
ModuleID = str
MessageID = str
Timestamp = float
Age = float
DevelopmentLevel = float  # 0.0 to 1.0 scale for module development
Vector = List[float]  # For neural activations and embeddings
StateDict = Dict[str, Any]

# Type variable for generic functions
T = TypeVar('T')

def generate_id() -> str:
    """Generate a unique ID string."""
    return str(uuid.uuid4())

def current_timestamp() -> Timestamp:
    """Get current timestamp in seconds since epoch."""
    return datetime.now().timestamp()
