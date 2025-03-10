"""
Core module for the LMM system.
Contains fundamental components for the cognitive architecture.
"""
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import key types
from .types import (
    ModuleType,
    MessageType,
    DevelopmentalStage,
    ModuleID,
    MessageID,
    Timestamp,
    Age,
    DevelopmentLevel,
    Vector,
    StateDict,
    generate_id,
    current_timestamp
)

# Import exceptions
from .exceptions import (
    LMMException,
    ModuleError,
    EventBusError,
    MessageError,
    StateError,
    DevelopmentError,
    StorageError,
    NeuralError,
    ConfigurationError,
    ResourceUnavailableError,
    InterfaceError
)

# Import message classes
from .message import (
    Content,
    TextContent,
    VectorContent,
    ImageContent,
    AudioContent,
    StructuredContent,
    Message
)

# Import event bus
from .event_bus import (
    EventBus,
    get_event_bus
)

# Import state manager
from .state_manager import (
    StateManager,
    get_state_manager
)

# Import mind
from .mind import (
    Mind,
    get_mind
)

# Version information
__version__ = "0.1.0"

# Export key components
__all__ = [
    # Types
    'ModuleType',
    'MessageType',
    'DevelopmentalStage',
    'ModuleID',
    'MessageID',
    'Timestamp',
    'Age',
    'DevelopmentLevel',
    'Vector',
    'StateDict',
    'generate_id',
    'current_timestamp',
    
    # Exceptions
    'LMMException',
    'ModuleError',
    'EventBusError',
    'MessageError',
    'StateError',
    'DevelopmentError',
    'StorageError',
    'NeuralError',
    'ConfigurationError',
    'ResourceUnavailableError',
    'InterfaceError',
    
    # Message classes
    'Content',
    'TextContent',
    'VectorContent',
    'ImageContent',
    'AudioContent',
    'StructuredContent',
    'Message',
    
    # Event bus
    'EventBus',
    'get_event_bus',
    
    # State manager
    'StateManager',
    'get_state_manager',
    
    # Mind
    'Mind',
    'get_mind',
    
    # Version
    '__version__'
]
