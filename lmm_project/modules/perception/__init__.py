"""
Perception Module

This module handles sensory input processing and pattern recognition,
serving as the input gateway for the LMM system. It receives sensory
data (primarily from the Mother interface), processes it to extract
features, and identifies patterns in the input.

Components:
    - SensoryInputProcessor: Processes raw sensory input and extracts features
    - PatternRecognizer: Recognizes patterns in processed inputs
    - PerceptionNetwork: Neural networks for perception processing

The perception module develops in sophistication with age, gradually
recognizing more complex patterns and features as the mind matures.
"""

from typing import Optional, Dict, List, Any, Set

from lmm_project.core.event_bus import EventBus
from lmm_project.utils.logging_utils import get_module_logger

from .models import (
    SensoryModality,
    SalienceLevel,
    SensoryInput,
    ProcessedInput,
    SensoryFeature,
    PatternType,
    RecognizedPattern,
    PerceptionEvent,
    PerceptionConfig,
    PerceptionState
)

from .sensory_input import SensoryInputProcessor
from .pattern_recognition import PatternRecognizer
from .neural_net import PerceptionNetwork

# Initialize logger
logger = get_module_logger("modules.perception")

# Module instances
_sensory_processor = None
_pattern_recognizer = None
_perception_network = None
_current_config = None
_current_age = 0.0
_initialized = False

def initialize(
    event_bus: EventBus,
    config: Optional[PerceptionConfig] = None,
    developmental_age: float = 0.0
) -> None:
    """
    Initialize the perception module.
    
    Args:
        event_bus: The event bus for communication
        config: Configuration for the perception module
        developmental_age: Current developmental age of the mind
    """
    global _sensory_processor, _pattern_recognizer, _perception_network
    global _current_config, _current_age, _initialized
    
    # Store configuration and age
    _current_config = config or PerceptionConfig()
    _current_age = developmental_age
    
    # Create components
    _sensory_processor = SensoryInputProcessor(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _pattern_recognizer = PatternRecognizer(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _perception_network = PerceptionNetwork(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _initialized = True
    
    logger.info(f"Perception module initialized with age {developmental_age}")

def process_input(sensory_input: SensoryInput) -> Optional[ProcessedInput]:
    """
    Process a sensory input.
    
    Args:
        sensory_input: The sensory input to process
        
    Returns:
        Processed input with extracted features, or None if below threshold
    """
    _ensure_initialized()
    return _sensory_processor.process_input(sensory_input)

def recognize_patterns(processed_input: ProcessedInput) -> List[RecognizedPattern]:
    """
    Recognize patterns in a processed input.
    
    Args:
        processed_input: The processed input to recognize patterns in
        
    Returns:
        List of recognized patterns
    """
    _ensure_initialized()
    return _pattern_recognizer.recognize_patterns(processed_input)

def create_sensory_input(
    content: Any,
    modality: SensoryModality = SensoryModality.TEXT,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    salience: float = SalienceLevel.MEDIUM
) -> SensoryInput:
    """
    Create a sensory input object.
    
    Args:
        content: The input content
        modality: The sensory modality
        source: Source of the input
        metadata: Additional metadata
        salience: Salience level of the input
        
    Returns:
        A SensoryInput object
    """
    return SensoryInput(
        modality=modality,
        content=content,
        source=source,
        metadata=metadata or {},
        salience=salience
    )

def get_recent_inputs(modality: SensoryModality, count: int = 5) -> List[SensoryInput]:
    """
    Get recent inputs for a specific modality.
    
    Args:
        modality: The sensory modality to get inputs for
        count: Maximum number of inputs to return
        
    Returns:
        List of recent inputs for the specified modality
    """
    _ensure_initialized()
    return _sensory_processor.get_recent_inputs(modality, count)

def get_recent_patterns(count: int = 10) -> List[RecognizedPattern]:
    """
    Get recently recognized patterns.
    
    Args:
        count: Maximum number of patterns to return
        
    Returns:
        List of recently recognized patterns
    """
    _ensure_initialized()
    return _pattern_recognizer.get_recent_patterns(count)

def get_known_patterns(pattern_type: Optional[PatternType] = None) -> Set[str]:
    """
    Get known pattern keys of a specific type or all types.
    
    Args:
        pattern_type: The pattern type to get, or None for all types
        
    Returns:
        Set of known pattern keys
    """
    _ensure_initialized()
    return _pattern_recognizer.get_known_patterns(pattern_type)

def get_state() -> PerceptionState:
    """
    Get the current state of the perception module.
    
    Returns:
        Current perception state
    """
    _ensure_initialized()
    
    return PerceptionState(
        active_modalities=list(_sensory_processor.get_active_inputs().keys()),
        current_salience_threshold=_current_config.base_salience_threshold,
        active_inputs=_sensory_processor.get_active_inputs(),
        recent_patterns=_pattern_recognizer.get_recent_patterns(),
        buffer_capacity=_current_config.buffer_capacity,
        developmental_age=_current_age
    )

def update_age(new_age: float) -> None:
    """
    Update the developmental age of the perception module.
    
    Args:
        new_age: The new developmental age
    """
    global _current_age
    _current_age = new_age
    logger.info(f"Perception module age updated to {new_age}")

def _ensure_initialized() -> None:
    """
    Ensure the perception module is initialized.
    
    Raises:
        RuntimeError: If the module is not initialized
    """
    if not _initialized:
        raise RuntimeError(
            "Perception module not initialized. Call initialize() first."
        )

# Export public API
__all__ = [
    # Classes
    'SensoryInputProcessor',
    'PatternRecognizer',
    'PerceptionNetwork',
    
    # Models
    'SensoryModality',
    'SalienceLevel',
    'SensoryInput',
    'ProcessedInput',
    'SensoryFeature',
    'PatternType',
    'RecognizedPattern',
    'PerceptionEvent',
    'PerceptionConfig',
    'PerceptionState',
    
    # Functions
    'initialize',
    'process_input',
    'recognize_patterns',
    'create_sensory_input',
    'get_recent_inputs',
    'get_recent_patterns',
    'get_known_patterns',
    'get_state',
    'update_age'
]
