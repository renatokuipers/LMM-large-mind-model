"""
Attention Module

This module handles attentional focus and salience detection,
serving as the cognitive filtering system for the LMM system.
It determines what sensory inputs and patterns deserve focus,
manages attention shifts, and modulates cognitive resources.

Components:
    - FocusController: Manages attention focus and allocation
    - SalienceDetector: Evaluates the salience of inputs and patterns
    - AttentionNetwork: Neural networks for attention processing

The attention module develops in sophistication with age, gradually
improving in focus duration, selective attention, and divided attention
capabilities as the mind matures.
"""

from typing import Optional, Dict, List, Any, Set, Tuple

from lmm_project.core.event_bus import EventBus
from lmm_project.utils.logging_utils import get_module_logger

from .models import (
    AttentionMode,
    FocusLevel,
    AttentionTarget,
    SalienceFeature, 
    AttentionFocus,
    SalienceAssessment,
    AttentionEvent,
    AttentionConfig,
    AttentionState
)

from .focus_controller import FocusController
from .salience_detector import SalienceDetector
from .neural_net import AttentionNetwork

# Initialize logger
logger = get_module_logger("modules.attention")

# Module instances
_focus_controller = None
_salience_detector = None
_attention_network = None
_current_config = None
_current_age = 0.0
_initialized = False

def initialize(
    event_bus: EventBus,
    config: Optional[AttentionConfig] = None,
    developmental_age: float = 0.0
) -> None:
    """
    Initialize the attention module.
    
    Args:
        event_bus: The event bus for communication
        config: Configuration for the attention module
        developmental_age: Current developmental age of the mind
    """
    global _focus_controller, _salience_detector, _attention_network
    global _current_config, _current_age, _initialized
    
    # Store configuration and age
    _current_config = config or AttentionConfig()
    _current_age = developmental_age
    
    # Create components
    _salience_detector = SalienceDetector(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _focus_controller = FocusController(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _attention_network = AttentionNetwork(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _initialized = True
    
    logger.info(f"Attention module initialized with age {developmental_age}")

def assess_salience(
    input_id: UUID,
    input_type: str,
    features: Dict[str, float],
    context_data: Optional[Dict[str, Any]] = None
) -> SalienceAssessment:
    """
    Assess the salience of an input.
    
    Args:
        input_id: ID of the input
        input_type: Type of the input (e.g., 'sensory_input', 'pattern')
        features: Dictionary of features that may affect salience
        context_data: Optional context that may affect salience assessment
        
    Returns:
        Salience assessment for the input
    """
    _ensure_initialized()
    return _salience_detector.assess_salience(input_id, input_type, features, context_data)

def focus_attention(
    target_type: str,
    target_id: UUID,
    description: str,
    priority: float = FocusLevel.MEDIUM,
    mode: AttentionMode = AttentionMode.FOCUSED,
    force_focus: bool = False
) -> Optional[AttentionFocus]:
    """
    Focus attention on a specific target.
    
    Args:
        target_type: Type of the target (e.g., 'sensory_input', 'pattern')
        target_id: ID of the target
        description: Description of the target
        priority: Priority level for this focus request
        mode: Attention mode to use
        force_focus: Whether to force focus even if below threshold
        
    Returns:
        The new attention focus if successful, None otherwise
    """
    _ensure_initialized()
    return _focus_controller.focus_attention(
        target_type, target_id, description, priority, mode, force_focus
    )

def create_attention_target(
    target_type: str,
    target_id: UUID,
    description: str,
    priority: float = FocusLevel.MEDIUM,
    relevance_score: float = 0.5
) -> AttentionTarget:
    """
    Create an attention target.
    
    Args:
        target_type: Type of the target (e.g., 'sensory_input', 'pattern')
        target_id: ID of the target
        description: Description of the target
        priority: Priority level for this target
        relevance_score: Relevance score for this target
        
    Returns:
        An AttentionTarget object
    """
    return AttentionTarget(
        target_type=target_type,
        target_id=target_id,
        description=description,
        priority=priority,
        relevance_score=relevance_score
    )

def get_current_focus() -> Optional[AttentionFocus]:
    """
    Get the current focus of attention.
    
    Returns:
        Current attention focus or None if no focus
    """
    _ensure_initialized()
    return _focus_controller.get_current_focus()

def get_focus_history(count: int = 5) -> List[AttentionFocus]:
    """
    Get recent attention focuses.
    
    Args:
        count: Maximum number of focuses to return
        
    Returns:
        List of recent attention focuses
    """
    _ensure_initialized()
    return _focus_controller.get_focus_history(count)

def get_active_targets() -> Dict[UUID, AttentionTarget]:
    """
    Get currently active attention targets.
    
    Returns:
        Dictionary of active attention targets by ID
    """
    _ensure_initialized()
    return _focus_controller.get_active_targets()

def get_state() -> AttentionState:
    """
    Get the current state of the attention module.
    
    Returns:
        Current attention state
    """
    _ensure_initialized()
    
    current_focus = get_current_focus()
    focus_history = get_focus_history()
    active_targets = get_active_targets()
    
    return AttentionState(
        current_mode=current_focus.mode if current_focus else _current_config.default_mode,
        current_focus=current_focus,
        focus_history=focus_history,
        active_targets=active_targets,
        salience_threshold=_current_config.base_salience_threshold,
        distractibility=max(0.1, 1.0 - min(1.0, _current_age * 2)),  # Distractibility decreases with age
        attention_span_ms=int(_current_config.base_attention_span_ms * (1 + _current_age)),
        developmental_age=_current_age
    )

def update_age(new_age: float) -> None:
    """
    Update the developmental age of the attention module.
    
    Args:
        new_age: The new developmental age
    """
    global _current_age
    
    _ensure_initialized()
    _current_age = new_age
    
    # Update components
    _salience_detector.update_developmental_age(new_age)
    _focus_controller.update_developmental_age(new_age)
    _attention_network.update_developmental_age(new_age)
    
    logger.info(f"Attention module age updated to {new_age}")

def _ensure_initialized() -> None:
    """
    Ensure the attention module is initialized.
    
    Raises:
        RuntimeError: If the module is not initialized
    """
    if not _initialized:
        raise RuntimeError(
            "Attention module not initialized. Call initialize() first."
        )

# Export public API
__all__ = [
    # Classes
    'FocusController',
    'SalienceDetector',
    'AttentionNetwork',
    
    # Models
    'AttentionMode',
    'FocusLevel',
    'AttentionTarget',
    'SalienceFeature',
    'AttentionFocus',
    'SalienceAssessment',
    'AttentionEvent',
    'AttentionConfig',
    'AttentionState',
    
    # Functions
    'initialize',
    'assess_salience',
    'focus_attention',
    'create_attention_target',
    'get_current_focus',
    'get_focus_history',
    'get_active_targets',
    'get_state',
    'update_age'
]
