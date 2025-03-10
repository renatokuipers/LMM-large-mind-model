"""
Cognitive Modules Package

This package provides all the specialized cognitive modules for the LMM system.
Each module handles a specific aspect of cognitive function.

The modular design allows each cognitive function to develop independently
while communicating through the event bus system. This mimics the specialized
yet interconnected nature of brain regions in human cognition.

Each module follows a developmental trajectory from simple capabilities in early
stages to sophisticated processing in later stages, in line with the LMM's
psychological development approach.
"""

from typing import Dict, Type, Any, TYPE_CHECKING, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Use conditional import to avoid circular references
if TYPE_CHECKING:
    from .base_module import BaseModule
    from lmm_project.core.event_bus import EventBus

def get_module_classes() -> Dict[str, Any]:
    """
    Get all available module classes
    
    This function returns factory functions for creating instances of each
    cognitive module. These factory functions follow a consistent interface
    that takes module_id, event_bus, and development_level parameters.
    
    Returns:
        Dictionary mapping module types to their factory functions
    """
    # Import here to avoid circular imports
    from lmm_project.modules.perception import get_module as get_perception_module
    from lmm_project.modules.attention import get_module as get_attention_module
    from lmm_project.modules.memory import get_module as get_memory_module
    from lmm_project.modules.emotion import get_module as get_emotion_module
    from lmm_project.modules.language import get_module as get_language_module
    from lmm_project.modules.consciousness import get_module as get_consciousness_module
    from lmm_project.modules.executive import get_module as get_executive_module
    from lmm_project.modules.social import get_module as get_social_module
    from lmm_project.modules.motivation import get_module as get_motivation_module
    from lmm_project.modules.temporal import get_module as get_temporal_module
    from lmm_project.modules.learning import get_module as get_learning_module
    from lmm_project.modules.identity import get_module as get_identity_module
    from lmm_project.modules.belief import get_module as get_belief_module
    
    # For modules with different naming patterns
    from lmm_project.modules.creativity import CreativityModule
    from lmm_project.modules.self_regulation import get_self_regulation_system
    
    # Create wrapper functions for modules that don't follow the standard pattern
    def get_creativity_module(module_id: str, event_bus: Optional['EventBus'] = None, development_level: float = 0.0, **kwargs):
        return CreativityModule(module_id=module_id, event_bus=event_bus, **kwargs)
    
    def get_self_regulation_module(module_id: str, event_bus: Optional['EventBus'] = None, development_level: float = 0.0, **kwargs):
        system = get_self_regulation_system(module_id)
        if system is None:
            # Create a new system if one doesn't exist
            from lmm_project.modules.self_regulation import SelfRegulationSystem
            system = SelfRegulationSystem(
                module_id=module_id, 
                event_bus=event_bus,
                development_level=development_level,
                parameters=kwargs
            )
        return system
    
    return {
        "perception": get_perception_module,
        "attention": get_attention_module,
        "memory": get_memory_module,
        "emotion": get_emotion_module,
        "language": get_language_module,
        "consciousness": get_consciousness_module,
        "executive": get_executive_module,
        "social": get_social_module,
        "motivation": get_motivation_module,
        "temporal": get_temporal_module,
        "learning": get_learning_module,
        "identity": get_identity_module,
        "belief": get_belief_module,
        "creativity": get_creativity_module,
        "self_regulation": get_self_regulation_module
    }

def initialize_all_modules(
    event_bus: Optional['EventBus'] = None, 
    development_level: float = 0.0,
    module_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, 'BaseModule']:
    """
    Initialize all cognitive modules at once
    
    This is a convenience function that initializes all available cognitive modules
    with the same event bus and initial development level, returning them in a
    dictionary for easy access.
    
    Args:
        event_bus: Event bus for module communication
        development_level: Initial developmental level for all modules
        module_params: Optional dictionary of module-specific parameters
                      Format: {"module_name": {"param1": value1, ...}}
        
    Returns:
        Dictionary mapping module types to initialized module instances
    """
    modules = {}
    module_classes = get_module_classes()
    
    if module_params is None:
        module_params = {}
    
    for module_type, get_module_func in module_classes.items():
        try:
            # Get module-specific parameters if available
            params = module_params.get(module_type, {})
            
            # Initialize the module
            modules[module_type] = get_module_func(
                module_id=module_type,
                event_bus=event_bus,
                development_level=development_level,
                **params
            )
            logger.info(f"Initialized {module_type} module")
        except Exception as e:
            logger.error(f"Failed to initialize {module_type} module: {str(e)}")
    
    return modules

def get_development_status(modules: Dict[str, 'BaseModule']) -> Dict[str, Dict[str, Any]]:
    """
    Get the development status of all modules
    
    Args:
        modules: Dictionary of initialized modules
        
    Returns:
        Dictionary mapping module types to their development information
    """
    result = {}
    for module_type, module in modules.items():
        try:
            result[module_type] = {
                "development_level": module.development_level,
                "milestone": _get_milestone_for_module(module),
                "module_id": module.module_id,
                "module_type": module.module_type
            }
        except Exception as e:
            logger.error(f"Error getting development status for {module_type}: {str(e)}")
            result[module_type] = {"error": str(e)}
    
    return result

def _get_milestone_for_module(module: 'BaseModule') -> str:
    """Get the current milestone description for a module based on its development level"""
    for level, description in sorted(module.development_milestones.items(), reverse=True):
        if module.development_level >= level:
            return description
    return "Unknown"

def save_all_modules_state(modules: Dict[str, 'BaseModule'], state_dir: str) -> Dict[str, str]:
    """
    Save the state of all modules to disk
    
    Args:
        modules: Dictionary of initialized modules
        state_dir: Directory to save state files
        
    Returns:
        Dictionary mapping module types to saved state file paths
    """
    saved_paths = {}
    for module_type, module in modules.items():
        try:
            path = module.save_state(state_dir)
            saved_paths[module_type] = path
            logger.info(f"Saved state for {module_type} module to {path}")
        except Exception as e:
            logger.error(f"Failed to save state for {module_type} module: {str(e)}")
            saved_paths[module_type] = f"ERROR: {str(e)}"
    
    return saved_paths

# Import BaseModule for convenience
from .base_module import BaseModule

__all__ = [
    'BaseModule',
    'get_module_classes',
    'initialize_all_modules',
    'get_development_status',
    'save_all_modules_state'
] 