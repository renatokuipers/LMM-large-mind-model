"""
Cognitive Modules Package

This package provides all the specialized cognitive modules for the LMM system.
Each module handles a specific aspect of cognitive function.
"""

from typing import Dict, Type, Any, TYPE_CHECKING

# Use conditional import to avoid circular references
if TYPE_CHECKING:
    from .base_module import BaseModule

def get_module_classes() -> Dict[str, Any]:
    """
    Get all available module classes
    
    Returns:
        Dictionary mapping module types to module classes
    """
    # Import here to avoid circular imports
    from lmm_project.modules.perception import get_module as get_perception_module
    from lmm_project.modules.attention import get_module as get_attention_module
    from lmm_project.modules.memory import get_module as get_memory_module
    from lmm_project.modules.emotion import get_module as get_emotion_module
    from lmm_project.modules.learning import get_module as get_learning_module
    # Add other module imports as they are implemented
    
    return {
        "perception": get_perception_module,
        "attention": get_attention_module,
        "memory": get_memory_module,
        "emotion": get_emotion_module,
        "learning": get_learning_module,
        # Add other modules here as they are implemented
    }

__all__ = [
    'BaseModule'
] 