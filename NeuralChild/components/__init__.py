"""
Neural components for the NeuralChild project.

This package contains specialized neural components that represent
psychological functions in the Neural Child's mind.
"""

from .base import NeuralComponent, ComponentState
from .emotion import EmotionComponent
from .language import LanguageComponent
from .memory import MemoryComponent
from .consciousness import ConsciousnessComponent

# Dictionary mapping component names to their classes for easy instantiation
COMPONENT_REGISTRY = {
    "Emotion": EmotionComponent,
    "Language": LanguageComponent,
    "Memory": MemoryComponent,
    "Consciousness": ConsciousnessComponent
}