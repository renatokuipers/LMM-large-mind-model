"""
Mother Interface Module

This module provides the "Mother" caregiver interface that nurtures,
teaches, and guides the developing mind through its formative stages.
The Mother acts as both a teacher and emotional support system, adapting
her interaction style, teaching methods, and language complexity based
on the developmental stage of the mind.

Components:
    - MotherLLM: Core class for generating nurturing responses
    - Personality: Manages the Mother's personality traits
    - TeachingStrategies: Handles educational approaches
    - InteractionPatterns: Controls communication styles
"""

from typing import Dict, Optional, Any, Union

from .models import (
    PersonalityDimension,
    PersonalityTrait,
    PersonalityProfile,
    TeachingMethod,
    TeachingStrategy,
    TeachingProfile,
    EmotionalTone,
    InteractionStyle,
    InteractionPattern,
    MotherInput,
    MotherResponse
)

from .personality import Personality
from .teaching_strategies import TeachingStrategies
from .interaction_patterns import InteractionPatterns
from .mother_llm import MotherLLM

# Singleton instance for easy access
_mother_instance = None

def get_mother(
    personality_preset: Optional[str] = None,
    teaching_preset: Optional[str] = None,
    use_tts: Optional[bool] = None,
    tts_voice: Optional[str] = None
) -> MotherLLM:
    """
    Get or create the singleton Mother LLM instance.
    
    Args:
        personality_preset: Name of personality preset to use
        teaching_preset: Name of teaching preset to use
        use_tts: Whether to use text-to-speech
        tts_voice: Voice to use for TTS
        
    Returns:
        MotherLLM instance
    """
    global _mother_instance
    
    if _mother_instance is None:
        _mother_instance = MotherLLM(
            personality_preset=personality_preset,
            teaching_preset=teaching_preset,
            use_tts=use_tts,
            tts_voice=tts_voice
        )
        
    return _mother_instance

def create_mother_input(
    content: str,
    age: float,
    context: Optional[Dict[str, Any]] = None
) -> MotherInput:
    """
    Create input for the Mother from the child.
    
    Args:
        content: The message content from the child
        age: Current developmental age
        context: Additional context information
        
    Returns:
        MotherInput object ready to be passed to MotherLLM.respond()
    """
    return MotherInput(
        content=content,
        age=age,
        context=context or {}
    )

__all__ = [
    # Classes
    'MotherLLM',
    'Personality',
    'TeachingStrategies',
    'InteractionPatterns',
    
    # Models
    'PersonalityDimension',
    'PersonalityTrait',
    'PersonalityProfile',
    'TeachingMethod',
    'TeachingStrategy',
    'TeachingProfile',
    'EmotionalTone',
    'InteractionStyle',
    'InteractionPattern',
    'MotherInput',
    'MotherResponse',
    
    # Factory functions
    'get_mother',
    'create_mother_input'
]
