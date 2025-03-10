"""
Mother Interface Module

This module provides the Mother LLM interface which acts as a nurturing caregiver
and teacher for the developing mind. The Mother LLM provides structured interactions,
educational guidance, and emotional support appropriate for the mind's developmental stage.

Key components include:
- Personality and teaching styles
- Interaction patterns for different developmental stages
- Teaching strategies and curriculum design
- Emotional response generation and adaptation

The Mother interface serves as the primary external interaction point for the developing mind.
"""

# Models
from lmm_project.interfaces.mother.models import (
    TeachingStyle,
    EmotionalValence,
    PersonalityProfile,
    InteractionType,
    InteractionComplexity,
    LearningGoalCategory,
    LearningMode,
    ComprehensionLevel,
    PersonalityTrait,
    MotherPersonality,
    InteractionPattern,
    ConversationEntry,
    TeachingStrategy
)

# Implementation classes
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.interfaces.mother.personality import PersonalityManager
from lmm_project.interfaces.mother.teaching_strategies import TeachingStrategyManager
from lmm_project.interfaces.mother.interaction_patterns import InteractionPatternManager

__all__ = [
    # Main classes
    'MotherLLM',
    'PersonalityManager',
    'TeachingStrategyManager',
    'InteractionPatternManager',
    
    # Enums
    'TeachingStyle',
    'EmotionalValence',
    'PersonalityProfile',
    'InteractionType',
    'InteractionComplexity',
    'LearningGoalCategory',
    'LearningMode',
    'ComprehensionLevel',
    
    # Models
    'PersonalityTrait',
    'MotherPersonality',
    'InteractionPattern',
    'ConversationEntry',
    'TeachingStrategy'
] 
