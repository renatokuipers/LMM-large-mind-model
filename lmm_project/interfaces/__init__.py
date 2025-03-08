# Interfaces module 

from .mother.mother_llm import MotherLLM
from .mother.models import (
    TeachingStyle, PersonalityTrait, MotherPersonality,
    InteractionPattern, ConversationEntry, TeachingStrategy
)

__all__ = [
    'MotherLLM',
    'TeachingStyle',
    'PersonalityTrait',
    'MotherPersonality',
    'InteractionPattern',
    'ConversationEntry',
    'TeachingStrategy'
] 
