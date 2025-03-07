"""
Core mind modules for the Large Mind Model (LMM) project.

This package contains the fundamental cognitive components that form the digital mind:
- Memory: Semantic and episodic memory storage and retrieval
- Consciousness: Self-awareness and reflection mechanisms
- Language: Language acquisition and comprehension
- Emotion: Emotional intelligence and processing
- Social: Social cognition and moral reasoning
- Thought: Autonomous thought generation and reasoning
- Imagination: Creative thinking and dreaming
"""

from lmm.core.memory import MemoryModule
from lmm.core.consciousness import ConsciousnessModule
from lmm.core.language import LanguageModule
from lmm.core.emotion import EmotionModule
from lmm.core.social import SocialModule  
from lmm.core.thought import ThoughtModule
from lmm.core.imagination import ImaginationModule

__all__ = [
    'MemoryModule',
    'ConsciousnessModule',
    'LanguageModule',
    'EmotionModule',
    'SocialModule',
    'ThoughtModule',
    'ImaginationModule',
] 