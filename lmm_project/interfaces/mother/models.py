from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class TeachingStyle(str, Enum):
    """Teaching styles for the Mother LLM"""
    SOCRATIC = "socratic"
    DIRECT = "direct"
    MONTESSORI = "montessori"
    CONSTRUCTIVIST = "constructivist"
    SCAFFOLDING = "scaffolding"

class EmotionalValence(str, Enum):
    """Types of emotional valence for responses"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    FIRM = "firm"

class PersonalityProfile(str, Enum):
    """Pre-defined personality profiles for the Mother LLM"""
    NURTURING = "nurturing"
    ACADEMIC = "academic"
    BALANCED = "balanced"
    PLAYFUL = "playful"
    SOCRATIC = "socratic"
    MINDFUL = "mindful"

class InteractionType(str, Enum):
    """Types of interaction patterns"""
    REPETITION = "repetition"
    MIRRORING = "mirroring"
    TURN_TAKING = "turn_taking"
    ELABORATION = "elaboration"
    QUESTIONING = "questioning"
    STORYTELLING = "storytelling"
    PLAYFUL = "playful"
    INSTRUCTIONAL = "instructional"
    CONVERSATIONAL = "conversational"
    SOCRATIC = "socratic"
    PROBLEM_SOLVING = "problem_solving"
    EMOTIONAL_SUPPORT = "emotional_support"

class InteractionComplexity(str, Enum):
    """Complexity levels for interactions"""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class LearningGoalCategory(str, Enum):
    """Categories of learning goals for curriculum development"""
    PATTERN_RECOGNITION = "pattern_recognition"
    LANGUAGE_ACQUISITION = "language_acquisition"
    OBJECT_PERMANENCE = "object_permanence"
    EMOTIONAL_UNDERSTANDING = "emotional_understanding"
    SOCIAL_AWARENESS = "social_awareness"
    CAUSAL_REASONING = "causal_reasoning"
    ABSTRACT_THINKING = "abstract_thinking"
    IDENTITY_FORMATION = "identity_formation"
    CREATIVE_THINKING = "creative_thinking"
    METACOGNITION = "metacognition"

class LearningMode(str, Enum):
    """Different modes of learning interaction"""
    EXPLORATION = "exploration"
    INSTRUCTION = "instruction"
    PRACTICE = "practice"
    REFLECTION = "reflection"
    ASSESSMENT = "assessment"
    PLAY = "play"
    CONVERSATION = "conversation"

class ComprehensionLevel(str, Enum):
    """Levels of comprehension for a concept or topic"""
    NONE = "none"
    MINIMAL = "minimal"
    PARTIAL = "partial"
    FUNCTIONAL = "functional"
    SOLID = "solid"
    MASTERY = "mastery"

class PersonalityTrait(BaseModel):
    """A personality trait with a value between 0 and 1"""
    name: str
    value: float = Field(default=0.5, ge=0.0, le=1.0)
    description: Optional[str] = None

class MotherPersonality(BaseModel):
    """Personality configuration for the Mother LLM"""
    traits: Dict[str, float] = Field(default_factory=lambda: {
        "nurturing": 0.8,
        "patient": 0.9,
        "encouraging": 0.8,
        "structured": 0.7,
        "responsive": 0.9
    })
    teaching_style: TeachingStyle = Field(default=TeachingStyle.SOCRATIC)
    voice_characteristics: Dict[str, Any] = Field(default_factory=dict)
    
    def to_prompt_section(self) -> str:
        """Convert personality to a prompt section for the LLM"""
        traits_str = ", ".join([f"{k} ({v:.1f})" for k, v in self.traits.items()])
        return f"""
        Your personality is defined by these traits: {traits_str}
        Your teaching style is {self.teaching_style.value}
        Please embody these traits in your responses.
        """

class InteractionPattern(BaseModel):
    """A pattern of interaction for the Mother LLM"""
    name: str
    description: str
    prompt_template: str
    suitable_stages: List[str] = Field(default_factory=list)
    
class ConversationEntry(BaseModel):
    """An entry in the conversation history"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TeachingStrategy(BaseModel):
    """A teaching strategy for the Mother LLM"""
    name: str
    description: str
    suitable_stages: List[str]
    prompt_guidance: str
    examples: List[str] = Field(default_factory=list)
