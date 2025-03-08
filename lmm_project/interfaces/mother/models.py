from typing import Dict, List, Any, Optional
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
        """Convert personality to a prompt section"""
        prompt = "Personality traits to embody:\n"
        
        for trait, value in self.traits.items():
            prompt += f"- {trait.capitalize()}: {value*10}/10\n"
            
        prompt += f"\nTeaching style: {self.teaching_style.value.capitalize()}\n"
        
        return prompt

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
