from enum import Enum
from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator

# Personality traits models
class PersonalityDimension(str, Enum):
    """Core personality dimensions for the Mother figure"""
    WARMTH = "warmth"                 # Cold vs. Warm
    PATIENCE = "patience"             # Impatient vs. Patient
    EXPRESSIVENESS = "expressiveness" # Reserved vs. Expressive
    STRUCTURE = "structure"           # Flexible vs. Structured
    CHALLENGE = "challenge"           # Nurturing vs. Challenging
    
class PersonalityTrait(BaseModel):
    """A specific personality trait with a value ranging from -1.0 to 1.0"""
    dimension: PersonalityDimension
    value: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    model_config = {"extra": "forbid"}
    
class PersonalityProfile(BaseModel):
    """Complete personality profile for the Mother"""
    traits: List[PersonalityTrait] = Field(default_factory=list)
    preset_name: Optional[str] = None
    
    model_config = {"extra": "forbid"}
    
    @field_validator('traits')
    @classmethod
    def validate_traits(cls, v: List[PersonalityTrait]) -> List[PersonalityTrait]:
        # Ensure we have a value for each dimension
        dimensions = {trait.dimension for trait in v}
        
        # Add missing dimensions with neutral values
        all_dimensions = set(PersonalityDimension)
        for dimension in all_dimensions - dimensions:
            v.append(PersonalityTrait(dimension=dimension, value=0.0))
            
        return v

# Teaching strategy models
class TeachingMethod(str, Enum):
    """Teaching methods available to the Mother"""
    DIRECT_INSTRUCTION = "direct_instruction"
    SCAFFOLDING = "scaffolding"
    SOCRATIC_QUESTIONING = "socratic_questioning"
    REPETITION = "repetition"
    REINFORCEMENT = "reinforcement"
    EXPERIENTIAL = "experiential"
    DISCOVERY = "discovery"
    ANALOGICAL = "analogical"

class TeachingStrategy(BaseModel):
    """Configuration for a specific teaching strategy"""
    method: TeachingMethod
    priority: float = Field(default=1.0, ge=0.0, le=10.0) 
    applicability: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}
    
class TeachingProfile(BaseModel):
    """Complete teaching profile for the Mother"""
    strategies: List[TeachingStrategy] = Field(default_factory=list)
    adaptive: bool = True
    preset_name: Optional[str] = None
    
    model_config = {"extra": "forbid"}

# Interaction models
class EmotionalTone(str, Enum):
    """Emotional tones for Mother's communication"""
    SOOTHING = "soothing"
    ENCOURAGING = "encouraging"
    PLAYFUL = "playful"
    CURIOUS = "curious"
    SERIOUS = "serious"
    EXCITED = "excited"
    CONCERNED = "concerned"
    FIRM = "firm"

class InteractionStyle(str, Enum):
    """Interaction styles for Mother's communication"""
    CONVERSATIONAL = "conversational"
    INSTRUCTIONAL = "instructional"
    QUESTIONING = "questioning"
    STORYTELLING = "storytelling"
    RESPONSIVE = "responsive"
    CORRECTIVE = "corrective"
    REFLECTIVE = "reflective"
    
class InteractionPattern(BaseModel):
    """Configuration for a specific interaction pattern"""
    style: InteractionStyle
    primary_tone: EmotionalTone
    secondary_tone: Optional[EmotionalTone] = None
    complexity_level: float = Field(default=1.0, ge=0.1, le=10.0)
    
    model_config = {"extra": "forbid"}

# Mother instruction and response models
class MotherInput(BaseModel):
    """Input from the child to the Mother"""
    content: str
    age: float
    context: Optional[Dict[str, Union[str, float, bool]]] = None
    
    model_config = {"extra": "forbid"}
    
class MotherResponse(BaseModel):
    """Response from the Mother to the child"""
    content: str
    tone: EmotionalTone
    teaching_method: Optional[TeachingMethod] = None
    voice_settings: Optional[Dict[str, Union[str, float]]] = None
    
    model_config = {"extra": "forbid"}
