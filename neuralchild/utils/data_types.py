"""
Data Types Module

This module defines all the Pydantic models used throughout the NeuralChild framework.
These models ensure type safety and data validation throughout the system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# Enums for developmental stages and other categorical values

class DevelopmentalStage(str, Enum):
    """Developmental stages of the child's mind."""
    INFANCY = "infancy"  # 0-2 years
    EARLY_CHILDHOOD = "early_childhood"  # 2-5 years
    MIDDLE_CHILDHOOD = "middle_childhood"  # 5-10 years
    ADOLESCENCE = "adolescence"  # 10-18 years
    EARLY_ADULTHOOD = "early_adulthood"  # 18+ years


class DevelopmentalSubstage(str, Enum):
    """Sub-stages within each developmental stage for more nuanced progression."""
    # Infancy substages
    EARLY_INFANCY = "early_infancy"            # 0-8 months
    MIDDLE_INFANCY = "middle_infancy"          # 8-16 months
    LATE_INFANCY = "late_infancy"              # 16-24 months
    
    # Early childhood substages
    EARLY_TODDLER = "early_toddler"            # 2-3 years
    LATE_TODDLER = "late_toddler"              # 3-4 years
    PRESCHOOL = "preschool"                    # 4-5 years
    
    # Middle childhood substages
    EARLY_ELEMENTARY = "early_elementary"      # 5-7 years
    MIDDLE_ELEMENTARY = "middle_elementary"    # 7-9 years
    LATE_ELEMENTARY = "late_elementary"        # 9-10 years
    
    # Adolescence substages
    EARLY_ADOLESCENCE = "early_adolescence"    # 10-13 years
    MIDDLE_ADOLESCENCE = "middle_adolescence"  # 13-16 years
    LATE_ADOLESCENCE = "late_adolescence"      # 16-18 years
    
    # Early adulthood substages
    EMERGING_ADULT = "emerging_adult"          # 18-21 years
    YOUNG_ADULT = "young_adult"                # 21-25 years
    ESTABLISHED_ADULT = "established_adult"    # 25+ years


# Mapping between main stages and substages
STAGE_TO_SUBSTAGES = {
    DevelopmentalStage.INFANCY: [
        DevelopmentalSubstage.EARLY_INFANCY,
        DevelopmentalSubstage.MIDDLE_INFANCY,
        DevelopmentalSubstage.LATE_INFANCY
    ],
    DevelopmentalStage.EARLY_CHILDHOOD: [
        DevelopmentalSubstage.EARLY_TODDLER,
        DevelopmentalSubstage.LATE_TODDLER,
        DevelopmentalSubstage.PRESCHOOL
    ],
    DevelopmentalStage.MIDDLE_CHILDHOOD: [
        DevelopmentalSubstage.EARLY_ELEMENTARY,
        DevelopmentalSubstage.MIDDLE_ELEMENTARY,
        DevelopmentalSubstage.LATE_ELEMENTARY
    ],
    DevelopmentalStage.ADOLESCENCE: [
        DevelopmentalSubstage.EARLY_ADOLESCENCE,
        DevelopmentalSubstage.MIDDLE_ADOLESCENCE,
        DevelopmentalSubstage.LATE_ADOLESCENCE
    ],
    DevelopmentalStage.EARLY_ADULTHOOD: [
        DevelopmentalSubstage.EMERGING_ADULT,
        DevelopmentalSubstage.YOUNG_ADULT,
        DevelopmentalSubstage.ESTABLISHED_ADULT
    ]
}

# Mapping between age in months and substages
AGE_TO_SUBSTAGE = {
    # Infancy (0-24 months)
    range(0, 8): DevelopmentalSubstage.EARLY_INFANCY,
    range(8, 16): DevelopmentalSubstage.MIDDLE_INFANCY,
    range(16, 24): DevelopmentalSubstage.LATE_INFANCY,
    
    # Early childhood (2-5 years, 24-60 months)
    range(24, 36): DevelopmentalSubstage.EARLY_TODDLER,
    range(36, 48): DevelopmentalSubstage.LATE_TODDLER,
    range(48, 60): DevelopmentalSubstage.PRESCHOOL,
    
    # Middle childhood (5-10 years, 60-120 months)
    range(60, 84): DevelopmentalSubstage.EARLY_ELEMENTARY,
    range(84, 108): DevelopmentalSubstage.MIDDLE_ELEMENTARY,
    range(108, 120): DevelopmentalSubstage.LATE_ELEMENTARY,
    
    # Adolescence (10-18 years, 120-216 months)
    range(120, 156): DevelopmentalSubstage.EARLY_ADOLESCENCE,
    range(156, 192): DevelopmentalSubstage.MIDDLE_ADOLESCENCE,
    range(192, 216): DevelopmentalSubstage.LATE_ADOLESCENCE,
    
    # Early adulthood (18+ years, 216+ months)
    range(216, 252): DevelopmentalSubstage.EMERGING_ADULT,
    range(252, 300): DevelopmentalSubstage.YOUNG_ADULT,
    range(300, 600): DevelopmentalSubstage.ESTABLISHED_ADULT
}


def get_substage_from_age(age_months: int) -> DevelopmentalSubstage:
    """
    Get the developmental substage based on age in months.
    
    Args:
        age_months: Age in months
        
    Returns:
        The corresponding developmental substage
    """
    for age_range, substage in AGE_TO_SUBSTAGE.items():
        if age_months in age_range:
            return substage
    
    # Default to established adult for very old ages
    return DevelopmentalSubstage.ESTABLISHED_ADULT


def get_stage_from_substage(substage: DevelopmentalSubstage) -> DevelopmentalStage:
    """
    Get the main developmental stage from a substage.
    
    Args:
        substage: The developmental substage
        
    Returns:
        The corresponding main developmental stage
    """
    for stage, substages in STAGE_TO_SUBSTAGES.items():
        if substage in substages:
            return stage
    
    # Default to early adulthood if not found
    return DevelopmentalStage.EARLY_ADULTHOOD


class EmotionType(str, Enum):
    """Basic emotion types."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    # Complex emotions develop later
    PRIDE = "pride"
    SHAME = "shame"
    GUILT = "guilt"
    ENVY = "envy"
    JEALOUSY = "jealousy"
    LOVE = "love"


class MemoryType(str, Enum):
    """Types of memory systems."""
    EPISODIC = "episodic"  # Memories of specific events
    SEMANTIC = "semantic"  # General knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Short-term processing


class MotherPersonality(str, Enum):
    """Mother personality types that influence interaction."""
    NURTURING = "nurturing"  # Warm, supportive, patient
    AUTHORITARIAN = "authoritarian"  # Strict, rule-based
    PERMISSIVE = "permissive"  # Indulgent, few rules
    NEGLECTFUL = "neglectful"  # Uninvolved, distant
    BALANCED = "balanced"  # Appropriate mix of support and boundaries


# Basic Models

class Emotion(BaseModel):
    """Model representing an emotional state."""
    type: EmotionType
    intensity: float = Field(..., ge=0.0, le=1.0)
    cause: Optional[str] = None
    
    model_config = {"extra": "forbid"}


class Memory(BaseModel):
    """Base model for all memory types."""
    id: str
    type: MemoryType
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    
    model_config = {"extra": "forbid"}
    
    @field_validator('strength')
    @classmethod
    def validate_strength(cls, v: float) -> float:
        """Ensure strength is within bounds."""
        return max(0.0, min(1.0, v))


class EpisodicMemory(Memory):
    """Model for episodic memories - specific events and experiences."""
    event_description: str
    emotional_valence: float = Field(..., ge=-1.0, le=1.0)
    associated_emotions: List[Emotion] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}


class SemanticMemory(Memory):
    """Model for semantic memories - facts and knowledge."""
    concept: str
    definition: str
    related_concepts: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    model_config = {"extra": "forbid"}


class Word(BaseModel):
    """Model representing a word in the child's vocabulary."""
    word: str
    first_encountered: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    understanding_level: float = Field(default=0.1, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    associations: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}


class MotherResponse(BaseModel):
    """Model representing a response from the Mother component."""
    text: str
    emotional_state: List[Emotion] = Field(default_factory=list)
    teaching_elements: Dict[str, Any] = Field(default_factory=dict)
    non_verbal_cues: Optional[str] = None
    
    model_config = {"extra": "forbid"}


class ChildResponse(BaseModel):
    """Model representing a response from the Child component."""
    text: Optional[str] = None
    vocalization: Optional[str] = None  # For pre-linguistic stages
    emotional_state: List[Emotion] = Field(default_factory=list)
    attention_focus: Optional[str] = None
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode='after')
    def validate_response_type(self) -> 'ChildResponse':
        """Ensure at least one response type is provided."""
        if self.text is None and self.vocalization is None:
            raise ValueError("Either text or vocalization must be provided")
        return self


class DevelopmentalMetrics(BaseModel):
    """Model tracking the child's developmental progress."""
    vocabulary_size: int = Field(default=0, ge=0)
    grammatical_complexity: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_regulation: float = Field(default=0.0, ge=0.0, le=1.0)
    social_awareness: float = Field(default=0.0, ge=0.0, le=1.0)
    object_permanence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstract_thinking: float = Field(default=0.0, ge=0.0, le=1.0)
    self_awareness: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Tracking history of each metric for visualization
    history: Dict[str, List[float]] = Field(default_factory=lambda: {
        "vocabulary_size": [],
        "grammatical_complexity": [],
        "emotional_regulation": [],
        "social_awareness": [],
        "object_permanence": [],
        "abstract_thinking": [],
        "self_awareness": []
    })
    
    model_config = {"extra": "forbid"}


class ComponentState(BaseModel):
    """Base model for the state of a neural component."""
    component_id: str
    component_type: str
    activation_level: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    model_config = {"extra": "forbid"}


class StageTransition(BaseModel):
    """Model representing a transition between developmental stages."""
    current_stage: DevelopmentalStage
    next_stage: DevelopmentalStage
    current_substage: DevelopmentalSubstage
    next_substage: DevelopmentalSubstage
    transition_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}


class ChildState(BaseModel):
    """Comprehensive state of the child's mind."""
    developmental_stage: DevelopmentalStage = Field(default=DevelopmentalStage.INFANCY)
    developmental_substage: DevelopmentalSubstage = Field(default=DevelopmentalSubstage.EARLY_INFANCY)
    simulated_age_months: int = Field(default=0, ge=0)
    vocabulary: Dict[str, Word] = Field(default_factory=dict)
    episodic_memories: Dict[str, EpisodicMemory] = Field(default_factory=dict)
    semantic_memories: Dict[str, SemanticMemory] = Field(default_factory=dict)
    current_emotional_state: List[Emotion] = Field(default_factory=list)
    metrics: DevelopmentalMetrics = Field(default_factory=DevelopmentalMetrics)
    component_states: Dict[str, ComponentState] = Field(default_factory=dict)
    stage_transition: Optional[StageTransition] = None
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode='after')
    def validate_substage_consistency(self) -> 'ChildState':
        """Ensure the substage is consistent with the main stage."""
        # Calculate expected substage based on age
        expected_substage = get_substage_from_age(self.simulated_age_months)
        
        # Get expected main stage from substage
        expected_stage = get_stage_from_substage(expected_substage)
        
        # Ensure substage consistency
        if self.developmental_substage not in STAGE_TO_SUBSTAGES[self.developmental_stage]:
            # Use the first substage of the current main stage
            self.developmental_substage = STAGE_TO_SUBSTAGES[self.developmental_stage][0]
        
        return self


class InteractionLog(BaseModel):
    """Log of an interaction between the Mother and Child."""
    timestamp: datetime = Field(default_factory=datetime.now)
    mother_response: MotherResponse
    child_response: ChildResponse
    developmental_effect: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}


class DevelopmentConfig(BaseModel):
    """Configuration for the development simulation."""
    time_acceleration_factor: int = Field(default=100, ge=1)
    random_seed: Optional[int] = None
    mother_personality: MotherPersonality = Field(default=MotherPersonality.BALANCED)
    start_age_months: int = Field(default=0, ge=0)
    enable_random_factors: bool = Field(default=True)
    
    model_config = {"extra": "forbid"}


class SystemState(BaseModel):
    """Overall system state including all components."""
    child_state: ChildState
    development_config: DevelopmentConfig
    interaction_history: List[InteractionLog] = Field(default_factory=list)
    system_start_time: datetime = Field(default_factory=datetime.now)
    last_save_time: Optional[datetime] = None
    
    model_config = {"extra": "forbid"} 