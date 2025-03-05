"""
Configuration module for the NeuralChild project.

This module defines the system-wide settings including:
- Time acceleration settings
- Persistence options
- Mother personality traits
- Developmental stage parameters
- Dashboard configuration
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class DevelopmentalStage(str, Enum):
    """Enum representing developmental stages a child goes through."""
    PRENATAL = "prenatal"
    INFANCY = "infancy"
    EARLY_CHILDHOOD = "early_childhood"
    MIDDLE_CHILDHOOD = "middle_childhood"
    ADOLESCENCE = "adolescence"
    EARLY_ADULTHOOD = "early_adulthood"
    MID_ADULTHOOD = "mid_adulthood"


class TimeConfig(BaseModel):
    """Configuration for time-related settings."""
    acceleration_factor: float = Field(
        default=720.0,
        description="Time acceleration factor (720x: 1 hour simulation = 30 days development)"
    )
    simulation_step_seconds: float = Field(
        default=1.0,
        description="Length of a single simulation step in seconds"
    )
    
    @validator('acceleration_factor')
    def validate_acceleration_factor(cls, v):
        if v <= 0:
            raise ValueError("Acceleration factor must be positive")
        return v


class PersistenceConfig(BaseModel):
    """Configuration for persistence-related settings."""
    autosave_enabled: bool = Field(
        default=True,
        description="Whether to automatically save the state"
    )
    autosave_interval_minutes: int = Field(
        default=60,
        description="Interval between automatic saves in minutes"
    )
    save_format: str = Field(
        default="json",
        description="Format to save data in (json or pickle)"
    )
    checkpoint_directory: str = Field(
        default="checkpoints",
        description="Directory to save checkpoints in"
    )
    
    @validator('save_format')
    def validate_save_format(cls, v):
        if v not in ["json", "pickle"]:
            raise ValueError("Save format must be either 'json' or 'pickle'")
        return v
    
    @validator('autosave_interval_minutes')
    def validate_autosave_interval(cls, v):
        if v <= 0:
            raise ValueError("Autosave interval must be positive")
        return v


class MotherPersonalityTraits(BaseModel):
    """Configuration for mother personality traits."""
    warmth: float = Field(
        default=0.7,
        description="Warmth (coldness to warmth)",
        ge=0.0,
        le=1.0
    )
    responsiveness: float = Field(
        default=0.8,
        description="Responsiveness to child's needs",
        ge=0.0,
        le=1.0
    )
    consistency: float = Field(
        default=0.9,
        description="Consistency in responses and behavior",
        ge=0.0,
        le=1.0
    )
    patience: float = Field(
        default=0.8,
        description="Patience with child's behaviors",
        ge=0.0,
        le=1.0
    )
    teaching_focus: float = Field(
        default=0.7,
        description="Focus on teaching vs. emotional support",
        ge=0.0,
        le=1.0
    )
    verbosity: float = Field(
        default=0.6,
        description="Verbosity in communication",
        ge=0.0,
        le=1.0
    )
    structure: float = Field(
        default=0.7,
        description="Structure vs. freedom provided",
        ge=0.0,
        le=1.0
    )
    
    # Custom traits can be added as a dictionary
    custom_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional custom personality traits"
    )
    
    @validator('custom_traits')
    def validate_custom_traits(cls, v):
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Custom trait '{key}' must have a value between 0.0 and 1.0")
        return v


class StageThresholds(BaseModel):
    """Thresholds for progressing to the next developmental stage."""
    language_vocabulary_size: int = Field(
        default=0,
        description="Minimum vocabulary size required"
    )
    language_grammar_complexity: float = Field(
        default=0.0,
        description="Minimum grammar complexity required",
        ge=0.0,
        le=1.0
    )
    emotional_stability: float = Field(
        default=0.0,
        description="Minimum emotional stability required",
        ge=0.0,
        le=1.0
    )
    emotional_diversity: float = Field(
        default=0.0,
        description="Minimum emotional diversity required",
        ge=0.0,
        le=1.0
    )
    cognitive_abstraction: float = Field(
        default=0.0,
        description="Minimum cognitive abstraction required",
        ge=0.0,
        le=1.0
    )
    social_understanding: float = Field(
        default=0.0,
        description="Minimum social understanding required",
        ge=0.0,
        le=1.0
    )
    minimum_age_days: int = Field(
        default=0,
        description="Minimum age in days required"
    )
    
    # Custom thresholds can be added as a dictionary
    custom_thresholds: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Additional custom thresholds"
    )


class DevelopmentConfig(BaseModel):
    """Configuration for developmental stages and parameters."""
    initial_stage: DevelopmentalStage = Field(
        default=DevelopmentalStage.PRENATAL,
        description="Starting developmental stage"
    )
    # Define thresholds for progressing from one stage to the next
    stage_thresholds: Dict[DevelopmentalStage, StageThresholds] = Field(
        default_factory=dict,
        description="Thresholds for progressing to the next stage"
    )
    
    # Learning rate multipliers for different components
    learning_rate_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "language": 1.0,
            "emotion": 1.0,
            "memory": 1.0,
            "perception": 1.0,
            "consciousness": 1.0,
            "self_concept": 1.0
        },
        description="Learning rate multipliers for different components"
    )
    
    # Random variation in development (to simulate individual differences)
    development_variation: float = Field(
        default=0.1,
        description="Random variation in development rates (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    @validator('stage_thresholds')
    def validate_stage_thresholds(cls, v):
        # Ensure all necessary stages have thresholds except for the final stage
        all_stages = list(DevelopmentalStage)
        required_stages = all_stages[:-1]  # All except the last
        
        for stage in required_stages:
            if stage not in v:
                v[stage] = StageThresholds()
        
        return v
    
    @validator('learning_rate_multipliers')
    def validate_learning_rate_multipliers(cls, v):
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"Learning rate multiplier for '{key}' must be positive")
        return v


class LLMConfig(BaseModel):
    """Configuration for the LLM used in the Mother component."""
    base_url: str = Field(
        default="http://192.168.2.12:1234",
        description="Base URL for the LLM API"
    )
    model: str = Field(
        default="qwen2.5-7b-instruct",
        description="Model name to use"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for generation",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens to generate"
    )
    system_prompt: str = Field(
        default="You are the mother of a child who is developing through different stages. "
                "Respond as a nurturing mother would, with warmth and attentiveness. "
                "Your responses will help shape the child's development.",
        description="System prompt for the Mother LLM"
    )


class DashboardConfig(BaseModel):
    """Configuration for the dashboard interface."""
    host: str = Field(
        default="127.0.0.1",
        description="Host to run the dashboard on"
    )
    port: int = Field(
        default=8050,
        description="Port to run the dashboard on"
    )
    debug: bool = Field(
        default=False,
        description="Whether to run in debug mode"
    )
    theme: str = Field(
        default="light",
        description="Dashboard theme (light or dark)"
    )
    update_interval_ms: int = Field(
        default=1000,
        description="Interval between dashboard updates in milliseconds"
    )
    
    @validator('port')
    def validate_port(cls, v):
        if not (1024 <= v <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        return v
    
    @validator('theme')
    def validate_theme(cls, v):
        if v not in ["light", "dark"]:
            raise ValueError("Theme must be either 'light' or 'dark'")
        return v


class NeuralChildConfig(BaseModel):
    """Main configuration for the NeuralChild project."""
    time: TimeConfig = Field(default_factory=TimeConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    mother_personality: MotherPersonalityTraits = Field(default_factory=MotherPersonalityTraits)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    # Initialize the default configuration with appropriate stage thresholds
    @classmethod
    def create_default_config(cls):
        """Create a default configuration with appropriate stage thresholds."""
        config = cls()
        
        # Set up stage thresholds
        config.development.stage_thresholds = {
            DevelopmentalStage.PRENATAL: StageThresholds(
                language_vocabulary_size=0,
                language_grammar_complexity=0.0,
                emotional_stability=0.0,
                emotional_diversity=0.0,
                cognitive_abstraction=0.0,
                social_understanding=0.0,
                minimum_age_days=0
            ),
            DevelopmentalStage.INFANCY: StageThresholds(
                language_vocabulary_size=10,
                language_grammar_complexity=0.1,
                emotional_stability=0.2,
                emotional_diversity=0.2,
                cognitive_abstraction=0.1,
                social_understanding=0.1,
                minimum_age_days=90
            ),
            DevelopmentalStage.EARLY_CHILDHOOD: StageThresholds(
                language_vocabulary_size=300,
                language_grammar_complexity=0.3,
                emotional_stability=0.3,
                emotional_diversity=0.4,
                cognitive_abstraction=0.3,
                social_understanding=0.3,
                minimum_age_days=365
            ),
            DevelopmentalStage.MIDDLE_CHILDHOOD: StageThresholds(
                language_vocabulary_size=1000,
                language_grammar_complexity=0.6,
                emotional_stability=0.5,
                emotional_diversity=0.6,
                cognitive_abstraction=0.5,
                social_understanding=0.5,
                minimum_age_days=1825  # ~5 years
            ),
            DevelopmentalStage.ADOLESCENCE: StageThresholds(
                language_vocabulary_size=5000,
                language_grammar_complexity=0.8,
                emotional_stability=0.6,
                emotional_diversity=0.8,
                cognitive_abstraction=0.7,
                social_understanding=0.7,
                minimum_age_days=3650  # ~10 years
            ),
            DevelopmentalStage.EARLY_ADULTHOOD: StageThresholds(
                language_vocabulary_size=10000,
                language_grammar_complexity=0.9,
                emotional_stability=0.8,
                emotional_diversity=0.9,
                cognitive_abstraction=0.9,
                social_understanding=0.85,
                minimum_age_days=6570  # ~18 years
            ),
        }
        
        return config


# Create and export the default configuration
CONFIG = NeuralChildConfig.create_default_config()