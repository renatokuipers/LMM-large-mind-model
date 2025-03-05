"""
Configuration module for the NeuralChild project.

This module contains the configuration settings for the NeuralChild project.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Base directory for storing data
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Development stages configuration
class DevelopmentStageConfig(BaseModel):
    """Configuration for a developmental stage."""
    name: str
    description: str
    min_age_months: float
    max_age_months: float
    language_milestones: Dict[str, Any]
    emotional_milestones: Dict[str, Any]
    cognitive_milestones: Dict[str, Any]
    social_milestones: Dict[str, Any]

# Default development stages
DEVELOPMENT_STAGES = [
    DevelopmentStageConfig(
        name="Prenatal",
        description="Neural architecture formation",
        min_age_months=-9.0,
        max_age_months=0.0,
        language_milestones={
            "receptive_language": 0.0,
            "expressive_language": 0.0,
            "vocabulary_size": 0,
        },
        emotional_milestones={
            "basic_emotions": 0.1,
            "emotional_regulation": 0.0,
            "emotional_complexity": 0.0,
        },
        cognitive_milestones={
            "attention": 0.1,
            "memory": 0.1,
            "problem_solving": 0.0,
            "abstract_thinking": 0.0,
        },
        social_milestones={
            "attachment": 0.1,
            "social_awareness": 0.0,
            "empathy": 0.0,
            "theory_of_mind": 0.0,
        },
    ),
    DevelopmentStageConfig(
        name="Infancy",
        description="Babbling, basic recognition",
        min_age_months=0.0,
        max_age_months=12.0,
        language_milestones={
            "receptive_language": 0.3,
            "expressive_language": 0.2,
            "vocabulary_size": 50,
        },
        emotional_milestones={
            "basic_emotions": 0.5,
            "emotional_regulation": 0.2,
            "emotional_complexity": 0.1,
        },
        cognitive_milestones={
            "attention": 0.3,
            "memory": 0.3,
            "problem_solving": 0.1,
            "abstract_thinking": 0.0,
        },
        social_milestones={
            "attachment": 0.5,
            "social_awareness": 0.2,
            "empathy": 0.1,
            "theory_of_mind": 0.0,
        },
    ),
    DevelopmentStageConfig(
        name="Early Childhood",
        description="Rapid vocabulary acquisition",
        min_age_months=12.0,
        max_age_months=36.0,
        language_milestones={
            "receptive_language": 0.6,
            "expressive_language": 0.5,
            "vocabulary_size": 1000,
        },
        emotional_milestones={
            "basic_emotions": 0.8,
            "emotional_regulation": 0.4,
            "emotional_complexity": 0.3,
        },
        cognitive_milestones={
            "attention": 0.5,
            "memory": 0.5,
            "problem_solving": 0.3,
            "abstract_thinking": 0.1,
        },
        social_milestones={
            "attachment": 0.8,
            "social_awareness": 0.5,
            "empathy": 0.3,
            "theory_of_mind": 0.1,
        },
    ),
    DevelopmentStageConfig(
        name="Middle Childhood",
        description="Grammar emergence",
        min_age_months=36.0,
        max_age_months=96.0,
        language_milestones={
            "receptive_language": 0.8,
            "expressive_language": 0.7,
            "vocabulary_size": 5000,
        },
        emotional_milestones={
            "basic_emotions": 0.9,
            "emotional_regulation": 0.6,
            "emotional_complexity": 0.5,
        },
        cognitive_milestones={
            "attention": 0.7,
            "memory": 0.7,
            "problem_solving": 0.5,
            "abstract_thinking": 0.3,
        },
        social_milestones={
            "attachment": 0.9,
            "social_awareness": 0.7,
            "empathy": 0.5,
            "theory_of_mind": 0.4,
        },
    ),
    DevelopmentStageConfig(
        name="Adolescence",
        description="Abstract thinking development",
        min_age_months=96.0,
        max_age_months=216.0,
        language_milestones={
            "receptive_language": 0.9,
            "expressive_language": 0.9,
            "vocabulary_size": 10000,
        },
        emotional_milestones={
            "basic_emotions": 1.0,
            "emotional_regulation": 0.8,
            "emotional_complexity": 0.7,
        },
        cognitive_milestones={
            "attention": 0.9,
            "memory": 0.9,
            "problem_solving": 0.7,
            "abstract_thinking": 0.6,
        },
        social_milestones={
            "attachment": 1.0,
            "social_awareness": 0.9,
            "empathy": 0.7,
            "theory_of_mind": 0.7,
        },
    ),
    DevelopmentStageConfig(
        name="Early Adulthood",
        description="Refinement and specialization",
        min_age_months=216.0,
        max_age_months=360.0,
        language_milestones={
            "receptive_language": 1.0,
            "expressive_language": 1.0,
            "vocabulary_size": 20000,
        },
        emotional_milestones={
            "basic_emotions": 1.0,
            "emotional_regulation": 0.9,
            "emotional_complexity": 0.9,
        },
        cognitive_milestones={
            "attention": 1.0,
            "memory": 1.0,
            "problem_solving": 0.9,
            "abstract_thinking": 0.9,
        },
        social_milestones={
            "attachment": 1.0,
            "social_awareness": 1.0,
            "empathy": 0.9,
            "theory_of_mind": 0.9,
        },
    ),
]

# Mother personality configuration
class MotherPersonalityConfig(BaseModel):
    """Configuration for the Mother's personality."""
    warmth: float = Field(0.7, ge=0.0, le=1.0)
    responsiveness: float = Field(0.7, ge=0.0, le=1.0)
    patience: float = Field(0.7, ge=0.0, le=1.0)
    teaching_style: str = Field("nurturing", pattern="^(nurturing|directive|authoritative|permissive)$")
    emotional_expressiveness: float = Field(0.7, ge=0.0, le=1.0)
    verbal_communication: float = Field(0.7, ge=0.0, le=1.0)
    consistency: float = Field(0.7, ge=0.0, le=1.0)

# Default mother personality
DEFAULT_MOTHER_PERSONALITY = MotherPersonalityConfig()

# Neural Child configuration
class NeuralChildConfig(BaseModel):
    """Configuration for the Neural Child."""
    initial_age_months: float = 0.0
    development_speed: float = Field(1.0, gt=0.0)  # 1.0 = real-time, 10.0 = 10x faster
    learning_rate: float = Field(0.01, gt=0.0)
    memory_capacity: int = Field(10000, gt=0)
    emotional_sensitivity: float = Field(0.7, ge=0.0, le=1.0)
    curiosity: float = Field(0.7, ge=0.0, le=1.0)
    attention_span: float = Field(0.5, ge=0.0, le=1.0)
    temperament: str = Field("balanced", pattern="^(easy|difficult|slow-to-warm-up|balanced)$")

# Default neural child configuration
DEFAULT_NEURAL_CHILD_CONFIG = NeuralChildConfig()

# LLM configuration
class LLMConfig(BaseModel):
    """Configuration for the LLM."""
    base_url: str = "http://192.168.2.12:1234"
    model: str = "qwen2.5-7b-instruct"
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(1000, gt=0)

# Default LLM configuration
DEFAULT_LLM_CONFIG = LLMConfig()

# Dashboard configuration
class DashboardConfig(BaseModel):
    """Configuration for the dashboard."""
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    theme: str = "dark"
    update_interval_ms: int = 1000

# Default dashboard configuration
DEFAULT_DASHBOARD_CONFIG = DashboardConfig()

# Global configuration
class GlobalConfig(BaseModel):
    """Global configuration for the NeuralChild project."""
    development_stages: List[DevelopmentStageConfig] = DEVELOPMENT_STAGES
    mother_personality: MotherPersonalityConfig = DEFAULT_MOTHER_PERSONALITY
    neural_child: NeuralChildConfig = DEFAULT_NEURAL_CHILD_CONFIG
    llm: LLMConfig = DEFAULT_LLM_CONFIG
    dashboard: DashboardConfig = DEFAULT_DASHBOARD_CONFIG

# Default global configuration
DEFAULT_CONFIG = GlobalConfig() 