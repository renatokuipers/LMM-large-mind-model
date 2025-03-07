"""
Configuration settings for the Large Mind Model (LMM) project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM interaction."""
    base_url: str = Field("http://192.168.2.12:1234", description="Base URL for the LLM API")
    default_model: str = Field("qwen2.5-7b-instruct", description="Default LLM model to use")
    embedding_model: str = Field(
        "text-embedding-nomic-embed-text-v1.5@q4_k_m", 
        description="Model for generating embeddings"
    )
    temperature: float = Field(0.7, description="Default temperature for LLM generation")
    max_tokens: int = Field(2048, description="Default maximum tokens for LLM generation")


class MemoryConfig(BaseModel):
    """Configuration for the memory systems."""
    vector_dimensions: int = Field(1024, description="Dimensions for vector embeddings")
    memory_store_path: Path = Field(
        Path("memory_store"), 
        description="Path to store persistent memory files"
    )
    faiss_index_path: Optional[Path] = Field(
        None, 
        description="Path to the FAISS index file"
    )


class MotherConfig(BaseModel):
    """Configuration for the 'Mother' LLM characteristics."""
    personality_traits: Dict[str, float] = Field(
        {
            "patience": 0.8,
            "kindness": 0.8,
            "strictness": 0.4,
            "curiosity": 0.7,
            "expressiveness": 0.6,
        },
        description="Personality traits of the 'Mother' LLM"
    )
    parenting_style: str = Field(
        "authoritative", 
        description="Parenting style: authoritative, permissive, authoritarian, or neglectful"
    )
    teaching_approach: str = Field(
        "guided_discovery", 
        description="Teaching approach: direct_instruction, guided_discovery, inquiry_based, etc."
    )


class DevelopmentConfig(BaseModel):
    """Configuration for developmental stages and progression."""
    current_stage: str = Field("prenatal", description="Current developmental stage")
    progression_speed: float = Field(1.0, description="Speed multiplier for development")
    stages: Dict[str, Dict[str, Any]] = Field(
        {
            "prenatal": {"duration": 1, "description": "Initialization phase"},
            "infancy": {"duration": 10, "description": "Basic language and memory formation"},
            "childhood": {"duration": 30, "description": "Developing social and emotional awareness"},
            "adolescence": {"duration": 50, "description": "Critical thinking and moral reasoning"},
            "adulthood": {"duration": -1, "description": "Mature self-awareness and reasoning"}
        },
        description="Developmental stages configuration"
    )


class VisualizationConfig(BaseModel):
    """Configuration for visualization tools."""
    update_interval: float = Field(1.0, description="Update interval for visualizations in seconds")
    enabled_visualizations: Dict[str, bool] = Field(
        {
            "neural_activations": True,
            "emotional_state": True,
            "language_progress": True,
            "memory_usage": True,
            "development_metrics": True
        },
        description="Enabled visualization components"
    )


class Config(BaseModel):
    """Main configuration for the LMM project."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mother: MotherConfig = Field(default_factory=MotherConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    debug_mode: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    workspace_path: Path = Field(
        Path(os.getcwd()), 
        description="Workspace path for the LMM project"
    )


# Create a default configuration instance
default_config = Config() 