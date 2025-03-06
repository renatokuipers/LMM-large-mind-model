"""
Configuration management for the Large Mind Model (LMM) project.
"""
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

class LLMConfig(BaseModel):
    """Configuration for the LLM client."""
    base_url: str = Field("http://192.168.2.12:1234", description="Base URL for the LLM API")
    chat_model: str = Field("qwen2.5-7b-instruct", description="Default model for chat completions")
    embedding_model: str = Field("text-embedding-nomic-embed-text-v1.5@q4_k_m", 
                                description="Default model for embeddings")
    temperature: float = Field(0.7, description="Default temperature for completions")
    max_tokens: int = Field(-1, description="Default max tokens for completions")

class MemoryConfig(BaseModel):
    """Configuration for the memory system."""
    vector_db_path: str = Field("./data/vector_db", description="Path to store vector database files")
    use_gpu: bool = Field(True, description="Whether to use GPU for vector operations")
    gpu_device: int = Field(0, description="GPU device ID to use")
    
    @field_validator('vector_db_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure the path uses OS-appropriate separators."""
        return os.path.normpath(v)

class MotherPersonalityConfig(BaseModel):
    """Configuration for the Mother LLM personality."""
    nurturing_level: float = Field(0.8, ge=0.0, le=1.0, 
                                  description="How nurturing the mother is (0.0-1.0)")
    patience_level: float = Field(0.7, ge=0.0, le=1.0, 
                                 description="How patient the mother is (0.0-1.0)")
    teaching_style: str = Field("supportive", 
                               description="Teaching style (supportive, challenging, etc.)")
    emotional_expressiveness: float = Field(0.6, ge=0.0, le=1.0, 
                                          description="How emotionally expressive the mother is (0.0-1.0)")
    
    model_config = {"extra": "forbid"}

class DevelopmentConfig(BaseModel):
    """Configuration for the developmental stages."""
    current_stage: str = Field("prenatal", 
                              description="Current developmental stage")
    acceleration_factor: float = Field(100.0, gt=0.0, 
                                      description="How much faster than real-time development occurs")
    enable_plateaus: bool = Field(True, 
                                 description="Whether to enable developmental plateaus")
    
    model_config = {"extra": "forbid"}

class VisualizationConfig(BaseModel):
    """Configuration for the visualization dashboard."""
    update_interval_seconds: float = Field(1.0, gt=0.0, 
                                         description="How often to update the dashboard")
    metrics_history_length: int = Field(1000, gt=0, 
                                       description="How many historical metrics to keep")
    
    model_config = {"extra": "forbid"}

class LMMConfig(BaseModel):
    """Main configuration for the Large Mind Model."""
    llm: LLMConfig = Field(default_factory=LLMConfig, 
                          description="LLM client configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, 
                                description="Memory system configuration")
    mother: MotherPersonalityConfig = Field(default_factory=MotherPersonalityConfig, 
                                          description="Mother personality configuration")
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig, 
                                         description="Development configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, 
                                             description="Visualization configuration")
    
    model_config = {"extra": "forbid"}

# Default configuration instance
default_config = LMMConfig()

# Global configuration instance that can be modified
config = LMMConfig()

def load_config_from_dict(config_dict: Dict) -> None:
    """Load configuration from a dictionary."""
    global config
    config = LMMConfig.model_validate(config_dict)

def load_config_from_env() -> None:
    """
    Load configuration from environment variables.
    
    Environment variables should be prefixed with LMM_ and follow the structure:
    LMM_SECTION_KEY=value
    
    Examples:
    LMM_LLM_BASE_URL=http://localhost:1234
    LMM_MEMORY_USE_GPU=False
    LMM_DEVELOPMENT_ACCELERATION_FACTOR=200.0
    """
    import os
    from typing import Any, get_type_hints
    import json
    
    global config
    
    # Get all environment variables with LMM_ prefix
    lmm_vars = {k: v for k, v in os.environ.items() if k.startswith("LMM_")}
    
    if not lmm_vars:
        return  # No environment variables to process
    
    # Create a dictionary to hold the configuration
    config_dict = {}
    
    # Process each environment variable
    for env_var, value in lmm_vars.items():
        try:
            # Remove prefix and split into sections
            parts = env_var.replace("LMM_", "", 1).lower().split("_", 1)
            
            if len(parts) != 2:
                continue  # Skip invalid format
            
            section, key = parts
            
            # Create section dictionary if it doesn't exist
            if section not in config_dict:
                config_dict[section] = {}
                
            # Convert value to appropriate type based on the current config
            # Get expected type from the model
            section_model = getattr(LMMConfig, section).default
            expected_type = None
            
            if hasattr(section_model, key):
                expected_type = type(getattr(section_model, key))
            elif hasattr(section_model, "model_fields") and key in section_model.model_fields:
                field_info = section_model.model_fields[key]
                expected_type = field_info.annotation
            
            # Perform type conversion
            if expected_type == bool:
                typed_value = value.lower() in ("true", "1", "yes", "y", "t")
            elif expected_type == int:
                typed_value = int(value)
            elif expected_type == float:
                typed_value = float(value)
            elif expected_type == list or expected_type == dict:
                # Parse JSON for complex types
                typed_value = json.loads(value)
            else:
                # Default to string
                typed_value = value
                
            # Assign the value to the configuration dictionary
            config_dict[section][key] = typed_value
            
        except Exception as e:
            import logging
            logging.warning(f"Error processing environment variable {env_var}: {str(e)}")
    
    # Update the configuration
    if config_dict:
        try:
            # Create a copy of current config as dict
            current_config = config.model_dump()
            
            # Update with new values
            for section, section_dict in config_dict.items():
                if section in current_config:
                    current_config[section].update(section_dict)
                else:
                    current_config[section] = section_dict
            
            # Load updated config
            config = LMMConfig.model_validate(current_config)
        except Exception as e:
            import logging
            logging.error(f"Failed to update configuration from environment variables: {str(e)}")
            # Keep using the existing configuration

def save_config_to_file(filepath: str) -> bool:
    """
    Save the current configuration to a JSON file.
    
    Args:
        filepath: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    import os
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save configuration to file
        with open(filepath, 'w') as f:
            json.dump(config.model_dump(), f, indent=4)
        return True
    except Exception as e:
        import logging
        logging.error(f"Failed to save configuration to {filepath}: {str(e)}")
        return False

def load_config_from_file(filepath: str) -> bool:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    
    global config
    
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Validate and load configuration
        config = LMMConfig.model_validate(config_dict)
        return True
    except Exception as e:
        import logging
        logging.error(f"Failed to load configuration from {filepath}: {str(e)}")
        return False

def get_config() -> LMMConfig:
    """Get the current configuration."""
    return config 