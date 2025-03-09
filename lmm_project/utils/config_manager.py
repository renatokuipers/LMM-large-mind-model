import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class NeuralSubstrateConfig(BaseModel):
    """Neural substrate configuration settings"""
    default_activation_function: str = "sigmoid"
    default_learning_rate: float = Field(0.01, ge=0.0, le=1.0)
    hebbian_learning_enabled: bool = True
    use_gpu: bool = True
    fallback_to_cpu: bool = True
    batch_size: int = 64

class MotherPersonalityConfig(BaseModel):
    """Mother LLM personality configuration"""
    nurturing: float = Field(0.8, ge=0.0, le=1.0)
    patient: float = Field(0.9, ge=0.0, le=1.0)
    encouraging: float = Field(0.8, ge=0.0, le=1.0)
    structured: float = Field(0.7, ge=0.0, le=1.0)
    responsive: float = Field(0.9, ge=0.0, le=1.0)

class MotherConfig(BaseModel):
    """Mother LLM configuration settings"""
    voice: str = "af_bella"
    teaching_style: str = "socratic"
    personality: MotherPersonalityConfig = Field(default_factory=MotherPersonalityConfig)

class DevelopmentConfig(BaseModel):
    """Development rate and stage configuration"""
    default_rate: float = Field(0.01, ge=0.0, le=1.0)
    default_cycles: int = 100
    save_interval: int = 50
    critical_periods_enabled: bool = True
    milestone_tracking_enabled: bool = True
    accelerated_mode: bool = False

class VisualizationConfig(BaseModel):
    """Visualization configuration settings"""
    enabled: bool = True
    update_interval: int = 5
    show_neural_activity: bool = True
    show_development_charts: bool = True
    show_memory_visualization: bool = True

class APIConfig(BaseModel):
    """API endpoint configuration"""
    llm_api_url: str = "http://192.168.2.12:1234"
    tts_api_url: str = "http://127.0.0.1:7860"

class StorageConfig(BaseModel):
    """Storage configuration settings"""
    checkpoint_dir: str = "storage/checkpoints"
    experience_dir: str = "storage/experiences"
    memory_dir: str = "storage/memories"
    backup_enabled: bool = True
    backup_interval: int = 1000
    max_backups: int = 10

class LMMConfig(BaseModel):
    """Main configuration model for the LMM project"""
    development_mode: bool = True
    log_level: str = "INFO"
    apis: APIConfig = Field(default_factory=APIConfig)
    mother: MotherConfig = Field(default_factory=MotherConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    neural_substrate: NeuralSubstrateConfig = Field(default_factory=NeuralSubstrateConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    active_modules: List[str] = Field(
        default_factory=lambda: [
            "perception", "attention", "memory", "language", 
            "emotion", "consciousness", "executive", "social", 
            "motivation", "temporal", "creativity", "self_regulation", 
            "learning", "identity", "belief"
        ]
    )

class ConfigManager:
    """Manages loading and accessing configuration settings for the LMM project"""
    
    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the config.yml file (default: project root config.yml)
            env_path: Path to the .env file (default: project root .env)
        """
        self.config_path = config_path or self._find_config_file()
        self.env_path = env_path or self._find_env_file()
        self.config = self._load_config()
        
    def _find_config_file(self) -> str:
        """Find the config.yml file in the project directory"""
        # Try different possible locations
        possible_paths = [
            "config.yml",
            "lmm_project/config.yml",
            "../config.yml",
            os.path.join(os.path.dirname(__file__), "..", "config.yml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Default to the one in the current directory
        logger.warning("Could not find config.yml file, using default configuration")
        return "config.yml"
        
    def _find_env_file(self) -> str:
        """Find the .env file in the project directory"""
        # Try different possible locations
        possible_paths = [
            ".env",
            "lmm_project/.env",
            "../.env",
            os.path.join(os.path.dirname(__file__), "..", ".env")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Default to the one in the current directory
        logger.warning("Could not find .env file, environment variables may not be loaded")
        return ".env"
        
    def _load_config(self) -> LMMConfig:
        """
        Load configuration from config.yml and .env files
        
        Returns:
            Validated LMMConfig object
        """
        # Load environment variables
        load_dotenv(self.env_path)
        
        # Read config file
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using default values")
                yaml_config = {}
                
            # Override with environment variables if they exist
            # API endpoints
            if "LLM_API_URL" in os.environ:
                if "apis" not in yaml_config:
                    yaml_config["apis"] = {}
                yaml_config["apis"]["llm_api_url"] = os.environ["LLM_API_URL"]
                
            if "TTS_API_URL" in os.environ:
                if "apis" not in yaml_config:
                    yaml_config["apis"] = {}
                yaml_config["apis"]["tts_api_url"] = os.environ["TTS_API_URL"]
                
            # Development mode based on environment
            if "ENVIRONMENT" in os.environ:
                env = os.environ["ENVIRONMENT"].lower()
                yaml_config["development_mode"] = env == "development"
                
            # Debug settings
            if "DEBUG" in os.environ:
                debug = os.environ["DEBUG"].lower() in ["true", "1", "yes"]
                if debug:
                    yaml_config["log_level"] = "DEBUG"
                    
            if "DETAILED_LOGGING" in os.environ and os.environ["DETAILED_LOGGING"].lower() in ["true", "1", "yes"]:
                yaml_config["log_level"] = "DEBUG"
                
            # GPU configuration
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                if "neural_substrate" not in yaml_config:
                    yaml_config["neural_substrate"] = {}
                # Only enable GPU if CUDA_VISIBLE_DEVICES is not -1
                yaml_config["neural_substrate"]["use_gpu"] = os.environ["CUDA_VISIBLE_DEVICES"] != "-1"
                
            # Development acceleration
            if "ACCELERATED_DEVELOPMENT" in os.environ:
                if "development" not in yaml_config:
                    yaml_config["development"] = {}
                yaml_config["development"]["accelerated_mode"] = os.environ["ACCELERATED_DEVELOPMENT"].lower() in ["true", "1", "yes"]
                
            # Create validated config
            return LMMConfig(**yaml_config)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Return default configuration
            return LMMConfig()
            
    def get_config(self) -> LMMConfig:
        """Get the full configuration object"""
        return self.config
        
    def update_config(self, **kwargs) -> None:
        """
        Update configuration with provided values
        
        Args:
            **kwargs: Configuration values to update
        """
        # Create a dictionary from current config
        config_dict = self.config.model_dump()
        
        # Update the dictionary with new values
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys (e.g., "mother.voice")
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
                
        # Recreate config object with updated values
        try:
            self.config = LMMConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Invalid configuration update: {str(e)}")
            
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file
        
        Args:
            path: Path to save the config file (default: original config path)
        """
        save_path = path or self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Convert config to dict
            config_dict = self.config.model_dump()
            
            # Save to file
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            if "." in key:
                # Handle nested keys
                parts = key.split(".")
                value = self.config
                for part in parts:
                    value = getattr(value, part)
                return value
            else:
                return getattr(self.config, key)
        except (AttributeError, KeyError):
            return default
            
# Create a global instance for easy imports
config_manager = ConfigManager()

def get_config() -> LMMConfig:
    """Get the global configuration object"""
    return config_manager.get_config() 