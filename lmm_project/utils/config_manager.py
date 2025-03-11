import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from lmm_project.core.exceptions import ConfigurationError
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger("config_manager")

# Singleton instance
_config_instance = None


class ConfigManager:
    """
    Configuration manager for LMM project.
    Handles loading configuration from YAML files and environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Parameters:
        config_path: Path to configuration YAML file
        """
        # Load environment variables first
        self._load_env_vars()
        
        # Set default config path if not provided
        if config_path is None:
            config_path = os.environ.get("CONFIG_PATH", "config.yml")
        
        # Store configuration
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
        
        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file if present."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug("Loaded environment variables from .env file")

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
                
            if not isinstance(self.config, dict):
                raise ConfigurationError("Invalid configuration format, expected dictionary")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Initialize with empty config
            self.config = {}
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Parameters:
        key_path: Dot-separated path to configuration value (e.g., "system.log_level")
        default: Default value to return if key not found
        
        Returns:
        Configuration value or default
        """
        # Check for environment variable override
        env_key = key_path.replace(".", "_").upper()
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return self._convert_env_value(env_value)
        
        # Navigate to the value in nested dict using the key path
        value = self.config
        for key in key_path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
            
        return value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Convert to boolean if applicable
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Convert to integer if applicable
        try:
            return int(value)
        except ValueError:
            pass
        
        # Convert to float if applicable
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

    def get_boolean(self, key_path: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key_path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)

    def get_int(self, key_path: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key_path, default)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key_path: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key_path, default)
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_string(self, key_path: str, default: str = "") -> str:
        """Get string configuration value."""
        value = self.get(key_path, default)
        if value is None:
            return default
        return str(value)

    def get_list(self, key_path: str, default: Optional[list] = None) -> list:
        """Get list configuration value."""
        if default is None:
            default = []
        value = self.get(key_path, default)
        if isinstance(value, list):
            return value
        # If it's a string, split by commas
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        return default

    def get_dict(self, key_path: str, default: Optional[dict] = None) -> dict:
        """Get dictionary configuration value."""
        if default is None:
            default = {}
        value = self.get(key_path, default)
        if isinstance(value, dict):
            return value
        return default


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the singleton configuration manager instance.
    
    Parameters:
    config_path: Path to configuration YAML file
    
    Returns:
    ConfigManager instance
    """
    global _config_instance
    
    # Try different locations for the config file if not provided
    if config_path is None:
        possible_paths = [
            "config.yml",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yml"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lmm_project", "config.yml")
        ]
        
        # Find the first path that exists
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from the specified path.
    
    Parameters:
    config_path: Path to the configuration file to load
    
    Returns:
    Dictionary containing the configuration
    """
    config_manager = get_config(config_path)
    return config_manager.config
