"""
Configuration utilities for AgenDev.

This module contains functions for loading and managing configuration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "workspace_path": os.getcwd(),
    "host": "127.0.0.1",
    "port": 8000,
    "llm": {
        "provider": "local",  # Default to local provider
        "model": "qwen2.5-7b-instruct",  # Use Qwen by default
        "endpoint": "http://192.168.2.12:1234"  # Local API endpoint
    }
}


def load_config(config_file: str = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Initialize with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Check for configuration file path
    if not config_file:
        # Look for config file in common locations
        possible_locations = [
            Path("config.json"),
            Path("agendev_config.json"),
            Path.home() / ".agendev" / "config.json",
            Path("/etc/agendev/config.json")
        ]
        
        for location in possible_locations:
            if location.exists():
                config_file = str(location)
                break
    
    # Load configuration file if it exists
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                
            # Merge with default configuration
            config = deep_merge(config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
    
    # Check for environment variables
    env_config = extract_env_config()
    if env_config:
        config = deep_merge(config, env_config)
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to override base
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def extract_env_config() -> Dict[str, Any]:
    """
    Extract configuration from environment variables.
    
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Extract LLM configuration
    if os.getenv("AGENDEV_LLM_PROVIDER"):
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["provider"] = os.getenv("AGENDEV_LLM_PROVIDER")
    
    if os.getenv("AGENDEV_LLM_MODEL"):
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["model"] = os.getenv("AGENDEV_LLM_MODEL")
    
    if os.getenv("AGENDEV_LLM_ENDPOINT"):
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["endpoint"] = os.getenv("AGENDEV_LLM_ENDPOINT")
    
    # Extract server configuration
    if os.getenv("AGENDEV_HOST"):
        config["host"] = os.getenv("AGENDEV_HOST")
    
    if os.getenv("AGENDEV_PORT"):
        try:
            config["port"] = int(os.getenv("AGENDEV_PORT"))
        except ValueError:
            logger.error(f"Invalid port number: {os.getenv('AGENDEV_PORT')}")
    
    # Extract workspace configuration
    if os.getenv("AGENDEV_WORKSPACE_PATH"):
        config["workspace_path"] = os.getenv("AGENDEV_WORKSPACE_PATH")
    
    return config 