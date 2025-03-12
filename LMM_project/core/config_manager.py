from typing import Dict, Any, Optional, Union, List, Type, TypeVar
from pathlib import Path
import json
import os
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationError

T = TypeVar('T', bound=BaseModel)

# TODO: Define ConfigEnvironment enum (DEVELOPMENT, TESTING, PRODUCTION)
# TODO: Create BaseConfig model with common configuration parameters
# TODO: Implement ConfigLoadError exception class

# TODO: Create ConfigManager class:
#   - __init__ with config directory initialization
#   - load_config method to load JSON into Pydantic model
#   - save_config method to save Pydantic model as JSON
#   - get_environment method to detect current environment
#   - validate_config method to check configuration integrity
#   - get_paths method for Windows-compatible file paths
#   - merge_configs method to combine different config sources
#   - get_module_config to retrieve module-specific settings

# TODO: Implement SystemConfig for global settings:
#   - paths: Dict[str, Path]
#   - memory: Dict[str, Any]
#   - development: Dict[str, Any]
#   - modules: Dict[str, Dict[str, Any]]
#   - inference: Dict[str, Any]
#   - mother: Dict[str, Any]

# TODO: Create config validation methods:
#   - validate_paths to check path existence
#   - validate_compatibility for Windows compatibility
#   - validate_dependencies for config dependencies

# TODO: Add Windows-specific path normalization
# TODO: Implement environment variable support
# TODO: Add config versioning for backward compatibility