import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# More detailed format for debugging
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"

# Log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Store configured loggers to avoid duplicate setup
_configured_loggers: Dict[str, logging.Logger] = {}

# Dictionary to store module-specific log level overrides
MODULE_LOG_LEVELS: Dict[str, str] = {}

# Global default log level that can be set by the main application
GLOBAL_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
    detailed_format: bool = False
) -> logging.Logger:
    """
    Get a configured logger.
    
    Parameters:
    name: Logger name (typically module name)
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file (if None, no file logging)
    max_file_size: Maximum log file size in bytes before rotation
    backup_count: Number of backup log files to keep
    console_output: Whether to output logs to console
    detailed_format: Whether to use detailed format with file and line info
    
    Returns:
    Configured logger
    """
    # Check if logger already configured
    if name in _configured_loggers:
        return _configured_loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatter
    log_format = DEBUG_FORMAT if detailed_format else DEFAULT_FORMAT
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Cache the configured logger
    _configured_loggers[name] = logger
    
    return logger


def setup_system_logging(
    log_dir: Union[str, Path] = "logs",
    log_level: str = "ERROR",
    detailed_format: bool = False
) -> logging.Logger:
    """
    Set up system-wide logging for the LMM project.
    
    Parameters:
    log_dir: Directory to store log files
    log_level: Logging level for the system logger
    detailed_format: Whether to use detailed format
    
    Returns:
    System logger
    """
    # Update the global log level
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = log_level
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"lmm_system_{timestamp}.log"
    
    # Get system logger
    logger = get_logger(
        name="lmm.system",
        log_level=log_level,
        log_file=log_file,
        detailed_format=detailed_format
    )
    
    # Log system startup
    logger.info(f"LMM System logging initialized (level: {log_level})")
    
    return logger


def get_module_logger(module_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Parameters:
    module_name: Name of the module
    log_level: Optional override for log level
    
    Returns:
    Module logger
    """
    full_name = f"lmm.{module_name}"
    
    # Get the log level with priority:
    # 1. Explicitly passed log_level parameter
    # 2. Module-specific override in MODULE_LOG_LEVELS
    # 3. Environment variable
    # 4. Global default
    if log_level is None:
        log_level = MODULE_LOG_LEVELS.get(module_name, 
                     os.environ.get(f"{module_name.upper()}_LOG_LEVEL",
                     GLOBAL_LOG_LEVEL))
    
    # Create logger with standard console output
    logger = get_logger(full_name, log_level=log_level)
    
    return logger


def set_module_log_level(module_name: str, log_level: str) -> None:
    """
    Set the log level for a specific module.
    
    Parameters:
    module_name: Name of the module (without 'lmm.' prefix)
    log_level: Log level to set
    """
    MODULE_LOG_LEVELS[module_name] = log_level
    
    # Update existing logger if it's already configured
    full_name = f"lmm.{module_name}"
    if full_name in _configured_loggers:
        logger = _configured_loggers[full_name]
        level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
        logger.setLevel(level)
