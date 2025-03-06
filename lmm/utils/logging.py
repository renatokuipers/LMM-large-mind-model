"""
Logging utilities for the Large Mind Model (LMM) project.
"""
import os
import logging
import datetime
from typing import Optional

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(
    name: str, 
    level: str = "info", 
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Log level (debug, info, warning, error, critical)
        log_file: Path to log file (if None, no file logging)
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    # Get the log level
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default loggers
main_logger = setup_logger("lmm.main", "info")
memory_logger = setup_logger("lmm.memory", "info")
mother_logger = setup_logger("lmm.mother", "info")
development_logger = setup_logger("lmm.development", "info")
mind_modules_logger = setup_logger("lmm.mind_modules", "info")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 