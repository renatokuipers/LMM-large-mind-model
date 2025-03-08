import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logger(
    name: str = "LMM",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up a logger with file and/or console output
    
    Parameters:
    name: Logger name
    log_level: Logging level (e.g., logging.INFO)
    log_file: Path to log file (if None, will use default path)
    console_output: Whether to output logs to console
    log_format: Format string for log messages
    
    Returns:
    Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    handlers = []
    
    # File handler
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Default log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{name.lower()}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

def log_state_change(
    logger: logging.Logger,
    component: str,
    old_state: Dict[str, Any],
    new_state: Dict[str, Any],
    message: Optional[str] = None
) -> None:
    """
    Log a state change with detailed information
    
    Parameters:
    logger: Logger to use
    component: Component name
    old_state: Previous state
    new_state: New state
    message: Optional message to include
    """
    # Find changed keys
    changed_keys = []
    for key in new_state:
        if key in old_state:
            if new_state[key] != old_state[key]:
                changed_keys.append(key)
        else:
            changed_keys.append(key)
    
    # Create log message
    log_msg = f"State change in {component}: "
    if message:
        log_msg += message + " "
    
    # Add changed values
    changes = []
    for key in changed_keys:
        old_val = old_state.get(key, "N/A")
        new_val = new_state.get(key, "N/A")
        changes.append(f"{key}: {old_val} -> {new_val}")
    
    log_msg += ", ".join(changes)
    
    # Log the message
    logger.info(log_msg)

def log_development_milestone(
    logger: logging.Logger,
    module: str,
    milestone: str,
    details: Dict[str, Any]
) -> None:
    """
    Log a developmental milestone
    
    Parameters:
    logger: Logger to use
    module: Module name
    milestone: Milestone description
    details: Additional details about the milestone
    """
    logger.info(f"MILESTONE: {module} - {milestone}")
    for key, value in details.items():
        logger.info(f"  {key}: {value}")

def get_log_level(level_name: str) -> int:
    """
    Convert a log level name to its numeric value
    
    Parameters:
    level_name: Log level name (e.g., "INFO", "DEBUG")
    
    Returns:
    Numeric log level
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    return levels.get(level_name.upper(), logging.INFO)
