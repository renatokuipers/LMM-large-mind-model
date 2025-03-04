# logging_utils.py - Enhanced logging with development tracking
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Global variables for tracking
interaction_log = []
development_milestones = []
current_log_file = None
log_start_time = None

class DevelopmentTrackingHandler(logging.Handler):
    """Custom log handler that tracks developmental milestones"""
    
    def __init__(self, milestone_path: Path):
        super().__init__()
        self.milestone_path = milestone_path
        self.milestones = []
        os.makedirs(milestone_path.parent, exist_ok=True)
    
    def emit(self, record):
        if record.levelno >= logging.INFO:
            msg = record.getMessage()
            
            # Look for developmental milestone keywords
            milestone_keywords = [
                "learned new word", "mastered", "developed", 
                "progressed", "milestone", "first time"
            ]
            
            if any(keyword in msg.lower() for keyword in milestone_keywords):
                milestone = {
                    "timestamp": datetime.now().isoformat(),
                    "logger": record.name,
                    "message": msg,
                    "level": record.levelname
                }
                
                self.milestones.append(milestone)
                
                # Save milestones periodically
                if len(self.milestones) % 10 == 0:
                    self._save_milestones()
    
    def _save_milestones(self):
        """Save milestones to file"""
        milestone_file = self.milestone_path / "development_milestones.json"
        
        try:
            with open(milestone_file, 'w') as f:
                json.dump(self.milestones, f, indent=2)
        except Exception as e:
            print(f"Error saving milestones: {str(e)}")

def setup_logging(
    log_dir: Optional[Path] = None, 
    log_level: int = logging.INFO,
    console_level: int = logging.INFO,
    track_development: bool = True
) -> None:
    """Set up logging with file and console output
    
    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Logging level for file output
        console_level: Logging level for console output
        track_development: Whether to track developmental milestones
    """
    global current_log_file, log_start_time
    
    # Create default log directory if not specified
    if log_dir is None:
        log_dir = Path("./logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"neural_child_{timestamp}.log"
    current_log_file = log_file
    log_start_time = time.time()
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level, console_level))  # Set to lowest level being used
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add development tracking handler if requested
    if track_development:
        milestone_dir = log_dir / "milestones"
        os.makedirs(milestone_dir, exist_ok=True)
        
        dev_handler = DevelopmentTrackingHandler(milestone_dir)
        dev_handler.setLevel(logging.INFO)
        dev_handler.setFormatter(formatter)
        root_logger.addHandler(dev_handler)
    
    # Log initialization
    logging.info(f"Logging initialized to {log_file}")

def log_interaction(
    child_message: str,
    mother_message: str,
    child_emotion: str,
    mother_emotion: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log an interaction between mother and child
    
    Args:
        child_message: Message from the child
        mother_message: Message from the mother
        child_emotion: Child's emotion
        mother_emotion: Mother's emotion
        metadata: Additional metadata
    """
    global interaction_log
    
    # Create interaction entry
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "child_message": child_message,
        "mother_message": mother_message,
        "child_emotion": child_emotion,
        "mother_emotion": mother_emotion
    }
    
    # Add metadata if provided
    if metadata:
        interaction["metadata"] = metadata
    
    # Add to interaction log
    interaction_log.append(interaction)
    
    # Log to system log
    logging.info(f"INTERACTION | Child: '{child_message}' [{child_emotion}] | "
                f"Mother: '{mother_message}' [{mother_emotion}]")
    
    # Save interaction log periodically
    if len(interaction_log) % 50 == 0:
        save_interaction_log()

def log_development_milestone(
    milestone: str,
    category: str,
    importance: int = 1,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log a developmental milestone
    
    Args:
        milestone: Description of the milestone
        category: Category of development (language, emotional, cognitive, etc.)
        importance: Importance level (1-3, with 3 being most important)
        details: Additional details about the milestone
    """
    global development_milestones
    
    # Create milestone entry
    milestone_entry = {
        "timestamp": datetime.now().isoformat(),
        "milestone": milestone,
        "category": category,
        "importance": importance
    }
    
    # Add details if provided
    if details:
        milestone_entry["details"] = details
    
    # Add to milestones
    development_milestones.append(milestone_entry)
    
    # Log to system log with appropriate emphasis
    if importance == 3:
        logging.info(f"MAJOR MILESTONE | {category.upper()}: {milestone}")
    elif importance == 2:
        logging.info(f"MILESTONE | {category}: {milestone}")
    else:
        logging.info(f"Development | {category}: {milestone}")
    
    # Save milestones periodically
    if len(development_milestones) % 10 == 0:
        save_development_milestones()

def save_interaction_log() -> None:
    """Save the interaction log to file"""
    global interaction_log, current_log_file
    
    if not interaction_log or not current_log_file:
        return
    
    try:
        log_dir = current_log_file.parent
        interaction_file = log_dir / "interaction_history.json"
        
        with open(interaction_file, 'w') as f:
            json.dump(interaction_log, f, indent=2)
            
        logging.debug(f"Interaction log saved to {interaction_file}")
        
    except Exception as e:
        logging.error(f"Error saving interaction log: {str(e)}")

def save_development_milestones() -> None:
    """Save development milestones to file"""
    global development_milestones, current_log_file
    
    if not development_milestones or not current_log_file:
        return
    
    try:
        log_dir = current_log_file.parent
        milestone_dir = log_dir / "milestones"
        os.makedirs(milestone_dir, exist_ok=True)
        
        milestone_file = milestone_dir / "development_milestones.json"
        
        with open(milestone_file, 'w') as f:
            json.dump(development_milestones, f, indent=2)
            
        logging.debug(f"Development milestones saved to {milestone_file}")
        
    except Exception as e:
        logging.error(f"Error saving development milestones: {str(e)}")

def log_metrics(metrics: Dict[str, Any], category: str = "general") -> None:
    """Log metrics for tracking and visualization
    
    Args:
        metrics: Dictionary of metrics
        category: Category of metrics
    """
    try:
        if not current_log_file:
            return
        
        log_dir = current_log_file.parent
        metrics_dir = log_dir / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Add timestamp
        timestamped_metrics = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - log_start_time,
            "metrics": metrics
        }
        
        # Append to category-specific metrics file
        metrics_file = metrics_dir / f"{category}_metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(timestamped_metrics) + '\n')
            
        logging.debug(f"Metrics logged to {metrics_file}")
        
    except Exception as e:
        logging.error(f"Error logging metrics: {str(e)}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def create_rotating_log_handler(
    log_dir: Path,
    base_filename: str,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5
) -> logging.Handler:
    """Create a rotating file handler
    
    Args:
        log_dir: Directory for log files
        base_filename: Base filename for log files
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        RotatingFileHandler instance
    """
    from logging.handlers import RotatingFileHandler
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f"{base_filename}.log"
    
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    handler.setFormatter(formatter)
    
    return handler

def configure_module_logging(
    module_name: str,
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    enable_file_logging: bool = True
) -> logging.Logger:
    """Configure logging for a specific module
    
    Args:
        module_name: Name of the module
        log_dir: Directory for log files
        log_level: Logging level
        enable_file_logging: Whether to log to file
    
    Returns:
        Logger instance
    """
    if log_dir is None and current_log_file is not None:
        log_dir = current_log_file.parent
    elif log_dir is None:
        log_dir = Path("./logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Get logger for module
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Add file handler if requested
    if enable_file_logging:
        # Use module-specific log file
        module_log_path = log_dir / f"{module_name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(module_log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_exception(e: Exception, context: str = "") -> None:
    """Log an exception with context
    
    Args:
        e: Exception to log
        context: Context description
    """
    if context:
        logging.error(f"Exception in {context}: {str(e)}", exc_info=True)
    else:
        logging.error(f"Exception: {str(e)}", exc_info=True)

def format_metrics_for_dashboard(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Format metrics for dashboard display
    
    Args:
        metrics: Raw metrics dictionary
    
    Returns:
        Formatted metrics for dashboard
    """
    # Convert values to appropriate formats for dashboard
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, float):
            # Round floats to 2 decimal places
            formatted[key] = round(value, 2)
        elif isinstance(value, datetime):
            # Format datetime as ISO string
            formatted[key] = value.isoformat()
        elif isinstance(value, list) and len(value) > 10:
            # Truncate long lists
            formatted[key] = value[:10]
        else:
            # Keep other values as is
            formatted[key] = value
    
    return formatted