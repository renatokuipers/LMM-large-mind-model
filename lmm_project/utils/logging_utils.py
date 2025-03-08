import logging 
from typing import Optional, Dict, Any 
 
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO): 
    """Set up a logger with the specified configuration""" 
    logger = logging.getLogger(name) 
    logger.setLevel(level) 
 
    formatter = logging.Formatter('(name)s - (message)s') 
 
    # Console handler 
    console_handler = logging.StreamHandler() 
    console_handler.setFormatter(formatter) 
    logger.addHandler(console_handler) 
 
    # File handler if specified 
    if log_file: 
        file_handler = logging.FileHandler(log_file) 
        file_handler.setFormatter(formatter) 
        logger.addHandler(file_handler) 
 
    return logger 
