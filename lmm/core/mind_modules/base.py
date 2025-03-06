"""
Base module for all mind modules in the Large Mind Model (LMM).

This module provides the base class for all mind modules, ensuring consistent
interfaces and functionality across the modules.
"""
from typing import Dict, Any, Optional

from lmm.utils.logging import get_logger

logger = get_logger("lmm.mind_modules.base")

class MindModule:
    """
    Base class for all mind modules.
    
    This class defines the common interface that all mind modules must implement
    and provides basic functionality shared across modules.
    """
    
    def __init__(self, name: str):
        """
        Initialize the mind module.
        
        Args:
            name: Name of the module
        """
        self.name = name
        self.active = True
        logger.info(f"Initialized {name} Mind Module")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data for this mind module.
        
        Args:
            input_data: Dictionary containing input data and operation instructions
            
        Returns:
            Dictionary with operation results
        """
        raise NotImplementedError("Mind modules must implement the process method")
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get the current status of the module.
        
        Returns:
            Dictionary with module status
        """
        return {
            "name": self.name,
            "status": "active" if self.active else "inactive"
        }
    
    def activate(self) -> None:
        """Activate the module."""
        self.active = True
        logger.info(f"Activated {self.name} Mind Module")
    
    def deactivate(self) -> None:
        """Deactivate the module."""
        self.active = False
        logger.info(f"Deactivated {self.name} Mind Module") 