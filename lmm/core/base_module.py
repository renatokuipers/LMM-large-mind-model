"""
Base Module for all cognitive components in the Large Mind Model.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch


class BaseModule(ABC):
    """
    Abstract base class for all cognitive modules in the Large Mind Model.
    
    Each module represents a specific psychological function or capability and must
    implement the required abstract methods.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive module.
        
        Args:
            name: The name of the module
            config: Optional configuration dictionary for the module
        """
        self.name = name
        self.config = config or {}
        self.initialized = False
        self.development_level = 0.0  # 0.0 to 1.0, representing development progression
        self.is_active = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network state
        self.model = None
        
        # Module connections
        self.connected_modules = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the module, setting up required resources and neural networks.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and generate outputs based on the module's function.
        
        Args:
            inputs: Dictionary of input data for processing
            
        Returns:
            Dictionary of output data after processing
        """
        pass
    
    @abstractmethod
    def update(self, feedback: Dict[str, Any]) -> None:
        """
        Update the module based on feedback, enabling learning and adaptation.
        
        Args:
            feedback: Dictionary containing feedback information
        """
        pass
    
    def connect(self, module_name: str, module: 'BaseModule') -> None:
        """
        Connect this module to another module to enable inter-module communication.
        
        Args:
            module_name: Name of the module to connect to
            module: Reference to the module object
        """
        self.connected_modules[module_name] = module
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module.
        
        Returns:
            Dictionary representing the module's current state
        """
        return {
            "name": self.name,
            "initialized": self.initialized,
            "development_level": self.development_level,
            "is_active": self.is_active,
            "connected_modules": list(self.connected_modules.keys())
        }
    
    def save_state(self, path: str) -> bool:
        """
        Save the module's state to disk.
        
        Args:
            path: Path where the state should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is not None:
                torch.save(self.model.state_dict(), f"{path}/{self.name}_model.pt")
            return True
        except Exception as e:
            print(f"Error saving {self.name} state: {e}")
            return False
    
    def load_state(self, path: str) -> bool:
        """
        Load the module's state from disk.
        
        Args:
            path: Path from which to load the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is not None:
                state_dict = torch.load(f"{path}/{self.name}_model.pt", map_location=self.device)
                self.model.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"Error loading {self.name} state: {e}")
            return False 