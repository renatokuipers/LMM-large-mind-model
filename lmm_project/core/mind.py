"""
Mind - Core cognitive architecture controller
"""

import logging
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union, Type
import os
import json
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message
from lmm_project.core.exceptions import ModuleInitializationError

# Type annotations with strings to avoid circular imports
if TYPE_CHECKING:
    from lmm_project.modules.base_module import BaseModule

logger = logging.getLogger(__name__)

class Mind:
    """
    Central coordinator for all cognitive modules
    
    The Mind integrates all cognitive modules, manages developmental progression,
    and coordinates information flow between components.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        state_manager: StateManager,
        initial_age: float = 0.0,
        developmental_stage: str = "prenatal"
    ):
        """
        Initialize the Mind
        
        Args:
            event_bus: Event bus for inter-module communication
            state_manager: State manager for tracking system state
            initial_age: Initial age of the mind
            developmental_stage: Initial developmental stage
        """
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.age = initial_age
        self.developmental_stage = developmental_stage
        self.modules: Dict[str, Any] = {}  # Use Any instead of BaseModule to avoid circular imports
        self.creation_time = datetime.now()
        
        logger.info(f"Mind initialized at age {initial_age} in {developmental_stage} stage")
        
    def initialize_modules(self):
        """
        Initialize all cognitive modules
        
        This method creates instances of all required cognitive modules and 
        establishes connections between them.
        """
        logger.info("Initializing cognitive modules...")
        
        # Import modules here to avoid circular imports
        from lmm_project.modules import get_module_classes
        
        module_classes = get_module_classes()
        
        # Create instances of all modules
        for module_type, module_class in module_classes.items():
            module_id = f"{module_type}_{int(time.time())}"
            try:
                module = module_class(
                    module_id=module_id,
                    event_bus=self.event_bus
                )
                self.modules[module_type] = module
                logger.info(f"Initialized {module_type} module")
            except Exception as e:
                logger.error(f"Failed to initialize {module_type} module: {str(e)}")
                
        logger.info(f"Initialized {len(self.modules)} cognitive modules")
        
    def update_development(self, delta_time: float):
        """
        Update the mind's developmental progression
        
        Args:
            delta_time: Amount of developmental time to add
        """
        # Update mind age
        prev_age = self.age
        self.age += delta_time
        
        # Update all modules with appropriate fraction of development
        for module_type, module in self.modules.items():
            # Different modules may develop at different rates
            # Here we use a simple approach where all modules develop equally
            module.update_development(delta_time)
            
        logger.debug(f"Mind development updated: age {prev_age:.2f} -> {self.age:.2f}")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the complete cognitive pipeline
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing processing results
        """
        results = {}
        
        # First, process through perception
        if "perception" in self.modules:
            results["perception"] = self.modules["perception"].process_input(input_data)
            
        # TODO: Implement full cognitive pipeline
        
        return results
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the mind
        
        Returns:
            Dictionary containing mind state
        """
        modules_state = {}
        for module_type, module in self.modules.items():
            modules_state[module_type] = module.get_state()
            
        return {
            "age": self.age,
            "developmental_stage": self.developmental_stage,
            "modules": modules_state,
            "creation_time": self.creation_time.isoformat()
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the mind state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Ensure directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get complete state
        state = self.get_state()
        
        # Save to file
        file_path = os.path.join(state_dir, f"mind_state_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Mind state saved to {file_path}")
        return file_path
        
    def load_state(self, state_path: str) -> bool:
        """
        Load the mind state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state file
            with open(state_path, "r") as f:
                state = json.load(f)
                
            # Update mind properties
            self.age = state.get("age", self.age)
            self.developmental_stage = state.get("developmental_stage", self.developmental_stage)
            
            # Update module states
            # Note: This would need more sophisticated logic in a real implementation
            
            logger.info(f"Mind state loaded from {state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load mind state: {str(e)}")
            return False
