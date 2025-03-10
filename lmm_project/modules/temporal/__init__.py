# Temporal module 

# TODO: Implement the temporal module factory function to return an integrated TemporalSystem
# This module should be responsible for sequence learning, prediction,
# causality understanding, and time perception.

# TODO: Create TemporalSystem class that integrates all temporal sub-components:
# - sequence_learning: learns patterns over time
# - prediction: anticipates future states
# - causality: understands cause-effect relationships
# - time_perception: tracks and estimates time intervals

# TODO: Implement development tracking for temporal cognition
# Temporal capabilities should develop from simple sequence recognition in early stages
# to sophisticated prediction and causal understanding in later stages

# TODO: Connect temporal module to memory, learning, and consciousness modules
# Temporal cognition should utilize episodic memories, inform
# learning processes, and contribute to conscious awareness

# TODO: Implement prospection capabilities
# Include mental time travel to imagine future scenarios,
# plan sequences of actions, and anticipate outcomes

from typing import Dict, List, Any, Optional
from lmm_project.core.event_bus import EventBus
import logging

from lmm_project.modules.temporal.sequence_learning import SequenceLearning
from lmm_project.modules.temporal.prediction import Prediction
from lmm_project.modules.temporal.causality import Causality
from lmm_project.modules.temporal.time_perception import TimePerception

logger = logging.getLogger(__name__)

class TemporalSystem:
    """
    Integrated temporal cognition system
    
    This class combines sequence learning, prediction,
    causality, and time perception components into a
    unified temporal processing system.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the temporal system with all components
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of the module
        """
        self.module_id = module_id
        self.event_bus = event_bus
        self.module_type = "temporal_system"
        
        # Initialize all temporal components
        self.sequence_learning = SequenceLearning(
            module_id=f"{module_id}/sequence_learning",
            event_bus=event_bus
        )
        
        self.prediction = Prediction(
            module_id=f"{module_id}/prediction",
            event_bus=event_bus
        )
        
        self.causality = Causality(
            module_id=f"{module_id}/causality",
            event_bus=event_bus
        )
        
        self.time_perception = TimePerception(
            module_id=f"{module_id}/time_perception",
            event_bus=event_bus
        )
        
        # Track overall system development
        self.development_level = development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through appropriate components
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing processing results
        """
        input_type = input_data.get("input_type", "")
        
        # Route input to appropriate component
        if "sequence" in input_type or "pattern" in input_type:
            return self.sequence_learning.process_input(input_data)
        elif "predict" in input_type or "forecast" in input_type:
            return self.prediction.process_input(input_data)
        elif "cause" in input_type or "causal" in input_type or "effect" in input_type:
            return self.causality.process_input(input_data)
        elif "time" in input_type or "duration" in input_type or "interval" in input_type:
            return self.time_perception.process_input(input_data)
        else:
            # For unspecified inputs, try each component in priority order
            # based on content keywords
            content = str(input_data).lower()
            
            if any(kw in content for kw in ["sequence", "pattern", "serial", "order"]):
                return self.sequence_learning.process_input(input_data)
            elif any(kw in content for kw in ["predict", "forecast", "future", "anticipate"]):
                return self.prediction.process_input(input_data)
            elif any(kw in content for kw in ["cause", "effect", "because", "result", "lead to"]):
                return self.causality.process_input(input_data)
            elif any(kw in content for kw in ["time", "duration", "interval", "when", "rhythm"]):
                return self.time_perception.process_input(input_data)
            
            # If no clear routing, return error
            return {
                "error": "Unrecognized input type",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update overall development level
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update sub-modules with proportional development
        self.sequence_learning.update_development(amount)
        self.prediction.update_development(amount)
        self.causality.update_development(amount)
        self.time_perception.update_development(amount)
        
        # Log development milestone transitions
        if int(prev_level * 10) != int(self.development_level * 10):
            logger.info(f"Temporal system reached development level {self.development_level:.2f}")
            
        return self.development_level
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of this module
        
        Args:
            level: New development level (0.0 to 1.0)
        """
        # Set overall development level
        self.development_level = max(0.0, min(1.0, level))
        
        # Set sub-modules to the same level
        self.sequence_learning.development_level = self.development_level
        self.prediction.development_level = self.development_level
        self.causality.development_level = self.development_level
        self.time_perception.development_level = self.development_level
        
        logger.info(f"Temporal system development level set to {self.development_level:.2f}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the temporal system
        
        Returns:
            Dictionary containing system state
        """
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "sequence_learning": self.sequence_learning.get_state(),
            "prediction": self.prediction.get_state(),
            "causality": self.causality.get_state(),
            "time_perception": self.time_perception.get_state()
        }
    
    def save_state(self, state_dir: str) -> Dict[str, str]:
        """
        Save the state of all components
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Dictionary mapping component names to saved state paths
        """
        saved_paths = {}
        saved_paths["sequence_learning"] = self.sequence_learning.save_state(state_dir)
        saved_paths["prediction"] = self.prediction.save_state(state_dir)
        saved_paths["causality"] = self.causality.save_state(state_dir)
        saved_paths["time_perception"] = self.time_perception.save_state(state_dir)
        return saved_paths
    
    def load_state(self, state_paths: Dict[str, str]) -> Dict[str, bool]:
        """
        Load the state of all components
        
        Args:
            state_paths: Dictionary mapping component names to state file paths
            
        Returns:
            Dictionary mapping component names to load success status
        """
        load_status = {}
        if "sequence_learning" in state_paths:
            load_status["sequence_learning"] = self.sequence_learning.load_state(state_paths["sequence_learning"])
        if "prediction" in state_paths:
            load_status["prediction"] = self.prediction.load_state(state_paths["prediction"])
        if "causality" in state_paths:
            load_status["causality"] = self.causality.load_state(state_paths["causality"])
        if "time_perception" in state_paths:
            load_status["time_perception"] = self.time_perception.load_state(state_paths["time_perception"])
        
        # Recalculate overall development level
        self.development_level = (
            self.sequence_learning.development_level + 
            self.prediction.development_level + 
            self.causality.development_level + 
            self.time_perception.development_level
        ) / 4.0
        
        return load_status


def get_module(module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0) -> Any:
    """
    Factory function to create a temporal module
    
    This function is responsible for creating a temporal system that can:
    - Recognize and learn sequential patterns
    - Predict future states based on current conditions
    - Understand and infer causal relationships
    - Track and estimate time intervals
    - Project into past and future (mental time travel)
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level of the module
        
    Returns:
        An instance of the TemporalSystem class
    """
    return TemporalSystem(module_id=module_id, event_bus=event_bus, development_level=development_level)
