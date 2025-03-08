# Attention module

from typing import Optional, Dict, Any

from lmm_project.core.event_bus import EventBus
from lmm_project.modules.attention.focus_controller import FocusController
from lmm_project.modules.attention.salience_detector import SalienceDetector
from lmm_project.modules.attention.models import AttentionFocus
from lmm_project.modules.attention.neural_net import AttentionNetwork

def get_module(module_id: str, event_bus: Optional[EventBus] = None) -> Any:
    """
    Factory function to create an attention module.
    
    The attention system controls the allocation of cognitive resources, 
    determining what information is selected for further processing. 
    It combines two key components:
    
    1. Salience detection - Identifies important stimuli
    2. Focus control - Manages limited attention resources
    
    Returns:
    An instance of AttentionSystem
    """
    return AttentionSystem(
        module_id=module_id,
        event_bus=event_bus
    )

class AttentionSystem:
    """
    Integrated attention system that manages focus and salience detection
    
    This class serves as a facade over the focus controller and salience
    detector components, providing a unified interface for attention processes.
    """
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        # Initialize core components
        self.module_id = module_id
        self.module_type = "attention"
        self.event_bus = event_bus
        
        # Create sub-components
        self.focus_controller = FocusController(f"{module_id}_focus", event_bus)
        self.salience_detector = SalienceDetector(f"{module_id}_salience", event_bus)
        
        # Neural network for attention processing
        input_dim = 128
        hidden_dim = 256
        output_dim = 64
        
        self.neural_network = AttentionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Track development level as average of sub-components
        self.development_level = 0.0
        
        # Current attention state
        self.current_focus = AttentionFocus()
        
        # Initialize system
        self.is_active = True
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the attention system
        
        Parameters:
        input_data: Dictionary with:
            - operation: The operation to perform
            - Additional parameters specific to the operation
            
        Returns:
        The result from the appropriate sub-component
        """
        operation = input_data.get("operation", "")
        
        # Route to appropriate sub-component or operation
        if operation == "detect_salience":
            return self.salience_detector.process_input(input_data)
            
        elif operation == "focus":
            return self.focus_controller.process_input(input_data)
            
        elif operation == "get_focus":
            return {
                "status": "success",
                "focus": self.current_focus.model_dump(),
                "capacity": self.current_focus.capacity,
                "targets": self.current_focus.targets
            }
            
        elif operation == "update_focus":
            # Combined operation that updates focus based on salience
            
            # First detect salience
            salience_result = self.salience_detector.process_input({
                "operation": "detect_salience",
                "inputs": input_data.get("inputs", {})
            })
            
            # Then update focus based on salience results
            focus_result = self.focus_controller.process_input({
                "operation": "update_focus",
                "salience_scores": salience_result.get("salience_scores", {})
            })
            
            # Update current focus
            self.current_focus = self.focus_controller.current_focus
            
            # Return combined result
            return {
                "status": "success",
                "salience_result": salience_result,
                "focus_result": focus_result,
                "current_focus": self.current_focus.model_dump()
            }
            
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update development of attention sub-components
        
        Parameters:
        amount: Development amount to apply
        
        Returns:
        Average development level across sub-components
        """
        # Update each sub-component
        focus_level = self.focus_controller.update_development(amount)
        salience_level = self.salience_detector.update_development(amount)
        
        # Calculate average development level
        self.development_level = (focus_level + salience_level) / 2.0
        
        # As development increases, increase attention capacity
        base_capacity = 3.0
        max_capacity = 7.0
        self.current_focus.capacity = base_capacity + (max_capacity - base_capacity) * self.development_level
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the attention system"""
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "focus_controller": self.focus_controller.get_state(),
            "salience_detector": self.salience_detector.get_state(),
            "current_focus": self.current_focus.model_dump()
        }
    
    def save_state(self) -> Dict[str, str]:
        """Save the state of all attention sub-components"""
        return {
            "focus_controller": self.focus_controller.save_state(),
            "salience_detector": self.salience_detector.save_state()
        }