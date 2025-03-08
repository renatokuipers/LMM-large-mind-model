from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel, Field
from datetime import datetime
import os
import importlib
import inspect

from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message
from lmm_project.core.exceptions import ModuleInitializationError
from lmm_project.modules.base_module import BaseModule

class Mind(BaseModel):
    """The integrated mind that coordinates all cognitive modules"""
    age: float = Field(default=0.0)
    developmental_stage: str = Field(default="prenatal")
    modules: Dict[str, BaseModule] = Field(default_factory=dict)
    initialization_time: datetime = Field(default_factory=datetime.now)
    event_bus: EventBus = Field(default_factory=EventBus)
    state_manager: StateManager = Field(default_factory=StateManager)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

    def initialize_modules(self):
        """Initialize all cognitive modules"""
        module_types = [
            "perception", "attention", "memory", "language",
            "emotion", "consciousness", "executive", "social",
            "motivation", "temporal", "creativity", "self_regulation",
            "learning", "identity", "belief"
        ]
        
        try:
            for module_type in module_types:
                module_path = f"lmm_project.modules.{module_type}"
                module_id = f"{module_type}_001"
                
                # Import the module's __init__ file
                try:
                    module_package = importlib.import_module(module_path)
                    
                    # Check if the module defines a get_module function
                    if hasattr(module_package, "get_module"):
                        module_instance = module_package.get_module(module_id, self.event_bus)
                        self.modules[module_type] = module_instance
                    else:
                        # Create a minimal placeholder module
                        from lmm_project.modules.base_module import BaseModule
                        class PlaceholderModule(BaseModule):
                            def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                                return {"status": "placeholder", "data": input_data}
                            
                            def update_development(self, amount: float) -> float:
                                self.development_level = min(1.0, self.development_level + amount)
                                return self.development_level
                                
                        self.modules[module_type] = PlaceholderModule(
                            module_id=module_id,
                            module_type=module_type,
                            is_active=True,
                            development_level=0.0
                        )
                        
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not initialize {module_type} module: {e}")
                    
            print(f"Initialized {len(self.modules)} cognitive modules")
        except Exception as e:
            raise ModuleInitializationError(f"Failed to initialize modules: {str(e)}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through all relevant modules"""
        results = {}
        
        # Route input to perception module first
        if "perception" in self.modules:
            perception_results = self.modules["perception"].process_input(input_data)
            
            # Create a message from perception results
            perception_message = Message(
                sender="perception",
                message_type="perception_results",
                content=perception_results
            )
            
            # Publish the message to the event bus
            self.event_bus.publish(perception_message)
            
            # Store in results
            results["perception"] = perception_results
        
        # Process the input through other modules as needed
        # This is a simplified implementation - in reality, the module
        # interactions would be more complex and based on the event_bus
        
        return results
    
    def update_development(self, delta_time: float):
        """Update the mind's developmental progress"""
        # Update age
        self.age += delta_time
        
        # Determine developmental stage based on age
        if self.age < 0.1:
            self.developmental_stage = "prenatal"
        elif self.age < 1.0:
            self.developmental_stage = "infant"
        elif self.age < 3.0:
            self.developmental_stage = "child"
        elif self.age < 6.0:
            self.developmental_stage = "adolescent"
        else:
            self.developmental_stage = "adult"
            
        # Update each module's development based on current stage
        for module_name, module in self.modules.items():
            # Different modules develop at different rates during different stages
            # This is a simplified implementation
            development_rate = 0.01 * delta_time
            
            # Adjust rate based on developmental stage and module type
            if self.developmental_stage == "infant":
                if module_name in ["perception", "attention"]:
                    development_rate *= 2.0
            elif self.developmental_stage == "child":
                if module_name in ["language", "memory"]:
                    development_rate *= 1.5
            elif self.developmental_stage == "adolescent":
                if module_name in ["social", "identity"]:
                    development_rate *= 1.8
                    
            # Update the module's development
            module.update_development(development_rate)
        
        # Update state
        self.state_manager.update_state({
            "age": self.age,
            "developmental_stage": self.developmental_stage,
            "module_development": {name: module.development_level for name, module in self.modules.items()}
        })
        
        return {
            "age": self.age,
            "stage": self.developmental_stage,
            "modules": {name: module.development_level for name, module in self.modules.items()}
        }
