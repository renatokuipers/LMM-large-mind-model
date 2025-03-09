"""
Perception Module

This module integrates various components for processing and understanding
sensory input. It serves as the primary interface between the Mind and
external stimuli, converting text input into meaningful patterns and
features for higher cognitive processing.

For this LMM implementation, perception is text-based, as the system does
not have physical sensory organs like eyes or ears.
"""

from typing import Optional, Dict, Any, List

from lmm_project.core.event_bus import EventBus
from lmm_project.modules.base_module import BaseModule
from lmm_project.core.message import Message
from lmm_project.modules.perception.sensory_input import SensoryInputProcessor
from lmm_project.modules.perception.pattern_recognition import PatternRecognizer
from lmm_project.modules.perception.models import PerceptionResult

def get_module(module_id: str, event_bus: Optional[EventBus] = None) -> Any:
    """
    Factory function to create a perception module.
    
    The perception module is the primary sensory interface for the LMM,
    responsible for processing and extracting patterns from text input.
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event communication system
        
    Returns:
        An instance of PerceptionSystem
    """
    return PerceptionSystem(
        module_id=module_id,
        event_bus=event_bus
    )

class PerceptionSystem(BaseModule):
    """
    Integrated perception system that processes sensory input and detects patterns
    
    This class serves as a facade over the constituent perception components,
    coordinating their operation and providing a unified interface for perception.
    """
    # Component modules
    sensory_processor: Optional[SensoryInputProcessor] = None
    pattern_recognizer: Optional[PatternRecognizer] = None
    
    # Result caching
    latest_results: Dict[str, Any] = {}
    result_history: List[str] = []  # Input IDs of processed results
    max_history: int = 20
    
    def __init__(
        self, 
        module_id: str, 
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        """Initialize the perception system"""
        super().__init__(
            module_id=module_id,
            module_type="perception",
            event_bus=event_bus,
            **kwargs
        )
        
        # Create sub-components
        self._initialize_components()
        
        # Subscribe to events
        if self.event_bus:
            self.subscribe_to_message("raw_text_input", self._handle_raw_text)
            self.subscribe_to_message("perception_result", self._handle_perception_result)
            
    def _initialize_components(self):
        """Initialize perception sub-components"""
        # Create sensory processor
        self.sensory_processor = SensoryInputProcessor(
            module_id=f"{self.module_id}_sensory",
            event_bus=self.event_bus,
            developmental_level=self.development_level
        )
        
        # Create pattern recognizer
        self.pattern_recognizer = PatternRecognizer(
            module_id=f"{self.module_id}_patterns",
            event_bus=self.event_bus,
            developmental_level=self.development_level
        )
    
    def update_development(self, amount: float) -> float:
        """
        Update developmental level of the perception system and its components
        
        As the perception system develops, it becomes better at:
        - Processing complex sensory input
        - Detecting subtle patterns
        - Handling sequential patterns
        - Distinguishing similar stimuli
        """
        # Update base module development
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update sub-components
        if self.sensory_processor:
            self.sensory_processor.update_development(amount)
            
        if self.pattern_recognizer:
            self.pattern_recognizer.update_development(amount)
            
        return self.development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the perception pipeline
        
        Args:
            input_data: Dictionary containing input data
                Must include 'text' key with the input text
                
        Returns:
            Dictionary with perception results
        """
        try:
            # Ensure text is in the input
            if "text" not in input_data:
                return {"error": "No text input provided"}
                
            # Process through sensory input processor
            sensory_result = self.sensory_processor.process_input(input_data)
            if "error" in sensory_result:
                return sensory_result
                
            # Process through pattern recognizer
            perception_result = self.pattern_recognizer.process_input(sensory_result)
            
            # Cache result
            if "input_id" in sensory_result:
                input_id = sensory_result["input_id"]
                self.latest_results[input_id] = perception_result
                self.result_history.append(input_id)
                
                # Maintain maximum history size
                if len(self.result_history) > self.max_history:
                    old_id = self.result_history.pop(0)
                    if old_id in self.latest_results:
                        del self.latest_results[old_id]
            
            return perception_result
            
        except Exception as e:
            import logging
            logging.error(f"Error in perception processing: {e}")
            return {"error": f"Perception processing failed: {str(e)}"}
    
    def _handle_raw_text(self, message: Message):
        """Handle raw text input event from event bus"""
        if not message.content:
            return
            
        # Process input directly through the pipeline
        self.process_input(message.content)
        
    def _handle_perception_result(self, message: Message):
        """Handle perception results for caching"""
        if not message.content or "input_id" not in message.content:
            return
            
        # Cache the result
        input_id = message.content["input_id"]
        self.latest_results[input_id] = message.content
        self.result_history.append(input_id)
        
        # Maintain maximum history size
        if len(self.result_history) > self.max_history:
            old_id = self.result_history.pop(0)
            if old_id in self.latest_results:
                del self.latest_results[old_id]
                
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent perception result"""
        if not self.result_history:
            return None
            
        latest_id = self.result_history[-1]
        return self.latest_results.get(latest_id)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the perception system"""
        base_state = super().get_state()
        
        # Add perception system specific state
        component_states = {}
        
        if self.sensory_processor:
            component_states["sensory_processor"] = self.sensory_processor.get_state()
            
        if self.pattern_recognizer:
            component_states["pattern_recognizer"] = self.pattern_recognizer.get_state()
        
        perception_state = {
            "component_states": component_states,
            "results_cached": len(self.latest_results),
            "has_latest_result": len(self.result_history) > 0
        }
        
        base_state.update(perception_state)
        return base_state 
