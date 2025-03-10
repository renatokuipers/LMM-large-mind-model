# Executive module 

# TODO: Implement the executive module factory function to return an integrated ExecutiveSystem
# This module should be responsible for planning, decision-making, inhibition,
# cognitive control, and working memory management.

# TODO: Create ExecutiveSystem class that integrates all executive sub-components:
# - planning: develops and executes plans to achieve goals
# - decision_making: evaluates options and makes choices
# - inhibition: suppresses inappropriate actions and thoughts
# - working_memory_control: manages contents of working memory

# TODO: Implement development tracking for executive function
# Executive capabilities should develop from minimal control in early stages
# to sophisticated planning and self-regulation in later stages

# TODO: Connect executive module to attention, consciousness, and motivation modules
# Executive function should direct attention resources, be influenced by
# conscious goals, and be driven by motivational priorities

# TODO: Implement resource management for executive functions
# The system should have limited executive resources that must be
# allocated efficiently across different control demands

from typing import Optional, Dict, Any, List, Tuple, Union
import uuid
import logging
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.base_module import BaseModule
from lmm_project.modules.executive.planning import Planning
from lmm_project.modules.executive.decision_making import DecisionMaking
from lmm_project.modules.executive.inhibition import Inhibition
from lmm_project.modules.executive.working_memory_control import WorkingMemoryControl
from lmm_project.modules.executive.models import ExecutiveSystemState, ExecutiveParameters, ExecutiveNeuralState, Plan, Decision

# Initialize logger
logger = logging.getLogger(__name__)

class ExecutiveSystem(BaseModule):
    """
    Coordinates executive functions for high-level cognitive control
    
    The executive system integrates planning, decision-making, inhibition,
    and working memory control to manage goal-directed behavior.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic reflexive control",
        0.2: "Simple goal-directed actions", 
        0.4: "Basic planning and inhibition",
        0.6: "Integrated executive functions",
        0.8: "Strategic planning and control",
        1.0: "Sophisticated executive functions"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the executive system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="executive", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create sub-modules with appropriate IDs
        base_id = module_id.split('.')[-1] if '.' in module_id else module_id
        
        self.planning = Planning(
            module_id=f"{base_id}.planning", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.decision_making = DecisionMaking(
            module_id=f"{base_id}.decision", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.inhibition = Inhibition(
            module_id=f"{base_id}.inhibition", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.working_memory = WorkingMemoryControl(
            module_id=f"{base_id}.working_memory", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create system state
        self.system_state = ExecutiveSystemState(
            module_id=module_id,
            developmental_level=development_level,
            parameters=ExecutiveParameters(),
            neural_state=ExecutiveNeuralState()
        )
        
        # Subscribe to events if event bus is provided
        if self.event_bus:
            self._subscribe_to_events()
        
        logger.info(f"Executive system initialized at development level {development_level:.2f}")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events from other modules"""
        # Subscribe to attention focus events
        self.event_bus.subscribe(
            "attention.focus_change",
            self._handle_attention_event
        )
        
        # Subscribe to emotion events for emotional regulation
        self.event_bus.subscribe(
            "emotion.state_change",
            self._handle_emotion_event
        )
        
        # Subscribe to perception events
        self.event_bus.subscribe(
            "perception.new_input",
            self._handle_perception_event
        )
        
        # Subscribe to memory events
        self.event_bus.subscribe(
            "memory.retrieval",
            self._handle_memory_event
        )
    
    def _handle_attention_event(self, message: Message):
        """Handle attention focus change events"""
        # Focus changes may require inhibition or working memory updates
        if self.development_level >= 0.4:
            # Extract focus data
            focus_data = message.content
            
            # Update working memory with new focus if appropriate
            if "focus_content" in focus_data and focus_data.get("focus_strength", 0) > 0.5:
                self.working_memory.process_input({
                    "operation": "store",
                    "content": focus_data["focus_content"],
                    "content_type": "attention_focus",
                    "tags": ["attention_focus"]
                })
    
    def _handle_emotion_event(self, message: Message):
        """Handle emotion state change events"""
        # Emotions can influence decision making and may require inhibition
        if self.development_level >= 0.4:
            emotion_data = message.content
            
            # If strong negative emotion, may need inhibitory control
            if emotion_data.get("valence", 0) < -0.7 and emotion_data.get("intensity", 0) > 0.7:
                self.inhibition.process_input({
                    "operation": "apply",
                    "stimulus": "emotional_response",
                    "context": {
                        "emotion_type": emotion_data.get("emotion_type", "unknown"),
                        "intensity": emotion_data.get("intensity", 0),
                        "valence": emotion_data.get("valence", 0)
                    }
                })
    
    def _handle_perception_event(self, message: Message):
        """Handle new perception input events"""
        # New perceptions may trigger planning or decision making
        perception_data = message.content
        
        # Store relevant perceptions in working memory
        if perception_data.get("importance", 0) > 0.6:
            self.working_memory.process_input({
                "operation": "store",
                "content": perception_data.get("content", {}),
                "content_type": "perception",
                "tags": ["perception", perception_data.get("modality", "unknown")]
            })
    
    def _handle_memory_event(self, message: Message):
        """Handle memory retrieval events"""
        # Retrieved memories may provide context for decisions or plans
        memory_data = message.content
        
        # Store retrieved memories in working memory if relevant
        if memory_data.get("relevance", 0) > 0.7:
            self.working_memory.process_input({
                "operation": "store",
                "content": memory_data.get("content", {}),
                "content_type": "memory_retrieval",
                "tags": ["memory", memory_data.get("memory_type", "unknown")]
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the executive system
        
        Args:
            input_data: Dictionary containing operation and parameters
                Required keys:
                - 'operation': The operation to perform
                  Options: 'plan', 'decide', 'inhibit', 'working_memory', 'status'
                
                For 'plan' operation, forwards to planning module
                For 'decide' operation, forwards to decision_making module
                For 'inhibit' operation, forwards to inhibition module
                For 'working_memory' operation, forwards to working_memory module
                For 'status' operation, returns overall executive system status
                
        Returns:
            Dictionary with the results of processing
        """
        operation = input_data.get("operation", "status")
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Process based on operation
        if operation == "plan":
            result = self.planning.process_input(input_data)
            
            # Update system state with new or updated plans
            if result.get("status") == "success" and "plan" in result:
                plan = result["plan"]
                self.system_state.add_plan(plan)
                
            return result
            
        elif operation == "decide":
            result = self.decision_making.process_input(input_data)
            
            # Update system state with new decisions
            if result.get("status") == "success" and "decision" in result:
                decision = result["decision"]
                self.system_state.add_decision(decision)
                
            return result
            
        elif operation == "inhibit":
            return self.inhibition.process_input(input_data)
            
        elif operation == "working_memory":
            return self.working_memory.process_input(input_data)
            
        elif operation == "status":
            # Return overall executive system status
            return {
                "status": "success",
                "process_id": process_id,
                "executive_system": {
                    "development_level": self.development_level,
                    "active_plans_count": len(self.system_state.active_plans),
                    "recent_decisions_count": len(self.system_state.recent_decisions),
                    "working_memory_utilization": self.working_memory.get_state().get("capacity_utilization", 0),
                    "inhibition_resources": self.inhibition.get_state().get("available_resources", 1.0)
                },
                "sub_modules": {
                    "planning": self.planning.get_state(),
                    "decision_making": self.decision_making.get_state(),
                    "inhibition": self.inhibition.get_state(),
                    "working_memory": self.working_memory.get_state()
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module and its submodules
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update submodules
        self.planning.update_development(amount)
        self.decision_making.update_development(amount)
        self.inhibition.update_development(amount)
        self.working_memory.update_development(amount)
        
        # Update system state
        self.system_state.developmental_level = new_level
        self.system_state.last_updated = datetime.now()
        
        logger.info(f"Executive system development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Get base state from parent
        base_state = super().get_state()
        
        # Update system state with current states from submodules
        planning_state = self.planning.get_state()
        decision_state = self.decision_making.get_state()
        inhibition_state = self.inhibition.get_state()
        working_memory_state = self.working_memory.get_state()
        
        # Update system state
        self.system_state.planning_state = planning_state
        self.system_state.decision_state = decision_state
        self.system_state.inhibition_state = inhibition_state.get("inhibition_state", {})
        self.system_state.working_memory_state = working_memory_state.get("working_memory_state", {})
        
        # Combine states
        combined_state = {
            **base_state,
            "system_state": self.system_state.dict(),
            "submodules": {
                "planning": planning_state,
                "decision_making": decision_state,
                "inhibition": inhibition_state,
                "working_memory": working_memory_state
            }
        }
        
        return combined_state


def get_module(module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0) -> Any:
    """
    Factory function to create an executive module
    
    This function is responsible for creating an executive system that can:
    - Plan and execute goal-directed actions
    - Make decisions based on available information
    - Inhibit inappropriate responses
    - Control working memory contents
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level of the module
        
    Returns:
        An instance of the ExecutiveSystem class
    """
    return ExecutiveSystem(module_id=module_id, event_bus=event_bus, development_level=development_level)
