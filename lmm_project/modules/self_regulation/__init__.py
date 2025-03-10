"""
Self-Regulation Module

This module integrates impulse control, emotional regulation, and self-monitoring
capabilities to enable the mind to modulate its own states and responses.

The self-regulation system serves as a critical component for managing internal 
states, allowing the mind to adapt to changing circumstances through 
emotional regulation, impulse control, and monitoring of states and processes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import os
from pathlib import Path
from datetime import datetime
import uuid

from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.base_module import BaseModule
from lmm_project.modules.self_regulation.emotional_regulation import EmotionalRegulation
from lmm_project.modules.self_regulation.impulse_control import ImpulseControl
from lmm_project.modules.self_regulation.self_monitoring import SelfMonitoring
from lmm_project.modules.self_regulation.models import (
    EmotionalState, ImpulseEvent, MonitoringEvent, 
    RegulationStrategy, SelfRegulationState
)
from lmm_project.modules.self_regulation.neural_net import RegulationController

# Export public classes
__all__ = [
    'SelfRegulationSystem',
    'EmotionalRegulation',
    'ImpulseControl',
    'SelfMonitoring',
    'EmotionalState',
    'ImpulseEvent',
    'MonitoringEvent',
    'RegulationStrategy',
    'SelfRegulationState',
]

# Registry of self-regulation systems
self_regulation_registry: Dict[str, 'SelfRegulationSystem'] = {}

def get_self_regulation_system(system_id: str) -> Optional['SelfRegulationSystem']:
    """Get a registered self-regulation system by ID"""
    return self_regulation_registry.get(system_id)

def register_self_regulation_system(system: 'SelfRegulationSystem') -> None:
    """Register a self-regulation system in the global registry"""
    self_regulation_registry[system.module_id] = system

def get_module(
    module_id: str = "self_regulation",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "SelfRegulationSystem":
    """
    Factory function to create and initialize a self-regulation module
    
    Args:
        module_id: Unique identifier for this module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level (0.0-1.0)
        
    Returns:
        Initialized self-regulation module
    """
    module = SelfRegulationSystem(module_id=module_id, event_bus=event_bus)
    module.set_development_level(development_level)
    return module

class SelfRegulationSystem(BaseModule):
    """
    Integrated self-regulation system
    
    This system coordinates emotional regulation, impulse control, and
    self-monitoring to enable the mind to modulate its own states and responses.
    
    The self-regulation system develops from basic external regulation in early
    stages to sophisticated internal regulation in later developmental stages.
    """
    
    # Developmental milestones for self-regulation
    development_milestones = {
        0.1: "Basic awareness of internal states",
        0.2: "Simple external regulation strategies",
        0.3: "Emerging basic self-regulation capabilities",
        0.4: "Basic emotional and impulse regulation",
        0.5: "Developing self-monitoring",
        0.6: "Integrated regulation strategies",
        0.7: "Advanced emotional regulation",
        0.8: "Sophisticated impulse control",
        0.9: "Highly effective self-monitoring",
        1.0: "Fully integrated self-regulation system"
    }
    
    def __init__(
        self, 
        module_id: str, 
        event_bus: Optional[EventBus] = None, 
        development_level: float = 0.0,
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the self-regulation system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
            parameters: Configuration parameters for the system
        """
        super().__init__(
            module_id=module_id,
            module_type="self_regulation",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.self_regulation.{module_id}")
        self.parameters = parameters or {}
        
        # Initialize component systems
        self.emotional_regulation = EmotionalRegulation(
            module_id=f"{module_id}.emotional_regulation",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.impulse_control = ImpulseControl(
            module_id=f"{module_id}.impulse_control",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.self_monitoring = SelfMonitoring(
            module_id=f"{module_id}.self_monitoring",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Neural controller for integrated processing
        self.neural_controller = RegulationController(
            developmental_level=development_level
        )
        
        # Apply configuration parameters
        self._apply_parameters(self.parameters)
        
        # Register message handlers
        if event_bus:
            self.subscribe_to_message("regulate_emotion")
            self.subscribe_to_message("control_impulse")
            self.subscribe_to_message("monitor_state")
            self.subscribe_to_message("regulation_query")
        
        # Register in global registry
        register_self_regulation_system(self)
        
        self.logger.info(f"Self-regulation system initialized at development level {development_level:.2f}")
    
    def _apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply configuration parameters to the system components"""
        # Apply emotional regulation parameters
        if "emotional_regulation" in parameters:
            er_params = parameters["emotional_regulation"]
            
            # Set regulation thresholds
            if "auto_regulation_threshold" in er_params:
                # This would require an adapter method in the EmotionalRegulation class
                # Example: self.emotional_regulation.set_auto_regulation_threshold(er_params["auto_regulation_threshold"])
                pass
                
        # Apply impulse control parameters
        if "impulse_control" in parameters:
            ic_params = parameters["impulse_control"]
            
            # Set inhibition thresholds
            # This would require an adapter method in the ImpulseControl class
            # Example: self.impulse_control.set_inhibition_threshold(ic_params["inhibition_threshold"])
            pass
            
        # Apply self-monitoring parameters
        if "self_monitoring" in parameters:
            sm_params = parameters["self_monitoring"]
            
            # Set monitoring frequency
            # This would require an adapter method in the SelfMonitoring class
            # Example: self.self_monitoring.set_monitoring_frequency(sm_params["monitoring_frequency"])
            pass
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for self-regulation
        
        Args:
            input_data: Dictionary with input data
                Required fields:
                - "type": Type of regulation ("emotion", "impulse", "monitoring", "query")
                - Other fields depend on the regulation type
                
        Returns:
            Dictionary with processing results
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input")
        
        if input_type == "emotion":
            return self.emotional_regulation.process_input({
                "type": "emotion",
                **{k: v for k, v in input_data.items() if k != "type"}
            })
            
        elif input_type == "impulse":
            return self.impulse_control.process_input({
                "type": "impulse",
                **{k: v for k, v in input_data.items() if k != "type"}
            })
            
        elif input_type == "monitoring":
            return self.self_monitoring.process_input({
                "type": "state",
                **{k: v for k, v in input_data.items() if k != "type"}
            })
            
        elif input_type == "regulation_request":
            target = input_data.get("target", "")
            
            if target == "emotion":
                return self.emotional_regulation.process_input({
                    "type": "regulation_request",
                    **{k: v for k, v in input_data.items() if k not in ["type", "target"]}
                })
                
            elif target == "impulse":
                return self.impulse_control.process_input({
                    "type": "control_request",
                    **{k: v for k, v in input_data.items() if k not in ["type", "target"]}
                })
                
            else:
                return {
                    "success": False,
                    "message": f"Unknown regulation target: {target}"
                }
                
        elif input_type == "query":
            query_target = input_data.get("target", "system")
            
            if query_target == "emotion":
                return self.emotional_regulation.process_input({
                    "type": "query",
                    **{k: v for k, v in input_data.items() if k not in ["type", "target"]}
                })
                
            elif query_target == "impulse":
                return self.impulse_control.process_input({
                    "type": "query",
                    **{k: v for k, v in input_data.items() if k not in ["type", "target"]}
                })
                
            elif query_target == "monitoring":
                return self.self_monitoring.process_input({
                    "type": "query",
                    **{k: v for k, v in input_data.items() if k not in ["type", "target"]}
                })
                
            elif query_target == "system":
                return self._process_system_query(input_data)
                
            else:
                return {
                    "success": False,
                    "message": f"Unknown query target: {query_target}"
                }
        else:
            return {
                "success": False,
                "message": f"Unknown input type: {input_type}"
            }
    
    def _process_system_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query about the overall self-regulation system"""
        query_type = query_data.get("query_type", "state")
        
        if query_type == "state":
            return {
                "success": True,
                "system_state": self.get_regulation_state().dict()
            }
            
        elif query_type == "development":
            return {
                "success": True,
                "development_level": self.development_level,
                "emotional_regulation": self.emotional_regulation.development_level,
                "impulse_control": self.impulse_control.development_level,
                "self_monitoring": self.self_monitoring.development_level,
                "milestones": list(self.get_development_progress()["achieved_milestones"])
            }
            
        elif query_type == "components":
            return {
                "success": True,
                "components": [
                    {
                        "id": self.emotional_regulation.module_id,
                        "type": self.emotional_regulation.module_type,
                        "development_level": self.emotional_regulation.development_level
                    },
                    {
                        "id": self.impulse_control.module_id,
                        "type": self.impulse_control.module_type,
                        "development_level": self.impulse_control.development_level
                    },
                    {
                        "id": self.self_monitoring.module_id,
                        "type": self.self_monitoring.module_type,
                        "development_level": self.self_monitoring.development_level
                    }
                ]
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown query type: {query_type}"
            }
    
    def get_regulation_state(self) -> SelfRegulationState:
        """Get the current state of the self-regulation system"""
        # Get the current state of each component
        emotional_state = self.emotional_regulation.get_state()
        impulse_state = self.impulse_control.get_state()
        monitoring_state = self.self_monitoring.get_state()
        
        # Calculate overall regulation capacities
        emotional_capacity = self.emotional_regulation.development_level
        impulse_capacity = self.impulse_control.development_level
        monitoring_capacity = self.self_monitoring.development_level
        
        # Get current emotional state and impulse
        current_emotion = None
        if "current_emotion" in emotional_state and emotional_state["current_emotion"]:
            current_emotion = EmotionalState(**emotional_state["current_emotion"])
            
        current_impulse = None
        if "current_impulse" in impulse_state and impulse_state["current_impulse"]:
            current_impulse = ImpulseEvent(**impulse_state["current_impulse"])
        
        # Calculate success rates
        emotional_success_rate = 0.0
        if emotional_state.get("regulation_attempts", 0) > 0:
            emotional_success_rate = emotional_state.get("successful_regulations", 0) / emotional_state.get("regulation_attempts", 1)
            
        impulse_success_rate = 0.0
        if impulse_state.get("control_attempts", 0) > 0:
            impulse_success_rate = impulse_state.get("successful_controls", 0) / impulse_state.get("control_attempts", 1)
            
        # Calculate overall regulation success rate
        total_attempts = emotional_state.get("regulation_attempts", 0) + impulse_state.get("control_attempts", 0)
        total_successes = emotional_state.get("successful_regulations", 0) + impulse_state.get("successful_controls", 0)
        
        regulation_success_rate = 0.5  # Default value
        if total_attempts > 0:
            regulation_success_rate = total_successes / total_attempts
        
        # Create the regulation state
        return SelfRegulationState(
            developmental_level=self.development_level,
            emotional_regulation_capacity=emotional_capacity,
            impulse_control_capacity=impulse_capacity,
            self_monitoring_capacity=monitoring_capacity,
            current_emotional_state=current_emotion,
            current_impulse_state=current_impulse,
            recent_monitoring_events=[],  # Would need to extract from monitoring_state
            available_emotional_strategies=emotional_state.get("available_strategies", []),
            available_impulse_strategies=[],  # Would need to extract from impulse_state
            regulation_success_rate=regulation_success_rate
        )
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "regulate_emotion":
            # Process emotion regulation request
            result = self.process_input({
                "type": "regulation_request",
                "target": "emotion",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="regulation_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "control_impulse":
            # Process impulse control request
            result = self.process_input({
                "type": "regulation_request",
                "target": "impulse",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="control_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "monitor_state":
            # Process monitoring request
            result = self.process_input({
                "type": "monitoring",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="monitor_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "regulation_query":
            # Process query
            result = self.process_input({
                "type": "query",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="regulation_query_response",
                    content=result,
                    reply_to=message.id
                ))
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        
        # Update base module development
        super().update_development(amount)
        
        # Update component development levels
        self.emotional_regulation.update_development(amount)
        self.impulse_control.update_development(amount)
        self.self_monitoring.update_development(amount)
        
        # Update neural controller
        self.neural_controller.update_development(self.development_level - old_level)
        
        # Check for developmental milestones
        self._check_development_milestones(old_level)
        
        self.logger.info(f"Updated self-regulation system development to {self.development_level:.2f}")
        return self.development_level
    
    def _check_development_milestones(self, previous_level: float) -> None:
        """
        Check if any developmental milestones have been reached
        
        Args:
            previous_level: The previous development level
        """
        # Check each milestone to see if we've crossed the threshold
        for level, description in self.development_milestones.items():
            # If we've crossed a milestone threshold
            if previous_level < level <= self.development_level:
                self.logger.info(f"Self-regulation system milestone reached at {level:.1f}: {description}")
                
                # Adjust system capabilities based on the new milestone
                if level == 0.1:
                    self.logger.info("Now capable of basic awareness of internal states")
                elif level == 0.2:
                    self.logger.info("Now capable of simple external regulation strategies")
                elif level == 0.3:
                    self.logger.info("Now capable of emerging basic self-regulation capabilities")
                elif level == 0.4:
                    self.logger.info("Now capable of basic emotional and impulse regulation")
                elif level == 0.5:
                    self.logger.info("Now capable of developing self-monitoring")
                elif level == 0.6:
                    self.logger.info("Now capable of integrated regulation strategies")
                elif level == 0.7:
                    self.logger.info("Now capable of advanced emotional regulation")
                elif level == 0.8:
                    self.logger.info("Now capable of sophisticated impulse control")
                elif level == 0.9:
                    self.logger.info("Now capable of highly effective self-monitoring")
                elif level == 1.0:
                    self.logger.info("Now capable of fully integrated self-regulation system")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with the module state
        """
        base_state = super().get_state()
        
        # Get combined state from all components
        regulation_state = self.get_regulation_state().dict()
        
        return {**base_state, **regulation_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save component states
        self.emotional_regulation.save_state(state_dir)
        self.impulse_control.save_state(state_dir)
        self.self_monitoring.save_state(state_dir)
        
        # Save system state
        state_path = os.path.join(module_dir, "system_state.json")
        with open(state_path, 'w') as f:
            state = self.get_state()
            # Remove redundant component states to avoid duplication
            for key in ["current_emotional_state", "current_impulse_state", "recent_monitoring_events"]:
                if key in state:
                    state[key] = None
            import json
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Saved self-regulation system state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load system state
            import json
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update base state
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Update neural controller
            self.neural_controller.update_development(0)  # Just to sync with new level
            
            # Find and load component states
            state_dir = os.path.dirname(state_path)
            parent_dir = os.path.dirname(state_dir)
            
            # Component paths
            er_path = os.path.join(parent_dir, "emotional_regulation", 
                                 f"{self.module_id}.emotional_regulation", "module_state.json")
            ic_path = os.path.join(parent_dir, "impulse_control", 
                                 f"{self.module_id}.impulse_control", "module_state.json")
            sm_path = os.path.join(parent_dir, "self_monitoring", 
                                 f"{self.module_id}.self_monitoring", "module_state.json")
            
            # Load component states if they exist
            if os.path.exists(er_path):
                self.emotional_regulation.load_state(er_path)
            if os.path.exists(ic_path):
                self.impulse_control.load_state(ic_path)
            if os.path.exists(sm_path):
                self.self_monitoring.load_state(sm_path)
                
            self.logger.info(f"Loaded self-regulation system state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False 
