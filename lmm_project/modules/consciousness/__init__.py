# Consciousness module for LMM
# Integrated system responsible for awareness, self-reflection, and
# the formation of a coherent subjective experience

from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
import numpy as np
import os
import json

from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.base_module import BaseModule
from lmm_project.modules.consciousness.awareness import Awareness
from lmm_project.modules.consciousness.global_workspace import GlobalWorkspace
from lmm_project.modules.consciousness.self_model import SelfModel
from lmm_project.modules.consciousness.introspection import Introspection
from lmm_project.modules.consciousness.models import ConsciousnessState
from lmm_project.modules.consciousness.neural_net import ConsciousnessNetwork

def get_module(
    module_id: str = "consciousness",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "ConsciousnessSystem":
    """
    Factory function to create a consciousness module.
    
    The consciousness system is responsible for:
    - Maintaining awareness of internal and external states
    - Providing a global workspace for information sharing
    - Developing and maintaining a self-model
    - Enabling introspection and self-reflection
    
    Args:
        module_id: Unique identifier for this module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level
        
    Returns:
        An instance of the ConsciousnessSystem class
    """
    return ConsciousnessSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class ConsciousnessSystem(BaseModule):
    """
    Integrated consciousness system that combines awareness, global workspace,
    self-model, and introspection capabilities.
    
    This system enables the emergence of consciousness through the integration
    of multiple submodules, each responsible for a different aspect of conscious
    experience.
    
    The system develops from basic awareness in early stages to sophisticated
    self-reflection and metacognition in later stages.
    """
    
    # Developmental milestones for the consciousness system
    development_milestones = {
        0.0: "basic_awareness",         # Basic awareness of stimuli
        0.2: "working_memory",          # Simple working memory
        0.4: "self_recognition",        # Recognition of self as entity
        0.6: "metacognition",           # Thinking about thinking
        0.8: "reflective_consciousness", # Full reflective capabilities
        0.95: "integrated_consciousness" # Fully integrated consciousness
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the consciousness system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(module_id=module_id, module_type="consciousness", event_bus=event_bus)
        
        # Set development level
        self._set_development_level(development_level)
        
        # Initialize state
        self.state = ConsciousnessState()
        
        # Create submodules
        self.awareness = Awareness(f"{module_id}_awareness", event_bus)
        self.global_workspace = GlobalWorkspace(f"{module_id}_workspace", event_bus)
        self.self_model = SelfModel(f"{module_id}_self", event_bus)
        self.introspection = Introspection(f"{module_id}_introspection", event_bus)
        
        # Initialize neural network
        self.network = ConsciousnessNetwork()
        
        # Synchronize development levels
        self._sync_development_levels()
        
        # Initialize communication between submodules
        self._initialize_submodule_communication()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.event_bus.subscribe("consciousness_query", self._handle_query)
            self.event_bus.subscribe("development_update", self._handle_development)
            self.event_bus.subscribe("save_state", self._handle_save_state)
            self.event_bus.subscribe("load_state", self._handle_load_state)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for the consciousness system
        
        Args:
            input_data: Dictionary containing input data for processing
            
        Returns:
            Dictionary with the results of consciousness processing
        """
        # Extract input information
        input_type = input_data.get("type", "unknown")
        content = input_data.get("content", {})
        source = input_data.get("source", "unknown")
        
        results = {}
        
        # Process different types of inputs
        if input_type == "perception":
            # Route to awareness
            awareness_result = self.awareness.process_input({
                "type": "perception",
                "state": content,
                "source": source
            })
            results["awareness"] = awareness_result
            
        elif input_type == "cognitive_state":
            # Route to both awareness and global workspace
            awareness_result = self.awareness.process_input({
                "type": "cognitive",
                "state": content,
                "source": source
            })
            
            workspace_result = self.global_workspace.process_input({
                "content": content,
                "source": source,
                "activation": content.get("importance", 0.5)
            })
            
            results["awareness"] = awareness_result
            results["global_workspace"] = workspace_result
            
        elif input_type == "self_update":
            # Route to self-model
            self_result = self.self_model.process_input({
                "type": content.get("update_type", "identity_update"),
                "data": content.get("data", {})
            })
            
            results["self_model"] = self_result
            
        elif input_type == "introspection_request":
            # Route to introspection
            introspection_result = self.introspection.process_input({
                "type": content.get("introspection_type", "general"),
                "mental_state": content.get("mental_state", {})
            })
            
            results["introspection"] = introspection_result
            
        elif input_type == "integrated_processing":
            # Perform integrated processing across all submodules
            results = self._integrated_processing(content)
            
        # Update overall consciousness state based on submodule states
        self._update_consciousness_state()
        
        # Add system-level information to results
        results["module_id"] = self.module_id
        results["module_type"] = self.module_type
        results["developmental_level"] = self.developmental_level
        results["current_milestone"] = self._get_current_milestone()
        results["consciousness_level"] = self.state.consciousness_level
        
        return results
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the consciousness system
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.developmental_level
        new_level = super().update_development(amount)
        
        # Update submodule development levels
        self._sync_development_levels()
        
        # Unlock new consciousness capabilities at milestones
        if previous_level < 0.2 and new_level >= 0.2:
            # Enhance global workspace at working memory milestone
            self.global_workspace.update_development(0.05)  # Extra boost
            
        if previous_level < 0.4 and new_level >= 0.4:
            # Enhance self-model at self-recognition milestone
            self.self_model.update_development(0.05)  # Extra boost
            
        if previous_level < 0.6 and new_level >= 0.6:
            # Enhance introspection at metacognition milestone
            self.introspection.update_development(0.05)  # Extra boost
            
        if previous_level < 0.8 and new_level >= 0.8:
            # Enhance awareness at reflective consciousness milestone
            self.awareness.update_development(0.05)  # Extra boost
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the consciousness system"""
        state = {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "developmental_level": self.developmental_level,
            "current_milestone": self._get_current_milestone(),
            "consciousness_state": self.state.model_dump(),
            "submodules": {
                "awareness": self.awareness.get_state() if hasattr(self.awareness, "get_state") else {},
                "global_workspace": self.global_workspace.get_state() if hasattr(self.global_workspace, "get_state") else {},
                "self_model": self.self_model.get_state() if hasattr(self.self_model, "get_state") else {},
                "introspection": self.introspection.get_state() if hasattr(self.introspection, "get_state") else {}
            }
        }
        return state
    
    def _set_development_level(self, level: float) -> None:
        """Set the development level manually"""
        self.developmental_level = max(0.0, min(1.0, level))
    
    def _sync_development_levels(self) -> None:
        """Synchronize development levels across all submodules"""
        level = self.developmental_level
        
        # Apply varying development rates to different submodules
        self.awareness.update_development(level - self.awareness.developmental_level)
        self.global_workspace.update_development(level - self.global_workspace.developmental_level)
        self.self_model.update_development(level - self.self_model.developmental_level)
        self.introspection.update_development(level - self.introspection.developmental_level)
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_conscious"
        for level, name in sorted(self.development_milestones.items()):
            if self.developmental_level >= level:
                milestone = name
        return milestone
    
    def _initialize_submodule_communication(self) -> None:
        """Initialize communication patterns between submodules"""
        # This function would set up direct communication between submodules
        # if they aren't already connected via the event bus
        pass
    
    def _update_consciousness_state(self) -> None:
        """Update the integrated consciousness state based on submodule states"""
        # Update awareness state
        if hasattr(self.awareness, "state"):
            self.state.awareness = self.awareness.state
            
        # Update global workspace state
        if hasattr(self.global_workspace, "state"):
            self.state.global_workspace = self.global_workspace.state
            
        # Update self-model state
        if hasattr(self.self_model, "state"):
            self.state.self_model = self.self_model.state
            
        # Update introspection state
        if hasattr(self.introspection, "state"):
            self.state.introspection = self.introspection.state
            
        # Update last update timestamp
        self.state.last_update = datetime.now()
        
        # Model validator will automatically update consciousness_level
    
    def _integrated_processing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated processing across all consciousness submodules"""
        results = {}
        
        # First, process with awareness to determine what enters consciousness
        awareness_result = self.awareness.process_input({
            "type": content.get("content_type", "unknown"),
            "state": content,
            "source": content.get("source", "integrated_request")
        })
        results["awareness"] = awareness_result
        
        # Next, place in global workspace if it passes awareness
        if awareness_result.get("state", {}).get("external_awareness", 0) > 0.3:
            workspace_result = self.global_workspace.process_input({
                "content": content,
                "source": content.get("source", "integrated_request"),
                "activation": 0.6  # Default activation for integrated processing
            })
            results["global_workspace"] = workspace_result
            
            # Get workspace contents for further processing
            workspace_contents = workspace_result.get("state", {}).get("active_items", {})
            
            # Update self-model based on workspace contents
            if workspace_contents and self.developmental_level >= 0.4:
                self_relevance = content.get("self_relevance", 0.0)
                
                if self_relevance > 0.4:
                    self_result = self.self_model.process_input({
                        "type": "identity_update",
                        "data": {
                            "updates": content.get("self_implications", {})
                        }
                    })
                    results["self_model"] = self_result
            
            # Perform introspection on the process if developed enough
            if self.developmental_level >= 0.6:
                introspection_result = self.introspection.process_input({
                    "type": "cognitive_process",
                    "mental_state": {
                        "process_type": "integrated_consciousness",
                        "state": content,
                        "self_involvement": content.get("self_relevance", 0.0)
                    }
                })
                results["introspection"] = introspection_result
        
        return results
    
    def _handle_query(self, message: Message) -> None:
        """Handle consciousness query messages"""
        query_type = message.content.get("query_type", "")
        
        response = {
            "module_id": self.module_id,
            "query_type": query_type
        }
        
        if query_type == "consciousness_state":
            response["state"] = self.state.model_dump()
            
        elif query_type == "development_level":
            response["developmental_level"] = self.developmental_level
            response["current_milestone"] = self._get_current_milestone()
            
        elif query_type == "self_model":
            response["self_model"] = self.self_model.state.model_dump() if hasattr(self.self_model, "state") else {}
            
        elif query_type == "awareness":
            response["awareness"] = self.awareness.state.model_dump() if hasattr(self.awareness, "state") else {}
            
        # Publish response if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                msg_type="consciousness_response",
                content=response,
                source=self.module_id,
                target=message.source
            )
    
    def _handle_development(self, message: Message) -> None:
        """Handle development update messages"""
        if message.target in [self.module_id, "all"]:
            amount = message.content.get("amount", 0.01)
            self.update_development(amount)
    
    def _handle_save_state(self, message: Message) -> None:
        """Handle save state messages"""
        if message.target in [self.module_id, "all"]:
            path = message.content.get("path", "")
            
            if not path:
                # Default path
                path = os.path.join("data", "states", f"{self.module_id}_state.json")
                
            # Create directories if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Get current state
            state = self.get_state()
            
            # Save to file
            try:
                with open(path, 'w') as f:
                    json.dump(state, f, indent=2)
                    
                # Publish success message
                if self.event_bus:
                    self.event_bus.publish(
                        msg_type="state_saved",
                        content={
                            "module_id": self.module_id,
                            "path": path
                        }
                    )
            except Exception as e:
                # Publish error message
                if self.event_bus:
                    self.event_bus.publish(
                        msg_type="save_error",
                        content={
                            "module_id": self.module_id,
                            "error": str(e)
                        }
                    )
    
    def _handle_load_state(self, message: Message) -> None:
        """Handle load state messages"""
        if message.target in [self.module_id, "all"]:
            path = message.content.get("path", "")
            
            if not path:
                # Default path
                path = os.path.join("data", "states", f"{self.module_id}_state.json")
                
            # Load from file
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        state = json.load(f)
                        
                    # Apply development level
                    if "developmental_level" in state:
                        self._set_development_level(state["developmental_level"])
                        self._sync_development_levels()
                        
                    # Publish success message
                    if self.event_bus:
                        self.event_bus.publish(
                            msg_type="state_loaded",
                            content={
                                "module_id": self.module_id,
                                "path": path
                            }
                        )
                else:
                    # Publish error message
                    if self.event_bus:
                        self.event_bus.publish(
                            msg_type="load_error",
                            content={
                                "module_id": self.module_id,
                                "error": f"File not found: {path}"
                            }
                        )
            except Exception as e:
                # Publish error message
                if self.event_bus:
                    self.event_bus.publish(
                        msg_type="load_error",
                        content={
                            "module_id": self.module_id,
                            "error": str(e)
                        }
                    )
