# TODO: Implement the Inhibition class to suppress inappropriate actions and thoughts
# This component should be able to:
# - Block prepotent but inappropriate responses
# - Filter out irrelevant or distracting information
# - Delay gratification for better long-term outcomes
# - Maintain focus despite competing demands

# TODO: Implement developmental progression in inhibition:
# - Minimal inhibitory control in early stages
# - Growing ability to delay responses in childhood
# - Improved resistance to distractions in adolescence
# - Sophisticated self-control in adulthood

# TODO: Create mechanisms for:
# - Response inhibition: Stop inappropriate actions
# - Interference control: Resist distractions
# - Delayed gratification: Wait for better rewards
# - Thought suppression: Control unwanted thoughts

# TODO: Implement resource modeling for inhibition:
# - Limited inhibitory resources that can be depleted
# - Recovery of inhibitory capacity over time
# - Factors affecting inhibitory strength (motivation, stress)
# - Individual differences in inhibitory capacity

# TODO: Connect to attention and emotion systems
# Inhibition should interact with attention for filtering
# and with emotion for emotional regulation

import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import torch
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.executive.models import InhibitionEvent, InhibitionState, ExecutiveNeuralState
from lmm_project.modules.executive.neural_net import InhibitionNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class Inhibition(BaseModule):
    """
    Suppresses inappropriate actions and thoughts
    
    This module provides control over behavior and cognition,
    blocking impulses and filtering information as needed.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic impulse control",
        0.2: "Simple distraction resistance",
        0.4: "Response inhibition",
        0.6: "Improved interference control",
        0.8: "Self-regulation strategies",
        1.0: "Sophisticated inhibitory control"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the inhibition module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="inhibition", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device
        self.device = get_device()
        
        # Initialize neural network
        self.inhibition_network = InhibitionNetwork(
            input_dim=128,
            hidden_dim=128
        ).to(self.device)
        
        # Set development level for network
        self.inhibition_network.set_development_level(development_level)
        
        # Create neural state for tracking
        self.neural_state = ExecutiveNeuralState()
        self.neural_state.inhibition_development = development_level
        
        # Initialize inhibition state
        self.inhibition_state = InhibitionState(
            available_resources=1.0,
            recovery_rate=0.1,
            threshold_adjustments={},
            last_updated=datetime.now()
        )
        
        # Last resource update timestamp
        self.last_resource_update = time.time()
        
        # Inhibition parameters
        self.params = {
            "base_inhibition_threshold": 0.5,  # Base threshold for inhibiting
            "resource_depletion_rate": 0.2,  # How quickly resources deplete
            "resource_recovery_rate": 0.1,  # How quickly resources recover
            "context_sensitivity": 0.3,  # How much context affects threshold
            "max_simultaneous_inhibitions": 1,  # Maximum number of simultaneous inhibitions
            "inhibition_duration": 5.0  # How long inhibition effect lasts (seconds)
        }
        
        # Update parameters based on development
        self._adjust_parameters_for_development()
        
        logger.info(f"Inhibition module initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust inhibition parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very basic inhibition at early stages
            self.params.update({
                "base_inhibition_threshold": 0.7,  # Higher threshold (less inhibition)
                "resource_depletion_rate": 0.3,  # Faster depletion
                "resource_recovery_rate": 0.05,  # Slower recovery
                "context_sensitivity": 0.1,  # Low context sensitivity
                "max_simultaneous_inhibitions": 1,
                "inhibition_duration": 3.0
            })
            
            # Update state parameters
            self.inhibition_state.recovery_rate = self.params["resource_recovery_rate"]
            
        elif self.development_level < 0.4:
            # Developing basic inhibition
            self.params.update({
                "base_inhibition_threshold": 0.6,
                "resource_depletion_rate": 0.25,
                "resource_recovery_rate": 0.07,
                "context_sensitivity": 0.2,
                "max_simultaneous_inhibitions": 1,
                "inhibition_duration": 4.0
            })
            
            # Update state parameters
            self.inhibition_state.recovery_rate = self.params["resource_recovery_rate"]
            
        elif self.development_level < 0.6:
            # Response inhibition development
            self.params.update({
                "base_inhibition_threshold": 0.5,
                "resource_depletion_rate": 0.2,
                "resource_recovery_rate": 0.1,
                "context_sensitivity": 0.3,
                "max_simultaneous_inhibitions": 2,
                "inhibition_duration": 5.0
            })
            
            # Update state parameters
            self.inhibition_state.recovery_rate = self.params["resource_recovery_rate"]
            
        elif self.development_level < 0.8:
            # Improved interference control
            self.params.update({
                "base_inhibition_threshold": 0.4,
                "resource_depletion_rate": 0.15,
                "resource_recovery_rate": 0.12,
                "context_sensitivity": 0.4,
                "max_simultaneous_inhibitions": 3,
                "inhibition_duration": 6.0
            })
            
            # Update state parameters
            self.inhibition_state.recovery_rate = self.params["resource_recovery_rate"]
            
        else:
            # Sophisticated inhibitory control
            self.params.update({
                "base_inhibition_threshold": 0.3,
                "resource_depletion_rate": 0.1,
                "resource_recovery_rate": 0.15,
                "context_sensitivity": 0.5,
                "max_simultaneous_inhibitions": 4,
                "inhibition_duration": 8.0
            })
            
            # Update state parameters
            self.inhibition_state.recovery_rate = self.params["resource_recovery_rate"]
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to apply inhibitory control
        
        Args:
            input_data: Dictionary containing stimulus and context information
                Required keys:
                - 'stimulus': Information about what might need to be inhibited
                Optional keys:
                - 'context': Contextual information affecting inhibition
                - 'type': Type of inhibition ('response', 'distraction', 'thought')
                - 'urgency': How urgent the inhibition decision is (0-1)
                - 'operation': Specific operation ('inhibit', 'query', 'recover')
            
        Returns:
            Dictionary with the results of inhibition
        """
        # Update resource recovery
        self._update_resources()
        
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        operation = input_data.get("operation", "inhibit")
        
        # Different operations based on the request
        if operation == "inhibit":
            return self._apply_inhibition(input_data, process_id)
        elif operation == "query":
            return self._query_state(input_data, process_id)
        elif operation == "recover":
            return self._force_recovery(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _apply_inhibition(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Apply inhibitory control to a stimulus"""
        # Extract required data
        if "stimulus" not in input_data:
            return {"status": "error", "message": "No stimulus provided", "process_id": process_id}
        
        stimulus = input_data.get("stimulus", {})
        context = input_data.get("context", {})
        inhibition_type = input_data.get("type", "response")
        
        # Check resource availability
        if self.inhibition_state.available_resources < 0.1:
            return {
                "status": "insufficient_resources",
                "inhibit_success": False,
                "available_resources": self.inhibition_state.available_resources,
                "message": "Insufficient inhibitory resources available",
                "process_id": process_id
            }
        
        # Convert stimulus and context to tensors for neural processing
        stimulus_features = self._extract_features(stimulus)
        context_features = self._extract_features(context)
        
        # Process through neural network
        with torch.no_grad():
            inhibition_result = self.inhibition_network(
                stimulus=stimulus_features.to(self.device),
                context=context_features.to(self.device)
            )
        
        # Extract results
        inhibit_probability = inhibition_result["inhibit_probability"].cpu().item()
        inhibition_strength = inhibition_result["inhibition_strength"].cpu().item()
        resource_cost = inhibition_result["resource_cost"].cpu().item()
        
        # Record activation in neural state
        self.neural_state.add_activation('inhibition', {
            'inhibit_probability': inhibit_probability,
            'inhibition_strength': inhibition_strength,
            'resource_cost': resource_cost,
            'inhibition_type': inhibition_type
        })
        
        # Apply context-specific threshold adjustments
        threshold = self.params["base_inhibition_threshold"]
        
        # Check for context-specific adjustments
        context_key = str(hash(str(sorted(context.items() if isinstance(context, dict) else context))))
        if context_key in self.inhibition_state.threshold_adjustments:
            threshold_adjustment = self.inhibition_state.threshold_adjustments[context_key]
            threshold = max(0.1, min(0.9, threshold + threshold_adjustment))
        
        # Determine whether to inhibit
        should_inhibit = inhibit_probability > threshold
        
        # Create inhibition event regardless of decision (for records)
        event = InhibitionEvent(
            inhibition_type=inhibition_type,
            trigger=str(stimulus)[:100],  # Truncate long stimuli
            strength=inhibition_strength,
            success=should_inhibit,
            resource_cost=resource_cost,
            context={k: str(v)[:50] for k, v in context.items()} if isinstance(context, dict) else {"data": str(context)[:50]}
        )
        
        # Add to history in state
        self.inhibition_state.recent_events.append(event)
        if len(self.inhibition_state.recent_events) > 20:
            self.inhibition_state.recent_events = self.inhibition_state.recent_events[-20:]
        
        # Update resources if inhibition was applied
        if should_inhibit:
            # Deplete resources based on cost
            self.inhibition_state.available_resources = max(
                0.0, 
                self.inhibition_state.available_resources - resource_cost
            )
            
            # Update state timestamp
            self.inhibition_state.last_updated = datetime.now()
            
            # Learn from this inhibition for future similar contexts
            if self.development_level >= 0.4:
                # Higher development enables learning from inhibition events
                if context_key not in self.inhibition_state.threshold_adjustments:
                    # Initialize adjustment
                    self.inhibition_state.threshold_adjustments[context_key] = 0.0
                
                # Adjust threshold slightly (strengthen or weaken based on outcome)
                # This simple learning rule could be replaced with more sophisticated approaches
                self.inhibition_state.threshold_adjustments[context_key] -= 0.01
                
                # Limit adjustment range
                self.inhibition_state.threshold_adjustments[context_key] = max(
                    -0.3, min(0.3, self.inhibition_state.threshold_adjustments[context_key])
                )
        
        return {
            "status": "success",
            "inhibit_decision": should_inhibit,
            "inhibition_strength": inhibition_strength,
            "available_resources": self.inhibition_state.available_resources,
            "resource_cost": resource_cost,
            "inhibition_event": event.dict(),
            "process_id": process_id
        }
    
    def _query_state(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Query the current inhibition state"""
        # Update resources first
        self._update_resources()
        
        query_type = input_data.get("query_type", "resources")
        
        if query_type == "resources":
            return {
                "status": "success",
                "available_resources": self.inhibition_state.available_resources,
                "recovery_rate": self.inhibition_state.recovery_rate,
                "process_id": process_id
            }
        elif query_type == "events":
            # Get recent events, optionally filtered by type
            event_type = input_data.get("event_type", None)
            events = self.inhibition_state.recent_events
            
            if event_type:
                events = [e for e in events if e.inhibition_type == event_type]
            
            return {
                "status": "success",
                "events": [e.dict() for e in events],
                "event_count": len(events),
                "process_id": process_id
            }
        elif query_type == "threshold_adjustments":
            return {
                "status": "success",
                "threshold_adjustments": self.inhibition_state.threshold_adjustments,
                "base_threshold": self.params["base_inhibition_threshold"],
                "process_id": process_id
            }
        else:
            # Full state
            return {
                "status": "success",
                "inhibition_state": self.inhibition_state.dict(),
                "inhibition_params": self.params,
                "development_level": self.development_level,
                "process_id": process_id
            }
    
    def _force_recovery(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Force recovery of inhibitory resources"""
        amount = input_data.get("amount", 0.5)
        
        # Cap the amount based on development level
        max_recovery = 0.2 + (0.8 * self.development_level)
        actual_amount = min(amount, max_recovery)
        
        # Apply recovery
        old_resources = self.inhibition_state.available_resources
        self.inhibition_state.available_resources = min(
            1.0, self.inhibition_state.available_resources + actual_amount
        )
        
        # Update timestamp
        self.inhibition_state.last_updated = datetime.now()
        self.last_resource_update = time.time()
        
        return {
            "status": "success",
            "previous_resources": old_resources,
            "new_resources": self.inhibition_state.available_resources,
            "recovery_amount": actual_amount,
            "process_id": process_id
        }
    
    def _update_resources(self):
        """Update resource recovery based on elapsed time"""
        current_time = time.time()
        elapsed_seconds = current_time - self.last_resource_update
        
        if elapsed_seconds > 0.1:  # Only update if enough time has passed
            # Calculate recovery amount
            recovery_amount = self.inhibition_state.recovery_rate * elapsed_seconds
            
            # Apply recovery up to maximum
            self.inhibition_state.available_resources = min(
                1.0, self.inhibition_state.available_resources + recovery_amount
            )
            
            # Update timestamp
            self.last_resource_update = current_time
            
            # Update state timestamp if resources changed significantly
            if recovery_amount > 0.01:
                self.inhibition_state.last_updated = datetime.now()
    
    def _extract_features(self, data) -> torch.Tensor:
        """
        Extract features from input data for neural processing
        
        Args:
            data: Text, dict, or other data to extract features from
            
        Returns:
            Tensor of features [1, feature_dim]
        """
        # For demonstration, create simple random features
        # In a real implementation, this would use proper feature extraction
        feature_dim = 64
        
        if isinstance(data, str):
            # Seed random generator with hash of string to ensure consistent features
            seed = hash(data) % 10000
            np.random.seed(seed)
            
            # Generate "features" based on the text
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        elif isinstance(data, dict):
            # For dictionary data, use keys and values to generate features
            seed = hash(str(sorted(data.items()))) % 10000
            np.random.seed(seed)
            
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        else:
            # Default random features
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update network development level
        self.inhibition_network.set_development_level(new_level)
        
        # Update neural state
        self.neural_state.inhibition_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # Adjust parameters based on new development level
        self._adjust_parameters_for_development()
        
        logger.info(f"Inhibition module development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Update resources first
        self._update_resources()
        
        # Get base state from parent
        base_state = super().get_state()
        
        # Add inhibition-specific state
        inhibition_state_dict = self.inhibition_state.dict()
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.inhibition_development,
            "accuracy": self.neural_state.inhibition_accuracy,
            "recent_activations_count": len(self.neural_state.recent_inhibition_activations)
        }
        
        # Combine states
        combined_state = {
            **base_state, 
            **inhibition_state_dict, 
            "params": self.params,
            "neural_state": neural_state
        }
        
        return combined_state
