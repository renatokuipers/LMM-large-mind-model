"""
Synapse implementation for the neural substrate.

This module defines the Synapse class, which represents connections between neurons
in the LMM neural substrate. Synapses have modifiable weights and implement
various forms of plasticity.
"""
import uuid
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("neural_substrate.synapse")


class PlasticityType(Enum):
    """Types of synaptic plasticity available in the system."""
    NONE = auto()
    HEBBIAN = auto()
    ANTI_HEBBIAN = auto()
    OJA = auto()
    STDP = auto()  # Spike-timing-dependent plasticity
    HOMEOSTATIC = auto()


class SynapseConfig(BaseModel):
    """Configuration for a synapse."""
    initial_weight: float = Field(default=0.1, ge=0.0)
    learning_rate: float = Field(default=0.01, ge=0.0)
    min_weight: float = Field(default=0.0)
    max_weight: float = Field(default=1.0)
    plasticity_type: PlasticityType = Field(default=PlasticityType.HEBBIAN)
    plasticity_enabled: bool = Field(default=True)
    decay_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    stdp_window: float = Field(default=20.0)  # Milliseconds for STDP window
    weight_noise: float = Field(default=0.0, ge=0.0)  # Random weight noise
    eligibility_trace_decay: float = Field(default=0.9, ge=0.0, le=1.0)


class Synapse:
    """
    Connection between neurons with modifiable weights.
    
    Synapses transmit signals between neurons and modify their weights
    based on the chosen plasticity mechanism, allowing the network to learn.
    """
    
    def __init__(
        self,
        pre_neuron_id: str,
        post_neuron_id: str,
        synapse_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a synapse between two neurons.
        
        Parameters:
        pre_neuron_id: ID of the presynaptic neuron (source)
        post_neuron_id: ID of the postsynaptic neuron (target)
        synapse_id: Unique identifier for this synapse
        config: Configuration dictionary
        device: Torch device for tensor operations
        """
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.synapse_id = synapse_id or str(uuid.uuid4())
        self.config = SynapseConfig(**(config or {}))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize weight as a tensor for gradient-based learning
        self.weight = torch.tensor(
            self.config.initial_weight, 
            device=self.device, 
            requires_grad=True
        )
        
        # Add noise to initial weight if configured
        if self.config.weight_noise > 0:
            with torch.no_grad():
                noise = torch.normal(
                    mean=0.0, 
                    std=self.config.weight_noise, 
                    size=(1,), 
                    device=self.device
                )
                self.weight += noise
            
        # Ensure weight is within bounds
        self._constrain_weight()
        
        # Learning parameters
        self.learning_rate = self.config.learning_rate
        self.eligibility_trace = 0.0
        
        # Timing variables for STDP
        self.last_pre_spike_time = 0.0
        self.last_post_spike_time = 0.0
        
        # Activity variables for plasticity
        self.pre_activity = 0.0
        self.post_activity = 0.0
        
        # Usage statistics
        self.activation_count = 0
        self.created_time = 0.0
        self.last_update_time = 0.0
    
    def transmit(
        self, 
        pre_activation: float, 
        current_time: float = 0.0
    ) -> float:
        """
        Transmit the presynaptic neuron's activation through this synapse.
        
        Parameters:
        pre_activation: Activation level of the presynaptic neuron
        current_time: Current simulation time
        
        Returns:
        Weighted input for the postsynaptic neuron
        """
        # Store the presynaptic activity for learning
        self.pre_activity = pre_activation
        
        # Apply weight with optional decay
        self._apply_weight_decay()
        
        # Track timing for STDP if activation is above threshold (spike)
        if pre_activation > 0.5:  # Simple threshold for spike detection
            self.last_pre_spike_time = current_time
            self.activation_count += 1
        
        # Update the last usage time
        self.last_update_time = current_time
        
        # Get the final weight and apply it to activation
        weighted_activation = pre_activation * self.weight.item()
        
        return weighted_activation
    
    def update_weight(
        self, 
        post_activation: float, 
        current_time: float = 0.0, 
        error_signal: Optional[float] = None
    ) -> None:
        """
        Update the synapse weight based on pre/post activities and plasticity type.
        
        Parameters:
        post_activation: Activation level of the postsynaptic neuron
        current_time: Current simulation time
        error_signal: Optional error signal for supervised learning
        """
        if not self.config.plasticity_enabled:
            return
            
        # Store the postsynaptic activity for learning
        self.post_activity = post_activation
        
        # Track timing for STDP if activation is above threshold (spike)
        if post_activation > 0.5:  # Simple threshold for spike detection
            self.last_post_spike_time = current_time
        
        # Determine weight update based on plasticity type
        weight_change = 0.0
        
        if self.config.plasticity_type == PlasticityType.HEBBIAN:
            # Standard Hebbian: weight += lr * pre * post
            weight_change = self.learning_rate * self.pre_activity * self.post_activity
            
        elif self.config.plasticity_type == PlasticityType.ANTI_HEBBIAN:
            # Anti-Hebbian: weight -= lr * pre * post
            weight_change = -self.learning_rate * self.pre_activity * self.post_activity
            
        elif self.config.plasticity_type == PlasticityType.OJA:
            # Oja's rule: weight += lr * post * (pre - post * weight)
            weight_change = self.learning_rate * self.post_activity * (
                self.pre_activity - self.post_activity * self.weight.item()
            )
            
        elif self.config.plasticity_type == PlasticityType.STDP:
            # Spike-Timing-Dependent Plasticity
            weight_change = self._compute_stdp_change(current_time)
            
        elif self.config.plasticity_type == PlasticityType.HOMEOSTATIC:
            # Homeostatic plasticity: maintain target postsynaptic activity
            target_activity = 0.5  # Target activity level
            weight_change = self.learning_rate * self.pre_activity * (
                target_activity - self.post_activity
            )
        
        # If error signal is provided (supervised learning), use it
        if error_signal is not None:
            weight_change = self.learning_rate * error_signal * self.pre_activity
        
        # Update eligibility trace
        self.eligibility_trace = (self.config.eligibility_trace_decay * self.eligibility_trace +
                                 (1 - self.config.eligibility_trace_decay) * weight_change)
        
        # Apply the weight change with eligibility trace modulation
        with torch.no_grad():
            self.weight += self.eligibility_trace
            
        # Constrain weight within bounds
        self._constrain_weight()
    
    def _compute_stdp_change(self, current_time: float) -> float:
        """
        Compute weight change based on spike-timing-dependent plasticity.
        
        Parameters:
        current_time: Current simulation time
        
        Returns:
        Weight change amount
        """
        # If either spike hasn't happened, no change
        if self.last_pre_spike_time == 0 or self.last_post_spike_time == 0:
            return 0.0
        
        # Calculate time difference between pre and post spikes
        time_diff = self.last_post_spike_time - self.last_pre_spike_time
        
        # STDP window (typically ~20ms)
        window = self.config.stdp_window
        
        # If post spike follows pre spike (within window), strengthen
        # If pre spike follows post spike (within window), weaken
        if abs(time_diff) > window:
            return 0.0  # Outside STDP window
        elif time_diff > 0:  # Post after pre (LTP - strengthen)
            # Exponential decay with time difference
            return self.learning_rate * np.exp(-time_diff / window)
        else:  # Pre after post (LTD - weaken)
            # Exponential decay with time difference (negative change)
            return -self.learning_rate * np.exp(time_diff / window)
    
    def _apply_weight_decay(self) -> None:
        """Apply weight decay if configured."""
        if self.config.decay_rate > 0:
            with torch.no_grad():
                self.weight *= (1.0 - self.config.decay_rate)
    
    def _constrain_weight(self) -> None:
        """Ensure the weight stays within the configured min/max bounds."""
        with torch.no_grad():
            self.weight.clamp_(self.config.min_weight, self.config.max_weight)
    
    def reset(self) -> None:
        """Reset the synapse to its initial state."""
        with torch.no_grad():
            self.weight = torch.tensor(
                self.config.initial_weight, 
                device=self.device,
                requires_grad=True
            )
        self.eligibility_trace = 0.0
        self.pre_activity = 0.0
        self.post_activity = 0.0
        self.last_pre_spike_time = 0.0
        self.last_post_spike_time = 0.0
        self.activation_count = 0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the synapse.
        
        Returns:
        Dictionary containing the synapse state
        """
        return {
            "synapse_id": self.synapse_id,
            "pre_neuron_id": self.pre_neuron_id,
            "post_neuron_id": self.post_neuron_id,
            "weight": self.weight.item(),
            "plasticity_type": self.config.plasticity_type.name,
            "plasticity_enabled": self.config.plasticity_enabled,
            "learning_rate": self.learning_rate,
            "eligibility_trace": self.eligibility_trace,
            "activation_count": self.activation_count,
            "last_update_time": self.last_update_time
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load synapse state from a dictionary.
        
        Parameters:
        state: Dictionary containing synapse state
        """
        if "weight" in state:
            with torch.no_grad():
                self.weight = torch.tensor(state["weight"], device=self.device, requires_grad=True)
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
        if "eligibility_trace" in state:
            self.eligibility_trace = state["eligibility_trace"]
        if "activation_count" in state:
            self.activation_count = state["activation_count"]
        if "last_update_time" in state:
            self.last_update_time = state["last_update_time"]
        if "plasticity_enabled" in state:
            self.config.plasticity_enabled = state["plasticity_enabled"]
