"""
Neuron implementation for the neural substrate.

This module defines the Neuron class, which is the basic processing unit
in the LMM neural substrate. Neurons receive inputs, apply activation
functions, and produce outputs.
"""
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from lmm_project.neural_substrate.activation_functions import (
    ActivationType, get_activation_function
)
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("neural_substrate.neuron")


class NeuronConfig(BaseModel):
    """Configuration for a neuron."""
    activation_type: ActivationType = Field(default=ActivationType.SIGMOID)
    bias: float = Field(default=0.0)
    threshold: float = Field(default=0.5)
    learning_rate: float = Field(default=0.01)
    is_inhibitory: bool = Field(default=False)
    refractory_period: float = Field(default=0.0)  # In milliseconds
    adaptation_rate: float = Field(default=0.0)
    noise_level: float = Field(default=0.0)
    leak_rate: float = Field(default=0.1)


class Neuron:
    """
    Basic processing unit in the neural substrate.
    
    Neurons receive inputs from other neurons, process them using
    an activation function, and produce an output that can be sent
    to other neurons.
    """
    
    def __init__(
        self,
        neuron_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a neuron.
        
        Parameters:
        neuron_id: Unique identifier for this neuron
        config: Configuration dictionary
        device: Torch device for tensor operations
        """
        self.neuron_id = neuron_id or str(uuid.uuid4())
        self.config = NeuronConfig(**(config or {}))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get activation function and its derivative
        self.activation_fn, self.activation_derivative = get_activation_function(
            self.config.activation_type
        )
        
        # Initialize state
        self.bias = torch.tensor(self.config.bias, device=self.device)
        self.activation = 0.0
        self.last_activation = 0.0
        self.threshold = self.config.threshold
        
        # Learning and adaptation parameters
        self.learning_rate = self.config.learning_rate
        self.adaptation = 0.0
        self.adaptation_rate = self.config.adaptation_rate
        
        # Timing and refractory properties
        self.last_spike_time = 0.0
        self.refractory_period = self.config.refractory_period
        
        # Tracking input connections
        self.input_connections: Set[str] = set()
        
        # Input buffers for incoming signals
        self.input_buffer: Dict[str, float] = {}
        
        # Track learning history
        self.activation_history: List[float] = []
        self.max_history_size = 100
    
    def add_input_connection(self, source_id: str) -> None:
        """
        Register an input connection from another neuron.
        
        Parameters:
        source_id: ID of the source neuron
        """
        self.input_connections.add(source_id)
        self.input_buffer[source_id] = 0.0
    
    def remove_input_connection(self, source_id: str) -> None:
        """
        Remove an input connection.
        
        Parameters:
        source_id: ID of the source neuron
        """
        if source_id in self.input_connections:
            self.input_connections.remove(source_id)
            del self.input_buffer[source_id]
    
    def receive_input(self, source_id: str, value: float, weight: float) -> None:
        """
        Receive input from a connected neuron.
        
        Parameters:
        source_id: ID of the source neuron
        value: Activation value from source neuron
        weight: Connection weight
        """
        # Apply weight to the incoming signal
        weighted_input = value * weight
        
        # If inhibitory connection, negate the signal
        if self.config.is_inhibitory:
            weighted_input = -abs(weighted_input)
            
        # Store in input buffer
        self.input_buffer[source_id] = weighted_input
    
    def compute_activation(self, current_time: float = 0.0) -> float:
        """
        Compute neuron activation based on current inputs and state.
        
        Parameters:
        current_time: Current simulation time in milliseconds
        
        Returns:
        New activation value
        """
        # Check if neuron is in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.activation = 0.0
            return self.activation
        
        # Store previous activation for learning
        self.last_activation = self.activation
        
        # Sum all inputs with bias
        total_input = sum(self.input_buffer.values()) + self.bias.item()
        
        # Apply adaptation
        total_input -= self.adaptation
        
        # Add noise if configured
        if self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level)
            total_input += noise
        
        # Apply activation function
        self.activation = float(self.activation_fn(total_input))
        
        # Update adaptation
        if self.adaptation_rate > 0:
            self.adaptation += self.adaptation_rate * self.activation
            self.adaptation *= (1.0 - self.config.leak_rate)  # Leaky adaptation
        
        # Track if neuron spiked (exceeded threshold)
        if self.activation >= self.threshold:
            self.last_spike_time = current_time
            
        # Track activation history
        self.activation_history.append(self.activation)
        if len(self.activation_history) > self.max_history_size:
            self.activation_history = self.activation_history[-self.max_history_size:]
        
        return self.activation
    
    def adapt(self, factor: float) -> None:
        """
        Apply homeostatic adaptation to the neuron.
        
        Parameters:
        factor: Adaptation factor
        """
        self.adaptation += factor * self.adaptation_rate
    
    def update_bias(self, error: float) -> None:
        """
        Update the bias based on error signal.
        
        Parameters:
        error: Error signal for learning
        """
        # Calculate gradient using activation derivative
        input_sum = sum(self.input_buffer.values())
        gradient = error * self.activation_derivative(input_sum + self.bias.item())
        
        # Update bias using gradient descent
        with torch.no_grad():
            self.bias -= self.learning_rate * gradient
    
    def reset_state(self) -> None:
        """Reset the neuron's state."""
        self.activation = 0.0
        self.last_activation = 0.0
        self.adaptation = 0.0
        self.last_spike_time = 0.0
        self.input_buffer = {source_id: 0.0 for source_id in self.input_connections}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the neuron.
        
        Returns:
        Dictionary containing the neuron state
        """
        return {
            "neuron_id": self.neuron_id,
            "activation": self.activation,
            "bias": self.bias.item(),
            "is_inhibitory": self.config.is_inhibitory,
            "input_connections": list(self.input_connections),
            "activation_type": self.config.activation_type.name,
            "adaptation": self.adaptation,
            "last_spike_time": self.last_spike_time
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load neuron state from a dictionary.
        
        Parameters:
        state: Dictionary containing neuron state
        """
        if "activation" in state:
            self.activation = state["activation"]
        if "bias" in state:
            self.bias = torch.tensor(state["bias"], device=self.device)
        if "adaptation" in state:
            self.adaptation = state["adaptation"]
        if "last_spike_time" in state:
            self.last_spike_time = state["last_spike_time"]
