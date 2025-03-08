from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
import uuid
import numpy as np
import math
from datetime import datetime

from .activation_functions import get_activation_function

class Neuron(BaseModel):
    """
    Basic neuron implementation for the neural substrate
    
    This represents a single neuron in the neural network, with
    activation state, connections, and learning capability.
    """
    neuron_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_function: str = "sigmoid"
    activation_threshold: float = Field(default=0.5)
    refractory_period: float = Field(default=0.0)
    last_fired: Optional[datetime] = None
    connections: Dict[str, float] = Field(default_factory=dict)
    bias: float = Field(default=0.0)
    learning_rate: float = Field(default=0.01)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def activate(self, input_value: float) -> float:
        """
        Activate the neuron with the given input
        
        Parameters:
        input_value: Input value to the neuron
        
        Returns:
        Activation level after processing input
        """
        # Check if neuron is in refractory period
        if self.last_fired:
            time_since_fired = (datetime.now() - self.last_fired).total_seconds()
            if time_since_fired < self.refractory_period:
                return self.activation
        
        # Apply activation function
        activation_func = get_activation_function(self.activation_function)
        self.activation = activation_func(input_value + self.bias)
        
        # Record firing time if threshold exceeded
        if self.activation >= self.activation_threshold:
            self.last_fired = datetime.now()
            
        return self.activation
    
    def connect_to(self, target_neuron_id: str, weight: float = 0.1) -> None:
        """
        Create or update a connection to another neuron
        
        Parameters:
        target_neuron_id: ID of the target neuron
        weight: Connection weight
        """
        self.connections[target_neuron_id] = weight
    
    def get_outgoing_activation(self, target_neuron_id: str) -> float:
        """
        Get the activation value being sent to a specific target neuron
        
        Parameters:
        target_neuron_id: ID of the target neuron
        
        Returns:
        Weighted activation value
        """
        if target_neuron_id not in self.connections:
            return 0.0
            
        return self.activation * self.connections[target_neuron_id]
    
    def adjust_weight(self, target_neuron_id: str, delta: float) -> float:
        """
        Adjust the weight of a connection
        
        Parameters:
        target_neuron_id: ID of the target neuron
        delta: Amount to adjust the weight
        
        Returns:
        New weight value
        """
        if target_neuron_id not in self.connections:
            return 0.0
            
        self.connections[target_neuron_id] += delta * self.learning_rate
        return self.connections[target_neuron_id]
    
    def reset(self) -> None:
        """Reset the neuron's activation"""
        self.activation = 0.0
        self.last_fired = None
