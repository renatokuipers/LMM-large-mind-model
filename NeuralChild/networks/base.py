"""
Base network module providing the foundation for all specialized neural networks.
Implements common functionality for activation, connection management, and learning.
"""

import time
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Type, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import ValidationError

from ..models.network_models import (
    BaseNetwork, BaseNetworkConfig, Connection, ConnectionType,
    ActivationFunction, NetworkState
)
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

class Network:
    """
    Base implementation for all neural networks in the system.
    Provides core functionality for network activation, connection management, and learning.
    """
    
    def __init__(
        self, 
        model: BaseNetwork,
        activation_fn: Optional[callable] = None
    ):
        """
        Initialize a network with a base model.
        
        Args:
            model: BaseNetwork model containing network parameters
            activation_fn: Optional custom activation function
        """
        self.model = model
        self._activation_fn = activation_fn
        self._input_buffer = []
        self._last_update = time.time()
        logger.debug(f"Initialized network {self.model.name} of type {self.model.type}")
    
    @property
    def id(self) -> UUID:
        """Get the network ID"""
        return self.model.id
    
    @property
    def name(self) -> str:
        """Get the network name"""
        return self.model.name
    
    @property
    def type(self) -> str:
        """Get the network type"""
        return self.model.type
    
    @property
    def activation(self) -> float:
        """Get the current activation level"""
        return self.model.activation
    
    @property
    def state(self) -> NetworkState:
        """Get the current network state"""
        return self.model.state
    
    def set_state(self, state: NetworkState) -> None:
        """Set the network state"""
        self.model.state = state
        self.model.last_updated = datetime.now()
    
    def apply_activation_function(self, value: float) -> float:
        """
        Apply the network's activation function to a value
        
        Args:
            value: The input value to the activation function
            
        Returns:
            The result after applying the activation function
        """
        if self._activation_fn is not None:
            return self._activation_fn(value)
        
        # Use the specified activation function from the model
        activation_fn = self.model.config.activation_function
        
        if activation_fn == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + np.exp(-value))
        elif activation_fn == ActivationFunction.TANH:
            return np.tanh(value)
        elif activation_fn == ActivationFunction.RELU:
            return max(0.0, value)
        elif activation_fn == ActivationFunction.LEAKY_RELU:
            return value if value > 0 else 0.01 * value
        elif activation_fn == ActivationFunction.LINEAR:
            return value
        elif activation_fn == ActivationFunction.SOFTMAX:
            # Softmax doesn't really make sense for a single value,
            # but we'll just normalize it to [0, 1]
            return min(1.0, max(0.0, value))
        else:
            return value
    
    def add_connection(
        self, 
        target_network_id: UUID, 
        connection_type: ConnectionType,
        weight: float = 1.0
    ) -> None:
        """
        Add a connection to another network.
        
        Args:
            target_network_id: The ID of the target network
            connection_type: The type of connection to create
            weight: Initial connection weight
        """
        self.model.add_connection(target_network_id, connection_type, weight)
        logger.debug(f"Added {connection_type} connection from {self.name} to network {target_network_id}")
    
    def get_connections(self, connection_type: Optional[ConnectionType] = None) -> List[Connection]:
        """
        Get all outgoing connections, optionally filtered by type.
        
        Args:
            connection_type: Optional filter for connection type
            
        Returns:
            List of connections
        """
        if connection_type is None:
            return self.model.connections
        
        return [conn for conn in self.model.connections if conn.connection_type == connection_type]
    
    def get_incoming_connections(
        self, 
        network_registry: Dict[UUID, 'Network'],
        connection_type: Optional[ConnectionType] = None
    ) -> List[Tuple[Connection, 'Network']]:
        """
        Get all incoming connections from other networks.
        
        Args:
            network_registry: Dictionary mapping network IDs to Network instances
            connection_type: Optional filter for connection type
            
        Returns:
            List of (connection, source_network) tuples
        """
        incoming = []
        
        for network in network_registry.values():
            for conn in network.get_connections(connection_type):
                if conn.target_id == self.id:
                    incoming.append((conn, network))
        
        return incoming
    
    def update_activation(self, new_value: float) -> None:
        """
        Update the network's activation level.
        
        Args:
            new_value: The new activation value
        """
        clamped_value = max(0.0, min(new_value, 1.0))
        self.model.update_activation(clamped_value)
        
        # Update network state based on activation
        if clamped_value >= self.model.config.threshold:
            if self.state != NetworkState.LEARNING and self.state != NetworkState.CONSOLIDATING:
                self.set_state(NetworkState.ACTIVE)
        else:
            if self.state != NetworkState.LEARNING and self.state != NetworkState.CONSOLIDATING:
                self.set_state(NetworkState.INACTIVE)
    
    def add_input(self, value: float, source_id: Optional[UUID] = None) -> None:
        """
        Add an input to the network's input buffer.
        
        Args:
            value: The input value
            source_id: Optional ID of the source network
        """
        self._input_buffer.append((value, source_id))
    
    def process_inputs(self) -> None:
        """Process all inputs in the buffer and update activation"""
        if not self._input_buffer:
            return
        
        # Simple weighted sum of inputs
        total_input = sum(value for value, _ in self._input_buffer)
        
        # Apply activation function
        new_activation = self.apply_activation_function(total_input)
        
        # Update network activation
        self.update_activation(new_activation)
        
        # Clear the input buffer
        self._input_buffer = []
    
    def propagate_activation(self, network_registry: Dict[UUID, 'Network']) -> None:
        """
        Propagate this network's activation to connected networks.
        
        Args:
            network_registry: Dictionary mapping network IDs to Network instances
        """
        if self.activation <= 0.0:
            return
        
        for conn in self.model.connections:
            if conn.target_id in network_registry:
                target_network = network_registry[conn.target_id]
                
                # Calculate the output value based on connection type
                if conn.connection_type == ConnectionType.EXCITATORY:
                    output = self.activation * conn.weight
                elif conn.connection_type == ConnectionType.INHIBITORY:
                    output = -self.activation * conn.weight
                elif conn.connection_type == ConnectionType.MODULATORY:
                    # Modulatory connections don't directly activate but
                    # will be processed separately by target networks
                    output = self.activation * conn.weight * 0.5
                else:
                    output = self.activation * conn.weight
                
                # Add the output to the target network's input buffer
                target_network.add_input(output, self.id)
                
                # Mark the connection as active
                conn.last_active = datetime.now()
    
    def update(self, network_registry: Dict[UUID, 'Network'], time_delta: float) -> None:
        """
        Update the network state based on time elapsed.
        
        Args:
            network_registry: Dictionary mapping network IDs to Network instances
            time_delta: Time elapsed since last update (in seconds)
        """
        # Process incoming activations
        self.process_inputs()
        
        # Apply activation decay over time
        if self.activation > 0:
            decay_rate = self.model.config.decay_rate * time_delta
            new_activation = self.activation * (1.0 - decay_rate)
            self.update_activation(new_activation)
        
        # Propagate activation to connected networks
        self.propagate_activation(network_registry)
        
        # Update last update time
        self._last_update = time.time()
    
    def learn(
        self, 
        target_network_id: UUID, 
        connection_type: ConnectionType,
        adjustment: float
    ) -> bool:
        """
        Adjust a connection weight through learning.
        
        Args:
            target_network_id: The ID of the target network
            connection_type: The type of connection to adjust
            adjustment: The weight adjustment amount
            
        Returns:
            True if the connection was adjusted, False otherwise
        """
        # Find the connection
        for conn in self.model.connections:
            if conn.target_id == target_network_id and conn.connection_type == connection_type:
                # Apply learning rate
                effective_adjustment = adjustment * self.model.config.learning_rate
                
                # Update weight
                conn.weight = max(0.0, min(2.0, conn.weight + effective_adjustment))
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the network to a dictionary for serialization"""
        return self.model.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], network_model_cls: Type[BaseNetwork] = BaseNetwork) -> 'Network':
        """
        Create a network from a dictionary.
        
        Args:
            data: Dictionary containing network data
            network_model_cls: The specific BaseNetwork subclass to use
            
        Returns:
            A new Network instance
        """
        try:
            model = network_model_cls(**data)
            return cls(model)
        except ValidationError as e:
            logger.error(f"Error creating network from dictionary: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """String representation of the network"""
        return (f"Network {self.name} "
                f"(type={self.type}, "
                f"state={self.state}, "
                f"activation={self.activation:.2f})")