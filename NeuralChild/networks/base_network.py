# base_network.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("BaseNetwork")

class NetworkConnection(BaseModel):
    """Connection between neural networks"""
    target_network: str
    connection_type: ConnectionType
    strength: float = Field(0.0, ge=0.0, le=1.0) 
    bidirectional: bool = False
    last_activation: datetime = Field(default_factory=datetime.now)
    
    def activate(self, intensity: float = 0.5) -> float:
        """Activate this connection with given intensity"""
        transmitted_signal = intensity * self.strength
        self.last_activation = datetime.now()
        return transmitted_signal

class NetworkState(BaseModel):
    """Current state of a neural network"""
    network_type: NetworkType
    activation: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    training_progress: float = Field(0.0, ge=0.0, le=1.0)
    last_active: datetime = Field(default_factory=datetime.now)
    error_rate: float = Field(0.0, ge=0.0)
    
    def update_activation(self, stimulation: float) -> None:
        """Update the activation level based on input stimulation"""
        decay_factor = 0.9  # Previous activation decays by this factor
        self.activation = min(1.0, max(0.0, self.activation * decay_factor + stimulation))
        self.last_active = datetime.now()

class BaseNetwork:
    """Base class for all neural networks in the system"""
    
    def __init__(
        self, 
        network_type: NetworkType,
        initial_state: Optional[NetworkState] = None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.2,
        name: Optional[str] = None
    ):
        """Initialize the neural network"""
        self.network_type = network_type
        self.name = name or str(network_type.value)
        
        # Initialize network state
        self.state = initial_state or NetworkState(
            network_type=network_type,
            learning_rate=0.01 * learning_rate_multiplier
        )
        
        # Configure network properties
        self.activation_threshold = activation_threshold
        self.learning_rate_multiplier = learning_rate_multiplier
        
        # Connections to other networks
        self.connections: Dict[str, NetworkConnection] = {}
        
        # Input queue for processing
        self.input_buffer: List[Dict[str, Any]] = []
        
        # Processing history
        self.activation_history: List[Tuple[datetime, float]] = []
        
        logger.info(f"Initialized {self.name} neural network")
    
    def connect_to(
        self, 
        target_network: str,
        connection_type: ConnectionType,
        initial_strength: float = 0.5,
        bidirectional: bool = False
    ) -> None:
        """Establish connection to another network"""
        if target_network not in self.connections:
            self.connections[target_network] = NetworkConnection(
                target_network=target_network,
                connection_type=connection_type,
                strength=initial_strength,
                bidirectional=bidirectional
            )
            logger.info(f"Connected {self.name} to {target_network} with strength {initial_strength}")
    
    def receive_input(self, input_data: Dict[str, Any], source_network: Optional[str] = None) -> None:
        """Receive input from environment or another network"""
        # Add to input buffer with metadata
        self.input_buffer.append({
            "data": input_data,
            "source": source_network,
            "timestamp": datetime.now()
        })
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process all inputs in the buffer and generate output
        This method should be overridden by subclasses to implement specific processing
        """
        if not self.input_buffer:
            return {}
        
        # Default implementation just averages numerical values
        aggregated_data = {}
        for input_item in self.input_buffer:
            data = input_item["data"]
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if key not in aggregated_data:
                        aggregated_data[key] = []
                    aggregated_data[key].append(value)
        
        # Average the values
        result = {key: sum(values) / len(values) for key, values in aggregated_data.items()}
        
        # Clear input buffer
        self.input_buffer = []
        
        return result
    
    def update(self, external_stimulation: float = 0.0) -> Dict[str, Any]:
        """Update network state and propagate signals"""
        # Process inputs to determine activation level
        processing_result = self.process_inputs()
        
        # Calculate new activation level
        internal_stimulation = sum(processing_result.values()) / max(1, len(processing_result))
        total_stimulation = (internal_stimulation + external_stimulation) / 2
        
        # Update activation level
        previous_activation = self.state.activation
        self.state.update_activation(total_stimulation)
        
        # Record activation
        self.activation_history.append((datetime.now(), self.state.activation))
        if len(self.activation_history) > 1000:  # Keep history limited
            self.activation_history.pop(0)
        
        # Only propagate signals if activation threshold is reached
        outputs = {}
        if self.state.activation >= self.activation_threshold:
            # Update training progress when network is significantly activated
            progress_increment = self.state.activation * self.state.learning_rate
            self.state.training_progress = min(1.0, self.state.training_progress + progress_increment)
            
            # Propagate signals to connected networks if activation increased
            if self.state.activation > previous_activation:
                outputs = self._propagate_signals()
            
            # Error rate decreases with training
            self.state.error_rate = max(0.01, self.state.error_rate * 0.999)
            
            # Confidence increases with training progress
            self.state.confidence = min(1.0, self.state.training_progress * 0.8 + 0.1)
        
        return outputs
    
    def _propagate_signals(self) -> Dict[str, Dict[str, Any]]:
        """Propagate signals to connected networks"""
        outputs = {}
        for network_name, connection in self.connections.items():
            # Only propagate if connection is strong enough
            if connection.strength >= 0.1:
                # Signal strength depends on current activation and connection strength
                signal_strength = connection.activate(self.state.activation)
                
                # Prepare data to send to target network
                outputs[network_name] = {
                    "signal_strength": signal_strength,
                    "source_network": self.name,
                    "connection_type": connection.connection_type.value,
                    "data": self._prepare_output_data()
                }
        
        return outputs
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks
        This method should be overridden by subclasses
        """
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value
        }
    
    def strengthen_connection(self, target_network: str, amount: float = 0.01) -> None:
        """Strengthen connection to another network"""
        if target_network in self.connections:
            connection = self.connections[target_network]
            connection.strength = min(1.0, connection.strength + amount)
    
    def weaken_connection(self, target_network: str, amount: float = 0.01) -> None:
        """Weaken connection to another network"""
        if target_network in self.connections:
            connection = self.connections[target_network]
            connection.strength = max(0.0, connection.strength - amount)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the network"""
        return {
            "network_type": self.network_type.value,
            "name": self.name,
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "training_progress": self.state.training_progress,
            "error_rate": self.state.error_rate,
            "last_active": self.state.last_active.isoformat(),
            "connections": {
                name: {
                    "target": conn.target_network,
                    "type": conn.connection_type.value,
                    "strength": conn.strength
                } for name, conn in self.connections.items()
            }
        }
    
    def load_state(self, state_data: Dict[str, Any]) -> None:
        """Load network state from saved data"""
        if "activation" in state_data:
            self.state.activation = state_data["activation"]
        if "confidence" in state_data:
            self.state.confidence = state_data["confidence"]
        if "training_progress" in state_data:
            self.state.training_progress = state_data["training_progress"]
        if "error_rate" in state_data:
            self.state.error_rate = state_data["error_rate"]
        
        # Restore connections
        if "connections" in state_data:
            for name, conn_data in state_data["connections"].items():
                if name not in self.connections:
                    # Create missing connection
                    self.connect_to(
                        target_network=conn_data["target"],
                        connection_type=ConnectionType(conn_data["type"]),
                        initial_strength=conn_data["strength"]
                    )
                else:
                    # Update existing connection
                    self.connections[name].strength = conn_data["strength"]