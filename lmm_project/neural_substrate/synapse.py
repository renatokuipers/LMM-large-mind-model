from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from ..core.types import ConnectionType

class Synapse(BaseModel):
    """
    Represents a connection between neurons
    
    A synapse connects two neurons and has a weight that determines
    the strength of the connection. It can be excitatory (positive weight)
    or inhibitory (negative weight).
    """
    synapse_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    weight: float = Field(default=0.1)
    connection_type: ConnectionType = ConnectionType.EXCITATORY
    plasticity: float = Field(default=0.01, ge=0.0, le=1.0)
    last_activation: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def transmit(self, activation: float) -> float:
        """
        Transmit activation from source to target
        
        Parameters:
        activation: Activation value from source neuron
        
        Returns:
        Weighted activation value for target neuron
        """
        self.last_activation = activation
        
        if self.connection_type == ConnectionType.INHIBITORY:
            # Inhibitory connections have negative effect
            return -abs(activation * self.weight)
        elif self.connection_type == ConnectionType.MODULATORY:
            # Modulatory connections don't directly activate but modify other inputs
            # This is a simplified implementation
            return 0.0
        else:  # EXCITATORY
            return activation * self.weight
    
    def update_weight(self, delta: float) -> float:
        """
        Update the synapse weight
        
        Parameters:
        delta: Amount to change the weight
        
        Returns:
        New weight value
        """
        # Apply plasticity factor to weight change
        effective_delta = delta * self.plasticity
        
        # Update weight
        self.weight += effective_delta
        
        # Update connection type based on weight sign
        if self.weight > 0:
            self.connection_type = ConnectionType.EXCITATORY
        elif self.weight < 0:
            self.connection_type = ConnectionType.INHIBITORY
            
        return self.weight
    
    def hebbian_update(self, pre_activation: float, post_activation: float, learning_rate: float = 0.01) -> float:
        """
        Update weight using Hebbian learning rule
        
        Parameters:
        pre_activation: Activation of source neuron
        post_activation: Activation of target neuron
        learning_rate: Learning rate
        
        Returns:
        New weight value
        """
        # Hebbian learning: neurons that fire together, wire together
        delta = learning_rate * pre_activation * post_activation
        return self.update_weight(delta)
