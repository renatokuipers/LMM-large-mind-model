from typing import Dict, List, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field

from ..core.exceptions import NeuralSubstrateError

class HebbianLearning(BaseModel):
    """
    Implementation of Hebbian learning rule
    
    Hebbian learning is based on the principle that "neurons that fire together, wire together."
    This class provides methods to apply Hebbian learning to neural networks.
    """
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.001, ge=0.0, le=1.0)
    min_weight: float = Field(default=-1.0)
    max_weight: float = Field(default=1.0)
    
    def update_weight(self, pre_activation: float, post_activation: float, current_weight: float) -> float:
        """
        Update a connection weight using the Hebbian rule
        
        Parameters:
        pre_activation: Activation of the presynaptic neuron
        post_activation: Activation of the postsynaptic neuron
        current_weight: Current weight of the connection
        
        Returns:
        Updated weight
        """
        # Basic Hebbian rule: weight change proportional to pre * post activation
        delta_weight = self.learning_rate * pre_activation * post_activation
        
        # Apply weight decay (prevents unbounded growth)
        decay = self.decay_rate * current_weight
        
        # Update weight
        new_weight = current_weight + delta_weight - decay
        
        # Clip weight to bounds
        return max(self.min_weight, min(self.max_weight, new_weight))
    
    def update_weights_batch(self, activations: Dict[str, float], connections: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """
        Update multiple connection weights in batch
        
        Parameters:
        activations: Dictionary mapping neuron IDs to activation values
        connections: Dictionary mapping (source_id, target_id) tuples to weights
        
        Returns:
        Updated connections dictionary
        """
        updated_connections = connections.copy()
        
        for (source_id, target_id), weight in connections.items():
            if source_id in activations and target_id in activations:
                pre_activation = activations[source_id]
                post_activation = activations[target_id]
                
                updated_connections[(source_id, target_id)] = self.update_weight(
                    pre_activation, post_activation, weight
                )
                
        return updated_connections
    
    def apply_oja_rule(self, pre_activation: float, post_activation: float, current_weight: float) -> float:
        """
        Apply Oja's rule, a normalized Hebbian rule
        
        Parameters:
        pre_activation: Activation of the presynaptic neuron
        post_activation: Activation of the postsynaptic neuron
        current_weight: Current weight of the connection
        
        Returns:
        Updated weight
        """
        # Oja's rule: prevents unbounded growth by normalizing
        delta_weight = self.learning_rate * (pre_activation * post_activation - post_activation**2 * current_weight)
        
        # Update weight
        new_weight = current_weight + delta_weight
        
        # Clip weight to bounds
        return max(self.min_weight, min(self.max_weight, new_weight))
    
    def apply_bcm_rule(self, pre_activation: float, post_activation: float, current_weight: float, threshold: float) -> float:
        """
        Apply BCM (Bienenstock-Cooper-Munro) rule
        
        Parameters:
        pre_activation: Activation of the presynaptic neuron
        post_activation: Activation of the postsynaptic neuron
        current_weight: Current weight of the connection
        threshold: Modification threshold
        
        Returns:
        Updated weight
        """
        # BCM rule: LTP when post > threshold, LTD when post < threshold
        delta_weight = self.learning_rate * pre_activation * post_activation * (post_activation - threshold)
        
        # Update weight
        new_weight = current_weight + delta_weight
        
        # Clip weight to bounds
        return max(self.min_weight, min(self.max_weight, new_weight))
