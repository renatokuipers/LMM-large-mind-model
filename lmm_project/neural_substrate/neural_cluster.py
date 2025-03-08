from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field
import uuid
import numpy as np
from datetime import datetime

from .neuron import Neuron
from .synapse import Synapse
from ..core.exceptions import NeuralSubstrateError

class NeuralCluster(BaseModel):
    """
    A cluster of neurons that function as a unit
    
    Neural clusters group neurons that serve a related function,
    allowing for higher-level organization of the neural substrate.
    """
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    neurons: Dict[str, Neuron] = Field(default_factory=dict)
    internal_synapses: Dict[str, Synapse] = Field(default_factory=dict)
    external_inputs: Dict[str, List[str]] = Field(default_factory=dict)
    external_outputs: Dict[str, List[str]] = Field(default_factory=dict)
    activation_pattern: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def add_neuron(self, neuron: Neuron) -> None:
        """
        Add a neuron to the cluster
        
        Parameters:
        neuron: The neuron to add
        """
        self.neurons[neuron.neuron_id] = neuron
        self.activation_pattern[neuron.neuron_id] = 0.0
    
    def connect_neurons(self, source_id: str, target_id: str, weight: float = 0.1) -> Synapse:
        """
        Create a connection between two neurons in the cluster
        
        Parameters:
        source_id: ID of the source neuron
        target_id: ID of the target neuron
        weight: Connection weight
        
        Returns:
        The created synapse
        
        Raises:
        NeuralSubstrateError: If either neuron is not in the cluster
        """
        if source_id not in self.neurons:
            raise NeuralSubstrateError(f"Source neuron {source_id} not in cluster")
        if target_id not in self.neurons:
            raise NeuralSubstrateError(f"Target neuron {target_id} not in cluster")
            
        synapse = Synapse(
            source_id=source_id,
            target_id=target_id,
            weight=weight
        )
        
        synapse_id = synapse.synapse_id
        self.internal_synapses[synapse_id] = synapse
        
        # Update the source neuron's connections
        self.neurons[source_id].connect_to(target_id, weight)
        
        return synapse
    
    def process_inputs(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Process inputs through the cluster
        
        Parameters:
        inputs: Dictionary mapping input neuron IDs to activation values
        
        Returns:
        Dictionary of output neuron activations
        """
        # Apply inputs to neurons
        for neuron_id, input_value in inputs.items():
            if neuron_id in self.neurons:
                self.neurons[neuron_id].activate(input_value)
                
        # Propagate activations through the cluster
        for _ in range(3):  # Simple fixed-point iteration, could be more sophisticated
            self._propagate_activations()
            
        # Collect outputs
        outputs = {}
        for neuron_id, neuron in self.neurons.items():
            outputs[neuron_id] = neuron.activation
            self.activation_pattern[neuron_id] = neuron.activation
            
        return outputs
    
    def _propagate_activations(self) -> None:
        """
        Propagate activations through the cluster's internal connections
        """
        # Collect all inputs for each neuron
        neuron_inputs = {neuron_id: 0.0 for neuron_id in self.neurons}
        
        # Calculate inputs from internal synapses
        for synapse_id, synapse in self.internal_synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id in self.neurons and target_id in self.neurons:
                source_activation = self.neurons[source_id].activation
                weighted_activation = synapse.transmit(source_activation)
                neuron_inputs[target_id] += weighted_activation
                
        # Apply inputs to neurons
        for neuron_id, input_value in neuron_inputs.items():
            if input_value != 0.0:  # Only update if there's input
                self.neurons[neuron_id].activate(input_value)
    
    def get_activation_vector(self) -> np.ndarray:
        """
        Get the cluster's activation pattern as a vector
        
        Returns:
        Numpy array of neuron activations
        """
        return np.array([self.neurons[n_id].activation for n_id in sorted(self.neurons.keys())])
    
    def reset(self) -> None:
        """
        Reset all neurons in the cluster
        """
        for neuron in self.neurons.values():
            neuron.reset()
        
        for neuron_id in self.activation_pattern:
            self.activation_pattern[neuron_id] = 0.0
