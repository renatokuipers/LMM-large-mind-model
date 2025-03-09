import torch 
import torch.nn as nn 
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
import uuid
import numpy as np
from datetime import datetime

from .neuron import Neuron
from .synapse import Synapse
from .neural_cluster import NeuralCluster
from .hebbian_learning import HebbianLearning
from ..core.exceptions import NeuralSubstrateError

class NeuralNetwork(BaseModel):
    """
    Neural network implementation for the neural substrate
    
    This class represents a complete neural network with neurons,
    synapses, and clusters. It provides methods for creating,
    connecting, and activating neurons.
    """
    network_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    neurons: Dict[str, Neuron] = Field(default_factory=dict)
    synapses: Dict[str, Synapse] = Field(default_factory=dict)
    clusters: Dict[str, NeuralCluster] = Field(default_factory=dict)
    input_neurons: List[str] = Field(default_factory=list)
    output_neurons: List[str] = Field(default_factory=list)
    learning_mechanism: HebbianLearning = Field(default_factory=HebbianLearning)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def create_neuron(self, activation_function: str = "sigmoid", activation_threshold: float = 0.5) -> Neuron:
        """
        Create a new neuron in the network
        
        Parameters:
        activation_function: Activation function to use
        activation_threshold: Activation threshold
        
        Returns:
        The created neuron
        """
        neuron = Neuron(
            activation_function=activation_function,
            activation_threshold=activation_threshold
        )
        
        self.neurons[neuron.neuron_id] = neuron
        return neuron
    
    def create_synapse(self, source_id: str, target_id: str, weight: float = 0.1) -> Synapse:
        """
        Create a connection between two neurons
        
        Parameters:
        source_id: ID of the source neuron
        target_id: ID of the target neuron
        weight: Connection weight
        
        Returns:
        The created synapse
        
        Raises:
        NeuralSubstrateError: If either neuron doesn't exist
        """
        if source_id not in self.neurons:
            raise NeuralSubstrateError(f"Source neuron {source_id} not found")
        if target_id not in self.neurons:
            raise NeuralSubstrateError(f"Target neuron {target_id} not found")
            
        synapse = Synapse(
            source_id=source_id,
            target_id=target_id,
            weight=weight
        )
        
        synapse_id = synapse.synapse_id
        self.synapses[synapse_id] = synapse
        
        # Update the source neuron's connections
        self.neurons[source_id].connect_to(target_id, weight)
        
        return synapse
    
    def create_cluster(self, name: str, neuron_ids: Optional[List[str]] = None) -> NeuralCluster:
        """
        Create a neural cluster
        
        Parameters:
        name: Name of the cluster
        neuron_ids: Optional list of neuron IDs to include in the cluster
        
        Returns:
        The created cluster
        """
        cluster = NeuralCluster(name=name)
        
        if neuron_ids:
            for neuron_id in neuron_ids:
                if neuron_id in self.neurons:
                    cluster.add_neuron(self.neurons[neuron_id])
                    
        self.clusters[cluster.cluster_id] = cluster
        return cluster
    
    def activate(self, inputs: Dict[str, float], steps: int = 3) -> Dict[str, float]:
        """
        Activate the network with the given inputs
        
        Parameters:
        inputs: Dictionary mapping input neuron IDs to activation values
        steps: Number of propagation steps
        
        Returns:
        Dictionary of output neuron activations
        """
        # Reset all neuron activations to ensure clean state
        for neuron_id, neuron in self.neurons.items():
            neuron.reset()
            
        # Apply inputs to input neurons
        for i, input_id in enumerate(self.input_neurons):
            if input_id in self.neurons:
                # Use the input value directly as the activation for input neurons
                input_key = f"input_{i}"
                if input_key in inputs:
                    input_value = inputs[input_key]
                    # Set activation directly instead of using activate method
                    self.neurons[input_id].activation = input_value
                    
        # Propagate activations through the network
        for _ in range(steps):
            self._propagate_activations()
            
        # Collect outputs
        outputs = {}
        for neuron_id in self.output_neurons:
            if neuron_id in self.neurons:
                outputs[neuron_id] = self.neurons[neuron_id].activation
                
        return outputs
    
    def _propagate_activations(self) -> None:
        """
        Propagate activations through the network's connections
        """
        # Collect all inputs for each neuron
        neuron_inputs = {neuron_id: 0.0 for neuron_id in self.neurons}
        
        # Calculate inputs from synapses
        for synapse_id, synapse in self.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id in self.neurons and target_id in self.neurons:
                source_activation = self.neurons[source_id].activation
                # Only propagate if source has activation
                if source_activation > 0.0:
                    weighted_activation = synapse.transmit(source_activation)
                    neuron_inputs[target_id] += weighted_activation
                
        # Apply inputs to neurons (except input neurons which should keep their direct inputs)
        for neuron_id, input_value in neuron_inputs.items():
            # Skip input neurons to preserve their direct input values
            if neuron_id not in self.input_neurons and input_value > 0.0:
                # Use the activate method to apply the activation function
                self.neurons[neuron_id].activate(input_value)
    
    def apply_hebbian_learning(self) -> None:
        """
        Apply Hebbian learning to all synapses in the network
        """
        # Get all neuron activations
        activations = {neuron_id: neuron.activation for neuron_id, neuron in self.neurons.items()}
        
        # Update each synapse
        for synapse_id, synapse in self.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id in activations and target_id in activations:
                pre_activation = activations[source_id]
                post_activation = activations[target_id]
                
                # Apply Hebbian learning
                new_weight = self.learning_mechanism.update_weight(
                    pre_activation, post_activation, synapse.weight
                )
                
                # Update synapse weight
                synapse.update_weight(new_weight - synapse.weight)
                
                # Update neuron connection
                if source_id in self.neurons:
                    self.neurons[source_id].adjust_weight(target_id, new_weight - synapse.weight)
    
    def reset(self) -> None:
        """
        Reset all neurons in the network
        """
        for neuron in self.neurons.values():
            neuron.reset()
            
        for cluster in self.clusters.values():
            cluster.reset()
