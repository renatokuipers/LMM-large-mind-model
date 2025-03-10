"""
Neural Substrate Module

This module provides the foundational neural building blocks from which higher
cognitive functions emerge. It implements the primitive neural architecture
that underpins all cognitive modules.

The neural substrate serves as the biological analogue in the system, providing
the low-level neural components that are then organized into specialized
cognitive modules.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from .neural_network import NeuralNetwork
from .neuron import Neuron
from .synapse import Synapse
from .neural_cluster import NeuralCluster
from .hebbian_learning import HebbianLearning
from .activation_functions import (
    sigmoid, relu, leaky_relu, tanh, softmax,
    get_activation_function
)

logger = logging.getLogger(__name__)

class NeuralSubstrateManager:
    """
    Central manager for neural substrate components
    
    This class provides a unified interface for creating and managing
    neural networks, neurons, synapses, and clusters. It ensures that
    components are properly connected and configured.
    """
    
    def __init__(self):
        self.networks: Dict[str, NeuralNetwork] = {}
        self.clusters: Dict[str, NeuralCluster] = {}
        self.learning_mechanisms: Dict[str, HebbianLearning] = {}
        
    def create_network(
        self, 
        network_id: str, 
        layers: List[int], 
        activation_function: str = 'sigmoid',
        learning_rate: float = 0.01,
        plasticity: float = 1.0
    ) -> NeuralNetwork:
        """
        Create a new neural network with the specified configuration
        
        Args:
            network_id: Unique identifier for the network
            layers: List of neuron counts per layer
            activation_function: Activation function to use
            learning_rate: Initial learning rate
            plasticity: Initial plasticity level
            
        Returns:
            The created neural network
        """
        if network_id in self.networks:
            logger.warning(f"Overwriting existing network with ID {network_id}")
            
        network = NeuralNetwork(
            network_id=network_id,
            layers=layers,
            activation_function=get_activation_function(activation_function),
            learning_rate=learning_rate
        )
        
        self.networks[network_id] = network
        return network
    
    def create_cluster(
        self, 
        cluster_id: str, 
        neuron_count: int,
        activation_function: str = 'sigmoid',
        connection_density: float = 0.5,
        plasticity: float = 1.0
    ) -> NeuralCluster:
        """
        Create a new neural cluster
        
        Args:
            cluster_id: Unique identifier for the cluster
            neuron_count: Number of neurons in the cluster
            activation_function: Activation function to use
            connection_density: Density of internal connections
            plasticity: Plasticity level for connections
            
        Returns:
            The created neural cluster
        """
        if cluster_id in self.clusters:
            logger.warning(f"Overwriting existing cluster with ID {cluster_id}")
            
        activation_func = get_activation_function(activation_function)
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            neuron_count=neuron_count,
            activation_function=activation_func,
            connection_density=connection_density,
            plasticity=plasticity
        )
        
        self.clusters[cluster_id] = cluster
        return cluster
    
    def create_hebbian_learning(
        self, 
        learning_id: str,
        network: Optional[NeuralNetwork] = None,
        learning_rate: float = 0.01,
        hebbian_rule: str = 'oja'
    ) -> HebbianLearning:
        """
        Create a Hebbian learning mechanism
        
        Args:
            learning_id: Unique identifier for the learning mechanism
            network: Neural network to apply learning to
            learning_rate: Learning rate
            hebbian_rule: Type of Hebbian learning rule
            
        Returns:
            The created Hebbian learning mechanism
        """
        if learning_id in self.learning_mechanisms:
            logger.warning(f"Overwriting existing learning mechanism with ID {learning_id}")
            
        learning = HebbianLearning(
            learning_id=learning_id,
            learning_rate=learning_rate,
            hebbian_rule=hebbian_rule
        )
        
        if network is not None:
            learning.attach_network(network)
            
        self.learning_mechanisms[learning_id] = learning
        return learning
    
    def connect_clusters(
        self, 
        source_cluster_id: str, 
        target_cluster_id: str,
        connection_strength: float = 0.5,
        connection_density: float = 0.3,
        bidirectional: bool = False
    ) -> int:
        """
        Connect two neural clusters
        
        Args:
            source_cluster_id: ID of source cluster
            target_cluster_id: ID of target cluster
            connection_strength: Initial connection strength
            connection_density: Density of connections between clusters
            bidirectional: Whether to create connections in both directions
            
        Returns:
            Number of connections created
        """
        if source_cluster_id not in self.clusters:
            raise ValueError(f"Source cluster {source_cluster_id} not found")
        if target_cluster_id not in self.clusters:
            raise ValueError(f"Target cluster {target_cluster_id} not found")
            
        source = self.clusters[source_cluster_id]
        target = self.clusters[target_cluster_id]
        
        conn_count = source.connect_to_cluster(
            target, 
            strength=connection_strength,
            density=connection_density
        )
        
        if bidirectional:
            conn_count += target.connect_to_cluster(
                source,
                strength=connection_strength,
                density=connection_density
            )
            
        return conn_count
    
    def get_network(self, network_id: str) -> Optional[NeuralNetwork]:
        """Get a neural network by ID"""
        return self.networks.get(network_id)
    
    def get_cluster(self, cluster_id: str) -> Optional[NeuralCluster]:
        """Get a neural cluster by ID"""
        return self.clusters.get(cluster_id)
    
    def get_learning_mechanism(self, learning_id: str) -> Optional[HebbianLearning]:
        """Get a learning mechanism by ID"""
        return self.learning_mechanisms.get(learning_id)
    
    def apply_hebbian_learning(self, network_id: str) -> Dict[str, Any]:
        """
        Apply Hebbian learning to a network
        
        Args:
            network_id: ID of the network to apply learning to
            
        Returns:
            Dictionary with learning results
        """
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
            
        for learning_id, mechanism in self.learning_mechanisms.items():
            if mechanism.is_attached_to(self.networks[network_id]):
                return mechanism.apply_learning()
                
        # If no mechanism is attached to this network, create one
        learning = self.create_hebbian_learning(
            f"{network_id}_learning",
            self.networks[network_id]
        )
        return learning.apply_learning()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of all neural components"""
        return {
            "networks": {nid: net.get_state() for nid, net in self.networks.items()},
            "clusters": {cid: cluster.get_state() for cid, cluster in self.clusters.items()},
            "learning_mechanisms": {lid: lm.get_state() for lid, lm in self.learning_mechanisms.items()}
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state from a dictionary"""
        # Implement state loading logic here
        pass

def create_neural_substrate() -> NeuralSubstrateManager:
    """
    Create and return a new neural substrate manager
    
    Returns:
        Initialized NeuralSubstrateManager
    """
    return NeuralSubstrateManager()

# Exports
__all__ = [
    # Classes
    'NeuralNetwork',
    'Neuron',
    'Synapse',
    'NeuralCluster',
    'HebbianLearning',
    'NeuralSubstrateManager',
    
    # Functions
    'create_neural_substrate',
    'sigmoid',
    'relu',
    'leaky_relu',
    'tanh',
    'softmax',
    'get_activation_function'
] 
