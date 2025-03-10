"""
Neural network implementation for the neural substrate.

This module defines the NeuralNetwork class, which provides a high-level interface
for creating and managing networks of neurons, synapses, and neural clusters.
"""
import uuid
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
import torch
from pydantic import BaseModel, Field

from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse, PlasticityType
from lmm_project.neural_substrate.neural_cluster import NeuralCluster, ClusterType
from lmm_project.neural_substrate.hebbian_learning import HebbianLearner, HebbianRule
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("neural_substrate.neural_network")


class NetworkType(Enum):
    """Types of neural network architectures."""
    CUSTOM = auto()         # Custom architecture
    FEEDFORWARD = auto()    # Standard feedforward network
    RECURRENT = auto()      # Recurrent network
    MODULAR = auto()        # Network of interconnected clusters
    RESERVOIR = auto()      # Reservoir computing network
    COMPETITIVE = auto()    # Competitive learning network


class NetworkConfig(BaseModel):
    """Configuration for a neural network."""
    network_type: NetworkType = Field(default=NetworkType.FEEDFORWARD)
    input_size: int = Field(default=1, gt=0)
    output_size: int = Field(default=1, gt=0)
    hidden_layers: List[int] = Field(default=[10])
    use_bias: bool = Field(default=True)
    activation_noise: float = Field(default=0.0, ge=0.0)
    plasticity_enabled: bool = Field(default=True)
    hebbian_rule: HebbianRule = Field(default=HebbianRule.BASIC)
    inhibitory_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    connection_density: float = Field(default=0.7, gt=0.0, le=1.0)
    
    # Modular network specific settings
    cluster_sizes: List[int] = Field(default=[])
    cluster_types: List[ClusterType] = Field(default=[])


class NeuralNetwork:
    """
    High-level network of neurons and/or neural clusters.
    
    The NeuralNetwork class provides a way to create and manage complex
    neural architectures composed of neurons, synapses, and neural clusters.
    It supports various network types and learning mechanisms.
    """
    
    def __init__(
        self,
        network_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a neural network.
        
        Parameters:
        network_id: Unique identifier for this network
        config: Configuration dictionary
        device: Torch device for tensor operations
        """
        self.network_id = network_id or str(uuid.uuid4())
        self.config = NetworkConfig(**(config or {}))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Track components
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.clusters: Dict[str, NeuralCluster] = {}
        
        # Track structure
        self.input_neuron_ids: List[str] = []
        self.output_neuron_ids: List[str] = []
        self.input_cluster_ids: List[str] = []
        self.output_cluster_ids: List[str] = []
        
        # Learning engine
        self.hebbian_learner = HebbianLearner(
            rule=self.config.hebbian_rule,
            device=self.device
        )
        
        # Simulation state
        self.current_time = 0.0
        self.time_step = 1.0
        self.activation_history: Dict[str, List[float]] = {}
        self.learning_enabled = self.config.plasticity_enabled
        
        # Initialize network based on type
        self._construct_network()
        
        logger.info(
            f"Created {self.config.network_type.name} neural network with "
            f"{len(self.neurons)} neurons, {len(self.synapses)} synapses, "
            f"and {len(self.clusters)} clusters"
        )
    
    def _construct_network(self) -> None:
        """Construct the network according to its type and configuration."""
        if self.config.network_type == NetworkType.FEEDFORWARD:
            self._construct_feedforward()
        elif self.config.network_type == NetworkType.RECURRENT:
            self._construct_recurrent()
        elif self.config.network_type == NetworkType.MODULAR:
            self._construct_modular()
        elif self.config.network_type == NetworkType.RESERVOIR:
            self._construct_reservoir()
        elif self.config.network_type == NetworkType.COMPETITIVE:
            self._construct_competitive()
        else:  # CUSTOM or fallback
            # Just create input and output layers
            self._create_io_neurons()
    
    def _construct_feedforward(self) -> None:
        """Construct a standard feedforward neural network."""
        # Create a single feed-forward cluster
        cluster_config = {
            "cluster_type": ClusterType.FEED_FORWARD,
            "neuron_count": (self.config.input_size + 
                           sum(self.config.hidden_layers) + 
                           self.config.output_size),
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "hidden_layers": self.config.hidden_layers,
            "inhibitory_ratio": self.config.inhibitory_ratio,
            "connection_density": self.config.connection_density,
            "plasticity_enabled": self.config.plasticity_enabled,
            "use_bias": self.config.use_bias
        }
        
        cluster_id = f"{self.network_id}_main_cluster"
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            config=cluster_config,
            device=self.device
        )
        
        # Register cluster
        self.clusters[cluster_id] = cluster
        self.input_cluster_ids.append(cluster_id)
        self.output_cluster_ids.append(cluster_id)
        
        # Add all neurons and synapses to our tracking
        self.neurons.update(cluster.neurons)
        self.synapses.update(cluster.synapses)
        
        # Track input and output neurons
        self.input_neuron_ids.extend(cluster.input_neuron_ids)
        self.output_neuron_ids.extend(cluster.output_neuron_ids)
    
    def _construct_recurrent(self) -> None:
        """Construct a recurrent neural network."""
        # Create a single recurrent cluster
        cluster_config = {
            "cluster_type": ClusterType.RECURRENT,
            "neuron_count": (self.config.input_size + 
                           sum(self.config.hidden_layers) + 
                           self.config.output_size),
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "hidden_layers": self.config.hidden_layers,
            "inhibitory_ratio": self.config.inhibitory_ratio,
            "connection_density": self.config.connection_density,
            "plasticity_enabled": self.config.plasticity_enabled,
            "use_bias": self.config.use_bias
        }
        
        cluster_id = f"{self.network_id}_recurrent_cluster"
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            config=cluster_config,
            device=self.device
        )
        
        # Register cluster
        self.clusters[cluster_id] = cluster
        self.input_cluster_ids.append(cluster_id)
        self.output_cluster_ids.append(cluster_id)
        
        # Add all neurons and synapses to our tracking
        self.neurons.update(cluster.neurons)
        self.synapses.update(cluster.synapses)
        
        # Track input and output neurons
        self.input_neuron_ids.extend(cluster.input_neuron_ids)
        self.output_neuron_ids.extend(cluster.output_neuron_ids)
    
    def _construct_modular(self) -> None:
        """Construct a network of interconnected clusters."""
        # Default cluster configuration if none provided
        if not self.config.cluster_sizes:
            self.config.cluster_sizes = [self.config.input_size, 20, self.config.output_size]
            
        if not self.config.cluster_types:
            self.config.cluster_types = [ClusterType.FEED_FORWARD] * len(self.config.cluster_sizes)
            
        # Ensure cluster_types is at least as long as cluster_sizes
        while len(self.config.cluster_types) < len(self.config.cluster_sizes):
            self.config.cluster_types.append(ClusterType.FEED_FORWARD)
        
        # Create each cluster
        created_clusters = []
        for i, (size, c_type) in enumerate(zip(self.config.cluster_sizes, self.config.cluster_types)):
            # Determine input/output configuration
            is_input = i == 0
            is_output = i == len(self.config.cluster_sizes) - 1
            
            # Set up cluster config
            cluster_config = {
                "cluster_type": c_type,
                "neuron_count": size,
                "input_size": self.config.input_size if is_input else 0,
                "output_size": self.config.output_size if is_output else 0,
                "hidden_layers": [size] if not (is_input or is_output) else [],
                "inhibitory_ratio": self.config.inhibitory_ratio,
                "connection_density": self.config.connection_density,
                "plasticity_enabled": self.config.plasticity_enabled,
                "use_bias": self.config.use_bias
            }
            
            # Create cluster
            cluster_id = f"{self.network_id}_cluster_{i}"
            cluster = NeuralCluster(
                cluster_id=cluster_id,
                config=cluster_config,
                device=self.device
            )
            
            # Register cluster
            self.clusters[cluster_id] = cluster
            created_clusters.append(cluster)
            
            if is_input:
                self.input_cluster_ids.append(cluster_id)
                self.input_neuron_ids.extend(cluster.input_neuron_ids)
                
            if is_output:
                self.output_cluster_ids.append(cluster_id)
                self.output_neuron_ids.extend(cluster.output_neuron_ids)
            
            # Add all neurons and synapses to our tracking
            self.neurons.update(cluster.neurons)
            self.synapses.update(cluster.synapses)
        
        # Connect clusters to each other (feed-forward connections)
        for i in range(len(created_clusters) - 1):
            source_cluster = created_clusters[i]
            target_cluster = created_clusters[i + 1]
            
            # Get output neurons from source cluster
            source_neurons = source_cluster.output_neuron_ids or list(source_cluster.neurons.keys())
            
            # Get non-input neurons from target cluster
            target_neurons = [n_id for n_id in target_cluster.neurons.keys() 
                             if n_id not in target_cluster.input_neuron_ids]
            
            # Connect with probability based on connection density
            for src_id in source_neurons:
                for tgt_id in target_neurons:
                    if np.random.random() < self.config.connection_density * 0.5:
                        # Create inter-cluster synapse
                        synapse_id = f"{src_id}_to_{tgt_id}"
                        synapse = Synapse(
                            pre_neuron_id=src_id,
                            post_neuron_id=tgt_id,
                            synapse_id=synapse_id,
                            config={
                                "plasticity_enabled": self.config.plasticity_enabled,
                                "initial_weight": np.random.uniform(-0.1, 0.1)
                            },
                            device=self.device
                        )
                        
                        # Add to tracking
                        self.synapses[synapse_id] = synapse
                        
                        # Register input in target neuron
                        self.neurons[tgt_id].add_input_connection(src_id)
    
    def _construct_reservoir(self) -> None:
        """Construct a reservoir computing network."""
        # Create a reservoir cluster
        cluster_config = {
            "cluster_type": ClusterType.RESERVOIR,
            "neuron_count": (self.config.input_size + 
                           sum(self.config.hidden_layers) + 
                           self.config.output_size),
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "inhibitory_ratio": self.config.inhibitory_ratio,
            "connection_density": 0.3,  # Sparser for reservoir
            "plasticity_enabled": False,  # Fixed weights for reservoir
            "use_bias": self.config.use_bias
        }
        
        cluster_id = f"{self.network_id}_reservoir_cluster"
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            config=cluster_config,
            device=self.device
        )
        
        # Register cluster
        self.clusters[cluster_id] = cluster
        self.input_cluster_ids.append(cluster_id)
        self.output_cluster_ids.append(cluster_id)
        
        # Add all neurons and synapses to our tracking
        self.neurons.update(cluster.neurons)
        self.synapses.update(cluster.synapses)
        
        # Track input and output neurons
        self.input_neuron_ids.extend(cluster.input_neuron_ids)
        self.output_neuron_ids.extend(cluster.output_neuron_ids)
        
        # Enable plasticity only for output connections
        for synapse_id, synapse in self.synapses.items():
            if synapse.post_neuron_id in self.output_neuron_ids:
                synapse.config.plasticity_enabled = self.config.plasticity_enabled
    
    def _construct_competitive(self) -> None:
        """Construct a competitive learning network."""
        # Create a competitive cluster
        cluster_config = {
            "cluster_type": ClusterType.COMPETITIVE,
            "neuron_count": (self.config.input_size + 
                           self.config.output_size),
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "inhibitory_ratio": 0.0,  # No inhibitory neurons, competitive connections instead
            "connection_density": 1.0,  # Full connections for competitive
            "plasticity_enabled": self.config.plasticity_enabled,
            "use_bias": self.config.use_bias
        }
        
        cluster_id = f"{self.network_id}_competitive_cluster"
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            config=cluster_config,
            device=self.device
        )
        
        # Register cluster
        self.clusters[cluster_id] = cluster
        self.input_cluster_ids.append(cluster_id)
        self.output_cluster_ids.append(cluster_id)
        
        # Add all neurons and synapses to our tracking
        self.neurons.update(cluster.neurons)
        self.synapses.update(cluster.synapses)
        
        # Track input and output neurons
        self.input_neuron_ids.extend(cluster.input_neuron_ids)
        self.output_neuron_ids.extend(cluster.output_neuron_ids)
    
    def _create_io_neurons(self) -> None:
        """Create input and output neurons for a custom network."""
        # Create input neurons
        for i in range(self.config.input_size):
            neuron_id = f"{self.network_id}_input_{i}"
            neuron = Neuron(
                neuron_id=neuron_id,
                config={"is_inhibitory": False},
                device=self.device
            )
            self.neurons[neuron_id] = neuron
            self.input_neuron_ids.append(neuron_id)
        
        # Create output neurons
        for i in range(self.config.output_size):
            neuron_id = f"{self.network_id}_output_{i}"
            neuron = Neuron(
                neuron_id=neuron_id,
                config={"is_inhibitory": False, "use_bias": self.config.use_bias},
                device=self.device
            )
            self.neurons[neuron_id] = neuron
            self.output_neuron_ids.append(neuron_id)
            
        # Connect input to output for minimal functionality
        for in_id in self.input_neuron_ids:
            for out_id in self.output_neuron_ids:
                if np.random.random() < self.config.connection_density:
                    synapse_id = f"{in_id}_to_{out_id}"
                    synapse = Synapse(
                        pre_neuron_id=in_id,
                        post_neuron_id=out_id,
                        synapse_id=synapse_id,
                        config={
                            "plasticity_enabled": self.config.plasticity_enabled,
                            "plasticity_type": PlasticityType.HEBBIAN,
                            "initial_weight": np.random.uniform(-0.1, 0.1)
                        },
                        device=self.device
                    )
                    self.synapses[synapse_id] = synapse
                    self.neurons[out_id].add_input_connection(in_id)
    
    def add_custom_neuron(
        self,
        neuron_id: Optional[str] = None,
        neuron_config: Optional[Dict[str, Any]] = None,
        is_input: bool = False,
        is_output: bool = False
    ) -> str:
        """
        Add a custom neuron to the network.
        
        Parameters:
        neuron_id: Optional ID for the neuron
        neuron_config: Configuration for the neuron
        is_input: Whether this is an input neuron
        is_output: Whether this is an output neuron
        
        Returns:
        ID of the created neuron
        """
        # Generate ID if not provided
        if neuron_id is None:
            neuron_id = f"{self.network_id}_neuron_{len(self.neurons)}"
            
        # Create the neuron
        neuron = Neuron(
            neuron_id=neuron_id,
            config=neuron_config,
            device=self.device
        )
        
        # Add to tracking
        self.neurons[neuron_id] = neuron
        
        if is_input:
            self.input_neuron_ids.append(neuron_id)
        if is_output:
            self.output_neuron_ids.append(neuron_id)
            
        return neuron_id
    
    def add_custom_synapse(
        self,
        pre_neuron_id: str,
        post_neuron_id: str,
        synapse_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a custom synapse to the network.
        
        Parameters:
        pre_neuron_id: ID of the presynaptic neuron
        post_neuron_id: ID of the postsynaptic neuron
        synapse_config: Configuration for the synapse
        
        Returns:
        ID of the created synapse
        """
        # Verify neurons exist
        if pre_neuron_id not in self.neurons or post_neuron_id not in self.neurons:
            raise ValueError("Pre or post neuron does not exist in the network")
            
        # Create synapse ID
        synapse_id = f"{pre_neuron_id}_to_{post_neuron_id}"
        
        # Create synapse
        synapse = Synapse(
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            synapse_id=synapse_id,
            config=synapse_config,
            device=self.device
        )
        
        # Add to tracking
        self.synapses[synapse_id] = synapse
        
        # Register input connection
        self.neurons[post_neuron_id].add_input_connection(pre_neuron_id)
        
        return synapse_id
    
    def add_custom_cluster(
        self,
        cluster_id: Optional[str] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        connect_to_network: bool = True
    ) -> str:
        """
        Add a custom cluster to the network.
        
        Parameters:
        cluster_id: Optional ID for the cluster
        cluster_config: Configuration for the cluster
        connect_to_network: Whether to connect the cluster to existing network
        
        Returns:
        ID of the created cluster
        """
        # Generate ID if not provided
        if cluster_id is None:
            cluster_id = f"{self.network_id}_cluster_{len(self.clusters)}"
            
        # Create the cluster
        cluster = NeuralCluster(
            cluster_id=cluster_id,
            config=cluster_config,
            device=self.device
        )
        
        # Add to tracking
        self.clusters[cluster_id] = cluster
        
        # Add neurons and synapses to tracking
        self.neurons.update(cluster.neurons)
        self.synapses.update(cluster.synapses)
        
        # Connect to network if requested
        if connect_to_network and (self.input_neuron_ids or self.output_neuron_ids):
            # Connect inputs to cluster if appropriate
            if self.input_neuron_ids and cluster.input_neuron_ids:
                for in_id in self.input_neuron_ids:
                    for cluster_in_id in cluster.input_neuron_ids:
                        if np.random.random() < self.config.connection_density * 0.5:
                            self.add_custom_synapse(
                                pre_neuron_id=in_id,
                                post_neuron_id=cluster_in_id
                            )
            
            # Connect cluster to outputs if appropriate
            if self.output_neuron_ids and cluster.output_neuron_ids:
                for cluster_out_id in cluster.output_neuron_ids:
                    for out_id in self.output_neuron_ids:
                        if np.random.random() < self.config.connection_density * 0.5:
                            self.add_custom_synapse(
                                pre_neuron_id=cluster_out_id,
                                post_neuron_id=out_id
                            )
        
        return cluster_id
    
    def process(
        self,
        inputs: Union[List[float], np.ndarray, torch.Tensor],
        learning_enabled: Optional[bool] = None
    ) -> List[float]:
        """
        Process inputs through the neural network.
        
        Parameters:
        inputs: Input values for the network
        learning_enabled: Override for learning enablement
        
        Returns:
        Output activations from the network
        """
        # Convert inputs to list if needed
        if isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().tolist()
            
        # Check input size
        if len(inputs) != self.config.input_size:
            logger.warning(
                f"Input size mismatch: expected {self.config.input_size}, "
                f"got {len(inputs)}"
            )
            # Pad or truncate inputs
            if len(inputs) < self.config.input_size:
                inputs = inputs + [0.0] * (self.config.input_size - len(inputs))
            else:
                inputs = inputs[:self.config.input_size]
        
        # Update simulation time
        self.current_time += self.time_step
        
        # Determine learning state
        learning_active = self.learning_enabled
        if learning_enabled is not None:
            learning_active = learning_enabled
        
        # If using clusters, process through them
        if self.clusters:
            output_values = [0.0] * self.config.output_size
            
            # Set inputs to all input clusters
            for cluster_id in self.input_cluster_ids:
                self.clusters[cluster_id].set_input(inputs)
                
            # Process all clusters
            for cluster_id, cluster in self.clusters.items():
                cluster_outputs = cluster.process(self.current_time)
                
                # Collect outputs from output clusters
                if cluster_id in self.output_cluster_ids:
                    for i, val in enumerate(cluster_outputs):
                        if i < len(output_values):
                            output_values[i] += val  # Sum outputs from multiple clusters
            
            # If multiple output clusters, average the results
            if len(self.output_cluster_ids) > 1:
                output_values = [val / len(self.output_cluster_ids) for val in output_values]
                
            return output_values
        
        # Otherwise process through individual neurons
        # Set input neuron activations
        for i, neuron_id in enumerate(self.input_neuron_ids):
            if i < len(inputs):
                self.neurons[neuron_id].activation = inputs[i]
                
                # Add noise if configured
                if self.config.activation_noise > 0:
                    noise = np.random.normal(0, self.config.activation_noise)
                    self.neurons[neuron_id].activation = max(0, min(1, 
                                                               self.neurons[neuron_id].activation + noise))
        
        # Process activation through network
        output_values = [0.0] * self.config.output_size
        
        # Process all synapses to propagate signals
        for synapse_id, synapse in self.synapses.items():
            pre_id = synapse.pre_neuron_id
            pre_activation = self.neurons[pre_id].activation
            
            # Transmit through synapse
            weighted_value = synapse.transmit(pre_activation, self.current_time)
            
            # Send to post-synaptic neuron
            post_id = synapse.post_neuron_id
            self.neurons[post_id].receive_input(pre_id, pre_activation, synapse.weight.item())
        
        # Compute activations for output neurons
        for i, neuron_id in enumerate(self.output_neuron_ids):
            if i < len(output_values):
                activation = self.neurons[neuron_id].compute_activation(self.current_time)
                output_values[i] = activation
                
                # Track activation history
                if neuron_id not in self.activation_history:
                    self.activation_history[neuron_id] = []
                self.activation_history[neuron_id].append(activation)
                
                # Trim history if needed
                if len(self.activation_history[neuron_id]) > 100:
                    self.activation_history[neuron_id] = self.activation_history[neuron_id][-100:]
        
        # Apply Hebbian learning if enabled
        if learning_active:
            for synapse_id, synapse in self.synapses.items():
                if synapse.config.plasticity_enabled:
                    pre_id = synapse.pre_neuron_id
                    post_id = synapse.post_neuron_id
                    
                    pre_activation = self.neurons[pre_id].activation
                    post_activation = self.neurons[post_id].activation
                    
                    self.hebbian_learner.update_synapse(synapse, pre_activation, post_activation)
        
        return output_values
    
    def reset(self) -> None:
        """Reset the state of the network."""
        # Reset neurons
        for neuron_id, neuron in self.neurons.items():
            neuron.reset_state()
            
        # Reset synapses
        for synapse_id, synapse in self.synapses.items():
            synapse.reset()
            
        # Reset clusters
        for cluster_id, cluster in self.clusters.items():
            cluster.reset()
            
        # Reset Hebbian learner
        self.hebbian_learner.reset()
        
        # Reset simulation state
        self.current_time = 0.0
        self.activation_history = {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the network.
        
        Returns:
        Dictionary containing network state
        """
        # Only include essential state to avoid excessive size
        basic_neuron_states = {
            neuron_id: {
                "activation": neuron.activation,
                "bias": neuron.bias.item()
            } for neuron_id, neuron in self.neurons.items()
        }
        
        basic_synapse_states = {
            synapse_id: {
                "weight": synapse.weight.item(),
                "plasticity_enabled": synapse.config.plasticity_enabled
            } for synapse_id, synapse in self.synapses.items()
        }
        
        cluster_states = {
            cluster_id: {
                "cluster_type": cluster.config.cluster_type.name,
                "neuron_count": len(cluster.neurons)
            } for cluster_id, cluster in self.clusters.items()
        }
        
        return {
            "network_id": self.network_id,
            "network_type": self.config.network_type.name,
            "current_time": self.current_time,
            "learning_enabled": self.learning_enabled,
            "hebbian_rule": self.hebbian_learner.rule.name,
            "neuron_count": len(self.neurons),
            "synapse_count": len(self.synapses),
            "cluster_count": len(self.clusters),
            "neurons": basic_neuron_states,
            "synapses": basic_synapse_states,
            "clusters": cluster_states
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load network state from a dictionary.
        
        Parameters:
        state: Dictionary containing network state
        """
        # Load global state
        if "current_time" in state:
            self.current_time = state["current_time"]
        if "learning_enabled" in state:
            self.learning_enabled = state["learning_enabled"]
        
        # Load neuron states
        if "neurons" in state:
            for neuron_id, neuron_state in state["neurons"].items():
                if neuron_id in self.neurons:
                    self.neurons[neuron_id].load_state(neuron_state)
        
        # Load synapse states
        if "synapses" in state:
            for synapse_id, synapse_state in state["synapses"].items():
                if synapse_id in self.synapses:
                    self.synapses[synapse_id].load_state(synapse_state)
        
        # Load cluster states
        if "clusters" in state:
            for cluster_id, cluster_state in state["clusters"].items():
                if cluster_id in self.clusters:
                    self.clusters[cluster_id].load_state(cluster_state)
    
    def set_learning_enabled(self, enabled: bool) -> None:
        """
        Enable or disable learning in the network.
        
        Parameters:
        enabled: Whether learning should be enabled
        """
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'} in network {self.network_id}")
    
    def set_hebbian_rule(self, rule: HebbianRule) -> None:
        """
        Set the Hebbian learning rule for the network.
        
        Parameters:
        rule: Hebbian rule to use
        """
        self.hebbian_learner.set_rule(rule)
        logger.info(f"Hebbian rule set to {rule.name} in network {self.network_id}")
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate for the network.
        
        Parameters:
        learning_rate: New learning rate
        """
        self.hebbian_learner.set_learning_rate(learning_rate)
        logger.info(f"Learning rate set to {learning_rate} in network {self.network_id}")
    
    def set_plasticity_for_synapse(self, synapse_id: str, enabled: bool) -> None:
        """
        Enable or disable plasticity for a specific synapse.
        
        Parameters:
        synapse_id: ID of the synapse
        enabled: Whether plasticity should be enabled
        """
        if synapse_id in self.synapses:
            self.synapses[synapse_id].config.plasticity_enabled = enabled
