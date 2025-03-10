"""
Neural cluster implementation for the neural substrate.

This module defines the NeuralCluster class, which represents a functional
grouping of neurons with structured connectivity patterns and specialized
behavior.
"""
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from lmm_project.neural_substrate.neuron import Neuron, NeuronConfig
from lmm_project.neural_substrate.synapse import Synapse, SynapseConfig, PlasticityType
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("neural_substrate.neural_cluster")


class ClusterType(Enum):
    """Types of neural clusters with different connectivity patterns."""
    FULLY_CONNECTED = auto()
    FEED_FORWARD = auto()
    RECURRENT = auto()
    COMPETITIVE = auto()
    RESERVOIR = auto()
    MODULAR = auto()


class ClusterConfig(BaseModel):
    """Configuration for a neural cluster."""
    cluster_type: ClusterType = Field(default=ClusterType.FEED_FORWARD)
    neuron_count: int = Field(default=10, gt=0)
    input_size: int = Field(default=0, ge=0)
    output_size: int = Field(default=0, ge=0)
    hidden_layers: List[int] = Field(default=[])
    inhibitory_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    connection_density: float = Field(default=0.5, gt=0.0, le=1.0)
    plasticity_enabled: bool = Field(default=True)
    plasticity_type: PlasticityType = Field(default=PlasticityType.HEBBIAN)
    use_bias: bool = Field(default=True)
    initial_weight_range: Tuple[float, float] = Field(default=(-0.1, 0.1))
    

class NeuralCluster:
    """
    Functional grouping of neurons with structured connectivity.
    
    Neural clusters organize neurons into functional units with specific
    connectivity patterns and behaviors, representing higher-level structures
    in the neural substrate.
    """
    
    def __init__(
        self,
        cluster_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a neural cluster.
        
        Parameters:
        cluster_id: Unique identifier for this cluster
        config: Configuration dictionary
        device: Torch device for tensor operations
        """
        self.cluster_id = cluster_id or str(uuid.uuid4())
        self.config = ClusterConfig(**(config or {}))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create storage for neurons and synapses
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        
        # Track cluster structure
        self.input_neuron_ids: List[str] = []
        self.output_neuron_ids: List[str] = []
        self.hidden_neuron_ids: List[str] = []
        
        # Track cluster state
        self.activation_levels: Dict[str, float] = {}
        self.input_buffer: Dict[int, float] = {}
        self.output_buffer: Dict[int, float] = {}
        
        # Initialize neurons and connectivity
        self._initialize_cluster()
        
        logger.info(
            f"Created {self.config.cluster_type.name} cluster with "
            f"{len(self.neurons)} neurons and {len(self.synapses)} synapses"
        )
    
    def _initialize_cluster(self) -> None:
        """Initialize the cluster structure based on configuration."""
        # Create neurons based on cluster type
        self._create_neurons()
        
        # Create synaptic connections based on cluster type
        if self.config.cluster_type == ClusterType.FULLY_CONNECTED:
            self._create_fully_connected_pattern()
        elif self.config.cluster_type == ClusterType.FEED_FORWARD:
            self._create_feed_forward_pattern()
        elif self.config.cluster_type == ClusterType.RECURRENT:
            self._create_recurrent_pattern()
        elif self.config.cluster_type == ClusterType.COMPETITIVE:
            self._create_competitive_pattern()
        elif self.config.cluster_type == ClusterType.RESERVOIR:
            self._create_reservoir_pattern()
        elif self.config.cluster_type == ClusterType.MODULAR:
            self._create_modular_pattern()
        
        # Initialize activation levels
        self.activation_levels = {neuron_id: 0.0 for neuron_id in self.neurons}
        
        # Initialize I/O buffers
        if self.config.input_size > 0:
            self.input_buffer = {i: 0.0 for i in range(self.config.input_size)}
        if self.config.output_size > 0:
            self.output_buffer = {i: 0.0 for i in range(self.config.output_size)}
    
    def _create_neurons(self) -> None:
        """Create neurons according to the cluster configuration."""
        # Determine total neuron count
        total_neurons = self.config.neuron_count
        
        # Determine which neurons are inhibitory
        inhibitory_count = int(total_neurons * self.config.inhibitory_ratio)
        inhibitory_indices = set(np.random.choice(
            total_neurons, 
            size=inhibitory_count, 
            replace=False
        ))
        
        # Create all neurons
        for i in range(total_neurons):
            # Determine if this neuron is inhibitory
            is_inhibitory = i in inhibitory_indices
            
            # Create neuron config
            neuron_config = {
                "is_inhibitory": is_inhibitory,
                "threshold": 0.5,
                "bias": 0.0 if not self.config.use_bias else np.random.uniform(-0.1, 0.1)
            }
            
            # Create the neuron
            neuron_id = f"{self.cluster_id}_n{i}"
            neuron = Neuron(
                neuron_id=neuron_id,
                config=neuron_config,
                device=self.device
            )
            
            # Store the neuron
            self.neurons[neuron_id] = neuron
            
            # Categorize neuron (input, hidden, output)
            if i < self.config.input_size:
                self.input_neuron_ids.append(neuron_id)
            elif i >= (total_neurons - self.config.output_size):
                self.output_neuron_ids.append(neuron_id)
            else:
                self.hidden_neuron_ids.append(neuron_id)
    
    def _create_synapse(
        self, 
        pre_neuron_id: str, 
        post_neuron_id: str, 
        weight: Optional[float] = None
    ) -> Synapse:
        """
        Create a synapse between two neurons.
        
        Parameters:
        pre_neuron_id: ID of presynaptic neuron
        post_neuron_id: ID of postsynaptic neuron
        weight: Initial weight (if None, randomized)
        
        Returns:
        Created synapse
        """
        # Generate random weight if not provided
        if weight is None:
            min_w, max_w = self.config.initial_weight_range
            weight = np.random.uniform(min_w, max_w)
        
        # Create synapse configuration
        synapse_config = {
            "initial_weight": weight,
            "plasticity_enabled": self.config.plasticity_enabled,
            "plasticity_type": self.config.plasticity_type
        }
        
        # Create synapse
        synapse_id = f"{pre_neuron_id}_to_{post_neuron_id}"
        synapse = Synapse(
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            synapse_id=synapse_id,
            config=synapse_config,
            device=self.device
        )
        
        # Register input connection in post neuron
        self.neurons[post_neuron_id].add_input_connection(pre_neuron_id)
        
        # Store synapse
        self.synapses[synapse_id] = synapse
        
        return synapse
    
    def _create_fully_connected_pattern(self) -> None:
        """Create a fully connected pattern where every neuron connects to every other."""
        neuron_ids = list(self.neurons.keys())
        
        # Connect each neuron to every other neuron
        for pre_id in neuron_ids:
            for post_id in neuron_ids:
                # Skip self-connections
                if pre_id != post_id:
                    # Create the synapse
                    self._create_synapse(pre_id, post_id)
    
    def _create_feed_forward_pattern(self) -> None:
        """Create a feed-forward network with layers."""
        # Determine layer sizes
        layer_sizes = [self.config.input_size] + self.config.hidden_layers + [self.config.output_size]
        
        # Create connections between layers
        neuron_ids = list(self.neurons.keys())
        current_idx = 0
        
        for layer_idx in range(len(layer_sizes) - 1):
            # Get current and next layer size
            current_size = layer_sizes[layer_idx]
            next_size = layer_sizes[layer_idx + 1]
            
            # Skip if either layer is empty
            if current_size == 0 or next_size == 0:
                continue
                
            # Get neuron IDs for current layer
            current_layer_ids = neuron_ids[current_idx:current_idx + current_size]
            
            # Get neuron IDs for next layer
            next_idx = current_idx + current_size
            next_layer_ids = neuron_ids[next_idx:next_idx + next_size]
            
            # Connect each neuron in current layer to each in next layer
            for pre_id in current_layer_ids:
                for post_id in next_layer_ids:
                    # Probabilistic connection based on density
                    if np.random.random() < self.config.connection_density:
                        self._create_synapse(pre_id, post_id)
            
            # Update current index
            current_idx += current_size
    
    def _create_recurrent_pattern(self) -> None:
        """Create a recurrent network with feedback connections."""
        # First create a feed-forward network
        self._create_feed_forward_pattern()
        
        # Now add recurrent connections within each hidden layer
        for layer_idx, layer_size in enumerate(self.config.hidden_layers):
            # Skip if layer is empty
            if layer_size == 0:
                continue
                
            # Calculate start index for this hidden layer
            start_idx = self.config.input_size + sum(self.config.hidden_layers[:layer_idx])
            
            # Get neuron IDs for this layer
            layer_ids = list(self.neurons.keys())[start_idx:start_idx + layer_size]
            
            # Add recurrent connections within layer
            for pre_id in layer_ids:
                for post_id in layer_ids:
                    # Skip self-connections
                    if pre_id != post_id:
                        # Probabilistic connection based on density/2 (sparser recurrence)
                        if np.random.random() < self.config.connection_density / 2:
                            self._create_synapse(pre_id, post_id)
    
    def _create_competitive_pattern(self) -> None:
        """Create a competitive network where neurons inhibit each other."""
        # Create feed-forward pattern for input connections
        self._create_feed_forward_pattern()
        
        # Now add competitive inhibitory connections between neurons in output layer
        for i, pre_id in enumerate(self.output_neuron_ids):
            for j, post_id in enumerate(self.output_neuron_ids):
                # Skip self-connections
                if pre_id != post_id:
                    # Create inhibitory connection
                    synapse = self._create_synapse(pre_id, post_id)
                    
                    # Force weight to be negative
                    with torch.no_grad():
                        synapse.weight = torch.tensor(-0.5, device=self.device)
    
    def _create_reservoir_pattern(self) -> None:
        """Create a reservoir computing network (echo state network)."""
        # Create sparse, random connections between all neurons
        neuron_ids = list(self.neurons.keys())
        
        for pre_id in neuron_ids:
            for post_id in neuron_ids:
                # Sparse connectivity
                if np.random.random() < self.config.connection_density * 0.3:
                    self._create_synapse(pre_id, post_id)
                    
        # Create input connections
        for in_id in self.input_neuron_ids:
            for other_id in neuron_ids:
                if in_id != other_id and np.random.random() < self.config.connection_density:
                    self._create_synapse(in_id, other_id)
                    
        # Create output connections
        for hid_id in self.hidden_neuron_ids:
            for out_id in self.output_neuron_ids:
                if np.random.random() < self.config.connection_density:
                    self._create_synapse(hid_id, out_id)
    
    def _create_modular_pattern(self) -> None:
        """Create a modular network with clusters of densely connected neurons."""
        total_neurons = len(self.neurons)
        
        # Skip if too few neurons
        if total_neurons < 10:
            self._create_fully_connected_pattern()
            return
            
        # Determine number of modules
        module_count = max(2, total_neurons // 10)
        
        # Assign neurons to modules
        neuron_ids = list(self.neurons.keys())
        module_assignments = np.random.randint(0, module_count, size=total_neurons)
        
        # Create dense connections within modules, sparse between modules
        for i, pre_id in enumerate(neuron_ids):
            pre_module = module_assignments[i]
            
            for j, post_id in enumerate(neuron_ids):
                # Skip self-connections
                if pre_id == post_id:
                    continue
                    
                post_module = module_assignments[j]
                
                # Within-module connections (dense)
                if pre_module == post_module:
                    if np.random.random() < self.config.connection_density * 2:
                        self._create_synapse(pre_id, post_id)
                # Between-module connections (sparse)
                else:
                    if np.random.random() < self.config.connection_density * 0.2:
                        self._create_synapse(pre_id, post_id)
    
    def set_input(self, input_values: Union[List[float], np.ndarray, torch.Tensor]) -> None:
        """
        Set input values for the cluster.
        
        Parameters:
        input_values: Input activation values
        """
        # Convert to list if needed
        if isinstance(input_values, np.ndarray):
            input_values = input_values.tolist()
        elif isinstance(input_values, torch.Tensor):
            input_values = input_values.cpu().tolist()
            
        # Ensure correct input size
        if len(input_values) != self.config.input_size:
            logger.warning(
                f"Input size mismatch: expected {self.config.input_size}, "
                f"got {len(input_values)}"
            )
            return
            
        # Update input buffer
        for i, value in enumerate(input_values):
            self.input_buffer[i] = value
            
        # Set input neuron activations directly
        for i, neuron_id in enumerate(self.input_neuron_ids):
            if i < len(input_values):
                self.neurons[neuron_id].activation = input_values[i]
                self.activation_levels[neuron_id] = input_values[i]
    
    def process(self, time_step: float = 1.0) -> List[float]:
        """
        Process signals through the cluster.
        
        Parameters:
        time_step: Time step for simulation
        
        Returns:
        Output activations
        """
        # Current simulation time
        current_time = time_step
        
        # Process in topological order (input -> hidden -> output)
        # First, propagate signals from input neurons
        for pre_id in self.input_neuron_ids:
            pre_activation = self.neurons[pre_id].activation
            
            # Find all synapses from this neuron
            for synapse_id, synapse in self.synapses.items():
                if synapse.pre_neuron_id == pre_id:
                    # Transmit signal through synapse
                    weighted_input = synapse.transmit(pre_activation, current_time)
                    
                    # Send to post-synaptic neuron
                    post_id = synapse.post_neuron_id
                    self.neurons[post_id].receive_input(pre_id, pre_activation, synapse.weight.item())
        
        # Then process hidden neurons
        for neuron_id in self.hidden_neuron_ids:
            # Compute activation
            activation = self.neurons[neuron_id].compute_activation(current_time)
            self.activation_levels[neuron_id] = activation
            
            # Propagate signals
            for synapse_id, synapse in self.synapses.items():
                if synapse.pre_neuron_id == neuron_id:
                    # Transmit signal through synapse
                    weighted_input = synapse.transmit(activation, current_time)
                    
                    # Send to post-synaptic neuron
                    post_id = synapse.post_neuron_id
                    self.neurons[post_id].receive_input(neuron_id, activation, synapse.weight.item())
        
        # Finally, process output neurons
        output_activations = []
        for i, neuron_id in enumerate(self.output_neuron_ids):
            # Compute activation
            activation = self.neurons[neuron_id].compute_activation(current_time)
            self.activation_levels[neuron_id] = activation
            
            # Store in output buffer
            if i < len(self.output_buffer):
                self.output_buffer[i] = activation
                output_activations.append(activation)
        
        # Update plasticity
        self._update_plasticity(current_time)
        
        return output_activations
    
    def _update_plasticity(self, current_time: float) -> None:
        """
        Update synaptic weights based on plasticity rules.
        
        Parameters:
        current_time: Current simulation time
        """
        if not self.config.plasticity_enabled:
            return
            
        for synapse_id, synapse in self.synapses.items():
            if synapse.config.plasticity_enabled:
                # Get pre and post activations
                pre_id = synapse.pre_neuron_id
                post_id = synapse.post_neuron_id
                pre_activation = self.activation_levels[pre_id]
                post_activation = self.activation_levels[post_id]
                
                # Update synapse weight
                synapse.update_weight(post_activation, current_time)
    
    def reset(self) -> None:
        """Reset the state of all neurons and synapses in the cluster."""
        # Reset neurons
        for neuron_id, neuron in self.neurons.items():
            neuron.reset_state()
            self.activation_levels[neuron_id] = 0.0
            
        # Reset synapses
        for synapse_id, synapse in self.synapses.items():
            synapse.reset()
            
        # Reset I/O buffers
        self.input_buffer = {i: 0.0 for i in range(self.config.input_size)}
        self.output_buffer = {i: 0.0 for i in range(self.config.output_size)}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the cluster.
        
        Returns:
        Dictionary containing cluster state
        """
        neuron_states = {neuron_id: neuron.get_state() for neuron_id, neuron in self.neurons.items()}
        synapse_states = {synapse_id: synapse.get_state() for synapse_id, synapse in self.synapses.items()}
        
        return {
            "cluster_id": self.cluster_id,
            "cluster_type": self.config.cluster_type.name,
            "neuron_count": len(self.neurons),
            "synapse_count": len(self.synapses),
            "activation_levels": self.activation_levels.copy(),
            "neurons": neuron_states,
            "synapses": synapse_states,
            "input_buffer": self.input_buffer.copy(),
            "output_buffer": self.output_buffer.copy()
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load cluster state from a dictionary.
        
        Parameters:
        state: Dictionary containing cluster state
        """
        # Load activation levels
        if "activation_levels" in state:
            self.activation_levels = state["activation_levels"]
            
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
                    
        # Load I/O buffers
        if "input_buffer" in state:
            self.input_buffer = state["input_buffer"]
        if "output_buffer" in state:
            self.output_buffer = state["output_buffer"]
