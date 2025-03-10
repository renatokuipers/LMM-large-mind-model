"""
Neural substrate module for the LMM project.

This module provides the foundational neural components upon which the
cognitive architecture is built. It implements neurons, synapses, neural clusters,
networks, and learning mechanisms.
"""

from lmm_project.neural_substrate.activation_functions import (
    ActivationType,
    get_activation_function,
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    elu,
    gaussian
)

from lmm_project.neural_substrate.neuron import (
    Neuron,
    NeuronConfig
)

from lmm_project.neural_substrate.synapse import (
    Synapse,
    SynapseConfig,
    PlasticityType
)

from lmm_project.neural_substrate.neural_cluster import (
    NeuralCluster,
    ClusterConfig,
    ClusterType
)

from lmm_project.neural_substrate.neural_network import (
    NeuralNetwork,
    NetworkConfig,
    NetworkType
)

from lmm_project.neural_substrate.hebbian_learning import (
    HebbianLearner,
    HebbianRule
)

__all__ = [
    # Activation functions
    'ActivationType', 'get_activation_function',
    'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'gaussian',
    
    # Neuron components
    'Neuron', 'NeuronConfig',
    
    # Synapse components
    'Synapse', 'SynapseConfig', 'PlasticityType',
    
    # Cluster components
    'NeuralCluster', 'ClusterConfig', 'ClusterType',
    
    # Network components
    'NeuralNetwork', 'NetworkConfig', 'NetworkType',
    
    # Learning components
    'HebbianLearner', 'HebbianRule'
]
