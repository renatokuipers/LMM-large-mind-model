import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from uuid import UUID, uuid4
from datetime import datetime
import time

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.neural_substrate.neural_network import NeuralNetwork, NetworkType
from lmm_project.neural_substrate.neural_cluster import NeuralCluster, ClusterType
from lmm_project.neural_substrate.activation_functions import ActivationType
from lmm_project.neural_substrate.hebbian_learning import HebbianLearner, HebbianRule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

from .models import (
    MemoryType,
    MemoryConfig,
    MemoryItem,
    WorkingMemoryItem
)

# Initialize logger
logger = get_module_logger("modules.memory.neural_net")

class MemoryNetwork:
    """
    Neural networks for memory operations that build on the neural substrate.
    Provides specialized networks for memory encoding, retrieval, association,
    and consolidation.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the memory neural network.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the network
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Networks for different memory operations
        self._encoding_network: Optional[NeuralNetwork] = None
        self._retrieval_network: Optional[NeuralNetwork] = None
        self._consolidation_network: Optional[NeuralNetwork] = None
        self._association_network: Optional[NeuralNetwork] = None
        
        # Neural clusters for memory functions
        self._function_clusters: Dict[str, NeuralCluster] = {}
        
        # Hebbian learning for memory associations
        self._hebbian_learning = HebbianLearner()
        
        # Recent activations
        self._recent_activations: Dict[str, List[float]] = {
            "encoding": [],
            "retrieval": [],
            "consolidation": [],
            "association": []
        }
        
        # Create initial networks based on developmental age
        self._initialize_networks()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Memory neural network initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("memory_encoding_requested", self._handle_encoding_request)
        self._event_bus.subscribe("memory_retrieval_requested", self._handle_retrieval_request)
        self._event_bus.subscribe("memory_consolidation_requested", self._handle_consolidation_request)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        self._event_bus.subscribe("memory_item_accessed", self._handle_memory_access)
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks based on developmental age"""
        # Encoding network (develops early)
        self._create_encoding_network()
        
        # Retrieval network (develops next)
        if self._developmental_age >= 0.1:
            self._create_retrieval_network()
        
        # Consolidation network (develops later)
        if self._developmental_age >= 0.3:
            self._create_consolidation_network()
            
        # Association network (develops last)
        if self._developmental_age >= 0.4:
            self._create_association_network()
            
        # Create function clusters
        self._create_function_clusters()
    
    def _create_encoding_network(self) -> None:
        """Create the memory encoding neural network"""
        # Skip if network already exists
        if self._encoding_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create network
        input_size = int(self._config.embedding_dimension * 0.5)  # Half of embedding dimension
        hidden_size = int(30 * age_factor)  # Size of hidden layer
        output_size = int(40 * age_factor)  # Size of encoded memory
        
        # Create network configuration
        config = {
            "network_type": NetworkType.FEEDFORWARD,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers": [hidden_size],
            "activation_type": ActivationType.RELU,
            "plasticity_enabled": True,
            "connection_density": 0.6,
            "hebbian_rule": HebbianRule.BASIC,
            "initial_weight_range": (0.01, 0.1)  # Ensure positive weights
        }
        
        # Create network with appropriate activation functions
        try:
            self._encoding_network = NeuralNetwork(
                network_id="memory_encoding",
                config=config
            )
            logger.debug(f"Created encoding network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
        except Exception as e:
            logger.error(f"Failed to initialize encoding network: {e}")
            self._encoding_network = None
    
    def _create_retrieval_network(self) -> None:
        """Create the memory retrieval neural network"""
        # Skip if network already exists
        if self._retrieval_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create network
        input_size = int(self._config.embedding_dimension * 0.3)  # Query embedding
        hidden_size = int(25 * age_factor)  # Size of hidden layer
        output_size = int(self._config.embedding_dimension * 0.8)  # Retrieved memory
        
        # Create network configuration
        config = {
            "network_type": NetworkType.FEEDFORWARD,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers": [hidden_size],
            "activation_type": ActivationType.SIGMOID,
            "plasticity_enabled": True,
            "connection_density": 0.7,
            "hebbian_rule": HebbianRule.BASIC,
            "initial_weight_range": (0.01, 0.1)  # Ensure positive weights
        }
        
        # Create network with appropriate activation functions
        try:
            self._retrieval_network = NeuralNetwork(
                network_id="memory_retrieval",
                config=config
            )
            logger.debug(f"Created retrieval network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval network: {e}")
            self._retrieval_network = None
    
    def _create_consolidation_network(self) -> None:
        """Create the memory consolidation neural network"""
        # Skip if network already exists
        if self._consolidation_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create network
        input_size = int(self._config.embedding_dimension * 0.6)  # Working memory embedding
        hidden_size = int(35 * age_factor)  # Size of hidden layer
        output_size = int(self._config.embedding_dimension * 0.9)  # Consolidated memory
        
        # Create network configuration
        config = {
            "network_type": NetworkType.FEEDFORWARD,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers": [hidden_size],
            "activation_type": ActivationType.TANH,
            "plasticity_enabled": True,
            "connection_density": 0.5,
            "hebbian_rule": HebbianRule.BASIC,
            "initial_weight_range": (0.01, 0.1)  # Ensure positive weights
        }
        
        # Create network with appropriate activation functions
        try:
            self._consolidation_network = NeuralNetwork(
                network_id="memory_consolidation",
                config=config
            )
            logger.debug(f"Created consolidation network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
        except Exception as e:
            logger.error(f"Failed to initialize consolidation network: {e}")
            self._consolidation_network = None
    
    def _create_association_network(self) -> None:
        """Create the memory association neural network"""
        # Skip if network already exists
        if self._association_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create network
        input_size = int(self._config.embedding_dimension * 0.4)  # Memory embedding
        hidden_size = int(20 * age_factor)  # Size of hidden layer
        output_size = int(self._config.embedding_dimension * 0.4)  # Associated memory
        
        # Create network configuration
        config = {
            "network_type": NetworkType.FEEDFORWARD,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers": [hidden_size],
            "activation_type": ActivationType.SIGMOID,
            "plasticity_enabled": True,
            "connection_density": 0.4,
            "hebbian_rule": HebbianRule.BASIC,
            "initial_weight_range": (0.01, 0.1)  # Ensure positive weights
        }
        
        # Create network with appropriate activation functions
        try:
            self._association_network = NeuralNetwork(
                network_id="memory_association",
                config=config
            )
            logger.debug(f"Created association network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
        except Exception as e:
            logger.error(f"Failed to initialize association network: {e}")
            self._association_network = None
    
    def _create_function_clusters(self) -> None:
        """Create neural clusters for specific memory functions"""
        # Define clusters based on developmental stage
        clusters_to_create = []
        
        # Basic encoding/storage (always available)
        clusters_to_create.append(("encoding", 12))
        clusters_to_create.append(("short_term_storage", 10))
        
        # Retrieval (develops around 0.2)
        if self._developmental_age >= 0.2:
            clusters_to_create.append(("retrieval_cue", 10))
            clusters_to_create.append(("pattern_completion", 12))
        
        # Consolidation (develops around 0.3)
        if self._developmental_age >= 0.3:
            clusters_to_create.append(("working_to_long_term", 14))
            clusters_to_create.append(("memory_stabilization", 10))
        
        # Association (develops around 0.4)
        if self._developmental_age >= 0.4:
            clusters_to_create.append(("associative_binding", 15))
            
        # Advanced memory functions (develops around 0.6)
        if self._developmental_age >= 0.6:
            clusters_to_create.append(("episodic_context", 12))
            clusters_to_create.append(("semantic_abstraction", 14))
        
        # Create clusters
        for cluster_name, size in clusters_to_create:
            if cluster_name not in self._function_clusters:
                try:
                    self._function_clusters[cluster_name] = NeuralCluster(
                        cluster_id=f"memory_{cluster_name}",
                        config={
                            "cluster_type": ClusterType.FEED_FORWARD,
                            "neuron_count": size,
                            "input_size": int(size * 0.7),
                            "output_size": int(size * 0.3),
                            "inhibitory_ratio": 0.3,
                            "plasticity_enabled": True,
                            "initial_weight_range": (0.01, 0.1)  # Ensure positive weights
                        }
                    )
                    
                    logger.debug(f"Created neural cluster: {cluster_name} with {size} neurons")
                except Exception as e:
                    logger.error(f"Failed to create neural cluster {cluster_name}: {e}")
                    # Continue with other clusters
    
    def _handle_encoding_request(self, event: Message) -> None:
        """
        Handle a memory encoding request.
        
        Args:
            event: The event containing data to encode
        """
        try:
            # Extract data
            memory_data = event.data.get("memory_data")
            if not memory_data:
                return
                
            # Skip if encoding network not initialized
            if not self._encoding_network:
                logger.warning("Encoding request received but network not initialized")
                return
                
            # Convert data to features
            features = self._extract_features(memory_data)
            
            if not features:
                logger.warning("Could not extract features from memory data")
                return
                
            # Process in encoding network
            result = self.encode_memory(features)
            
            # Publish result
            self._event_bus.publish("memory_encoding_result", {
                "result": result.tolist() if isinstance(result, np.ndarray) else result,
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling encoding request: {e}")
    
    def _handle_retrieval_request(self, event: Message) -> None:
        """
        Handle a memory retrieval request.
        
        Args:
            event: The event containing retrieval request
        """
        try:
            # Extract query
            query_data = event.data.get("query")
            if not query_data:
                return
                
            # Skip if retrieval network not initialized
            if not self._retrieval_network:
                logger.warning("Retrieval request received but network not initialized")
                return
                
            # Convert query to features
            features = self._extract_features(query_data)
            
            if not features:
                logger.warning("Could not extract features from query data")
                return
                
            # Process in retrieval network
            result = self.retrieve_memory(features)
            
            # Publish result
            self._event_bus.publish("memory_retrieval_neural_result", {
                "result": result.tolist() if isinstance(result, np.ndarray) else result,
                "query": query_data,
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling retrieval request: {e}")
    
    def _handle_consolidation_request(self, event: Message) -> None:
        """
        Handle a memory consolidation request.
        
        Args:
            event: The event containing consolidation data
        """
        try:
            # Extract working memory data
            memory_item = event.data.get("memory_item")
            if not memory_item:
                return
                
            # Skip if consolidation network not initialized
            if not self._consolidation_network:
                logger.warning("Consolidation request received but network not initialized")
                return
                
            # Process in consolidation network
            result = self.consolidate_memory(memory_item)
            
            # Publish result
            self._event_bus.publish("memory_consolidation_neural_result", {
                "result": result.tolist() if isinstance(result, np.ndarray) else result,
                "memory_id": memory_item.get("id"),
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling consolidation request: {e}")
    
    def _handle_memory_access(self, event: Message) -> None:
        """
        Handle a memory access event for learning.
        
        Args:
            event: The event containing memory access data
        """
        try:
            # Extract memory data and retrieve patterns
            memory_data = event.data.get("memory_item")
            if not memory_data or not memory_data.get("vector_embedding"):
                return
                
            # Learn from memory access (strengthen retrieval patterns)
            if self._retrieval_network:
                # Extract features from memory
                features = self._extract_features(memory_data)
                
                if features and len(features) > 0:
                    # Create scaled vector for network
                    input_vec = self._scale_to_network_inputs(
                        np.array(features),
                        self._retrieval_network.input_size
                    )
                    
                    # Get embedding
                    embedding = np.array(memory_data.get("vector_embedding"))
                    embedding = self._scale_to_network_inputs(
                        embedding,
                        self._retrieval_network.output_size
                    )
                    
                    # Train retrieval network with this sample
                    # (helps it learn to retrieve this memory in the future)
                    self._retrieval_network.forward(input_vec)
                    self._retrieval_network.backward(embedding)
                    
                    logger.debug(f"Learned from memory access for item {memory_data.get('id')}")
        except Exception as e:
            logger.error(f"Error handling memory access for learning: {e}")
    
    def _handle_age_update(self, event: Message) -> None:
        """
        Handle development age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.data.get("new_age")
        if new_age is not None:
            self.update_developmental_age(new_age)
    
    def encode_memory(self, features: List[float]) -> np.ndarray:
        """
        Encode memory features into neural representation.
        
        Args:
            features: Memory features to encode
            
        Returns:
            Encoded memory representation
        """
        if not self._encoding_network:
            logger.warning("Encoding network not initialized")
            return np.zeros(10)  # Default empty encoding
            
        # Create input vector
        input_vec = self._scale_to_network_inputs(
            np.array(features),
            self._encoding_network.input_size
        )
        
        # Forward pass through network
        activation = self._encoding_network.forward(input_vec)
        
        # Record activation
        self._recent_activations["encoding"].append(np.mean(activation))
        if len(self._recent_activations["encoding"]) > 50:
            self._recent_activations["encoding"].pop(0)
        
        # Activate relevant clusters
        self._activate_function_clusters(
            "encoding", np.mean(activation)
        )
        
        # Publish activation event
        self._publish_neural_activation_event(
            "encoding_network", np.mean(activation)
        )
        
        return activation
    
    def retrieve_memory(self, query_features: List[float]) -> np.ndarray:
        """
        Generate memory retrieval pattern from query.
        
        Args:
            query_features: Query features
            
        Returns:
            Retrieval pattern
        """
        if not self._retrieval_network:
            logger.warning("Retrieval network not initialized")
            return np.zeros(10)  # Default empty pattern
            
        # Create input vector
        input_vec = self._scale_to_network_inputs(
            np.array(query_features),
            self._retrieval_network.input_size
        )
        
        # Forward pass through network
        activation = self._retrieval_network.forward(input_vec)
        
        # Record activation
        self._recent_activations["retrieval"].append(np.mean(activation))
        if len(self._recent_activations["retrieval"]) > 50:
            self._recent_activations["retrieval"].pop(0)
        
        # Activate relevant clusters
        self._activate_function_clusters(
            "retrieval", np.mean(activation)
        )
        
        # Publish activation event
        self._publish_neural_activation_event(
            "retrieval_network", np.mean(activation)
        )
        
        return activation
    
    def consolidate_memory(self, memory_item: Dict[str, Any]) -> np.ndarray:
        """
        Consolidate working memory to long-term representation.
        
        Args:
            memory_item: Working memory item to consolidate
            
        Returns:
            Consolidated memory representation
        """
        if not self._consolidation_network:
            logger.warning("Consolidation network not initialized")
            return np.zeros(10)  # Default empty representation
            
        # Extract features from memory item
        features = self._extract_features(memory_item)
        
        if not features or len(features) == 0:
            logger.warning("Could not extract features for consolidation")
            return np.zeros(10)
            
        # Create input vector
        input_vec = self._scale_to_network_inputs(
            np.array(features),
            self._consolidation_network.input_size
        )
        
        # Forward pass through network
        activation = self._consolidation_network.forward(input_vec)
        
        # Record activation
        self._recent_activations["consolidation"].append(np.mean(activation))
        if len(self._recent_activations["consolidation"]) > 50:
            self._recent_activations["consolidation"].pop(0)
        
        # Activate relevant clusters
        self._activate_function_clusters(
            "consolidation", np.mean(activation)
        )
        
        # Publish activation event
        self._publish_neural_activation_event(
            "consolidation_network", np.mean(activation)
        )
        
        return activation
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the network.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        
        # Update networks based on new age
        self._initialize_networks()
        
        logger.info(f"Memory network age updated to {new_age}")
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """
        Extract features from memory data.
        
        Args:
            data: Memory data
            
        Returns:
            Extracted features
        """
        features = []
        
        # Try to use vector embedding if available
        if "vector_embedding" in data and data["vector_embedding"]:
            return data["vector_embedding"]
            
        # Extract numeric features
        for key, value in data.items():
            if key in ["id", "timestamp", "created_at", "last_accessed"]:
                continue  # Skip metadata
                
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple hash-based encoding of strings
                # (In a real system, this would use proper text embedding)
                hash_val = hash(value) % 1000 / 1000.0  # Simple hash normalization
                features.append(hash_val)
                
        # If no features extracted, return default
        if not features:
            return [0.5] * 10  # Default feature vector
            
        return features
    
    def _scale_to_network_inputs(
        self, 
        vector: np.ndarray, 
        target_size: int
    ) -> np.ndarray:
        """
        Scale a feature vector to match the input size of a network.
        
        Args:
            vector: The input vector
            target_size: The target size
            
        Returns:
            Scaled vector matching target size
        """
        current_size = vector.shape[0]
        
        if current_size == target_size:
            return vector
        elif current_size > target_size:
            # Downsample by averaging
            return vector[:target_size]
        else:
            # Upsample by padding with zeros
            result = np.zeros(target_size)
            result[:current_size] = vector
            return result
    
    def _activate_function_clusters(
        self,
        activation_type: str,
        intensity: float
    ) -> None:
        """
        Activate relevant function clusters based on input.
        
        Args:
            activation_type: Type of activation ('encoding', 'retrieval', 'consolidation')
            intensity: Activation intensity
        """
        # Determine which clusters to activate based on activation type
        clusters_to_activate = []
        
        if activation_type == "encoding":
            clusters_to_activate.append("encoding")
            clusters_to_activate.append("short_term_storage")
            
        elif activation_type == "retrieval":
            # Retrieval clusters develop around 0.2
            if self._developmental_age >= 0.2:
                clusters_to_activate.append("retrieval_cue")
                clusters_to_activate.append("pattern_completion")
            
            # Associative retrieval develops later
            if self._developmental_age >= 0.4:
                clusters_to_activate.append("associative_binding")
                
        elif activation_type == "consolidation":
            # Consolidation clusters develop around 0.3
            if self._developmental_age >= 0.3:
                clusters_to_activate.append("working_to_long_term")
                clusters_to_activate.append("memory_stabilization")
                
            # Advanced memory functions develop around 0.6
            if self._developmental_age >= 0.6:
                clusters_to_activate.append("episodic_context")
                clusters_to_activate.append("semantic_abstraction")
        
        # Activate selected clusters
        for cluster_name in clusters_to_activate:
            if cluster_name in self._function_clusters:
                cluster = self._function_clusters[cluster_name]
                
                # Create cluster input (use intensity for all neurons)
                cluster_input = np.ones(cluster.size) * intensity
                
                # Activate cluster
                activation = cluster.activate(cluster_input)
                
                # Publish cluster activation event
                self._publish_neural_activation_event(
                    f"cluster_{cluster_name}", np.mean(activation)
                )
    
    def _publish_neural_activation_event(self, network_name: str, activation_level: float) -> None:
        """
        Publish an event for neural activation.
        
        Args:
            network_name: Name of the activated network
            activation_level: Level of activation
        """
        self._event_bus.publish(
            "neural_activation",
            {
                "module": "memory",
                "network": network_name,
                "activation": activation_level,
                "developmental_age": self._developmental_age,
                "timestamp": "now"  # In production would use actual timestamp
            }
        )
    
    def get_encoding_network(self) -> Optional[NeuralNetwork]:
        """
        Get the encoding network.
        
        Returns:
            Encoding network or None if not initialized
        """
        return self._encoding_network
    
    def get_retrieval_network(self) -> Optional[NeuralNetwork]:
        """
        Get the retrieval network.
        
        Returns:
            Retrieval network or None if not initialized
        """
        return self._retrieval_network
    
    def get_consolidation_network(self) -> Optional[NeuralNetwork]:
        """
        Get the consolidation network.
        
        Returns:
            Consolidation network or None if not initialized
        """
        return self._consolidation_network
    
    def get_association_network(self) -> Optional[NeuralNetwork]:
        """
        Get the association network.
        
        Returns:
            Association network or None if not initialized
        """
        return self._association_network
    
    def get_function_clusters(self) -> Dict[str, NeuralCluster]:
        """
        Get the function-specific neural clusters.
        
        Returns:
            Dictionary of neural clusters by name
        """
        return self._function_clusters.copy()
    
    def get_recent_activations(self) -> Dict[str, List[float]]:
        """
        Get recent network activations.
        
        Returns:
            Dictionary of recent activations by network type
        """
        return self._recent_activations.copy()
