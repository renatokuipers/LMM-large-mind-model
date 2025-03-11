import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from uuid import UUID, uuid4

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neural_cluster import NeuralCluster, ClusterType
from lmm_project.neural_substrate.activation_functions import ActivationFunction, ActivationType
from lmm_project.core.event_bus import EventBus

from .models import (
    AttentionMode,
    AttentionTarget,
    AttentionFocus,
    SalienceAssessment,
    AttentionConfig
)

# Initialize logger
logger = get_module_logger("modules.attention.neural_net")

class AttentionNetwork:
    """
    Neural network for attention processing that builds on the neural substrate.
    Provides specialized networks for attention-related tasks like
    salience detection, focus maintenance, and distraction inhibition.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config: Optional[AttentionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the attention neural network.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the network
            developmental_age: Current developmental age of the mind
        """
        self._config = config or AttentionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Networks for different attention functions
        self._focus_network: Optional[NeuralNetwork] = None
        self._salience_network: Optional[NeuralNetwork] = None
        self._inhibition_network: Optional[NeuralNetwork] = None
        
        # Neural clusters for specific functions
        self._function_clusters: Dict[str, NeuralCluster] = {}
        
        # Recent activations
        self._recent_activations: Dict[str, List[float]] = {
            "focus": [],
            "salience": [],
            "inhibition": []
        }
        
        # Create initial networks based on developmental age
        self._initialize_networks()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Attention neural network initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("salience_assessment_created", self._handle_salience_assessment)
        self._event_bus.subscribe("attention_focus_shifted", self._handle_focus_shift)
        self._event_bus.subscribe("attention_focus_cleared", self._handle_focus_clear)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks based on developmental age"""
        # Salience network (develops early)
        self._create_salience_network()
        
        # Focus network (develops next)
        if self._developmental_age >= 0.1:
            self._create_focus_network()
        
        # Inhibition network (develops later)
        if self._developmental_age >= 0.3:
            self._create_inhibition_network()
            
        # Create function clusters
        self._create_function_clusters()
    
    def _create_salience_network(self) -> None:
        """Create the salience detection neural network"""
        # Skip if network already exists
        if self._salience_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create network
        input_size = int(20 * age_factor)  # Number of feature dimensions
        hidden_size = int(30 * age_factor)  # Size of hidden layer
        output_size = 1  # Salience score
        
        # Create the salience detection network
        self._salience_network = NeuralNetwork(
            network_id="attention_salience",
            config={
                "input_size": input_size,
                "hidden_layers": [hidden_size],
                "output_size": output_size,
                "activation_type": ActivationType.SIGMOID,
                "learning_rate": 0.01,
                "connection_density": 0.6
            }
        )
        
        logger.debug(f"Created salience network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
    
    def _create_focus_network(self) -> None:
        """Create the focus control neural network"""
        # Skip if network already exists
        if self._focus_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age * 2))
        
        # Create network
        input_size = int(15 * age_factor)  # Target features + context
        hidden_size = int(25 * age_factor)  # Size of hidden layer
        output_size = len(AttentionMode)  # One output per attention mode
        
        # Create network with appropriate activation functions
        self._focus_network = NeuralNetwork(
            name="attention_focus",
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=ActivationType.RELU,
            learning_rate=0.005,
            connection_density=0.7,
            plastic=True
        )
        
        logger.debug(f"Created focus network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
    
    def _create_inhibition_network(self) -> None:
        """Create the distraction inhibition neural network"""
        # Skip if network already exists
        if self._inhibition_network:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age))
        
        # Create network
        input_size = int(10 * age_factor)  # Current focus + distractor features
        hidden_size = int(20 * age_factor)  # Size of hidden layer
        output_size = 1  # Inhibition strength
        
        # Create network with appropriate activation functions
        self._inhibition_network = NeuralNetwork(
            name="attention_inhibition",
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=ActivationType.TANH,
            learning_rate=0.003,
            connection_density=0.5,
            plastic=True
        )
        
        logger.debug(f"Created inhibition network: {input_size} inputs, {hidden_size} hidden, {output_size} outputs")
    
    def _create_function_clusters(self) -> None:
        """Create neural clusters for specific attention functions"""
        # Define clusters based on developmental stage
        clusters_to_create = []
        
        # Basic salience detection (always available)
        clusters_to_create.append(("novelty_detection", 8))
        clusters_to_create.append(("intensity_detection", 6))
        
        # Focus maintenance (develops around 0.2)
        if self._developmental_age >= 0.2:
            clusters_to_create.append(("focus_maintenance", 10))
            clusters_to_create.append(("feature_binding", 12))
        
        # Selective attention (develops around 0.3)
        if self._developmental_age >= 0.3:
            clusters_to_create.append(("distractor_suppression", 8))
            clusters_to_create.append(("contextual_modulation", 10))
        
        # Divided attention (develops around 0.4)
        if self._developmental_age >= 0.4:
            clusters_to_create.append(("multi_target_tracking", 15))
            
        # Advanced attention (develops around 0.6)
        if self._developmental_age >= 0.6:
            clusters_to_create.append(("attention_switching", 12))
            clusters_to_create.append(("volitional_control", 14))
        
        # Create clusters
        for cluster_name, size in clusters_to_create:
            if cluster_name not in self._function_clusters:
                self._function_clusters[cluster_name] = NeuralCluster(
                    cluster_id=f"attention_{cluster_name}",
                    config={
                        "neuron_count": size,
                        "cluster_type": ClusterType.RECURRENT,
                        "activation_type": ActivationType.SIGMOID,
                        "plasticity_enabled": True
                    }
                )
                logger.debug(f"Created neural cluster: {cluster_name} with {size} neurons")
    
    def _handle_salience_assessment(self, event: Dict[str, Any]) -> None:
        """
        Handle a salience assessment event.
        
        Args:
            event: The event containing assessment data
        """
        try:
            # Extract assessment data
            assessment_data = event.get("assessment")
            if not assessment_data:
                return  # Silently ignore
                
            # Skip if network not initialized
            if not self._salience_network:
                return
                
            # Extract features from assessment
            features = assessment_data.get("contributing_features", [])
            salience_score = assessment_data.get("salience_score", 0.0)
            
            # Convert to feature vector
            feature_dict = {}
            for feature in features:
                feature_name = feature.get("name")
                feature_value = feature.get("value")
                if feature_name and feature_value is not None:
                    feature_dict[feature_name] = feature_value
            
            # Process in salience network if we have features
            if feature_dict:
                feature_vector = self._create_feature_vector(feature_dict)
                scaled_vector = self._scale_to_network_inputs(
                    feature_vector, self._salience_network.input_size
                )
                
                # Process in network
                activation = self._salience_network.forward(scaled_vector)
                
                # Record activation
                self._recent_activations["salience"].append(activation[0])
                if len(self._recent_activations["salience"]) > 50:
                    self._recent_activations["salience"].pop(0)
                
                # Learning step: use the actual salience score as target
                target = np.array([salience_score])
                self._salience_network.backward(target)
                
                # Activate relevant clusters
                self._activate_function_clusters(
                    "salience", feature_dict, salience_score
                )
                
                # Publish activation event
                self._publish_neural_activation_event(
                    "salience_network", activation[0]
                )
        except Exception as e:
            logger.error(f"Error handling salience assessment: {e}")
    
    def _handle_focus_shift(self, event: Dict[str, Any]) -> None:
        """
        Handle a focus shift event.
        
        Args:
            event: The focus shift event
        """
        try:
            # Extract focus data
            focus_data = event.get("focus")
            if not focus_data:
                return  # Silently ignore
                
            # Skip if network not initialized
            if not self._focus_network:
                return
                
            # Extract focus details
            mode = focus_data.get("mode")
            primary_target = focus_data.get("primary_target")
            focus_intensity = focus_data.get("focus_intensity", 0.5)
            
            if not (mode and primary_target):
                return
                
            # Create feature dictionary
            feature_dict = {
                "focus_mode": mode,
                "focus_intensity": focus_intensity,
                "target_type": primary_target.get("target_type", "unknown"),
                "priority": primary_target.get("priority", 0.5),
                "relevance": primary_target.get("relevance_score", 0.5)
            }
            
            # Process in focus network
            feature_vector = self._create_feature_vector(feature_dict)
            scaled_vector = self._scale_to_network_inputs(
                feature_vector, self._focus_network.input_size
            )
            
            # Forward pass
            activation = self._focus_network.forward(scaled_vector)
            
            # Record activation
            self._recent_activations["focus"].append(np.mean(activation))
            if len(self._recent_activations["focus"]) > 50:
                self._recent_activations["focus"].pop(0)
            
            # Create target output (one-hot encoding of attention mode)
            target = np.zeros(len(AttentionMode))
            try:
                mode_index = list(AttentionMode).index(AttentionMode(mode))
                target[mode_index] = 1.0
            except (ValueError, IndexError):
                target[0] = 1.0  # Default to first mode
            
            # Learning step
            self._focus_network.backward(target)
            
            # Activate relevant clusters
            self._activate_function_clusters(
                "focus", feature_dict, focus_intensity
            )
            
            # Publish activation event
            self._publish_neural_activation_event(
                "focus_network", np.mean(activation)
            )
        except Exception as e:
            logger.error(f"Error handling focus shift: {e}")
    
    def _handle_focus_clear(self, event: Dict[str, Any]) -> None:
        """
        Handle a focus clear event.
        
        Args:
            event: The focus clear event
        """
        try:
            # No specific data needed, just reset the network
            self._reset_focus_activity()
            
        except Exception as e:
            logger.error(f"Error handling focus clear: {e}")
    
    def _handle_age_update(self, event: Dict[str, Any]) -> None:
        """
        Handle development age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.get("new_age")
        if new_age is not None:
            self.update_developmental_age(new_age)
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the network.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        
        # Update networks based on new age
        self._initialize_networks()
        
        logger.info(f"Attention network age updated to {new_age}")
    
    def _create_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Create a feature vector from a dictionary of features.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector as numpy array
        """
        # Extract numeric features
        numeric_features = []
        for name, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features.append(value)
            elif isinstance(value, str):
                # One-hot encode string features
                # (simplified version - in production would use proper encoding)
                hash_val = hash(value) % 1000 / 1000.0  # Simple hash normalization
                numeric_features.append(hash_val)
        
        return np.array(numeric_features)
    
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
        features: Dict[str, Any],
        intensity: float
    ) -> None:
        """
        Activate relevant function clusters based on input.
        
        Args:
            activation_type: Type of activation ('salience', 'focus', 'inhibition')
            features: Feature dictionary
            intensity: Activation intensity
        """
        # Determine which clusters to activate based on activation type
        clusters_to_activate = []
        
        if activation_type == "salience":
            clusters_to_activate.append("novelty_detection")
            clusters_to_activate.append("intensity_detection")
            
            # Contextual modulation for more mature minds
            if self._developmental_age >= 0.3:
                clusters_to_activate.append("contextual_modulation")
                
        elif activation_type == "focus":
            # Focus maintenance develops around 0.2
            if self._developmental_age >= 0.2:
                clusters_to_activate.append("focus_maintenance")
                clusters_to_activate.append("feature_binding")
            
            # Mode-specific clusters
            mode = features.get("focus_mode")
            if mode == AttentionMode.DIVIDED.value and self._developmental_age >= 0.4:
                clusters_to_activate.append("multi_target_tracking")
                
            if mode == AttentionMode.ALTERNATING.value and self._developmental_age >= 0.6:
                clusters_to_activate.append("attention_switching")
                
            if self._developmental_age >= 0.6:
                clusters_to_activate.append("volitional_control")
                
        elif activation_type == "inhibition":
            # Inhibition develops around 0.3
            if self._developmental_age >= 0.3:
                clusters_to_activate.append("distractor_suppression")
        
        # Activate selected clusters
        for cluster_name in clusters_to_activate:
            if cluster_name in self._function_clusters:
                cluster = self._function_clusters[cluster_name]
                
                # Create cluster input from features
                # (simplified - just use the intensity)
                cluster_input = np.ones(cluster.neuron_count) * intensity
                
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
        try:
            from lmm_project.core.message import Message, StructuredContent
            from lmm_project.core.types import ModuleType, MessageType
            
            message = Message(
                sender="attention",
                sender_type=ModuleType.ATTENTION,
                message_type=MessageType.ATTENTION_FOCUS,
                content=StructuredContent(
                    data={
                        "module": "attention",
                        "network": network_name,
                        "activation": activation_level,
                        "developmental_age": self._developmental_age,
                        "timestamp": "now"  # In production would use actual timestamp
                    }
                ),
                priority=3
            )
            
            self._event_bus.publish(message)
        except Exception as e:
            logger.error(f"Error publishing neural activation event: {e}")
    
    def get_salience_network(self) -> Optional[NeuralNetwork]:
        """
        Get the salience detection network.
        
        Returns:
            Salience network or None if not initialized
        """
        return self._salience_network
    
    def get_focus_network(self) -> Optional[NeuralNetwork]:
        """
        Get the focus control network.
        
        Returns:
            Focus network or None if not initialized
        """
        return self._focus_network
    
    def get_inhibition_network(self) -> Optional[NeuralNetwork]:
        """
        Get the inhibition network.
        
        Returns:
            Inhibition network or None if not initialized
        """
        return self._inhibition_network
    
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

    def _reset_focus_activity(self) -> None:
        """
        Reset activations in focus-related clusters and networks.
        Used when clearing attention focus.
        """
        try:
            # Reset activations in focus clusters
            for cluster_name in ["focus_maintenance", "feature_binding", 
                              "multi_target_tracking", "attention_switching"]:
                if cluster_name in self._function_clusters:
                    self._function_clusters[cluster_name].reset()
            
            # Publish deactivation event
            self._publish_neural_activation_event("focus_network", 0.0)
        except Exception as e:
            logger.error(f"Error resetting focus activity: {e}")
