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
    SensoryModality,
    ProcessedInput,
    RecognizedPattern,
    PatternType,
    PerceptionConfig
)

# Initialize logger
logger = get_module_logger("modules.perception.neural_net")

class PerceptionNetwork:
    """
    Neural network for perception processing that builds on the neural substrate.
    Provides specialized networks for different perception tasks like
    feature detection, pattern recognition, and sensory processing.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config: Optional[PerceptionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the perception neural network.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the network
            developmental_age: Current developmental age of the mind
        """
        self._config = config or PerceptionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Networks for different modalities
        self._modality_networks: Dict[SensoryModality, NeuralNetwork] = {}
        
        # Pattern recognition networks for different pattern types
        self._pattern_networks: Dict[PatternType, NeuralNetwork] = {}
        
        # Feature detector clusters
        self._feature_detectors: Dict[str, NeuralCluster] = {}
        
        # Create initial networks based on developmental age
        self._initialize_networks()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Perception neural network initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("sensory_input_processed", self._handle_processed_input)
        self._event_bus.subscribe("pattern_recognized", self._handle_recognized_pattern)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks based on developmental age"""
        # Create networks for active modalities
        for modality in self._config.default_modalities:
            self._create_modality_network(modality)
            
        # Create pattern networks appropriate for age
        if self._developmental_age >= 0.1:
            self._create_pattern_network(PatternType.SEMANTIC)
            
        if self._developmental_age >= 0.2:
            self._create_pattern_network(PatternType.ASSOCIATIVE)
            
        if self._developmental_age >= 0.3:
            self._create_pattern_network(PatternType.TEMPORAL)
    
    def _create_modality_network(self, modality: SensoryModality) -> None:
        """
        Create a neural network for a specific sensory modality.
        
        Args:
            modality: The sensory modality to create a network for
        """
        # Skip if network already exists
        if modality in self._modality_networks:
            return
            
        # Configure network size based on developmental age
        # (more mature networks have more neurons and connections)
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create a neural network for this modality
        network = NeuralNetwork(
            network_id=f"perception_{modality.value}",
            config={
                "input_size": int(10 * age_factor),
                "hidden_layers": [int(20 * age_factor)],
                "output_size": int(10 * age_factor),
                "activation_type": ActivationType.RELU,
                "learning_rate": 0.01,
                "batch_size": 1
            }
        )
        
        # Store the network
        self._modality_networks[modality] = network
        
        logger.info(f"Created neural network for {modality.value} modality")
    
    def _create_pattern_network(self, pattern_type: PatternType) -> None:
        """
        Create a neural network for a specific pattern type.
        
        Args:
            pattern_type: The pattern type to create a network for
        """
        # Skip if network already exists
        if pattern_type in self._pattern_networks:
            return
            
        # Configure network size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        
        # Create a neural network for this pattern type
        network = NeuralNetwork(
            network_id=f"pattern_{pattern_type.value}",
            config={
                "input_size": int(15 * age_factor),
                "hidden_layers": [int(30 * age_factor)],
                "output_size": int(15 * age_factor),
                "activation_type": ActivationType.SIGMOID,
                "learning_rate": 0.01,
                "batch_size": 1
            }
        )
        
        # Store the network
        self._pattern_networks[pattern_type] = network
        
        logger.info(f"Created neural network for {pattern_type.value} pattern type")
    
    def _create_feature_detector(self, feature_name: str, modality: SensoryModality) -> None:
        """
        Create a neural cluster for detecting a specific feature.
        
        Args:
            feature_name: The name of the feature to detect
            modality: The sensory modality of the feature
        """
        feature_key = f"{modality.value}:{feature_name}"
        
        # Skip if detector already exists
        if feature_key in self._feature_detectors:
            return
            
        # Configure cluster size based on developmental age
        age_factor = min(1.0, max(0.1, self._developmental_age + 0.1))
        size = int(10 * age_factor)
        
        # Create a neural cluster for this feature
        cluster = NeuralCluster(
            cluster_id=f"feature_{feature_name}_{modality.value}",
            config={
                "neuron_count": int(10 * age_factor),
                "cluster_type": ClusterType.FEED_FORWARD,
                "activation_type": ActivationType.SIGMOID,
                "threshold": 0.3,
                "decay_rate": 0.1
            }
        )
        
        # Store the cluster
        self._feature_detectors[feature_key] = cluster
        
        logger.debug(f"Created feature detector for {feature_key}")
    
    def _handle_processed_input(self, event: Dict[str, Any]) -> None:
        """
        Handle a processed input event by activating relevant neural networks.
        
        Args:
            event: The event containing the processed input
        """
        try:
            # Extract processed input from event
            processed_input_data = event.get("processed_input")
            if not processed_input_data:
                return
                
            # Create processed input model (already validated in pattern recognizer)
            processed_input = ProcessedInput(**processed_input_data)
            
            # Process in the appropriate modality network
            self._process_in_modality_network(processed_input)
            
            # Create feature detectors for new features
            for feature in processed_input.features:
                self._create_feature_detector(feature.name, feature.modality)
                
            # Activate feature detectors
            self._activate_feature_detectors(processed_input)
                
        except Exception as e:
            logger.error(f"Error processing input in neural network: {e}")
    
    def _handle_recognized_pattern(self, event: Dict[str, Any]) -> None:
        """
        Handle a recognized pattern event by reinforcing neural pathways.
        
        Args:
            event: The event containing the recognized pattern
        """
        try:
            # Extract pattern from event
            pattern_data = event.get("pattern")
            if not pattern_data:
                return
                
            # Create pattern model
            pattern = RecognizedPattern(**pattern_data)
            
            # Process in the appropriate pattern network
            self._process_in_pattern_network(pattern)
                
        except Exception as e:
            logger.error(f"Error processing pattern in neural network: {e}")
    
    def _handle_age_update(self, event: Dict[str, Any]) -> None:
        """
        Handle a developmental age update event by updating networks.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.get("age")
        if new_age is not None and isinstance(new_age, (int, float)):
            old_age = self._developmental_age
            self._developmental_age = float(new_age)
            
            # Check if we need to create new networks based on age
            if old_age < 0.1 <= self._developmental_age:
                self._create_pattern_network(PatternType.SEMANTIC)
                
            if old_age < 0.2 <= self._developmental_age:
                self._create_pattern_network(PatternType.ASSOCIATIVE)
                
            if old_age < 0.3 <= self._developmental_age:
                self._create_pattern_network(PatternType.TEMPORAL)
                
            if old_age < 0.5 <= self._developmental_age:
                self._create_pattern_network(PatternType.CATEGORICAL)
                
            if old_age < 0.7 <= self._developmental_age:
                if self._config.enable_emotional_processing:
                    self._create_pattern_network(PatternType.EMOTIONAL)
            
            logger.debug(f"Updated developmental age to {self._developmental_age}")
    
    def _process_in_modality_network(self, processed_input: ProcessedInput) -> None:
        """
        Process an input in the appropriate modality network.
        
        Args:
            processed_input: The processed input to process
        """
        # Skip if we don't have a network for this modality
        if processed_input.modality not in self._modality_networks:
            self._create_modality_network(processed_input.modality)
            
        # Get the network for this modality
        network = self._modality_networks[processed_input.modality]
        
        # Prepare input vector
        input_vector = self._create_input_vector(processed_input)
        
        # Process in the network
        try:
            # Forward pass
            output = network.forward(input_vector)
            
            # Backward pass (learning) - simple target is just the input for now
            # (autoencoder-like behavior for early developmental stages)
            target = input_vector
            network.backward(target)
            
            # Publish neural activation event
            self._publish_neural_activation_event(
                network_name=network.network_id,
                activation_level=float(np.mean(output))
            )
            
        except Exception as e:
            logger.error(f"Error in neural network processing: {e}")
    
    def _process_in_pattern_network(self, pattern: RecognizedPattern) -> None:
        """
        Process a recognized pattern in the appropriate pattern network.
        
        Args:
            pattern: The recognized pattern to process
        """
        # Skip if we don't have a network for this pattern type
        if pattern.pattern_type not in self._pattern_networks:
            return
            
        # Get the network for this pattern type
        network = self._pattern_networks[pattern.pattern_type]
        
        # Prepare input vector from pattern
        input_vector = self._create_pattern_vector(pattern)
        
        # Process in the network
        try:
            # Forward pass
            output = network.forward(input_vector)
            
            # Backward pass (learning)
            # Use confidence as a learning signal
            target = np.ones_like(output) * pattern.confidence
            network.backward(target)
            
            # Publish neural activation event
            self._publish_neural_activation_event(
                network_name=network.network_id,
                activation_level=float(np.mean(output))
            )
            
        except Exception as e:
            logger.error(f"Error in pattern network processing: {e}")
    
    def _activate_feature_detectors(self, processed_input: ProcessedInput) -> None:
        """
        Activate feature detectors for the features in a processed input.
        
        Args:
            processed_input: The processed input containing features
        """
        # Activate detectors for each feature
        for feature in processed_input.features:
            feature_key = f"{feature.modality.value}:{feature.name}"
            
            # Skip if we don't have a detector for this feature
            if feature_key not in self._feature_detectors:
                continue
                
            # Get the detector for this feature
            detector = self._feature_detectors[feature_key]
            
            # Activate the detector with activation level proportional to feature value
            activation_level = abs(feature.value)
            detector.activate(activation_level)
            
            # Publish neural activation event
            self._publish_neural_activation_event(
                network_name=detector.cluster_id,
                activation_level=activation_level
            )
    
    def _create_input_vector(self, processed_input: ProcessedInput) -> np.ndarray:
        """
        Create an input vector from a processed input.
        
        Args:
            processed_input: The processed input to create a vector from
            
        Returns:
            Input vector as a numpy array
        """
        # Get the network for this modality
        network = self._modality_networks[processed_input.modality]
        
        # Create a vector of the correct size for this network
        vector = np.zeros(network.input_size)
        
        # Fill the vector with feature values if available
        for i, feature in enumerate(processed_input.features):
            if i < network.input_size:
                vector[i] = feature.value
        
        # If we have a context vector, use it to populate remaining elements
        if processed_input.context_vector and network.input_size > len(processed_input.features):
            cv = np.array(processed_input.context_vector)
            # Normalize to range 0-1
            cv_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-10)
            # Resize to fit available space
            remaining_size = network.input_size - len(processed_input.features)
            if len(cv) > remaining_size:
                # Take a sampling of the context vector to fit
                stride = len(cv) // remaining_size
                samples = cv_norm[::stride][:remaining_size]
            else:
                # Pad with zeros if context vector is smaller
                samples = np.pad(cv_norm, (0, remaining_size - len(cv)))
                
            # Add to the input vector
            vector[len(processed_input.features):] = samples
        
        return vector
    
    def _create_pattern_vector(self, pattern: RecognizedPattern) -> np.ndarray:
        """
        Create an input vector from a recognized pattern.
        
        Args:
            pattern: The recognized pattern to create a vector from
            
        Returns:
            Input vector as a numpy array
        """
        # Get the network for this pattern type
        network = self._pattern_networks[pattern.pattern_type]
        
        # Create a vector of the correct size for this network
        vector = np.zeros(network.input_size)
        
        # Fill the vector with feature values if available
        for i, feature in enumerate(pattern.features):
            if i < network.input_size:
                vector[i] = feature.value
        
        # Add pattern confidence
        if len(pattern.features) < network.input_size:
            vector[len(pattern.features)] = pattern.confidence
            
        # Add pattern salience
        if len(pattern.features) + 1 < network.input_size:
            vector[len(pattern.features) + 1] = pattern.salience
            
        # If we have a vector representation, use it to populate remaining elements
        if pattern.vector_representation and network.input_size > len(pattern.features) + 2:
            vr = np.array(pattern.vector_representation)
            # Normalize to range 0-1
            vr_norm = (vr - vr.min()) / (vr.max() - vr.min() + 1e-10)
            # Resize to fit available space
            remaining_size = network.input_size - (len(pattern.features) + 2)
            if len(vr) > remaining_size:
                # Take a sampling of the vector to fit
                stride = len(vr) // remaining_size
                samples = vr_norm[::stride][:remaining_size]
            else:
                # Pad with zeros if vector is smaller
                samples = np.pad(vr_norm, (0, remaining_size - len(vr)))
                
            # Add to the input vector
            vector[len(pattern.features) + 2:] = samples
        
        return vector
    
    def _publish_neural_activation_event(self, network_name: str, activation_level: float) -> None:
        """
        Publish an event for neural network activation.
        
        Args:
            network_name: The name of the activated network
            activation_level: The level of activation
        """
        # Create event payload
        payload = {
            "network": network_name,
            "activation": activation_level,
            "developmental_age": self._developmental_age,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Create and publish event
        event = Event(
            type="neural_activation",
            source="perception.neural_net",
            data=payload
        )
        self._event_bus.publish(event)
    
    def get_modality_networks(self) -> Dict[SensoryModality, NeuralNetwork]:
        """
        Get all modality networks.
        
        Returns:
            Dictionary of modality networks by modality
        """
        return self._modality_networks.copy()
    
    def get_pattern_networks(self) -> Dict[PatternType, NeuralNetwork]:
        """
        Get all pattern networks.
        
        Returns:
            Dictionary of pattern networks by pattern type
        """
        return self._pattern_networks.copy()
    
    def get_feature_detectors(self) -> Dict[str, NeuralCluster]:
        """
        Get all feature detectors.
        
        Returns:
            Dictionary of feature detectors by feature key
        """
        return self._feature_detectors.copy()
