import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

class PerceptionNetwork(nn.Module): 
    """
    Neural network for processing textual perceptual inputs.
    
    This network processes text-based perceptual input through multiple stages:
    1. Feature extraction - Converts text tokens to feature vectors
    2. Pattern encoding - Encodes features into pattern representations
    3. Novelty detection - Identifies novel or unexpected inputs
    4. Salience estimation - Determines the importance of the input
    
    The network adapts its behavior based on developmental level.
    """
    def __init__(
        self, 
        input_dim: int = 64, 
        hidden_dim: int = 128, 
        pattern_dim: int = 32,
        developmental_level: float = 0.0
    ):
        super().__init__() 
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pattern_dim = pattern_dim
        self.developmental_level = developmental_level
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Pattern encoding network
        self.pattern_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pattern_dim)
        )
        
        # Novelty detection network
        self.novelty_detector = nn.Sequential(
            nn.Linear(pattern_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Salience estimation network
        self.salience_estimator = nn.Sequential(
            nn.Linear(pattern_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Pattern memory for tracking known patterns
        self.pattern_memory = {}
        
        # Recent patterns for novelty assessment
        self.recent_patterns = deque(maxlen=50)
        
        # Apply developmental scaling to adjust network behavior
        self._apply_developmental_scaling()
    
    def _apply_developmental_scaling(self):
        """
        Apply developmental level-based scaling to network parameters
        
        At lower developmental levels:
        - Higher dropout rates (simulating less reliable processing)
        - Lower sensitivity to subtle patterns
        - Stronger activation thresholds
        
        At higher developmental levels:
        - Lower dropout rates (more reliable processing)
        - Higher sensitivity to subtle patterns
        - More nuanced activation thresholds
        """
        # Scale dropout based on development (higher dropout at lower development)
        dropout_scale = max(0.1, 0.5 - (self.developmental_level * 0.4))
        
        # Update dropout layers
        for module in self.feature_extractor:
            if isinstance(module, nn.Dropout):
                module.p = dropout_scale
        
    def update_developmental_level(self, new_level: float):
        """
        Update the network's developmental level and adjust parameters
        
        Args:
            new_level: New developmental level (0.0 to 1.0)
        """
        if 0.0 <= new_level <= 1.0 and new_level != self.developmental_level:
            prev_level = self.developmental_level
            self.developmental_level = new_level
            
            # Adjust network parameters based on new level
            self._apply_developmental_scaling()
            
            # Log significant developmental changes
            if int(new_level * 10) > int(prev_level * 10):
                logger.info(f"Perception network advanced to developmental level {new_level:.2f}")
                
            return True
        return False
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the perception network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with:
                features: Extracted features
                patterns: Encoded patterns
                novelty: Novelty scores
                salience: Salience scores
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Encode patterns
        patterns = self.pattern_encoder(features)
        
        # Detect novelty
        novelty = self.novelty_detector(patterns)
        
        # Estimate salience
        salience = self.salience_estimator(patterns)
        
        # Apply developmental scaling to outputs
        if self.developmental_level < 0.3:
            # At early developmental stages, reduce sensitivity to subtle distinctions
            patterns = torch.tanh(patterns * (0.5 + self.developmental_level))
            
            # Make novelty detection more binary (less nuanced)
            novelty = torch.round(novelty * 2) / 2
        
        return {
            "features": features,
            "patterns": patterns,
            "novelty": novelty,
            "salience": salience
        }
    
    def detect_patterns(
        self, 
        input_vector: torch.Tensor, 
        known_patterns: Optional[Dict[str, torch.Tensor]] = None,
        activation_threshold: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Detect patterns in the input vector
        
        Args:
            input_vector: Input vector to detect patterns in
            known_patterns: Dictionary of known patterns
            activation_threshold: Threshold for pattern activation
            
        Returns:
            Tuple of (activated_patterns, updated_known_patterns)
        """
        # Process input through network
        with torch.no_grad():
            output = self.forward(input_vector)
            
        # Get encoded pattern
        pattern_vector = output["patterns"]
        
        # Create pattern memory if none provided
        if known_patterns is None:
            known_patterns = self.pattern_memory
            
        # List for storing activated patterns
        activated_patterns = []
        
        # Calculate similarities with known patterns
        max_similarity = 0.0
        most_similar_pattern = None
        
        for pattern_id, stored_pattern in known_patterns.items():
            # Calculate cosine similarity - fix the dimension issue
            # Ensure both tensors are properly shaped for cosine similarity
            pattern_vector_flat = pattern_vector.view(1, -1)  # Shape: [1, dim]
            stored_pattern_flat = stored_pattern.view(1, -1)  # Shape: [1, dim]
            
            # Calculate cosine similarity between the flattened vectors
            similarity = F.cosine_similarity(
                pattern_vector_flat, 
                stored_pattern_flat
            ).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pattern = pattern_id
                
            # If similarity exceeds threshold, pattern is activated
            if similarity > activation_threshold:
                activated_patterns.append({
                    "pattern_id": pattern_id,
                    "activation": similarity,
                    "novelty": 1.0 - similarity
                })
                
        # Adjust threshold based on developmental level
        creation_threshold = max(0.1, 0.6 - (self.developmental_level * 0.3))
        
        # If no patterns were significantly activated, create a new one
        if max_similarity < creation_threshold:
            new_pattern_id = str(uuid.uuid4())
            known_patterns[new_pattern_id] = pattern_vector.detach().clone()
            
            activated_patterns.append({
                "pattern_id": new_pattern_id,
                "activation": 1.0,  # New pattern is fully activated
                "novelty": 1.0,     # New pattern is maximally novel
                "is_new": True
            })
            
            # Add to recent patterns
            self.recent_patterns.append({
                "pattern_id": new_pattern_id,
                "timestamp": datetime.now(),
                "vector": pattern_vector.detach().clone()
            })
            
        # Update the pattern memory
        self.pattern_memory = known_patterns
            
        return activated_patterns, known_patterns
    
    def to_device(self, device: torch.device):
        """Move the model and all tensors to the specified device"""
        self.to(device)
        # Move all stored patterns to the device
        for pattern_id, pattern in self.pattern_memory.items():
            self.pattern_memory[pattern_id] = pattern.to(device)
            
    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the perception network"""
        return {
            "developmental_level": self.developmental_level,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "pattern_dim": self.pattern_dim,
            "pattern_count": len(self.pattern_memory),
            "recent_pattern_count": len(self.recent_patterns)
        }
            

class TemporalPatternNetwork(nn.Module):
    """
    Neural network for processing temporal sequences of patterns
    
    This network processes sequences of pattern vectors to detect
    temporal patterns and predict future patterns.
    """
    
    def __init__(
        self, 
        input_dim: int = 32, 
        hidden_dim: int = 64,
        sequence_length: int = 5,
        developmental_level: float = 0.0
    ):
        """
        Initialize the temporal pattern network
        
        Args:
            input_dim: Dimension of input pattern vectors
            hidden_dim: Dimension of hidden layer
            sequence_length: Maximum sequence length to process
            developmental_level: Initial developmental level
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.developmental_level = developmental_level
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Temporal pattern extraction
        self.pattern_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Next pattern prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Device for computation
        self.device = torch.device('cpu')
        
    def to_device(self, device: torch.device):
        """Move the network to the specified device"""
        self.device = device
        self.to(device)
        return self
        
    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of pattern vectors
        
        Args:
            sequence: Tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Dictionary with temporal pattern and next prediction
        """
        # Ensure sequence is on the correct device
        sequence = sequence.to(self.device)
        
        # Ensure LSTM and other components are on the same device
        self.lstm = self.lstm.to(self.device)
        self.pattern_extractor = self.pattern_extractor.to(self.device)
        self.predictor = self.predictor.to(self.device)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(sequence)
        
        # Extract temporal pattern from final hidden state
        temporal_pattern = self.pattern_extractor(hidden[-1])
        
        # Predict next pattern
        next_prediction = self.predictor(hidden[-1])
        
        return {
            "temporal_pattern": temporal_pattern,
            "next_prediction": next_prediction,
            "hidden_state": hidden[-1]
        }
        
    def update_developmental_level(self, level: float):
        """Update the developmental level of the network"""
        self.developmental_level = level
        
    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the temporal network"""
        return {
            "developmental_level": self.developmental_level,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "sequence_length": self.sequence_length
        }

class NeuralPatternDetector:
    """
    Neural network-based pattern detector
    
    Uses neural networks to detect patterns in input data and
    maintain a memory of known patterns.
    """
    
    def __init__(
        self, 
        vector_dim: int = 32,
        developmental_level: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the neural pattern detector
        
        Args:
            vector_dim: Dimension of pattern vectors
            developmental_level: Initial developmental level
            device: Device to use for computation (CPU or CUDA)
        """
        self.vector_dim = vector_dim
        self.developmental_level = developmental_level
        
        # Set device (CPU or CUDA if available)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.pattern_encoder = PatternEncoder(input_dim=vector_dim).to(self.device)
        self.temporal_network = TemporalPatternNetwork(
            input_dim=vector_dim,
            developmental_level=developmental_level
        ).to_device(self.device)
        
        # Storage for known patterns
        self.known_patterns = []
        self.pattern_vectors = []
        
        # Pattern activation threshold
        self.activation_threshold = 0.2
        
        # Initialize pattern sequence for temporal processing
        self.pattern_sequence = []
        self.max_sequence_length = 5
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def update_developmental_level(self, level: float):
        """Update the developmental level of all components"""
        if 0.0 <= level <= 1.0:
            self.developmental_level = level
            self.temporal_network.update_developmental_level(level)
            
            # Adjust activation threshold based on developmental level
            # Lower threshold as development increases (more sensitive)
            self.activation_threshold = max(0.1, 0.3 - (level * 0.2))
            
            return True
        return False
        
    def encode_pattern(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Encode an input vector into a pattern vector
        
        Args:
            input_vector: Input vector to encode
            
        Returns:
            Encoded pattern vector
        """
        # Ensure input is on the correct device
        input_vector = input_vector.to(self.device)
        
        # Encode the pattern
        with torch.no_grad():
            pattern_vector = self.pattern_encoder(input_vector)
            
        return pattern_vector
        
    def detect_patterns(self, input_vector: torch.Tensor, activation_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect patterns in the input vector
        
        Args:
            input_vector: Input vector to detect patterns in
            activation_threshold: Optional override for activation threshold
            
        Returns:
            List of detected patterns with confidence scores
        """
        if activation_threshold is None:
            activation_threshold = self.activation_threshold
            
        # Ensure input is on the correct device
        input_vector = input_vector.to(self.device)
        
        # Encode the pattern
        pattern_vector = self.encode_pattern(input_vector)
        
        detected_patterns = []
        
        # Compare with known patterns
        for i, stored_pattern in enumerate(self.pattern_vectors):
            # Ensure stored pattern is on the correct device
            stored_pattern = stored_pattern.to(self.device)
            
            # Reshape tensors for cosine similarity calculation
            pattern_vector_flat = pattern_vector.view(1, -1)
            stored_pattern_flat = stored_pattern.view(1, -1)
            
            # Calculate similarity
            similarity = F.cosine_similarity(pattern_vector_flat, stored_pattern_flat)
            confidence = similarity.item()
            
            if confidence >= activation_threshold:
                detected_patterns.append({
                    "pattern_id": i,
                    "pattern_name": self.known_patterns[i]["name"],
                    "confidence": confidence,
                    "pattern_type": "neural"
                })
                
        return detected_patterns
        
    def process_temporal_sequence(self, pattern_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Process a pattern vector through the temporal network
        
        Args:
            pattern_vector: Pattern vector to process
            
        Returns:
            Temporal processing results
        """
        # Ensure pattern vector is on the correct device
        pattern_vector = pattern_vector.to(self.device)
        
        # Add to sequence
        self.pattern_sequence.append(pattern_vector)
        
        # Keep sequence at max length
        if len(self.pattern_sequence) > self.max_sequence_length:
            self.pattern_sequence = self.pattern_sequence[-self.max_sequence_length:]
            
        # Process sequence if we have enough patterns
        if len(self.pattern_sequence) >= 2:
            # Stack sequence into a batch
            sequence_tensor = torch.stack(self.pattern_sequence).unsqueeze(0)
            
            # Process through temporal network
            with torch.no_grad():
                temporal_results = self.temporal_network(sequence_tensor)
                
            return temporal_results
            
        return {"temporal_pattern": None, "next_prediction": None}
        
    def create_pattern(self, pattern_vector: torch.Tensor, pattern_name: str) -> Dict[str, Any]:
        """
        Create a new pattern from the input vector
        
        Args:
            pattern_vector: Vector representation of the pattern
            pattern_name: Name/identifier for the pattern
            
        Returns:
            Created pattern information
        """
        # Ensure pattern vector is on the correct device
        pattern_vector = pattern_vector.to(self.device)
        
        # Create pattern entry
        pattern = {
            "name": pattern_name,
            "created_at": datetime.now().isoformat(),
            "developmental_level": self.developmental_level,
            "vector_dim": self.vector_dim
        }
        
        # Store pattern
        self.known_patterns.append(pattern)
        self.pattern_vectors.append(pattern_vector.detach().clone())
        
        return pattern
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the neural pattern detector
        
        Args:
            input_data: Input data dictionary with vector representation
            
        Returns:
            Processing results with detected patterns
        """
        # Extract input vector
        if "vector" not in input_data:
            self.logger.warning("No vector in input data for neural pattern detector")
            return {"error": "No vector in input data"}
            
        input_vector = input_data["vector"]
        
        # Ensure input vector is a tensor
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32)
            
        # Ensure input vector is on the correct device
        input_vector = input_vector.to(self.device)
        
        # Encode the pattern
        pattern_vector = self.encode_pattern(input_vector)
        
        # Detect patterns
        detected_patterns = self.detect_patterns(input_vector)
        
        # Process temporal sequence
        temporal_results = self.process_temporal_sequence(pattern_vector)
        
        # Create a new pattern if none detected and developmental level is sufficient
        if not detected_patterns and self.developmental_level >= 0.3:
            # Generate a unique pattern name
            pattern_name = f"neural_pattern_{len(self.known_patterns)}"
            
            # Create the pattern
            new_pattern = self.create_pattern(pattern_vector, pattern_name)
            
            # Add to detected patterns with high confidence
            detected_patterns.append({
                "pattern_id": len(self.known_patterns) - 1,
                "pattern_name": pattern_name,
                "confidence": 1.0,  # New pattern has perfect match
                "pattern_type": "neural",
                "is_new": True
            })
            
        return {
            "detected_patterns": detected_patterns,
            "temporal_results": temporal_results,
            "developmental_level": self.developmental_level
        }
