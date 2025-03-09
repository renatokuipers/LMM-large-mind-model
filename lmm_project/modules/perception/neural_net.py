import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)

class PerceptionNetwork(nn.Module): 
    """
    Neural network for processing textual perceptual inputs.
    
    This network processes text-based perceptual input through multiple stages:
    1. Feature extraction - Converts text tokens to feature vectors
    2. Pattern encoding - Encodes features into pattern representations
    3. Novelty detection - Identifies novel or unexpected inputs
    
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
            nn.Linear(pattern_dim, pattern_dim // 2),
            nn.ReLU(),
            nn.Linear(pattern_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Intensity estimation network
        self.intensity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize developmental weights
        self._apply_developmental_scaling()
        
    def _apply_developmental_scaling(self):
        """Apply developmental scaling to network parameters"""
        # Simplify processing for early development stages
        if self.developmental_level < 0.2:
            # Add noise to make early pattern recognition less precise
            with torch.no_grad():
                noise_scale = max(0.5 - self.developmental_level * 2, 0.1)
                for param in self.pattern_encoder.parameters():
                    param.add_(torch.randn_like(param) * noise_scale)
    
    def update_developmental_level(self, new_level: float):
        """
        Update the developmental level and adjust network accordingly
        
        As development progresses, the network becomes more sophisticated:
        - Reduced noise in processing
        - More precise pattern encoding
        - Better novelty detection
        """
        prev_level = self.developmental_level
        self.developmental_level = min(max(new_level, 0.0), 1.0)
        
        # Only update if significant change
        if abs(self.developmental_level - prev_level) > 0.05:
            self._apply_developmental_scaling()
            
            # Log development progress
            logger.info(f"Perception network updated to developmental level: {self.developmental_level:.2f}")
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through the perception network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
            - features: Extracted features
            - patterns: Encoded patterns
            - novelty: Novelty scores
            - intensity: Intensity scores
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Pattern encoding
        patterns = self.pattern_encoder(features)
        
        # Novelty detection
        novelty = self.novelty_detector(patterns)
        
        # Intensity estimation
        intensity = self.intensity_estimator(features)
        
        return {
            "features": features,
            "patterns": patterns,
            "novelty": novelty,
            "intensity": intensity
        }
    
    def detect_patterns(
        self, 
        input_vector: torch.Tensor,
        known_patterns: Optional[Dict[str, torch.Tensor]] = None,
        activation_threshold: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Detect patterns in the input vector and compare with known patterns
        
        Args:
            input_vector: Input feature vector
            known_patterns: Dictionary of known pattern vectors
            activation_threshold: Threshold for pattern activation
            
        Returns:
            Tuple of (activated_patterns, updated_known_patterns)
        """
        if known_patterns is None:
            known_patterns = {}
            
        # Process input
        with torch.no_grad():
            output = self.forward(input_vector)
            current_pattern = output["patterns"]
            
        # Find matching patterns
        activated_patterns = []
        best_match_score = 0.0
        best_match_id = None
        
        # Compare with known patterns
        for pattern_id, pattern_vector in known_patterns.items():
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                current_pattern, pattern_vector.unsqueeze(0), dim=1
            ).item()
            
            # If above threshold, consider it activated
            if similarity > activation_threshold:
                activated_patterns.append({
                    "pattern_id": pattern_id,
                    "activation": float(similarity),
                    "is_new": False
                })
                
                # Track best match
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_id = pattern_id
        
        # Create new pattern if no good matches found
        if not activated_patterns or best_match_score < activation_threshold + 0.1:
            # Developmental scaling - more new patterns in early stages, fewer later
            novelty_bonus = max(0.5 - self.developmental_level * 0.5, 0.0)
            creation_threshold = activation_threshold - novelty_bonus
            
            if best_match_score < creation_threshold:
                # Create new pattern
                new_pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                known_patterns[new_pattern_id] = current_pattern.squeeze(0).detach().clone()
                
                # Add to activated patterns
                activated_patterns.append({
                    "pattern_id": new_pattern_id,
                    "activation": 1.0,  # New pattern fully activated
                    "is_new": True
                })
        
        return activated_patterns, known_patterns
    
    def to_device(self, device: torch.device):
        """Move the network to the specified device"""
        self.to(device)
        return self


class TemporalPatternNetwork(nn.Module):
    """
    Neural network for detecting temporal patterns across sequential inputs
    """
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        sequence_length: int = 5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Pattern detection from sequence
        self.temporal_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
    
    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of inputs
        
        Args:
            sequence: Tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Dictionary with temporal pattern information
        """
        # Process sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(sequence)
        
        # Use the final hidden state for temporal pattern detection
        temporal_pattern = self.temporal_detector(hidden.squeeze(0))
        
        # Predict next element in sequence
        next_prediction = self.temporal_detector(lstm_out[:, -1, :])
        
        return {
            "temporal_pattern": temporal_pattern,
            "next_prediction": next_prediction,
            "sequence_encoding": hidden.squeeze(0)
        }
