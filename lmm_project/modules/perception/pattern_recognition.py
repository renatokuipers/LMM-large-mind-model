"""
Pattern Recognition Module

This module detects patterns in perceptual input, which is a fundamental
capability required for higher cognitive functions. It operates on the
pre-processed sensory inputs to identify recurring structures and relationships.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
import logging
from datetime import datetime
from collections import deque
import random

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.perception.models import (
    Pattern, PerceptionResult, SensoryInput, PerceptionParameters
)
from lmm_project.modules.perception.neural_net import PerceptionNetwork, TemporalPatternNetwork

logger = logging.getLogger(__name__)

class PatternRecognizer(BaseModule):
    """
    Detects and learns patterns from sensory input
    
    The pattern recognizer identifies recurring structures in perceptual input,
    from simple token-level patterns to complex semantic and temporal patterns.
    Its capabilities evolve with developmental level.
    """
    # Pattern storage
    known_patterns: Dict[str, Pattern] = {}
    
    # Sequence tracking for temporal patterns
    input_sequence: deque = deque(maxlen=10)
    temporal_patterns: Dict[str, Any] = {}
    
    # Configuration parameters
    parameters: PerceptionParameters = PerceptionParameters()
    
    # Neural networks
    perception_network: Optional[PerceptionNetwork] = None
    temporal_network: Optional[TemporalPatternNetwork] = None
    
    # Persistent feature vectors for known patterns
    pattern_vectors: Dict[str, torch.Tensor] = {}
    
    # Module state
    last_novel_pattern_time: Optional[datetime] = None
    pattern_activation_history: List[Dict[str, Any]] = []
    
    def __init__(
        self, 
        module_id: str,
        event_bus: Optional[EventBus] = None,
        developmental_level: float = 0.0,
        **kwargs
    ):
        """Initialize the pattern recognizer"""
        super().__init__(
            module_id=module_id,
            module_type="pattern_recognizer",
            event_bus=event_bus,
            development_level=developmental_level,
            **kwargs
        )
        
        # Initialize neural networks
        self._initialize_networks()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("sensory_input_processed", self._handle_sensory_input)
            
        # Initialize parameters based on developmental level
        self._adjust_parameters_for_development()
            
    def _initialize_networks(self):
        """Initialize neural networks for pattern recognition"""
        # Initialize perception network
        self.perception_network = PerceptionNetwork(
            input_dim=384,  # Dimension of text embeddings
            hidden_dim=256,
            pattern_dim=128,
            developmental_level=self.development_level
        )
        
        # Initialize temporal pattern network
        self.temporal_network = TemporalPatternNetwork(
            input_dim=128,
            hidden_dim=256,
            sequence_length=5
        )
        
        # Move to appropriate device if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perception_network.to_device(device)
        self.temporal_network.to(device)
        
    def _adjust_parameters_for_development(self):
        """Adjust recognition parameters based on developmental level"""
        # Early development - higher sensitivity to detect more patterns
        if self.development_level < 0.2:
            self.parameters.pattern_activation_threshold = 0.2
            self.parameters.token_sensitivity = 0.7
            self.parameters.ngram_sensitivity = 0.3
            self.parameters.semantic_sensitivity = 0.1
            self.parameters.novelty_threshold = 0.4
        
        # Mid development - more balanced parameters
        elif self.development_level < 0.6:
            self.parameters.pattern_activation_threshold = 0.3
            self.parameters.token_sensitivity = 0.5
            self.parameters.ngram_sensitivity = 0.5
            self.parameters.semantic_sensitivity = 0.3
            self.parameters.novelty_threshold = 0.5
            
        # Advanced development - more selective pattern recognition
        else:
            self.parameters.pattern_activation_threshold = 0.4
            self.parameters.token_sensitivity = 0.3
            self.parameters.ngram_sensitivity = 0.5
            self.parameters.semantic_sensitivity = 0.6
            self.parameters.novelty_threshold = 0.6
            
    def update_development(self, amount: float) -> float:
        """Update developmental level and adjust parameters"""
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # If significant change, update networks and parameters
        if abs(self.development_level - prev_level) > 0.05:
            self._adjust_parameters_for_development()
            
            # Update neural networks
            if self.perception_network:
                self.perception_network.update_developmental_level(self.development_level)
                
        return self.development_level
        
    def _detect_token_patterns(self, tokens: List[str]) -> List[Pattern]:
        """
        Detect patterns at the token level
        
        For early development, this focuses on detecting:
        - Repeated tokens
        - Common tokens
        - Token sequences (n-grams)
        """
        detected_patterns = []
        
        if not tokens:
            return detected_patterns
            
        # Track token frequencies in this input
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
            
        # Detect repeated tokens (frequency > 1)
        for token, freq in token_freq.items():
            if freq > 1:
                pattern_id = f"token_repeat_{token}_{uuid.uuid4().hex[:6]}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type="token",
                    content=token,
                    confidence=min(0.5 + 0.1 * freq, 0.9),
                    activation=min(0.3 + 0.1 * freq, 0.8),
                    frequency=1
                )
                detected_patterns.append(pattern)
                
        # Detect n-grams (adjacent tokens)
        if len(tokens) > 1 and self.parameters.ngram_sensitivity > 0.2:
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                pattern_id = f"ngram_{uuid.uuid4().hex[:6]}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type="n_gram",
                    content=bigram,
                    confidence=0.5 * self.parameters.ngram_sensitivity,
                    activation=0.4 * self.parameters.ngram_sensitivity,
                    frequency=1
                )
                detected_patterns.append(pattern)
                
        return detected_patterns
    
    def _detect_semantic_patterns(self, embedding_tensor: torch.Tensor) -> List[Pattern]:
        """
        Detect semantic patterns using neural network
        
        Uses the perception network to encode the input and detect patterns
        by comparing with known pattern vectors.
        """
        detected_patterns = []
        
        # Skip if network not initialized
        if not self.perception_network:
            return detected_patterns
            
        try:
            # Process through perception network
            with torch.no_grad():
                network_output = self.perception_network.forward(embedding_tensor)
                
            # Get detected patterns from network
            activated_patterns, updated_patterns = self.perception_network.detect_patterns(
                embedding_tensor,
                self.pattern_vectors,
                activation_threshold=self.parameters.pattern_activation_threshold
            )
            
            # Update pattern vectors with any new additions
            self.pattern_vectors = updated_patterns
            
            # Convert to Pattern models
            for pattern_info in activated_patterns:
                pattern_id = pattern_info["pattern_id"]
                is_new = pattern_info["is_new"]
                activation = pattern_info["activation"]
                
                # For new patterns, create a full Pattern object
                if is_new:
                    pattern = Pattern(
                        pattern_id=pattern_id,
                        pattern_type="semantic",
                        content=f"semantic_pattern_{pattern_id[-6:]}",
                        confidence=0.5,  # Initial confidence for new patterns
                        activation=activation,
                        frequency=1
                    )
                    self.known_patterns[pattern_id] = pattern
                    self.last_novel_pattern_time = datetime.now()
                    
                    # Log new pattern discovery
                    logger.debug(f"New semantic pattern discovered: {pattern_id}")
                
                # For existing patterns, retrieve and update activation
                elif pattern_id in self.known_patterns:
                    pattern = self.known_patterns[pattern_id]
                    pattern.activation = activation
                    pattern.frequency += 1
                    pattern.last_seen = datetime.now()
                    
                    # Update confidence with experience
                    pattern.confidence = min(
                        pattern.confidence + 0.02, 
                        0.9
                    )
                
                # Something went wrong, create a new pattern as fallback
                else:
                    pattern = Pattern(
                        pattern_id=pattern_id,
                        pattern_type="semantic",
                        content=f"semantic_pattern_{pattern_id[-6:]}",
                        confidence=0.3,
                        activation=activation,
                        frequency=1
                    )
                    self.known_patterns[pattern_id] = pattern
                
                detected_patterns.append(pattern)
                
            # Calculate novelty score from the network output
            novelty_score = float(network_output["novelty"].mean().item())
            
            # Record pattern activation for history
            self.pattern_activation_history.append({
                "timestamp": datetime.now(),
                "activated_count": len(activated_patterns),
                "novelty_score": novelty_score
            })
            
            # Limit history size
            if len(self.pattern_activation_history) > 100:
                self.pattern_activation_history = self.pattern_activation_history[-100:]
                
            return detected_patterns
                
        except Exception as e:
            logger.error(f"Error in semantic pattern detection: {e}")
            return []
    
    def _update_temporal_sequence(self, patterns: List[Pattern], embedding_tensor: torch.Tensor):
        """
        Update temporal pattern sequence and detect temporal patterns
        
        Records the current set of patterns in the sequence history and
        looks for recurring temporal sequences.
        """
        if not patterns or not self.temporal_network:
            return
            
        try:
            # Extract pattern IDs as a fingerprint for this input
            pattern_fingerprint = {p.pattern_id: p.activation for p in patterns}
            
            # Add to sequence
            self.input_sequence.append({
                "timestamp": datetime.now(),
                "patterns": pattern_fingerprint,
                "embedding": embedding_tensor.detach().clone()
            })
            
            # Need at least 3 items for temporal pattern detection
            if len(self.input_sequence) < 3:
                return
                
            # Check if we have enough development for temporal patterns
            if self.development_level < 0.3:
                return
                
            # Extract embeddings from sequence
            sequence_tensors = []
            for item in list(self.input_sequence)[-5:]:  # Use up to 5 recent items
                sequence_tensors.append(item["embedding"])
                
            # Stack tensors for batch processing
            if len(sequence_tensors) < 2:
                return
                
            # Create padded sequence tensor
            sequence_tensor = torch.cat(sequence_tensors, dim=0).unsqueeze(0)
            
            # Process through temporal network
            with torch.no_grad():
                temporal_output = self.temporal_network.forward(sequence_tensor)
                
            # Store temporal pattern
            temporal_pattern_id = f"temporal_{uuid.uuid4().hex[:8]}"
            self.temporal_patterns[temporal_pattern_id] = {
                "created_at": datetime.now(),
                "pattern_vector": temporal_output["temporal_pattern"].detach().clone(),
                "sequence_length": len(sequence_tensors)
            }
            
            # TODO: In more advanced implementations, compare with existing 
            # temporal patterns to find recurring sequences
            
        except Exception as e:
            logger.error(f"Error in temporal pattern update: {e}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to detect patterns
        
        Args:
            input_data: Dictionary containing processed sensory input
                Must include processed sensory features
                
        Returns:
            Dictionary with detected patterns and perceptual analysis
        """
        # Check for required fields
        if "features" not in input_data:
            return {"error": "Missing features in input data"}
            
        try:
            # Extract information from input
            features = input_data.get("features", {})
            tokens = features.get("tokens", [])
            embedding = features.get("embedding", [])
            input_id = input_data.get("input_id", f"unknown_{uuid.uuid4().hex[:8]}")
            
            # Get or create embedding tensor
            if "embedding_tensor" in input_data:
                embedding_tensor = input_data["embedding_tensor"]
            elif embedding is not None:
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            else:
                return {"error": "No valid embedding data found"}
                
            # Detect token-level patterns
            token_patterns = self._detect_token_patterns(tokens)
            
            # Detect semantic patterns
            semantic_patterns = self._detect_semantic_patterns(embedding_tensor)
            
            # Combine detected patterns
            all_patterns = token_patterns + semantic_patterns
            
            # Update temporal sequence
            self._update_temporal_sequence(all_patterns, embedding_tensor)
            
            # Get network output for additional metrics
            network_output = {}
            if self.perception_network:
                with torch.no_grad():
                    network_output = self.perception_network.forward(embedding_tensor)
            
            # Calculate novelty score
            if "novelty" in network_output:
                novelty_score = float(network_output["novelty"].mean().item())
            else:
                # Fallback calculation if network output unavailable
                novelty_score = 0.5
                if self.known_patterns:
                    # More patterns found = less novel
                    novelty_score = max(0.1, 1.0 - min(1.0, len(all_patterns) / 5))
            
            # Calculate intensity score
            if "intensity" in network_output:
                intensity_score = float(network_output["intensity"].mean().item())
            else:
                # Fallback intensity calculation
                avg_activation = 0.5
                if all_patterns:
                    avg_activation = sum(p.activation for p in all_patterns) / len(all_patterns)
                intensity_score = avg_activation
            
            # Create perception result
            perception_result = PerceptionResult(
                input_id=input_id,
                detected_patterns=all_patterns,
                novelty_score=novelty_score,
                intensity_score=intensity_score,
                feature_vector=embedding if isinstance(embedding, list) else embedding.tolist(),
                developmental_level=self.development_level
            )
            
            # Create semantic content summary for easier debugging/introspection
            semantic_content = {
                "token_pattern_count": len(token_patterns),
                "semantic_pattern_count": len(semantic_patterns),
                "novelty_level": "high" if novelty_score > 0.7 else "medium" if novelty_score > 0.4 else "low",
                "intensity_level": "high" if intensity_score > 0.7 else "medium" if intensity_score > 0.4 else "low"
            }
            perception_result.semantic_content = semantic_content
            
            # Create result object
            result = {
                "status": "success",
                "perception_result": perception_result.model_dump(),
                "input_id": input_id,
                "developmental_level": self.development_level
            }
            
            # Publish perception result
            if self.event_bus:
                self.publish_message(
                    message_type="perception_result",
                    content=result
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing input in pattern recognition: {e}")
            return {"error": f"Pattern recognition failed: {str(e)}"}
    
    def _handle_sensory_input(self, message: Message):
        """Handle processed sensory input"""
        if not message.content:
            return
            
        # Process the input
        self.process_input(message.content)
        
    def get_state(self) -> Dict[str, Any]:
        """Get the module's current state"""
        base_state = super().get_state()
        
        # Add pattern recognizer specific state information
        extended_state = {
            "known_pattern_count": len(self.known_patterns),
            "temporal_pattern_count": len(self.temporal_patterns),
            "pattern_activation_threshold": self.parameters.pattern_activation_threshold,
            "last_novelty_score": self.pattern_activation_history[-1]["novelty_score"] if self.pattern_activation_history else None,
            "has_temporal_sequence": len(self.input_sequence) > 0
        }
        
        base_state.update(extended_state)
        return base_state
        
    def get_top_patterns(self, count: int = 5) -> List[Pattern]:
        """Get the most frequent patterns"""
        patterns = list(self.known_patterns.values())
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns[:count] 
