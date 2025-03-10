import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class LearningNetwork(nn.Module):
    """
    Neural network for the learning module that processes learning experiences
    and adapts based on developmental level.
    
    This network handles:
    1. Experience encoding - Converting learning experiences to vector representations
    2. Pattern extraction - Identifying patterns in learning experiences
    3. Reinforcement - Strengthening relevant connections
    4. Strategy learning - Developing higher-level learning strategies
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        developmental_level: float = 0.0
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.developmental_level = developmental_level
        
        # Get device (CUDA if available)
        self.device = get_device()
        
        # Experience encoding network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Pattern extraction network
        self.pattern_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Reinforcement prediction network
        self.reinforcement_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1 for reward prediction
        )
        
        # Strategy selection network (develops with higher developmental levels)
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 4)  # 4 learning strategies
        )
        
        # Apply developmental scaling
        self._apply_developmental_scaling()
        
        # Initialize learning history
        self.learning_history = []
        
        # Move model to appropriate device
        self.to(self.device)
        logger.info(f"Learning network initialized on device: {self.device}")
        
        # Enable cuDNN benchmark for performance optimization if available
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
    def _apply_developmental_scaling(self):
        """
        Adjust network parameters based on developmental level
        
        At lower levels:
        - Higher dropout (less reliable learning)
        - Simpler pattern recognition
        - Basic reinforcement learning only
        
        At higher levels:
        - More reliable processing
        - Complex pattern recognition
        - Strategy-based learning
        """
        # Scale dropout based on development
        dropout_rate = max(0.1, 0.5 - (self.developmental_level * 0.4))
        
        # Update dropout layers
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
                
        # Scaling factor for weights (lower dev level = simplified processing)
        scaling = 0.5 + (self.developmental_level * 0.5)
        
        # Scale the strategy network based on development level
        # (less influence at lower developmental levels)
        for param in self.strategy_selector.parameters():
            param.data *= self.developmental_level
    
    def update_developmental_level(self, new_level: float):
        """
        Update the developmental level and adjust network accordingly
        
        Args:
            new_level: New developmental level (0.0 to 1.0)
        """
        self.developmental_level = max(0.0, min(1.0, new_level))
        self._apply_developmental_scaling()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the learning network
        
        Args:
            x: Input tensor representing a learning experience
            
        Returns:
            Dictionary with encoded experience, extracted patterns,
            reinforcement prediction, and strategy selection
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Encode the experience
        encoded = self.encoder(x)
        
        # Extract patterns
        patterns = self.pattern_extractor(encoded)
        
        # Predict reinforcement value
        reinforcement = self.reinforcement_predictor(encoded)
        
        # Select learning strategy (influenced by developmental level)
        strategy_logits = self.strategy_selector(encoded)
        strategy_probs = F.softmax(strategy_logits * self.developmental_level, dim=-1)
        
        return {
            "encoded_experience": encoded,
            "extracted_patterns": patterns,
            "reinforcement_prediction": reinforcement,
            "strategy_selection": strategy_probs
        }
    
    def adapt_to_reinforcement(self, 
                              experience: torch.Tensor, 
                              reward: float, 
                              learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Adapt network based on reinforcement signal
        
        Args:
            experience: Input tensor representing the experience
            reward: Actual reward/reinforcement value (-1.0 to 1.0)
            learning_rate: How quickly to adapt to the reinforcement
            
        Returns:
            Dictionary with prediction error and updated prediction
        """
        # Ensure experience is on the correct device
        experience = experience.to(self.device)
        
        # Get current prediction
        with torch.no_grad():
            output = self.forward(experience)
            current_prediction = output["reinforcement_prediction"].item()
        
        # Calculate prediction error
        prediction_error = reward - current_prediction
        
        # Record learning event
        self.learning_history.append({
            "timestamp": datetime.now(),
            "prediction": current_prediction,
            "actual": reward,
            "error": prediction_error,
            "developmental_level": self.developmental_level
        })
        
        # Return results
        return {
            "prediction_error": prediction_error,
            "updated_prediction": current_prediction + (prediction_error * learning_rate)
        }
    
    def batch_process_experiences(self, experiences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a batch of experiences efficiently
        
        Args:
            experiences: Batch of experience tensors [batch_size, input_dim]
            
        Returns:
            Dictionary with batch results
        """
        # Ensure experiences are on the correct device
        experiences = experiences.to(self.device)
        
        # Forward pass
        results = self.forward(experiences)
        
        return results
        
    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Move model to CPU for saving to avoid GPU memory issues
        device_backup = next(self.parameters()).device
        cpu_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        
        torch.save({
            "model_state": cpu_state_dict,
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "developmental_level": self.developmental_level
            },
            "learning_history": self.learning_history
        }, path)
        
        # Move model back to original device
        self.to(device_backup)
        logger.info(f"Learning network saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state"])
            self.developmental_level = checkpoint["config"]["developmental_level"]
            self.learning_history = checkpoint.get("learning_history", [])
            self._apply_developmental_scaling()
            logger.info(f"Learning network loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the network"""
        return {
            "developmental_level": self.developmental_level,
            "learning_history_length": len(self.learning_history),
            "recent_errors": [item["error"] for item in self.learning_history[-10:]] if self.learning_history else [],
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "device": str(self.device)
            }
        }
        
    def free_memory(self):
        """Free GPU memory if using CUDA"""
        if self.device.type == 'cuda':
            # Move model to CPU
            self.to(torch.device('cpu'))
            # Clear CUDA cache
            torch.cuda.empty_cache()
            logger.info("Freed GPU memory")
            
    def to_gpu(self):
        """Move model to GPU if available"""
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))
            self.device = torch.device('cuda')
            logger.info("Moved model to GPU")
        else:
            logger.warning("GPU not available, model remains on CPU")
            
    def to_cpu(self):
        """Move model to CPU"""
        self.to(torch.device('cpu'))
        self.device = torch.device('cpu')
        logger.info("Moved model to CPU")


class AssociativeLearningNetwork(LearningNetwork):
    """Specialized network for associative learning with pattern detection focus"""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        developmental_level: float = 0.0
    ):
        super().__init__(input_dim, hidden_dim, output_dim, developmental_level)
        
        # Add pattern similarity network
        self.similarity_network = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def compute_similarity(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """Compute similarity between two patterns"""
        # Ensure patterns are on the correct device
        pattern1 = pattern1.to(self.device)
        pattern2 = pattern2.to(self.device)
        
        # Concatenate patterns
        combined = torch.cat((pattern1, pattern2), dim=-1)
        
        # Compute similarity
        with torch.no_grad():
            similarity = self.similarity_network(combined)
            
        return similarity.item()


class ReinforcementLearningNetwork(LearningNetwork):
    """Specialized network for reinforcement learning with policy focus"""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        action_dim: int = 10,
        developmental_level: float = 0.0
    ):
        super().__init__(input_dim, hidden_dim, output_dim, developmental_level)
        
        # Add Q-value network for reinforcement learning
        self.action_dim = action_dim
        self.q_network = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)
    
    def compute_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a state"""
        # Ensure state is on the correct device
        state = state.to(self.device)
        
        # Extract patterns from state
        with torch.no_grad():
            output = self.forward(state)
            patterns = output["extracted_patterns"]
            
            # Compute Q-values
            q_values = self.q_network(patterns)
            
        return q_values


class ProceduralLearningNetwork(LearningNetwork):
    """Specialized network for procedural learning with sequence focus"""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        seq_length: int = 5,
        developmental_level: float = 0.0
    ):
        super().__init__(input_dim, hidden_dim, output_dim, developmental_level)
        
        # Add sequence prediction network
        self.seq_length = seq_length
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        ).to(self.device)
        
        self.sequence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
    
    def predict_next_in_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Predict the next item in a sequence"""
        # Ensure sequence is on the correct device
        sequence = sequence.to(self.device)
        
        # Process each item in sequence
        seq_features = []
        for i in range(sequence.shape[0]):
            with torch.no_grad():
                item = sequence[i:i+1]
                output = self.forward(item)
                seq_features.append(output["extracted_patterns"])
        
        # Stack features
        seq_features = torch.cat(seq_features, dim=0).unsqueeze(0)  # Add batch dimension
        
        # Process with LSTM
        with torch.no_grad():
            lstm_out, _ = self.lstm(seq_features)
            prediction = self.sequence_predictor(lstm_out[:, -1])
            
        return prediction


class MetaLearningNetwork(LearningNetwork):
    """Specialized network for meta-learning with strategy focus"""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        strategy_dim: int = 10,
        developmental_level: float = 0.0
    ):
        super().__init__(input_dim, hidden_dim, output_dim, developmental_level)
        
        # Add strategy optimization network
        self.strategy_dim = strategy_dim
        self.strategy_optimizer = nn.Sequential(
            nn.Linear(output_dim + strategy_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, strategy_dim)
        ).to(self.device)
    
    def optimize_strategy(self, state: torch.Tensor, current_strategy: torch.Tensor) -> torch.Tensor:
        """Optimize a learning strategy for the current state"""
        # Ensure tensors are on the correct device
        state = state.to(self.device)
        current_strategy = current_strategy.to(self.device)
        
        # Extract features
        with torch.no_grad():
            output = self.forward(state)
            features = output["extracted_patterns"]
            
            # Combine features with current strategy
            combined = torch.cat((features, current_strategy), dim=-1)
            
            # Optimize strategy
            optimized_strategy = self.strategy_optimizer(combined)
            
        return optimized_strategy
