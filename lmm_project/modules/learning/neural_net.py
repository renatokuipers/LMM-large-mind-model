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
    
    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "developmental_level": self.developmental_level
            },
            "learning_history": self.learning_history
        }, path)
        logger.info(f"Learning network saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path)
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
                "output_dim": self.output_dim
            }
        }
