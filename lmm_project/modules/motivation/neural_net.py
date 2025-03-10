import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class MotivationNeuralNetwork:
    """
    Neural network for the motivation system
    
    This network handles various motivation-related functions including
    drive processing, need evaluation, goal selection, and reward learning.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        network_type: str = "drive",
        learning_rate: float = 0.01,
        development_level: float = 0.0,
        device: str = "auto"  # "auto", "cpu", or "cuda"
    ):
        """
        Initialize the motivation neural network
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output vectors
            network_type: Type of motivation network ("drive", "need", "goal", "reward")
            learning_rate: Learning rate for training
            development_level: Current developmental level
            device: Device to use for tensor operations
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.learning_rate = learning_rate
        self.development_level = development_level
        
        # Set device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device} for {network_type} network")
        
        # Create the appropriate network based on type
        if network_type == "drive":
            self.network = self._create_drive_network().to(self.device)
        elif network_type == "need":
            self.network = self._create_need_network().to(self.device)
        elif network_type == "goal":
            self.network = self._create_goal_network().to(self.device)
        elif network_type == "reward":
            self.network = self._create_reward_network().to(self.device)
        else:
            self.network = self._create_generic_network().to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.learning_rate
        )
        
        # Hebbian learning matrix for associative learning
        self.hebbian_matrix = torch.zeros(
            (output_dim, input_dim), 
            device=self.device
        )
        
        # Hebbian learning rate
        self.hebbian_rate = 0.001
        
        # Perform initialization based on development level
        self._adjust_for_development()
    
    def _create_drive_network(self) -> nn.Module:
        """
        Create a neural network for drive processing
        
        This network maps internal states to drive intensities
        """
        class DriveNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
                self.output_layer = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                # Process internal state to produce drive intensities
                x = F.relu(self.input_layer(x))
                x = self.dropout(x)
                x = F.relu(self.hidden_layer(x))
                x = torch.sigmoid(self.output_layer(x))  # Drive intensities between 0-1
                return x
                
        return DriveNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_need_network(self) -> nn.Module:
        """
        Create a neural network for need evaluation
        
        This network evaluates need satisfaction based on current state
        """
        class NeedNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
                self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
                self.dropout = nn.Dropout(0.2)
                
                # Hierarchical weights for different need levels
                self.hierarchy_weights = nn.Parameter(
                    torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
                )
                
            def forward(self, x, need_levels=None):
                # Process state to evaluate need satisfaction
                x = F.relu(self.input_layer(x))
                x = self.dropout(x)
                x = F.relu(self.hidden_layer1(x))
                x = F.relu(self.hidden_layer2(x))
                satisfaction = torch.sigmoid(self.output_layer(x))
                
                # Apply hierarchical weighting if need levels provided
                if need_levels is not None:
                    weights = torch.zeros_like(satisfaction)
                    for i, level in enumerate(need_levels):
                        if level < 5:  # 1-indexed levels, 0-4 after subtraction
                            weights[i] = self.hierarchy_weights[level-1]
                    satisfaction = satisfaction * weights
                
                return satisfaction
                
        return NeedNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_goal_network(self) -> nn.Module:
        """
        Create a neural network for goal selection and prioritization
        
        This network evaluates and prioritizes goals based on current drives and needs
        """
        class GoalNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                # Need and drive inputs
                self.need_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
                self.drive_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
                
                # Goal evaluation
                self.goal_hidden = nn.Linear(hidden_dim, hidden_dim)
                self.goal_output = nn.Linear(hidden_dim, output_dim)
                
                # Urgency evaluation
                self.urgency_output = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                # Split input into need and drive components
                batch_size = x.shape[0]
                mid_point = self.input_dim // 2
                
                need_input = x[:, :mid_point]
                drive_input = x[:, mid_point:]
                
                # Encode need and drive states
                need_encoding = F.relu(self.need_encoder(need_input))
                drive_encoding = F.relu(self.drive_encoder(drive_input))
                
                # Combine encodings
                combined = torch.cat([need_encoding, drive_encoding], dim=1)
                
                # Generate goal relevance scores
                hidden = F.relu(self.goal_hidden(combined))
                relevance = torch.sigmoid(self.goal_output(hidden))
                
                # Generate urgency scores
                urgency = torch.sigmoid(self.urgency_output(hidden))
                
                return relevance, urgency
                
        return GoalNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_reward_network(self) -> nn.Module:
        """
        Create a neural network for reward processing and learning
        
        This network processes reward signals and updates learned values
        """
        class RewardNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                # State encoder
                self.state_encoder = nn.Linear(input_dim, hidden_dim)
                
                # Action encoder
                self.action_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
                
                # Value prediction
                self.value_hidden = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
                self.value_output = nn.Linear(hidden_dim, 1)
                
                # Action preference
                self.action_hidden = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
                self.action_output = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, state, action=None):
                # Encode state
                state_encoding = F.relu(self.state_encoder(state))
                
                # If action provided, predict value
                if action is not None:
                    action_encoding = F.relu(self.action_encoder(action))
                    combined = torch.cat([state_encoding, action_encoding], dim=1)
                    
                    # Predict value
                    value_hidden = F.relu(self.value_hidden(combined))
                    value = self.value_output(value_hidden)
                    
                    # Predict action preferences
                    action_hidden = F.relu(self.action_hidden(combined))
                    action_prefs = F.softmax(self.action_output(action_hidden), dim=1)
                    
                    return value, action_prefs
                else:
                    # Just return state encoding for further processing
                    return state_encoding
                
        return RewardNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_generic_network(self) -> nn.Module:
        """Create a generic fallback neural network"""
        class GenericNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
                self.output_layer = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = F.relu(self.input_layer(x))
                x = F.relu(self.hidden_layer(x))
                x = self.output_layer(x)
                return x
                
        return GenericNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _adjust_for_development(self):
        """Adjust network parameters based on developmental level"""
        # Higher development = lower learning rate (more stable)
        self.learning_rate = max(0.001, 0.01 - self.development_level * 0.009)
        
        # Update optimizer with new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
        # Higher development = higher hebbian learning rate
        self.hebbian_rate = min(0.01, 0.001 + self.development_level * 0.009)
        
        logger.debug(f"Adjusted {self.network_type} network for development level {self.development_level}")
    
    def forward(self, x: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor or array
            **kwargs: Additional arguments for specific network types
            
        Returns:
            Network output and any additional information
        """
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Ensure tensor is on the correct device
        x = x.to(self.device)
        
        # Forward pass depends on network type
        if self.network_type == "drive":
            output = self.network(x)
            return output, None
            
        elif self.network_type == "need":
            need_levels = kwargs.get("need_levels", None)
            output = self.network(x, need_levels)
            return output, None
            
        elif self.network_type == "goal":
            relevance, urgency = self.network(x)
            return relevance, urgency
            
        elif self.network_type == "reward":
            action = kwargs.get("action", None)
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                action = action.to(self.device)
                
            output = self.network(x, action)
            return output, None
            
        else:
            output = self.network(x)
            
            # Apply hebbian learning
            if self.development_level >= 0.4 and x.shape[0] == 1:  # Only apply to single inputs
                self._update_hebbian_matrix(x.detach(), output.detach())
                
            return output, None
    
    def train(
        self, 
        inputs: Union[np.ndarray, torch.Tensor], 
        targets: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the network
        
        Args:
            inputs: Input data
            targets: Target outputs
            **kwargs: Additional arguments for specific network types
            
        Returns:
            Training metrics
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Ensure tensors are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass depends on network type
        if self.network_type == "drive":
            outputs = self.network(inputs)
            loss = F.mse_loss(outputs, targets)
            
        elif self.network_type == "need":
            need_levels = kwargs.get("need_levels", None)
            outputs = self.network(inputs, need_levels)
            loss = F.mse_loss(outputs, targets)
            
        elif self.network_type == "goal":
            target_relevance = targets[:, :self.output_dim]
            target_urgency = targets[:, self.output_dim:]
            
            relevance, urgency = self.network(inputs)
            
            # Combine losses
            loss_relevance = F.mse_loss(relevance, target_relevance)
            loss_urgency = F.mse_loss(urgency, target_urgency)
            loss = loss_relevance + loss_urgency
            
        elif self.network_type == "reward":
            actions = kwargs.get("actions", None)
            if actions is not None:
                if isinstance(actions, np.ndarray):
                    actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                actions = actions.to(self.device)
                
                values, prefs = self.network(inputs, actions)
                
                # Target format: [value, action_prefs]
                target_values = targets[:, 0].unsqueeze(1)
                target_prefs = targets[:, 1:]
                
                # Value prediction loss
                value_loss = F.mse_loss(values, target_values)
                
                # Action preference loss (cross entropy)
                pref_loss = F.cross_entropy(prefs, torch.argmax(target_prefs, dim=1))
                
                loss = value_loss + pref_loss
            else:
                # No actions provided, can't train
                return {"error": "Actions required for reward network training"}
                
        else:
            outputs = self.network(inputs)
            loss = F.mse_loss(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "network_type": self.network_type
        }
    
    def _update_hebbian_matrix(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        """
        Update the Hebbian learning matrix
        
        This implements a simple Hebbian learning rule: "Neurons that fire
        together, wire together"
        
        Args:
            inputs: Input activations
            outputs: Output activations
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze(0)  # Remove batch dimension
        if outputs.dim() > 1:
            outputs = outputs.squeeze(0)
            
        # Compute outer product of outputs and inputs
        outer_product = torch.outer(outputs, inputs)
        
        # Update Hebbian matrix
        self.hebbian_matrix = (1 - self.hebbian_rate) * self.hebbian_matrix + self.hebbian_rate * outer_product
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level
        
        Args:
            amount: Amount to increase development level by
            
        Returns:
            New development level
        """
        self.development_level = min(1.0, max(0.0, self.development_level + amount))
        self._adjust_for_development()
        return self.development_level
    
    def save(self, path: str) -> None:
        """
        Save the network to a file
        
        Args:
            path: Path to save the network
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare save data
        save_data = {
            "network_type": self.network_type,
            "model_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "hebbian_matrix": self.hebbian_matrix.cpu().numpy(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "development_level": self.development_level,
            "hebbian_rate": self.hebbian_rate
        }
        
        # Save to file
        torch.save(save_data, path)
        logger.info(f"Saved {self.network_type} network to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the network from a file
        
        Args:
            path: Path to load the network from
        """
        if not os.path.exists(path):
            logger.error(f"Network file not found: {path}")
            return
            
        # Load data
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check network type
        if checkpoint["network_type"] != self.network_type:
            logger.warning(f"Loading {checkpoint['network_type']} network into {self.network_type} network")
        
        # Load network parameters if dimensions match
        if (checkpoint["input_dim"] == self.input_dim and 
            checkpoint["hidden_dim"] == self.hidden_dim and
            checkpoint["output_dim"] == self.output_dim):
            
            self.network.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            # Update parameters
            self.learning_rate = checkpoint["learning_rate"]
            self.development_level = checkpoint["development_level"]
            self.hebbian_rate = checkpoint["hebbian_rate"]
            
            # Load Hebbian matrix
            self.hebbian_matrix = torch.tensor(
                checkpoint["hebbian_matrix"],
                dtype=torch.float32,
                device=self.device
            )
            
            logger.info(f"Loaded {self.network_type} network from {path}")
        else:
            logger.error(f"Cannot load network due to dimension mismatch")
    
    def to_gpu(self):
        """Move the network to GPU if available"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.network.to(device)
            self.hebbian_matrix = self.hebbian_matrix.to(device)
            self.device = device
            
            # Recreate optimizer after moving model to GPU
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), 
                lr=self.learning_rate
            )
            
            logger.info(f"Moved {self.network_type} network to GPU")
    
    def to_cpu(self):
        """Move the network to CPU"""
        device = torch.device("cpu")
        self.network.to(device)
        self.hebbian_matrix = self.hebbian_matrix.to(device)
        self.device = device
        
        # Recreate optimizer after moving model to CPU
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.learning_rate
        )
        
        logger.info(f"Moved {self.network_type} network to CPU")
    
    def free_memory(self):
        """Free GPU memory"""
        if self.device.type == "cuda":
            self.to_cpu()
            torch.cuda.empty_cache()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the network
        
        Returns:
            Dictionary with state information
        """
        return {
            "network_type": self.network_type,
            "dimensions": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            },
            "development_level": self.development_level,
            "learning_rate": self.learning_rate,
            "hebbian_rate": self.hebbian_rate,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.network.parameters())
        }
