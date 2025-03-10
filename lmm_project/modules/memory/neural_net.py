from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import os
import pickle
import logging

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.neural_substrate.activation_functions import sigmoid, relu, tanh

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class MemoryNeuralNetwork:
    """
    Neural network architecture for memory systems
    
    This class provides specialized neural network models for different 
    memory types, implementing the computational aspects of memory processes.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        memory_type: str = "generic",
        learning_rate: float = 0.01,
        device: str = "auto"  # "auto", "cpu", or "cuda"
    ):
        """
        Initialize memory neural network
        
        Parameters:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        memory_type: Type of memory network (working, semantic, episodic, associative)
        learning_rate: Learning rate for training
        device: Device to run computations on ("auto", "cpu" or "cuda")
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_type = memory_type
        self.learning_rate = learning_rate
        
        # Set device based on availability
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)
            
        logger.info(f"Memory neural network created on device: {self.device}")
        
        # Create neural network based on memory type
        if memory_type == "working":
            self.network = self._create_working_memory_network()
        elif memory_type == "semantic":
            self.network = self._create_semantic_memory_network()
        elif memory_type == "episodic":
            self.network = self._create_episodic_memory_network()
        elif memory_type == "associative":
            self.network = self._create_associative_memory_network()
        else:
            self.network = self._create_generic_memory_network()
            
        # Move network to the appropriate device
        self.network.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Hebbian learning matrix for associative memory
        self.hebbian_matrix = None
        if memory_type == "associative":
            self.hebbian_matrix = np.zeros((output_dim, output_dim))
            
        # Track development level
        self.development_level = 0.0
        
        # Optimize for performance if using CUDA
        if self.device.type == "cuda":
            # Enable cuDNN benchmark for potentially faster performance
            torch.backends.cudnn.benchmark = True
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Use mixed precision if available (PyTorch 1.6+)
            if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
                logger.info("Enabling mixed precision training for memory networks")
                self.scaler = torch.cuda.amp.GradScaler()
                self.use_mixed_precision = True
            else:
                self.use_mixed_precision = False
                
    def _create_working_memory_network(self) -> nn.Module:
        """
        Create neural network for working memory
        
        Working memory maintains active representations for current processing.
        """
        # Define working memory network
        class WorkingMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                
                # LSTM for maintaining items over time
                self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 2, batch_first=True)
                
                # Output layer
                self.output = nn.Linear(hidden_dim // 2, output_dim)
                
            def forward(self, x, hidden=None):
                # Encode input
                encoded = self.encoder(x)
                
                # Reshape for LSTM if needed (batch, seq, features)
                if len(encoded.shape) == 2:
                    encoded = encoded.unsqueeze(1)
                
                # Process through LSTM
                if hidden is None:
                    lstm_out, hidden = self.lstm(encoded)
                else:
                    lstm_out, hidden = self.lstm(encoded, hidden)
                
                # Get output
                output = self.output(lstm_out[:, -1])
                
                return output, hidden
        
        return WorkingMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_semantic_memory_network(self) -> nn.Module:
        """
        Create neural network for semantic memory
        
        Semantic memory focuses on knowledge representation and concept relationships.
        """
        # Define semantic memory network
        class SemanticMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                
                # Concept encoder
                self.concept_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                # Concept embedding
                self.embedding = nn.Linear(hidden_dim, output_dim)
                
                # Concept completion network
                self.completion = nn.Sequential(
                    nn.Linear(output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
                
                # Relation network
                self.relation = nn.Bilinear(output_dim, output_dim, 1)
                
            def forward(self, x, completion_target=None):
                # Encode features
                encoded = self.concept_encoder(x)
                
                # Get concept embedding
                embedding = self.embedding(encoded)
                
                # If completion target is provided, compute reconstruction loss
                if completion_target is not None:
                    completion = self.completion(embedding)
                    completion_loss = F.mse_loss(completion, completion_target)
                else:
                    completion_loss = torch.tensor(0.0)
                    
                return {
                    "embedding": embedding,
                    "completion_loss": completion_loss
                }
                
        return SemanticMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_episodic_memory_network(self) -> nn.Module:
        """
        Create neural network for episodic memory
        
        Episodic memory stores specific events and experiences with temporal context.
        """
        # Define episodic memory network
        class EpisodicMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                
                # Content encoder
                self.content_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                
                # Context encoder
                self.context_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                
                # LSTM for temporal sequence
                self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                
                # Output layer
                self.output = nn.Linear(hidden_dim, output_dim)
                
                # Emotional salience layer
                self.emotional = nn.Sequential(
                    nn.Linear(hidden_dim, 2),  # valence and arousal
                    nn.Tanh()  # -1 to 1 for valence, 0 to 1 for arousal after adjustment
                )
                
            def forward(self, content, context=None, prev_hidden=None):
                # Encode content
                content_encoded = self.content_encoder(content)
                
                # Encode context if provided
                if context is not None:
                    context_encoded = self.context_encoder(context)
                    # Combine content and context
                    combined = torch.cat([content_encoded, context_encoded], dim=-1)
                else:
                    # Just use content with zero padding for context portion
                    padding = torch.zeros_like(content_encoded)
                    combined = torch.cat([content_encoded, padding], dim=-1)
                
                # Reshape for LSTM if needed (batch, seq, features)
                if len(combined.shape) == 2:
                    combined = combined.unsqueeze(1)
                
                # Process through LSTM
                if prev_hidden is None:
                    lstm_out, hidden = self.temporal(combined)
                else:
                    lstm_out, hidden = self.temporal(combined, prev_hidden)
                
                # Get episode embedding
                embedding = self.output(lstm_out[:, -1])
                
                # Get emotional salience
                emotion = self.emotional(lstm_out[:, -1])
                valence = emotion[:, 0]  # -1 to 1
                arousal = (emotion[:, 1] + 1) / 2  # Scale to 0 to 1
                
                return embedding, valence, arousal, hidden
                
        return EpisodicMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_associative_memory_network(self) -> nn.Module:
        """
        Create neural network for associative memory
        
        Associative memory focuses on connections between different memories.
        """
        # Define associative memory network
        class AssociativeMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                
                # Feature encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                
                # Association strength predictor (for two patterns)
                self.association_strength = nn.Sequential(
                    nn.Linear(output_dim * 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()  # 0 to 1 strength
                )
                
                # Stored patterns (for Hebbian learning)
                self.patterns = {}
                self.pattern_count = 0
                self.max_patterns = 1000
                
            def store_pattern(self, pattern):
                """Store a pattern for Hebbian learning"""
                pattern_id = str(self.pattern_count)
                self.patterns[pattern_id] = pattern
                self.pattern_count += 1
                
                # Remove oldest if at capacity
                if len(self.patterns) > self.max_patterns:
                    oldest_id = str(self.pattern_count - self.max_patterns - 1)
                    if oldest_id in self.patterns:
                        del self.patterns[oldest_id]
                
                return pattern_id
                
            def forward(self, x, hebbian_matrix=None):
                # Extract features
                features = self.encoder(x)
                
                # If Hebbian matrix provided, compute associative activations
                if hebbian_matrix is not None:
                    # Convert features to numpy for Hebbian computation
                    features_np = features.detach().cpu().numpy()
                    # Apply Hebbian associative recall
                    associations = np.dot(features_np, hebbian_matrix)
                    # Convert back to tensor
                    associations_tensor = torch.from_numpy(associations).to(features.device)
                    # Combine with direct features (residual connection)
                    output = features + 0.5 * torch.tensor(associations_tensor, dtype=features.dtype)
                else:
                    output = features
                
                return output
                
        return AssociativeMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_generic_memory_network(self) -> nn.Module:
        """Create a generic memory network used as fallback"""
        # Define a simple generic network
        class GenericMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                
            def forward(self, x):
                return self.network(x)
                
        return GenericMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def forward(self, x: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass through the memory network
        
        Parameters:
        x: Input tensor/array
        **kwargs: Additional arguments for specific memory types
        
        Returns:
        Tuple of output tensor and additional data
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Forward through appropriate network
        if self.memory_type == "working":
            hidden = kwargs.get("hidden", None)
            output, new_hidden = self.network(x, hidden)
            return output, {"hidden": new_hidden}
            
        elif self.memory_type == "semantic":
            completion_target = kwargs.get("completion_target", None)
            if completion_target is not None and isinstance(completion_target, np.ndarray):
                completion_target = torch.tensor(completion_target, dtype=torch.float32).to(self.device)
                
            results = self.network(x, completion_target)
            return results["embedding"], {"completion_loss": results["completion_loss"]}
            
        elif self.memory_type == "episodic":
            context = kwargs.get("context", None)
            prev_hidden = kwargs.get("prev_hidden", None)
            
            if context is not None and isinstance(context, np.ndarray):
                context = torch.tensor(context, dtype=torch.float32).to(self.device)
                
            embedding, valence, arousal, hidden = self.network(x, context, prev_hidden)
            return embedding, {
                "valence": valence, 
                "arousal": arousal, 
                "hidden": hidden
            }
            
        elif self.memory_type == "associative":
            # Use Hebbian matrix if available
            output = self.network(x, self.hebbian_matrix)
            return output, {}
            
        else:
            # Generic memory network
            output = self.network(x)
            return output, {}
    
    def train(
        self, 
        inputs: Union[np.ndarray, torch.Tensor], 
        targets: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the memory network
        
        Parameters:
        inputs: Training inputs
        targets: Training targets
        **kwargs: Additional training parameters
        
        Returns:
        Dictionary with training metrics
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32)
            
        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Set network to training mode
        self.network.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Use mixed precision if available and enabled
        if hasattr(self, 'use_mixed_precision') and self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs, additional = self.forward(inputs, **kwargs)
                
                # Compute loss
                if self.memory_type == "semantic":
                    # For semantic memory, combine reconstruction and target losses
                    target_loss = F.mse_loss(outputs, targets)
                    completion_loss = additional.get("completion_loss", 0.0)
                    loss = target_loss + completion_loss
                else:
                    # Standard loss for other memory types
                    loss = F.mse_loss(outputs, targets)
                
            # Scale gradients and optimize
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training process
            # Forward pass
            outputs, additional = self.forward(inputs, **kwargs)
            
            # Compute loss
            if self.memory_type == "semantic":
                # For semantic memory, combine reconstruction and target losses
                target_loss = F.mse_loss(outputs, targets)
                completion_loss = additional.get("completion_loss", 0.0)
                loss = target_loss + completion_loss
            else:
                # Standard loss for other memory types
                loss = F.mse_loss(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
        
        # Update Hebbian matrix for associative memory
        if self.memory_type == "associative" and outputs.shape[0] > 0:
            output_np = outputs.detach().cpu().numpy()
            # Extract first example if batch
            if len(output_np.shape) > 1:
                output_np = output_np[0]
            self._update_hebbian_matrix(output_np)
        
        # Return training metrics
        return {
            "loss": loss.item(),
            "type": self.memory_type
        }
    
    def _update_hebbian_matrix(self, features: np.ndarray) -> None:
        """Update Hebbian matrix for associative memory"""
        if self.hebbian_matrix is None:
            return
            
        # Update Hebbian connections
        # Outer product represents connection strengths between all pairs of neurons
        outer_product = np.outer(features, features)
        
        # Scale learning based on development level
        # More primitive reinforcement at early stages, more nuanced at later stages
        hebbian_lr = 0.01 + (0.04 * self.development_level)
        
        # Apply development-dependent learning rate
        self.hebbian_matrix = (1.0 - hebbian_lr) * self.hebbian_matrix + hebbian_lr * outer_product
        
        # Apply normalization to prevent runaway values
        max_val = np.max(np.abs(self.hebbian_matrix))
        if max_val > 0:
            self.hebbian_matrix /= max_val
    
    def update_development(self, amount: float) -> float:
        """
        Update developmental level and adjust network accordingly
        
        Parameters:
        amount: Amount to increase development level by
        
        Returns:
        New development level
        """
        # Update development level
        old_level = self.development_level
        self.development_level = min(1.0, max(0.0, self.development_level + amount))
        
        # Only make adjustments if significant change
        if abs(self.development_level - old_level) < 0.01:
            return self.development_level
            
        # Adjust network based on development level
        if self.memory_type == "working":
            # Working memory LSTM layers might get more complex with development
            if hasattr(self.network, 'lstm'):
                # Adjust dropout based on development (lower dropout at higher levels)
                for module in self.network.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = max(0.1, 0.5 - (self.development_level * 0.4))
        
        elif self.memory_type == "associative":
            # Associative memory might have more patterns with development
            if hasattr(self.network, 'max_patterns'):
                self.network.max_patterns = int(1000 + (9000 * self.development_level))
        
        # Adjust learning rate based on development
        # Higher development = more refined, slower learning
        self.learning_rate = max(0.001, 0.01 - (0.008 * self.development_level))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
            
        logger.info(f"Updated memory network development to {self.development_level:.2f}")
        return self.development_level
    
    def save(self, path: str) -> None:
        """
        Save memory network to disk
        
        Parameters:
        path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to CPU to avoid issues with GPU-specific tensors
        device_backup = next(self.network.parameters()).device
        cpu_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        
        # Prepare data to save
        save_data = {
            "network_state": cpu_state_dict,
            "memory_type": self.memory_type,
            "dimensions": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            },
            "development_level": self.development_level,
            "learning_rate": self.learning_rate
        }
        
        # Add Hebbian matrix if applicable
        if self.hebbian_matrix is not None:
            save_data["hebbian_matrix"] = self.hebbian_matrix
        
        # Save data
        torch.save(save_data, path)
        
        # Restore model to original device
        self.network.to(device_backup)
        logger.info(f"Memory network saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load memory network from disk
        
        Parameters:
        path: Path to the saved model
        """
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return
            
        try:
            # Load data using the current device
            save_data = torch.load(path, map_location=self.device)
            
            # Check if memory type matches
            if save_data["memory_type"] != self.memory_type:
                logger.warning(f"Memory type mismatch: saved={save_data['memory_type']}, current={self.memory_type}")
                # Create a new network of the correct type
                self.memory_type = save_data["memory_type"]
                if self.memory_type == "working":
                    self.network = self._create_working_memory_network()
                elif self.memory_type == "semantic":
                    self.network = self._create_semantic_memory_network()
                elif self.memory_type == "episodic":
                    self.network = self._create_episodic_memory_network()
                elif self.memory_type == "associative":
                    self.network = self._create_associative_memory_network()
                else:
                    self.network = self._create_generic_memory_network()
                
                # Move to correct device
                self.network.to(self.device)
            
            # Load network state
            self.network.load_state_dict(save_data["network_state"])
            
            # Update parameters
            self.input_dim = save_data["dimensions"]["input_dim"]
            self.hidden_dim = save_data["dimensions"]["hidden_dim"]
            self.output_dim = save_data["dimensions"]["output_dim"]
            self.development_level = save_data["development_level"]
            self.learning_rate = save_data["learning_rate"]
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            
            # Load Hebbian matrix if available
            if "hebbian_matrix" in save_data and self.memory_type == "associative":
                self.hebbian_matrix = save_data["hebbian_matrix"]
            
            logger.info(f"Memory network loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading memory network: {e}")
            
    def to_gpu(self):
        """Move network to GPU if available"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.network.to(device)
            self.device = device
            logger.info("Memory network moved to GPU")
            
            # Enable mixed precision if available
            if hasattr(torch.cuda, 'amp'):
                logger.info("Enabling mixed precision for memory network")
                self.scaler = torch.cuda.amp.GradScaler()
                self.use_mixed_precision = True
        else:
            logger.warning("GPU not available, network remains on CPU")
            
    def to_cpu(self):
        """Move network to CPU"""
        device = torch.device("cpu")
        self.network.to(device)
        self.device = device
        self.use_mixed_precision = False
        logger.info("Memory network moved to CPU")
        
    def free_memory(self):
        """Free memory used by the network (useful for GPU memory management)"""
        if self.device.type == "cuda":
            # Move to CPU temporarily
            self.to_cpu()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            logger.info("Memory network GPU memory freed")
                
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the memory network
        
        Returns:
        Dictionary with state information
        """
        return {
            "memory_type": self.memory_type,
            "dimensions": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            },
            "development_level": self.development_level,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.network.parameters())
        }
