from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import os
import pickle

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.neural_substrate.activation_functions import sigmoid, relu, tanh

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
        device: str = "cpu"
    ):
        """
        Initialize memory neural network
        
        Parameters:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        memory_type: Type of memory network (working, semantic, episodic, associative)
        learning_rate: Learning rate for training
        device: Device to run computations on ("cpu" or "cuda")
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_type = memory_type
        self.learning_rate = learning_rate
        self.device = device
        
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
            
        # Move network to device
        self.network.to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Track development level (affects network complexity)
        self.development_level = 0.0
        
        # Hebbian learning components
        self.hebbian_learning_enabled = True
        self.hebbian_learning_rate = 0.001
        self.hebbian_decay_rate = 0.0001
        
        # Association matrices for hebbian learning
        self.association_matrix = np.zeros((hidden_dim, hidden_dim))
    
    def _create_working_memory_network(self) -> nn.Module:
        """
        Create neural network for working memory
        
        Working memory requires:
        - Fast update capability
        - Limited capacity
        - Decay over time
        - Attention-based gating
        """
        class WorkingMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, hidden_dim)
                self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
                self.decoder = nn.Linear(hidden_dim, output_dim)
                self.gate = nn.Linear(hidden_dim, 1)
                
            def forward(self, x, hidden=None):
                # Encode input
                x = F.relu(self.encoder(x))
                
                # Reshape for GRU if needed
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                # Process with GRU
                if hidden is None:
                    output, hidden = self.gru(x)
                else:
                    output, hidden = self.gru(x, hidden)
                
                # Apply attention
                attn_output, _ = self.attention(output, output, output)
                
                # Apply gating (determines what enters working memory)
                gate_values = torch.sigmoid(self.gate(attn_output))
                gated_output = gate_values * attn_output
                
                # Decode output
                output = self.decoder(gated_output)
                
                return output, hidden
                
        return WorkingMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_semantic_memory_network(self) -> nn.Module:
        """
        Create neural network for semantic memory
        
        Semantic memory requires:
        - Pattern completion
        - Hierarchical structure
        - Concept association
        """
        class SemanticMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                # Lower-level feature extraction
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                # Concept association layers
                self.association = nn.Linear(hidden_dim, hidden_dim)
                
                # Hierarchical structure
                self.hierarchy_up = nn.Linear(hidden_dim, hidden_dim // 2)
                self.hierarchy_down = nn.Linear(hidden_dim // 2, hidden_dim)
                
                # Output layer
                self.decoder = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x, completion_target=None):
                # Encode features
                features = self.encoder(x)
                
                # Apply associative activation
                assoc = torch.sigmoid(self.association(features))
                features = features * assoc
                
                # Hierarchical processing (abstraction)
                abstract = F.relu(self.hierarchy_up(features))
                
                # Pattern completion (if partial input)
                if completion_target is not None:
                    # Use abstract representation to fill in missing details
                    completed = self.hierarchy_down(abstract)
                    # Blend with target
                    mask = (completion_target != 0).float()
                    features = mask * completion_target + (1 - mask) * completed
                
                # Final output
                output = self.decoder(features)
                return output, features
                
        return SemanticMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_episodic_memory_network(self) -> nn.Module:
        """
        Create neural network for episodic memory
        
        Episodic memory requires:
        - Temporal sequence encoding
        - Context binding
        - Emotional tagging
        - Vividness modulation
        """
        class EpisodicMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                # Content encoder
                self.content_encoder = nn.Linear(input_dim, hidden_dim)
                
                # Context encoder
                self.context_encoder = nn.Linear(input_dim, hidden_dim // 2)
                
                # Temporal sequence processing
                self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                
                # Emotional tagging
                self.emotion_encoder = nn.Linear(input_dim, 2)  # valence, arousal
                
                # Context binding mechanism
                self.context_binding = nn.Bilinear(hidden_dim, hidden_dim // 2, hidden_dim)
                
                # Vividness modulation
                self.vividness = nn.Linear(hidden_dim, 1)
                
                # Decoder
                self.decoder = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, content, context=None, prev_hidden=None):
                # Encode content
                content_features = F.relu(self.content_encoder(content))
                
                # Reshape for LSTM if needed
                if len(content_features.shape) == 2:
                    content_features = content_features.unsqueeze(1)
                
                # Process temporal aspects
                if prev_hidden is None:
                    temporal_output, (h, c) = self.temporal(content_features)
                else:
                    temporal_output, (h, c) = self.temporal(content_features, prev_hidden)
                
                # Emotional tagging
                emotion = torch.tanh(self.emotion_encoder(content))
                
                # Apply context binding if context provided
                if context is not None:
                    context_features = F.relu(self.context_encoder(context))
                    bound_features = self.context_binding(
                        temporal_output.squeeze(1), context_features
                    )
                else:
                    bound_features = temporal_output.squeeze(1)
                
                # Apply vividness modulation
                vividness = torch.sigmoid(self.vividness(bound_features))
                modulated_features = vividness * bound_features
                
                # Generate output
                output = self.decoder(modulated_features)
                
                return output, (h, c), emotion, vividness
                
        return EpisodicMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_associative_memory_network(self) -> nn.Module:
        """
        Create neural network for associative memory
        
        Associative memory requires:
        - Pattern association
        - Hebbian learning
        - Pattern completion
        - Spreading activation
        """
        class AssociativeMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                # Feature extraction
                self.encoder = nn.Linear(input_dim, hidden_dim)
                
                # Associative network (fully connected)
                self.association_layer = nn.Linear(hidden_dim, hidden_dim)
                
                # Pattern completion
                self.completion_layer = nn.Linear(hidden_dim, hidden_dim)
                
                # Output generation
                self.decoder = nn.Linear(hidden_dim, output_dim)
                
                # Stored patterns for association
                self.register_buffer("stored_patterns", torch.zeros(100, hidden_dim))
                self.pattern_count = 0
                
            def store_pattern(self, pattern):
                """Store a pattern for future association"""
                if self.pattern_count < 100:
                    self.stored_patterns[self.pattern_count] = pattern
                    self.pattern_count += 1
                else:
                    # Replace oldest pattern (simple circular buffer)
                    self.stored_patterns[self.pattern_count % 100] = pattern
                    self.pattern_count += 1
                
            def forward(self, x, hebbian_matrix=None):
                # Extract features
                features = F.relu(self.encoder(x))
                
                # Apply associative layer
                associations = torch.tanh(self.association_layer(features))
                
                # If hebbian matrix provided, apply Hebbian learning
                if hebbian_matrix is not None:
                    # Convert to tensor if numpy array
                    if isinstance(hebbian_matrix, np.ndarray):
                        hebbian_matrix = torch.tensor(
                            hebbian_matrix, 
                            device=x.device, 
                            dtype=torch.float32
                        )
                    
                    # Apply Hebbian associations
                    hebbian_associations = torch.matmul(features, hebbian_matrix)
                    associations = associations + hebbian_associations
                
                # Check for pattern completion with stored patterns
                if self.pattern_count > 0:
                    # Calculate similarity with stored patterns
                    similarities = torch.matmul(
                        features, 
                        self.stored_patterns[:self.pattern_count].t()
                    )
                    # Get most similar pattern
                    max_sim, idx = torch.max(similarities, dim=1)
                    # If similarity above threshold, blend with pattern
                    threshold = 0.7
                    mask = (max_sim > threshold).float().unsqueeze(1)
                    most_similar = self.stored_patterns[idx]
                    completion = self.completion_layer(most_similar)
                    features = mask * completion + (1 - mask) * features
                
                # Generate output
                output = self.decoder(features)
                
                return output, features
                
        return AssociativeMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def _create_generic_memory_network(self) -> nn.Module:
        """Create a generic neural network for memory"""
        class GenericMemoryNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                
            def forward(self, x):
                return self.model(x), None
                
        return GenericMemoryNetwork(self.input_dim, self.hidden_dim, self.output_dim)
    
    def forward(self, x: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass through the neural network
        
        Parameters:
        x: Input data
        **kwargs: Additional arguments for specific memory types
        
        Returns:
        Tuple of (output, additional_outputs)
        """
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Make sure x is float32
        x = x.to(dtype=torch.float32, device=self.device)
        
        # Forward pass depends on memory type
        with torch.no_grad():
            if self.memory_type == "working":
                hidden = kwargs.get("hidden", None)
                output, hidden = self.network(x, hidden)
                return output, hidden
                
            elif self.memory_type == "semantic":
                completion_target = kwargs.get("completion_target", None)
                output, features = self.network(x, completion_target)
                return output, features
                
            elif self.memory_type == "episodic":
                context = kwargs.get("context", None)
                prev_hidden = kwargs.get("prev_hidden", None)
                output, hidden, emotion, vividness = self.network(x, context, prev_hidden)
                return output, (hidden, emotion, vividness)
                
            elif self.memory_type == "associative":
                # Apply Hebbian learning if enabled
                if self.hebbian_learning_enabled:
                    output, features = self.network(x, self.association_matrix)
                    # Update association matrix with new pattern
                    self._update_hebbian_matrix(features.detach().cpu().numpy())
                    return output, features
                else:
                    output, features = self.network(x)
                    return output, features
                    
            else:
                output, _ = self.network(x)
                return output, None
    
    def train(
        self, 
        inputs: Union[np.ndarray, torch.Tensor], 
        targets: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the neural network
        
        Parameters:
        inputs: Input data
        targets: Target outputs
        **kwargs: Additional arguments for specific memory types
        
        Returns:
        Dictionary with training metrics
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Set network to training mode
        self.network.train()
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.memory_type == "working":
            hidden = kwargs.get("hidden", None)
            outputs, hidden = self.network(inputs, hidden)
            loss = F.mse_loss(outputs, targets)
            
        elif self.memory_type == "semantic":
            completion_target = kwargs.get("completion_target", None)
            outputs, features = self.network(inputs, completion_target)
            loss = F.mse_loss(outputs, targets)
            
        elif self.memory_type == "episodic":
            context = kwargs.get("context", None)
            prev_hidden = kwargs.get("prev_hidden", None)
            outputs, (hidden, emotion, vividness) = self.network(inputs, context, prev_hidden)
            
            # Optional emotional target
            emotion_targets = kwargs.get("emotion_targets", None)
            if emotion_targets is not None:
                # Combine content loss and emotion loss
                content_loss = F.mse_loss(outputs, targets)
                emotion_loss = F.mse_loss(emotion, emotion_targets)
                loss = content_loss + 0.5 * emotion_loss
            else:
                loss = F.mse_loss(outputs, targets)
                
        elif self.memory_type == "associative":
            outputs, features = self.network(inputs)
            loss = F.mse_loss(outputs, targets)
            
            # Store pattern for future associations
            if kwargs.get("store_pattern", False):
                self.network.store_pattern(features.detach())
                
        else:
            outputs, _ = self.network(inputs)
            loss = F.mse_loss(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item()
        }
    
    def _update_hebbian_matrix(self, features: np.ndarray) -> None:
        """
        Update Hebbian association matrix
        
        Implements Hebbian learning rule: "Neurons that fire together, wire together"
        
        Parameters:
        features: Feature activations
        """
        if len(features.shape) > 1:
            # Use first example if batch
            features = features[0]
            
        # Outer product of features with itself
        associations = np.outer(features, features)
        
        # Apply learning rate
        delta = self.hebbian_learning_rate * associations
        
        # Update association matrix (with decay)
        self.association_matrix = (1 - self.hebbian_decay_rate) * self.association_matrix + delta
    
    def update_development(self, amount: float) -> float:
        """
        Update the network's developmental level
        
        As the network develops:
        - More complex representations emerge
        - Learning becomes more efficient
        - Connections become more stable
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Development affects learning parameters
        delta = self.development_level - prev_level
        
        # Adjust learning rate (decreases as network matures)
        self.learning_rate = max(0.001, self.learning_rate - delta * 0.005)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
            
        # Increase Hebbian learning capabilities
        self.hebbian_learning_rate = min(0.01, self.hebbian_learning_rate + delta * 0.001)
        self.hebbian_decay_rate = max(0.00001, self.hebbian_decay_rate - delta * 0.0001)
        
        return self.development_level
    
    def save(self, path: str) -> None:
        """
        Save the neural network to disk
        
        Parameters:
        path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save network parameters
        torch.save(self.network.state_dict(), f"{path}_weights.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), f"{path}_optimizer.pt")
        
        # Save Hebbian matrix and other parameters
        with open(f"{path}_config.pkl", 'wb') as f:
            pickle.dump({
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'memory_type': self.memory_type,
                'learning_rate': self.learning_rate,
                'development_level': self.development_level,
                'hebbian_learning_enabled': self.hebbian_learning_enabled,
                'hebbian_learning_rate': self.hebbian_learning_rate,
                'hebbian_decay_rate': self.hebbian_decay_rate,
                'association_matrix': self.association_matrix
            }, f)
    
    def load(self, path: str) -> None:
        """
        Load the neural network from disk
        
        Parameters:
        path: Path to load the model from
        """
        # Load network parameters
        self.network.load_state_dict(torch.load(f"{path}_weights.pt"))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(f"{path}_optimizer.pt"))
        
        # Load Hebbian matrix and other parameters
        with open(f"{path}_config.pkl", 'rb') as f:
            config = pickle.load(f)
            
            # Update parameters
            self.development_level = config['development_level']
            self.learning_rate = config['learning_rate']
            self.hebbian_learning_enabled = config['hebbian_learning_enabled']
            self.hebbian_learning_rate = config['hebbian_learning_rate']
            self.hebbian_decay_rate = config['hebbian_decay_rate']
            self.association_matrix = config['association_matrix']
