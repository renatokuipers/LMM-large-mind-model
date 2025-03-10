import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU")
        return torch.device("cuda")
    logger.info("CUDA is not available, using CPU")
    return torch.device("cpu")

class MentalStateEncoder(nn.Module):
    """
    Neural network for encoding and processing mental states
    
    This network converts belief, desire, and intention representations 
    into a condensed vector encoding for theory of mind processing.
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 hidden_dim: int = 64, 
                 output_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the mental state encoder"""
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def encode_mental_state(self, mental_state_dict: Dict[str, Any], embeddings: Dict[str, List[float]]) -> torch.Tensor:
        """
        Convert a mental state dictionary into a tensor representation
        
        Args:
            mental_state_dict: Dictionary containing mental state elements
            embeddings: Dictionary mapping concepts to embedding vectors
            
        Returns:
            Tensor representation of the mental state
        """
        # Combine embeddings weighted by strength/confidence
        vector_list = []
        
        # Process beliefs
        for belief, confidence in mental_state_dict.get("beliefs", {}).items():
            if belief in embeddings:
                weighted_vector = np.array(embeddings[belief]) * confidence
                vector_list.append(weighted_vector)
        
        # Process desires
        for desire, strength in mental_state_dict.get("desires", {}).items():
            if desire in embeddings:
                weighted_vector = np.array(embeddings[desire]) * strength
                vector_list.append(weighted_vector)
        
        # Process intentions
        for intention, commitment in mental_state_dict.get("intentions", {}).items():
            if intention in embeddings:
                weighted_vector = np.array(embeddings[intention]) * commitment
                vector_list.append(weighted_vector)
        
        # Process emotions
        for emotion, intensity in mental_state_dict.get("emotions", {}).items():
            if emotion in embeddings:
                weighted_vector = np.array(embeddings[emotion]) * intensity
                vector_list.append(weighted_vector)
        
        # Combine vectors (average)
        if not vector_list:
            # If no embeddings, return a zero vector
            return torch.zeros(self.input_dim, device=self.device)
        
        combined_vector = np.mean(vector_list, axis=0)
        
        # Ensure the vector has the correct dimension
        if len(combined_vector) < self.input_dim:
            # Pad if too small
            padding = np.zeros(self.input_dim - len(combined_vector))
            combined_vector = np.concatenate([combined_vector, padding])
        elif len(combined_vector) > self.input_dim:
            # Truncate if too large
            combined_vector = combined_vector[:self.input_dim]
        
        return torch.tensor(combined_vector, dtype=torch.float32, device=self.device)

class RelationshipNetwork(nn.Module):
    """
    Neural network for modeling relationships between agents
    
    This network processes relationship attributes and history to predict
    future relationship dynamics and expectations.
    """
    
    def __init__(self, 
                 input_dim: int = 64, 
                 hidden_dim: int = 128, 
                 output_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process input through the relationship network"""
        x = x.to(self.device)
        embedded = F.relu(self.embedding(x))
        embedded = self.dropout(embedded)
        
        if hidden is None:
            output, (hidden_state, cell_state) = self.lstm(embedded.unsqueeze(0))
        else:
            hidden_state, cell_state = hidden
            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)
            output, (hidden_state, cell_state) = self.lstm(embedded.unsqueeze(0), (hidden_state, cell_state))
        
        output = self.dropout(output.squeeze(0))
        output = self.fc(output)
        
        return output, (hidden_state, cell_state)

class MoralReasoningNetwork(nn.Module):
    """
    Neural network for moral reasoning
    
    This network evaluates actions and situations using ethical principles
    and produces moral judgments.
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 principle_dim: int = 32,
                 hidden_dim: int = 64, 
                 output_dim: int = 3,  # judgment, confidence, reasoning_vector
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.principle_dim = principle_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Principle encoder
        self.principle_encoder = nn.Sequential(
            nn.Linear(principle_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined processing
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, action_vector: torch.Tensor, principle_vectors: torch.Tensor) -> torch.Tensor:
        """
        Process action and principles through the moral reasoning network
        
        Args:
            action_vector: Vector representing the action being evaluated
            principle_vectors: Tensor of principle vectors (batch_size, num_principles, principle_dim)
            
        Returns:
            Tensor with moral judgment, confidence, and reasoning vector
        """
        action_vector = action_vector.to(self.device)
        principle_vectors = principle_vectors.to(self.device)
        
        # Encode action
        action_encoding = self.action_encoder(action_vector)
        
        # Process each principle
        batch_size, num_principles, _ = principle_vectors.shape
        principle_vectors = principle_vectors.view(-1, self.principle_dim)
        principle_encodings = self.principle_encoder(principle_vectors)
        principle_encodings = principle_encodings.view(batch_size, num_principles, -1)
        
        # Apply attention to principles
        attention_scores = self.attention(principle_encodings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        weighted_principles = torch.sum(attention_weights * principle_encodings, dim=1)
        
        # Combine action and weighted principles
        combined = torch.cat([action_encoding, weighted_principles], dim=1)
        output = self.combiner(combined)
        
        return output

class NormProcessingNetwork(nn.Module):
    """
    Neural network for processing social norms
    
    This network identifies relevant norms for contexts and
    detects norm violations.
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 norm_dim: int = 64,
                 hidden_dim: int = 96, 
                 output_dim: int = 3,  # relevance, violation_score, confidence
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.norm_dim = norm_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Norm encoder
        self.norm_encoder = nn.Sequential(
            nn.Linear(norm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Relevance predictor
        self.relevance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, context_vector: torch.Tensor, norm_vectors: torch.Tensor) -> torch.Tensor:
        """
        Process context and norms to determine relevance and violations
        
        Args:
            context_vector: Vector representing the social context
            norm_vectors: Tensor of norm vectors (batch_size, num_norms, norm_dim)
            
        Returns:
            Tensor with relevance scores, violation scores, and confidence for each norm
        """
        context_vector = context_vector.to(self.device)
        norm_vectors = norm_vectors.to(self.device)
        
        # Encode context
        context_encoding = self.context_encoder(context_vector)
        
        # Process each norm
        batch_size, num_norms, _ = norm_vectors.shape
        norm_vectors = norm_vectors.view(-1, self.norm_dim)
        norm_encodings = self.norm_encoder(norm_vectors)
        norm_encodings = norm_encodings.view(batch_size, num_norms, -1)
        
        # Expand context encoding for combination with each norm
        expanded_context = context_encoding.unsqueeze(1).expand(-1, num_norms, -1)
        
        # Combine context and norms
        combined = torch.cat([expanded_context, norm_encodings], dim=2)
        combined = combined.view(-1, self.hidden_dim * 2)
        
        # Predict relevance, violation, and confidence
        output = self.relevance_predictor(combined)
        output = output.view(batch_size, num_norms, -1)
        
        return output
