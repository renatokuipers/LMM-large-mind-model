import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

class EmotionEncoder(nn.Module):
    """
    Neural network for encoding emotion-relevant features into embedding space
    
    This model takes features from stimuli and context and maps them to
    dimensional (valence-arousal) emotion space.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 32):
        """
        Initialize the emotion encoder network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of emotion embedding space
        """
        super().__init__()
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers for feature extraction
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output projection to valence-arousal space
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Separate heads for valence and arousal prediction
        self.valence_head = nn.Linear(output_dim, 1)  # valence (-1 to 1)
        self.arousal_head = nn.Linear(output_dim, 1)  # arousal (0 to 1)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Developmental parameters (controlled externally)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input features to predict valence and arousal
        
        Args:
            x: Input tensor of features [batch_size, input_dim]
            
        Returns:
            Dictionary with emotion embedding, valence, and arousal
        """
        # Project input to hidden dimension
        hidden = self.input_projection(x)
        hidden = F.relu(hidden)
        
        # Process through hidden layers with residual connection
        hidden_output = self.hidden_layers(hidden)
        hidden = hidden + self.layer_norm(hidden_output)  # Residual connection
        
        # Project to embedding space
        embedding = self.output_projection(hidden)
        
        # Developmental modulation - more complex processing at higher development
        if self.developmental_factor.item() < 0.3:
            # Very basic processing at early development
            embedding = 0.5 * embedding + 0.5 * torch.randn_like(embedding) * 0.1
        
        # Predict valence (-1 to 1)
        valence = torch.tanh(self.valence_head(embedding))
        
        # Predict arousal (0 to 1)
        arousal = torch.sigmoid(self.arousal_head(embedding))
        
        return {
            "embedding": embedding,
            "valence": valence,
            "arousal": arousal
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the encoder
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class EmotionClassifierNetwork(nn.Module):
    """
    Neural network for mapping dimensional emotions to categorical emotions
    
    This model takes valence-arousal representations and classifies them
    into discrete emotion categories with confidence scores.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_emotions: int = 8):
        """
        Initialize the emotion classifier network
        
        Args:
            input_dim: Dimension of input (typically 2 for valence and arousal)
            hidden_dim: Dimension of hidden layers
            num_emotions: Number of discrete emotion categories
        """
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Emotion category prediction (logits)
        self.emotion_classifier = nn.Linear(hidden_dim, num_emotions)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Emotion centers in valence-arousal space (for basic classification)
        # Format: [valence, arousal] where valence is -1 to 1, arousal is 0 to 1
        self.emotion_centers = {
            "joy": torch.tensor([0.8, 0.7]),
            "sadness": torch.tensor([-0.7, 0.3]), 
            "anger": torch.tensor([-0.6, 0.8]),
            "fear": torch.tensor([-0.8, 0.9]),
            "disgust": torch.tensor([-0.5, 0.5]),
            "surprise": torch.tensor([0.1, 0.9]),
            "trust": torch.tensor([0.7, 0.4]),
            "anticipation": torch.tensor([0.5, 0.7])
        }
    
    def forward(self, valence: torch.Tensor, arousal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify valence-arousal values into discrete emotions
        
        Args:
            valence: Tensor of valence values [-1,1]
            arousal: Tensor of arousal values [0,1]
            
        Returns:
            Dictionary with emotion logits and probabilities
        """
        # Combine valence and arousal
        x = torch.cat([valence, arousal], dim=-1)
        
        # Check developmental level to determine classification approach
        dev_level = self.developmental_factor.item()
        
        if dev_level < 0.3:
            # Simple classification based on closest emotion center
            emotion_probs = self._basic_classification(valence, arousal)
            return {
                "emotion_probs": emotion_probs
            }
        
        # Neural classification for more developed stages
        hidden = F.relu(self.input_layer(x))
        hidden = self.hidden_layers(hidden)
        emotion_logits = self.emotion_classifier(hidden)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs
        }
    
    def _basic_classification(self, valence: torch.Tensor, arousal: torch.Tensor) -> torch.Tensor:
        """
        Basic classification based on distance to emotion centers
        
        Args:
            valence: Tensor of valence values [-1,1]
            arousal: Tensor of arousal values [0,1]
            
        Returns:
            Tensor of emotion probabilities
        """
        # Combine valence and arousal into coordinates
        coords = torch.cat([valence, arousal], dim=-1)  # [batch_size, 2]
        
        # Get available emotions based on development level
        if self.developmental_factor.item() < 0.2:
            # Very basic emotions only
            available_emotions = ["joy", "sadness"]
        elif self.developmental_factor.item() < 0.5:
            # Primary emotions
            available_emotions = ["joy", "sadness", "anger", "fear"]
        else:
            # All emotions
            available_emotions = list(self.emotion_centers.keys())
        
        # Create a tensor for emotion centers
        centers = torch.stack([self.emotion_centers[e] for e in available_emotions])  # [num_emotions, 2]
        
        # Calculate distances to all emotion centers
        # Reshape coordinates to [batch_size, 1, 2] for broadcasting
        coords_expanded = coords.unsqueeze(1)  # [batch_size, 1, 2]
        
        # Calculate squared Euclidean distance
        # This gives us [batch_size, num_emotions]
        distances = torch.sum((coords_expanded - centers) ** 2, dim=2)
        
        # Convert distances to probabilities (inverse relationship)
        # Use softmax with negative distances (smaller distance = higher probability)
        probs = F.softmax(-distances, dim=1)
        
        return probs
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the classifier
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class SentimentNetwork(nn.Module):
    """
    Neural network for analyzing sentiment in text
    
    This model processes text features to detect emotional tone,
    sentiment polarity, and specific emotion indicators.
    """
    
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 64, 
                 hidden_dim: int = 128, output_dim: int = 4):
        """
        Initialize the sentiment analysis network
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (typically 4 for pos/neg/neutral/compound)
        """
        super().__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layers for different sentiment aspects
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, output_dim)  # positive, negative, neutral, compound
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process text to analyze sentiment
        
        Args:
            x: Input tensor of token ids [batch_size, seq_length]
            lengths: Tensor of sequence lengths [batch_size]
            
        Returns:
            Dictionary with sentiment scores and features
        """
        # Convert token ids to embeddings
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Pack padded sequence for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, seq_length, hidden_dim*2]
        
        # Apply attention to focus on sentiment-relevant parts
        attention_scores = self.attention(output).squeeze(-1)  # [batch_size, seq_length]
        
        # Create attention mask based on sequence lengths
        mask = torch.arange(output.size(1), device=x.device).expand(output.size(0), -1) < lengths.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_length, 1]
        
        # Apply attention weights to get context vector
        context = torch.sum(attention_weights * output, dim=1)  # [batch_size, hidden_dim*2]
        
        # Classify sentiment
        sentiment_logits = self.sentiment_classifier(context)  # [batch_size, 4]
        
        # Split into individual sentiment components
        positive = torch.sigmoid(sentiment_logits[:, 0])
        negative = torch.sigmoid(sentiment_logits[:, 1])
        neutral = torch.sigmoid(sentiment_logits[:, 2])
        
        # Compound score derived from positive and negative
        # (scaled to range from -1 to 1)
        compound = positive - negative
        
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "compound": compound,
            "features": context
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the sentiment analyzer
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class EmotionRegulationNetwork(nn.Module):
    """
    Neural network for emotion regulation
    
    This model processes current emotional state and regulation goals
    to produce regulated emotional state.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 2):
        """
        Initialize the emotion regulation network
        
        Args:
            input_dim: Dimension of input (emotional state + regulation goals)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (typically 2 for valence and arousal)
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Context encoding (for regulation context)
        self.context_encoder = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Success prediction (how successful the regulation attempt will be)
        self.success_predictor = nn.Linear(hidden_dim, 1)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
    
    def forward(self, 
                current_state: torch.Tensor, 
                target_state: torch.Tensor, 
                regulation_strategy: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply emotion regulation to current state
        
        Args:
            current_state: Current emotional state [batch_size, 2] (valence, arousal)
            target_state: Target emotional state [batch_size, 2] (valence, arousal)
            regulation_strategy: One-hot encoded regulation strategy [batch_size, num_strategies]
            context: Optional context information [batch_size, context_dim]
            
        Returns:
            Dictionary with regulated state and success prediction
        """
        # Combine inputs
        inputs = torch.cat([current_state, target_state, regulation_strategy], dim=-1)
        if context is not None:
            inputs = torch.cat([inputs, context], dim=-1)
        
        # Project input
        hidden = F.relu(self.input_projection(inputs))
        
        # Encode context (or use a default if none provided)
        if context is not None:
            context_hidden = F.relu(self.context_encoder(context))
        else:
            context_hidden = torch.zeros_like(hidden)
        
        # Combine input and context
        combined = torch.cat([hidden, context_hidden], dim=-1)
        
        # Process through hidden layers
        hidden = self.hidden_layers(combined)
        
        # Development-based modulation
        dev_level = self.developmental_factor.item()
        
        # Limited regulation capabilities at early development
        if dev_level < 0.3:
            # Early development - limited regulation capability
            # Regulated state is closer to current than target
            weight = 0.2 * dev_level  # Very limited movement toward target
            regulated_state = current_state * (1 - weight) + target_state * weight
            success = torch.tensor([[0.2 * dev_level]], device=current_state.device)
            
            return {
                "regulated_state": regulated_state,
                "success": success
            }
        
        # More advanced regulation for higher development
        
        # Project to regulated state
        regulated_delta = self.output_projection(hidden)
        
        # Apply delta to current state (bounded to prevent extreme changes)
        regulated_state = current_state + torch.tanh(regulated_delta) * (0.5 + 0.5 * dev_level)
        
        # Ensure valence stays in [-1,1] and arousal in [0,1]
        regulated_state_constrained = torch.stack([
            torch.clamp(regulated_state[:, 0], -1.0, 1.0),  # valence
            torch.clamp(regulated_state[:, 1], 0.0, 1.0)    # arousal
        ], dim=1)
        
        # Predict regulation success
        success = torch.sigmoid(self.success_predictor(hidden))
        
        # Developmental factor affects success
        success = success * (0.5 + 0.5 * dev_level)
        
        return {
            "regulated_state": regulated_state_constrained,
            "success": success
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the regulator
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


def get_device() -> torch.device:
    """
    Get the appropriate device (GPU if available, otherwise CPU)
    
    Returns:
        torch.device: The device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
