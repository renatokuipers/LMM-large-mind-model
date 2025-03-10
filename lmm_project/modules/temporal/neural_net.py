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

# Sequence Learning Networks
class SequenceEncoder(nn.Module):
    """
    Neural network for encoding and processing sequences
    
    This network encodes sequential patterns and learns transition probabilities
    between sequence elements, enabling pattern recognition and prediction.
    """
    
    def __init__(self, 
                 input_dim: int = 128, 
                 hidden_dim: int = 256, 
                 embedding_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Element embedding
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Sequence processing with LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process sequence through the network"""
        x = x.to(self.device)
        
        # Embed input elements
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Process through LSTM
        if hidden is None:
            output, (hidden_state, cell_state) = self.lstm(embedded)
        else:
            hidden_state, cell_state = hidden
            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)
            output, (hidden_state, cell_state) = self.lstm(embedded, (hidden_state, cell_state))
        
        # Apply dropout to output
        output = self.dropout(output)
        
        # Project to embedding space
        output = self.output_projection(output)
        
        return output, (hidden_state, cell_state)
    
    def predict_next(self, sequence: torch.Tensor) -> torch.Tensor:
        """Predict the next element in a sequence"""
        # Forward pass through the model
        output, _ = self.forward(sequence)
        
        # Get the last output (prediction for next element)
        next_element = output[:, -1, :]
        
        return next_element

# Hierarchical Sequence Processing
class HierarchicalSequenceNetwork(nn.Module):
    """
    Network for processing hierarchical sequence structures
    
    This network identifies patterns at multiple levels of abstraction,
    allowing for the recognition of nested sequence structures.
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: List[int] = [128, 256, 128],
                 output_dim: int = 64,
                 num_levels: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_levels = num_levels
        
        # Create LSTMs for each hierarchical level
        self.level_lstms = nn.ModuleList()
        
        # First level processes raw input
        self.level_lstms.append(
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dims[0],
                batch_first=True,
                dropout=dropout
            )
        )
        
        # Higher levels process information from lower levels
        for level in range(1, num_levels):
            idx = min(level, len(hidden_dims)-1)
            prev_idx = min(level-1, len(hidden_dims)-1)
            self.level_lstms.append(
                nn.LSTM(
                    input_size=hidden_dims[prev_idx],
                    hidden_size=hidden_dims[idx],
                    batch_first=True,
                    dropout=dropout
                )
            )
        
        # Output projection from highest level
        self.output = nn.Linear(hidden_dims[min(num_levels-1, len(hidden_dims)-1)], output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, return_all_levels: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Process input through all hierarchical levels
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            return_all_levels: Whether to return outputs from all levels
            
        Returns:
            Output tensor or list of output tensors from each level
        """
        x = x.to(self.device)
        
        # Process through each level
        level_outputs = []
        current_input = x
        
        for level, lstm in enumerate(self.level_lstms):
            # Process through LSTM
            output, _ = lstm(current_input)
            level_outputs.append(output)
            
            # Prepare input for next level (downsample by factor of 2)
            if level < self.num_levels - 1:
                seq_len = output.size(1)
                if seq_len >= 2:
                    # Downsample by taking every other element
                    current_input = output[:, ::2, :]
                else:
                    current_input = output
        
        # Final output projection
        final_output = self.output(self.dropout(level_outputs[-1]))
        
        if return_all_levels:
            return [final_output] + level_outputs
        
        return final_output

# Causality Networks
class CausalityNetwork(nn.Module):
    """
    Neural network for causal inference and modeling
    
    This network analyzes temporal data to identify causal relationships
    between variables and events.
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder for variables/events
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoder (for analyzing time series)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Causal strength estimator
        self.causal_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, cause_series: torch.Tensor, effect_series: torch.Tensor) -> torch.Tensor:
        """
        Evaluate causal relationship between two time series
        
        Args:
            cause_series: Potential cause time series [batch_size, sequence_length, input_dim]
            effect_series: Potential effect time series [batch_size, sequence_length, input_dim]
            
        Returns:
            Causal relationship representation [batch_size, output_dim]
        """
        cause_series = cause_series.to(self.device)
        effect_series = effect_series.to(self.device)
        
        # Encode individual time points
        encoded_cause = self.encoder(cause_series)
        encoded_effect = self.encoder(effect_series)
        
        # Process temporal dynamics
        _, cause_hidden = self.temporal_encoder(encoded_cause)
        _, effect_hidden = self.temporal_encoder(encoded_effect)
        
        # Combine cause and effect representations
        cause_hidden = cause_hidden[-1]  # Take last layer
        effect_hidden = effect_hidden[-1]  # Take last layer
        
        combined = torch.cat([cause_hidden, effect_hidden], dim=1)
        
        # Estimate causal relationship
        causal_output = self.causal_estimator(combined)
        
        return causal_output
    
    def evaluate_intervention(self, pre_intervention: torch.Tensor, post_intervention: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the effect of an intervention
        
        Args:
            pre_intervention: State before intervention [batch_size, sequence_length, input_dim]
            post_intervention: State after intervention [batch_size, sequence_length, input_dim]
            
        Returns:
            Intervention effect representation [batch_size, output_dim]
        """
        # Similar to forward, but specifically for intervention analysis
        return self.forward(pre_intervention, post_intervention)

# Prediction Networks
class PredictionNetwork(nn.Module):
    """
    Neural network for making predictions about future states
    
    This network generates predictions at various time horizons
    and estimates confidence in those predictions.
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_horizons: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_horizons = num_horizons
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Prediction heads for different time horizons
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) 
            for _ in range(num_horizons)
        ])
        
        # Confidence estimators for each prediction
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_horizons)
        ])
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate predictions at different time horizons
        
        Args:
            x: Input state sequence [batch_size, sequence_length, input_dim]
            
        Returns:
            Tuple of (predictions, confidences) at each time horizon
        """
        x = x.to(self.device)
        
        # Encode input states
        encoded = self.encoder(x)
        
        # Process temporal dynamics
        outputs, (hidden, _) = self.lstm(encoded)
        
        # Use the final output for predictions
        final_output = outputs[:, -1, :]
        
        # Generate predictions for each time horizon
        predictions = []
        confidences = []
        
        for pred_head, conf_head in zip(self.prediction_heads, self.confidence_heads):
            predictions.append(pred_head(final_output))
            confidences.append(conf_head(final_output))
        
        return predictions, confidences

# Time Perception Networks
class TimePerceptionNetwork(nn.Module):
    """
    Neural network for time perception and estimation
    
    This network tracks and estimates time intervals and
    detects temporal rhythms and patterns.
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Event encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal processing
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Duration estimator
        self.duration_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive duration estimate
        )
        
        # Rhythm detector
        self.rhythm_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize to device
        self.device = get_device()
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal information
        
        Args:
            x: Input event sequence [batch_size, sequence_length, input_dim]
            
        Returns:
            Tuple of (duration_estimate, rhythm_representation)
        """
        x = x.to(self.device)
        
        # Encode events
        encoded = self.encoder(x)
        
        # Process temporal dynamics
        outputs, hidden = self.gru(encoded)
        
        # Use final state for estimates
        final_state = hidden[-1]
        
        # Estimate duration
        duration = self.duration_estimator(final_state)
        
        # Detect rhythms
        rhythm = self.rhythm_detector(final_state)
        
        return duration, rhythm
    
    def estimate_interval(self, start_events: torch.Tensor, end_events: torch.Tensor) -> torch.Tensor:
        """
        Estimate the duration between two event sequences
        
        Args:
            start_events: Events at interval start [batch_size, sequence_length, input_dim]
            end_events: Events at interval end [batch_size, sequence_length, input_dim]
            
        Returns:
            Estimated duration between events [batch_size, 1]
        """
        start_events = start_events.to(self.device)
        end_events = end_events.to(self.device)
        
        # Encode events
        encoded_start = self.encoder(start_events)
        encoded_end = self.encoder(end_events)
        
        # Process through GRU
        _, start_hidden = self.gru(encoded_start)
        _, end_hidden = self.gru(encoded_end)
        
        # Take final layers
        start_hidden = start_hidden[-1]
        end_hidden = end_hidden[-1]
        
        # Compute difference representation
        diff = end_hidden - start_hidden
        
        # Estimate duration from difference
        duration = self.duration_estimator(diff)
        
        return duration
