import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

class AttentionNetwork(nn.Module):
    """
    Neural network architecture for attention processing
    
    This network handles the computational aspects of attention, including
    salience detection, focus control, and attention modulation.
    """
    def __init__(
        self, 
        input_dim: int = 128, 
        hidden_dim: int = 256, 
        output_dim: int = 64,
        num_heads: int = 4
    ):
        """
        Initialize attention neural network
        
        Parameters:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        """
        super().__init__()
        
        # Input embedding
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Salience detection network
        self.salience_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 components: novelty, emotion, goal, intensity
            nn.Sigmoid()  # Constrain outputs to 0-1 range
        )
        
        # Attention focus network
        self.focus_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # Input + salience components
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Activation level (0-1)
        )
        
        # Multi-head attention mechanism
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gating mechanism to control information flow
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        # State for working memory
        self.working_memory = None
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        current_focus: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention network
        
        Parameters:
        inputs: Input tensor [batch_size, num_items, input_dim]
        current_focus: Optional current focus state [batch_size, hidden_dim]
        context: Optional context information [batch_size, hidden_dim]
        
        Returns:
        Tuple of (output, attention_scores, salience_scores)
        """
        batch_size, num_items, _ = inputs.shape
        
        # Encode inputs
        encoded = self.input_encoder(inputs)  # [batch_size, num_items, hidden_dim]
        
        # Calculate salience for each item
        salience_scores = self.salience_network(encoded)  # [batch_size, num_items, 4]
        
        # Overall salience is average of components
        overall_salience = salience_scores.mean(dim=2, keepdim=True)  # [batch_size, num_items, 1]
        
        # Concatenate encoded inputs with salience components
        enhanced_inputs = torch.cat(
            [encoded, salience_scores], 
            dim=2
        )  # [batch_size, num_items, hidden_dim + 4]
        
        # Calculate attention activation for each item
        attention_scores = self.focus_network(enhanced_inputs)  # [batch_size, num_items, 1]
        attention_scores = attention_scores.squeeze(2)  # [batch_size, num_items]
        
        # Create attention mask (0 for items below threshold, 1 for above)
        attention_threshold = 0.3
        attention_mask = (attention_scores > attention_threshold).float()
        
        # Apply mask to encoded inputs
        masked_encoded = encoded * attention_mask.unsqueeze(2)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attention(
            masked_encoded, masked_encoded, masked_encoded
        )
        
        # Apply residual connection
        attn_output = attn_output + masked_encoded
        
        # Calculate attention gating
        gate_values = self.attention_gate(attn_output)
        
        # Apply gate to control information flow
        gated_output = attn_output * gate_values
        
        # Generate final output
        output = self.decoder(gated_output)
        
        # Update working memory with current state
        self.working_memory = attn_output.detach()
        
        return output, attention_scores, salience_scores
    
    def detect_salience(
        self, 
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Only run the salience detection part of the network
        
        Parameters:
        inputs: Input tensor [batch_size, num_items, input_dim]
        context: Optional context information [batch_size, hidden_dim]
        
        Returns:
        Tuple of (overall_salience, salience_components)
        """
        # Encode inputs
        encoded = self.input_encoder(inputs)
        
        # Calculate salience components
        salience_components = self.salience_network(encoded)
        
        # Extract components
        novelty = salience_components[:, :, 0]
        emotional = salience_components[:, :, 1]
        goal = salience_components[:, :, 2]
        intensity = salience_components[:, :, 3]
        
        # Calculate overall salience
        overall_salience = salience_components.mean(dim=2)
        
        components_dict = {
            "novelty": novelty,
            "emotional_significance": emotional,
            "goal_relevance": goal,
            "intensity": intensity
        }
        
        return overall_salience, components_dict
    
    def update_focus(
        self,
        encoded_items: torch.Tensor,
        salience_scores: torch.Tensor,
        current_focus: Optional[torch.Tensor] = None,
        capacity: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update attention focus based on salience
        
        Parameters:
        encoded_items: Encoded input items [batch_size, num_items, hidden_dim]
        salience_scores: Salience scores for items [batch_size, num_items, 4]
        current_focus: Optional current focus state [batch_size, hidden_dim]
        capacity: Attention capacity
        
        Returns:
        Tuple of (new_focus_state, attention_scores)
        """
        batch_size, num_items, _ = encoded_items.shape
        
        # Concatenate encoded items with salience components
        enhanced_inputs = torch.cat(
            [encoded_items, salience_scores], 
            dim=2
        )
        
        # Calculate attention activation for each item
        attention_scores = self.focus_network(enhanced_inputs)
        attention_scores = attention_scores.squeeze(2)  # [batch_size, num_items]
        
        # Apply capacity constraint - keep only top-k items
        if num_items > capacity:
            # Find threshold value that keeps capacity items
            sorted_scores, _ = torch.sort(attention_scores, dim=1, descending=True)
            threshold_values = sorted_scores[:, capacity-1].unsqueeze(1)
            
            # Create mask for items above threshold
            capacity_mask = (attention_scores >= threshold_values).float()
            
            # Apply mask to attention scores
            masked_attention = attention_scores * capacity_mask
        else:
            masked_attention = attention_scores
        
        # Calculate new focus state
        new_focus_state = torch.sum(
            encoded_items * masked_attention.unsqueeze(2),
            dim=1
        )
        
        return new_focus_state, masked_attention
    
    def process_with_attention(
        self,
        inputs: torch.Tensor,
        attention_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process inputs modulated by attention
        
        Parameters:
        inputs: Input tensor [batch_size, num_items, input_dim]
        attention_scores: Attention scores [batch_size, num_items]
        
        Returns:
        Processed output
        """
        # Apply attention modulation
        modulated_inputs = inputs * attention_scores.unsqueeze(2)
        
        # Encode modulated inputs
        encoded = self.input_encoder(modulated_inputs)
        
        # Apply multi-head attention mechanism
        attn_output, _ = self.multihead_attention(
            encoded, encoded, encoded
        )
        
        # Apply residual connection
        attn_output = attn_output + encoded
        
        # Generate final output
        output = self.decoder(attn_output)
        
        return output
    
    def reset_working_memory(self) -> None:
        """Reset working memory state"""
        self.working_memory = None