import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class ConsciousnessAttention(nn.Module):
    """
    Neural attention mechanism for consciousness processing.
    Enables the system to focus on relevant information.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to input
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (attended output, attention weights)
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attention, v)
        return out, attention

class GlobalWorkspaceNetwork(nn.Module):
    """
    Neural network implementation of Global Workspace Theory.
    Represents competition and integration of information.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.competition = nn.Linear(hidden_dim, hidden_dim)
        self.integration = nn.Linear(hidden_dim, output_dim)
        self.attention = ConsciousnessAttention(hidden_dim)
        
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process inputs through the global workspace
        
        Args:
            inputs: List of tensor inputs from different sources
            
        Returns:
            Tuple of (integrated representation, attention weights)
        """
        if not inputs:
            # Return zero tensor if no inputs
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.integration.out_features, device=device), None
            
        # Concatenate inputs if multiple
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[0]
            
        # Project to hidden space
        hidden = F.relu(self.input_projection(x))
        
        # Competition phase
        competition = F.relu(self.competition(hidden))
        
        # Apply attention for selection
        attended, weights = self.attention(competition.unsqueeze(0))
        
        # Integration phase
        output = F.relu(self.integration(attended.squeeze(0)))
        
        return output, weights

class SelfModelNetwork(nn.Module):
    """
    Neural network for self-modeling and identity representation
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.identity_projection = nn.Linear(hidden_dim, output_dim)
        self.capability_estimation = nn.Linear(hidden_dim, output_dim)
        self.goal_representation = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through self-model network
        
        Args:
            x: Input tensor representing current state
            
        Returns:
            Dictionary of self-model outputs (identity, capabilities, goals)
        """
        encoded = self.encoder(x)
        
        return {
            "identity": self.identity_projection(encoded),
            "capabilities": torch.sigmoid(self.capability_estimation(encoded)),
            "goals": self.goal_representation(encoded)
        }

class IntrospectionNetwork(nn.Module):
    """
    Neural network for introspective processing and metacognition
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Linear(input_dim, hidden_dim)
        self.process_analyzer = nn.Linear(hidden_dim, hidden_dim)
        self.metacognitive_output = nn.Linear(hidden_dim, output_dim)
        self.confidence_estimator = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through introspection network
        
        Args:
            x: Input tensor representing system state
            
        Returns:
            Dictionary of introspection outputs
        """
        encoded = F.relu(self.state_encoder(x))
        analyzed = F.relu(self.process_analyzer(encoded))
        
        return {
            "metacognition": self.metacognitive_output(analyzed),
            "confidence": torch.sigmoid(self.confidence_estimator(analyzed))
        }

class ConsciousnessNetwork(nn.Module):
    """
    Integrated neural network for consciousness processing.
    Combines global workspace, self-model, and introspection.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.global_workspace = GlobalWorkspaceNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.self_model = SelfModelNetwork(hidden_dim, hidden_dim, output_dim)
        self.introspection = IntrospectionNetwork(hidden_dim * 2, hidden_dim, output_dim)
        self.development_gate = nn.Parameter(torch.tensor(0.1))  # Learnable developmental parameter
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Process inputs through the consciousness network
        
        Args:
            inputs: Dictionary of input tensors from different sources
            
        Returns:
            Dictionary of consciousness processing results
        """
        # Get device
        device = next(self.parameters()).device
        
        # Convert inputs to embeddings
        embedded_inputs = []
        for source, tensor in inputs.items():
            if tensor is not None:
                embedded = F.relu(self.input_embedding(tensor))
                embedded_inputs.append(embedded)
        
        # Global workspace processing
        if embedded_inputs:
            workspace_output, attention_weights = self.global_workspace(embedded_inputs)
        else:
            workspace_output = torch.zeros(1, hidden_dim, device=device)
            attention_weights = None
            
        # Self-model processing
        self_model_output = self.self_model(workspace_output)
        
        # Introspection processing (takes both workspace and self-model as input)
        combined = torch.cat([workspace_output, self_model_output["identity"]], dim=-1)
        introspection_output = self.introspection(combined)
        
        # Apply developmental gating to introspection
        dev_level = torch.sigmoid(self.development_gate)
        gated_introspection = {k: v * dev_level for k, v in introspection_output.items()}
        
        return {
            "global_workspace": workspace_output,
            "attention": attention_weights,
            "self_model": self_model_output,
            "introspection": gated_introspection,
            "developmental_level": dev_level.item()
        }
        
    def update_development(self, amount: float) -> float:
        """
        Update the developmental parameter
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        with torch.no_grad():
            current = torch.sigmoid(self.development_gate).item()
            target = min(1.0, current + amount)
            # Convert from probability space back to unbounded space
            if target >= 0.99:
                self.development_gate.data = torch.tensor(6.0)  # Approximately sigmoid(6) ≈ 0.998
            else:
                self.development_gate.data = torch.tensor(np.log(target / (1 - target)))
            return torch.sigmoid(self.development_gate).item()
