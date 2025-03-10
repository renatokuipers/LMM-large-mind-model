import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

class PlanningNetwork(nn.Module):
    """
    Neural network for plan generation and execution monitoring
    
    This network processes goals and states to generate plan steps and
    monitor execution progress.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the planning network
        
        Args:
            input_dim: Dimension of input features (goal + state)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
        """
        super().__init__()
        
        # Goal and state encoders
        self.goal_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
        self.state_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
        
        # Combined processing layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Plan generation head
        self.plan_generator = nn.Linear(hidden_dim, output_dim)
        
        # Execution monitoring head
        self.execution_monitor = nn.Linear(hidden_dim, 3)  # [progress, success_prob, revision_needed]
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
    def forward(self, goal: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process goal and state to generate plan information
        
        Args:
            goal: Tensor representing the goal [batch_size, input_dim//2]
            state: Tensor representing current state [batch_size, input_dim//2]
            
        Returns:
            Dictionary with plan features and monitoring outputs
        """
        # Encode goal and state
        goal_features = F.relu(self.goal_encoder(goal))
        state_features = F.relu(self.state_encoder(state))
        
        # Combine features
        combined = torch.cat([goal_features, state_features], dim=1)
        
        # Process through hidden layers with residual connection
        hidden_output = self.hidden_layers(combined)
        combined = combined + self.layer_norm(hidden_output)  # Residual connection
        
        # Generate plan features
        plan_features = self.plan_generator(combined)
        
        # Monitor execution
        monitor_outputs = self.execution_monitor(combined)
        progress = torch.sigmoid(monitor_outputs[:, 0])
        success_prob = torch.sigmoid(monitor_outputs[:, 1])
        revision_needed = torch.sigmoid(monitor_outputs[:, 2])
        
        # Developmental modulation
        dev_factor = self.developmental_factor.item()
        if dev_factor < 0.3:
            # Very basic planning at early development
            # Add randomness and limit planning horizon
            plan_features = 0.7 * plan_features + 0.3 * torch.randn_like(plan_features) * 0.1
            # Simple binary outcomes at early stages
            success_prob = torch.round(success_prob * 2) / 2
        
        return {
            "plan_features": plan_features,
            "progress": progress,
            "success_probability": success_prob,
            "revision_needed": revision_needed
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the planning network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class DecisionNetwork(nn.Module):
    """
    Neural network for decision making
    
    This network evaluates options and selects the best one based on
    multiple criteria and contextual factors.
    """
    
    def __init__(self, option_dim: int = 64, criteria_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize the decision network
        
        Args:
            option_dim: Dimension of option features
            criteria_dim: Dimension of criteria features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Option encoder
        self.option_encoder = nn.Linear(option_dim, hidden_dim)
        
        # Criteria encoder
        self.criteria_encoder = nn.Linear(criteria_dim, hidden_dim // 2)
        
        # Context encoder
        self.context_encoder = nn.Linear(option_dim, hidden_dim // 2)
        
        # Evaluation layers
        self.evaluation_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Option scoring head
        self.option_scorer = nn.Linear(hidden_dim, 1)
        
        # Confidence estimation head
        self.confidence_estimator = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
    def forward(self, 
                options: torch.Tensor, 
                criteria: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate options based on criteria and context
        
        Args:
            options: Tensor of option features [batch_size, num_options, option_dim]
            criteria: Tensor of criteria features [batch_size, criteria_dim]
            context: Optional tensor of context features [batch_size, option_dim]
            
        Returns:
            Dictionary with option scores and confidence
        """
        batch_size, num_options, _ = options.shape
        
        # Encode criteria
        criteria_encoded = F.relu(self.criteria_encoder(criteria))  # [batch_size, hidden_dim//2]
        
        # Encode context (or use zeros if not provided)
        if context is not None:
            context_encoded = F.relu(self.context_encoder(context))  # [batch_size, hidden_dim//2]
        else:
            context_encoded = torch.zeros(batch_size, self.hidden_dim // 2, device=options.device)
            
        # Combine criteria and context
        criteria_context = torch.cat([criteria_encoded, context_encoded], dim=1)  # [batch_size, hidden_dim]
        
        # Repeat for each option
        criteria_context = criteria_context.unsqueeze(1).expand(-1, num_options, -1)  # [batch_size, num_options, hidden_dim]
        
        # Process each option
        options_flat = options.view(-1, options.size(-1))  # [batch_size*num_options, option_dim]
        options_encoded = F.relu(self.option_encoder(options_flat))  # [batch_size*num_options, hidden_dim]
        options_encoded = options_encoded.view(batch_size, num_options, -1)  # [batch_size, num_options, hidden_dim]
        
        # Combine options with criteria and context
        combined = torch.cat([options_encoded, criteria_context], dim=2)  # [batch_size, num_options, hidden_dim*2]
        
        # Reshape for processing
        combined_flat = combined.view(-1, combined.size(-1))  # [batch_size*num_options, hidden_dim*2]
        
        # Evaluate options
        evaluated = self.evaluation_layers(combined_flat)  # [batch_size*num_options, hidden_dim]
        evaluated = evaluated.view(batch_size, num_options, -1)  # [batch_size, num_options, hidden_dim]
        
        # Score options
        scores_flat = self.option_scorer(evaluated.view(-1, evaluated.size(-1)))  # [batch_size*num_options, 1]
        scores = scores_flat.view(batch_size, num_options)  # [batch_size, num_options]
        
        # Apply developmental modulation
        dev_factor = self.developmental_factor.item()
        if dev_factor < 0.3:
            # At early development, decisions are more random and less nuanced
            scores = scores * 0.7 + torch.randn_like(scores) * 0.3
        
        # Convert to probabilities
        probabilities = F.softmax(scores, dim=1)
        
        # Estimate confidence (based on score distribution entropy)
        # High entropy = low confidence, Low entropy = high confidence
        log_probs = F.log_softmax(scores, dim=1)
        entropy = -torch.sum(probabilities * log_probs, dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(num_options, dtype=torch.float))
        confidence = 1 - entropy / max_entropy
        
        # Best option
        best_option_idx = torch.argmax(scores, dim=1)
        
        return {
            "scores": scores,
            "probabilities": probabilities,
            "confidence": confidence,
            "best_option_idx": best_option_idx
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the decision network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class InhibitionNetwork(nn.Module):
    """
    Neural network for inhibitory control
    
    This network processes stimuli and context to determine whether
    inhibitory control should be applied and with what strength.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        """
        Initialize the inhibition network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Stimulus encoder
        self.stimulus_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
        
        # Context encoder
        self.context_encoder = nn.Linear(input_dim // 2, hidden_dim // 2)
        
        # Processing layers
        self.processing_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Inhibition decision head
        self.inhibition_head = nn.Linear(hidden_dim, 1)  # Inhibit or not
        
        # Inhibition strength head
        self.strength_head = nn.Linear(hidden_dim, 1)  # How strongly to inhibit
        
        # Resource cost head
        self.cost_head = nn.Linear(hidden_dim, 1)  # Resource cost of inhibition
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
    def forward(self, stimulus: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Determine whether and how strongly to apply inhibitory control
        
        Args:
            stimulus: Tensor representing the stimulus to potentially inhibit
            context: Tensor representing contextual information
            
        Returns:
            Dictionary with inhibition decision and parameters
        """
        # Encode stimulus and context
        stimulus_encoded = F.relu(self.stimulus_encoder(stimulus))
        context_encoded = F.relu(self.context_encoder(context))
        
        # Combine features
        combined = torch.cat([stimulus_encoded, context_encoded], dim=1)
        
        # Process through hidden layers with residual connection
        processed = self.processing_layers(combined)
        combined = combined + self.layer_norm(processed)  # Residual connection
        
        # Inhibition decision
        inhibit_logit = self.inhibition_head(combined)
        inhibit_prob = torch.sigmoid(inhibit_logit)
        
        # Inhibition strength
        strength = torch.sigmoid(self.strength_head(combined))
        
        # Resource cost
        cost = F.softplus(self.cost_head(combined))  # Always positive
        
        # Apply developmental modulation
        dev_factor = self.developmental_factor.item()
        if dev_factor < 0.3:
            # Low development: weak/inconsistent inhibition
            inhibit_prob = inhibit_prob * 0.7 + torch.rand_like(inhibit_prob) * 0.3
            strength = strength * 0.5  # Weaker inhibition
            cost = cost * 1.5  # Higher cost
        
        return {
            "inhibit_probability": inhibit_prob,
            "inhibition_strength": strength,
            "resource_cost": cost
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the inhibition network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))


class WorkingMemoryNetwork(nn.Module):
    """
    Neural network for working memory control
    
    This network handles maintenance, updating, and manipulation
    of items in working memory.
    """
    
    def __init__(self, item_dim: int = 64, control_dim: int = 32, hidden_dim: int = 128, capacity: int = 7):
        """
        Initialize the working memory network
        
        Args:
            item_dim: Dimension of item features
            control_dim: Dimension of control signals
            hidden_dim: Dimension of hidden layers
            capacity: Maximum number of items in working memory
        """
        super().__init__()
        
        self.item_dim = item_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        
        # Item encoder
        self.item_encoder = nn.Linear(item_dim, hidden_dim)
        
        # Control signal encoder
        self.control_encoder = nn.Linear(control_dim, hidden_dim)
        
        # Memory slots (as parameters)
        self.memory_slots = nn.Parameter(torch.zeros(capacity, hidden_dim))
        
        # Attention mechanism for slot selection
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Gate for memory updating
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, item_dim)
        
        # Memory decay parameters
        self.decay_rate = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        
        # Developmental parameter
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
    def forward(self, 
                operation: str,
                items: Optional[torch.Tensor] = None,
                control: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform operations on working memory
        
        Args:
            operation: Type of operation ('store', 'retrieve', 'update', 'clear')
            items: Optional tensor of items to store [batch_size, item_dim]
            control: Optional tensor of control signals [batch_size, control_dim]
            
        Returns:
            Dictionary with operation results
        """
        batch_size = 1
        if items is not None:
            batch_size = items.size(0)
            
        # Get effective capacity based on development
        dev_factor = self.developmental_factor.item()
        effective_capacity = max(1, int(self.capacity * (0.3 + 0.7 * dev_factor)))
        
        # Initialize result dictionary
        result = {}
        
        if operation == 'store' and items is not None:
            # Encode items
            item_encoded = F.relu(self.item_encoder(items))  # [batch_size, hidden_dim]
            
            # Find least active slot or empty slot
            # In a real implementation, this would use actual memory activations
            # Here we'll use a simple heuristic for demonstration
            slot_idx = torch.randint(0, effective_capacity, (batch_size,))
            
            # Store items in selected slots
            # In practice, this would update a persistent memory state
            memory_update = F.sigmoid(self.update_gate(
                torch.cat([item_encoded, self.memory_slots[slot_idx]], dim=1)
            ))
            
            # Apply developmental modulation
            if dev_factor < 0.4:
                # Lower development: more forgetting, less precise storage
                memory_update = memory_update * 0.8 + torch.randn_like(memory_update) * 0.2
            
            result['stored_indices'] = slot_idx
            result['store_success'] = torch.ones(batch_size)
            
        elif operation == 'retrieve' and control is not None:
            # Encode control signal
            control_encoded = F.relu(self.control_encoder(control))  # [batch_size, hidden_dim]
            
            # Calculate attention over memory slots
            attention_inputs = []
            for i in range(effective_capacity):
                slot_expand = self.memory_slots[i:i+1].expand(batch_size, -1)
                combined = torch.cat([control_encoded, slot_expand], dim=1)
                attention_inputs.append(self.attention(combined))
            
            attention_logits = torch.cat(attention_inputs, dim=1)  # [batch_size, capacity]
            attention_weights = F.softmax(attention_logits, dim=1)
            
            # Retrieve weighted combination of memory slots
            retrieved_items = torch.zeros(batch_size, self.hidden_dim, device=control.device)
            for i in range(effective_capacity):
                slot_expand = self.memory_slots[i:i+1].expand(batch_size, -1)
                retrieved_items += attention_weights[:, i:i+1] * slot_expand
            
            # Project back to item space
            retrieved_items = self.output_projection(retrieved_items)
            
            # Apply developmental modulation
            if dev_factor < 0.4:
                # Lower development: noisier retrieval
                retrieved_items = retrieved_items * 0.8 + torch.randn_like(retrieved_items) * 0.2
            
            result['retrieved_items'] = retrieved_items
            result['attention_weights'] = attention_weights
            
        elif operation == 'update' and items is not None and control is not None:
            # For simplicity, this is similar to store but with target slot
            # Encode items and control
            item_encoded = F.relu(self.item_encoder(items))
            control_encoded = F.relu(self.control_encoder(control))
            
            # Calculate attention to find target slot
            attention_inputs = []
            for i in range(effective_capacity):
                slot_expand = self.memory_slots[i:i+1].expand(batch_size, -1)
                combined = torch.cat([control_encoded, slot_expand], dim=1)
                attention_inputs.append(self.attention(combined))
            
            attention_logits = torch.cat(attention_inputs, dim=1)
            attention_weights = F.softmax(attention_logits, dim=1)
            slot_idx = torch.argmax(attention_weights, dim=1)
            
            # Update selected slots
            memory_update = F.sigmoid(self.update_gate(
                torch.cat([item_encoded, self.memory_slots[slot_idx]], dim=1)
            ))
            
            result['updated_indices'] = slot_idx
            result['update_success'] = torch.ones(batch_size)
            
        elif operation == 'clear':
            # Reset all memory slots
            # In practice, this would update a persistent memory state
            # with torch.no_grad():
            #     self.memory_slots.zero_()
            
            result['clear_success'] = torch.ones(1)
        
        return result
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the working memory network
        
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
