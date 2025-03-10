import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import numpy as np

class ConceptEncoder(nn.Module):
    """Neural encoder for concept representation"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into a concept representation"""
        return self.encoder(x)

class ConceptCombiner(nn.Module):
    """Neural network for combining concepts"""
    def __init__(self, concept_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        # Encode individual concepts
        self.concept_encoder = ConceptEncoder(concept_dim, hidden_dim, concept_dim)
        
        # Combination networks for different patterns
        self.blend_network = nn.Sequential(
            nn.Linear(concept_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.property_transfer_network = nn.Sequential(
            nn.Linear(concept_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.analogy_network = nn.Sequential(
            nn.Linear(concept_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for feature selection
        self.attention = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                concepts: List[torch.Tensor], 
                combination_type: str = "blend") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Combine concepts using specified combination type
        
        Args:
            concepts: List of concept tensors to combine
            combination_type: Type of combination ("blend", "property_transfer", "analogy")
            
        Returns:
            Tuple of (combined concept tensor, attention weights)
        """
        # Ensure we have enough concepts
        if len(concepts) < 2:
            if len(concepts) == 1:
                return concepts[0], {"attention": None}
            else:
                # Return zero tensor if no concepts
                device = next(self.parameters()).device
                return torch.zeros(1, self.blend_network[-1].out_features, device=device), {"attention": None}
        
        # Encode concepts
        encoded_concepts = [self.concept_encoder(c) for c in concepts]
        
        # Apply different combination patterns
        if combination_type == "blend" and len(concepts) >= 2:
            # Conceptual blending of two concepts
            c1, c2 = encoded_concepts[0], encoded_concepts[1]
            
            # Calculate attention weights for each concept
            attn1 = self.attention(c1)
            attn2 = self.attention(c2)
            
            # Apply attention weighting
            c1_weighted = c1 * attn1
            c2_weighted = c2 * attn2
            
            # Concatenate and process through blend network
            combined = torch.cat([c1_weighted, c2_weighted], dim=-1)
            result = self.blend_network(combined)
            
            return result, {"attention": (attn1, attn2)}
            
        elif combination_type == "property_transfer" and len(concepts) >= 2:
            # Transfer properties from one concept to another
            c1, c2 = encoded_concepts[0], encoded_concepts[1]
            
            # Calculate attention weights for feature selection
            attn = self.attention(c2)  # Features to transfer
            
            # Weighted combination
            combined = torch.cat([c1, c2 * attn], dim=-1)
            result = self.property_transfer_network(combined)
            
            return result, {"attention": attn}
            
        elif combination_type == "analogy" and len(concepts) >= 3:
            # Analogical mapping: A is to B as C is to ?
            c1, c2, c3 = encoded_concepts[0], encoded_concepts[1], encoded_concepts[2]
            
            # Calculate relationship between A and B
            relation = c2 - c1
            
            # Apply relationship to C
            combined = torch.cat([c1, c2, c3], dim=-1)
            result = self.analogy_network(combined)
            
            return result, {"relation": relation}
            
        else:
            # Default to simple blending
            c1, c2 = encoded_concepts[0], encoded_concepts[1]
            combined = torch.cat([c1, c2], dim=-1)
            result = self.blend_network(combined)
            
            return result, {"attention": None}

class DivergentGenerator(nn.Module):
    """Neural network for divergent thinking and idea generation"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # For mean and log variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Additional heads for generating diverse solutions
        self.solution_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(4)  # Multiple solution paths
        ])
        
        self.latent_dim = latent_dim
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters (mean, log_var)"""
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var
        
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent space using reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
        
    def generate_diverse_solutions(self, z: torch.Tensor, 
                                  diversity_factor: float = 1.0) -> List[torch.Tensor]:
        """Generate multiple diverse solutions from the latent representation"""
        solutions = []
        
        # Base solution
        solutions.append(self.decode(z))
        
        # Generate diverse variations
        for head in self.solution_heads:
            # Add noise to latent vector proportional to diversity factor
            noise = torch.randn_like(z) * diversity_factor
            z_diverse = z + noise
            solutions.append(head(z_diverse))
            
        return solutions
        
    def forward(self, x: torch.Tensor, 
               diversity_factor: float = 1.0, 
               num_solutions: int = 5) -> Dict[str, Any]:
        """
        Process input for divergent thinking
        
        Args:
            x: Input tensor representing problem or prompt
            diversity_factor: Controls how diverse the solutions should be
            num_solutions: Number of solutions to generate
            
        Returns:
            Dictionary with generated solutions and latent representations
        """
        # Encode to latent space
        mean, log_var = self.encode(x)
        
        # Sample from latent space
        z = self.reparameterize(mean, log_var)
        
        # Generate diverse solutions
        solutions = self.generate_diverse_solutions(z, diversity_factor)
        
        # Limit solutions to requested number
        solutions = solutions[:num_solutions]
        
        return {
            "solutions": solutions,
            "latent": z,
            "mean": mean,
            "log_var": log_var
        }

class ImaginationNetwork(nn.Module):
    """Neural network for imagination and scenario generation"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128, 
                 sequence_length: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Recurrent network for temporal coherence
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Scene element generators
        self.object_generator = nn.Linear(hidden_dim, output_dim)
        self.agent_generator = nn.Linear(hidden_dim, output_dim)
        self.action_generator = nn.Linear(hidden_dim, output_dim)
        self.setting_generator = nn.Linear(hidden_dim, output_dim)
        
        # Integration layer
        self.scene_integrator = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.sequence_length = sequence_length
        
    def forward(self, x: torch.Tensor, 
               sequence_length: Optional[int] = None,
               initial_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate imaginative scene or sequence
        
        Args:
            x: Input tensor (seed or prompt)
            sequence_length: Length of sequence to generate
            initial_state: Initial hidden state for RNN
            
        Returns:
            Dictionary with scene elements and integrated scene
        """
        batch_size = x.shape[0]
        seq_len = sequence_length or self.sequence_length
        
        # Encode input
        encoded = self.encoder(x).unsqueeze(1)  # Add sequence dimension
        
        # Expand to desired sequence length
        encoded = encoded.expand(-1, seq_len, -1)
        
        # Generate sequence with temporal coherence
        if initial_state is None:
            hidden_states, final_state = self.rnn(encoded)
        else:
            hidden_states, final_state = self.rnn(encoded, initial_state)
        
        # Generate scene elements for each step in sequence
        objects = self.object_generator(hidden_states)
        agents = self.agent_generator(hidden_states)
        actions = self.action_generator(hidden_states)
        settings = self.setting_generator(hidden_states)
        
        # Integrate scene elements
        scene_elements = torch.cat([objects, agents, actions, settings], dim=-1)
        integrated_scene = self.scene_integrator(scene_elements)
        
        return {
            "objects": objects,
            "agents": agents,
            "actions": actions,
            "settings": settings,
            "integrated_scene": integrated_scene,
            "hidden_states": hidden_states,
            "final_state": final_state
        }

class NoveltyDetector(nn.Module):
    """Neural network for detecting novelty in inputs"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, memory_size: int = 100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Memory module to store recent experiences
        self.register_buffer("memory", torch.zeros(memory_size, input_dim))
        self.memory_counter = 0
        self.memory_size = memory_size
        
        # Novelty scoring network
        self.novelty_scorer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def update_memory(self, encoded: torch.Tensor) -> None:
        """Update memory with new encoded inputs"""
        batch_size = encoded.shape[0]
        
        for i in range(batch_size):
            idx = self.memory_counter % self.memory_size
            self.memory[idx] = encoded[i].detach()
            self.memory_counter += 1
    
    def compute_novelty(self, encoded: torch.Tensor) -> torch.Tensor:
        """Compute novelty by comparing to memory"""
        # If memory is empty, everything is novel
        if self.memory_counter == 0:
            return torch.ones(encoded.shape[0], 1, device=encoded.device)
        
        # Compute minimum distance to memory items
        memory_size = min(self.memory_size, self.memory_counter)
        memory = self.memory[:memory_size]
        
        # Compute distance to all memory items
        distances = torch.cdist(encoded, memory)
        
        # Take minimum distance for each input
        min_distances, _ = torch.min(distances, dim=1, keepdim=True)
        
        # Normalize distances to [0, 1] range
        normalized_distances = torch.tanh(min_distances)
        
        return normalized_distances
        
    def forward(self, x: torch.Tensor, update_memory: bool = True) -> Dict[str, torch.Tensor]:
        """
        Detect novelty in input
        
        Args:
            x: Input tensor to evaluate for novelty
            update_memory: Whether to update memory with this input
            
        Returns:
            Dictionary with novelty scores and encoded representation
        """
        # Encode input
        encoded = self.encoder(x)
        
        # Compute novelty score
        novelty_score = self.compute_novelty(encoded)
        
        # Update memory if requested
        if update_memory:
            self.update_memory(encoded)
            
        return {
            "novelty_score": novelty_score,
            "encoded": encoded
        }

class CreativityNetwork(nn.Module):
    """Integrated neural network for creative processing"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Creativity components
        self.concept_combiner = ConceptCombiner(hidden_dim, hidden_dim, output_dim)
        self.divergent_generator = DivergentGenerator(hidden_dim, hidden_dim, output_dim)
        self.imagination = ImaginationNetwork(hidden_dim, hidden_dim, output_dim)
        self.novelty_detector = NoveltyDetector(hidden_dim, hidden_dim)
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Developmental gate
        self.development_gate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Process inputs through the creativity network
        
        Args:
            inputs: Dictionary of input tensors for different creativity processes
            
        Returns:
            Dictionary of creativity processing results
        """
        # Get device
        device = next(self.parameters()).device
        
        # Process concept combination if inputs available
        concept_result = None
        if "concepts" in inputs and inputs["concepts"] is not None:
            concepts = [self.input_embedding(c) for c in inputs["concepts"]]
            combination_type = inputs.get("combination_type", "blend")
            concept_result, attention = self.concept_combiner(concepts, combination_type)
        
        # Process divergent thinking if inputs available
        divergent_result = None
        if "problem" in inputs and inputs["problem"] is not None:
            embedded_problem = self.input_embedding(inputs["problem"])
            diversity_factor = inputs.get("diversity_factor", 1.0)
            num_solutions = inputs.get("num_solutions", 5)
            divergent_result = self.divergent_generator(
                embedded_problem, 
                diversity_factor, 
                num_solutions
            )
        
        # Process imagination if inputs available
        imagination_result = None
        if "seed" in inputs and inputs["seed"] is not None:
            embedded_seed = self.input_embedding(inputs["seed"])
            sequence_length = inputs.get("sequence_length", 10)
            initial_state = inputs.get("initial_state", None)
            imagination_result = self.imagination(
                embedded_seed,
                sequence_length,
                initial_state
            )
        
        # Process novelty detection if inputs available
        novelty_result = None
        if "input" in inputs and inputs["input"] is not None:
            embedded_input = self.input_embedding(inputs["input"])
            update_memory = inputs.get("update_memory", True)
            novelty_result = self.novelty_detector(embedded_input, update_memory)
        
        # Apply developmental gating
        dev_level = torch.sigmoid(self.development_gate)
        
        # Create integrated result
        result = {
            "concept_combination": concept_result,
            "divergent_thinking": divergent_result,
            "imagination": imagination_result,
            "novelty_detection": novelty_result,
            "developmental_level": dev_level.item()
        }
        
        # Integrate results if all components have produced output
        if all(v is not None for v in [concept_result, divergent_result, imagination_result, novelty_result]):
            # Extract main outputs from each component
            concept_out = concept_result
            divergent_out = divergent_result["solutions"][0]  # Take first solution
            imagination_out = imagination_result["integrated_scene"][:, -1]  # Take last frame
            novelty_out = novelty_result["encoded"]
            
            # Concatenate and integrate
            combined = torch.cat([concept_out, divergent_out, imagination_out, novelty_out], dim=-1)
            integrated = self.integration(combined)
            
            result["integrated_output"] = integrated
        
        return result
    
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
