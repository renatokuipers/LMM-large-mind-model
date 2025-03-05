"""
Cognitive component for the Neural Child's mind.

This module contains the implementation of the cognitive component that handles
cognitive functions like attention, problem-solving, and abstract thinking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import random
import time
import json
from pathlib import Path
import requests

from neural_child.mind.base import NeuralComponent

class CognitiveComponent(NeuralComponent):
    """Cognitive component that handles cognitive processes like attention, perception, and reasoning."""
    
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        output_size: int = 128,
        learning_rate: float = 0.01,
        device: str = "cpu",
        name: str = "cognitive_component",
        embedding_api_url: str = "http://192.168.2.12:1234/v1/embeddings",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ):
        """Initialize the cognitive component.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layer
            output_size: Dimension of output features
            learning_rate: Learning rate for training
            device: Device to run the model on (cpu or cuda)
            name: Name of the component
            embedding_api_url: URL for the embedding API
            embedding_model: Model to use for embeddings
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size, name=name)
        
        self.learning_rate = learning_rate
        self.device = device
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        
        # Neural network for cognitive processing
        self.cognitive_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.cognitive_network.parameters(), 
            lr=learning_rate
        )
        
        # Cognitive development metrics
        self.cognitive_development = {
            "attention": 0.1,  # Ability to focus and sustain attention
            "memory": 0.1,     # Working memory capacity
            "problem_solving": 0.0,  # Ability to solve problems
            "abstract_thinking": 0.0  # Ability to think abstractly
        }
        
        # Attention parameters
        self.attention_span_seconds = 5.0  # Initial attention span in seconds
        self.attention_decay_rate = 0.1   # Rate at which attention decays
        self.current_attention_level = 1.0  # Current attention level (0.0 to 1.0)
        self.last_attention_update = time.time()
        
        # Problem-solving parameters
        self.problem_complexity_threshold = 0.1  # Complexity of problems that can be solved
        self.tool_use_capability = 0.0  # Ability to use tools to solve problems
        self.causal_reasoning_ability = 0.0  # Ability to understand cause and effect
        
        # Abstract thinking parameters
        self.symbol_understanding = 0.0  # Ability to understand symbols
        self.metaphor_understanding = 0.0  # Ability to understand metaphors
        self.hypothetical_reasoning = 0.0  # Ability to reason about hypothetical scenarios
        
        # Cognitive biases (initially neutral)
        self.cognitive_biases = {
            "confirmation_bias": 0.0,  # Tendency to favor information that confirms existing beliefs
            "recency_bias": 0.0,       # Tendency to weigh recent events more heavily
            "availability_bias": 0.0,  # Tendency to overestimate likelihood of events based on availability
            "anchoring_bias": 0.0      # Tendency to rely too heavily on first piece of information
        }
        
        # Cognitive load (0.0 to 1.0)
        self.cognitive_load = 0.0
        
        # Curiosity level (0.0 to 1.0)
        self.curiosity_level = 0.5
        
        # Learning history
        self.learning_history = []
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the cognitive component.
        
        Args:
            input_data: Dictionary containing input data
                - mother_utterance: Text from mother
                - emotional_state: Current emotional state
                - context: Context of the interaction
                - complexity: Complexity of the input (0.0 to 1.0)
                
        Returns:
            Dictionary containing processed output
        """
        # Update attention based on elapsed time
        self._update_attention()
        
        # Extract relevant information from input
        mother_utterance = input_data.get("mother_utterance", "")
        emotional_state = input_data.get("emotional_state", {})
        context = input_data.get("context", {})
        complexity = input_data.get("complexity", 0.5)
        
        # Check if complexity exceeds current capabilities
        if complexity > self.problem_complexity_threshold:
            # Increase cognitive load
            self.cognitive_load = min(1.0, self.cognitive_load + 0.2)
            
            # Decrease attention if cognitive load is high
            if self.cognitive_load > 0.7:
                self.current_attention_level = max(0.1, self.current_attention_level - 0.3)
        
        # Apply attention filter (simulating attention mechanism)
        attention_factor = self.current_attention_level
        
        # If attention is low, only process part of the input
        if attention_factor < 0.5 and len(mother_utterance) > 10:
            # Simulate partial attention by truncating input
            effective_length = int(len(mother_utterance) * attention_factor)
            mother_utterance = mother_utterance[:effective_length] + "..."
        
        # Convert input to tensor for neural processing
        # This is a simplified representation - in a real system, you'd use proper NLP embeddings
        input_embedding = self._create_input_embedding(
            mother_utterance, emotional_state, context
        )
        
        # Process through neural network
        with torch.no_grad():
            output_embedding = self.cognitive_network(input_embedding)
        
        # Interpret the output embedding
        understanding_level = self._calculate_understanding_level(
            output_embedding, complexity
        )
        
        # Generate cognitive response
        cognitive_response = self._generate_cognitive_response(
            understanding_level, mother_utterance, context
        )
        
        # Update curiosity based on novelty
        novelty = self._calculate_novelty(input_data)
        self.curiosity_level = min(1.0, self.curiosity_level + novelty * 0.2)
        
        # Prepare output
        output = {
            "understanding_level": understanding_level,
            "cognitive_response": cognitive_response,
            "attention_level": self.current_attention_level,
            "cognitive_load": self.cognitive_load,
            "curiosity_level": self.curiosity_level,
            "cognitive_development": self.cognitive_development
        }
        
        # Record learning if understanding was achieved
        if understanding_level > 0.3:
            self._record_learning(input_data, understanding_level)
        
        return output
    
    def _update_attention(self):
        """Update attention level based on elapsed time."""
        current_time = time.time()
        elapsed_seconds = current_time - self.last_attention_update
        self.last_attention_update = current_time
        
        # Natural decay of attention over time
        decay = self.attention_decay_rate * elapsed_seconds / self.attention_span_seconds
        self.current_attention_level = max(0.1, self.current_attention_level - decay)
        
        # Gradually recover attention
        recovery_rate = 0.05 * elapsed_seconds
        self.current_attention_level = min(1.0, self.current_attention_level + recovery_rate)
        
        # Cognitive load reduces attention recovery
        if self.cognitive_load > 0.5:
            self.current_attention_level = max(0.1, self.current_attention_level - 0.1)
        
        # Gradually reduce cognitive load over time
        self.cognitive_load = max(0.0, self.cognitive_load - 0.05 * elapsed_seconds)
    
    def _create_input_embedding(
        self, 
        utterance: str, 
        emotional_state: Dict[str, float], 
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Create an embedding vector from the input data using the embedding API.
        
        Args:
            utterance: Text utterance
            emotional_state: Emotional state dictionary
            context: Context dictionary
            
        Returns:
            Tensor embedding of the input
        """
        # Create a context-enriched input for better embeddings
        emotional_context = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_state.items()])
        
        # Combine utterance with emotional context
        if emotional_context:
            enriched_input = f"{utterance} [Emotional state: {emotional_context}]"
        else:
            enriched_input = utterance
            
        # Add any relevant context information
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if isinstance(v, (str, int, float))])
            if context_str:
                enriched_input += f" [Context: {context_str}]"
        
        try:
            # Call the embedding API
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.embedding_model,
                "input": enriched_input
            }
            
            response = requests.post(self.embedding_api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Extract the embedding from the response
            embedding_data = response.json()
            embedding = embedding_data["data"][0]["embedding"]
            
            # Convert to numpy array and ensure correct dimension
            embedding_np = np.array(embedding, dtype=np.float32)
            
            # If the embedding dimension doesn't match input_size, resize it
            if len(embedding_np) != self.input_size:
                if len(embedding_np) > self.input_size:
                    # Truncate if too large
                    embedding_np = embedding_np[:self.input_size]
                else:
                    # Pad with zeros if too small
                    padding = np.zeros(self.input_size - len(embedding_np), dtype=np.float32)
                    embedding_np = np.concatenate([embedding_np, padding])
            
            return torch.tensor(embedding_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            
        except Exception as e:
            # If API call fails, use a random embedding
            print(f"Error creating embedding: {e}")
            
            # Create a random embedding
            embedding = np.random.randn(self.input_size).astype(np.float32)
            
            # Add some influence from emotional state
            if emotional_state:
                for i, (emotion, value) in enumerate(emotional_state.items()):
                    if i < self.input_size // 4:
                        embedding[i] += value
            
            return torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _calculate_understanding_level(
        self, 
        output_embedding: torch.Tensor, 
        complexity: float
    ) -> float:
        """Calculate the level of understanding based on output embedding and complexity.
        
        Args:
            output_embedding: Output embedding from neural network
            complexity: Complexity of the input
            
        Returns:
            Understanding level (0.0 to 1.0)
        """
        # Extract the output embedding
        output_np = output_embedding.cpu().numpy().flatten()
        
        # Calculate coherence of the output (using variance as a proxy)
        coherence = 1.0 - min(1.0, np.var(output_np))
        
        # Adjust understanding based on complexity and cognitive development
        problem_solving_factor = self.cognitive_development["problem_solving"]
        abstract_thinking_factor = self.cognitive_development["abstract_thinking"]
        
        # Calculate understanding level
        understanding = coherence * (1.0 - complexity * 0.8)
        
        # Adjust based on cognitive development
        if complexity > 0.5:
            understanding *= (0.2 + 0.8 * problem_solving_factor)
        
        if complexity > 0.7:
            understanding *= (0.2 + 0.8 * abstract_thinking_factor)
        
        # Apply attention factor
        understanding *= self.current_attention_level
        
        return float(max(0.0, min(1.0, understanding)))
    
    def _generate_cognitive_response(
        self, 
        understanding_level: float, 
        utterance: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a cognitive response based on understanding level.
        
        Args:
            understanding_level: Level of understanding (0.0 to 1.0)
            utterance: Input utterance
            context: Context dictionary
            
        Returns:
            Dictionary containing cognitive response
        """
        # Determine response type based on understanding level
        if understanding_level < 0.2:
            response_type = "confusion"
        elif understanding_level < 0.5:
            response_type = "partial_understanding"
        else:
            response_type = "understanding"
        
        # Determine if curiosity is triggered
        curiosity_triggered = (
            self.curiosity_level > 0.6 and 
            random.random() < self.curiosity_level * 0.5
        )
        
        # Generate response
        response = {
            "response_type": response_type,
            "understanding_level": understanding_level,
            "curiosity_triggered": curiosity_triggered,
            "cognitive_processes": {
                "attention_focused": self.current_attention_level > 0.7,
                "memory_activated": understanding_level > 0.3,
                "problem_solving_engaged": understanding_level > 0.5 and self.cognitive_development["problem_solving"] > 0.3,
                "abstract_thinking_engaged": understanding_level > 0.7 and self.cognitive_development["abstract_thinking"] > 0.3
            }
        }
        
        return response
    
    def _calculate_novelty(self, input_data: Dict[str, Any]) -> float:
        """Calculate the novelty of the input data.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Novelty score (0.0 to 1.0)
        """
        # Extract utterance
        utterance = input_data.get("mother_utterance", "")
        
        # Check if we have learning history
        if not self.learning_history:
            return 0.8  # High novelty if no history
        
        # Calculate similarity to previous inputs
        similarities = []
        for entry in self.learning_history[-10:]:  # Check last 10 entries
            prev_utterance = entry.get("input", {}).get("mother_utterance", "")
            if prev_utterance:
                # Simple similarity based on length difference
                # In a real system, you'd use proper semantic similarity
                len_diff = abs(len(utterance) - len(prev_utterance)) / max(len(utterance), len(prev_utterance), 1)
                similarity = 1.0 - len_diff
                similarities.append(similarity)
        
        # Calculate average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            novelty = 1.0 - avg_similarity
        else:
            novelty = 0.8
        
        return max(0.1, min(1.0, novelty))
    
    def _record_learning(self, input_data: Dict[str, Any], understanding_level: float):
        """Record learning from an interaction.
        
        Args:
            input_data: Input data dictionary
            understanding_level: Level of understanding achieved
        """
        learning_entry = {
            "timestamp": time.time(),
            "input": input_data,
            "understanding_level": understanding_level,
            "cognitive_state": {
                "attention": self.current_attention_level,
                "cognitive_load": self.cognitive_load,
                "curiosity": self.curiosity_level
            }
        }
        
        self.learning_history.append(learning_entry)
        
        # Limit history size
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
        
        # Update cognitive development based on learning
        self._update_cognitive_development(understanding_level, input_data)
    
    def _update_cognitive_development(
        self, 
        understanding_level: float, 
        input_data: Dict[str, Any]
    ):
        """Update cognitive development based on learning.
        
        Args:
            understanding_level: Level of understanding achieved
            input_data: Input data dictionary
        """
        # Extract complexity
        complexity = input_data.get("complexity", 0.5)
        
        # Calculate learning amount
        learning_amount = understanding_level * 0.01  # Small incremental learning
        
        # Update attention development if sustained attention was required
        if self.current_attention_level > 0.7:
            self.cognitive_development["attention"] = min(
                1.0, self.cognitive_development["attention"] + learning_amount
            )
        
        # Update memory development if memory was involved
        if understanding_level > 0.3:
            self.cognitive_development["memory"] = min(
                1.0, self.cognitive_development["memory"] + learning_amount
            )
        
        # Update problem-solving development if complex problem
        if complexity > 0.5 and understanding_level > 0.5:
            self.cognitive_development["problem_solving"] = min(
                1.0, self.cognitive_development["problem_solving"] + learning_amount * 1.5
            )
        
        # Update abstract thinking if very complex problem
        if complexity > 0.7 and understanding_level > 0.6:
            self.cognitive_development["abstract_thinking"] = min(
                1.0, self.cognitive_development["abstract_thinking"] + learning_amount * 2.0
            )
        
        # Update problem complexity threshold based on development
        self.problem_complexity_threshold = 0.1 + 0.9 * (
            self.cognitive_development["problem_solving"] * 0.7 +
            self.cognitive_development["abstract_thinking"] * 0.3
        )
        
        # Update attention span based on attention development
        self.attention_span_seconds = 5.0 + 55.0 * self.cognitive_development["attention"]
    
    def focus_attention(self, stimulus_importance: float = 0.5):
        """Focus attention on a stimulus.
        
        Args:
            stimulus_importance: Importance of the stimulus (0.0 to 1.0)
        """
        # Increase attention based on stimulus importance
        attention_boost = stimulus_importance * 0.5
        self.current_attention_level = min(1.0, self.current_attention_level + attention_boost)
        
        # Reset attention timer
        self.last_attention_update = time.time()
    
    def train(self, input_data: Dict[str, Any], target_data: Dict[str, Any]) -> float:
        """Train the cognitive component.
        
        Args:
            input_data: Input data dictionary
            target_data: Target data dictionary
            
        Returns:
            Loss value
        """
        # Create input embedding
        input_embedding = self._create_input_embedding(
            input_data.get("mother_utterance", ""),
            input_data.get("emotional_state", {}),
            input_data.get("context", {})
        )
        
        # Create target embedding (simplified)
        target_embedding = torch.randn(1, self.output_size, device=self.device)
        
        # Forward pass
        output_embedding = self.cognitive_network(input_embedding)
        
        # Calculate loss (MSE)
        loss = nn.functional.mse_loss(output_embedding, target_embedding)
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_developmental_metrics(self) -> Dict[str, float]:
        """Get the developmental metrics for the cognitive component.
        
        Returns:
            Dictionary of developmental metrics
        """
        return dict(self.cognitive_development)
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save neural network
        model_path = directory / f"{self.name}_model.pt"
        torch.save(self.cognitive_network.state_dict(), model_path)
        
        # Save state
        state = {
            "name": self.name,
            "cognitive_development": self.cognitive_development,
            "attention_span_seconds": self.attention_span_seconds,
            "attention_decay_rate": self.attention_decay_rate,
            "current_attention_level": self.current_attention_level,
            "last_attention_update": self.last_attention_update,
            "problem_complexity_threshold": self.problem_complexity_threshold,
            "tool_use_capability": self.tool_use_capability,
            "causal_reasoning_ability": self.causal_reasoning_ability,
            "symbol_understanding": self.symbol_understanding,
            "metaphor_understanding": self.metaphor_understanding,
            "hypothetical_reasoning": self.hypothetical_reasoning,
            "cognitive_biases": self.cognitive_biases,
            "cognitive_load": self.cognitive_load,
            "curiosity_level": self.curiosity_level,
            "learning_history": self.learning_history
        }
        
        # Save state
        state_path = directory / f"{self.name}_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Load neural network
        model_path = directory / f"{self.name}_model.pt"
        if model_path.exists():
            self.cognitive_network.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load state
        state_path = directory / f"{self.name}_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
                self.name = state["name"]
                self.cognitive_development = state["cognitive_development"]
                self.attention_span_seconds = state["attention_span_seconds"]
                self.attention_decay_rate = state["attention_decay_rate"]
                self.current_attention_level = state["current_attention_level"]
                self.last_attention_update = state["last_attention_update"]
                self.problem_complexity_threshold = state["problem_complexity_threshold"]
                self.tool_use_capability = state["tool_use_capability"]
                self.causal_reasoning_ability = state["causal_reasoning_ability"]
                self.symbol_understanding = state["symbol_understanding"]
                self.metaphor_understanding = state["metaphor_understanding"]
                self.hypothetical_reasoning = state["hypothetical_reasoning"]
                self.cognitive_biases = state["cognitive_biases"]
                self.cognitive_load = state["cognitive_load"]
                self.curiosity_level = state["curiosity_level"]
                self.learning_history = state["learning_history"]
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return cognitive outputs.
        
        Args:
            inputs: Dictionary containing input data such as:
                - mother_utterance: Text from the mother
                - mother_emotional_state: Emotional state of the mother
                - context: Contextual information
                - teaching_elements: Any explicit teaching content
                
        Returns:
            Dictionary containing cognitive outputs such as:
                - understanding_level: How well the input was understood (0.0 to 1.0)
                - attention_focus: What the child is focusing on
                - cognitive_response: Generated cognitive response
                - novelty: How novel the input is (0.0 to 1.0)
        """
        # Process the input data
        return self.process_input(inputs) 