"""
Social component for the Neural Child's mind.

This module contains the implementation of the social component that handles
social interactions, attachment, and social development for the simulated mind.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import random
import time
import json
import requests
from pathlib import Path
from collections import defaultdict

from neural_child.mind.base import NeuralComponent

class SocialComponent(NeuralComponent):
    """Social component that handles social interactions and relationship development."""
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        output_size: int = 64,
        learning_rate: float = 0.01,
        device: str = "cpu",
        name: str = "social_component",
        embedding_api_url: str = "http://192.168.2.12:1234/v1/embeddings",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ):
        """Initialize the social component.
        
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
        
        # Neural network for social processing
        self.social_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.social_network.parameters(), 
            lr=learning_rate
        )
        
        # Social development metrics
        self.social_development = {
            "attachment": 0.1,  # Attachment to primary caregiver
            "social_awareness": 0.0,  # Awareness of social cues and norms
            "empathy": 0.0,  # Ability to understand others' emotions
            "theory_of_mind": 0.0  # Understanding that others have different mental states
        }
        
        # Attachment parameters
        self.attachment_figures = {
            "mother": {
                "strength": 0.2,  # Initial attachment strength
                "security": 0.5,  # Security of attachment (0.0 = insecure, 1.0 = secure)
                "interactions": 0,  # Number of interactions
                "positive_interactions": 0,  # Number of positive interactions
                "negative_interactions": 0,  # Number of negative interactions
                "last_interaction": time.time()  # Time of last interaction
            }
        }
        
        # Social relationships (beyond primary attachment figures)
        self.relationships = {}
        
        # Social interaction history
        self.interaction_history = []
        
        # Social schemas (understanding of social situations)
        self.social_schemas = {}
        
        # Social rules learned
        self.social_rules = []
        
        # Empathy development
        self.emotional_recognition = 0.1  # Ability to recognize emotions
        self.perspective_taking = 0.0  # Ability to take others' perspectives
        
        # Theory of mind development
        self.false_belief_understanding = 0.0  # Understanding that others can have false beliefs
        self.intention_understanding = 0.0  # Understanding others' intentions
        
        # Social preferences
        self.social_preferences = {
            "familiarity_preference": 0.8,  # Preference for familiar individuals (starts high)
            "novelty_seeking": 0.2,  # Interest in new social interactions (starts low)
            "social_approach": 0.5  # Tendency to approach vs avoid social interaction
        }
        
        # Stranger anxiety (develops around 8-12 months)
        self.stranger_anxiety = 0.0
        
        # Social referencing (checking caregiver's reaction in uncertain situations)
        self.social_referencing = 0.0
    
    def process_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a social interaction.
        
        Args:
            interaction_data: Dictionary containing interaction data
                - agent: The person interacting with the child (e.g., "mother", "stranger")
                - content: Content of the interaction
                - emotional_tone: Emotional tone of the interaction
                - context: Context of the interaction
                - age_months: Child's age in months
                
        Returns:
            Dictionary containing processed output
        """
        # Extract relevant information from input
        agent = interaction_data.get("agent", "unknown")
        content = interaction_data.get("content", "")
        emotional_tone = interaction_data.get("emotional_tone", {})
        context = interaction_data.get("context", {})
        age_months = interaction_data.get("age_months", 0.0)
        
        # Update stranger anxiety based on age
        self._update_stranger_anxiety(age_months)
        
        # Calculate social response
        social_response = self._calculate_social_response(agent, content, emotional_tone, context)
        
        # Update attachment based on interaction
        if agent in self.attachment_figures:
            self._update_attachment(agent, content, emotional_tone, context)
        else:
            # If this is a new agent, initialize relationship
            self._initialize_relationship(agent)
        
        # Update social development based on interaction
        self._update_social_development(interaction_data, social_response)
        
        # Record interaction in history
        self._record_interaction(interaction_data, social_response)
        
        # Prepare output
        output = {
            "social_response": social_response,
            "attachment_status": self._get_attachment_status(agent),
            "social_development": self.social_development,
            "stranger_anxiety_triggered": self._is_stranger_anxiety_triggered(agent),
            "social_referencing_used": social_response.get("social_referencing_used", False)
        }
        
        return output
    
    def _calculate_social_response(
        self, 
        agent: str, 
        content: str, 
        emotional_tone: Dict[str, float], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the social response to an interaction.
        
        Args:
            agent: The person interacting with the child
            content: Content of the interaction
            emotional_tone: Emotional tone of the interaction
            context: Context of the interaction
            
        Returns:
            Dictionary containing social response
        """
        # Create input embedding
        input_embedding = self._create_social_embedding(agent, content, emotional_tone, context)
        
        # Process through neural network
        with torch.no_grad():
            output_embedding = self.social_network(input_embedding)
        
        # Convert output to response parameters
        output_np = output_embedding.cpu().numpy().flatten()
        
        # Determine if stranger anxiety is triggered
        stranger_anxiety_triggered = (
            agent not in self.attachment_figures and 
            self.stranger_anxiety > 0.3 and
            random.random() < self.stranger_anxiety
        )
        
        # Determine if social referencing is used
        social_referencing_used = (
            self.social_referencing > 0.3 and
            random.random() < self.social_referencing and
            context.get("uncertainty", 0.0) > 0.5
        )
        
        # Calculate approach vs avoidance
        # Higher values = more approach, lower values = more avoidance
        approach_tendency = self.social_preferences["social_approach"]
        
        # Adjust for familiarity
        if agent in self.attachment_figures:
            # More likely to approach attachment figures
            approach_tendency += self.attachment_figures[agent]["strength"] * 0.3
        elif agent in self.relationships:
            # Adjust based on relationship quality
            approach_tendency += self.relationships[agent]["quality"] * 0.2
        else:
            # Less likely to approach strangers, especially with stranger anxiety
            approach_tendency -= self.stranger_anxiety * 0.5
        
        # Adjust for emotional tone
        if emotional_tone:
            # Positive emotions increase approach, negative decrease it
            positivity = (
                emotional_tone.get("joy", 0.0) + 
                emotional_tone.get("interest", 0.0) - 
                emotional_tone.get("fear", 0.0) - 
                emotional_tone.get("anger", 0.0)
            ) * 0.2
            approach_tendency += positivity
        
        # Ensure within bounds
        approach_tendency = max(0.0, min(1.0, approach_tendency))
        
        # Generate response
        response = {
            "approach_tendency": approach_tendency,
            "engagement_level": output_np[0] if len(output_np) > 0 else 0.5,
            "emotional_resonance": output_np[1] if len(output_np) > 1 else 0.3,
            "social_understanding": output_np[2] if len(output_np) > 2 else 0.2,
            "stranger_anxiety_triggered": stranger_anxiety_triggered,
            "social_referencing_used": social_referencing_used,
            "attachment_seeking": self._calculate_attachment_seeking(agent, context)
        }
        
        return response
    
    def _create_social_embedding(
        self, 
        agent: str, 
        content: str, 
        emotional_tone: Dict[str, float], 
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Create an embedding vector for social interaction using the embedding API.
        
        Args:
            agent: The person interacting with the child
            content: Content of the interaction
            emotional_tone: Emotional tone of the interaction
            context: Context of the interaction
            
        Returns:
            Tensor embedding of the social interaction
        """
        # Create a context-enriched input for better embeddings
        emotional_context = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in emotional_tone.items()])
        
        # Combine agent, content, and emotional context
        enriched_input = f"Agent: {agent} | Content: {content}"
        if emotional_context:
            enriched_input += f" | Emotional tone: {emotional_context}"
            
        # Add any relevant context information
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if isinstance(v, (str, int, float))])
            if context_str:
                enriched_input += f" | Context: {context_str}"
        
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
            
            # Convert to tensor
            return torch.tensor(embedding_np, device=self.device).float().unsqueeze(0)
            
        except Exception as e:
            # Fallback to a simpler method if API call fails
            print(f"Embedding API call failed: {e}. Using fallback embedding method.")
            
            # Create a random embedding based on the hash of the input (fallback method)
            seed = hash(enriched_input) % 10000
            np.random.seed(seed)
            
            # Create a random embedding
            embedding = np.random.randn(self.input_size).astype(np.float32)
            
            # Add some influence from emotional tone
            if emotional_tone:
                for i, (emotion, value) in enumerate(emotional_tone.items()):
                    if i < self.input_size // 4:
                        embedding[i] += value
            
            # Normalize the embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return torch.tensor(embedding, device=self.device).float().unsqueeze(0)
    
    def _update_attachment(
        self, 
        agent: str, 
        content: str, 
        emotional_tone: Dict[str, float], 
        context: Dict[str, Any]
    ):
        """Update attachment based on interaction.
        
        Args:
            agent: The person interacting with the child
            content: Content of the interaction
            emotional_tone: Emotional tone of the interaction
            context: Context of the interaction
        """
        if agent not in self.attachment_figures:
            return
        
        # Get attachment figure data
        attachment = self.attachment_figures[agent]
        
        # Update interaction counts
        attachment["interactions"] += 1
        attachment["last_interaction"] = time.time()
        
        # Calculate interaction quality (-1.0 to 1.0)
        quality = 0.0
        
        # Emotional tone affects quality
        if emotional_tone:
            # Positive emotions contribute positively
            quality += emotional_tone.get("joy", 0.0) * 0.3
            quality += emotional_tone.get("interest", 0.0) * 0.2
            
            # Negative emotions contribute negatively
            quality -= emotional_tone.get("anger", 0.0) * 0.3
            quality -= emotional_tone.get("fear", 0.0) * 0.2
            quality -= emotional_tone.get("disgust", 0.0) * 0.1
        
        # Context can affect quality
        responsiveness = context.get("responsiveness", 0.5)
        quality += (responsiveness - 0.5) * 0.4
        
        # Update positive/negative interaction counts
        if quality > 0.2:
            attachment["positive_interactions"] += 1
        elif quality < -0.2:
            attachment["negative_interactions"] += 1
        
        # Calculate attachment strength change
        strength_change = quality * 0.01  # Small incremental changes
        
        # Attachment is more malleable in early development
        if self.social_development["attachment"] < 0.3:
            strength_change *= 2.0
        
        # Update attachment strength
        attachment["strength"] = max(0.0, min(1.0, attachment["strength"] + strength_change))
        
        # Update attachment security
        # Security is based on consistency and responsiveness
        if attachment["interactions"] > 10:
            consistency = 1.0 - (abs(0.8 - (attachment["positive_interactions"] / attachment["interactions"])))
            attachment["security"] = 0.3 * attachment["security"] + 0.7 * consistency
    
    def _initialize_relationship(self, agent: str):
        """Initialize a new relationship.
        
        Args:
            agent: The person to initialize relationship with
        """
        if agent not in self.relationships and agent not in self.attachment_figures:
            self.relationships[agent] = {
                "quality": 0.5,  # Neutral initial quality
                "familiarity": 0.1,  # Low initial familiarity
                "interactions": 1,  # Start with 1 for this interaction
                "positive_interactions": 0,
                "negative_interactions": 0,
                "last_interaction": time.time()
            }
    
    def _update_social_development(
        self, 
        interaction_data: Dict[str, Any], 
        social_response: Dict[str, Any]
    ):
        """Update social development based on interaction.
        
        Args:
            interaction_data: Dictionary containing interaction data
            social_response: Dictionary containing social response
        """
        # Extract relevant information
        agent = interaction_data.get("agent", "unknown")
        emotional_tone = interaction_data.get("emotional_tone", {})
        context = interaction_data.get("context", {})
        age_months = interaction_data.get("age_months", 0.0)
        
        # Calculate learning amount (small incremental learning)
        learning_amount = 0.005
        
        # Attachment development
        if agent in self.attachment_figures:
            # More significant attachment development with primary caregivers
            attachment_learning = learning_amount * 2.0
            self.social_development["attachment"] = min(
                1.0, self.social_development["attachment"] + attachment_learning
            )
        
        # Social awareness development
        if context.get("social_complexity", 0.0) > 0.3:
            # Complex social situations develop social awareness
            awareness_learning = learning_amount * context.get("social_complexity", 0.0) * 2.0
            self.social_development["social_awareness"] = min(
                1.0, self.social_development["social_awareness"] + awareness_learning
            )
        
        # Empathy development
        if emotional_tone and self.social_development["attachment"] > 0.3:
            # Emotional interactions develop empathy, especially after secure attachment forms
            empathy_learning = learning_amount * sum(emotional_tone.values()) / len(emotional_tone)
            self.social_development["empathy"] = min(
                1.0, self.social_development["empathy"] + empathy_learning
            )
            
            # Update emotional recognition
            self.emotional_recognition = min(
                1.0, self.emotional_recognition + empathy_learning * 0.5
            )
        
        # Theory of mind development
        # This typically develops around 3-4 years (36-48 months)
        if age_months > 30 and context.get("perspective_taking", 0.0) > 0.5:
            # Interactions involving different perspectives develop theory of mind
            tom_learning = learning_amount * context.get("perspective_taking", 0.0) * 2.0
            self.social_development["theory_of_mind"] = min(
                1.0, self.social_development["theory_of_mind"] + tom_learning
            )
            
            # Update false belief understanding
            if age_months > 36:
                self.false_belief_understanding = min(
                    1.0, self.false_belief_understanding + tom_learning * 0.5
                )
        
        # Update perspective taking ability
        if context.get("perspective_taking", 0.0) > 0.3:
            self.perspective_taking = min(
                1.0, self.perspective_taking + learning_amount
            )
        
        # Update intention understanding
        if context.get("intention_clarity", 0.0) > 0.5:
            self.intention_understanding = min(
                1.0, self.intention_understanding + learning_amount
            )
        
        # Update social referencing
        if age_months > 8:  # Social referencing typically emerges around 8-10 months
            self.social_referencing = min(
                1.0, 
                max(
                    (age_months - 8) / 10,  # Age-based development
                    self.social_referencing + learning_amount * 0.5
                )
            )
    
    def _update_stranger_anxiety(self, age_months: float):
        """Update stranger anxiety based on age.
        
        Args:
            age_months: Child's age in months
        """
        # Stranger anxiety typically develops around 8-12 months and peaks around 12-18 months
        if 8 <= age_months <= 24:
            # Ramp up from 8-12 months
            if 8 <= age_months <= 12:
                self.stranger_anxiety = (age_months - 8) / 4
            # Peak from 12-18 months
            elif 12 < age_months <= 18:
                self.stranger_anxiety = 1.0
            # Gradually decrease after 18 months
            else:  # 18 < age_months <= 24
                self.stranger_anxiety = 1.0 - (age_months - 18) / 12
        elif age_months > 24:
            # Maintain a low level after 24 months
            self.stranger_anxiety = max(0.1, 1.0 - (age_months - 18) / 12)
        else:
            # No stranger anxiety before 8 months
            self.stranger_anxiety = 0.0
    
    def _is_stranger_anxiety_triggered(self, agent: str) -> bool:
        """Check if stranger anxiety is triggered.
        
        Args:
            agent: The person interacting with the child
            
        Returns:
            Boolean indicating if stranger anxiety is triggered
        """
        # Stranger anxiety is triggered for unfamiliar people
        if agent not in self.attachment_figures:
            # Check if this is a relatively unfamiliar person
            if agent not in self.relationships or self.relationships[agent]["familiarity"] < 0.4:
                # Probability based on stranger anxiety level
                return random.random() < self.stranger_anxiety
        
        return False
    
    def _calculate_attachment_seeking(self, agent: str, context: Dict[str, Any]) -> float:
        """Calculate attachment seeking behavior.
        
        Args:
            agent: The person interacting with the child
            context: Context of the interaction
            
        Returns:
            Attachment seeking level (0.0 to 1.0)
        """
        # Base attachment seeking
        attachment_seeking = 0.2
        
        # Increase in stressful situations
        stress_level = context.get("stress", 0.0)
        attachment_seeking += stress_level * 0.5
        
        # Increase if this is an attachment figure
        if agent in self.attachment_figures:
            attachment_figure = self.attachment_figures[agent]
            attachment_seeking += attachment_figure["strength"] * 0.3
            
            # Secure attachment leads to more confident exploration
            if context.get("exploration", 0.0) > 0.5 and attachment_figure["security"] > 0.7:
                attachment_seeking -= 0.2
        
        # Ensure within bounds
        return max(0.0, min(1.0, attachment_seeking))
    
    def _get_attachment_status(self, agent: str) -> Dict[str, Any]:
        """Get attachment status for an agent.
        
        Args:
            agent: The person to get attachment status for
            
        Returns:
            Dictionary containing attachment status
        """
        if agent in self.attachment_figures:
            attachment = self.attachment_figures[agent]
            
            # Determine attachment style
            style = "developing"
            if attachment["interactions"] > 20:
                if attachment["security"] > 0.7:
                    style = "secure"
                elif attachment["strength"] > 0.6:
                    style = "insecure-ambivalent"
                elif attachment["strength"] < 0.3:
                    style = "insecure-avoidant"
                else:
                    style = "insecure-disorganized"
            
            return {
                "exists": True,
                "strength": attachment["strength"],
                "security": attachment["security"],
                "style": style,
                "interactions": attachment["interactions"]
            }
        elif agent in self.relationships:
            relationship = self.relationships[agent]
            return {
                "exists": True,
                "strength": relationship["quality"] * 0.5,  # Relationships are weaker than attachments
                "security": 0.5,  # Neutral security for non-attachment relationships
                "style": "relationship",
                "interactions": relationship["interactions"]
            }
        else:
            return {
                "exists": False,
                "strength": 0.0,
                "security": 0.0,
                "style": "none",
                "interactions": 0
            }
    
    def _record_interaction(
        self, 
        interaction_data: Dict[str, Any], 
        social_response: Dict[str, Any]
    ):
        """Record an interaction in history.
        
        Args:
            interaction_data: Dictionary containing interaction data
            social_response: Dictionary containing social response
        """
        # Create interaction record
        interaction_record = {
            "timestamp": time.time(),
            "agent": interaction_data.get("agent", "unknown"),
            "content": interaction_data.get("content", ""),
            "emotional_tone": interaction_data.get("emotional_tone", {}),
            "context": interaction_data.get("context", {}),
            "social_response": social_response,
            "social_development": dict(self.social_development)
        }
        
        # Add to history
        self.interaction_history.append(interaction_record)
        
        # Limit history size
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
    
    def learn_social_rule(self, rule: Dict[str, Any]):
        """Learn a new social rule.
        
        Args:
            rule: Dictionary containing rule information
                - description: Description of the rule
                - context: Context where the rule applies
                - importance: Importance of the rule (0.0 to 1.0)
        """
        # Add rule to social rules
        if rule not in self.social_rules:
            self.social_rules.append(rule)
            
            # Update social awareness
            importance = rule.get("importance", 0.5)
            self.social_development["social_awareness"] = min(
                1.0, self.social_development["social_awareness"] + importance * 0.02
            )
    
    def get_social_development_metrics(self) -> Dict[str, float]:
        """Get the social development metrics.
        
        Returns:
            Dictionary of social development metrics
        """
        metrics = dict(self.social_development)
        
        # Add additional metrics
        metrics["emotional_recognition"] = self.emotional_recognition
        metrics["perspective_taking"] = self.perspective_taking
        metrics["false_belief_understanding"] = self.false_belief_understanding
        metrics["intention_understanding"] = self.intention_understanding
        metrics["social_referencing"] = self.social_referencing
        metrics["stranger_anxiety"] = self.stranger_anxiety
        
        return metrics
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save neural network
        model_path = directory / f"{self.name}_model.pt"
        torch.save(self.social_network.state_dict(), model_path)
        
        # Save state
        state = {
            "name": self.name,
            "social_development": self.social_development,
            "attachment_figures": self.attachment_figures,
            "relationships": self.relationships,
            "social_schemas": self.social_schemas,
            "social_rules": self.social_rules,
            "emotional_recognition": self.emotional_recognition,
            "perspective_taking": self.perspective_taking,
            "false_belief_understanding": self.false_belief_understanding,
            "intention_understanding": self.intention_understanding,
            "social_preferences": self.social_preferences,
            "stranger_anxiety": self.stranger_anxiety,
            "social_referencing": self.social_referencing
        }
        
        # Save state
        state_path = directory / f"{self.name}_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Save interaction history separately (could be large)
        history_path = directory / f"{self.name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.interaction_history, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Load neural network
        model_path = directory / f"{self.name}_model.pt"
        if model_path.exists():
            self.social_network.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load state
        state_path = directory / f"{self.name}_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
                self.name = state["name"]
                self.social_development = state["social_development"]
                self.attachment_figures = state["attachment_figures"]
                self.relationships = state["relationships"]
                self.social_schemas = state["social_schemas"]
                self.social_rules = state["social_rules"]
                self.emotional_recognition = state["emotional_recognition"]
                self.perspective_taking = state["perspective_taking"]
                self.false_belief_understanding = state["false_belief_understanding"]
                self.intention_understanding = state["intention_understanding"]
                self.social_preferences = state["social_preferences"]
                self.stranger_anxiety = state["stranger_anxiety"]
                self.social_referencing = state["social_referencing"]
        
        # Load interaction history
        history_path = directory / f"{self.name}_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.interaction_history = json.load(f)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process social inputs and return social outputs.
        
        Args:
            inputs: Dictionary containing input data such as:
                - agent: The agent interacting with the child (e.g., "mother")
                - utterance: Text from the agent
                - emotional_state: Emotional state of the agent
                - context: Contextual information
                
        Returns:
            Dictionary containing social outputs such as:
                - social_response: Generated social response
                - attachment_level: Current attachment level to the agent
                - social_interest: Level of social interest
                - stranger_anxiety: Whether stranger anxiety is triggered
        """
        # Process the interaction
        return self.process_interaction(inputs) 