"""
Emotional component for the Neural Child's mind.

This module contains the implementation of the emotional component that handles
emotional development and processing for the simulated mind.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import random
from pathlib import Path

from neural_child.mind.base import NeuralComponent

class EmotionalComponent(NeuralComponent):
    """Emotional component that handles emotional development and processing."""
    
    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 64,
        output_size: int = 8,
        name: str = "emotional_component"
    ):
        """Initialize the emotional component.
        
        Args:
            input_size: Size of the input layer
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer
            name: Name of the component
        """
        super().__init__(input_size, hidden_size, output_size, name)
        
        # Emotion labels
        self.emotion_labels = [
            "joy", "sadness", "fear", "anger", 
            "surprise", "disgust", "trust", "anticipation"
        ]
        
        # Emotional development metrics
        self.basic_emotions_development = 0.1  # Start with minimal basic emotions
        self.emotional_regulation_development = 0.0  # Start with no emotional regulation
        self.emotional_complexity_development = 0.0  # Start with no emotional complexity
        
        # Emotional memory
        self.emotional_memory: List[Dict[str, Any]] = []
        
        # Emotional contagion factor (how much the child is influenced by others' emotions)
        self.emotional_contagion_factor = 0.8  # Start with high emotional contagion
        
        # Emotional stability (how stable the child's emotions are)
        self.emotional_stability = 0.2  # Start with low emotional stability
        
        # Current emotional state
        self.current_emotional_state = {
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.0,
            "anticipation": 0.0
        }
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs.
        
        Args:
            inputs: Dictionary of inputs to the component
                - mother_emotional_state: Dictionary of the mother's emotional state
                - interaction_context: Dictionary of the interaction context
                - developmental_stage: String representing the developmental stage
                - age_months: Float representing the age in months
                
        Returns:
            Dictionary of outputs from the component
                - emotional_state: Dictionary of the child's emotional state
                - dominant_emotion: Tuple of (emotion, intensity)
                - emotional_development: Dictionary of emotional development metrics
        """
        # Extract inputs
        mother_emotional_state = inputs.get("mother_emotional_state", {})
        interaction_context = inputs.get("interaction_context", {})
        developmental_stage = inputs.get("developmental_stage", "Prenatal")
        age_months = inputs.get("age_months", 0.0)
        
        # Update emotional state based on mother's emotional state (emotional contagion)
        self._update_emotional_state_from_mother(mother_emotional_state)
        
        # Update emotional state based on interaction context
        self._update_emotional_state_from_context(interaction_context)
        
        # Apply emotional regulation based on developmental stage
        self._apply_emotional_regulation(developmental_stage)
        
        # Update emotional development metrics based on age
        self._update_emotional_development(age_months)
        
        # Get dominant emotion
        dominant_emotion = self._get_dominant_emotion()
        
        # Store emotional memory
        self._store_emotional_memory(interaction_context, dominant_emotion)
        
        # Prepare outputs
        outputs = {
            "emotional_state": self.current_emotional_state.copy(),
            "dominant_emotion": dominant_emotion,
            "emotional_development": {
                "basic_emotions": self.basic_emotions_development,
                "emotional_regulation": self.emotional_regulation_development,
                "emotional_complexity": self.emotional_complexity_development,
                "emotional_contagion_factor": self.emotional_contagion_factor,
                "emotional_stability": self.emotional_stability
            }
        }
        
        # Update activation level
        self.update_activation(max(self.current_emotional_state.values()))
        
        return outputs
    
    def _update_emotional_state_from_mother(self, mother_emotional_state: Dict[str, float]):
        """Update the emotional state based on the mother's emotional state.
        
        Args:
            mother_emotional_state: Dictionary of the mother's emotional state
        """
        if not mother_emotional_state:
            return
        
        # Apply emotional contagion
        for emotion, intensity in mother_emotional_state.items():
            if emotion in self.current_emotional_state:
                # Child's emotion is influenced by mother's emotion based on contagion factor
                self.current_emotional_state[emotion] = (
                    self.current_emotional_state[emotion] * (1 - self.emotional_contagion_factor) +
                    intensity * self.emotional_contagion_factor
                )
    
    def _update_emotional_state_from_context(self, interaction_context: Dict[str, Any]):
        """Update the emotional state based on the interaction context.
        
        Args:
            interaction_context: Dictionary of the interaction context
        """
        if not interaction_context:
            return
        
        # Extract relevant information from context
        context_type = interaction_context.get("type", "")
        context_valence = interaction_context.get("valence", 0.0)
        context_arousal = interaction_context.get("arousal", 0.0)
        
        # Update emotions based on context type and valence/arousal
        if context_type == "nurturing":
            # Nurturing interactions increase joy and trust
            self.current_emotional_state["joy"] += 0.1 * context_valence
            self.current_emotional_state["trust"] += 0.1 * context_valence
        elif context_type == "playful":
            # Playful interactions increase joy and surprise
            self.current_emotional_state["joy"] += 0.1 * context_valence
            self.current_emotional_state["surprise"] += 0.1 * context_arousal
        elif context_type == "educational":
            # Educational interactions increase trust and anticipation
            self.current_emotional_state["trust"] += 0.1 * context_valence
            self.current_emotional_state["anticipation"] += 0.1 * context_arousal
        elif context_type == "disciplinary":
            # Disciplinary interactions may increase fear or sadness
            self.current_emotional_state["fear"] += 0.1 * context_arousal
            self.current_emotional_state["sadness"] += 0.1 * (1 - context_valence)
        
        # Add some random variation to emotions
        for emotion in self.current_emotional_state:
            self.current_emotional_state[emotion] += random.uniform(-0.05, 0.05)
        
        # Ensure all emotions are within bounds
        for emotion in self.current_emotional_state:
            self.current_emotional_state[emotion] = max(0.0, min(1.0, self.current_emotional_state[emotion]))
    
    def _apply_emotional_regulation(self, developmental_stage: str):
        """Apply emotional regulation based on developmental stage.
        
        Args:
            developmental_stage: String representing the developmental stage
        """
        # Different stages have different levels of emotional regulation
        regulation_factor = 0.0
        
        if developmental_stage == "Prenatal":
            regulation_factor = 0.0
        elif developmental_stage == "Infancy":
            regulation_factor = 0.1
        elif developmental_stage == "Early Childhood":
            regulation_factor = 0.3
        elif developmental_stage == "Middle Childhood":
            regulation_factor = 0.5
        elif developmental_stage == "Adolescence":
            regulation_factor = 0.7
        elif developmental_stage == "Early Adulthood":
            regulation_factor = 0.9
        
        # Apply emotional regulation
        regulation_strength = regulation_factor * self.emotional_regulation_development
        
        # Regulate emotions by moving them towards a neutral state
        for emotion in self.current_emotional_state:
            # Move towards neutral (0.5 for positive emotions, 0.0 for negative emotions)
            neutral_value = 0.5 if emotion in ["joy", "trust", "anticipation"] else 0.0
            self.current_emotional_state[emotion] = (
                self.current_emotional_state[emotion] * (1 - regulation_strength) +
                neutral_value * regulation_strength
            )
    
    def _update_emotional_development(self, age_months: float):
        """Update emotional development metrics based on age.
        
        Args:
            age_months: Float representing the age in months
        """
        # Basic emotions develop early
        if age_months <= 12:
            # First year: rapid development of basic emotions
            self.basic_emotions_development = min(0.5, 0.1 + age_months * 0.033)
        elif age_months <= 36:
            # 1-3 years: continued development of basic emotions
            self.basic_emotions_development = min(0.8, 0.5 + (age_months - 12) * 0.0125)
        else:
            # After 3 years: slow refinement of basic emotions
            self.basic_emotions_development = min(1.0, 0.8 + (age_months - 36) * 0.001)
        
        # Emotional regulation develops later
        if age_months <= 24:
            # First 2 years: minimal emotional regulation
            self.emotional_regulation_development = min(0.2, age_months * 0.008)
        elif age_months <= 60:
            # 2-5 years: developing emotional regulation
            self.emotional_regulation_development = min(0.5, 0.2 + (age_months - 24) * 0.008)
        elif age_months <= 144:
            # 5-12 years: significant development of emotional regulation
            self.emotional_regulation_development = min(0.8, 0.5 + (age_months - 60) * 0.0035)
        else:
            # After 12 years: refinement of emotional regulation
            self.emotional_regulation_development = min(1.0, 0.8 + (age_months - 144) * 0.001)
        
        # Emotional complexity develops gradually
        if age_months <= 36:
            # First 3 years: minimal emotional complexity
            self.emotional_complexity_development = min(0.3, age_months * 0.008)
        elif age_months <= 96:
            # 3-8 years: developing emotional complexity
            self.emotional_complexity_development = min(0.6, 0.3 + (age_months - 36) * 0.005)
        elif age_months <= 216:
            # 8-18 years: significant development of emotional complexity
            self.emotional_complexity_development = min(0.9, 0.6 + (age_months - 96) * 0.0025)
        else:
            # After 18 years: refinement of emotional complexity
            self.emotional_complexity_development = min(1.0, 0.9 + (age_months - 216) * 0.0005)
        
        # Emotional contagion factor decreases with age (becomes less influenced by others)
        self.emotional_contagion_factor = max(0.2, 0.8 - age_months * 0.002)
        
        # Emotional stability increases with age
        self.emotional_stability = min(0.9, 0.2 + age_months * 0.003)
    
    def _get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the dominant emotion.
        
        Returns:
            Tuple of (emotion, intensity)
        """
        if not self.current_emotional_state:
            return ("neutral", 0.0)
        
        dominant_emotion = max(self.current_emotional_state.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def _store_emotional_memory(self, interaction_context: Dict[str, Any], dominant_emotion: Tuple[str, float]):
        """Store emotional memory.
        
        Args:
            interaction_context: Dictionary of the interaction context
            dominant_emotion: Tuple of (emotion, intensity)
        """
        if not interaction_context:
            return
        
        # Create emotional memory entry
        memory_entry = {
            "context": interaction_context,
            "dominant_emotion": dominant_emotion,
            "emotional_state": self.current_emotional_state.copy()
        }
        
        # Add to emotional memory
        self.emotional_memory.append(memory_entry)
        
        # Limit memory size
        if len(self.emotional_memory) > 100:
            self.emotional_memory.pop(0)
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Get the current emotional state.
        
        Returns:
            Dictionary of the current emotional state
        """
        return self.current_emotional_state.copy()
    
    def get_emotional_development_metrics(self) -> Dict[str, float]:
        """Get the emotional development metrics.
        
        Returns:
            Dictionary of emotional development metrics
        """
        return {
            "basic_emotions": self.basic_emotions_development,
            "emotional_regulation": self.emotional_regulation_development,
            "emotional_complexity": self.emotional_complexity_development,
            "emotional_contagion_factor": self.emotional_contagion_factor,
            "emotional_stability": self.emotional_stability
        }
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Call parent save method
        super().save(directory)
        
        # Save additional state
        additional_state = {
            "basic_emotions_development": self.basic_emotions_development,
            "emotional_regulation_development": self.emotional_regulation_development,
            "emotional_complexity_development": self.emotional_complexity_development,
            "emotional_contagion_factor": self.emotional_contagion_factor,
            "emotional_stability": self.emotional_stability,
            "current_emotional_state": self.current_emotional_state,
            "emotional_memory": self.emotional_memory
        }
        
        # Save additional state
        additional_state_path = directory / f"{self.name}_additional_state.json"
        with open(additional_state_path, "w") as f:
            import json
            json.dump(additional_state, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Call parent load method
        super().load(directory)
        
        # Load additional state
        additional_state_path = directory / f"{self.name}_additional_state.json"
        if additional_state_path.exists():
            with open(additional_state_path, "r") as f:
                import json
                additional_state = json.load(f)
                self.basic_emotions_development = additional_state["basic_emotions_development"]
                self.emotional_regulation_development = additional_state["emotional_regulation_development"]
                self.emotional_complexity_development = additional_state["emotional_complexity_development"]
                self.emotional_contagion_factor = additional_state["emotional_contagion_factor"]
                self.emotional_stability = additional_state["emotional_stability"]
                self.current_emotional_state = additional_state["current_emotional_state"]
                self.emotional_memory = additional_state["emotional_memory"] 