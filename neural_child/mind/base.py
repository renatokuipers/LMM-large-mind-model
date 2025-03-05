"""
Base models for the Neural Child's mind components.

This module contains the base classes for the neural components representing
psychological functions of the simulated mind.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path

class NeuralComponent(nn.Module, ABC):
    """Base class for all neural components in the Neural Child's mind."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, name: str):
        """Initialize the neural component.
        
        Args:
            input_size: Size of the input layer
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer
            name: Name of the component
        """
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Basic neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
        # Component state
        self.activation_level = 0.0
        self.confidence = 0.0
        self.training_progress = 0.0
        self.last_input = None
        self.last_output = None
        self.experience_count = 0
        
        # Developmental metrics
        self.developmental_metrics = {
            "activation_history": [],
            "confidence_history": [],
            "training_progress_history": [],
            "experience_count_history": []
        }
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs.
        
        Args:
            inputs: Dictionary of inputs to the component
            
        Returns:
            Dictionary of outputs from the component
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)
    
    def update_activation(self, activation_level: float):
        """Update the activation level of the component.
        
        Args:
            activation_level: New activation level (0.0 to 1.0)
        """
        self.activation_level = max(0.0, min(1.0, activation_level))
        self.developmental_metrics["activation_history"].append(self.activation_level)
    
    def update_confidence(self, confidence: float):
        """Update the confidence level of the component.
        
        Args:
            confidence: New confidence level (0.0 to 1.0)
        """
        self.confidence = max(0.0, min(1.0, confidence))
        self.developmental_metrics["confidence_history"].append(self.confidence)
    
    def update_training_progress(self, progress: float):
        """Update the training progress of the component.
        
        Args:
            progress: New training progress (0.0 to 1.0)
        """
        self.training_progress = max(0.0, min(1.0, progress))
        self.developmental_metrics["training_progress_history"].append(self.training_progress)
    
    def increment_experience(self):
        """Increment the experience count of the component."""
        self.experience_count += 1
        self.developmental_metrics["experience_count_history"].append(self.experience_count)
    
    def train_component(self, inputs: torch.Tensor, targets: torch.Tensor, learning_rate: float = 0.01) -> float:
        """Train the component on a batch of inputs and targets.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            learning_rate: Learning rate for training
            
        Returns:
            Loss value
        """
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        self.increment_experience()
        self.update_confidence(1.0 - loss.item())
        
        return loss.item()
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save model state
        torch.save(self.state_dict(), directory / f"{self.name}_model.pt")
        
        # Save component state
        component_state = {
            "name": self.name,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "activation_level": self.activation_level,
            "confidence": self.confidence,
            "training_progress": self.training_progress,
            "experience_count": self.experience_count,
            "developmental_metrics": self.developmental_metrics
        }
        with open(directory / f"{self.name}_state.json", "w") as f:
            json.dump(component_state, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Load model state
        model_path = directory / f"{self.name}_model.pt"
        if model_path.exists():
            self.load_state_dict(torch.load(model_path))
        
        # Load component state
        state_path = directory / f"{self.name}_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                component_state = json.load(f)
                self.activation_level = component_state["activation_level"]
                self.confidence = component_state["confidence"]
                self.training_progress = component_state["training_progress"]
                self.experience_count = component_state["experience_count"]
                self.developmental_metrics = component_state["developmental_metrics"]

class MindState(BaseModel):
    """Model representing the current state of the Neural Child's mind."""
    
    # Basic state information
    age_months: float = Field(0.0, ge=-9.0)
    developmental_stage: str = "Prenatal"
    
    # Component activation levels
    component_activations: Dict[str, float] = {}
    
    # Current mental state
    attention_focus: Optional[str] = None
    emotional_state: Dict[str, float] = {
        "joy": 0.0,
        "sadness": 0.0,
        "fear": 0.0,
        "anger": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
    }
    
    # Language capabilities
    vocabulary_size: int = 0
    language_comprehension: float = 0.0
    language_production: float = 0.0
    
    # Memory state
    working_memory: List[Dict[str, Any]] = []
    recent_experiences: List[Dict[str, Any]] = []
    
    # Developmental metrics
    developmental_metrics: Dict[str, Any] = {
        "language": {
            "receptive_language": 0.0,
            "expressive_language": 0.0,
            "vocabulary_size_history": []
        },
        "emotional": {
            "basic_emotions": 0.0,
            "emotional_regulation": 0.0,
            "emotional_complexity": 0.0
        },
        "cognitive": {
            "attention": 0.0,
            "memory": 0.0,
            "problem_solving": 0.0,
            "abstract_thinking": 0.0
        },
        "social": {
            "attachment": 0.0,
            "social_awareness": 0.0,
            "empathy": 0.0,
            "theory_of_mind": 0.0
        }
    }
    
    # Needs and drives
    needs: Dict[str, float] = {
        "physical": 1.0,
        "safety": 1.0,
        "love": 0.5,
        "esteem": 0.0,
        "self_actualization": 0.0
    }
    
    def update_emotional_state(self, emotion: str, value: float):
        """Update the emotional state.
        
        Args:
            emotion: Emotion to update
            value: New value for the emotion (0.0 to 1.0)
        """
        if emotion in self.emotional_state:
            self.emotional_state[emotion] = max(0.0, min(1.0, value))
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the dominant emotion.
        
        Returns:
            Tuple of (emotion, intensity)
        """
        if not self.emotional_state:
            return ("neutral", 0.0)
        
        dominant_emotion = max(self.emotional_state.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def update_developmental_metrics(self, category: str, metric: str, value: float):
        """Update a developmental metric.
        
        Args:
            category: Category of the metric (language, emotional, cognitive, social)
            metric: Name of the metric
            value: New value for the metric (0.0 to 1.0)
        """
        if category in self.developmental_metrics and metric in self.developmental_metrics[category]:
            self.developmental_metrics[category][metric] = max(0.0, min(1.0, value))
    
    def add_to_working_memory(self, item: Dict[str, Any], max_size: int = 5):
        """Add an item to working memory.
        
        Args:
            item: Item to add to working memory
            max_size: Maximum size of working memory
        """
        self.working_memory.append(item)
        if len(self.working_memory) > max_size:
            self.working_memory.pop(0)
    
    def add_experience(self, experience: Dict[str, Any], max_size: int = 20):
        """Add an experience to recent experiences.
        
        Args:
            experience: Experience to add
            max_size: Maximum size of recent experiences
        """
        self.recent_experiences.append(experience)
        if len(self.recent_experiences) > max_size:
            self.recent_experiences.pop(0)
    
    def update_vocabulary_size(self, size: int):
        """Update the vocabulary size.
        
        Args:
            size: New vocabulary size
        """
        self.vocabulary_size = max(0, size)
        self.developmental_metrics["language"]["vocabulary_size_history"].append(self.vocabulary_size)
    
    def update_needs(self, need: str, value: float):
        """Update a need.
        
        Args:
            need: Need to update
            value: New value for the need (0.0 to 1.0)
        """
        if need in self.needs:
            self.needs[need] = max(0.0, min(1.0, value))

class InteractionState(BaseModel):
    """Model representing the state of an interaction between the Mother and the Neural Child."""
    
    # Interaction metadata
    interaction_id: str
    timestamp: float
    age_months: float
    developmental_stage: str
    
    # Mother's state
    mother_state: Dict[str, Any] = {
        "verbal_response": "",
        "emotional_state": {
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.0,
            "anticipation": 0.0
        },
        "non_verbal_cues": [],
        "teaching_elements": []
    }
    
    # Child's state
    child_state: Dict[str, Any] = {
        "verbal_response": "",
        "emotional_state": {
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.0,
            "anticipation": 0.0
        },
        "attention_focus": None,
        "needs": {
            "physical": 1.0,
            "safety": 1.0,
            "love": 0.5,
            "esteem": 0.0,
            "self_actualization": 0.0
        }
    }
    
    # Learning outcomes
    learning_outcomes: Dict[str, Any] = {
        "vocabulary_additions": [],
        "emotional_development": {},
        "cognitive_development": {},
        "social_development": {}
    } 