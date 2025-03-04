# attention.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("AttentionNetwork")

class AttentionNetwork(BaseNetwork):
    """
    Focus on relevant input network using Transformer-like processing
    
    The attention network determines what sensory and cognitive inputs
    are selected for conscious processing. It models the child's ability
    to focus on relevant stimuli and ignore distractions.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.2,
        name: str = "Attention",
        attention_span_base: float = 0.2  # Base attention span
    ):
        """Initialize the attention network"""
        super().__init__(
            network_type=NetworkType.ATTENTION,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Attention parameters
        self.attention_span_base = attention_span_base  # Base attention span (increases with development)
        self.focus_objects = []  # Currently attended objects/concepts
        self.focus_duration = {}  # How long each object/concept has been in focus
        self.distractibility = 0.7  # Initially high, decreases with development
        self.novelty_bias = 0.8  # Bias towards novel stimuli, decreases with age
        
        # Attention history
        self.focus_history = deque(maxlen=100)  # Recent focus objects
        self.attention_scores = {}  # Attention scores for different inputs
        
        logger.info(f"Initialized attention network with span base {attention_span_base} and distractibility {self.distractibility}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to determine what to focus on"""
        if not self.input_buffer:
            return {}
        
        # Extract stimuli from inputs
        all_stimuli = []
        emotional_state = {}
        
        for input_item in self.input_buffer:
            data = input_item["data"]
            
            # Extract stimuli - could be objects, words, concepts, etc.
            if "stimuli" in data and isinstance(data["stimuli"], list):
                all_stimuli.extend(data["stimuli"])
            elif "vocabulary" in data and isinstance(data["vocabulary"], list):
                all_stimuli.extend(data["vocabulary"])
            elif "concepts" in data and isinstance(data["concepts"], list):
                all_stimuli.extend(data["concepts"])
            
            # Extract emotional state which affects attention
            if "emotional_state" in data:
                for emotion, intensity in data["emotional_state"].items():
                    emotional_state[emotion] = intensity
        
        # Update attention scores for all stimuli
        self._update_attention_scores(all_stimuli, emotional_state)
        
        # Apply attention allocation algorithm (winner-take-most)
        selected_focus, attention_allocation = self._allocate_attention()
        
        # Update focus objects and durations
        for obj in self.focus_objects:
            if obj not in selected_focus:
                # Object is no longer in focus
                if obj in self.focus_duration:
                    del self.focus_duration[obj]
        
        self.focus_objects = selected_focus
        
        # Update focus durations
        for obj in self.focus_objects:
            if obj not in self.focus_duration:
                self.focus_duration[obj] = 0
            self.focus_duration[obj] += 1
        
        # Add to focus history
        for obj in self.focus_objects:
            self.focus_history.append(obj)
        
        # Calculate total attention level (how focused the child is)
        if self.focus_objects:
            # Higher score when focusing on fewer objects more intensely
            total_attention = sum(attention_allocation.values()) / len(self.focus_objects)
        else:
            total_attention = 0.0
        
        # Clear input buffer
        self.input_buffer = []
        
        return {
            "network_activation": total_attention,
            "focus_objects": self.focus_objects,
            "attention_allocation": attention_allocation
        }
    
    def _update_attention_scores(self, stimuli: List[str], emotional_state: Dict[str, float]) -> None:
        """Update attention scores for all stimuli based on various factors"""
        # Decay existing scores
        for obj in self.attention_scores:
            self.attention_scores[obj] *= 0.9  # Gradual decay
        
        # Process each stimulus
        for stimulus in stimuli:
            # Initialize if new
            if stimulus not in self.attention_scores:
                self.attention_scores[stimulus] = 0.3  # Base score
            
            # Novelty factor - increases score for new or infrequent stimuli
            novelty = self._calculate_novelty(stimulus)
            
            # Emotional salience - certain emotions increase attention
            emotional_salience = 0.0
            if "surprise" in emotional_state:
                emotional_salience += emotional_state["surprise"] * 0.3
            if "fear" in emotional_state:
                emotional_salience += emotional_state["fear"] * 0.4
            if "joy" in emotional_state:
                emotional_salience += emotional_state["joy"] * 0.2
            
            # Continuity bias - tendency to keep attending to current focus
            continuity_bias = 0.0
            if stimulus in self.focus_objects:
                continuity_bias = 0.2
            
            # Update score
            self.attention_scores[stimulus] += (novelty * self.novelty_bias) + emotional_salience + continuity_bias
            
            # Cap score
            self.attention_scores[stimulus] = min(1.0, self.attention_scores[stimulus])
    
    def _calculate_novelty(self, stimulus: str) -> float:
        """Calculate how novel a stimulus is based on recent history"""
        # Count occurrences in recent history
        occurrences = list(self.focus_history).count(stimulus)
        
        # Higher score for less frequent items
        if occurrences == 0:
            return 0.5  # Completely new
        else:
            return max(0.1, 0.5 - (occurrences * 0.05))  # Decreases with familiarity
    
    def _allocate_attention(self) -> Tuple[List[str], Dict[str, float]]:
        """Allocate attention to stimuli using a competitive process"""
        # Sort stimuli by attention score
        sorted_stimuli = sorted(
            self.attention_scores.keys(),
            key=lambda x: self.attention_scores.get(x, 0),
            reverse=True
        )
        
        # Calculate how many items can be attended to based on development
        effective_span = self.get_attention_span()
        max_items = max(1, round(effective_span * 5))  # 1-5 items based on span
        
        # Apply distractibility - chance to include a random stimulus
        selected_focus = []
        if sorted_stimuli:
            # Add top scoring stimuli
            top_stimuli = sorted_stimuli[:max_items]
            selected_focus.extend(top_stimuli)
            
            # Possibly add a distractor
            if random.random() < self.distractibility and len(sorted_stimuli) > max_items:
                # Pick a random lower-scoring stimulus
                potential_distractors = sorted_stimuli[max_items:]
                if potential_distractors:
                    distractor = random.choice(potential_distractors)
                    if distractor not in selected_focus:
                        selected_focus.append(distractor)
        
        # Calculate attention allocation (how attention is distributed)
        total_score = sum(self.attention_scores.get(obj, 0) for obj in selected_focus)
        attention_allocation = {}
        
        if total_score > 0:
            for obj in selected_focus:
                attention_allocation[obj] = self.attention_scores.get(obj, 0) / total_score
        
        return selected_focus, attention_allocation
    
    def get_attention_span(self) -> float:
        """Get the current attention span (0-1) based on development"""
        # Base span plus training progress contribution
        return min(1.0, self.attention_span_base + (self.state.training_progress * 0.8))
    
    def update_development(self, age_days: float, interactions_count: int) -> None:
        """Update developmental parameters based on age and experiences"""
        # Attention span increases with age and training
        age_factor = min(0.5, age_days / 200)  # Max +0.5 from age
        interactions_factor = min(0.3, interactions_count / 1000)  # Max +0.3 from interactions
        
        self.attention_span_base = min(0.8, 0.2 + age_factor + interactions_factor)
        
        # Distractibility decreases with age
        self.distractibility = max(0.2, 0.7 - (age_days / 400))
        
        # Novelty bias decreases with age
        self.novelty_bias = max(0.4, 0.8 - (age_days / 500))
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "attention_span": self.get_attention_span(),
            "focus_objects": self.focus_objects,
            "attention_allocation": {obj: self.attention_scores.get(obj, 0) 
                                    for obj in self.focus_objects},
            "distractibility": self.distractibility
        }
    
    def reset_focus(self) -> None:
        """Reset focus objects (e.g., for sleep or major interruption)"""
        self.focus_objects = []
        self.focus_duration = {}