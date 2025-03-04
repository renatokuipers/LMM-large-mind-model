# consciousness.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("ConsciousnessNetwork")

class ConsciousnessNetwork(BaseNetwork):
    """
    Awareness integration network using RNN-like processing
    
    The consciousness network integrates information from various neural networks
    to form a coherent sense of self and awareness. It represents the child's
    emerging self-awareness and conscious experience.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.3,  # Higher threshold for consciousness
        name: str = "Consciousness",
        self_awareness_base: float = 0.1  # Base self-awareness level
    ):
        """Initialize the consciousness network"""
        super().__init__(
            network_type=NetworkType.CONSCIOUSNESS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Consciousness parameters
        self.self_awareness = self_awareness_base  # Self-awareness level
        self.integration_capacity = 0.2  # Ability to integrate information
        self.continuity_strength = 0.3  # Continuity of consciousness
        
        # Active contents of consciousness
        self.active_contents = {
            "perceptions": [],
            "emotions": {},
            "thoughts": [],
            "self_representations": []  # Emerging concepts of self
        }
        
        # Consciousness stream (temporal record of conscious states)
        self.consciousness_stream = deque(maxlen=20)  # Recent conscious states
        
        # Network integration weights (how strongly each network influences consciousness)
        self.network_weights = {
            NetworkType.PERCEPTION.value: 0.8,  # Strong influence initially
            NetworkType.EMOTIONS.value: 0.7,     # Strong influence initially
            NetworkType.ATTENTION.value: 0.6,    # Moderate influence
            NetworkType.THOUGHTS.value: 0.3,     # Weaker initially, grows with development
            NetworkType.ARCHETYPES.value: 0.1,   # Weak conscious influence
            NetworkType.INSTINCTS.value: 0.2,    # Weak conscious influence
            NetworkType.DRIVES.value: 0.4,       # Moderate influence
            NetworkType.MOODS.value: 0.5         # Moderate influence
        }
        
        logger.info(f"Initialized consciousness network with self-awareness {self_awareness_base}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """
        Process inputs from other networks to create the current state of consciousness
        """
        if not self.input_buffer:
            return {}
        
        # Extract input data from various networks
        perceptions = []
        thoughts = []
        emotions = {}
        attention_focus = []
        
        network_activations = {}
        
        for input_item in self.input_buffer:
            data = input_item["data"]
            source = input_item.get("source", "unknown")
            
            # Track network activations
            if "activation" in data:
                network_activations[source] = data["activation"]
            
            # Process perceptions
            if source == NetworkType.PERCEPTION.value and "percepts" in data:
                perceptions.extend(data["percepts"])
            
            # Process emotions
            if source == NetworkType.EMOTIONS.value and "emotional_state" in data:
                for emotion, intensity in data["emotional_state"].items():
                    emotions[emotion] = intensity
            
            # Process thoughts
            if source == NetworkType.THOUGHTS.value and "thoughts" in data:
                thoughts.extend(data["thoughts"])
            
            # Process attention focus
            if source == NetworkType.ATTENTION.value and "focus_objects" in data:
                attention_focus.extend(data["focus_objects"])
        
        # Apply attention filter - consciousness primarily contains what's attended to
        filtered_perceptions = []
        for percept in perceptions:
            # Items in attention focus are more likely to enter consciousness
            if percept in attention_focus:
                filtered_perceptions.append(percept)
            elif random.random() < 0.3:  # Small chance for unattended items
                filtered_perceptions.append(percept)
        
        # Apply emotional filter - strong emotions color consciousness
        dominant_emotions = {}
        for emotion, intensity in emotions.items():
            if intensity > 0.4:  # Only strong emotions enter consciousness fully
                dominant_emotions[emotion] = intensity
        
        # Process self-representations
        self_representations = self._process_self_awareness(
            filtered_perceptions, dominant_emotions, thoughts
        )
        
        # Update active contents of consciousness
        self.active_contents = {
            "perceptions": filtered_perceptions,
            "emotions": dominant_emotions,
            "thoughts": thoughts,
            "self_representations": self_representations
        }
        
        # Add to consciousness stream (temporal record)
        self.consciousness_stream.append({
            "timestamp": datetime.now(),
            "contents": self.active_contents.copy(),
            "integration_level": self.integration_capacity
        })
        
        # Calculate overall consciousness level
        # Integration of network activations weighted by their influence on consciousness
        weighted_activations = 0
        weight_sum = 0
        
        for network, activation in network_activations.items():
            weight = self.network_weights.get(network, 0.1)
            weighted_activations += activation * weight
            weight_sum += weight
        
        if weight_sum > 0:
            consciousness_level = weighted_activations / weight_sum
        else:
            consciousness_level = 0.0
        
        # Apply integration capacity and continuity
        if self.consciousness_stream and len(self.consciousness_stream) > 1:
            # Continuity from previous state
            prev_state = self.consciousness_stream[-2]
            consciousness_level = (consciousness_level * (1 - self.continuity_strength) + 
                                  self.state.activation * self.continuity_strength)
        
        consciousness_level *= self.integration_capacity
        
        # Clear input buffer
        self.input_buffer = []
        
        return {
            "network_activation": consciousness_level,
            "active_contents": self.active_contents,
            "self_awareness": self.self_awareness
        }
    
    def _process_self_awareness(
        self, 
        perceptions: List[str], 
        emotions: Dict[str, float], 
        thoughts: List[str]
    ) -> List[str]:
        """
        Process self-awareness - the child's emerging sense of self
        
        This function generates self-representations based on current experiences
        and developmental stage.
        """
        self_representations = []
        
        # Only process self-awareness if development is sufficient
        if self.self_awareness < 0.2:
            return self_representations
        
        # Basic bodily self-awareness (early development)
        bodily_terms = ["me", "my", "mine", "body", "hands", "face", "mouth"]
        for term in bodily_terms:
            if term in perceptions and random.random() < self.self_awareness:
                self_representations.append(f"physical:{term}")
        
        # Emotional self-awareness (develops later)
        if self.self_awareness > 0.3:
            for emotion, intensity in emotions.items():
                if intensity > 0.5 and random.random() < self.self_awareness:
                    self_representations.append(f"feeling:{emotion}")
        
        # Cognitive self-awareness (develops latest)
        if self.self_awareness > 0.5:
            cognitive_terms = ["think", "know", "want", "like", "need"]
            for term in cognitive_terms:
                if term in thoughts and random.random() < self.self_awareness:
                    self_representations.append(f"thinking:{term}")
        
        return self_representations
    
    def update_development(self, age_days: float, vocabulary_size: int) -> None:
        """
        Update developmental parameters based on age and language development
        
        Self-awareness and integration capacity increase with development.
        """
        # Self-awareness increases with age
        age_factor = min(0.6, age_days / 200)  # Max +0.6 from age
        
        # Language influences self-awareness development
        language_factor = min(0.3, vocabulary_size / 1000)  # Max +0.3 from language
        
        self.self_awareness = min(0.9, 0.1 + age_factor + language_factor)
        
        # Integration capacity increases with age and self-awareness
        self.integration_capacity = min(0.9, 0.2 + (age_days / 300) + (self.self_awareness * 0.2))
        
        # Continuity strengthens with development
        self.continuity_strength = min(0.8, 0.3 + (age_days / 400))
        
        # Network weights shift with development
        # Lower-level processes dominate early, higher-level later
        if age_days > 30:
            self.network_weights[NetworkType.THOUGHTS.value] = min(0.7, 0.3 + (age_days / 200))
            self.network_weights[NetworkType.ARCHETYPES.value] = min(0.5, 0.1 + (age_days / 300))
        
        if age_days > 100:
            # Balance shifts from perception/emotion to thought/attention
            self.network_weights[NetworkType.PERCEPTION.value] = max(0.5, 0.8 - (age_days / 500))
            self.network_weights[NetworkType.EMOTIONS.value] = max(0.5, 0.7 - (age_days / 600))
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Extract the dominant emotion
        dominant_emotion = None
        dominant_intensity = 0.0
        
        for emotion, intensity in self.active_contents["emotions"].items():
            if intensity > dominant_intensity:
                dominant_intensity = intensity
                dominant_emotion = emotion
        
        # Count frequency of each perception in consciousness stream
        perception_frequency = {}
        for state in self.consciousness_stream:
            for percept in state["contents"]["perceptions"]:
                perception_frequency[percept] = perception_frequency.get(percept, 0) + 1
        
        # Get most stable conscious contents (appear most frequently)
        stable_contents = sorted(
            perception_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "self_awareness": self.self_awareness,
            "integration_capacity": self.integration_capacity,
            "dominant_emotion": dominant_emotion,
            "active_perceptions": self.active_contents["perceptions"],
            "active_thoughts": self.active_contents["thoughts"],
            "self_representations": self.active_contents["self_representations"],
            "stable_contents": [item[0] for item in stable_contents]
        }
    
    def get_consciousness_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current state of consciousness"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_contents": self.active_contents,
            "self_awareness": self.self_awareness,
            "activation_level": self.state.activation,
            "confidence": self.state.confidence
        }