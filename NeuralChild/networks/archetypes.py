# archetypes.py
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import random
import numpy as np

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("ArchetypesNetwork")

class ArchetypesNetwork(BaseNetwork):
    """
    Deep-seated patterns network using GRU-like processing
    
    Archetypes represent fundamental psychological patterns that form the 
    foundation of personality and behavior. This network models the instinctual
    and collective patterns that influence higher-level cognitive processes.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.15,  # Lower threshold for archetypes
        name: str = "Archetypes"
    ):
        """Initialize the archetypes network"""
        super().__init__(
            network_type=NetworkType.ARCHETYPES,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Initialize archetype patterns
        self.archetypes = {
            "caregiver": {
                "activation": 0.0,
                "development": 0.1,
                "associations": ["nurture", "protect", "comfort", "mother", "father"]
            },
            "explorer": {
                "activation": 0.0,
                "development": 0.1,
                "associations": ["curiosity", "discover", "adventure", "new"]
            },
            "creator": {
                "activation": 0.0,
                "development": 0.1,
                "associations": ["make", "build", "imagine", "play"]
            },
            "seeker": {
                "activation": 0.0,
                "development": 0.1,
                "associations": ["meaning", "purpose", "understand", "why"]
            },
            "survivor": {
                "activation": 0.0,
                "development": 0.1,
                "associations": ["safety", "fear", "protect", "danger"]
            }
        }
        
        # Archetype temporal memory (recent activations)
        self.archetype_memory = {name: [] for name in self.archetypes}
        
        logger.info(f"Initialized archetypes network with {len(self.archetypes)} fundamental patterns")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to activate relevant archetypes"""
        if not self.input_buffer:
            return {}
        
        # Extract emotional and contextual data from inputs
        emotional_data = {}
        contextual_keywords = []
        
        for input_item in self.input_buffer:
            data = input_item.get("data", {})
            source = input_item.get("source", "unknown")
            
            # Extract emotional information
            emotional_state = data.get("emotional_state", {})
            for emotion, intensity in emotional_state.items():
                if emotion not in emotional_data:
                    emotional_data[emotion] = []
                emotional_data[emotion].append(intensity)
            
            # Extract contextual keywords
            context = data.get("context", "")
            if isinstance(context, str):
                words = context.lower().split()
                contextual_keywords.extend(words)
            
            # Extract direct vocabulary
            vocabulary = data.get("vocabulary", [])
            if isinstance(vocabulary, list):
                contextual_keywords.extend(vocabulary)
        
        # Process emotional data (average intensities)
        emotions = {emotion: sum(values) / len(values) 
                   for emotion, values in emotional_data.items()}
        
        # Activate archetypes based on contextual and emotional inputs
        archetype_activations = {}
        
        for name, archetype in self.archetypes.items():
            # Calculate activation based on keyword associations
            keyword_match = 0.0
            if contextual_keywords:
                for keyword in contextual_keywords:
                    if keyword in archetype["associations"]:
                        keyword_match += 0.1
                
                # Normalize to avoid excessive activation from many words
                keyword_match = min(0.5, keyword_match)
            
            # Emotional influence on activation
            emotional_influence = 0.0
            
            # Different archetypes respond to different emotions
            if name == "caregiver" and "trust" in emotions:
                emotional_influence += emotions["trust"] * 0.3
            elif name == "explorer" and "anticipation" in emotions:
                emotional_influence += emotions["anticipation"] * 0.3
            elif name == "creator" and "joy" in emotions:
                emotional_influence += emotions["joy"] * 0.3
            elif name == "seeker" and "surprise" in emotions:
                emotional_influence += emotions["surprise"] * 0.3
            elif name == "survivor" and "fear" in emotions:
                emotional_influence += emotions["fear"] * 0.3
            
            # Combined activation with some randomness for emergent behavior
            activation = keyword_match + emotional_influence
            
            # Add some variability (to simulate the non-deterministic nature of archetypes)
            activation += random.uniform(0.0, 0.1) * archetype["development"]
            
            # Update archetype activation
            archetype["activation"] = min(1.0, activation)
            
            # Record activation for temporal patterns
            self.archetype_memory[name].append(archetype["activation"])
            if len(self.archetype_memory[name]) > 10:  # Keep last 10 activations
                self.archetype_memory[name].pop(0)
            
            archetype_activations[name] = archetype["activation"]
        
        # Calculate overall network activation as weighted sum of archetype activations
        total_activation = sum(activation for activation in archetype_activations.values())
        count = max(1, len(archetype_activations))
        
        # Clean input buffer
        self.input_buffer = []
        
        return {
            "network_activation": total_activation / count,
            "archetype_activations": archetype_activations
        }
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Get the currently dominant archetype
        dominant_archetype = max(self.archetypes.keys(), 
                                key=lambda a: self.archetypes[a]["activation"])
        
        # Calculate average activation per archetype over recent history
        temporal_patterns = {}
        for name, history in self.archetype_memory.items():
            if history:
                temporal_patterns[name] = sum(history) / len(history)
            else:
                temporal_patterns[name] = 0.0
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "dominant_archetype": dominant_archetype,
            "archetype_activations": {name: archetype["activation"] 
                                      for name, archetype in self.archetypes.items()},
            "temporal_patterns": temporal_patterns
        }
    
    def update_development(self, age_days: float, emotional_state: Dict[str, float]) -> None:
        """Update the development of archetypes based on age and emotional state"""
        # Archetypes develop at different rates based on age
        
        # Early development focuses on caregiver and survivor archetypes
        if age_days < 30:
            self.archetypes["caregiver"]["development"] = min(0.5, 0.1 + (age_days / 100))
            self.archetypes["survivor"]["development"] = min(0.5, 0.1 + (age_days / 120))
        
        # Explorer and creator develop more in early to middle childhood
        elif age_days < 100:
            self.archetypes["explorer"]["development"] = min(0.7, 0.1 + ((age_days - 30) / 100))
            self.archetypes["creator"]["development"] = min(0.7, 0.1 + ((age_days - 30) / 120))
        
        # Seeker develops more in later stages
        else:
            self.archetypes["seeker"]["development"] = min(0.9, 0.1 + ((age_days - 100) / 200))
        
        # Emotional experiences influence archetype development
        for emotion, intensity in emotional_state.items():
            if emotion == "trust" and intensity > 0.5:
                self.archetypes["caregiver"]["development"] += 0.001
            elif emotion == "joy" and intensity > 0.5:
                self.archetypes["creator"]["development"] += 0.001
            elif emotion == "anticipation" and intensity > 0.5:
                self.archetypes["explorer"]["development"] += 0.001
            elif emotion == "surprise" and intensity > 0.5:
                self.archetypes["seeker"]["development"] += 0.001
            elif emotion == "fear" and intensity > 0.5:
                self.archetypes["survivor"]["development"] += 0.001
        
        # Cap development values
        for archetype in self.archetypes.values():
            archetype["development"] = min(1.0, archetype["development"])