# unconsciousness.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("UnconsciousnessNetwork")

class UnconsciousAssociation:
    """An unconscious association between concepts or stimuli"""
    def __init__(
        self,
        source: str,
        target: str,
        strength: float = 0.1,
        affect: str = "neutral",
        formed_at: Optional[datetime] = None
    ):
        self.source = source
        self.target = target
        self.strength = strength  # How strong the association is (0-1)
        self.affect = affect      # Emotional tone: positive, negative, neutral
        self.formed_at = formed_at or datetime.now()
        self.activation_count = 0 # Times this association has been activated
    
    def activate(self) -> float:
        """Activate this association and return its strength"""
        self.activation_count += 1
        return self.strength
    
    def __eq__(self, other):
        if not isinstance(other, UnconsciousAssociation):
            return False
        return self.source == other.source and self.target == other.target
    
    def __hash__(self):
        return hash((self.source, self.target))

class UnconsciousnessNetwork(BaseNetwork):
    """
    Deep automatic processes network using DBN-like architecture
    
    The unconsciousness network represents processes that occur below the 
    threshold of awareness. It handles automatic associations, implicit
    learning, and background processing that influences behavior and thought
    without conscious awareness.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 0.8,
        activation_threshold: float = 0.1,  # Low threshold for unconscious activation
        name: str = "Unconsciousness"
    ):
        """Initialize the unconsciousness network"""
        super().__init__(
            network_type=NetworkType.UNCONSCIOUSNESS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Unconscious parameters
        self.implicit_learning_rate = 0.05  # Rate of forming unconscious associations
        self.decay_rate = 0.01            # Rate of decay for associations
        self.emotional_influence = 0.7    # How much emotions influence associations
        self.activation_sensitivity = 0.3 # How easily associations are triggered
        
        # Associations storage
        self.associations = set()          # Set of all associations
        self.source_index = {}             # Index from source to associations
        self.target_index = {}             # Index from target to associations
        
        # Implicit memories and patterns
        self.implicit_memories = {}        # Implicitly learned memories
        self.pattern_fragments = []        # Fragments of patterns being learned
        
        # Active associations from last processing cycle
        self.active_associations = []
        
        # Developmental factors
        self.developmental_stage = "early"  # early, middle, late
        
        logger.info(f"Initialized unconsciousness network with implicit learning rate {self.implicit_learning_rate}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs at an unconscious level"""
        # Reset active associations
        self.active_associations = []
        
        if not self.input_buffer:
            # Apply background processing even without input
            return self._background_processing()
        
        # Extract all stimuli and context from inputs
        stimuli = []
        emotional_context = {}
        perceptual_context = []
        
        for input_item in self.input_buffer:
            data = input_item["data"]
            source = input_item.get("source", "unknown")
            
            # Process percepts
            if source == NetworkType.PERCEPTION.value and "percepts" in data:
                stimuli.extend(data["percepts"])
                perceptual_context = data["percepts"]
            
            # Process emotional states
            if source == NetworkType.EMOTIONS.value and "emotional_state" in data:
                emotional_context = data["emotional_state"]
            
            # Process consciousness contents
            if source == NetworkType.CONSCIOUSNESS.value and "active_contents" in data:
                for category, contents in data["active_contents"].items():
                    if isinstance(contents, list):
                        stimuli.extend(contents)
                    elif isinstance(contents, dict):
                        stimuli.extend(contents.keys())
            
            # Process direct stimuli
            if "stimuli" in data:
                stimuli.extend(data["stimuli"])
        
        # Trim duplicates
        stimuli = list(set(stimuli))
        
        # 1. Form new associations between co-occurring stimuli
        if len(stimuli) > 1:
            self._form_associations(stimuli, emotional_context)
        
        # 2. Activate existing associations based on current stimuli
        activated_associations = self._activate_associations(stimuli)
        
        # 3. Learn patterns implicitly
        if len(stimuli) > 2:
            self._implicit_pattern_learning(stimuli, perceptual_context, emotional_context)
        
        # 4. Update active associations
        self.active_associations = activated_associations
        
        # 5. Apply decay to all associations
        self._apply_association_decay()
        
        # Calculate activation level based on active associations
        if activated_associations:
            total_strength = sum(assoc.strength for assoc in activated_associations)
            activation = min(1.0, total_strength / 3.0)  # Normalize
        else:
            # Background activation
            activation = 0.2
        
        # Clear input buffer
        self.input_buffer = []
        
        # Extract triggered concepts
        triggered_concepts = []
        for assoc in activated_associations:
            triggered_concepts.append(assoc.target)
        
        return {
            "network_activation": activation,
            "active_associations_count": len(activated_associations),
            "triggered_concepts": triggered_concepts
        }
    
    def _background_processing(self) -> Dict[str, Any]:
        """Perform background processing in the absence of input"""
        # Sometimes reactivate random associations to strengthen them
        if random.random() < 0.3 and self.associations:
            # Select a few random associations to activate
            sample_size = min(3, len(self.associations))
            random_associations = random.sample(list(self.associations), sample_size)
            
            for assoc in random_associations:
                assoc.activate()
                self.active_associations.append(assoc)
            
            activation = 0.2  # Low background activation
            
            # Extract triggered concepts
            triggered_concepts = [assoc.target for assoc in random_associations]
            
            return {
                "network_activation": activation,
                "active_associations_count": len(random_associations),
                "triggered_concepts": triggered_concepts
            }
        
        return {
            "network_activation": 0.1,  # Minimal baseline activation
            "active_associations_count": 0,
            "triggered_concepts": []
        }
    
    def _form_associations(self, stimuli: List[str], emotional_context: Dict[str, float]) -> None:
        """Form new unconscious associations between co-occurring stimuli"""
        # Determine dominant emotion and intensity
        dominant_emotion = "neutral"
        dominant_intensity = 0.0
        
        for emotion, intensity in emotional_context.items():
            if intensity > dominant_intensity:
                dominant_intensity = intensity
                dominant_emotion = emotion
        
        # Convert emotion to affect
        if dominant_emotion in ["joy", "trust", "anticipation"]:
            affect = "positive"
        elif dominant_emotion in ["anger", "fear", "sadness", "disgust"]:
            affect = "negative"
        else:
            affect = "neutral"
        
        # Form associations between stimuli
        for i in range(len(stimuli)):
            for j in range(i+1, len(stimuli)):
                source = stimuli[i]
                target = stimuli[j]
                
                # Check if association already exists
                existing = False
                for assoc in self.associations:
                    if (assoc.source == source and assoc.target == target) or \
                       (assoc.source == target and assoc.target == source):
                        # Update existing association
                        assoc.strength = min(1.0, assoc.strength + self.implicit_learning_rate)
                        assoc.affect = affect  # Update affect based on current emotion
                        existing = True
                        break
                
                if not existing and random.random() < self.implicit_learning_rate:
                    # Form new association with strength influenced by emotional intensity
                    strength = self.implicit_learning_rate * (1.0 + dominant_intensity * self.emotional_influence)
                    
                    new_assoc = UnconsciousAssociation(
                        source=source,
                        target=target,
                        strength=strength,
                        affect=affect
                    )
                    
                    # Add to associations and indices
                    self.associations.add(new_assoc)
                    
                    # Update indices
                    if source not in self.source_index:
                        self.source_index[source] = []
                    self.source_index[source].append(new_assoc)
                    
                    if target not in self.target_index:
                        self.target_index[target] = []
                    self.target_index[target].append(new_assoc)
                    
                    # Also create the reverse association with slightly lower strength
                    if random.random() < 0.7:  # 70% chance for bidirectional association
                        reverse_assoc = UnconsciousAssociation(
                            source=target,
                            target=source,
                            strength=strength * 0.8,
                            affect=affect
                        )
                        
                        self.associations.add(reverse_assoc)
                        
                        if target not in self.source_index:
                            self.source_index[target] = []
                        self.source_index[target].append(reverse_assoc)
                        
                        if source not in self.target_index:
                            self.target_index[source] = []
                        self.target_index[source].append(reverse_assoc)
    
    def _activate_associations(self, stimuli: List[str]) -> List[UnconsciousAssociation]:
        """Activate associations based on current stimuli"""
        activated = []
        
        for stimulus in stimuli:
            # Check if stimulus is a source in any associations
            if stimulus in self.source_index:
                for assoc in self.source_index[stimulus]:
                    # Probability of activation based on strength and sensitivity
                    activation_prob = assoc.strength * self.activation_sensitivity
                    
                    if random.random() < activation_prob:
                        assoc.activate()
                        activated.append(assoc)
        
        return activated
    
    def _implicit_pattern_learning(self, stimuli: List[str], 
                                 perceptual_context: List[str], 
                                 emotional_context: Dict[str, float]) -> None:
        """Learn patterns implicitly from repeated exposures"""
        # Convert stimuli to a stable representation
        pattern_key = tuple(sorted(stimuli))
        
        # Check if this pattern already exists in implicit memories
        if pattern_key in self.implicit_memories:
            # Update existing pattern
            mem = self.implicit_memories[pattern_key]
            mem["count"] += 1
            
            # Update emotional association
            for emotion, intensity in emotional_context.items():
                if emotion not in mem["emotional_associations"]:
                    mem["emotional_associations"][emotion] = 0
                mem["emotional_associations"][emotion] += intensity
            
            # Increase strength based on repetition
            repetition_factor = min(10, mem["count"]) / 10  # Caps at 10 repetitions
            mem["strength"] = min(1.0, mem["strength"] + (self.implicit_learning_rate * repetition_factor))
            
        else:
            # Create new implicit memory
            self.implicit_memories[pattern_key] = {
                "stimuli": list(pattern_key),
                "count": 1,
                "first_seen": datetime.now(),
                "strength": self.implicit_learning_rate,
                "emotional_associations": {k: v for k, v in emotional_context.items() if v > 0.3}
            }
        
        # Also store pattern fragments for partial matching
        if len(stimuli) > 2:
            # Store random subsets as fragments
            for _ in range(2):  # Store 2 random fragments
                fragment_size = random.randint(2, len(stimuli)-1)
                fragment = random.sample(stimuli, fragment_size)
                
                self.pattern_fragments.append({
                    "fragment": tuple(sorted(fragment)),
                    "original": pattern_key,
                    "created": datetime.now()
                })
            
            # Limit pattern fragments to avoid memory explosion
            if len(self.pattern_fragments) > 100:
                self.pattern_fragments = self.pattern_fragments[-100:]
    
    def _apply_association_decay(self) -> None:
        """Apply decay to associations over time"""
        to_remove = []
        
        for assoc in self.associations:
            # Apply decay based on developmental stage
            if self.developmental_stage == "early":
                # Faster decay in early development
                decay_amount = self.decay_rate * 1.5
            elif self.developmental_stage == "middle":
                decay_amount = self.decay_rate
            else:
                # Slower decay in later development
                decay_amount = self.decay_rate * 0.7
            
            # Less decay for frequently activated associations
            activation_factor = min(1.0, assoc.activation_count / 10)
            effective_decay = decay_amount * (1.0 - activation_factor * 0.5)
            
            # Apply decay
            assoc.strength = max(0.0, assoc.strength - effective_decay)
            
            # Mark for removal if too weak
            if assoc.strength < 0.05:
                to_remove.append(assoc)
        
        # Remove weak associations
        for assoc in to_remove:
            self.associations.remove(assoc)
            
            # Also remove from indices
            if assoc.source in self.source_index and assoc in self.source_index[assoc.source]:
                self.source_index[assoc.source].remove(assoc)
                
            if assoc.target in self.target_index and assoc in self.target_index[assoc.target]:
                self.target_index[assoc.target].remove(assoc)
    
    def update_development(self, age_days: float) -> None:
        """Update developmental parameters based on age"""
        # Update developmental stage
        if age_days < 30:
            self.developmental_stage = "early"
        elif age_days < 180:
            self.developmental_stage = "middle"
        else:
            self.developmental_stage = "late"
        
        # Update learning parameters based on developmental stage
        if self.developmental_stage == "early":
            # Early stage: high implicit learning but poor retention
            self.implicit_learning_rate = 0.05
            self.decay_rate = 0.015
            self.emotional_influence = 0.8
            self.activation_sensitivity = 0.4
        elif self.developmental_stage == "middle":
            # Middle stage: balanced
            self.implicit_learning_rate = 0.04
            self.decay_rate = 0.01
            self.emotional_influence = 0.7
            self.activation_sensitivity = 0.35
        else:
            # Late stage: slower learning but better retention
            self.implicit_learning_rate = 0.03
            self.decay_rate = 0.005
            self.emotional_influence = 0.6
            self.activation_sensitivity = 0.3
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Get strongest associations
        sorted_assocs = sorted(self.associations, key=lambda x: x.strength, reverse=True)
        top_associations = sorted_assocs[:10]
        
        # Extract trigger words (things that activate unconscious associations)
        trigger_concepts = {}
        for assoc in sorted_assocs[:20]:  # Look at top 20 associations
            if assoc.source not in trigger_concepts:
                trigger_concepts[assoc.source] = 0
            trigger_concepts[assoc.source] += assoc.strength
        
        # Find dominant affects
        affect_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for assoc in self.active_associations:
            affect_counts[assoc.affect] += 1
        
        # Get active implicit memories
        active_memories = []
        for pattern, mem in self.implicit_memories.items():
            # Check if any part of the pattern is in our active stimuli
            active_stimuli = [a.source for a in self.active_associations] + [a.target for a in self.active_associations]
            if any(stim in active_stimuli for stim in mem["stimuli"]) and mem["strength"] > 0.3:
                active_memories.append({
                    "pattern": mem["stimuli"],
                    "strength": mem["strength"],
                    "emotions": {k: v for k, v in mem["emotional_associations"].items() 
                                if v > 0.3}  # Only include significant emotions
                })
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "active_associations": [(a.source, a.target, a.strength) for a in self.active_associations],
            "dominant_affect": max(affect_counts, key=affect_counts.get),
            "trigger_concepts": trigger_concepts,
            "association_count": len(self.associations),
            "implicit_memories": len(self.implicit_memories),
            "active_memories": active_memories,
            "developmental_stage": self.developmental_stage
        }
    
    def get_associations_for(self, concept: str) -> List[str]:
        """Get all concepts associated with the given concept"""
        results = []
        
        # Check source index
        if concept in self.source_index:
            for assoc in self.source_index[concept]:
                results.append((assoc.target, assoc.strength, assoc.affect))
        
        # Check target index for bidirectional associations
        if concept in self.target_index:
            for assoc in self.target_index[concept]:
                results.append((assoc.source, assoc.strength * 0.8, assoc.affect))
        
        # Sort by strength
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in results]  # Return just the concept names