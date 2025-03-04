# emotions.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("EmotionsNetwork")

class EmotionEvent(dict):
    """An emotion event that occurs in response to a stimulus"""
    def __init__(
        self,
        emotion: str,
        intensity: float,
        stimulus: Optional[str] = None,
        source_network: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__({
            "emotion": emotion,
            "intensity": intensity,
            "stimulus": stimulus,
            "source_network": source_network,
            "timestamp": timestamp or datetime.now()
        })

class EmotionalMemory:
    """Memory of emotional events for temporal processing"""
    def __init__(self, max_size: int = 100):
        self.events = deque(maxlen=max_size)
    
    def add_event(self, event: EmotionEvent) -> None:
        """Add an emotional event to memory"""
        self.events.append(event)
    
    def get_recent_events(self, count: int = 10) -> List[EmotionEvent]:
        """Get the most recent emotional events"""
        return list(self.events)[-count:]
    
    def get_events_by_emotion(self, emotion: str) -> List[EmotionEvent]:
        """Get events for a specific emotion"""
        return [e for e in self.events if e["emotion"] == emotion]

class EmotionsNetwork(BaseNetwork):
    """
    Affective responses network using LSTM-like processing
    
    The emotions network represents the child's emotional responses to
    stimuli. It models the development of emotional understanding and
    regulation over time.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.15,  # Lower threshold for emotions
        name: str = "Emotions",
        emotional_sensitivity: float = 0.7
    ):
        """Initialize the emotions network"""
        super().__init__(
            network_type=NetworkType.EMOTIONS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Emotional state configuration
        self.emotional_state = {
            "joy": 0.2,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.1,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.3,
            "anticipation": 0.1
        }
        
        # Emotional development parameters
        self.emotional_sensitivity = emotional_sensitivity  # How sensitive to stimuli
        self.emotional_volatility = 0.7  # How quickly emotions change (high for young children)
        self.emotional_complexity = 0.2  # Ability to experience mixed emotions
        self.emotional_regulation = 0.1  # Ability to regulate emotions (low for young children)
        
        # Emotional thresholds (when an emotion becomes consciously felt)
        self.emotional_thresholds = {emotion: 0.2 for emotion in self.emotional_state}
        
        # Emotional memory (for temporal patterns)
        self.emotional_memory = EmotionalMemory()
        
        # Emotional response patterns (develops over time)
        self.response_patterns = {}
        
        logger.info(f"Initialized emotions network with sensitivity {emotional_sensitivity}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to generate emotional responses"""
        self._apply_emotional_decay()
        
        if not self.input_buffer:
            return self._calculate_activation()
        
        for input_item in self.input_buffer:
            data = input_item.get("data", {})
            source = input_item.get("source", "unknown")
            
            # Process perceptual inputs
            if source == NetworkType.PERCEPTION.value:
                self._process_perceptual_input(data.get("percepts", []), source)
            
            # Process drive inputs
            elif source == NetworkType.DRIVES.value:
                drive_levels = data.get("drive_levels", {})
                self._process_drive_input(drive_levels, source)
            
            # Process interpersonal inputs
            interpersonal_data = data.get("interpersonal", {})
            if interpersonal_data:
                self._process_interpersonal_input(interpersonal_data, source)
            
            # Process direct emotional stimuli
            emotional_stimulus = data.get("emotional_stimulus", {})
            if emotional_stimulus:
                self._process_emotional_stimulus(emotional_stimulus, source)
            
            # Special handling for consciousness influence
            if source == NetworkType.CONSCIOUSNESS.value:
                regulation_data = data.get("regulation", {})
                if regulation_data:
                    self._apply_emotional_regulation(regulation_data)
        
        # Apply emotional complexity (blending of emotions)
        self._apply_emotional_complexity()
        
        # Clear input buffer
        self.input_buffer = []
        
        return self._calculate_activation()
    
    def _calculate_activation(self) -> Dict[str, Any]:
        """Calculate overall emotional activation"""
        # Get the dominant emotion and intensity
        dominant_emotion, dominant_intensity = self._get_dominant_emotion()
        
        # Calculate overall activation as sum of emotional intensities
        total_intensity = sum(self.emotional_state.values())
        
        # Normalize to 0-1 range, with bias toward higher activation
        normalized_activation = min(1.0, total_intensity / 3.0)
        
        # Record emotional state
        for emotion, intensity in self.emotional_state.items():
            if intensity > self.emotional_thresholds[emotion]:
                # Only record emotions above threshold
                self.emotional_memory.add_event(EmotionEvent(
                    emotion=emotion,
                    intensity=intensity,
                    stimulus=None,  # No specific stimulus in this case
                    source_network=None
                ))
        
        return {
            "network_activation": normalized_activation,
            "emotional_state": self.emotional_state.copy(),
            "dominant_emotion": dominant_emotion,
            "dominant_intensity": dominant_intensity
        }
    
    def _apply_emotional_decay(self) -> None:
        """Apply natural decay to emotions over time"""
        # Different emotions decay at different rates
        decay_rates = {
            "joy": 0.05,
            "sadness": 0.02,  # Sadness lingers longer
            "anger": 0.04,
            "fear": 0.03,
            "surprise": 0.08,  # Surprise fades quickly
            "disgust": 0.04,
            "trust": 0.01,  # Trust decays very slowly
            "anticipation": 0.05
        }
        
        # Apply decay to each emotion
        for emotion, intensity in self.emotional_state.items():
            decay_amount = decay_rates.get(emotion, 0.05) * (1.0 - self.emotional_regulation)
            self.emotional_state[emotion] = max(0.0, intensity - decay_amount)
    
    def _process_perceptual_input(self, percepts: List[str], source: str) -> None:
        """Process perceptual input to generate emotional responses"""
        # Simple keyword-based emotional responses
        emotional_keywords = {
            "joy": ["happy", "play", "fun", "laugh", "smile", "love"],
            "sadness": ["sad", "cry", "alone", "hurt", "lost"],
            "anger": ["angry", "mad", "bad", "wrong", "no"],
            "fear": ["scary", "afraid", "dark", "loud", "strange"],
            "surprise": ["wow", "new", "sudden", "unexpected"],
            "disgust": ["yucky", "dirty", "gross", "smelly"],
            "trust": ["mommy", "daddy", "family", "safe", "help"],
            "anticipation": ["soon", "wait", "going", "will", "later"]
        }
        
        # Check percepts against emotional keywords
        for percept in percepts:
            percept_lower = percept.lower()
            
            for emotion, keywords in emotional_keywords.items():
                if any(keyword in percept_lower for keyword in keywords):
                    # Generate emotional response with some randomness
                    intensity = 0.3 + (random.random() * 0.3)
                    intensity *= self.emotional_sensitivity
                    
                    # Record the emotional event
                    self.emotional_memory.add_event(EmotionEvent(
                        emotion=emotion,
                        intensity=intensity,
                        stimulus=percept,
                        source_network=source
                    ))
                    
                    # Update emotional state
                    self._update_emotion(emotion, intensity)
    
    def _process_drive_input(self, drive_levels: Dict[str, float], source: str) -> None:
        """Process drive inputs to generate emotional responses"""
        # Map drives to emotions
        drive_emotion_map = {
            "physiological": [("joy", 0.3), ("anger", 0.4)],  # Satisfied/unsatisfied
            "safety": [("fear", 0.5), ("trust", 0.4)],  # Threatened/secure
            "attachment": [("sadness", 0.4), ("joy", 0.5), ("trust", 0.5)],  # Lonely/connected
            "exploration": [("anticipation", 0.4), ("surprise", 0.3)],  # Discovery
            "rest": [("joy", 0.2), ("anger", 0.3)]  # Rested/tired
        }
        
        # Process each drive
        for drive, level in drive_levels.items():
            if drive in drive_emotion_map:
                for emotion, base_intensity in drive_emotion_map[drive]:
                    # High drive levels (unsatisfied needs) generate negative emotions
                    if emotion in ["anger", "fear", "sadness"] and level > 0.6:
                        intensity = base_intensity * level * self.emotional_sensitivity
                        self._update_emotion(emotion, intensity)
                        
                        # Record emotional event
                        self.emotional_memory.add_event(EmotionEvent(
                            emotion=emotion,
                            intensity=intensity,
                            stimulus=f"drive:{drive}",
                            source_network=source
                        ))
                    
                    # Low drive levels (satisfied needs) generate positive emotions
                    elif emotion in ["joy", "trust"] and level < 0.4:
                        intensity = base_intensity * (1.0 - level) * self.emotional_sensitivity
                        self._update_emotion(emotion, intensity)
                        
                        # Record emotional event
                        self.emotional_memory.add_event(EmotionEvent(
                            emotion=emotion,
                            intensity=intensity,
                            stimulus=f"drive:{drive}",
                            source_network=source
                        ))
    
    def _process_interpersonal_input(self, interpersonal_data: Dict[str, Any], source: str) -> None:
        """Process interpersonal interactions (especially with mother)"""
        if "mother_emotion" in interpersonal_data:
            # Emotional contagion - child picks up mother's emotions
            mother_emotion = interpersonal_data["mother_emotion"]
            contagion_strength = 0.4  # How strongly mother's emotions transfer
            intensity = interpersonal_data.get("intensity", 0.5)  # FIXED! Always define with a default
            
            if mother_emotion in self.emotional_state:
                # Direct emotional contagion
                self._update_emotion(mother_emotion, intensity * contagion_strength)
            
                # Record emotional event
                self.emotional_memory.add_event(EmotionEvent(
                    emotion=mother_emotion,
                    intensity=intensity,
                    stimulus="mother",
                    source_network=source
                ))
        
        if "mother_actions" in interpersonal_data:
            actions = interpersonal_data["mother_actions"]
            
            # Process nurturing actions
            nurturing_actions = ["hug", "kiss", "cuddle", "hold", "comfort"]
            if any(action in " ".join(actions).lower() for action in nurturing_actions):
                self._update_emotion("joy", 0.4 * self.emotional_sensitivity)
                self._update_emotion("trust", 0.5 * self.emotional_sensitivity)
            
            # Process corrective actions
            corrective_actions = ["no", "stop", "don't", "correct", "scold"]
            if any(action in " ".join(actions).lower() for action in corrective_actions):
                self._update_emotion("sadness", 0.2 * self.emotional_sensitivity)
                self._update_emotion("surprise", 0.3 * self.emotional_sensitivity)
    
    def _process_emotional_stimulus(self, stimulus: Dict[str, Any], source: str) -> None:
        """Process direct emotional stimuli"""
        if "emotion" in stimulus and "intensity" in stimulus:
            emotion = stimulus["emotion"]
            intensity = stimulus["intensity"] * self.emotional_sensitivity
            
            if emotion in self.emotional_state:
                self._update_emotion(emotion, intensity)
                
                # Record emotional event
                self.emotional_memory.add_event(EmotionEvent(
                    emotion=emotion,
                    intensity=intensity,
                    stimulus=stimulus.get("stimulus", "unknown"),
                    source_network=source
                ))
    
    def _apply_emotional_regulation(self, regulation_data: Dict[str, Any]) -> None:
        """Apply emotional regulation from consciousness"""
        if "target_emotion" in regulation_data and "regulation_strength" in regulation_data:
            target = regulation_data["target_emotion"]
            strength = regulation_data["regulation_strength"] * self.emotional_regulation
            
            if target in self.emotional_state and strength > 0:
                # Reduce the target emotion
                self.emotional_state[target] = max(0.0, self.emotional_state[target] - strength)
    
    def _apply_emotional_complexity(self) -> None:
        """Apply emotional complexity (blending of emotions)"""
        if self.emotional_complexity < 0.3:
            # Young children have simple emotions - intensify the dominant one
            dominant, _ = self._get_dominant_emotion()
            if dominant:
                for emotion in self.emotional_state:
                    if emotion != dominant:
                        # Suppress non-dominant emotions
                        self.emotional_state[emotion] *= (1.0 - 0.3 + self.emotional_complexity)
        else:
            # Older children can have more complex emotional blends
            # No additional processing needed as multiple emotions can coexist
            pass
    
    def _update_emotion(self, emotion: str, intensity: float) -> None:
        """Update an emotion with the given intensity"""
        if emotion in self.emotional_state:
            # Apply volatility - how rapidly emotions change
            current = self.emotional_state[emotion]
            change = intensity * self.emotional_volatility
            
            # Update with some randomness
            self.emotional_state[emotion] = min(1.0, current + change + (random.random() * 0.1 * change))
    
    def _get_dominant_emotion(self) -> Tuple[Optional[str], float]:
        """Get the dominant emotion and its intensity"""
        if not self.emotional_state:
            return None, 0.0
            
        dominant = max(self.emotional_state.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
    
    def update_development(self, age_days: float) -> None:
        """Update emotional development based on age"""
        # Emotional volatility decreases with age (emotions become more stable)
        self.emotional_volatility = max(0.3, 0.7 - (age_days / 400))
        
        # Emotional complexity increases with age (mixed emotions become possible)
        self.emotional_complexity = min(0.9, 0.2 + (age_days / 300))
        
        # Emotional regulation increases with age (better control of emotions)
        self.emotional_regulation = min(0.8, 0.1 + (age_days / 250))
        
        # Emotional thresholds change with age
        for emotion in self.emotional_thresholds:
            # Most emotional thresholds increase (takes more to trigger an emotion)
            if emotion in ["anger", "fear", "sadness"]:
                self.emotional_thresholds[emotion] = min(0.4, 0.2 + (age_days / 600))
            # Except positive emotions which decrease (more easily felt)
            elif emotion in ["joy", "trust"]:
                self.emotional_thresholds[emotion] = max(0.1, 0.2 - (age_days / 800))
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        dominant_emotion, dominant_intensity = self._get_dominant_emotion()
        
        # Get emotions above threshold
        active_emotions = {
            emotion: intensity for emotion, intensity in self.emotional_state.items()
            if intensity > self.emotional_thresholds[emotion]
        }
        
        # Recent emotional patterns (last 10 events)
        recent_events = self.emotional_memory.get_recent_events(10)
        recent_pattern = {}
        for event in recent_events:
            emotion = event["emotion"]
            if emotion not in recent_pattern:
                recent_pattern[emotion] = []
            recent_pattern[emotion].append(event["intensity"])
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "emotional_state": self.emotional_state.copy(),
            "dominant_emotion": dominant_emotion,
            "dominant_intensity": dominant_intensity,
            "active_emotions": active_emotions,
            "emotional_stability": 1.0 - self.emotional_volatility,
            "emotional_complexity": self.emotional_complexity,
            "emotional_regulation": self.emotional_regulation,
            "recent_pattern": recent_pattern
        }