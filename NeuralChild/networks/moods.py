# moods.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import Field
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("MoodsNetwork")

class MoodState(BaseNetwork):
    """A mood state that persists over time"""
    name: str
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    onset: datetime = Field(default_factory=datetime.now)
    duration: timedelta = Field(default_factory=lambda: timedelta(hours=1))
    causal_factors: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def remaining_duration(self) -> timedelta:
        """Get the remaining duration of this mood"""
        elapsed = datetime.now() - self.onset
        return max(timedelta(0), self.duration - elapsed)
    
    @property
    def is_active(self) -> bool:
        """Check if this mood is still active"""
        return self.remaining_duration > timedelta(0)
    
    def update_intensity(self, delta: float) -> None:
        """Update the intensity of this mood"""
        self.intensity = max(0.0, min(1.0, self.intensity + delta))
        
        # Adjust duration based on intensity changes
        if delta > 0:
            # Intensifying extends the duration
            self.duration += timedelta(minutes=int(30 * delta))
        elif delta < 0:
            # Weakening shortens the duration
            self.duration -= timedelta(minutes=int(15 * abs(delta)))

class MoodsNetwork(BaseNetwork):
    """
    Longer-lasting emotional states network using GRU-like processing
    
    The moods network represents the child's longer-lasting emotional states
    that persist beyond immediate emotional reactions. These moods influence
    perception, cognition, and behavior over extended periods.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 0.8,
        activation_threshold: float = 0.2,
        name: str = "Moods",
        baseline_mood: str = "neutral"
    ):
        """Initialize the moods network"""
        super().__init__(
            network_type=NetworkType.MOODS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Mood parameters
        self.baseline_mood = baseline_mood
        self.mood_stability = 0.3  # How stable moods are (increases with age)
        self.mood_inertia = 0.6  # Resistance to mood changes
        self.mood_memory_length = 48  # Hours of mood history to maintain
        
        # Available mood states
        self.available_moods = [
            "playful",    # Energetic, happy, ready to engage
            "content",    # Calm, satisfied, balanced
            "curious",    # Interested, exploring, wondering
            "irritable",  # Easily annoyed, grumpy
            "anxious",    # Worried, uneasy, distressed
            "tired",      # Low energy, sleepy
            "sad",        # Down, unhappy, low
            "excited",    # Very happy, enthusiastic
            "neutral"     # Baseline state, neither positive nor negative
        ]
        
        # Current and historical mood states
        self.active_moods: Dict[str, MoodState] = {}
        self.mood_history = deque(maxlen=self.mood_memory_length)
        
        # Initialize with neutral mood
        self.active_moods[baseline_mood] = MoodState(
            name=baseline_mood,
            intensity=0.5,
            duration=timedelta(hours=4)
        )
        
        # Mood transitions (which moods can transform into other moods)
        # This creates a more realistic mood progression
        self.mood_transitions = {
            "playful": ["content", "excited", "tired", "neutral"],
            "content": ["playful", "curious", "neutral", "tired"],
            "curious": ["playful", "content", "excited", "neutral"],
            "irritable": ["tired", "anxious", "sad", "neutral"],
            "anxious": ["irritable", "tired", "neutral", "sad"],
            "tired": ["irritable", "content", "sad", "neutral"],
            "sad": ["tired", "irritable", "anxious", "neutral"],
            "excited": ["playful", "curious", "tired", "neutral"],
            "neutral": ["content", "curious", "tired", "playful", "irritable"]
        }
        
        # Mood impacts (how moods affect other systems)
        self.mood_impacts = {
            "playful": {
                "perception": 0.2,
                "attention": 0.3,
                "emotion_bias": {"joy": 0.3, "surprise": 0.2}
            },
            "content": {
                "perception": 0.1,
                "attention": 0.1,
                "emotion_bias": {"joy": 0.2, "trust": 0.3}
            },
            "curious": {
                "perception": 0.3,
                "attention": 0.4,
                "emotion_bias": {"anticipation": 0.4, "surprise": 0.3}
            },
            "irritable": {
                "perception": -0.1,
                "attention": -0.2,
                "emotion_bias": {"anger": 0.3, "disgust": 0.2}
            },
            "anxious": {
                "perception": -0.1,
                "attention": -0.1,
                "emotion_bias": {"fear": 0.4, "anticipation": 0.2}
            },
            "tired": {
                "perception": -0.3,
                "attention": -0.4,
                "emotion_bias": {"sadness": 0.2}
            },
            "sad": {
                "perception": -0.2,
                "attention": -0.2,
                "emotion_bias": {"sadness": 0.4, "fear": 0.1}
            },
            "excited": {
                "perception": 0.2,
                "attention": 0.2,
                "emotion_bias": {"joy": 0.4, "anticipation": 0.3}
            },
            "neutral": {
                "perception": 0.0,
                "attention": 0.0,
                "emotion_bias": {}
            }
        }
        
        logger.info(f"Initialized moods network with baseline mood: {baseline_mood}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to update mood states"""
        # Clean up expired moods
        self._clean_expired_moods()
        
        # Process emotional influences from input buffer
        emotion_influences = {}
        environmental_factors = {}
        physiological_state = {}
        
        for input_item in self.input_buffer:
            data = input_item["data"]
            source = input_item.get("source", "unknown")
            
            # Extract emotional state influences
            if source == NetworkType.EMOTIONS.value and "emotional_state" in data:
                for emotion, intensity in data["emotional_state"].items():
                    emotion_influences[emotion] = intensity
            
            # Extract environmental factors
            if "environment" in data:
                for factor, value in data["environment"].items():
                    environmental_factors[factor] = value
            
            # Extract physiological states
            if "physiological" in data:
                for state, value in data["physiological"].items():
                    physiological_state[state] = value
            
            # Handle explicit mood triggers
            if "mood_trigger" in data:
                self._process_mood_trigger(data["mood_trigger"])
        
        # Update moods based on inputs
        if emotion_influences:
            self._update_from_emotions(emotion_influences)
        
        if environmental_factors:
            self._update_from_environment(environmental_factors)
        
        if physiological_state:
            self._update_from_physiology(physiological_state)
        
        # Apply mood decay and transitions
        self._apply_mood_dynamics()
        
        # Calculate overall mood state
        dominant_mood, mood_state = self._calculate_mood_state()
        
        # Add to mood history
        self.mood_history.append({
            "timestamp": datetime.now(),
            "dominant_mood": dominant_mood,
            "mood_intensities": {name: mood.intensity for name, mood in self.active_moods.items()},
            "causal_factors": {name: list(mood.causal_factors.keys()) 
                              for name, mood in self.active_moods.items()}
        })
        
        # Clear input buffer
        self.input_buffer = []
        
        return mood_state
    
    def _clean_expired_moods(self) -> None:
        """Remove moods that have expired"""
        expired = []
        for name, mood in self.active_moods.items():
            if not mood.is_active and name != self.baseline_mood:
                expired.append(name)
        
        for name in expired:
            logger.info(f"Mood '{name}' has expired")
            del self.active_moods[name]
    
    def _process_mood_trigger(self, trigger: Dict[str, Any]) -> None:
        """Process an explicit mood trigger"""
        if "mood" in trigger and "intensity" in trigger:
            mood_name = trigger["mood"]
            intensity = trigger["intensity"]
            
            if mood_name in self.available_moods:
                # Create or update mood
                if mood_name in self.active_moods:
                    self.active_moods[mood_name].update_intensity(intensity)
                    if "cause" in trigger:
                        self.active_moods[mood_name].causal_factors[trigger["cause"]] = intensity
                else:
                    # Calculate duration based on intensity and mood stability
                    hours = 1 + (intensity * 4 * self.mood_stability)
                    
                    self.active_moods[mood_name] = MoodState(
                        name=mood_name,
                        intensity=intensity,
                        duration=timedelta(hours=hours),
                        causal_factors={trigger.get("cause", "direct"): intensity}
                    )
                
                logger.info(f"Triggered mood '{mood_name}' with intensity {intensity:.2f}")
    
    def _update_from_emotions(self, emotions: Dict[str, float]) -> None:
        """Update moods based on emotional state"""
        # Emotion to mood mappings
        emotion_mood_map = {
            "joy": [("playful", 0.7), ("content", 0.5), ("excited", 0.8)],
            "sadness": [("sad", 0.8), ("tired", 0.4)],
            "anger": [("irritable", 0.7)],
            "fear": [("anxious", 0.8), ("tired", 0.3)],
            "surprise": [("curious", 0.6), ("excited", 0.4)],
            "disgust": [("irritable", 0.5)],
            "trust": [("content", 0.7), ("playful", 0.3)],
            "anticipation": [("curious", 0.7), ("excited", 0.5)]
        }
        
        # Process each emotion
        for emotion, intensity in emotions.items():
            if intensity > 0.4 and emotion in emotion_mood_map:  # Only strong emotions affect mood
                for mood_name, influence_strength in emotion_mood_map[emotion]:
                    mood_influence = intensity * influence_strength * (1.0 - self.mood_inertia)
                    
                    if mood_name in self.active_moods:
                        self.active_moods[mood_name].update_intensity(mood_influence)
                        self.active_moods[mood_name].causal_factors[f"emotion:{emotion}"] = mood_influence
                    else:
                        # Only create new mood if influence is significant
                        if mood_influence > 0.3:
                            hours = 1 + (mood_influence * 3 * self.mood_stability)
                            
                            self.active_moods[mood_name] = MoodState(
                                name=mood_name,
                                intensity=mood_influence,
                                duration=timedelta(hours=hours),
                                causal_factors={f"emotion:{emotion}": mood_influence}
                            )
    
    def _update_from_environment(self, environment: Dict[str, Any]) -> None:
        """Update moods based on environmental factors"""
        # Examples of environmental influences on mood
        if "weather" in environment:
            weather = environment["weather"]
            if weather == "sunny":
                self._influence_mood("playful", 0.2, f"environment:{weather}")
                self._influence_mood("content", 0.2, f"environment:{weather}")
            elif weather == "rainy":
                self._influence_mood("tired", 0.2, f"environment:{weather}")
                self._influence_mood("content", 0.1, f"environment:{weather}")
        
        if "noise_level" in environment:
            noise = environment["noise_level"]
            if noise > 0.7:  # High noise
                self._influence_mood("irritable", 0.3, "environment:noise")
                self._influence_mood("anxious", 0.2, "environment:noise")
            elif noise < 0.2:  # Quiet
                self._influence_mood("content", 0.2, "environment:quiet")
                self._influence_mood("tired", 0.1, "environment:quiet")
        
        if "novelty" in environment:
            novelty = environment["novelty"]
            if novelty > 0.6:
                self._influence_mood("curious", 0.3, "environment:novelty")
                self._influence_mood("excited", 0.2, "environment:novelty")
    
    def _update_from_physiology(self, physiology: Dict[str, Any]) -> None:
        """Update moods based on physiological states"""
        # Examples of physiological influences on mood
        if "hunger" in physiology:
            hunger = physiology["hunger"]
            if hunger > 0.7:  # Very hungry
                self._influence_mood("irritable", 0.4, "physiology:hunger")
                self._influence_mood("tired", 0.2, "physiology:hunger")
        
        if "fatigue" in physiology:
            fatigue = physiology["fatigue"]
            if fatigue > 0.6:
                self._influence_mood("tired", 0.5, "physiology:fatigue")
                self._influence_mood("irritable", 0.3, "physiology:fatigue")
        
        if "pain" in physiology:
            pain = physiology["pain"]
            if pain > 0.4:
                self._influence_mood("irritable", 0.4, "physiology:pain")
                self._influence_mood("sad", 0.3, "physiology:pain")
    
    def _influence_mood(self, mood_name: str, amount: float, cause: str) -> None:
        """Influence a mood by the given amount with a cause"""
        if mood_name not in self.available_moods:
            return
            
        # Apply mood inertia (resistance to change)
        effective_amount = amount * (1.0 - self.mood_inertia)
        
        if mood_name in self.active_moods:
            self.active_moods[mood_name].update_intensity(effective_amount)
            self.active_moods[mood_name].causal_factors[cause] = effective_amount
        else:
            # Only create new mood if amount is significant
            if effective_amount > 0.3:
                hours = 1 + (effective_amount * 3 * self.mood_stability)
                
                self.active_moods[mood_name] = MoodState(
                    name=mood_name,
                    intensity=effective_amount,
                    duration=timedelta(hours=hours),
                    causal_factors={cause: effective_amount}
                )
    
    def _apply_mood_dynamics(self) -> None:
        """Apply mood decay and transitions"""
        # Apply natural mood decay
        for mood in self.active_moods.values():
            if mood.name != self.baseline_mood:
                # More intense moods decay slower
                decay_rate = 0.01 * (1.0 + self.mood_stability)
                decay_amount = decay_rate * (1.0 - mood.intensity * 0.5)
                mood.update_intensity(-decay_amount)
        
        # Ensure baseline mood is always present
        if self.baseline_mood not in self.active_moods:
            self.active_moods[self.baseline_mood] = MoodState(
                name=self.baseline_mood,
                intensity=0.5,
                duration=timedelta(hours=24)
            )
        
        # Apply mood transitions
        # Strong moods might trigger other related moods
        dominant_mood, _ = self._get_dominant_mood()
        
        if dominant_mood and random.random() < 0.1:  # 10% chance of mood transition
            dominant_intensity = self.active_moods[dominant_mood].intensity
            
            if dominant_intensity > 0.7 and dominant_mood in self.mood_transitions:
                # Get possible transition targets
                targets = self.mood_transitions[dominant_mood]
                
                if targets:
                    # Select a target mood
                    target = random.choice(targets)
                    
                    # Calculate transition strength
                    transition_strength = 0.3 * dominant_intensity * (1.0 - self.mood_stability)
                    
                    # Apply transition
                    self._influence_mood(target, transition_strength, f"transition:{dominant_mood}")
    
    def _get_dominant_mood(self) -> Tuple[Optional[str], float]:
        """Get the dominant mood and its intensity"""
        if not self.active_moods:
            return None, 0.0
            
        # Find mood with highest intensity
        dominant_mood = max(self.active_moods.items(), key=lambda x: x[1].intensity)
        return dominant_mood[0], dominant_mood[1].intensity
    
    def _calculate_mood_state(self) -> Tuple[str, Dict[str, Any]]:
        """Calculate the overall mood state for output"""
        dominant_mood, dominant_intensity = self._get_dominant_mood()
        
        # Prepare mood impacts data
        impacts = {}
        for name, mood in self.active_moods.items():
            if name in self.mood_impacts and mood.intensity > 0.3:
                for target, impact in self.mood_impacts[name].items():
                    if target not in impacts:
                        impacts[target] = 0.0
                    impacts[target] += impact * mood.intensity
        
        # Calculate emotion bias
        emotion_bias = {}
        for name, mood in self.active_moods.items():
            if name in self.mood_impacts and "emotion_bias" in self.mood_impacts[name]:
                for emotion, bias in self.mood_impacts[name]["emotion_bias"].items():
                    if emotion not in emotion_bias:
                        emotion_bias[emotion] = 0.0
                    emotion_bias[emotion] += bias * mood.intensity
        
        # Build result
        result = {
            "network_activation": dominant_intensity if dominant_intensity else 0.0,
            "dominant_mood": dominant_mood,
            "dominant_intensity": dominant_intensity,
            "active_moods": {name: mood.intensity for name, mood in self.active_moods.items()},
            "mood_impacts": impacts,
            "emotion_bias": emotion_bias
        }
        
        return dominant_mood or self.baseline_mood, result
    
    def update_development(self, age_days: float) -> None:
        """Update mood dynamics based on developmental age"""
        # Mood stability increases with age (moods become less volatile)
        self.mood_stability = min(0.8, 0.3 + (age_days / 400))
        
        # Mood inertia increases with age (harder to shift moods)
        self.mood_inertia = min(0.8, 0.6 + (age_days / 500))
        
        # Mood memory increases with age
        self.mood_memory_length = min(120, int(48 + (age_days / 10)))
        
        logger.info(f"Updated mood dynamics for age {age_days:.1f} days: "
                   f"stability={self.mood_stability:.2f}, inertia={self.mood_inertia:.2f}")
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        dominant_mood, mood_data = self._calculate_mood_state()
        
        # Extract mood durations
        mood_durations = {}
        for name, mood in self.active_moods.items():
            hours_remaining = mood.remaining_duration.total_seconds() / 3600
            mood_durations[name] = hours_remaining
        
        # Analyze mood patterns from history
        mood_frequency = {}
        if self.mood_history:
            # Count frequency of dominant moods
            for entry in self.mood_history:
                mood = entry["dominant_mood"]
                if mood not in mood_frequency:
                    mood_frequency[mood] = 0
                mood_frequency[mood] += 1
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "dominant_mood": dominant_mood,
            "active_moods": {name: mood.intensity for name, mood in self.active_moods.items()},
            "mood_durations": mood_durations,
            "mood_impacts": mood_data["mood_impacts"],
            "emotion_bias": mood_data["emotion_bias"],
            "mood_stability": self.mood_stability,
            "mood_frequency": mood_frequency
        }