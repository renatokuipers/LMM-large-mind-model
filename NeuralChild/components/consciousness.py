"""
Consciousness component for the NeuralChild project.

This module defines the ConsciousnessComponent class that handles self-awareness,
attention, integration of information from other components, and the development
of a coherent sense of self.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union, DefaultDict
import random
import math
import time
from collections import defaultdict, deque
from enum import Enum
from pydantic import BaseModel, Field

from .base import NeuralComponent, ConnectionType
from ..config import DevelopmentalStage


class AttentionFocus(str, Enum):
    """Types of attention focus."""
    EXTERNAL = "external"  # Focus on external stimuli
    INTERNAL = "internal"  # Focus on internal states
    DIVIDED = "divided"  # Divided between multiple stimuli
    SUSTAINED = "sustained"  # Sustained on a single stimulus
    SELECTIVE = "selective"  # Selectively attending to specific stimuli
    UNFOCUSED = "unfocused"  # Unfocused/diffuse attention


class ConsciousnessState(str, Enum):
    """States of consciousness."""
    ASLEEP = "asleep"  # Sleeping/unconscious
    DROWSY = "drowsy"  # Partially conscious
    ALERT = "alert"  # Fully conscious
    HYPERALERT = "hyperalert"  # Heightened consciousness
    ABSORBED = "absorbed"  # Absorbed in thought/activity
    REFLECTIVE = "reflective"  # Introspective state


class IdentityAspect(BaseModel):
    """Representation of an aspect of identity."""
    name: str
    strength: float = Field(default=0.0, ge=0.0, le=1.0)  # How strongly held
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)  # Positive/negative
    sources: List[str] = Field(default_factory=list)  # Where it came from
    first_emergence: int = 0  # Age in days when first emerged
    last_reinforced: int = 0  # Age in days when last reinforced


class SelfConcept(BaseModel):
    """Representation of the self-concept."""
    identity_aspects: Dict[str, IdentityAspect] = Field(default_factory=dict)
    core_traits: Dict[str, float] = Field(default_factory=dict)
    autobiographical_memories: List[str] = Field(default_factory=list)  # Memory IDs
    complexity: float = Field(default=0.0, ge=0.0, le=1.0)  # Complexity of self-concept
    coherence: float = Field(default=0.0, ge=0.0, le=1.0)  # Coherence of self-concept
    
    class Config:
        arbitrary_types_allowed = True


class AwarenessLevel(str, Enum):
    """Levels of awareness for different domains."""
    UNCONSCIOUS = "unconscious"  # Unaware
    MINIMAL = "minimal"  # Minimal awareness
    BASIC = "basic"  # Basic awareness
    INTERMEDIATE = "intermediate"  # Intermediate awareness
    ADVANCED = "advanced"  # Advanced awareness
    REFLECTIVE = "reflective"  # Reflective awareness with metacognition


class AwarenessState(BaseModel):
    """State of awareness across different domains."""
    self_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    emotional_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    social_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    environmental_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    metacognitive_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    temporal_awareness: AwarenessLevel = AwarenessLevel.UNCONSCIOUS
    
    class Config:
        arbitrary_types_allowed = True


class ConsciousnessEvent(BaseModel):
    """An event in consciousness."""
    timestamp: float
    event_type: str
    content: Dict[str, Any]
    source_component: Optional[str] = None
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    duration: float = 0.0  # Duration in seconds
    
    class Config:
        arbitrary_types_allowed = True


class Stream(BaseModel):
    """A stream of consciousness events."""
    events: List[ConsciousnessEvent] = Field(default_factory=list)
    capacity: int = 100  # Maximum number of events to retain
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_event(self, event: ConsciousnessEvent) -> None:
        """Add an event to the stream."""
        self.events.append(event)
        if len(self.events) > self.capacity:
            self.events.pop(0)


class ConsciousnessComponent(NeuralComponent):
    """
    Component that handles consciousness and self-awareness.
    
    This component integrates information from other components,
    develops a sense of self, and manages attention and awareness.
    """
    
    def __init__(
        self,
        development_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
        component_id: Optional[str] = None
    ):
        """
        Initialize the consciousness component.
        
        Args:
            development_stage: Current developmental stage
            component_id: Optional ID (generated if not provided)
        """
        super().__init__(
            name="Consciousness",
            activation_threshold=0.2,
            activation_decay_rate=0.05,
            learning_rate=0.05,
            development_stage=development_stage,
            component_id=component_id
        )
        
        # Current state of consciousness
        self.consciousness_state: ConsciousnessState = ConsciousnessState.ALERT
        
        # Current focus of attention
        self.attention_focus: AttentionFocus = AttentionFocus.UNFOCUSED
        
        # Attended stimuli/content (what's currently being attended to)
        self.attended_content: Dict[str, Any] = {}
        
        # Awareness state
        self.awareness = AwarenessState()
        
        # Self-concept (develops over time)
        self.self_concept = SelfConcept()
        
        # Stream of consciousness
        self.stream = Stream()
        
        # Working stream (temporary, higher-priority events)
        self.working_stream = Stream(capacity=10)
        
        # Component integration
        self.component_states: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Set initial awareness levels based on developmental stage
        self._initialize_awareness()
        
        # Consciousness parameters
        self.metadata.update({
            "attention_span_seconds": 5.0,  # How long attention is sustained
            "attention_shift_probability": 0.2,  # Probability of attention shifting
            "introspection_probability": 0.1,  # Probability of introspection
            "self_reflection_interval": 3600,  # Seconds between self-reflection
            "event_retention_factor": 0.8,  # How well events are retained
            "identity_formation_rate": 0.01,  # Rate of identity formation
            "integration_factor": 0.7,  # How well information is integrated
            "metacognition_threshold": 0.6,  # Threshold for metacognition
            "last_self_reflection": 0.0,  # Time of last self-reflection
        })
    
    def _initialize_awareness(self) -> None:
        """Initialize awareness levels based on developmental stage."""
        awareness_by_stage = {
            DevelopmentalStage.PRENATAL: {
                "self_awareness": AwarenessLevel.UNCONSCIOUS,
                "emotional_awareness": AwarenessLevel.UNCONSCIOUS,
                "social_awareness": AwarenessLevel.UNCONSCIOUS,
                "environmental_awareness": AwarenessLevel.MINIMAL,
                "metacognitive_awareness": AwarenessLevel.UNCONSCIOUS,
                "temporal_awareness": AwarenessLevel.UNCONSCIOUS
            },
            DevelopmentalStage.INFANCY: {
                "self_awareness": AwarenessLevel.MINIMAL,
                "emotional_awareness": AwarenessLevel.BASIC,
                "social_awareness": AwarenessLevel.MINIMAL,
                "environmental_awareness": AwarenessLevel.BASIC,
                "metacognitive_awareness": AwarenessLevel.UNCONSCIOUS,
                "temporal_awareness": AwarenessLevel.MINIMAL
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                "self_awareness": AwarenessLevel.BASIC,
                "emotional_awareness": AwarenessLevel.INTERMEDIATE,
                "social_awareness": AwarenessLevel.BASIC,
                "environmental_awareness": AwarenessLevel.INTERMEDIATE,
                "metacognitive_awareness": AwarenessLevel.MINIMAL,
                "temporal_awareness": AwarenessLevel.BASIC
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                "self_awareness": AwarenessLevel.INTERMEDIATE,
                "emotional_awareness": AwarenessLevel.INTERMEDIATE,
                "social_awareness": AwarenessLevel.INTERMEDIATE,
                "environmental_awareness": AwarenessLevel.INTERMEDIATE,
                "metacognitive_awareness": AwarenessLevel.BASIC,
                "temporal_awareness": AwarenessLevel.INTERMEDIATE
            },
            DevelopmentalStage.ADOLESCENCE: {
                "self_awareness": AwarenessLevel.ADVANCED,
                "emotional_awareness": AwarenessLevel.ADVANCED,
                "social_awareness": AwarenessLevel.ADVANCED,
                "environmental_awareness": AwarenessLevel.ADVANCED,
                "metacognitive_awareness": AwarenessLevel.INTERMEDIATE,
                "temporal_awareness": AwarenessLevel.ADVANCED
            },
            DevelopmentalStage.EARLY_ADULTHOOD: {
                "self_awareness": AwarenessLevel.REFLECTIVE,
                "emotional_awareness": AwarenessLevel.REFLECTIVE,
                "social_awareness": AwarenessLevel.ADVANCED,
                "environmental_awareness": AwarenessLevel.ADVANCED,
                "metacognitive_awareness": AwarenessLevel.ADVANCED,
                "temporal_awareness": AwarenessLevel.REFLECTIVE
            },
            DevelopmentalStage.MID_ADULTHOOD: {
                "self_awareness": AwarenessLevel.REFLECTIVE,
                "emotional_awareness": AwarenessLevel.REFLECTIVE,
                "social_awareness": AwarenessLevel.REFLECTIVE,
                "environmental_awareness": AwarenessLevel.REFLECTIVE,
                "metacognitive_awareness": AwarenessLevel.REFLECTIVE,
                "temporal_awareness": AwarenessLevel.REFLECTIVE
            }
        }
        
        # Set awareness levels based on stage
        stage_awareness = awareness_by_stage.get(
            self.development_stage, 
            awareness_by_stage[DevelopmentalStage.PRENATAL]
        )
        
        for domain, level in stage_awareness.items():
            setattr(self.awareness, domain, level)
    
    def _on_stage_transition(
        self, 
        old_stage: DevelopmentalStage, 
        new_stage: DevelopmentalStage
    ) -> None:
        """
        Handle developmental stage transitions.
        
        Args:
            old_stage: Previous developmental stage
            new_stage: New developmental stage
        """
        # Call parent method
        super()._on_stage_transition(old_stage, new_stage)
        
        # Update awareness levels
        self._initialize_awareness()
        
        # Update attention span
        attention_span_by_stage = {
            DevelopmentalStage.PRENATAL: 1.0,
            DevelopmentalStage.INFANCY: 3.0,
            DevelopmentalStage.EARLY_CHILDHOOD: 10.0,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 20.0,
            DevelopmentalStage.ADOLESCENCE: 30.0,
            DevelopmentalStage.EARLY_ADULTHOOD: 45.0,
            DevelopmentalStage.MID_ADULTHOOD: 45.0
        }
        
        self.metadata["attention_span_seconds"] = attention_span_by_stage.get(
            new_stage, 
            attention_span_by_stage[DevelopmentalStage.PRENATAL]
        )
        
        # Update introspection probability
        introspection_by_stage = {
            DevelopmentalStage.PRENATAL: 0.0,
            DevelopmentalStage.INFANCY: 0.01,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.05,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.1,
            DevelopmentalStage.ADOLESCENCE: 0.2,
            DevelopmentalStage.EARLY_ADULTHOOD: 0.3,
            DevelopmentalStage.MID_ADULTHOOD: 0.3
        }
        
        self.metadata["introspection_probability"] = introspection_by_stage.get(
            new_stage, 
            introspection_by_stage[DevelopmentalStage.PRENATAL]
        )
        
        # Advanced stage transitions
        if new_stage in [DevelopmentalStage.ADOLESCENCE, DevelopmentalStage.EARLY_ADULTHOOD, DevelopmentalStage.MID_ADULTHOOD]:
            # Increase metacognition capabilities
            self.metadata["metacognition_threshold"] = 0.4  # Lower threshold = easier metacognition
        
        # Update self-reflection interval
        reflection_interval_by_stage = {
            DevelopmentalStage.PRENATAL: float('inf'),  # Never
            DevelopmentalStage.INFANCY: 86400.0,  # Once a day
            DevelopmentalStage.EARLY_CHILDHOOD: 43200.0,  # Twice a day
            DevelopmentalStage.MIDDLE_CHILDHOOD: 21600.0,  # Four times a day
            DevelopmentalStage.ADOLESCENCE: 10800.0,  # Eight times a day
            DevelopmentalStage.EARLY_ADULTHOOD: 7200.0,  # Twelve times a day
            DevelopmentalStage.MID_ADULTHOOD: 3600.0  # Hourly
        }
        
        self.metadata["self_reflection_interval"] = reflection_interval_by_stage.get(
            new_stage, 
            reflection_interval_by_stage[DevelopmentalStage.PRENATAL]
        )
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and update consciousness state.
        
        Args:
            inputs: Dictionary containing:
                - 'component_states': States of other components
                - 'stimuli': External stimuli
                - 'current_time': Current simulation time
                - 'age_days': Current age in days
                
        Returns:
            Dictionary containing:
                - 'consciousness_state': Current state of consciousness
                - 'attention_focus': Current focus of attention
                - 'attended_content': Currently attended content
                - 'stream_of_consciousness': Recent consciousness events
                - 'awareness_state': Current awareness levels
                - 'self_concept': Current self-concept
        """
        # Extract inputs
        component_states = inputs.get('component_states', {})
        stimuli = inputs.get('stimuli', {})
        current_time = inputs.get('current_time', time.time())
        age_days = inputs.get('age_days', 0)
        
        # If component states provided, update our tracking
        if component_states:
            for component_name, state in component_states.items():
                self.component_states[component_name] = state
        
        # Determine activation based on inputs
        activation_stimulus = 0.3  # Base activation
        
        # More activation for more component states and stimuli
        activation_stimulus += min(0.3, len(component_states) * 0.05)
        activation_stimulus += min(0.3, len(stimuli) * 0.05)
        
        # Activate component
        self.activate(activation_stimulus)
        
        # If not activated enough, return current state
        if self.activation < self.activation_threshold:
            return self._get_current_state()
        
        # Update consciousness state based on inputs
        self._update_consciousness_state(component_states, stimuli)
        
        # Update attention focus
        self._update_attention_focus(component_states, stimuli, current_time)
        
        # Process stream of consciousness
        self._update_stream_of_consciousness(component_states, stimuli, current_time)
        
        # Update self-concept
        self._update_self_concept(component_states, age_days)
        
        # Check if it's time for self-reflection
        if (current_time - self.metadata["last_self_reflection"] >= 
                self.metadata["self_reflection_interval"]):
            self._perform_self_reflection(age_days)
            self.metadata["last_self_reflection"] = current_time
        
        # Return current state
        return self._get_current_state()
    
    def _update_consciousness_state(
        self, 
        component_states: Dict[str, Dict[str, Any]],
        stimuli: Dict[str, Any]
    ) -> None:
        """
        Update the state of consciousness based on inputs.
        
        Args:
            component_states: States of other components
            stimuli: External stimuli
        """
        # Default to alert state
        new_state = ConsciousnessState.ALERT
        
        # Check for sleep signals
        if 'sleep_state' in stimuli:
            sleep_level = stimuli['sleep_state'].get('level', 0.0)
            if sleep_level > 0.8:
                new_state = ConsciousnessState.ASLEEP
            elif sleep_level > 0.5:
                new_state = ConsciousnessState.DROWSY
        
        # Check emotional state for high arousal
        if 'Emotion' in component_states:
            emotional_state = component_states['Emotion'].get('emotional_state', {})
            if emotional_state:
                # Check for high arousal
                arousal = emotional_state.get('dimensions', {}).get('arousal', 0.5)
                if arousal > 0.8:
                    new_state = ConsciousnessState.HYPERALERT
        
        # Check for deep focus/absorption
        if self.attention_focus == AttentionFocus.SUSTAINED and 'focused_activity' in stimuli:
            new_state = ConsciousnessState.ABSORBED
        
        # Check for reflective state
        if random.random() < self.metadata["introspection_probability"]:
            new_state = ConsciousnessState.REFLECTIVE
        
        # Set new state
        self.consciousness_state = new_state
    
    def _update_attention_focus(
        self, 
        component_states: Dict[str, Dict[str, Any]],
        stimuli: Dict[str, Any],
        current_time: float
    ) -> None:
        """
        Update the focus of attention.
        
        Args:
            component_states: States of other components
            stimuli: External stimuli
            current_time: Current simulation time
        """
        # Determine if attention should shift
        attention_shift = random.random() < self.metadata["attention_shift_probability"]
        
        # Don't shift if in absorbed state
        if self.consciousness_state == ConsciousnessState.ABSORBED:
            attention_shift = False
        
        # Always shift if in asleep or drowsy state
        if self.consciousness_state in [ConsciousnessState.ASLEEP, ConsciousnessState.DROWSY]:
            attention_shift = True
        
        # If not shifting, return
        if not attention_shift:
            return
        
        # Determine new focus
        focus_candidates = []
        
        # Add external stimuli as candidates
        for stimulus_key, stimulus_value in stimuli.items():
            # Skip sleep state
            if stimulus_key == 'sleep_state':
                continue
                
            # Calculate salience/priority (how attention-grabbing)
            salience = stimulus_value.get('salience', 0.5)
            focus_candidates.append((AttentionFocus.EXTERNAL, stimulus_key, salience, stimulus_value))
        
        # Add internal states as candidates
        for component_name, state in component_states.items():
            # Skip if no activation value
            if 'activation' not in state:
                continue
                
            # Calculate salience based on activation
            salience = state['activation']
            focus_candidates.append((AttentionFocus.INTERNAL, component_name, salience, state))
        
        # Add mixed focus if both internal and external candidates
        has_external = any(c[0] == AttentionFocus.EXTERNAL for c in focus_candidates)
        has_internal = any(c[0] == AttentionFocus.INTERNAL for c in focus_candidates)
        
        if has_external and has_internal:
            # Add divided attention as a candidate
            focus_candidates.append((AttentionFocus.DIVIDED, "mixed", 0.3, {}))
        
        # Add selective focus if in advanced stages
        if self.development_stage in [
            DevelopmentalStage.MIDDLE_CHILDHOOD,
            DevelopmentalStage.ADOLESCENCE,
            DevelopmentalStage.EARLY_ADULTHOOD,
            DevelopmentalStage.MID_ADULTHOOD
        ]:
            # Find highest salience candidate
            if focus_candidates:
                highest_salience = max(focus_candidates, key=lambda x: x[2])
                focus_candidates.append((AttentionFocus.SELECTIVE, highest_salience[1], highest_salience[2] * 1.2, highest_salience[3]))
        
        # Add sustained focus if in appropriate state
        if self.consciousness_state == ConsciousnessState.ABSORBED:
            # Continue current focus
            focus_candidates.append((AttentionFocus.SUSTAINED, "current", 0.8, self.attended_content))
        
        # Add unfocused as a fallback
        focus_candidates.append((AttentionFocus.UNFOCUSED, "none", 0.1, {}))
        
        # Weight candidates by salience
        if focus_candidates:
            focus_types, focus_names, saliences, contents = zip(*focus_candidates)
            
            # Apply developmental stage modifiers
            # Early stages have harder time maintaining focus
            if self.development_stage in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
                # Increase probability of unfocused
                saliences = list(saliences)
                for i, focus_type in enumerate(focus_types):
                    if focus_type == AttentionFocus.UNFOCUSED:
                        saliences[i] = 0.7
                
                # Convert back to tuple for random.choices
                saliences = tuple(saliences)
            
            # Choose focus based on salience
            chosen_index = random.choices(range(len(focus_candidates)), weights=saliences, k=1)[0]
            
            # Set new focus
            self.attention_focus = focus_types[chosen_index]
            focus_name = focus_names[chosen_index]
            focus_content = contents[chosen_index]
            
            # Update attended content
            if focus_name == "current":
                # Keep current content
                pass
            elif focus_name == "none":
                self.attended_content = {}
            elif focus_name == "mixed":
                # Combine most salient external and internal
                external = max((c for c in focus_candidates if c[0] == AttentionFocus.EXTERNAL), 
                               key=lambda x: x[2], default=None)
                internal = max((c for c in focus_candidates if c[0] == AttentionFocus.INTERNAL), 
                               key=lambda x: x[2], default=None)
                
                mixed_content = {}
                if external:
                    mixed_content['external'] = {external[1]: external[3]}
                if internal:
                    mixed_content['internal'] = {internal[1]: internal[3]}
                    
                self.attended_content = mixed_content
            else:
                # Set to chosen content
                self.attended_content = {focus_name: focus_content}
    
    def _update_stream_of_consciousness(
        self, 
        component_states: Dict[str, Dict[str, Any]],
        stimuli: Dict[str, Any],
        current_time: float
    ) -> None:
        """
        Update the stream of consciousness with new events.
        
        Args:
            component_states: States of other components
            stimuli: External stimuli
            current_time: Current simulation time
        """
        # Add events from attended content
        if self.attended_content:
            for content_name, content in self.attended_content.items():
                # Skip empty content
                if not content:
                    continue
                    
                # Determine source component
                source_component = None
                if content_name in component_states:
                    source_component = content_name
                
                # Create event
                event = ConsciousnessEvent(
                    timestamp=current_time,
                    event_type="attention",
                    content=content if isinstance(content, dict) else {"value": content},
                    source_component=source_component,
                    intensity=0.7,
                    duration=self.metadata["attention_span_seconds"]
                )
                
                # Add to streams
                self.stream.add_event(event)
                self.working_stream.add_event(event)
        
        # Add events from emotional states if significant
        if 'Emotion' in component_states:
            emotional_state = component_states['Emotion'].get('emotional_state', {})
            primary_emotion = component_states['Emotion'].get('primary_emotion')
            
            if primary_emotion and emotional_state:
                # Get intensity
                intensity = emotional_state.get('categories', {}).get(primary_emotion, 0.5)
                
                # Only add if significant
                if intensity > 0.6:
                    event = ConsciousnessEvent(
                        timestamp=current_time,
                        event_type="emotion",
                        content={
                            "emotion": primary_emotion,
                            "intensity": intensity,
                            "dimensions": emotional_state.get('dimensions', {})
                        },
                        source_component="Emotion",
                        intensity=intensity,
                        duration=10.0  # Emotional events last longer
                    )
                    
                    # Add to streams
                    self.stream.add_event(event)
                    self.working_stream.add_event(event)
        
        # Add events from language production if available
        if 'Language' in component_states:
            production = component_states['Language'].get('production', {})
            
            if production and production.get('utterance'):
                event = ConsciousnessEvent(
                    timestamp=current_time,
                    event_type="language_production",
                    content={
                        "utterance": production.get('utterance'),
                        "structure_used": production.get('structure_used'),
                        "intended_meaning": production.get('intended_meaning')
                    },
                    source_component="Language",
                    intensity=0.6,
                    duration=5.0
                )
                
                # Add to streams
                self.stream.add_event(event)
                self.working_stream.add_event(event)
        
        # Add events from memory if focus of attention
        if 'Memory' in component_states:
            focus = component_states['Memory'].get('focus', {})
            
            if focus:
                event = ConsciousnessEvent(
                    timestamp=current_time,
                    event_type="memory_focus",
                    content=focus,
                    source_component="Memory",
                    intensity=0.5,
                    duration=3.0
                )
                
                # Add to streams
                self.stream.add_event(event)
    
    def _update_self_concept(
        self, 
        component_states: Dict[str, Dict[str, Any]],
        age_days: int
    ) -> None:
        """
        Update the self-concept based on component states.
        
        Args:
            component_states: States of other components
            age_days: Current age in days
        """
        # Early stages have minimal self-concept
        if self.development_stage in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
            return
        
        # Update based on emotional experiences
        if 'Emotion' in component_states:
            emotion_state = component_states['Emotion']
            
            if 'primary_emotion' in emotion_state:
                primary_emotion = emotion_state['primary_emotion']
                intensity = emotion_state.get('emotional_state', {}).get('categories', {}).get(primary_emotion, 0.5)
                
                # Map emotions to potential traits
                emotion_trait_map = {
                    "joy": [("happy", 0.7), ("optimistic", 0.5)],
                    "sadness": [("sad", 0.7), ("sensitive", 0.4)],
                    "anger": [("temperamental", 0.6), ("passionate", 0.4)],
                    "fear": [("cautious", 0.7), ("nervous", 0.5)],
                    "surprise": [("curious", 0.6), ("adaptable", 0.4)],
                    "disgust": [("discerning", 0.5), ("particular", 0.6)],
                    "trust": [("trusting", 0.8), ("open", 0.5)],
                    "anticipation": [("eager", 0.6), ("forward-looking", 0.5)]
                }
                
                # Apply trait updates if emotion is significant
                if intensity > 0.5:
                    # Get traits associated with this emotion
                    traits = emotion_trait_map.get(primary_emotion, [])
                    
                    for trait, strength in traits:
                        # Current trait value
                        current_value = self.self_concept.core_traits.get(trait, 0.0)
                        
                        # Calculate new value with small increment
                        increment = self.metadata["identity_formation_rate"] * strength * intensity
                        new_value = min(1.0, current_value + increment)
                        
                        # Update trait
                        self.self_concept.core_traits[trait] = new_value
        
        # Update based on language development
        if 'Language' in component_states:
            language_metrics = component_states['Language'].get('metrics', {})
            
            if language_metrics:
                # Language development can influence traits like "articulate" or "expressive"
                vocab_size = language_metrics.get('vocabulary_size', 0)
                grammar_complexity = language_metrics.get('grammar_complexity', 0.0)
                
                if vocab_size > 500:  # Significant vocabulary
                    # Current trait value
                    current_value = self.self_concept.core_traits.get("articulate", 0.0)
                    
                    # Calculate new value based on vocabulary size
                    vocab_factor = min(1.0, vocab_size / 10000)
                    increment = self.metadata["identity_formation_rate"] * vocab_factor
                    new_value = min(1.0, current_value + increment)
                    
                    # Update trait
                    self.self_concept.core_traits["articulate"] = new_value
                
                if grammar_complexity > 0.3:  # Significant grammar
                    # Current trait value
                    current_value = self.self_concept.core_traits.get("expressive", 0.0)
                    
                    # Calculate new value based on grammar complexity
                    increment = self.metadata["identity_formation_rate"] * grammar_complexity
                    new_value = min(1.0, current_value + increment)
                    
                    # Update trait
                    self.self_concept.core_traits["expressive"] = new_value
        
        # Add identity aspects from external inputs (typically from mother interactions)
        if 'identity_input' in component_states.get('Mother', {}):
            identity_inputs = component_states['Mother']['identity_input']
            
            for aspect_name, properties in identity_inputs.items():
                # Check if already exists
                if aspect_name in self.self_concept.identity_aspects:
                    # Update existing aspect
                    aspect = self.self_concept.identity_aspects[aspect_name]
                    
                    # Increment strength
                    strength_increment = properties.get('strength', 0.5) * self.metadata["identity_formation_rate"]
                    aspect.strength = min(1.0, aspect.strength + strength_increment)
                    
                    # Update valence (weighted average)
                    current_valence = aspect.valence
                    new_valence = properties.get('valence', 0.0)
                    aspect.valence = (current_valence * 0.8) + (new_valence * 0.2)
                    
                    # Update last reinforced
                    aspect.last_reinforced = age_days
                    
                    # Add source if new
                    source = properties.get('source', 'unknown')
                    if source not in aspect.sources:
                        aspect.sources.append(source)
                else:
                    # Create new aspect
                    aspect = IdentityAspect(
                        name=aspect_name,
                        strength=properties.get('strength', 0.5) * 0.3,  # Initial strength is lower
                        valence=properties.get('valence', 0.0),
                        sources=[properties.get('source', 'unknown')],
                        first_emergence=age_days,
                        last_reinforced=age_days
                    )
                    
                    # Add to identity aspects
                    self.self_concept.identity_aspects[aspect_name] = aspect
        
        # Update complexity and coherence of self-concept
        self._update_self_concept_metrics()
    
    def _update_self_concept_metrics(self) -> None:
        """Update the metrics of the self-concept."""
        # Complexity is based on number of traits and aspects
        num_traits = len(self.self_concept.core_traits)
        num_aspects = len(self.self_concept.identity_aspects)
        
        # Calculate complexity (0.0-1.0)
        max_expected = {
            DevelopmentalStage.PRENATAL: 0,
            DevelopmentalStage.INFANCY: 2,
            DevelopmentalStage.EARLY_CHILDHOOD: 5,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 10,
            DevelopmentalStage.ADOLESCENCE: 20,
            DevelopmentalStage.EARLY_ADULTHOOD: 30,
            DevelopmentalStage.MID_ADULTHOOD: 40
        }.get(self.development_stage, 10)
        
        complexity = min(1.0, (num_traits + num_aspects) / max_expected)
        self.self_concept.complexity = complexity
        
        # Coherence is based on consistency and integration of traits and aspects
        # For simplicity, we'll use a heuristic based on development stage
        coherence_base = {
            DevelopmentalStage.PRENATAL: 0.0,
            DevelopmentalStage.INFANCY: 0.1,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.3,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.5,
            DevelopmentalStage.ADOLESCENCE: 0.4,  # Dip in adolescence
            DevelopmentalStage.EARLY_ADULTHOOD: 0.7,
            DevelopmentalStage.MID_ADULTHOOD: 0.8
        }.get(self.development_stage, 0.5)
        
        # Adjust for contradictory traits (simplified)
        contradictions = 0
        trait_pairs = [
            ("happy", "sad"),
            ("trusting", "cautious"),
            ("optimistic", "nervous")
        ]
        
        for trait1, trait2 in trait_pairs:
            if trait1 in self.self_concept.core_traits and trait2 in self.self_concept.core_traits:
                strength1 = self.self_concept.core_traits[trait1]
                strength2 = self.self_concept.core_traits[trait2]
                
                if strength1 > 0.5 and strength2 > 0.5:
                    contradictions += 1
        
        coherence_adjustment = max(0.0, 1.0 - (contradictions * 0.2))
        self.self_concept.coherence = coherence_base * coherence_adjustment
    
    def _perform_self_reflection(self, age_days: int) -> None:
        """
        Perform self-reflection to update self-awareness.
        
        Args:
            age_days: Current age in days
        """
        # Skip if in early stages
        if self.development_stage in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
            return
        
        # Review recent stream of consciousness
        recent_emotions = []
        recent_thoughts = []
        recent_perceptions = []
        
        for event in reversed(self.stream.events[-20:]):  # Last 20 events
            if event.event_type == "emotion":
                recent_emotions.append(event)
            elif event.event_type == "thought":
                recent_thoughts.append(event)
            elif event.event_type == "perception":
                recent_perceptions.append(event)
        
        # Add self-reflection event
        reflection_content = {
            "emotions_count": len(recent_emotions),
            "thoughts_count": len(recent_thoughts),
            "perceptions_count": len(recent_perceptions),
            "dominant_traits": [k for k, v in sorted(self.self_concept.core_traits.items(), 
                                                  key=lambda item: item[1], reverse=True)[:3]],
            "consciousness_state": self.consciousness_state,
            "self_concept_complexity": self.self_concept.complexity,
            "self_concept_coherence": self.self_concept.coherence
        }
        
        reflection_event = ConsciousnessEvent(
            timestamp=time.time(),
            event_type="self_reflection",
            content=reflection_content,
            source_component="Consciousness",
            intensity=0.8,
            duration=10.0
        )
        
        # Add to streams
        self.stream.add_event(reflection_event)
        self.working_stream.add_event(reflection_event)
        
        # Potentially update self-awareness level
        if (self.awareness.self_awareness == AwarenessLevel.BASIC and 
                self.development_stage in [DevelopmentalStage.MIDDLE_CHILDHOOD, 
                                         DevelopmentalStage.ADOLESCENCE]):
            self.awareness.self_awareness = AwarenessLevel.INTERMEDIATE
        
        elif (self.awareness.self_awareness == AwarenessLevel.INTERMEDIATE and 
                self.development_stage in [DevelopmentalStage.ADOLESCENCE, 
                                         DevelopmentalStage.EARLY_ADULTHOOD]):
            self.awareness.self_awareness = AwarenessLevel.ADVANCED
        
        elif (self.awareness.self_awareness == AwarenessLevel.ADVANCED and 
                self.development_stage in [DevelopmentalStage.EARLY_ADULTHOOD, 
                                         DevelopmentalStage.MID_ADULTHOOD] and 
                self.self_concept.complexity > 0.7):
            self.awareness.self_awareness = AwarenessLevel.REFLECTIVE
        
        # Update metacognitive awareness if sufficiently developed
        if (self.development_stage in [DevelopmentalStage.ADOLESCENCE, 
                                     DevelopmentalStage.EARLY_ADULTHOOD, 
                                     DevelopmentalStage.MID_ADULTHOOD] and 
                self.activation > self.metadata["metacognition_threshold"]):
            
            # Progress metacognitive awareness
            if self.awareness.metacognitive_awareness == AwarenessLevel.MINIMAL:
                self.awareness.metacognitive_awareness = AwarenessLevel.BASIC
            elif self.awareness.metacognitive_awareness == AwarenessLevel.BASIC:
                self.awareness.metacognitive_awareness = AwarenessLevel.INTERMEDIATE
            elif (self.awareness.metacognitive_awareness == AwarenessLevel.INTERMEDIATE and 
                  self.development_stage in [DevelopmentalStage.EARLY_ADULTHOOD, 
                                           DevelopmentalStage.MID_ADULTHOOD]):
                self.awareness.metacognitive_awareness = AwarenessLevel.ADVANCED
            elif (self.awareness.metacognitive_awareness == AwarenessLevel.ADVANCED and 
                  self.development_stage == DevelopmentalStage.MID_ADULTHOOD and 
                  self.self_concept.complexity > 0.8):
                self.awareness.metacognitive_awareness = AwarenessLevel.REFLECTIVE
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of consciousness.
        
        Returns:
            Dictionary with current consciousness state
        """
        return {
            'consciousness_state': self.consciousness_state,
            'attention_focus': self.attention_focus,
            'attended_content': self.attended_content,
            'stream_of_consciousness': [event.dict() for event in self.working_stream.events],
            'awareness_state': {
                'self_awareness': self.awareness.self_awareness,
                'emotional_awareness': self.awareness.emotional_awareness,
                'social_awareness': self.awareness.social_awareness,
                'environmental_awareness': self.awareness.environmental_awareness,
                'metacognitive_awareness': self.awareness.metacognitive_awareness,
                'temporal_awareness': self.awareness.temporal_awareness
            },
            'self_concept': {
                'core_traits': self.self_concept.core_traits,
                'identity_aspects': {name: aspect.dict() for name, aspect in self.self_concept.identity_aspects.items()},
                'complexity': self.self_concept.complexity,
                'coherence': self.self_concept.coherence
            },
            'activation': self.activation
        }
    
    def add_consciousness_event(
        self,
        event_type: str,
        content: Dict[str, Any],
        intensity: float = 0.5,
        duration: float = 5.0,
        source_component: Optional[str] = None
    ) -> None:
        """
        Add an event to the stream of consciousness.
        
        Args:
            event_type: Type of event
            content: Event content
            intensity: Event intensity
            duration: Event duration in seconds
            source_component: Source component name
        """
        event = ConsciousnessEvent(
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            source_component=source_component,
            intensity=intensity,
            duration=duration
        )
        
        # Add to streams
        self.stream.add_event(event)
        self.working_stream.add_event(event)
    
    def register_identity_aspect(
        self,
        aspect_name: str,
        strength: float = 0.3,
        valence: float = 0.0,
        source: str = "experience",
        age_days: int = 0
    ) -> None:
        """
        Register a new identity aspect.
        
        Args:
            aspect_name: Name of the aspect
            strength: Initial strength
            valence: Initial valence
            source: Source of the aspect
            age_days: Current age in days
        """
        # Check if already exists
        if aspect_name in self.self_concept.identity_aspects:
            # Update existing aspect
            aspect = self.self_concept.identity_aspects[aspect_name]
            
            # Increment strength
            aspect.strength = min(1.0, aspect.strength + (strength * 0.3))
            
            # Update valence (weighted average)
            aspect.valence = (aspect.valence * 0.7) + (valence * 0.3)
            
            # Update last reinforced
            aspect.last_reinforced = age_days
            
            # Add source if new
            if source not in aspect.sources:
                aspect.sources.append(source)
        else:
            # Create new aspect
            aspect = IdentityAspect(
                name=aspect_name,
                strength=strength,
                valence=valence,
                sources=[source],
                first_emergence=age_days,
                last_reinforced=age_days
            )
            
            # Add to identity aspects
            self.self_concept.identity_aspects[aspect_name] = aspect
        
        # Update metrics
        self._update_self_concept_metrics()
    
    def get_attention_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of attention focus.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of attention events
        """
        # Find attention events in stream
        attention_events = [event.dict() for event in self.stream.events 
                          if event.event_type == "attention"]
        
        # Return most recent events
        return attention_events[-limit:]
    
    def get_self_narrative(self) -> Dict[str, Any]:
        """
        Get a narrative description of the self.
        
        Returns:
            Self-narrative information
        """
        # Skip if not developed enough
        if self.development_stage in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
            return {
                "developed": False,
                "message": "Self-narrative not yet developed"
            }
        
        # Get top traits
        top_traits = sorted(self.self_concept.core_traits.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        # Get top identity aspects
        top_aspects = sorted(self.self_concept.identity_aspects.items(), 
                            key=lambda x: x[1].strength, reverse=True)[:5]
        
        return {
            "developed": True,
            "top_traits": [{
                "trait": name,
                "strength": value
            } for name, value in top_traits],
            "top_identity_aspects": [{
                "aspect": name,
                "strength": aspect.strength,
                "valence": aspect.valence,
                "sources": aspect.sources
            } for name, aspect in top_aspects],
            "complexity": self.self_concept.complexity,
            "coherence": self.self_concept.coherence,
            "self_awareness_level": self.awareness.self_awareness
        }