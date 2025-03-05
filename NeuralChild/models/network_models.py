from typing import Dict, List, Optional, Union, Literal, Any
from enum import Enum, auto
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
import numpy as np
from datetime import datetime

class ConnectionType(str, Enum):
    """Types of connections between neural networks"""
    EXCITATORY = "excitatory"  # Increases activation
    INHIBITORY = "inhibitory"  # Decreases activation  
    MODULATORY = "modulatory"  # Changes behavior without direct activation
    FEEDBACK = "feedback"      # Provides feedback signals
    ASSOCIATIVE = "associative"  # Creates associations between concepts

class ActivationFunction(str, Enum):
    """Activation functions for neural networks"""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"
    LINEAR = "linear"

class NetworkState(str, Enum):
    """Possible states of a neural network"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    LEARNING = "learning"
    CONSOLIDATING = "consolidating"

class Connection(BaseModel):
    """Represents a connection between networks"""
    source_id: UUID = Field(..., description="ID of the source network")
    target_id: UUID = Field(..., description="ID of the target network")
    connection_type: ConnectionType = Field(..., description="Type of connection")
    weight: float = Field(1.0, description="Connection weight/strength")
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True

class BaseNetworkConfig(BaseModel):
    """Base configuration for any neural network"""
    activation_function: ActivationFunction = Field(ActivationFunction.SIGMOID)
    learning_rate: float = Field(0.01, ge=0, le=1)
    decay_rate: float = Field(0.001, ge=0, le=1)
    threshold: float = Field(0.5, ge=0, le=1)
    max_capacity: int = Field(1000, gt=0)
    initial_activation: float = Field(0.1, ge=0, le=1)

class BaseNetwork(BaseModel):
    """Base model for all neural networks in the system"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: str = Field(..., description="Type of neural network")
    state: NetworkState = Field(NetworkState.INACTIVE)
    activation: float = Field(0.0, ge=0.0, le=1.0, description="Current activation level")
    connections: List[Connection] = Field(default_factory=list)
    config: BaseNetworkConfig = Field(default_factory=BaseNetworkConfig)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_connection(self, target_network_id: UUID, 
                       connection_type: ConnectionType,
                       weight: float = 1.0) -> None:
        """Add a connection to another network"""
        connection = Connection(
            source_id=self.id,
            target_id=target_network_id,
            connection_type=connection_type,
            weight=weight
        )
        self.connections.append(connection)
        self.last_updated = datetime.now()
    
    def update_activation(self, new_value: float) -> None:
        """Update the activation level of the network"""
        self.activation = max(0.0, min(new_value, 1.0))  # Clamp between 0 and 1
        self.last_updated = datetime.now()
    
    class Config:
        arbitrary_types_allowed = True

# Specialized Network Models

class ArchetypeNetworkConfig(BaseNetworkConfig):
    """Configuration for the archetypes network"""
    depth_factor: float = Field(0.8, description="How deeply archetypes influence behavior")
    emergence_threshold: float = Field(0.7, description="Threshold for archetype emergence")

class ArchetypeNetwork(BaseNetwork):
    """Network for modeling deep personality patterns"""
    type: Literal["archetype"] = "archetype"
    config: ArchetypeNetworkConfig = Field(default_factory=ArchetypeNetworkConfig)
    archetypes: Dict[str, float] = Field(default_factory=dict)
    
    def add_archetype(self, name: str, strength: float) -> None:
        """Add or update an archetype pattern"""
        self.archetypes[name] = max(0.0, min(strength, 1.0))
        self.last_updated = datetime.now()

class InstinctNetworkConfig(BaseNetworkConfig):
    """Configuration for the instincts network"""
    response_speed: float = Field(0.9, description="Speed of instinctual responses")
    inhibition_factor: float = Field(0.3, description="How strongly instincts can be inhibited")

class InstinctNetwork(BaseNetwork):
    """Network for modeling hardwired responses to stimuli"""
    type: Literal["instinct"] = "instinct"
    config: InstinctNetworkConfig = Field(default_factory=InstinctNetworkConfig)
    instincts: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    def add_instinct(self, name: str, trigger: str, strength: float) -> None:
        """Add or update an instinctual response"""
        if name not in self.instincts:
            self.instincts[name] = {}
        self.instincts[name][trigger] = max(0.0, min(strength, 1.0))
        self.last_updated = datetime.now()

class UnconsciousnessNetworkConfig(BaseNetworkConfig):
    """Configuration for the unconsciousness network"""
    depth_levels: int = Field(5, description="Levels of unconscious processing")
    association_strength: float = Field(0.6, description="Strength of unconscious associations")

class UnconsciousnessNetwork(BaseNetwork):
    """Network for modeling automatic associations and implicit learning"""
    type: Literal["unconsciousness"] = "unconsciousness"
    config: UnconsciousnessNetworkConfig = Field(default_factory=UnconsciousnessNetworkConfig)
    associations: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    def add_association(self, source: str, target: str, strength: float) -> None:
        """Add or update an unconscious association"""
        if source not in self.associations:
            self.associations[source] = {}
        self.associations[source][target] = max(0.0, min(strength, 1.0))
        self.last_updated = datetime.now()

class DriveType(str, Enum):
    """Types of drives/needs"""
    PHYSIOLOGICAL = "physiological"  # Food, water, etc.
    SAFETY = "safety"                # Security, stability
    BELONGING = "belonging"          # Connection, attachment
    ESTEEM = "esteem"                # Respect, recognition
    SELF_ACTUALIZATION = "self_actualization"  # Growth, fulfillment
    CURIOSITY = "curiosity"          # Knowledge, exploration

class DriveNetworkConfig(BaseNetworkConfig):
    """Configuration for the drives network"""
    urgency_factor: float = Field(0.7, description="How urgent drives become")
    satisfaction_rate: float = Field(0.2, description="Rate at which drives can be satisfied")

class DriveNetwork(BaseNetwork):
    """Network for modeling core motivational forces"""
    type: Literal["drive"] = "drive"
    config: DriveNetworkConfig = Field(default_factory=DriveNetworkConfig)
    drives: Dict[DriveType, float] = Field(default_factory=dict)
    
    def update_drive(self, drive_type: DriveType, intensity: float) -> None:
        """Update the intensity of a drive"""
        self.drives[drive_type] = max(0.0, min(intensity, 1.0))
        self.last_updated = datetime.now()
        
    def satisfy_drive(self, drive_type: DriveType, amount: float) -> None:
        """Reduce the intensity of a drive (satisfy it)"""
        if drive_type in self.drives:
            self.drives[drive_type] = max(0.0, self.drives[drive_type] - amount)
            self.last_updated = datetime.now()

class Emotion(str, Enum):
    """Basic emotions"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    # More complex emotions develop over time
    GUILT = "guilt"
    SHAME = "shame"
    PRIDE = "pride"
    JEALOUSY = "jealousy"
    LOVE = "love"
    GRIEF = "grief"

class EmotionNetworkConfig(BaseNetworkConfig):
    """Configuration for the emotions network"""
    intensity_factor: float = Field(0.6, description="How intensely emotions are felt")
    decay_curve: str = Field("exponential", description="How emotions fade over time")
    complexity_development: float = Field(0.2, description="Rate at which emotional complexity develops")

class EmotionNetwork(BaseNetwork):
    """Network for modeling affective responses"""
    type: Literal["emotion"] = "emotion"
    config: EmotionNetworkConfig = Field(default_factory=EmotionNetworkConfig)
    emotions: Dict[Emotion, float] = Field(default_factory=dict)
    emotional_complexity: float = Field(0.1, description="Current emotional complexity level")
    
    def feel_emotion(self, emotion: Emotion, intensity: float) -> None:
        """Generate an emotional response"""
        self.emotions[emotion] = max(0.0, min(intensity, 1.0))
        self.activation = max(self.activation, intensity * 0.5)  # Emotions affect overall activation
        self.last_updated = datetime.now()
    
    def decay_emotions(self, time_delta: float) -> None:
        """Allow emotions to decay over time"""
        for emotion in self.emotions:
            if self.config.decay_curve == "exponential":
                self.emotions[emotion] *= (1 - self.config.decay_rate * time_delta)
            else:  # Linear decay
                self.emotions[emotion] = max(0.0, self.emotions[emotion] - self.config.decay_rate * time_delta)
        self.last_updated = datetime.now()

class Mood(str, Enum):
    """Basic moods"""
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    CALM = "calm"
    IRRITABLE = "irritable"
    CONTENT = "content"
    NEUTRAL = "neutral"

class MoodNetworkConfig(BaseNetworkConfig):
    """Configuration for the moods network"""
    stability_factor: float = Field(0.8, description="How stable moods are")
    influence_threshold: float = Field(0.4, description="Threshold for emotions to influence mood")

class MoodNetwork(BaseNetwork):
    """Network for modeling longer-lasting emotional states"""
    type: Literal["mood"] = "mood"
    config: MoodNetworkConfig = Field(default_factory=MoodNetworkConfig)
    current_mood: Mood = Field(Mood.NEUTRAL)
    mood_intensity: float = Field(0.5, ge=0.0, le=1.0)
    
    def update_mood(self, emotions: Dict[Emotion, float]) -> None:
        """Update mood based on emotional inputs"""
        # Simple implementation - would be more complex in reality
        if emotions.get(Emotion.JOY, 0) > self.config.influence_threshold:
            self.current_mood = Mood.HAPPY
            self.mood_intensity = emotions[Emotion.JOY]
        elif emotions.get(Emotion.SADNESS, 0) > self.config.influence_threshold:
            self.current_mood = Mood.SAD
            self.mood_intensity = emotions[Emotion.SADNESS]
        # Add more mood transitions as needed
        self.last_updated = datetime.now()

class AttentionNetworkConfig(BaseNetworkConfig):
    """Configuration for the attention network"""
    focus_strength: float = Field(0.7, description="Strength of attentional focus")
    distraction_threshold: float = Field(0.6, description="Threshold for distractions")
    capacity: int = Field(4, description="Number of items that can be attended to")

class AttentionNetwork(BaseNetwork):
    """Network for controlling focus on relevant inputs"""
    type: Literal["attention"] = "attention"
    config: AttentionNetworkConfig = Field(default_factory=AttentionNetworkConfig)
    focused_items: List[UUID] = Field(default_factory=list)
    focus_level: float = Field(0.5, ge=0.0, le=1.0)
    
    def focus_on(self, item_id: UUID) -> bool:
        """Focus attention on an item"""
        if len(self.focused_items) < self.config.capacity or item_id in self.focused_items:
            if item_id not in self.focused_items:
                self.focused_items.append(item_id)
            self.last_updated = datetime.now()
            return True
        return False  # Attention capacity exceeded
    
    def remove_focus(self, item_id: UUID) -> None:
        """Stop focusing on an item"""
        if item_id in self.focused_items:
            self.focused_items.remove(item_id)
            self.last_updated = datetime.now()

class SensoryModality(str, Enum):
    """Types of sensory input"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"
    PROPRIOCEPTIVE = "proprioceptive"  # Body position sense
    VESTIBULAR = "vestibular"  # Balance sense

class PerceptionNetworkConfig(BaseNetworkConfig):
    """Configuration for the perception network"""
    sensory_thresholds: Dict[SensoryModality, float] = Field(
        default_factory=lambda: {modality: 0.2 for modality in SensoryModality}
    )
    integration_strength: float = Field(0.6, description="How strongly modalities are integrated")

class Percept(BaseModel):
    """A perceptual unit of information"""
    id: UUID = Field(default_factory=uuid4)
    modality: SensoryModality
    content: Any
    intensity: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class PerceptionNetwork(BaseNetwork):
    """Network for interpreting sensory information"""
    type: Literal["perception"] = "perception"
    config: PerceptionNetworkConfig = Field(default_factory=PerceptionNetworkConfig)
    active_percepts: Dict[UUID, Percept] = Field(default_factory=dict)
    
    def process_input(self, modality: SensoryModality, content: Any, intensity: float) -> UUID:
        """Process a sensory input and create a percept"""
        percept = Percept(modality=modality, content=content, intensity=intensity)
        self.active_percepts[percept.id] = percept
        self.last_updated = datetime.now()
        return percept.id
    
    def get_percept(self, percept_id: UUID) -> Optional[Percept]:
        """Retrieve a specific percept"""
        return self.active_percepts.get(percept_id)

class ConsciousnessNetworkConfig(BaseNetworkConfig):
    """Configuration for the consciousness network"""
    awareness_threshold: float = Field(0.7, description="Threshold for conscious awareness")
    integration_capacity: int = Field(7, description="Items that can be integrated at once")

class ConsciousnessNetwork(BaseNetwork):
    """Network for integrating information for awareness"""
    type: Literal["consciousness"] = "consciousness"
    config: ConsciousnessNetworkConfig = Field(default_factory=ConsciousnessNetworkConfig)
    conscious_contents: List[UUID] = Field(default_factory=list)
    awareness_level: float = Field(0.5, ge=0.0, le=1.0)
    
    def bring_to_consciousness(self, item_id: UUID) -> bool:
        """Attempt to bring an item to consciousness"""
        if len(self.conscious_contents) < self.config.integration_capacity:
            if item_id not in self.conscious_contents:
                self.conscious_contents.append(item_id)
            self.last_updated = datetime.now()
            return True
        return False  # Consciousness capacity exceeded
    
    def remove_from_consciousness(self, item_id: UUID) -> None:
        """Remove an item from consciousness"""
        if item_id in self.conscious_contents:
            self.conscious_contents.remove(item_id)
            self.last_updated = datetime.now()

class ThoughtType(str, Enum):
    """Types of thoughts"""
    PERCEPTION = "perception"  # Direct sensory interpretation
    MEMORY = "memory"  # Recalled information
    CONCEPT = "concept"  # Abstract concept
    BELIEF = "belief"  # Held belief
    GOAL = "goal"  # Desired outcome
    PLAN = "plan"  # Action sequence
    REFLECTION = "reflection"  # Self-reflective thought

class Thought(BaseModel):
    """A unit of thought"""
    id: UUID = Field(default_factory=uuid4)
    type: ThoughtType
    content: Any
    connections: List[UUID] = Field(default_factory=list)  # Other related thoughts
    created_at: datetime = Field(default_factory=datetime.now)

class ThoughtsNetworkConfig(BaseNetworkConfig):
    """Configuration for the thoughts network"""
    association_strength: float = Field(0.6, description="Strength of thought associations")
    divergence_factor: float = Field(0.3, description="How much thoughts diverge/explore")

class ThoughtsNetwork(BaseNetwork):
    """Network for active information processing"""
    type: Literal["thoughts"] = "thoughts"
    config: ThoughtsNetworkConfig = Field(default_factory=ThoughtsNetworkConfig)
    active_thoughts: Dict[UUID, Thought] = Field(default_factory=dict)
    
    def create_thought(self, thought_type: ThoughtType, content: Any, 
                       related_thought_ids: List[UUID] = None) -> UUID:
        """Create a new thought"""
        thought = Thought(
            type=thought_type, 
            content=content,
            connections=related_thought_ids or []
        )
        self.active_thoughts[thought.id] = thought
        self.last_updated = datetime.now()
        return thought.id
    
    def connect_thoughts(self, thought_id1: UUID, thought_id2: UUID) -> bool:
        """Connect two thoughts associatively"""
        if thought_id1 in self.active_thoughts and thought_id2 in self.active_thoughts:
            if thought_id2 not in self.active_thoughts[thought_id1].connections:
                self.active_thoughts[thought_id1].connections.append(thought_id2)
            if thought_id1 not in self.active_thoughts[thought_id2].connections:
                self.active_thoughts[thought_id2].connections.append(thought_id1)
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_thought(self, thought_id: UUID) -> Optional[Thought]:
        """Retrieve a specific thought"""
        return self.active_thoughts.get(thought_id)