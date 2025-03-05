"""
NeuralChild main class that integrates all components of the cognitive architecture.
Acts as the central hub connecting networks, memory systems, and developmental tracking.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np

from ..networks import (
    Network, base
)
from ..networks.archetypes import ArchetypeNetwork
from ..networks.instincts import InstinctNetwork
from ..networks.unconsciousness import UnconsciousnessNetwork
from ..networks.drives import DriveNetwork, DriveType
from ..networks.emotions import EmotionNetwork, Emotion
from ..networks.moods import MoodNetwork, Mood
from ..networks.attention import AttentionNetwork
from ..networks.perception import PerceptionNetwork, SensoryModality, Percept
from ..networks.consciousness import ConsciousnessNetwork
from ..networks.thoughts import ThoughtsNetwork, ThoughtType, Thought

from ..memory import (
    WorkingMemorySystem, EpisodicMemorySystem, 
    LongTermMemorySystem, MemoryConsolidation
)
from ..models.memory_models import (
    MemoryItem, MemoryType, MemoryAttributes, MemoryStage, 
    EmotionalValence, LongTermMemoryDomain
)

from ..core.development import DevelopmentTracker
from ..models.development_models import (
    DevelopmentalStage, DevelopmentalDomain, DevelopmentalMilestone,
    DevelopmentalMetrics
)

from ..language.vocabulary import VocabularyManager
from ..language.syntax import SyntaxProcessor
from ..language.production import LanguageProduction

from .. import config

# Configure logging
logger = logging.getLogger(__name__)

class NeuralChild:
    """
    Main class for the NeuralChild cognitive architecture.
    Integrates all components into a unified system that simulates
    the developing mind of a child.
    """
    
    def __init__(self, name: str = "Child", load_from_checkpoint: Optional[str] = None):
        """
        Initialize the NeuralChild system.
        
        Args:
            name: Name of the child
            load_from_checkpoint: Optional path to load state from a checkpoint
        """
        self.name = name
        self._creation_time = datetime.now()
        self._last_update = time.time()
        self._running = False
        self._update_thread = None
        self._update_interval = 0.1  # seconds
        self._update_lock = threading.RLock()
        
        # Initialize neural networks
        self._networks: Dict[str, Network] = {}
        
        # Initialize memory systems
        self.working_memory = WorkingMemorySystem()
        self.episodic_memory = EpisodicMemorySystem()
        self.long_term_memory = LongTermMemorySystem()
        
        # Initialize memory consolidation
        self.memory_consolidation = MemoryConsolidation(
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            long_term_memory=self.long_term_memory
        )
        
        # Initialize development tracker
        self.development_tracker = DevelopmentTracker()
        
        # Initialize language components
        self.vocabulary = VocabularyManager()
        self.syntax = SyntaxProcessor()
        self.language_production = LanguageProduction()
        
        # Development metrics
        self.metrics = DevelopmentalMetrics()
        
        # Initialize core networks if not loading from checkpoint
        if not load_from_checkpoint:
            self._initialize_networks()
        else:
            self._load_from_checkpoint(load_from_checkpoint)
        
        logger.info(f"NeuralChild '{name}' initialized")
    
    def _initialize_networks(self) -> None:
        """Initialize all neural networks in the system"""
        from ..models.network_models import (
            ArchetypeNetwork as ArchetypeModel,
            InstinctNetwork as InstinctModel,
            UnconsciousnessNetwork as UnconsciousnessModel,
            DriveNetwork as DriveModel,
            EmotionNetwork as EmotionModel,
            MoodNetwork as MoodModel,
            AttentionNetwork as AttentionModel,
            PerceptionNetwork as PerceptionModel, 
            ConsciousnessNetwork as ConsciousnessModel,
            ThoughtsNetwork as ThoughtsModel,
            BaseNetworkConfig, ActivationFunction, NetworkState, ConnectionType
        )
        
        # Create network models
        network_configs = config.NETWORKS
        
        # Archetypes Network (deep personality patterns)
        archetypes_model = ArchetypeModel(
            name="Archetypes",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["archetypes"],
                decay_rate=network_configs["activation_decay_rate"] * 0.1,  # Slower decay
                threshold=0.3
            )
        )
        self._networks["archetypes"] = Network(archetypes_model)
        
        # Instincts Network (hardwired responses)
        instincts_model = InstinctModel(
            name="Instincts",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["instincts"],
                decay_rate=network_configs["activation_decay_rate"] * 0.5,  # Medium decay
                threshold=0.4
            )
        )
        self._networks["instincts"] = Network(instincts_model)
        
        # Unconsciousness Network (implicit associations)
        unconsciousness_model = UnconsciousnessModel(
            name="Unconsciousness",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["unconsciousness"],
                decay_rate=network_configs["activation_decay_rate"] * 0.3,  # Slower decay
                threshold=0.3
            )
        )
        self._networks["unconsciousness"] = Network(unconsciousness_model)
        
        # Drives Network (motivational forces)
        drives_model = DriveModel(
            name="Drives",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["drives"],
                decay_rate=network_configs["activation_decay_rate"] * 0.4,  # Medium decay
                threshold=0.4
            )
        )
        self._networks["drives"] = Network(drives_model)
        
        # Initialize basic drives
        for drive_name, intensity in network_configs["initial_drives"].items():
            drive_type = DriveType(drive_name)
            drives_model.update_drive(drive_type, intensity)
        
        # Emotions Network (affective responses)
        emotions_model = EmotionModel(
            name="Emotions",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["emotions"],
                decay_rate=network_configs["activation_decay_rate"] * 1.2,  # Faster decay
                threshold=0.3
            )
        )
        self._networks["emotions"] = Network(emotions_model)
        
        # Moods Network (longer-lasting emotional states)
        moods_model = MoodModel(
            name="Moods",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["moods"],
                decay_rate=network_configs["activation_decay_rate"] * 0.1,  # Very slow decay
                threshold=0.3
            )
        )
        self._networks["moods"] = Network(moods_model)
        
        # Attention Network (focus control)
        attention_model = AttentionModel(
            name="Attention",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["attention"],
                decay_rate=network_configs["activation_decay_rate"] * 1.5,  # Faster decay
                threshold=0.4
            )
        )
        self._networks["attention"] = Network(attention_model)
        
        # Perception Network (sensory interpretation)
        perception_model = PerceptionModel(
            name="Perception",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["perception"],
                decay_rate=network_configs["activation_decay_rate"] * 1.2,  # Faster decay
                threshold=0.3
            )
        )
        self._networks["perception"] = Network(perception_model)
        
        # Consciousness Network (awareness integration)
        consciousness_model = ConsciousnessModel(
            name="Consciousness",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["consciousness"],
                decay_rate=network_configs["activation_decay_rate"] * 0.8,  # Medium decay
                threshold=0.5
            )
        )
        self._networks["consciousness"] = Network(consciousness_model)
        
        # Thoughts Network (active processing)
        thoughts_model = ThoughtsModel(
            name="Thoughts",
            config=BaseNetworkConfig(
                activation_function=ActivationFunction.SIGMOID,
                learning_rate=network_configs["learning_rates"]["thoughts"],
                decay_rate=network_configs["activation_decay_rate"] * 1.0,  # Standard decay
                threshold=0.4
            )
        )
        self._networks["thoughts"] = Network(thoughts_model)
        
        # Create network connections
        self._initialize_network_connections()
    
    def _initialize_network_connections(self) -> None:
        """Initialize connections between neural networks"""
        from ..models.network_models import ConnectionType
        
        # Connection configurations
        connection_weights = config.NETWORKS["connection_initial_weights"]
        
        # Dictionary of network IDs for easy reference
        network_ids = {name: network.id for name, network in self._networks.items()}
        
        # Archetypes -> Unconsciousness (deep patterns influence implicit associations)
        self._networks["archetypes"].add_connection(
            network_ids["unconsciousness"],
            ConnectionType.MODULATORY,
            connection_weights["modulatory"]
        )
        
        # Archetypes -> Drives (personality influences motivational forces)
        self._networks["archetypes"].add_connection(
            network_ids["drives"],
            ConnectionType.MODULATORY,
            connection_weights["modulatory"] * 0.8
        )
        
        # Instincts -> Drives (hardwired responses affect needs)
        self._networks["instincts"].add_connection(
            network_ids["drives"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.9
        )
        
        # Instincts -> Emotions (instinctual reactions trigger emotions)
        self._networks["instincts"].add_connection(
            network_ids["emotions"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"]
        )
        
        # Drives -> Attention (needs direct focus)
        self._networks["drives"].add_connection(
            network_ids["attention"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.7
        )
        
        # Drives -> Emotions (unmet needs trigger emotions)
        self._networks["drives"].add_connection(
            network_ids["emotions"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.8
        )
        
        # Emotions -> Moods (emotions influence longer-term states)
        self._networks["emotions"].add_connection(
            network_ids["moods"],
            ConnectionType.MODULATORY,
            connection_weights["modulatory"] * 1.2
        )
        
        # Emotions -> Attention (emotional salience draws attention)
        self._networks["emotions"].add_connection(
            network_ids["attention"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.9
        )
        
        # Attention -> Perception (attention enhances perception)
        self._networks["attention"].add_connection(
            network_ids["perception"],
            ConnectionType.MODULATORY,
            connection_weights["modulatory"] * 1.1
        )
        
        # Attention -> Consciousness (attended items enter consciousness)
        self._networks["attention"].add_connection(
            network_ids["consciousness"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 1.2
        )
        
        # Perception -> Attention (salient percepts attract attention)
        self._networks["perception"].add_connection(
            network_ids["attention"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.8
        )
        
        # Perception -> Emotions (perceptions trigger emotional responses)
        self._networks["perception"].add_connection(
            network_ids["emotions"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.7
        )
        
        # Perception -> Consciousness (perceptions enter awareness)
        self._networks["perception"].add_connection(
            network_ids["consciousness"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"]
        )
        
        # Consciousness -> Thoughts (conscious content generates thoughts)
        self._networks["consciousness"].add_connection(
            network_ids["thoughts"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 1.1
        )
        
        # Thoughts -> Consciousness (thoughts enter consciousness)
        self._networks["thoughts"].add_connection(
            network_ids["consciousness"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.9
        )
        
        # Thoughts -> Emotions (thoughts trigger emotions)
        self._networks["thoughts"].add_connection(
            network_ids["emotions"],
            ConnectionType.EXCITATORY,
            connection_weights["excitatory"] * 0.6
        )
        
        # Unconsciousness -> Thoughts (unconscious influences conscious thought)
        self._networks["unconsciousness"].add_connection(
            network_ids["thoughts"],
            ConnectionType.MODULATORY,
            connection_weights["modulatory"] * 0.7
        )
        
        # More connections can be added as needed
        logger.debug("Initialized network connections")
    
    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load system state from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Not implemented yet
        raise NotImplementedError("Loading from checkpoint not implemented yet")
    
    def start(self) -> None:
        """Start the NeuralChild system"""
        if self._running:
            logger.warning("NeuralChild system is already running")
            return
            
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("NeuralChild system started")
    
    def stop(self) -> None:
        """Stop the NeuralChild system"""
        if not self._running:
            logger.warning("NeuralChild system is not running")
            return
            
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
            self._update_thread = None
        logger.info("NeuralChild system stopped")
    
    def _update_loop(self) -> None:
        """Main update loop that runs in a separate thread"""
        while self._running:
            try:
                with self._update_lock:
                    self.update()
                time.sleep(self._update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}", exc_info=True)
                time.sleep(1.0)  # Sleep a bit longer on error
    
    def update(self, elapsed_seconds: Optional[float] = None) -> None:
        """
        Update all systems and networks based on elapsed time.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds = current_time - self._last_update
            self._last_update = current_time
        
        # Update development time
        self.development_tracker.update_simulated_time(elapsed_seconds)
        
        # Update networks
        network_registry = {network.id: network for network in self._networks.values()}
        for network in self._networks.values():
            network.update(network_registry, elapsed_seconds)
        
        # Update memory systems
        self.working_memory.update(elapsed_seconds)
        self.episodic_memory.update(elapsed_seconds)
        self.long_term_memory.update(elapsed_seconds)
        
        # Update memory consolidation
        self.memory_consolidation.update(elapsed_seconds)
        
        # Update language components
        current_stage = self.development_tracker.current_stage
        self.vocabulary.update(current_stage)
        self.syntax.update(current_stage)
        self.language_production.update(
            vocabulary=self.vocabulary,
            syntax=self.syntax,
            developmental_stage=current_stage
        )
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update developmental metrics based on current state"""
        # This method would be more complex in a full implementation
        # For now, we'll just update a few key metrics
        
        # Language metrics
        self.metrics.language.vocabulary_size = self.vocabulary.total_vocabulary_size
        self.metrics.language.active_vocabulary = self.vocabulary.active_vocabulary_size
        self.metrics.language.mean_utterance_length = self.language_production.mean_utterance_length
        
        # Memory metrics
        self.metrics.memory.working_memory_capacity = self.working_memory.current_load
        
        # Cognitive metrics - from various network stats
        self.metrics.cognitive.attention_span_seconds = self._get_attention_span()
        
        # Emotional metrics
        self.metrics.emotional.emotional_complexity = self._get_emotional_complexity()
        
        # Update timestamp
        self.metrics.last_updated = datetime.now()
    
    def _get_attention_span(self) -> float:
        """Calculate the current attention span in seconds"""
        # Simple calculation based on attention network activation and development stage
        base_span = 5.0  # 5 seconds base for newborn
        attention_multiplier = 1.0 + self._networks["attention"].activation * 2.0
        
        # Different base spans by developmental stage
        stage_spans = {
            DevelopmentalStage.NEWBORN: 5.0,
            DevelopmentalStage.EARLY_INFANCY: 10.0,
            DevelopmentalStage.MIDDLE_INFANCY: 20.0,
            DevelopmentalStage.LATE_INFANCY: 30.0,
            DevelopmentalStage.EARLY_TODDLER: 45.0,
            DevelopmentalStage.LATE_TODDLER: 60.0,
            DevelopmentalStage.EARLY_PRESCHOOL: 120.0,
            DevelopmentalStage.LATE_PRESCHOOL: 180.0,
            DevelopmentalStage.EARLY_CHILDHOOD: 300.0,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 600.0
        }
        
        if self.development_tracker.current_stage in stage_spans:
            base_span = stage_spans[self.development_tracker.current_stage]
        
        return base_span * attention_multiplier
    
    def _get_emotional_complexity(self) -> float:
        """Calculate the current emotional complexity"""
        # Access the emotional complexity directly from the emotions network
        if "emotions" in self._networks:
            emotions_network = self._networks["emotions"].model
            if hasattr(emotions_network, "emotional_complexity"):
                return emotions_network.emotional_complexity
        
        # Fallback calculation
        return 0.1 + (
            self.development_tracker.domain_development.get(DevelopmentalDomain.EMOTIONAL, 0.0) * 0.9
        )
    
    def process_perception(self, modality: SensoryModality, content: Any, 
                          intensity: float = 0.5) -> UUID:
        """
        Process a sensory input through the perception network.
        
        Args:
            modality: The sensory modality of the input
            content: The content of the perception
            intensity: The intensity of the perception
            
        Returns:
            ID of the created percept
        """
        # Get the perception network
        perception_network = self._networks["perception"].model
        
        # Process the input and create a percept
        percept_id = perception_network.process_input(modality, content, intensity)
        
        # Increase perception network activation
        self._networks["perception"].update_activation(
            self._networks["perception"].activation + intensity * 0.3
        )
        
        # Create a memory item for the perception
        memory_attributes = MemoryAttributes(
            strength=0.7,
            emotional_valence=EmotionalValence.NEUTRAL,
            emotional_intensity=0.1,
            accessibility=1.0,
            importance=0.5
        )
        
        # Adjust attributes based on percept intensity
        memory_attributes.strength = min(1.0, memory_attributes.strength + intensity * 0.2)
        memory_attributes.importance = min(1.0, memory_attributes.importance + intensity * 0.3)
        
        # Create a memory item
        memory_id = self.working_memory.add_item(
            content={"type": "percept", "modality": modality.value, "content": content, "percept_id": percept_id},
            memory_type=MemoryType.WORKING,
            attributes=memory_attributes,
            tags=["percept", f"modality:{modality.value}"]
        )
        
        # Try to focus attention on this percept if it's intense enough
        if intensity > 0.6:
            attention_network = self._networks["attention"].model
            attention_network.focus_on(percept_id)
            self.working_memory.set_focus(memory_id, True)
            
            # Increase attention network activation
            self._networks["attention"].update_activation(
                self._networks["attention"].activation + intensity * 0.4
            )
        
        return percept_id
    
    def feel_emotion(self, emotion: Emotion, intensity: float = 0.5) -> None:
        """
        Generate an emotional response.
        
        Args:
            emotion: The emotion to feel
            intensity: The intensity of the emotion
        """
        # Get the emotions network
        emotions_network = self._networks["emotions"].model
        
        # Generate the emotional response
        emotions_network.feel_emotion(emotion, intensity)
        
        # Update moods based on emotion
        moods_network = self._networks["moods"].model
        moods_network.update_mood({emotion: intensity})
        
        # Create a memory item for the emotion
        memory_attributes = MemoryAttributes(
            strength=0.8,
            emotional_valence=self._map_emotion_to_valence(emotion),
            emotional_intensity=intensity,
            accessibility=0.9,
            importance=0.6
        )
        
        # Create a memory item
        self.working_memory.add_item(
            content={"type": "emotion", "emotion": emotion.value, "intensity": intensity},
            memory_type=MemoryType.EMOTIONAL,
            attributes=memory_attributes,
            tags=["emotion", f"emotion:{emotion.value}"]
        )
    
    def _map_emotion_to_valence(self, emotion: Emotion) -> EmotionalValence:
        """Map an emotion to an emotional valence"""
        positive_emotions = [Emotion.JOY, Emotion.TRUST, Emotion.ANTICIPATION, Emotion.PRIDE, Emotion.LOVE]
        negative_emotions = [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST, 
                           Emotion.GUILT, Emotion.SHAME, Emotion.GRIEF, Emotion.JEALOUSY]
        
        if emotion in positive_emotions:
            return EmotionalValence.POSITIVE
        elif emotion in negative_emotions:
            return EmotionalValence.NEGATIVE
        else:
            return EmotionalValence.NEUTRAL
    
    def create_thought(self, thought_type: ThoughtType, content: Any,
                      related_thought_ids: List[UUID] = None) -> UUID:
        """
        Create a new thought.
        
        Args:
            thought_type: Type of thought
            content: Content of the thought
            related_thought_ids: Optional IDs of related thoughts
            
        Returns:
            ID of the created thought
        """
        # Get the thoughts network
        thoughts_network = self._networks["thoughts"].model
        
        # Create the thought
        thought_id = thoughts_network.create_thought(thought_type, content, related_thought_ids)
        
        # Increase thoughts network activation
        self._networks["thoughts"].update_activation(
            self._networks["thoughts"].activation + 0.3
        )
        
        # Create a memory item for the thought
        memory_attributes = MemoryAttributes(
            strength=0.7,
            emotional_valence=EmotionalValence.NEUTRAL,
            emotional_intensity=0.2,
            accessibility=0.8,
            importance=0.5
        )
        
        # Create a memory item
        self.working_memory.add_item(
            content={"type": "thought", "thought_type": thought_type.value, "content": content, "thought_id": thought_id},
            memory_type=MemoryType.WORKING,
            attributes=memory_attributes,
            tags=["thought", f"thought_type:{thought_type.value}"]
        )
        
        # Try to bring thought to consciousness
        consciousness_network = self._networks["consciousness"].model
        consciousness_network.bring_to_consciousness(thought_id)
        
        return thought_id
    
    def update_drive(self, drive_type: DriveType, intensity: float) -> None:
        """
        Update the intensity of a drive.
        
        Args:
            drive_type: The drive to update
            intensity: The new intensity level
        """
        # Get the drives network
        drives_network = self._networks["drives"].model
        
        # Update the drive
        drives_network.update_drive(drive_type, intensity)
        
        # If the drive is strong, create a thought about it
        if intensity > 0.7:
            self.create_thought(
                ThoughtType.GOAL,
                f"Need to satisfy {drive_type.value} drive"
            )
    
    def satisfy_drive(self, drive_type: DriveType, amount: float = 0.3) -> None:
        """
        Satisfy a drive by reducing its intensity.
        
        Args:
            drive_type: The drive to satisfy
            amount: Amount to reduce the drive intensity
        """
        # Get the drives network
        drives_network = self._networks["drives"].model
        
        # Initial intensity
        initial_intensity = 0
        if drive_type in drives_network.drives:
            initial_intensity = drives_network.drives[drive_type]
        
        # Satisfy the drive
        drives_network.satisfy_drive(drive_type, amount)
        
        # If the drive was significantly reduced, generate a positive emotion
        if initial_intensity > 0.6 and initial_intensity - amount <= 0.3:
            self.feel_emotion(Emotion.JOY, 0.6)
    
    def produce_utterance(self) -> Optional[str]:
        """
        Generate a language utterance based on current state.
        
        Returns:
            A generated utterance or None if unable to produce language
        """
        # Check if language production is developmentally appropriate
        current_stage = self.development_tracker.current_stage
        if current_stage in [DevelopmentalStage.NEWBORN, DevelopmentalStage.EARLY_INFANCY]:
            # Pre-linguistic - just produce coos or cries
            if self._networks["emotions"].activation > 0.7:
                # High emotional activation - cry or laugh
                emotion = max(self._networks["emotions"].model.emotions.items(), 
                             key=lambda x: x[1], default=(None, 0))
                if emotion[0] == Emotion.JOY:
                    return "*laughs*"
                else:
                    return "*cries*"
            else:
                return "*coos*"
        
        # Get content for utterance based on active thoughts and drives
        utterance_content = self._generate_utterance_content()
        
        # Use language production to generate the utterance
        utterance = self.language_production.generate_utterance(
            content=utterance_content,
            vocabulary=self.vocabulary,
            syntax=self.syntax,
            developmental_stage=current_stage
        )
        
        return utterance
    
    def _generate_utterance_content(self) -> Dict[str, Any]:
        """
        Generate content for an utterance based on internal state.
        
        Returns:
            Dictionary containing content for utterance generation
        """
        # Content will be based on:
        # 1. Active thoughts
        # 2. Strong drives
        # 3. Current emotions
        # 4. Current percepts
        
        content = {
            "subject": None,
            "action": None,
            "object": None,
            "modifier": None,
            "emotion": None
        }
        
        # Check for active thoughts
        thoughts_network = self._networks["thoughts"].model
        active_thoughts = list(thoughts_network.active_thoughts.values())
        if active_thoughts:
            # Take the most recent thought
            thought = active_thoughts[-1]
            if thought.type == ThoughtType.GOAL:
                content["action"] = "want"
                if isinstance(thought.content, str) and "drive" in thought.content:
                    content["object"] = thought.content.split()[3]  # Extract drive name
            elif thought.type == ThoughtType.PERCEPTION:
                content["action"] = "see"
                content["object"] = str(thought.content)
        
        # Check for strong drives
        drives_network = self._networks["drives"].model
        strong_drives = [(d, v) for d, v in drives_network.drives.items() if v > 0.6]
        if strong_drives:
            strongest_drive = max(strong_drives, key=lambda x: x[1])
            content["subject"] = "I"
            content["action"] = "want"
            if strongest_drive[0] == DriveType.PHYSIOLOGICAL:
                content["object"] = "food"
            elif strongest_drive[0] == DriveType.SAFETY:
                content["object"] = "safe"
            elif strongest_drive[0] == DriveType.BELONGING:
                content["object"] = "hug"
        
        # Check for strong emotions
        emotions_network = self._networks["emotions"].model
        strong_emotions = [(e, v) for e, v in emotions_network.emotions.items() if v > 0.5]
        if strong_emotions:
            strongest_emotion = max(strong_emotions, key=lambda x: x[1])
            content["emotion"] = strongest_emotion[0].value
            
            # Simple emotion expressions
            if strongest_emotion[0] == Emotion.JOY:
                content["subject"] = "I"
                content["action"] = "happy"
            elif strongest_emotion[0] == Emotion.SADNESS:
                content["subject"] = "I"
                content["action"] = "sad"
        
        return content
    
    def process_verbal_input(self, utterance: str) -> None:
        """
        Process a verbal input (e.g., from mother).
        
        Args:
            utterance: The verbal input to process
        """
        # Process through perception
        percept_id = self.process_perception(
            modality=SensoryModality.AUDITORY,
            content=utterance,
            intensity=0.8  # Verbal inputs tend to be salient
        )
        
        # Try to understand the utterance based on current language capabilities
        understanding = self.syntax.parse_utterance(
            utterance=utterance,
            vocabulary=self.vocabulary,
            developmental_stage=self.development_tracker.current_stage
        )
        
        # Create a thought about the understood content
        if understanding:
            self.create_thought(
                thought_type=ThoughtType.PERCEPTION,
                content=understanding
            )
            
            # Add to vocabulary if new words
            if "new_words" in understanding and understanding["new_words"]:
                for word in understanding["new_words"]:
                    self.vocabulary.add_word(word)
        else:
            # Create a thought about not understanding
            self.create_thought(
                thought_type=ThoughtType.PERCEPTION,
                content="Unknown sounds"
            )
    
    def get_observable_state(self) -> Dict[str, Any]:
        """
        Get the current observable state of the child.
        This represents what the mother can observe.
        
        Returns:
            Dictionary containing observable state information
        """
        # Get active emotions
        emotions_network = self._networks["emotions"].model
        active_emotions = {e.value: v for e, v in emotions_network.emotions.items() if v > 0.2}
        
        # Get current mood
        moods_network = self._networks["moods"].model
        current_mood = {
            "mood": moods_network.current_mood.value,
            "intensity": moods_network.mood_intensity
        }
        
        # Get strong drives
        drives_network = self._networks["drives"].model
        active_drives = {d.value: v for d, v in drives_network.drives.items() if v > 0.5}
        
        # Get attention focus
        attention_network = self._networks["attention"].model
        focused_items = attention_network.focused_items
        
        # Determine facial expressions and body language
        facial_expression = self._determine_facial_expression(active_emotions)
        body_language = self._determine_body_language(active_emotions, active_drives)
        
        # Construct observable state
        observable_state = {
            "developmental_stage": self.development_tracker.current_stage.value,
            "simulated_age_months": round(self.development_tracker.simulated_age_months, 1),
            "facial_expression": facial_expression,
            "body_language": body_language,
            "vocalizations": self.produce_utterance(),
            "emotional_state": {
                "visible_emotions": active_emotions,
                "current_mood": current_mood
            },
            "attention": {
                "focused": len(focused_items) > 0,
                "attention_span_seconds": self._get_attention_span()
            },
            "activity_level": self._calculate_activity_level(),
            "language_development": {
                "vocabulary_size": self.vocabulary.total_vocabulary_size,
                "mean_utterance_length": self.language_production.mean_utterance_length
            }
        }
        
        return observable_state
    
    def _determine_facial_expression(self, active_emotions: Dict[str, float]) -> str:
        """Determine the facial expression based on active emotions"""
        if not active_emotions:
            return "neutral"
            
        strongest_emotion = max(active_emotions.items(), key=lambda x: x[1])
        
        if strongest_emotion[0] == "joy":
            return "smiling"
        elif strongest_emotion[0] == "sadness":
            return "frowning"
        elif strongest_emotion[0] == "anger":
            return "furrowed brow"
        elif strongest_emotion[0] == "fear":
            return "wide-eyed"
        elif strongest_emotion[0] == "disgust":
            return "scrunched nose"
        elif strongest_emotion[0] == "surprise":
            return "open-mouthed"
        else:
            return "neutral"
    
    def _determine_body_language(self, active_emotions: Dict[str, float], 
                                active_drives: Dict[str, float]) -> str:
        """Determine body language based on emotions and drives"""
        # Handle strong physiological drives first
        if "physiological" in active_drives and active_drives["physiological"] > 0.7:
            return "restless, signaling hunger"
            
        # Then safety/belonging needs
        if "safety" in active_drives and active_drives["safety"] > 0.7:
            return "seeking proximity, clingy"
            
        # Then emotional states
        if active_emotions:
            strongest_emotion = max(active_emotions.items(), key=lambda x: x[1])
            
            if strongest_emotion[0] == "joy":
                return "animated movement, reaching out"
            elif strongest_emotion[0] == "sadness":
                return "slumped, inactive"
            elif strongest_emotion[0] == "anger":
                return "tense, stiff"
            elif strongest_emotion[0] == "fear":
                return "withdrawn, protective posture"
            else:
                return "normal movement"
        
        # Default body language
        return "relaxed, normal movement"
    
    def _calculate_activity_level(self) -> float:
        """Calculate the overall activity level of the child"""
        # Based on emotions, drives, and attention
        emotions_activation = self._networks["emotions"].activation
        drives_activation = self._networks["drives"].activation
        attention_activation = self._networks["attention"].activation
        
        # Weighted combination
        activity_level = (
            emotions_activation * 0.4 +
            drives_activation * 0.4 +
            attention_activation * 0.2
        )
        
        return activity_level
    
    def process_mother_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Process an interaction from the mother.
        
        Args:
            interaction: Dictionary containing structured interaction from mother
        """
        # Process different types of interactions
        if "speech" in interaction:
            # Verbal interaction
            self.process_verbal_input(interaction["speech"])
            
            # Create an emotional response based on tone
            if "tone" in interaction:
                if interaction["tone"] == "soothing":
                    self.feel_emotion(Emotion.JOY, 0.6)
                elif interaction["tone"] == "firm":
                    self.feel_emotion(Emotion.FEAR, 0.3)
                elif interaction["tone"] == "encouraging":
                    self.feel_emotion(Emotion.JOY, 0.5)
                    
        elif "action" in interaction and "physical_interaction" in interaction:
            # Physical interaction
            action = interaction["action"]
            intensity = interaction.get("intensity", 0.5)
            
            # Process through tactile perception
            self.process_perception(
                modality=SensoryModality.TACTILE,
                content=action,
                intensity=intensity
            )
            
            # Generate emotional response based on action
            if "hug" in action.lower() or "cuddle" in action.lower():
                self.feel_emotion(Emotion.JOY, 0.7)
                self.satisfy_drive(DriveType.BELONGING, 0.4)
            elif "feed" in action.lower():
                self.satisfy_drive(DriveType.PHYSIOLOGICAL, 0.6)
                self.feel_emotion(Emotion.JOY, 0.5)
            elif "comfort" in action.lower():
                self.feel_emotion(Emotion.TRUST, 0.6)
                
        elif "concept" in interaction and "teaching_moment" in interaction:
            # Teaching moment
            concept = interaction["concept"]
            speech = interaction.get("speech", "")
            
            # Process the verbal component
            self.process_verbal_input(speech)
            
            # Attempt to learn the concept
            if self.vocabulary.can_learn_word(concept):
                self.vocabulary.add_word(concept)
                
                # Create a memory with the learned concept
                memory_attributes = MemoryAttributes(
                    strength=0.8,
                    emotional_valence=EmotionalValence.POSITIVE,
                    emotional_intensity=0.4,
                    accessibility=0.9,
                    importance=0.7
                )
                
                self.working_memory.add_item(
                    content={"type": "learned_concept", "concept": concept},
                    memory_type=MemoryType.SEMANTIC,
                    attributes=memory_attributes,
                    tags=["concept", "learning", f"word:{concept}"]
                )
                
                # Satisfy curiosity drive
                self.satisfy_drive(DriveType.CURIOSITY, 0.3)
                
                # Feel positive emotion from learning
                self.feel_emotion(Emotion.JOY, 0.5)
        
        # Handle other interaction types as needed