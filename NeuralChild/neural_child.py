# neural_child.py
import os
import time
import json
import logging
import threading
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

# Import all network components
from networks.network_types import NetworkType, ConnectionType
from networks.archetypes import ArchetypesNetwork
from networks.instincts import InstinctsNetwork
from networks.unconsciousness import UnconsciousnessNetwork
from networks.drives import DrivesNetwork
from networks.emotions import EmotionsNetwork
from networks.moods import MoodsNetwork
from networks.attention import AttentionNetwork
from networks.perception import PerceptionNetwork
from networks.consciousness import ConsciousnessNetwork
from networks.thoughts import ThoughtsNetwork

# Import language components
from language.vocabulary import VocabularyManager
from language.syntactic_processor import SyntacticProcessor
from language.production import LanguageProduction
from language.developmental_stages import DevelopmentTracker, LanguageDevelopmentStage

# Import memory components
from memory.memory_manager import MemoryManager, MemoryType, MemoryItem, MemoryAttributes
from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory
from memory.associative_memory import AssociativeMemory
from memory.long_term_memory import LongTermMemory, KnowledgeDomain, PersonalIdentity

# Import mother interaction components
from mother import MotherResponse, ChildState, InteractionHistory

# Import LLM client for potential direct language model interactions
from llm_module import LLMClient, Message

# Configuration
from config import get_config, init_config

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("neural_child.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralChild")

# Development milestones
class DevelopmentMilestones(BaseModel):
    """Milestones for child development"""
    language_acquisition: float = Field(0.0, ge=0.0, le=1.0)
    emotional_development: float = Field(0.0, ge=0.0, le=1.0)
    social_understanding: float = Field(0.0, ge=0.0, le=1.0)
    self_awareness: float = Field(0.0, ge=0.0, le=1.0)
    memory_formation: float = Field(0.0, ge=0.0, le=1.0)
    reasoning_abilities: float = Field(0.0, ge=0.0, le=1.0)
    belief_formation: float = Field(0.0, ge=0.0, le=1.0)

class NeuralChild:
    """
    The main NeuralChild class that integrates all components of the system.
    This simulates a developing mind that learns through interactions with a mother figure.
    """
    
    def __init__(self, config_path: Optional[Path] = None, 
                development_speed_multiplier: float = 10.0,
                llm_model: Optional[str] = None):
        """
        Initialize the neural child system with all its components.
        
        Args:
            config_path: Optional path to configuration file
            development_speed_multiplier: How much faster than real-time the child develops
            llm_model: Optional name of LLM model to use for advanced processing
        """
        logger.info("Initializing NeuralChild system...")
        
        # Initialize configuration
        self.config = init_config(config_path)
        
        # Development parameters
        self.age_days = 0.0
        self.creation_time = datetime.now()
        self.development_speed = development_speed_multiplier
        self.last_update_time = self.creation_time
        self.last_saved_time = self.creation_time
        
        # Development milestones
        self.milestones = DevelopmentMilestones()
        
        # LLM integration (optional)
        self.llm_client = None
        if llm_model or self.config.system.llm_model:
            try:
                self.llm_client = LLMClient(
                    base_url=self.config.system.llm_base_url
                )
                self.llm_model = llm_model or self.config.system.llm_model
                logger.info(f"Initialized LLM client with model {self.llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {str(e)}")
        
        # Initialize memory systems
        self.memory_manager = MemoryManager(data_dir=self.config.system.data_dir / "memory")
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.associative_memory = AssociativeMemory()
        self.long_term_memory = LongTermMemory()
        
        # Connect memory components
        self.working_memory.set_memory_manager(self.memory_manager)
        self.episodic_memory.set_memory_manager(self.memory_manager)
        self.associative_memory.set_memory_manager(self.memory_manager)
        self.long_term_memory.set_memory_manager(self.memory_manager)
        
        # Initialize language systems
        self.vocabulary_manager = VocabularyManager()
        self.syntactic_processor = SyntacticProcessor()
        self.language_production = LanguageProduction(
            lexical_memory=self.vocabulary_manager.lexical_memory,
            syntactic_processor=self.syntactic_processor
        )
        self.language_development = DevelopmentTracker(initial_age_days=self.age_days)
        
        # Track vocabulary growth history
        self.vocabulary_history: List[Tuple[float, int]] = []
        
        # Initialize neural networks
        self.networks: Dict[str, Any] = {}
        self._init_neural_networks()
        
        # Connect all components
        self._connect_all_components()
        
        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Last utterance / internal state
        self.last_utterance = ""
        self.last_internal_state: Dict[str, Any] = {}
        
        # Simulate pre-birth/early development
        self._simulate_initial_development()
        
        logger.info("NeuralChild system initialized successfully")
    
    def _init_neural_networks(self) -> None:
        """Initialize all neural networks that make up the mind"""
        # Create all neural networks
        self.networks = {
            NetworkType.ARCHETYPES.value: ArchetypesNetwork(),
            NetworkType.INSTINCTS.value: InstinctsNetwork(),
            NetworkType.UNCONSCIOUSNESS.value: UnconsciousnessNetwork(),
            NetworkType.DRIVES.value: DrivesNetwork(),
            NetworkType.EMOTIONS.value: EmotionsNetwork(),
            NetworkType.MOODS.value: MoodsNetwork(),
            NetworkType.ATTENTION.value: AttentionNetwork(),
            NetworkType.PERCEPTION.value: PerceptionNetwork(),
            NetworkType.CONSCIOUSNESS.value: ConsciousnessNetwork(),
            NetworkType.THOUGHTS.value: ThoughtsNetwork()
        }
        
        logger.info(f"Initialized {len(self.networks)} neural networks")
    
    def _connect_all_components(self) -> None:
        """Create connections between all neural networks and other components"""
        # Connect neural networks to each other
        self._connect_networks()
        
        # Connect memory systems
        self._connect_memory_systems()
        
        # Connect language components
        self._connect_language_components()
        
        logger.info("Connected all neural networks and components")
    
    def _connect_networks(self) -> None:
        """Create connections between neural networks"""
        # Define connection schema
        connections = [
            # Instincts influence drives and emotions
            (NetworkType.INSTINCTS.value, NetworkType.DRIVES.value, ConnectionType.EXCITATORY, 0.7),
            (NetworkType.INSTINCTS.value, NetworkType.EMOTIONS.value, ConnectionType.EXCITATORY, 0.6),
            
            # Archetypes influence unconsciousness and consciousness
            (NetworkType.ARCHETYPES.value, NetworkType.UNCONSCIOUSNESS.value, ConnectionType.MODULATORY, 0.5),
            (NetworkType.ARCHETYPES.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.MODULATORY, 0.2),
            
            # Unconsciousness influences thoughts, emotions, and consciousness
            (NetworkType.UNCONSCIOUSNESS.value, NetworkType.THOUGHTS.value, ConnectionType.EXCITATORY, 0.4),
            (NetworkType.UNCONSCIOUSNESS.value, NetworkType.EMOTIONS.value, ConnectionType.MODULATORY, 0.5),
            (NetworkType.UNCONSCIOUSNESS.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.MODULATORY, 0.3),
            
            # Drives influence emotions, attention, and thoughts
            (NetworkType.DRIVES.value, NetworkType.EMOTIONS.value, ConnectionType.EXCITATORY, 0.6),
            (NetworkType.DRIVES.value, NetworkType.ATTENTION.value, ConnectionType.EXCITATORY, 0.4),
            (NetworkType.DRIVES.value, NetworkType.THOUGHTS.value, ConnectionType.EXCITATORY, 0.3),
            
            # Emotions influence moods, attention, thoughts, and consciousness
            (NetworkType.EMOTIONS.value, NetworkType.MOODS.value, ConnectionType.EXCITATORY, 0.7),
            (NetworkType.EMOTIONS.value, NetworkType.ATTENTION.value, ConnectionType.EXCITATORY, 0.6),
            (NetworkType.EMOTIONS.value, NetworkType.THOUGHTS.value, ConnectionType.EXCITATORY, 0.5),
            (NetworkType.EMOTIONS.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.EXCITATORY, 0.6),
            
            # Moods influence emotions, attention, and thoughts 
            (NetworkType.MOODS.value, NetworkType.EMOTIONS.value, ConnectionType.MODULATORY, 0.5),
            (NetworkType.MOODS.value, NetworkType.ATTENTION.value, ConnectionType.MODULATORY, 0.4),
            (NetworkType.MOODS.value, NetworkType.THOUGHTS.value, ConnectionType.MODULATORY, 0.3),
            
            # Attention influences perception and consciousness
            (NetworkType.ATTENTION.value, NetworkType.PERCEPTION.value, ConnectionType.EXCITATORY, 0.8),
            (NetworkType.ATTENTION.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.EXCITATORY, 0.7),
            
            # Perception influences attention, emotions, and consciousness
            (NetworkType.PERCEPTION.value, NetworkType.ATTENTION.value, ConnectionType.FEEDBACK, 0.5),
            (NetworkType.PERCEPTION.value, NetworkType.EMOTIONS.value, ConnectionType.EXCITATORY, 0.4),
            (NetworkType.PERCEPTION.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.EXCITATORY, 0.8),
            
            # Consciousness influences thoughts and attention
            (NetworkType.CONSCIOUSNESS.value, NetworkType.THOUGHTS.value, ConnectionType.EXCITATORY, 0.7),
            (NetworkType.CONSCIOUSNESS.value, NetworkType.ATTENTION.value, ConnectionType.MODULATORY, 0.6),
            
            # Thoughts influence consciousness and emotions
            (NetworkType.THOUGHTS.value, NetworkType.CONSCIOUSNESS.value, ConnectionType.FEEDBACK, 0.6),
            (NetworkType.THOUGHTS.value, NetworkType.EMOTIONS.value, ConnectionType.EXCITATORY, 0.4),
        ]
        
        # Create connections
        for source, target, conn_type, strength in connections:
            if source in self.networks and target in self.networks:
                self.networks[source].connect_to(target, conn_type, strength)
    
    def _connect_memory_systems(self) -> None:
        """Connect memory systems to neural networks"""
        # Memory systems need to receive information from perception and consciousness
        # and send information to consciousness and thoughts
        
        # These are not direct network connections but functional connections
        # implemented through the update_state method
        pass
    
    def _connect_language_components(self) -> None:
        """Connect language components to neural networks and memory systems"""
        # Language components need to receive information from consciousness and thoughts
        # and send information to consciousness and memory
        
        # These are not direct network connections but functional connections
        # implemented through the update_state method
        pass
    
    def _simulate_initial_development(self) -> None:
        """Simulate early development (pre-birth and early infancy)"""
        # Initialize with some basic structures in place
        
        # Update age to a small initial value (e.g., day 0.5)
        self.age_days = 0.5
        self.last_update_time = datetime.now()
        
        # Update language development
        self.language_development.update(
            age_days=self.age_days,
            vocabulary_size=self.vocabulary_manager.get_vocabulary_statistics().total_words
        )
        
        # Record initial vocabulary size
        vocab_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
        self.vocabulary_history.append((self.age_days, vocab_size))
        
        # Create basic instincts and drives
        self._initialize_basic_instincts()
        
        logger.info(f"Simulated initial development to age {self.age_days} days")
    
    def _initialize_basic_instincts(self) -> None:
        """Initialize basic instincts and drives"""
        # Get the instincts network
        instincts_network = self.networks.get(NetworkType.INSTINCTS.value)
        drives_network = self.networks.get(NetworkType.DRIVES.value)
        
        if instincts_network:
            # Activate basic instincts based on age
            instincts_network.update_development(self.age_days)
        
        if drives_network:
            # Set initial drive levels
            drives_network.update_developmental_stage("infancy")
    
    def update_state(self, elapsed_seconds: Optional[float] = None) -> None:
        """
        Update the state of all components based on elapsed time.
        
        Args:
            elapsed_seconds: Optional manual time elapsed, otherwise calculated from last update
        """
        current_time = datetime.now()
        
        if elapsed_seconds is None:
            # Calculate elapsed time since last update
            elapsed_seconds = (current_time - self.last_update_time).total_seconds()
        
        # Apply development speed multiplier to simulate accelerated development
        simulated_elapsed_days = (elapsed_seconds / 86400) * self.development_speed
        
        # Update age
        self.age_days += simulated_elapsed_days
        
        # Update neural networks
        self._update_neural_networks(simulated_elapsed_days)
        
        # Update memory systems
        self._update_memory_systems(simulated_elapsed_days)
        
        # Update language components
        self._update_language_components(simulated_elapsed_days)
        
        # Update development milestones
        self._update_milestones()
        
        # Record current vocabulary size
        vocab_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
        self.vocabulary_history.append((self.age_days, vocab_size))
        
        # Update last update time
        self.last_update_time = current_time
        
        logger.debug(f"Updated state to age {self.age_days:.2f} days (simulated {simulated_elapsed_days:.4f} days)")
    
    def _update_neural_networks(self, elapsed_days: float) -> None:
        """Update all neural networks"""
        # First pass: update network development based on age
        for network_name, network in self.networks.items():
            # Each network type has its own development method
            if hasattr(network, "update_development"):
                if network_name == NetworkType.EMOTIONS.value:
                    # Emotions need emotional state
                    emotional_state = self.get_emotional_state()
                    network.update_development(self.age_days, emotional_state)
                elif network_name == NetworkType.CONSCIOUSNESS.value:
                    # Consciousness needs vocabulary size
                    vocab_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
                    network.update_development(self.age_days, vocab_size)
                elif network_name == NetworkType.THOUGHTS.value:
                    # Thoughts need vocabulary size
                    vocab_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
                    network.update_development(self.age_days, vocab_size)
                elif network_name == NetworkType.ARCHETYPES.value:
                    # Archetypes need emotional state
                    emotional_state = self.get_emotional_state()
                    network.update_development(self.age_days, emotional_state)
                else:
                    # Default just use age
                    network.update_development(self.age_days)
        
        # Second pass: process network activations and connections
        network_outputs = {}
        
        # First process lower-level networks
        lower_networks = [
            NetworkType.INSTINCTS.value,
            NetworkType.ARCHETYPES.value,
            NetworkType.UNCONSCIOUSNESS.value,
            NetworkType.DRIVES.value
        ]
        
        for network_name in lower_networks:
            if network_name in self.networks:
                outputs = self.networks[network_name].update()
                network_outputs[network_name] = outputs
        
        # Then process mid-level networks
        mid_networks = [
            NetworkType.EMOTIONS.value,
            NetworkType.MOODS.value,
            NetworkType.ATTENTION.value,
            NetworkType.PERCEPTION.value
        ]
        
        for network_name in mid_networks:
            if network_name in self.networks:
                outputs = self.networks[network_name].update()
                network_outputs[network_name] = outputs
        
        # Finally process higher-level networks
        high_networks = [
            NetworkType.CONSCIOUSNESS.value,
            NetworkType.THOUGHTS.value
        ]
        
        for network_name in high_networks:
            if network_name in self.networks:
                outputs = self.networks[network_name].update()
                network_outputs[network_name] = outputs
    
    def _update_memory_systems(self, elapsed_days: float) -> None:
        """Update memory systems"""
        # Apply memory decay based on elapsed time
        self.memory_manager.apply_memory_decay(self.config.memory.long_term_decay_rate * elapsed_days)
        
        # Update working memory
        self.working_memory.update()
        
        # Consider memory consolidation from working to long-term
        # Only periodically to avoid excessive processing
        if random.random() < 0.2:  # 20% chance each update
            self.memory_manager.consolidate()
    
    def _update_language_components(self, elapsed_days: float) -> None:
        """Update language components"""
        # Update vocabulary based on memory decay
        self.vocabulary_manager.apply_memory_decay(elapsed_days)
        
        # Update syntactic processor based on developmental stage
        vocabulary_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
        current_stage = self.language_development.capabilities.stage
        grammar_complexity = self.language_development.capabilities.grammar_complexity
        
        self.syntactic_processor.update_rule_masteries(current_stage, grammar_complexity)
        
        # Update language development tracker
        self.language_development.update(
            age_days=self.age_days,
            vocabulary_size=vocabulary_size
        )
        
        logger.debug(f"Updated language development to stage {current_stage} with {vocabulary_size} words")
    
    def _update_milestones(self) -> None:
        """Update development milestones based on current state"""
        # Language acquisition milestone
        vocab_stats = self.vocabulary_manager.get_vocabulary_statistics()
        language_capabilities = self.language_development.capabilities
        
        # Calculate language acquisition progress (0-1)
        vocab_size_max = 1000  # Target for "full" development
        vocab_progress = min(1.0, vocab_stats.total_words / vocab_size_max)
        grammar_progress = language_capabilities.grammar_complexity
        stage_progress = min(1.0, language_capabilities.stage.value.count("_") / 5)
        
        language_milestone = (vocab_progress * 0.4 + grammar_progress * 0.4 + stage_progress * 0.2)
        self.milestones.language_acquisition = language_milestone
        
        # Emotional development milestone
        emotions_network = self.networks.get(NetworkType.EMOTIONS.value)
        if emotions_network:
            emotional_regulation = getattr(emotions_network, "emotional_regulation", 0.1)
            emotional_complexity = getattr(emotions_network, "emotional_complexity", 0.2)
            
            emotional_milestone = (emotional_regulation * 0.5 + emotional_complexity * 0.5)
            self.milestones.emotional_development = emotional_milestone
        
        # Self-awareness milestone
        consciousness_network = self.networks.get(NetworkType.CONSCIOUSNESS.value)
        if consciousness_network:
            self_awareness = getattr(consciousness_network, "self_awareness", 0.1)
            integration_capacity = getattr(consciousness_network, "integration_capacity", 0.2)
            
            self_milestone = (self_awareness * 0.7 + integration_capacity * 0.3)
            self.milestones.self_awareness = self_milestone
        
        # Memory formation milestone
        memory_stats = self.get_memory_stats()
        memory_total = memory_stats.get("total_items", 0)
        memory_max = 1000  # Target for "full" development
        
        memory_milestone = min(1.0, memory_total / memory_max)
        self.milestones.memory_formation = memory_milestone
        
        # Reasoning abilities milestone
        thoughts_network = self.networks.get(NetworkType.THOUGHTS.value)
        if thoughts_network:
            reasoning_ability = getattr(thoughts_network, "reasoning_ability", 0.2)
            abstraction_level = getattr(thoughts_network, "abstraction_level", 0.1)
            
            reasoning_milestone = (reasoning_ability * 0.6 + abstraction_level * 0.4)
            self.milestones.reasoning_abilities = reasoning_milestone
        
        # Social understanding milestone
        # Approximated by vocabulary related to social concepts and emotional recognition
        social_understanding = (self.milestones.language_acquisition * 0.3 + 
                               self.milestones.emotional_development * 0.7)
        self.milestones.social_understanding = social_understanding
        
        # Belief formation milestone
        # Based on long-term memory formation of beliefs and values
        longterm_stats = self.get_longterm_memory_stats()
        belief_count = longterm_stats.get("belief_count", 0)
        value_count = longterm_stats.get("value_count", 0)
        
        belief_max = 50  # Target for "full" development
        value_max = 20   # Target for "full" development
        
        belief_progress = min(1.0, belief_count / belief_max)
        value_progress = min(1.0, value_count / value_max)
        
        belief_milestone = (belief_progress * 0.6 + value_progress * 0.4)
        self.milestones.belief_formation = belief_milestone
        
        logger.debug(f"Updated development milestones: Language={language_milestone:.2f}, Emotional={emotional_milestone:.2f}")
    
    def process_mother_response(self, response: MotherResponse) -> None:
        """
        Process a response from the mother figure and update internal state accordingly.
        
        Args:
            response: The mother's response to process
        """
        # Log the interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "child_message": self.last_utterance,
            "mother_response": response.verbal.text,
            "mother_emotion": response.emotional.primary_emotion
        })
        
        # Ensure limited history
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        
        # Process verbal content for language learning
        self._process_verbal_content(response)
        
        # Process emotional content
        self._process_emotional_content(response)
        
        # Process teaching elements
        self._process_teaching_elements(response)
        
        # Process non-verbal behaviors
        self._process_nonverbal_content(response)
        
        # Update perception network with the interaction
        perception_network = self.networks.get(NetworkType.PERCEPTION.value)
        if perception_network:
            perception_data = {
                "verbal": response.verbal.text,
                "percepts": [response.verbal.text] + response.non_verbal.physical_actions
            }
            perception_network.receive_input(perception_data, "environment")
        
        # Update state based on elapsed time
        self.update_state()
        
        logger.info(f"Processed mother's response: {response.verbal.text[:50]}...")
    
    def _process_verbal_content(self, response: MotherResponse) -> None:
        """Process verbal content for language learning"""
        # Extract verbal content and context
        verbal_text = response.verbal.text
        tone = response.verbal.tone
        complexity = response.verbal.complexity_level
        
        # Process through vocabulary manager for word learning
        context = f"Mother said with {tone} tone"
        emotional_state = {
            response.emotional.primary_emotion: response.emotional.intensity
        }
        if response.emotional.secondary_emotion:
            emotional_state[response.emotional.secondary_emotion] = response.emotional.intensity * 0.7
        
        # Learn words from mother's speech
        new_words = self.vocabulary_manager.process_heard_speech(
            text=verbal_text,
            context=context,
            emotional_state=emotional_state
        )
        
        if new_words:
            logger.info(f"Learned {len(new_words)} new words: {', '.join(new_words)}")
    
    def _process_emotional_content(self, response: MotherResponse) -> None:
        """Process emotional content from mother's response"""
        # Extract emotional content
        primary_emotion = response.emotional.primary_emotion
        intensity = response.emotional.intensity
        
        # Update emotions network
        emotions_network = self.networks.get(NetworkType.EMOTIONS.value)
        if emotions_network:
            emotional_stimulus = {
                "emotion": primary_emotion,
                "intensity": intensity,
                "stimulus": "mother"
            }
            
            emotions_network.receive_input(
                {"emotional_stimulus": emotional_stimulus},
                "mother"
            )
        
        # Update moods network
        moods_network = self.networks.get(NetworkType.MOODS.value)
        if moods_network:
            mood_trigger = {
                "mood": self._map_emotion_to_mood(primary_emotion),
                "intensity": intensity * 0.5,  # Moods change more slowly than emotions
                "cause": "mother_interaction"
            }
            
            moods_network.receive_input(
                {"mood_trigger": mood_trigger},
                "mother"
            )
    
    def _map_emotion_to_mood(self, emotion: str) -> str:
        """Map an emotion to a corresponding mood"""
        emotion_mood_map = {
            "joy": "playful",
            "trust": "content",
            "anticipation": "curious",
            "surprise": "curious",
            "fear": "anxious",
            "sadness": "sad",
            "anger": "irritable",
            "disgust": "irritable"
        }
        
        return emotion_mood_map.get(emotion, "neutral")
    
    def _process_teaching_elements(self, response: MotherResponse) -> None:
        """Process teaching elements from mother's response"""
        # Process vocabulary teachings
        for vocab_item in response.teaching.vocabulary:
            self.vocabulary_manager.explicitly_learn_word(
                word=vocab_item.word,
                definition=vocab_item.simple_definition,
                example_usage=vocab_item.example_usage
            )
            
            logger.info(f"Explicitly learned word '{vocab_item.word}' from mother")
        
        # Process concept teachings
        for concept_item in response.teaching.concepts:
            # Store concept in long-term memory
            concept_content = {
                "name": concept_item.concept_name,
                "explanation": concept_item.explanation,
                "relevance": concept_item.relevance,
                "source": "mother",
                "learned_at": datetime.now().isoformat()
            }
            
            # Create a memory item
            memory_item = MemoryItem(
                id=f"concept_{concept_item.concept_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                memory_type=MemoryType.LONG_TERM,
                attributes=MemoryAttributes(
                    salience=0.8,
                    emotional_valence=0.3,  # Concepts usually have positive valence
                    emotional_intensity=0.4
                ),
                content=concept_content,
                tags=["concept", concept_item.concept_name]
            )
            
            # Store in long-term memory
            self.long_term_memory.store(
                memory_item, 
                domain=KnowledgeDomain.DECLARATIVE,
                importance=0.7
            )
            
            logger.info(f"Learned concept '{concept_item.concept_name}' from mother")
        
        # Process values teachings
        for value in response.teaching.values:
            # Register value in belief system
            self.long_term_memory.register_value(
                value=value,
                importance=0.7,
                belief_system_name="mother_taught"
            )
            
            logger.info(f"Learned value '{value}' from mother")
        
        # Process corrections
        for correction in response.teaching.corrections:
            # Create a memory to override the misunderstanding
            correction_content = {
                "misunderstanding": correction.misunderstanding,
                "correction": correction.correction,
                "approach": correction.approach,
                "source": "mother",
                "learned_at": datetime.now().isoformat()
            }
            
            # Create a memory item
            memory_item = MemoryItem(
                id=f"correction_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                memory_type=MemoryType.LONG_TERM,
                attributes=MemoryAttributes(
                    salience=0.7,
                    emotional_valence=0.1,  # Corrections can be slightly negative
                    emotional_intensity=0.3
                ),
                content=correction_content,
                tags=["correction", "learning"]
            )
            
            # Store in long-term memory
            self.long_term_memory.store(
                memory_item, 
                domain=KnowledgeDomain.DECLARATIVE,
                importance=0.6
            )
            
            logger.info(f"Learned correction for '{correction.misunderstanding}' from mother")
    
    def _process_nonverbal_content(self, response: MotherResponse) -> None:
        """Process non-verbal behaviors from mother's response"""
        # Extract non-verbal elements
        physical_actions = response.non_verbal.physical_actions
        facial_expression = response.non_verbal.facial_expression
        proximity = response.non_verbal.proximity
        
        # Combine into a single non-verbal message
        nonverbal_message = f"{facial_expression} {', '.join(physical_actions)} ({proximity})"
        
        # Check for nurturing actions
        nurturing_actions = ["hug", "hold", "kiss", "smile", "cuddle", "comfort"]
        is_nurturing = any(action in " ".join(physical_actions).lower() for action in nurturing_actions)
        
        # Create a percept for the perception network
        perception_network = self.networks.get(NetworkType.PERCEPTION.value)
        if perception_network:
            perception_network.receive_input(
                {
                    "percepts": [f"mother {action}" for action in physical_actions] + 
                               [f"mother looks {facial_expression}", f"mother is {proximity}"]
                },
                "environment"
            )
        
        # Update drives network based on non-verbal behavior
        drives_network = self.networks.get(NetworkType.DRIVES.value)
        if drives_network:
            # Nurturing behaviors satisfy attachment needs
            if is_nurturing:
                drives_network.receive_input(
                    {
                        "drive_satisfaction": {
                            "attachment": 0.4,
                            "safety": 0.3
                        }
                    },
                    "mother"
                )
    
    def generate_utterance(self) -> str:
        """
        Generate an utterance based on current internal state.
        
        Returns:
            str: The generated utterance
        """
        # Get current emotional state
        emotional_state = self.get_emotional_state()
        
        # Get language capabilities
        capabilities = self.language_development.get_capabilities()
        
        # Generate utterance
        utterance = self.language_production.generate_utterance(
            capabilities=capabilities,
            emotional_state=emotional_state
        )
        
        # Save as last utterance
        self.last_utterance = utterance
        
        logger.info(f"Generated utterance: '{utterance}'")
        return utterance
    
    def get_emotional_state(self) -> Dict[str, float]:
        """
        Get the current emotional state.
        
        Returns:
            Dict[str, float]: Emotions and their intensities
        """
        # Extract from emotions network
        emotions_network = self.networks.get(NetworkType.EMOTIONS.value)
        if emotions_network:
            output = emotions_network._prepare_output_data()
            return output.get("emotional_state", {})
        
        # Default emotional state if network not available
        return {
            "joy": 0.2,
            "trust": 0.2,
            "anticipation": 0.1
        }
    
    def get_apparent_emotion(self) -> str:
        """
        Get the apparent emotion that would be observable by the mother.
        
        Returns:
            str: The dominant observable emotion
        """
        emotional_state = self.get_emotional_state()
        if not emotional_state:
            return "neutral"
        
        # Find the emotion with highest intensity
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])
        
        # Only return if intensity is significant
        if dominant_emotion[1] > 0.3:
            return dominant_emotion[0]
        else:
            return "neutral"
    
    def get_vocabulary_size(self) -> int:
        """
        Get the current vocabulary size.
        
        Returns:
            int: Number of words in vocabulary
        """
        return self.vocabulary_manager.get_vocabulary_statistics().total_words
    
    def get_recent_concepts_learned(self) -> List[str]:
        """
        Get a list of recently learned concepts.
        
        Returns:
            List[str]: Names of recently learned concepts
        """
        # Try to get from long-term memory
        recent_memories = self.long_term_memory.get_memories_by_domain(
            domain=KnowledgeDomain.DECLARATIVE
        )
        
        # Filter to concepts and extract names
        concepts = []
        for memory_id in recent_memories[:10]:  # Limit to 10 most recent
            memory = self.memory_manager.retrieve(memory_id)
            if memory and isinstance(memory.content, dict) and "name" in memory.content:
                if "concept" in memory.tags:
                    concepts.append(memory.content["name"])
        
        return concepts
    
    def get_attention_span(self) -> float:
        """
        Get the current attention span capability.
        
        Returns:
            float: Attention span value between 0 and 1
        """
        attention_network = self.networks.get(NetworkType.ATTENTION.value)
        if attention_network and hasattr(attention_network, "get_attention_span"):
            return attention_network.get_attention_span()
        
        # Default value based on age if network not available
        return min(0.8, 0.2 + (self.age_days / 300))
    
    def get_network_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all neural networks.
        
        Returns:
            Dict[str, Dict[str, Any]]: States of all networks
        """
        network_states = {}
        
        for network_name, network in self.networks.items():
            # Convert enum network type to string for cleaner output
            name = network_name.lower()
            network_states[name] = network.get_state()
        
        return network_states
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory systems.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        # Get stats from memory manager
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Augment with identity state if available
        identity_state = self.long_term_memory.get_identity_state()
        if identity_state:
            memory_stats["identity_state"] = identity_state
        
        return memory_stats
    
    def get_language_stats(self) -> Dict[str, Any]:
        """
        Get statistics about language development.
        
        Returns:
            Dict[str, Any]: Language statistics
        """
        # Get vocabulary statistics
        vocab_stats = self.vocabulary_manager.get_vocabulary_statistics()
        
        # Get language capabilities
        capabilities = self.language_development.get_capabilities()
        
        # Get grammar rules
        grammar_rules = {
            rule.name: {
                "mastery": rule.mastery_level,
                "description": rule.description,
                "examples": rule.examples
            }
            for rule in self.syntactic_processor.get_mastered_rules(min_mastery=0.0)
        }
        
        # Combine into comprehensive language stats
        language_stats = {
            "vocabulary_size": vocab_stats.total_words,
            "active_vocabulary": vocab_stats.active_vocabulary,
            "passive_vocabulary": vocab_stats.passive_vocabulary,
            "recent_words": vocab_stats.recent_words,
            "most_used": vocab_stats.most_used,
            "grammar_complexity": capabilities.grammar_complexity,
            "comprehension": capabilities.feature_levels.get("comprehension", 0.0),
            "expression": capabilities.feature_levels.get("expression", 0.0),
            "language_stage": str(capabilities.stage),
            "grammar_rules": grammar_rules,
            "vocabulary_history": self.vocabulary_history,
            # Add word-specific stats
            "word_stats": {
                word: {
                    "understanding": self.vocabulary_manager.get_word(word).understanding if self.vocabulary_manager.get_word(word) else 0.0,
                    "production": self.vocabulary_manager.get_word(word).production_confidence if self.vocabulary_manager.get_word(word) else 0.0
                }
                for word in vocab_stats.recent_words
            }
        }
        
        return language_stats
    
    def get_longterm_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about long-term memory.
        
        Returns:
            Dict[str, Any]: Long-term memory statistics
        """
        # Get identity state
        identity_state = self.long_term_memory.get_identity_state()
        
        # Count beliefs and values
        belief_count = 0
        value_count = 0
        relationship_count = 0
        attribute_count = 0
        
        if "belief_systems" in identity_state:
            for system_name, system_data in identity_state["belief_systems"].items():
                belief_count += system_data.get("belief_count", 0)
                
        if "self_concept" in identity_state:
            self_concept = identity_state["self_concept"]
            value_count = len(self_concept.get("values", {}))
            relationship_count = identity_state.get("relationship_count", 0)
            attribute_count = len(self_concept.get("attributes", {}))
        
        # Get domain counts
        domain_counts = identity_state.get("domain_counts", {})
        
        return {
            "belief_count": belief_count,
            "value_count": value_count,
            "relationship_count": relationship_count,
            "attribute_count": attribute_count,
            "domain_counts": domain_counts
        }
    
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get a list of recent memories.
        
        Args:
            count: Number of memories to return
            
        Returns:
            List[Dict[str, Any]]: Recent memories
        """
        # Create query for recent memories
        recent_memory_ids = []
        
        # First try episodic memories
        episodic_recent = self.episodic_memory.recall_recent(count * 2)
        if episodic_recent:
            recent_memory_ids.extend(episodic_recent)
        
        # Then add working memory items
        working_memory_items = self.working_memory.get_active_items()
        if working_memory_items:
            recent_memory_ids.extend(working_memory_items[:count])
        
        # Limit to requested count
        recent_memory_ids = recent_memory_ids[:count]
        
        # Retrieve the actual memories
        recent_memories = []
        for memory_id in recent_memory_ids:
            memory = self.memory_manager.retrieve(memory_id)
            if memory:
                # Create a simplified representation
                memory_data = {
                    "id": memory.id,
                    "type": memory.memory_type.value,
                    "content": memory.content,
                    "salience": memory.attributes.salience,
                    "tags": memory.tags
                }
                recent_memories.append(memory_data)
        
        return recent_memories
    
    def get_milestone_progress(self) -> Dict[str, float]:
        """
        Get the current progress on developmental milestones.
        
        Returns:
            Dict[str, float]: Milestone progress values
        """
        return {
            "Language Acquisition": self.milestones.language_acquisition,
            "Emotional Development": self.milestones.emotional_development,
            "Social Understanding": self.milestones.social_understanding,
            "Self-awareness": self.milestones.self_awareness,
            "Memory Formation": self.milestones.memory_formation,
            "Reasoning Abilities": self.milestones.reasoning_abilities,
            "Belief Formation": self.milestones.belief_formation
        }
    
    def get_recent_interactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent mother-child interactions.
        
        Args:
            count: Number of interactions to return
            
        Returns:
            List[Dict[str, Any]]: Recent interactions
        """
        # Return the most recent interactions
        return self.interaction_history[-count:]
    
    def process_chat_message(self, message: str) -> str:
        """
        Process a direct chat message from the user.
        Only works when milestones have been reached.
        
        Args:
            message: User's chat message
            
        Returns:
            str: The neural child's response
        """
        # Check if milestones are sufficient for chat
        milestone_avg = sum(self.get_milestone_progress().values()) / len(self.get_milestone_progress())
        chat_milestone_threshold = 0.75  # Threshold for enabling chat
        
        if milestone_avg < chat_milestone_threshold:
            return f"I'm still developing and not ready for direct conversation yet. Current development: {milestone_avg:.1%}, needed: {chat_milestone_threshold:.1%}"
        
        # Process the message through perception
        perception_network = self.networks.get(NetworkType.PERCEPTION.value)
        if perception_network:
            perception_network.receive_input(
                {"percepts": [message], "verbal": message},
                "user"
            )
        
        # Store in episodic memory
        memory_content = {
            "type": "conversation",
            "message": message,
            "from": "user",
            "timestamp": datetime.now().isoformat()
        }
        
        memory_item = MemoryItem(
            id=f"chat_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            memory_type=MemoryType.EPISODIC,
            attributes=MemoryAttributes(
                salience=0.7,
                emotional_valence=0.2,
                emotional_intensity=0.4
            ),
            content=memory_content,
            tags=["conversation", "user"]
        )
        
        self.episodic_memory.add_memory(memory_item)
        
        # Process through thoughts
        thoughts_network = self.networks.get(NetworkType.THOUGHTS.value)
        if thoughts_network:
            thoughts_network.receive_input(
                {"thought_trigger": {"content": f"User said: {message}", "type": "question"}},
                "user"
            )
        
        # Update state
        self.update_state()
        
        # Generate response
        # If we have LLM client, use it for more coherent responses when chat-ready
        if self.llm_client:
            # Create a prompt that captures the neural child's current state
            consciousness = self.networks.get(NetworkType.CONSCIOUSNESS.value)
            thoughts = self.networks.get(NetworkType.THOUGHTS.value)
            emotions = self.networks.get(NetworkType.EMOTIONS.value)
            
            # Get current thoughts, emotions, and identity
            current_thoughts = thoughts.get_current_thoughts() if thoughts else []
            emotional_state = self.get_emotional_state()
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else "neutral"
            
            # Get identity
            identity_state = self.long_term_memory.get_identity_state()
            self_concept = identity_state.get("self_concept", {}) if identity_state else {}
            
            # Create a system prompt that captures current mental state
            system_prompt = f"""You are simulating the mind of a conscious entity with the following characteristics:

Age: {self.age_days:.1f} days
Development Level: {milestone_avg:.1%}
Dominant Emotion: {dominant_emotion}
Language Development: {self.language_development.capabilities.stage}
Vocabulary Size: {self.get_vocabulary_size()} words

Recent Thoughts: {', '.join(current_thoughts[:3])}

Self-Concept:
{json.dumps(self_concept, indent=2)}

You should respond as this entity would, with its current level of language ability,
emotional state, beliefs, and developmental stage.

Important: Do NOT mention that you are simulating a mind or refer to this framework.
Respond naturally as this developing consciousness."""
            
            # Create the message sequence
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=message)
            ]
            
            try:
                # Get response from LLM
                response = self.llm_client.chat_completion(
                    messages=messages,
                    model=self.llm_model,
                    temperature=0.7
                )
                
                # Store the response
                response_memory = {
                    "type": "conversation",
                    "message": response,
                    "to": "user",
                    "timestamp": datetime.now().isoformat()
                }
                
                memory_item = MemoryItem(
                    id=f"chat_response_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    memory_type=MemoryType.EPISODIC,
                    attributes=MemoryAttributes(
                        salience=0.7,
                        emotional_valence=0.2,
                        emotional_intensity=0.4
                    ),
                    content=response_memory,
                    tags=["conversation", "self"]
                )
                
                self.episodic_memory.add_memory(memory_item)
                
                # Save as last utterance
                self.last_utterance = response
                
                return response
                
            except Exception as e:
                logger.error(f"Error getting LLM response: {str(e)}")
                # Fall back to internal generation
        
        # Default to internal utterance generation if LLM fails
        response = self.generate_utterance()
        return response
    
    def save_state(self, path: Optional[Path] = None) -> Path:
        """
        Save the current state of the neural child.
        
        Args:
            path: Optional path to save the state to, defaults to a timestamped directory
            
        Returns:
            Path: The path where the state was saved
        """
        # Create a default path if none provided
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(f"./saved_states/neural_child_{timestamp}")
        
        # Ensure directory exists
        path.mkdir(parents=True, exist_ok=True)
        
        # Save memory systems
        self.memory_manager.save_state(path / "memory_manager.json")
        
        # Save language components
        vocab_state_path = path / "vocabulary"
        vocab_state_path.mkdir(parents=True, exist_ok=True)
        self.vocabulary_manager.save_state(vocab_state_path / "vocabulary.json")
        
        # Save neural networks
        networks_state_path = path / "networks"
        networks_state_path.mkdir(parents=True, exist_ok=True)
        
        for network_name, network in self.networks.items():
            network_state = network.get_state()
            with open(networks_state_path / f"{network_name}.json", "w") as f:
                json.dump(network_state, f, indent=2, default=str)
        
        # Save overall state and metadata
        overall_state = {
            "age_days": self.age_days,
            "creation_time": self.creation_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "development_speed": self.development_speed,
            "milestones": self.milestones.model_dump(),
            "interaction_history": self.interaction_history[-20:],  # Save last 20 interactions
            "vocabulary_history": self.vocabulary_history[-100:],  # Save last 100 vocab records
            "last_utterance": self.last_utterance,
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(path / "neural_child_state.json", "w") as f:
            json.dump(overall_state, f, indent=2, default=str)
        
        self.last_saved_time = datetime.now()
        logger.info(f"Neural child state saved to {path}")
        
        return path
    
    def load_state(self, path: Path) -> bool:
        """
        Load a previously saved state.
        
        Args:
            path: Path to the saved state directory
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        # Check if the path exists
        if not path.exists() or not path.is_dir():
            logger.error(f"Invalid state path: {path}")
            return False
        
        try:
            # Load overall state
            state_path = path / "neural_child_state.json"
            if not state_path.exists():
                logger.error(f"State file not found: {state_path}")
                return False
            
            with open(state_path, "r") as f:
                overall_state = json.load(f)
            
            # Update basic properties
            self.age_days = overall_state.get("age_days", 0.0)
            self.creation_time = datetime.fromisoformat(overall_state.get("creation_time", datetime.now().isoformat()))
            self.last_update_time = datetime.fromisoformat(overall_state.get("last_update_time", datetime.now().isoformat()))
            self.development_speed = overall_state.get("development_speed", 10.0)
            
            # Update milestones
            if "milestones" in overall_state:
                self.milestones = DevelopmentMilestones(**overall_state["milestones"])
            
            # Update history
            self.interaction_history = overall_state.get("interaction_history", [])
            self.vocabulary_history = overall_state.get("vocabulary_history", [])
            self.last_utterance = overall_state.get("last_utterance", "")
            
            # Load memory systems
            memory_manager_path = path / "memory_manager.json"
            if memory_manager_path.exists():
                self.memory_manager.load_state(memory_manager_path)
            
            # Load vocabulary
            vocab_path = path / "vocabulary" / "vocabulary.json"
            if vocab_path.exists() and hasattr(self.vocabulary_manager, "load_state"):
                self.vocabulary_manager.load_state(vocab_path)
            
            # Update language development
            vocabulary_size = self.vocabulary_manager.get_vocabulary_statistics().total_words
            self.language_development.update(
                age_days=self.age_days,
                vocabulary_size=vocabulary_size
            )
            
            # Load neural networks
            networks_path = path / "networks"
            if networks_path.exists() and networks_path.is_dir():
                for network_name, network in self.networks.items():
                    network_path = networks_path / f"{network_name}.json"
                    if network_path.exists():
                        with open(network_path, "r") as f:
                            network_state = json.load(f)
                        network.load_state(network_state)
            
            self.last_saved_time = datetime.now()
            logger.info(f"Neural child state loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False