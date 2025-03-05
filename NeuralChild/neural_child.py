# neural_child.py
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import logging
import json
import os
import numpy as np
import time
from pydantic import BaseModel, Field, field_validator, model_validator

# Import configuration
from config import get_config, GlobalConfig

# Import mother component
from mother import Mother, ChildState, MotherResponse

# Import network components
from networks.base_network import BaseNetwork, NetworkState, ConnectionType
from networks.network_types import NetworkType
from networks.emotions import EmotionsNetwork
from networks.consciousness import ConsciousnessNetwork
from networks.unconsciousness import UnconsciousnessNetwork
from networks.archetypes import ArchetypesNetwork
from networks.instincts import InstinctsNetwork
from networks.drives import DrivesNetwork
from networks.attention import AttentionNetwork
from networks.perception import PerceptionNetwork
from networks.thoughts import ThoughtsNetwork
from networks.moods import MoodsNetwork

# Import memory components
from memory.memory_manager import MemoryManager, MemoryType, MemoryItem, MemoryAttributes, MemoryQuery
from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemory
from memory.associative_memory import AssociativeMemory
from memory.long_term_memory import LongTermMemory, KnowledgeDomain

# Import language components
from language.developmental_stages import (
    LanguageDevelopmentStage, LanguageFeature, 
    LanguageCapabilities, DevelopmentTracker
)
from language.lexical_memory import LexicalMemory, LexicalItem
from language.semantic_network import SemanticNetwork, SemanticRelation
from language.syntactic_processor import SyntacticProcessor, GrammaticalCategory
from language.vocabulary import VocabularyManager, VocabularyStatistics
from language.production import LanguageProduction

# Import LLM client
from llm_module import LLMClient, Message

logger = logging.getLogger("NeuralChild")

class ChildMetrics(BaseModel):
    """Metrics tracking the child's development"""
    age_days: float = Field(0.0, ge=0.0, description="Simulated age in days")
    language_stage: LanguageDevelopmentStage = Field(LanguageDevelopmentStage.PRE_LINGUISTIC)
    vocabulary_size: int = Field(0, ge=0, description="Number of known words")
    grammar_complexity: float = Field(0.0, ge=0.0, le=1.0)
    emotional_development: float = Field(0.0, ge=0.0, le=1.0)
    attention_span: float = Field(0.1, ge=0.0, le=1.0)
    self_awareness: float = Field(0.0, ge=0.0, le=1.0)
    memory_capacity: float = Field(0.0, ge=0.0, le=1.0)
    social_understanding: float = Field(0.0, ge=0.0, le=1.0)
    biological_needs: Dict[str, float] = Field(default_factory=dict)
    active_emotions: Dict[str, float] = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=datetime.now)
    interaction_count: int = Field(0, ge=0)
    
    def update(self, age_increment: float = 0.0) -> None:
        """Update metrics with new values"""
        # Age increases with time
        self.age_days += age_increment
        self.last_update = datetime.now()
        self.interaction_count += 1

class DevelopmentalState(BaseModel):
    """Overall developmental state of the neural child"""
    stage: str = Field("infancy", description="Overall developmental stage")
    age_days: float = Field(0.0, ge=0.0, description="Simulated age in days")
    has_language: bool = Field(False, description="Whether language has emerged")
    has_self_awareness: bool = Field(False, description="Whether self-awareness has emerged")
    can_recognize_mother: bool = Field(False, description="Can recognize the mother")
    can_form_memories: bool = Field(False, description="Can form lasting memories")
    capabilities: Dict[str, float] = Field(default_factory=dict, description="Capability levels")
    stage_transitions: Dict[str, float] = Field(default_factory=dict, description="Ages at which transitions occurred")
    
    def update_from_metrics(self, metrics: ChildMetrics, config: GlobalConfig) -> None:
        """Update developmental state based on metrics"""
        self.age_days = metrics.age_days
        
        # Determine stage transitions
        if self.stage == "infancy" and metrics.age_days >= config.development.stage_transition_thresholds["infancy_to_early_childhood"]:
            self.stage = "early_childhood"
            self.stage_transitions["infancy_to_early_childhood"] = metrics.age_days
            
        elif self.stage == "early_childhood" and metrics.age_days >= config.development.stage_transition_thresholds["early_to_middle_childhood"]:
            self.stage = "middle_childhood"
            self.stage_transitions["early_to_middle_childhood"] = metrics.age_days
            
        elif self.stage == "middle_childhood" and metrics.age_days >= config.development.stage_transition_thresholds["middle_childhood_to_adolescence"]:
            self.stage = "adolescence"
            self.stage_transitions["middle_childhood_to_adolescence"] = metrics.age_days
        
        # Update capability flags
        self.has_language = metrics.language_stage != LanguageDevelopmentStage.PRE_LINGUISTIC
        self.has_self_awareness = metrics.self_awareness >= 0.4
        self.can_recognize_mother = metrics.age_days >= 3.0
        self.can_form_memories = metrics.memory_capacity >= 0.2
        
        # Update capabilities dictionary
        self.capabilities = {
            "language": float(metrics.language_stage.value.split("_")[-1] == "advanced"),
            "emotional": metrics.emotional_development,
            "attention": metrics.attention_span,
            "self_awareness": metrics.self_awareness,
            "memory": metrics.memory_capacity,
            "social": metrics.social_understanding
        }

class NeuralChild:
    """
    The main class representing the neural child's mind.
    
    This class coordinates all the components of the simulated mind,
    including neural networks, memory systems, and language capabilities.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize the neural child's mind"""
        # Load configuration
        self.config = get_config()
        if config_path:
            self.config = self.config.load_from_file(config_path)
        
        # Set up data directory
        self.data_dir = data_dir or self.config.system.data_dir / "neural_child"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(base_url=self.config.system.llm_base_url)
        
        # Initialize development metrics
        self.metrics = ChildMetrics()
        self.developmental_state = DevelopmentalState()
        
        # Initialize memory systems
        logger.info("Initializing memory systems...")
        self.memory_manager = MemoryManager(data_dir=self.data_dir / "memory")
        self.working_memory = WorkingMemory(capacity=self.config.memory.working_memory_capacity)
        self.episodic_memory = EpisodicMemory(data_dir=self.data_dir / "episodic")
        self.associative_memory = AssociativeMemory(data_dir=self.data_dir / "associative")
        self.long_term_memory = LongTermMemory(data_dir=self.data_dir / "long_term")
        
        # Connect memory systems to memory manager
        self.working_memory.set_memory_manager(self.memory_manager)
        self.episodic_memory.set_memory_manager(self.memory_manager)
        self.associative_memory.set_memory_manager(self.memory_manager)
        self.long_term_memory.set_memory_manager(self.memory_manager)
        
        # Initialize language components
        logger.info("Initializing language components...")
        self.lexical_memory = LexicalMemory()
        self.syntactic_processor = SyntacticProcessor()
        self.semantic_network = SemanticNetwork()
        self.vocabulary_manager = VocabularyManager(data_dir=self.data_dir / "vocabulary")
        self.language_tracker = DevelopmentTracker()
        self.language_production = LanguageProduction(
            lexical_memory=self.lexical_memory,
            syntactic_processor=self.syntactic_processor
        )
        
        # Initialize neural networks
        logger.info("Initializing neural networks...")
        self.networks: Dict[NetworkType, BaseNetwork] = {
            NetworkType.ARCHETYPES: ArchetypesNetwork(),
            NetworkType.INSTINCTS: InstinctsNetwork(),
            NetworkType.UNCONSCIOUSNESS: UnconsciousnessNetwork(),
            NetworkType.DRIVES: DrivesNetwork(),
            NetworkType.EMOTIONS: EmotionsNetwork(),
            NetworkType.MOODS: MoodsNetwork(),
            NetworkType.ATTENTION: AttentionNetwork(),
            NetworkType.PERCEPTION: PerceptionNetwork(),
            NetworkType.CONSCIOUSNESS: ConsciousnessNetwork(),
            NetworkType.THOUGHTS: ThoughtsNetwork()
        }
        
        # Establish network connections based on psychological theory
        self._establish_network_connections()
        
        # The Mother component
        self.mother = Mother(llm_client=self.llm_client)
        
        # Internal state
        self.last_mother_response: Optional[MotherResponse] = None
        self.current_response: str = ""
        self.response_ready = False
        self.active_networks: List[NetworkType] = []
        self.activation_history: Dict[NetworkType, List[Tuple[datetime, float]]] = {
            network_type: [] for network_type in self.networks.keys()
        }
        
        # Memory context for interactions
        self.interaction_context = {
            "recent_exchanges": [],
            "referenced_objects": [],
            "conversation_topic": None
        }
        
        # Working variables
        self.simulation_speed: float = 1.0  # How fast time progresses
        self.paused: bool = False
        self.processing: bool = False
        
        logger.info("Neural child initialized")
    
    def _establish_network_connections(self) -> None:
        """Establish connections between neural networks"""
        # Establish bottom-up connections
        # Perception -> Attention
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.ATTENTION.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8
        )
        
        # Perception -> Emotions
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        # Instincts -> Drives
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.DRIVES.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8
        )
        
        # Instincts -> Emotions
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        # Drives -> Emotions
        self.networks[NetworkType.DRIVES].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Attention -> Consciousness
        self.networks[NetworkType.ATTENTION].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        # Emotions -> Consciousness
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        # Emotions -> Moods
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.MOODS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Consciousness -> Thoughts
        self.networks[NetworkType.CONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        # Establish top-down connections
        # Consciousness -> Attention
        self.networks[NetworkType.CONSCIOUSNESS].connect_to(
            target_network=NetworkType.ATTENTION.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        
        # Thoughts -> Consciousness
        self.networks[NetworkType.THOUGHTS].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.FEEDBACK,
            initial_strength=0.5
        )
        
        # Archetypes -> Unconsciousness
        self.networks[NetworkType.ARCHETYPES].connect_to(
            target_network=NetworkType.UNCONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        
        # Unconsciousness -> Emotions
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.5
        )
        
        # Unconsciousness -> Thoughts
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.4
        )
        
        # Moods -> Emotions
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        
        # Moods -> Perception
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.PERCEPTION.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
        
        logger.info("Network connections established")
    
    def process_mother_response(self, mother_response: MotherResponse) -> None:
        """Process a response from the mother"""
        logger.info("Processing mother's response")
        self.processing = True
        self.last_mother_response = mother_response
        self.response_ready = False
        
        # Update interaction metrics
        self.metrics.interaction_count += 1
        
        # Extract verbal content
        verbal_text = mother_response.verbal.text
        
        # Process through perception
        percepts = [verbal_text]
        percepts.extend(mother_response.non_verbal.physical_actions)
        
        # Add emotional percepts
        if mother_response.emotional.primary_emotion:
            percepts.append(f"mother_emotion:{mother_response.emotional.primary_emotion}")
        
        # Create perception data
        perception_data = {
            "verbal": verbal_text,
            "percepts": percepts,
            "source": "mother"
        }
        
        # Feed to perception network
        self.networks[NetworkType.PERCEPTION].receive_input(perception_data)
        
        # Update working memory with mother's response
        memory_item = MemoryItem(
            id=f"mother_response_{self.metrics.interaction_count}",
            memory_type=MemoryType.WORKING,
            attributes=MemoryAttributes(
                emotional_valence=0.0,  # Will be updated based on emotional processing
                emotional_intensity=0.0,  # Will be updated based on emotional processing
                salience=0.8  # Mother's responses are important
            ),
            content=mother_response.model_dump(),
            tags=["mother", "interaction"]
        )
        self.memory_manager.store(memory_item)
        
        # Process through language systems
        self._process_language_input(mother_response)
        
        # Process through emotional systems
        self._process_emotional_input(mother_response)
        
        # Run network activation cycle
        self._run_network_activation_cycle()
        
        # Generate response
        self._generate_response()
        
        # Update development metrics
        self._update_development_metrics()
        
        # Memory maintenance
        self._perform_memory_maintenance()
        
        self.processing = False
        self.response_ready = True
        logger.info("Finished processing mother's response")
    
    def _process_language_input(self, mother_response: MotherResponse) -> None:
        """Process language input from the mother"""
        # Extract verbal content
        verbal_text = mother_response.verbal.text
        
        # Process through vocabulary manager
        emotional_state = {}
        if mother_response.emotional.primary_emotion:
            emotional_state[mother_response.emotional.primary_emotion] = mother_response.emotional.intensity
        
        # Learn words from mother's speech
        new_words = self.vocabulary_manager.process_heard_speech(
            text=verbal_text,
            context="mother_speech",
            emotional_state=emotional_state
        )
        
        # Process explicit teaching
        for vocab_item in mother_response.teaching.vocabulary:
            self.vocabulary_manager.explicitly_learn_word(
                word=vocab_item.word,
                definition=vocab_item.simple_definition,
                example_usage=vocab_item.example_usage,
                emotional_state=emotional_state
            )
        
        # Process through syntactic processor
        syntax_analysis = self.syntactic_processor.analyze_text(verbal_text)
        
        # Update language capabilities based on input
        vocab_stats = self.vocabulary_manager.get_vocabulary_statistics()
        self.language_tracker.update(self.metrics.age_days, vocab_stats.total_words)
        
        logger.info(f"Processed language input, learned {len(new_words)} new words")
    
    def _process_emotional_input(self, mother_response: MotherResponse) -> None:
        """Process emotional content from the mother"""
        # Extract mother's emotional state
        mother_emotion = mother_response.emotional.primary_emotion
        intensity = mother_response.emotional.intensity
        
        # Create emotional data
        emotional_data = {
            "mother_emotion": mother_emotion,
            "intensity": intensity,
            "mother_actions": mother_response.non_verbal.physical_actions
        }
        
        # Feed to emotions network
        self.networks[NetworkType.EMOTIONS].receive_input({
            "interpersonal": emotional_data
        })
        
        # Feed to drives network
        drive_satisfaction = {}
        
        # Process mother's actions for drive satisfaction
        actions = " ".join(mother_response.non_verbal.physical_actions)
        if any(word in actions.lower() for word in ["feed", "food", "milk", "eat"]):
            drive_satisfaction["physiological"] = 0.5
        
        if any(word in actions.lower() for word in ["hug", "hold", "cuddle", "comfort"]):
            drive_satisfaction["attachment"] = 0.6
        
        if any(word in actions.lower() for word in ["play", "show", "explore"]):
            drive_satisfaction["exploration"] = 0.4
        
        if len(drive_satisfaction) > 0:
            self.networks[NetworkType.DRIVES].receive_input({
                "drive_satisfaction": drive_satisfaction
            })
        
        logger.info(f"Processed emotional input from mother: {mother_emotion}")
    
    def _run_network_activation_cycle(self) -> None:
        """Run activation cycle across all networks"""
        # First, update all networks with time-based changes
        for network_type, network in self.networks.items():
            network.update()
        
        # Record active networks
        self.active_networks = []
        for network_type, network in self.networks.items():
            if network.state.activation >= network.activation_threshold:
                self.active_networks.append(network_type)
            
            # Record activation history
            self.activation_history[network_type].append(
                (datetime.now(), network.state.activation)
            )
            
            # Trim history to last 1000 points
            if len(self.activation_history[network_type]) > 1000:
                self.activation_history[network_type] = self.activation_history[network_type][-1000:]
        
        # Now propagate signals between networks
        # We're manually implementing the signal propagation here to make sure
        # we have control over the process and can monitor it
        for source_type, source_network in self.networks.items():
            if source_type in self.active_networks:
                # Get outgoing signals
                outgoing = source_network._propagate_signals()
                
                # Deliver to targets
                for target_name, signal in outgoing.items():
                    for target_type, target_network in self.networks.items():
                        if target_network.name == target_name:
                            target_network.receive_input(
                                signal["data"],
                                source_network=source_network.name
                            )
        
        # Run a second update to process the propagated signals
        for network_type, network in self.networks.items():
            network.update()
        
        logger.info(f"Completed network activation cycle. Active networks: {[n.value for n in self.active_networks]}")
    
    def _generate_response(self) -> None:
        """Generate a response based on internal state"""
        # Get language capabilities
        language_capabilities = self.language_tracker.get_capabilities()
        
        # Get emotional state
        emotional_state = {}
        emotions_network = self.networks[NetworkType.EMOTIONS]
        
        if NetworkType.EMOTIONS in self.active_networks:
            emotion_data = emotions_network._prepare_output_data()
            if "emotional_state" in emotion_data:
                emotional_state = emotion_data["emotional_state"]
        
        # Generate utterance
        if language_capabilities.stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            # In pre-linguistic stage, just generate babbling or crying
            if "distress" in emotional_state and emotional_state["distress"] > 0.5:
                self.current_response = "Waaah!"
            else:
                self.current_response = self.language_production.generate_babble()
        else:
            # Generate linguistic response based on capabilities
            self.current_response = self.language_production.generate_utterance(
                capabilities=language_capabilities,
                emotional_state=emotional_state,
                context="responding_to_mother",
                response_to=self.last_mother_response.verbal.text if self.last_mother_response else None
            )
        
        logger.info(f"Generated response: {self.current_response}")
    
    def _update_development_metrics(self) -> None:
        """Update development metrics based on current state"""
        # Update age - simulated time passes with each interaction
        age_increment = 0.2 * self.simulation_speed  # 0.2 days per interaction, scaled by simulation speed
        self.metrics.update(age_increment)
        
        # Update language metrics
        vocab_stats = self.vocabulary_manager.get_vocabulary_statistics()
        self.metrics.vocabulary_size = vocab_stats.total_words
        
        lang_capabilities = self.language_tracker.get_capabilities()
        self.metrics.language_stage = lang_capabilities.stage
        self.metrics.grammar_complexity = lang_capabilities.grammar_complexity
        
        # Update attention span
        attention_network = self.networks[NetworkType.ATTENTION]
        if hasattr(attention_network, "get_attention_span"):
            self.metrics.attention_span = attention_network.get_attention_span()
        else:
            self.metrics.attention_span = attention_network.state.activation
        
        # Update emotional development
        emotions_network = self.networks[NetworkType.EMOTIONS]
        if hasattr(emotions_network, "emotional_regulation"):
            self.metrics.emotional_development = emotions_network.emotional_regulation
        else:
            self.metrics.emotional_development = emotions_network.state.training_progress
        
        # Update self-awareness
        consciousness_network = self.networks[NetworkType.CONSCIOUSNESS]
        if hasattr(consciousness_network, "self_awareness"):
            self.metrics.self_awareness = consciousness_network.self_awareness
        else:
            self.metrics.self_awareness = consciousness_network.state.training_progress * 0.5
        
        # Update memory capacity
        working_memory_capacity = len(self.working_memory.items) / self.working_memory.capacity
        episodic_stats = self.episodic_memory.get_stats()
        ltm_stats = self.long_term_memory.get_memory_stats()
        
        self.metrics.memory_capacity = (
            working_memory_capacity * 0.2 + 
            (episodic_stats["total_episodes"] / 100) * 0.3 + 
            (ltm_stats["total_items"] / 200) * 0.5
        ) / 1.0  # Normalize to 0-1
        
        # Get current biological needs (drives)
        drives_network = self.networks[NetworkType.DRIVES]
        if hasattr(drives_network, "get_drive_status"):
            drive_status = drives_network.get_drive_status()
            self.metrics.biological_needs = {
                name: data["level"] for name, data in drive_status.items()
            }
        
        # Get current emotions
        if hasattr(emotions_network, "_prepare_output_data"):
            emotion_data = emotions_network._prepare_output_data()
            if "emotional_state" in emotion_data:
                self.metrics.active_emotions = emotion_data["emotional_state"]
        
        # Calculate social understanding based on relationship knowledge
        if hasattr(self.long_term_memory.identity, "relationships"):
            relationship_count = len(self.long_term_memory.identity.relationships)
            self.metrics.social_understanding = min(1.0, relationship_count / 10)
        else:
            self.metrics.social_understanding = self.metrics.self_awareness * 0.7
        
        # Update developmental state
        self.developmental_state.update_from_metrics(self.metrics, self.config)
        
        # Update networks based on developmental stage
        self._update_networks_for_development()
        
        logger.info(f"Updated developmental metrics. Age: {self.metrics.age_days:.1f} days")
    
    def _update_networks_for_development(self) -> None:
        """Update network parameters based on development"""
        # Update instincts network
        if hasattr(self.networks[NetworkType.INSTINCTS], "update_development"):
            self.networks[NetworkType.INSTINCTS].update_development(self.metrics.age_days)
        
        # Update emotions network
        if hasattr(self.networks[NetworkType.EMOTIONS], "update_development"):
            self.networks[NetworkType.EMOTIONS].update_development(self.metrics.age_days)
        
        # Update consciousness network
        if hasattr(self.networks[NetworkType.CONSCIOUSNESS], "update_development"):
            self.networks[NetworkType.CONSCIOUSNESS].update_development(
                self.metrics.age_days, 
                self.metrics.vocabulary_size
            )
        
        # Update attention network
        if hasattr(self.networks[NetworkType.ATTENTION], "update_development"):
            self.networks[NetworkType.ATTENTION].update_development(
                self.metrics.age_days,
                self.metrics.interaction_count
            )
        
        # Update unconsciousness network
        if hasattr(self.networks[NetworkType.UNCONSCIOUSNESS], "update_development"):
            self.networks[NetworkType.UNCONSCIOUSNESS].update_development(self.metrics.age_days)
        
        # Update archetypes network
        if hasattr(self.networks[NetworkType.ARCHETYPES], "update_development"):
            emotion_state = self.metrics.active_emotions
            self.networks[NetworkType.ARCHETYPES].update_development(self.metrics.age_days, emotion_state)
        
        # Update thoughts network
        if hasattr(self.networks[NetworkType.THOUGHTS], "update_development"):
            self.networks[NetworkType.THOUGHTS].update_development(
                self.metrics.age_days,
                self.metrics.vocabulary_size
            )
        
        # Update moods network
        if hasattr(self.networks[NetworkType.MOODS], "update_development"):
            self.networks[NetworkType.MOODS].update_development(self.metrics.age_days)
        
        # Update perception network
        if hasattr(self.networks[NetworkType.PERCEPTION], "update_development"):
            self.networks[NetworkType.PERCEPTION].update_development(
                self.metrics.age_days,
                self.metrics.vocabulary_size
            )
        
        # Update drives based on developmental stage
        if hasattr(self.networks[NetworkType.DRIVES], "update_developmental_stage"):
            self.networks[NetworkType.DRIVES].update_developmental_stage(self.developmental_state.stage)
        
        # Update language components
        self.vocabulary_manager.update_development(self.metrics.age_days)
        self.syntactic_processor.update_rule_masteries(
            self.language_tracker.get_capabilities().stage,
            self.metrics.grammar_complexity
        )
    
    def _perform_memory_maintenance(self) -> None:
        """Perform maintenance on memory systems"""
        # Update working memory
        self.working_memory.update()
        
        # Consolidate working memory to long-term memory
        if self.metrics.interaction_count % 5 == 0:  # Every 5 interactions
            self.memory_manager.consolidate()
        
        # Consolidate episodic memories
        if self.metrics.interaction_count % 10 == 0:  # Every 10 interactions
            self.episodic_memory.consolidate_recent_episodes()
        
        # Apply memory decay
        if self.metrics.interaction_count % 20 == 0:  # Every 20 interactions
            decay_rate = self.config.memory.long_term_decay_rate * self.simulation_speed
            self.memory_manager.apply_memory_decay(decay_rate)
        
        # Apply decay to associative memory
        if self.metrics.interaction_count % 15 == 0:  # Every 15 interactions
            decay_rate = 0.01 * self.simulation_speed
            self.associative_memory.apply_decay(decay_rate)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the neural child"""
        # Basic state information
        state = {
            "metrics": self.metrics.model_dump(),
            "developmental_state": self.developmental_state.model_dump(),
            "active_networks": [network.value for network in self.active_networks],
            "processing": self.processing,
            "response_ready": self.response_ready,
            "current_response": self.current_response
        }
        
        # Get network states
        network_states = {}
        for network_type, network in self.networks.items():
            network_states[network_type.value] = network.get_state()
        
        state["network_states"] = network_states
        
        # Get memory statistics
        memory_stats = {
            "working_memory": self.working_memory.get_state(),
            "episodic_memory": self.episodic_memory.get_stats(),
            "associative_memory": self.associative_memory.get_stats(),
            "long_term_memory": self.long_term_memory.get_memory_stats()
        }
        
        state["memory_stats"] = memory_stats
        
        # Get language statistics
        language_stats = {
            "vocabulary": self.vocabulary_manager.get_vocabulary_statistics().model_dump(),
            "language_capabilities": self.language_tracker.get_capabilities().model_dump(),
            "syntactic_processor": self.syntactic_processor.get_grammar_statistics()
        }
        
        state["language_stats"] = language_stats
        
        return state
    
    def get_child_state_for_mother(self) -> ChildState:
        """Get the child's state formatted for the mother"""
        # Get the apparent emotion
        apparent_emotion = "neutral"
        if self.metrics.active_emotions:
            strongest_emotion = max(
                self.metrics.active_emotions.items(),
                key=lambda x: x[1],
                default=("neutral", 0.0)
            )
            apparent_emotion = strongest_emotion[0]
        
        # Get recent concepts learned
        recent_words = self.vocabulary_manager.get_vocabulary_statistics().recent_words
        
        return ChildState(
            message=self.current_response,
            apparent_emotion=apparent_emotion,
            vocabulary_size=self.metrics.vocabulary_size,
            age_days=self.metrics.age_days,
            recent_concepts_learned=recent_words[:5],
            attention_span=self.metrics.attention_span
        )
    
    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """Save the current state to disk"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.data_dir / f"neural_child_state_{timestamp}.json"
        
        # Create state dictionary
        state = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.model_dump(),
            "developmental_state": self.developmental_state.model_dump(),
            "simulation_speed": self.simulation_speed
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save individual components
        components_dir = filepath.parent / filepath.stem
        components_dir.mkdir(exist_ok=True)
        
        # Save memory systems
        memory_dir = components_dir / "memory"
        memory_dir.mkdir(exist_ok=True)
        
        self.memory_manager.save_state(memory_dir / "memory_manager.json")
        self.vocabulary_manager.save_state(memory_dir / "vocabulary.json")
        
        # Save neural networks individually
        network_dir = components_dir / "networks"
        network_dir.mkdir(exist_ok=True)
        
        network_states = {}
        for network_type, network in self.networks.items():
            network_state = network.get_state()
            network_states[network_type.value] = network_state
        
        with open(network_dir / "network_states.json", 'w') as f:
            json.dump(network_states, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
        return filepath
    
    def load_state(self, filepath: Path) -> bool:
        """Load state from disk"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load core state
            self.metrics = ChildMetrics.model_validate(state["metrics"])
            self.developmental_state = DevelopmentalState.model_validate(state["developmental_state"])
            self.simulation_speed = state.get("simulation_speed", 1.0)
            
            # Load component states if available
            components_dir = filepath.parent / filepath.stem
            if os.path.exists(components_dir):
                # Load memory systems
                memory_dir = components_dir / "memory"
                if os.path.exists(memory_dir / "memory_manager.json"):
                    self.memory_manager.load_state(memory_dir / "memory_manager.json")
                
                if os.path.exists(memory_dir / "vocabulary.json"):
                    # Find the vocabulary file
                    vocab_files = list(memory_dir.glob("vocabulary*.json"))
                    if vocab_files:
                        vocab_file = sorted(vocab_files)[-1]  # Get the latest one
                        self.vocabulary_manager.load_state(vocab_file, None)  # No semantic network file needed
                
                # Load neural network states
                network_dir = components_dir / "networks"
                if os.path.exists(network_dir / "network_states.json"):
                    with open(network_dir / "network_states.json", 'r') as f:
                        network_states = json.load(f)
                    
                    for network_type_str, network_state in network_states.items():
                        for network_type, network in self.networks.items():
                            if network_type.value == network_type_str:
                                network.load_state(network_state)
            
            logger.info(f"State loaded from {filepath}")
            
            # Update networks for current developmental stage
            self._update_networks_for_development()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def set_simulation_speed(self, speed: float) -> None:
        """Set the simulation speed multiplier"""
        self.simulation_speed = max(0.1, min(10.0, speed))
        logger.info(f"Simulation speed set to {self.simulation_speed}x")