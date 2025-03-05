# neural_child.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import logging
import os
import json
from pathlib import Path
import time
from pydantic import BaseModel, Field, field_validator, model_validator

# Import system components
from config import get_config
from llm_module import LLMClient, Message

# Import networks
from networks.base_network import BaseNetwork, NetworkState
from networks.network_types import NetworkType, ConnectionType
from networks.emotions import EmotionsNetwork
from networks.consciousness import ConsciousnessNetwork
from networks.unconsciousness import UnconsciousnessNetwork
from networks.archetypes import ArchetypesNetwork
from networks.instincts import InstinctsNetwork
from networks.perception import PerceptionNetwork
from networks.attention import AttentionNetwork
from networks.thoughts import ThoughtsNetwork
from networks.moods import MoodsNetwork
from networks.drives import DrivesNetwork

# Import language components
from language.developmental_stages import (
    LanguageDevelopmentStage, 
    LanguageCapabilities, 
    DevelopmentTracker, 
    LanguageFeature
)
from language.vocabulary import VocabularyManager
from language.syntactic_processor import SyntacticProcessor
from language.production import LanguageProduction

# Import memory components
from memory.memory_manager import MemoryManager, MemoryItem, MemoryType
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory, KnowledgeDomain
from memory.episodic_memory import EpisodicMemory
from memory.associative_memory import AssociativeMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralChild")

class DevelopmentMetrics(BaseModel):
    """Metrics tracking the child's development"""
    age_days: float = Field(0.0, ge=0.0, description="Simulated age in days")
    vocabulary_size: int = Field(0, ge=0, description="Number of words in vocabulary")
    grammar_complexity: float = Field(0.0, ge=0.0, le=1.0, description="Complexity of grammar")
    emotional_development: float = Field(0.0, ge=0.0, le=1.0, description="Emotional development level")
    cognitive_development: float = Field(0.0, ge=0.0, le=1.0, description="Cognitive development level")
    social_development: float = Field(0.0, ge=0.0, le=1.0, description="Social development level")
    attention_span: float = Field(0.2, ge=0.0, le=1.0, description="Current attention span")
    memory_capacity: float = Field(0.1, ge=0.0, le=1.0, description="Memory capacity")
    interactions_count: int = Field(0, ge=0, description="Total number of interactions")
    developmental_stage: str = Field("infancy", description="Current developmental stage")
    feature_levels: Dict[str, float] = Field(
        default_factory=lambda: {
            "phonetics": 0.1,
            "vocabulary": 0.1, 
            "grammar": 0.0,
            "pragmatics": 0.1,
            "comprehension": 0.1,
            "expression": 0.05
        },
        description="Levels of different language features"
    )
    
    @property
    def overall_development(self) -> float:
        """Calculate the overall development level"""
        weighted_sum = (
            self.emotional_development * 0.25 +
            self.cognitive_development * 0.25 +
            self.social_development * 0.2 +
            self.grammar_complexity * 0.15 +
            self.memory_capacity * 0.15
        )
        return weighted_sum

class MindState(BaseModel):
    """Current state of the neural child's mind"""
    dominant_emotion: Optional[str] = Field(None, description="Dominant emotion currently felt")
    emotional_state: Dict[str, float] = Field(default_factory=dict, description="Current emotional state")
    attention_focus: List[str] = Field(default_factory=list, description="What the child is focusing on")
    active_thoughts: List[str] = Field(default_factory=list, description="Current active thoughts")
    active_networks: Dict[str, float] = Field(default_factory=dict, description="Currently active networks")
    dominant_drive: Optional[str] = Field(None, description="Dominant drive or motivation")
    dominant_mood: Optional[str] = Field(None, description="Dominant mood state")
    self_awareness_level: float = Field(0.1, ge=0.0, le=1.0, description="Level of self-awareness")
    internal_state: Dict[str, Any] = Field(default_factory=dict, description="Additional internal state data")

class NeuralChild:
    """The main neural child system that integrates all components"""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        simulation_speed: float = 24.0,  # 1 day = 1 hour by default
        random_seed: Optional[int] = None
    ):
        """Initialize the neural child system
        
        Args:
            data_dir: Directory for saving/loading data
            config_path: Path to configuration file
            llm_client: LLM client for natural language processing
            simulation_speed: How fast the simulation runs (days per real hour)
            random_seed: Seed for random number generation
        """
        # Set up basic parameters
        self.data_dir = data_dir or Path("./data/neural_child")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_config()
        self.simulation_speed = simulation_speed
        self.last_update_time = datetime.now()
        self.last_save_time = self.last_update_time
        
        # Set random seed if provided
        if random_seed is not None:
            import random
            import numpy as np
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Set up LLM client
        self.llm_client = llm_client or LLMClient(base_url=self.config.system.llm_base_url)
        
        # Initialize developmental metrics
        self.metrics = DevelopmentMetrics()
        
        # Initialize memory systems
        self.memory_manager = MemoryManager(data_dir=self.data_dir / "memory")
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory(data_dir=self.data_dir / "long_term")
        self.episodic_memory = EpisodicMemory(data_dir=self.data_dir / "episodic")
        self.associative_memory = AssociativeMemory(data_dir=self.data_dir / "associative")
        
        # Connect memory systems
        self.working_memory.set_memory_manager(self.memory_manager)
        self.long_term_memory.set_memory_manager(self.memory_manager)
        self.episodic_memory.set_memory_manager(self.memory_manager)
        self.associative_memory.set_memory_manager(self.memory_manager)
        
        # Initialize neural networks
        self.networks: Dict[NetworkType, BaseNetwork] = {}
        self._initialize_networks()
        
        # Initialize language components
        self.development_tracker = DevelopmentTracker(initial_age_days=self.metrics.age_days)
        self.syntactic_processor = SyntacticProcessor()
        self.vocabulary_manager = VocabularyManager(
            data_dir=self.data_dir / "vocabulary",
            simulation_speed=simulation_speed
        )
        self.language_production = LanguageProduction(
            lexical_memory=self.vocabulary_manager.lexical_memory,
            syntactic_processor=self.syntactic_processor
        )
        
        # Initialize mind state
        self.mind_state = MindState()
        
        # Track interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
        logger.info(f"Neural child initialized with simulation speed {simulation_speed}x")
    
    def _initialize_networks(self) -> None:
        """Initialize all neural networks"""
        # Create networks
        self.networks[NetworkType.EMOTIONS] = EmotionsNetwork()
        self.networks[NetworkType.CONSCIOUSNESS] = ConsciousnessNetwork()
        self.networks[NetworkType.UNCONSCIOUSNESS] = UnconsciousnessNetwork()
        self.networks[NetworkType.ARCHETYPES] = ArchetypesNetwork()
        self.networks[NetworkType.INSTINCTS] = InstinctsNetwork()
        self.networks[NetworkType.PERCEPTION] = PerceptionNetwork()
        self.networks[NetworkType.ATTENTION] = AttentionNetwork()
        self.networks[NetworkType.THOUGHTS] = ThoughtsNetwork()
        self.networks[NetworkType.MOODS] = MoodsNetwork()
        self.networks[NetworkType.DRIVES] = DrivesNetwork()
        
        # Initialize connections between networks based on psychological model
        self._establish_network_connections()
        
        logger.info(f"Initialized {len(self.networks)} neural networks")
    
    def _establish_network_connections(self) -> None:
        """Establish connections between networks"""
        # Perception connections
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.ATTENTION.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8
        )
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Attention connections
        self.networks[NetworkType.ATTENTION].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.7
        )
        self.networks[NetworkType.ATTENTION].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Emotions connections
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.MOODS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5
        )
        
        # Drives connections
        self.networks[NetworkType.DRIVES].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        self.networks[NetworkType.DRIVES].connect_to(
            target_network=NetworkType.ATTENTION.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5
        )
        
        # Instincts connections
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.DRIVES.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        
        # Archetypes connections
        self.networks[NetworkType.ARCHETYPES].connect_to(
            target_network=NetworkType.UNCONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.7
        )
        self.networks[NetworkType.ARCHETYPES].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5
        )
        
        # Unconsciousness connections
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.3
        )
        
        # Consciousness connections
        self.networks[NetworkType.CONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8
        )
        self.networks[NetworkType.CONSCIOUSNESS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.FEEDBACK,
            initial_strength=0.5,
            bidirectional=True
        )
        
        # Thoughts connections
        self.networks[NetworkType.THOUGHTS].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.FEEDBACK,
            initial_strength=0.7
        )
        
        # Moods connections
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5
        )
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
    
    def process_mother_response(self, mother_response: Any) -> Dict[str, Any]:
        """Process a response from the mother
        
        Args:
            mother_response: The mother's response object
            
        Returns:
            Dictionary with processing results and child's response
        """
        # Extract verbal component
        verbal_text = mother_response.verbal.text
        
        # Extract non-verbal components
        facial_expression = mother_response.non_verbal.facial_expression
        physical_actions = mother_response.non_verbal.physical_actions
        proximity = mother_response.non_verbal.proximity
        
        # Extract emotional state
        mother_emotion = mother_response.emotional.primary_emotion
        emotion_intensity = mother_response.emotional.intensity
        
        # Extract teaching elements if available
        new_vocabulary = mother_response.teaching.vocabulary
        new_concepts = mother_response.teaching.concepts
        
        # Process through perception network
        perception_input = {
            "data": {  # Wrap the input data in a "data" field
                "verbal": verbal_text,
                "visual": {
                    "facial_expression": facial_expression,
                    "physical_actions": physical_actions,
                    "proximity": proximity
                },
                "emotional_state": {
                    mother_emotion: emotion_intensity
                }
            },
            "source": "mother"  # Include source in the input data structure
        }
        
        # Pass input to perception network
        self.networks[NetworkType.PERCEPTION].receive_input(perception_input)
        
        # Process language input
        self._process_language_input(verbal_text)
        
        # Process teaching elements
        if new_vocabulary:
            self._process_vocabulary_teaching(new_vocabulary)
        
        if new_concepts:
            self._process_concept_teaching(new_concepts)
        
        # Update networks and propagate signals
        self._update_networks()
        
        # Generate response based on current state
        response = self._generate_response()
        
        # Update metrics after interaction
        self._update_metrics_after_interaction()
        
        # Record the interaction
        self._record_interaction(mother_response, response)
        
        # Return the processing results
        return {
            "response": response,
            "emotional_state": self.mind_state.emotional_state,
            "development_metrics": self.metrics.model_dump()
        }
    
    def _process_language_input(self, text: str) -> None:
        """Process language input through vocabulary and syntactic systems"""
        # Process text with vocabulary manager
        emotional_state = self.mind_state.emotional_state
        new_words = self.vocabulary_manager.process_heard_speech(text, context="mother_speech", emotional_state=emotional_state)
        
        # Process with syntactic processor to learn grammar patterns
        analysis = self.syntactic_processor.analyze_text(text)
        
        # Update grammar complexity based on input analysis
        if analysis["complexity"] > 0:
            # Gradually increase grammar complexity based on exposure
            current = self.metrics.grammar_complexity
            increment = min(0.01, analysis["complexity"] * 0.05)
            self.metrics.grammar_complexity = min(1.0, current + increment)
        
        # Update language development tracker with new vocabulary size
        vocab_size = len(self.vocabulary_manager.lexical_memory.words)
        self.metrics.vocabulary_size = vocab_size
        self.development_tracker.update(self.metrics.age_days, vocab_size)
        
        # Store interesting grammatical patterns in long-term memory
        if analysis["complexity"] > 0.3 and self.metrics.age_days > 10:
            pattern_memory = {
                "pattern_type": "grammatical",
                "example": text,
                "complexity": analysis["complexity"],
                "features": analysis["grammatical_features"]
            }
            
            self.memory_manager.store(
                content=pattern_memory,
                memory_type=MemoryType.LONG_TERM,
                tags=["language", "grammar", "pattern"],
                emotional_valence=0.1,
                emotional_intensity=0.3,
                salience=0.6
            )
    
    def _process_vocabulary_teaching(self, vocabulary_items: List[Any]) -> None:
        """Process vocabulary teaching from mother"""
        for item in vocabulary_items:
            # Explicitly learn the word with its definition
            self.vocabulary_manager.explicitly_learn_word(
                word=item.word,
                definition=item.simple_definition,
                example_usage=item.example_usage,
                emotional_state=self.mind_state.emotional_state
            )
            
            # Create an association in semantic network
            self.vocabulary_manager.semantic_network.add_concept(
                word=item.word,
                category=None,
                emotions=self.mind_state.emotional_state
            )
            
            # Store in long-term memory
            word_memory = {
                "word": item.word,
                "definition": item.simple_definition,
                "example": item.example_usage,
                "source": "mother_teaching"
            }
            
            self.memory_manager.store(
                content=word_memory,
                memory_type=MemoryType.LONG_TERM,
                tags=["vocabulary", "taught", item.word],
                emotional_valence=0.3,  # Learning is slightly positive
                emotional_intensity=0.4,
                salience=0.7  # Explicitly taught words are salient
            )
    
    def _process_concept_teaching(self, concept_items: List[Any]) -> None:
        """Process concept teaching from mother"""
        for item in concept_items:
            # Create a concept memory
            concept_memory = {
                "concept": item.concept_name,
                "explanation": item.explanation,
                "relevance": item.relevance,
                "source": "mother_teaching"
            }
            
            # Store in long-term memory with domain classification
            memory_id = self.memory_manager.store(
                content=concept_memory,
                memory_type=MemoryType.LONG_TERM,
                tags=["concept", item.concept_name],
                emotional_valence=0.2,
                emotional_intensity=0.3,
                salience=0.7
            )
            
            # Determine knowledge domain based on concept content
            domain = self._classify_concept_domain(item.concept_name, item.explanation)
            
            # Store in long-term memory with domain classification
            self.long_term_memory.store(
                memory_item=self.memory_manager.retrieve(memory_id),
                domain=domain,
                importance=0.6
            )
            
            # Create semantic network entries for the concept
            self.associative_memory.create_association(
                source="mother",
                target=item.concept_name,
                strength=0.7,
                link_type="teaching"
            )
    
    def _classify_concept_domain(self, concept: str, explanation: str) -> Optional[KnowledgeDomain]:
        """Classify a concept into a knowledge domain"""
        # Simple keyword-based classification for now
        personal_keywords = ["me", "my", "i", "mine", "self", "identity"]
        social_keywords = ["people", "friends", "family", "mother", "father", "social"]
        procedural_keywords = ["how to", "steps", "procedure", "do this", "process"]
        emotional_keywords = ["feeling", "emotion", "happy", "sad", "angry"]
        
        concept_lower = concept.lower()
        explanation_lower = explanation.lower()
        
        # Check concept and explanation for domain keywords
        for word in personal_keywords:
            if word in concept_lower or word in explanation_lower:
                return KnowledgeDomain.PERSONAL
                
        for word in social_keywords:
            if word in concept_lower or word in explanation_lower:
                return KnowledgeDomain.SOCIAL
                
        for word in procedural_keywords:
            if word in concept_lower or word in explanation_lower:
                return KnowledgeDomain.PROCEDURAL
                
        for word in emotional_keywords:
            if word in concept_lower or word in explanation_lower:
                return KnowledgeDomain.EMOTIONAL
        
        # Default to declarative knowledge
        return KnowledgeDomain.DECLARATIVE
    
    def _update_networks(self) -> None:
        """Update all networks and propagate signals between them"""
        # First, update each network individually
        network_outputs = {}
        active_networks = {}
        
        for network_type, network in self.networks.items():
            # Update the network
            outputs = network.update()
            network_outputs[network_type] = outputs
            
            # Track activation levels
            active_networks[network_type.value] = network.state.activation
        
        # Update mind state with network activations
        self.mind_state.active_networks = active_networks
        
        # Second pass: propagate outputs between networks
        for source_type, outputs in network_outputs.items():
            for target_network, output_data in outputs.items():
                if target_network in [nt.value for nt in self.networks]:
                    target_type = NetworkType(target_network)
                    # Find target network and send it the output
                    self.networks[target_type].receive_input(
                        input_data=output_data.get("data", {}),
                        source_network=source_type.value
                    )
        
        # Extract important state information from networks
        self._extract_network_states()
    
    def _extract_network_states(self) -> None:
        """Extract important state information from networks"""
        # Extract emotional state from emotions network
        emotions_network = self.networks[NetworkType.EMOTIONS]
        emotions_data = emotions_network._prepare_output_data()
        
        if "emotional_state" in emotions_data:
            self.mind_state.emotional_state = emotions_data["emotional_state"]
            
        if "dominant_emotion" in emotions_data:
            self.mind_state.dominant_emotion = emotions_data["dominant_emotion"]
        
        # Extract attention focus from attention network
        attention_network = self.networks[NetworkType.ATTENTION]
        attention_data = attention_network._prepare_output_data()
        
        if "focus_objects" in attention_data:
            self.mind_state.attention_focus = attention_data["focus_objects"]
        
        # Extract thoughts from thoughts network
        thoughts_network = self.networks[NetworkType.THOUGHTS]
        thoughts_data = thoughts_network._prepare_output_data()
        
        if "thoughts" in thoughts_data:
            self.mind_state.active_thoughts = thoughts_data["thoughts"]
        
        # Extract drive information from drives network
        drives_network = self.networks[NetworkType.DRIVES]
        drives_data = drives_network._prepare_output_data()
        
        if "dominant_drive" in drives_data:
            self.mind_state.dominant_drive = drives_data["dominant_drive"]
        
        # Extract mood information from moods network
        moods_network = self.networks[NetworkType.MOODS]
        moods_data = moods_network._prepare_output_data()
        
        if "dominant_mood" in moods_data:
            self.mind_state.dominant_mood = moods_data["dominant_mood"]
        
        # Extract self-awareness level from consciousness network
        consciousness_network = self.networks[NetworkType.CONSCIOUSNESS]
        consciousness_data = consciousness_network._prepare_output_data()
        
        if "self_awareness" in consciousness_data:
            self.mind_state.self_awareness_level = consciousness_data["self_awareness"]
    
    def _generate_response(self) -> str:
        """Generate a response based on the current mind state"""
        # Get language capabilities based on development
        language_capabilities = self.development_tracker.get_capabilities()
        
        # Get emotional state
        emotional_state = self.mind_state.emotional_state
        
        # Generate utterance based on current capabilities and state
        response = self.language_production.generate_utterance(
            capabilities=language_capabilities,
            emotional_state=emotional_state,
            context=str(self.mind_state.attention_focus),
            response_to=None  # No specific utterance to respond to
        )
        
        # If pre-linguistic, just return the babbling
        if language_capabilities.stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            return response
        
        # Apply developmental errors and simplifications
        response = self.syntactic_processor.apply_developmental_errors(
            response,
            language_capabilities.stage,
            language_capabilities.grammar_complexity
        )
        
        # Update vocabulary with used words
        words_used = response.lower().split()
        self.vocabulary_manager.update_after_child_production(words_used)
        
        # Store the utterance in episodic memory
        utterance_memory = {
            "utterance": response,
            "context": {
                "attention_focus": self.mind_state.attention_focus,
                "dominant_emotion": self.mind_state.dominant_emotion,
                "emotional_state": self.mind_state.emotional_state
            }
        }
        
        self.memory_manager.store(
            content=utterance_memory,
            memory_type=MemoryType.EPISODIC,
            tags=["utterance", "speech"],
            emotional_valence=0.0,  # Neutral unless specified
            emotional_intensity=0.3,
            salience=0.5
        )
        
        return response
    
    def _update_metrics_after_interaction(self) -> None:
        """Update developmental metrics after an interaction"""
        # Increment interaction count
        self.metrics.interactions_count += 1
        
        # Update language development features from tracker
        language_capabilities = self.development_tracker.get_capabilities()
        for feature, level in language_capabilities.feature_levels.items():
            self.metrics.feature_levels[feature.value] = level
        
        # Update vocabulary size
        vocab_stats = self.vocabulary_manager.get_vocabulary_statistics()
        self.metrics.vocabulary_size = vocab_stats.total_words
        
        # Update grammar complexity
        grammar_stats = self.syntactic_processor.get_grammar_statistics()
        mastery_percentage = grammar_stats.get("mastery_percentage", 0.0)
        self.metrics.grammar_complexity = mastery_percentage / 100.0
        
        # Update emotional development based on emotions network
        emotions_network = self.networks[NetworkType.EMOTIONS]
        emotions_data = emotions_network._prepare_output_data()
        emotional_complexity = emotions_data.get("emotional_complexity", 0.0)
        emotional_regulation = emotions_data.get("emotional_regulation", 0.0)
        self.metrics.emotional_development = (emotional_complexity + emotional_regulation) / 2.0
        
        # Update cognitive development based on consciousness and thoughts networks
        consciousness_network = self.networks[NetworkType.CONSCIOUSNESS]
        consciousness_data = consciousness_network._prepare_output_data()
        integration_capacity = consciousness_data.get("integration_capacity", 0.0)
        
        thoughts_network = self.networks[NetworkType.THOUGHTS]
        thoughts_data = thoughts_network._prepare_output_data()
        reasoning_ability = thoughts_data.get("reasoning_ability", 0.0)
        abstraction_level = thoughts_data.get("abstraction_level", 0.0)
        
        self.metrics.cognitive_development = (integration_capacity + reasoning_ability + abstraction_level) / 3.0
        
        # Update attention span
        attention_network = self.networks[NetworkType.ATTENTION]
        attention_data = attention_network._prepare_output_data()
        self.metrics.attention_span = attention_data.get("attention_span", 0.0)
        
        # Update memory capacity based on working memory and long-term memory
        working_memory_state = self.working_memory.get_state()
        memory_capacity = working_memory_state.get("used_capacity", 0) / self.working_memory.capacity
        
        long_term_stats = self.long_term_memory.get_memory_stats() or {"total_items": 0}
        long_term_size = long_term_stats.get("total_items", 0) / 1000  # Normalize to 0-1
        
        self.metrics.memory_capacity = (memory_capacity + min(1.0, long_term_size)) / 2.0
        
        # Update social development based on understanding of social dynamics
        social_concepts = self.long_term_memory.get_memories_by_domain(KnowledgeDomain.SOCIAL)
        social_development = min(1.0, len(social_concepts) / 100)  # Normalize to 0-1
        self.metrics.social_development = social_development
        
        # Update developmental stage
        self._update_developmental_stage()
    
    def _update_developmental_stage(self) -> None:
        """Update the developmental stage based on metrics"""
        overall_development = self.metrics.overall_development
        
        # Define stage thresholds
        if overall_development < 0.2:
            stage = "infancy"
        elif overall_development < 0.4:
            stage = "early_childhood"
        elif overall_development < 0.6:
            stage = "middle_childhood"
        elif overall_development < 0.8:
            stage = "adolescence"
        else:
            stage = "adulthood"
        
        # Update if changed
        if stage != self.metrics.developmental_stage:
            logger.info(f"Developmental stage changed from {self.metrics.developmental_stage} to {stage}")
            self.metrics.developmental_stage = stage
            
            # Update subsystems with new stage
            drive_network = self.networks[NetworkType.DRIVES]
            if hasattr(drive_network, "update_developmental_stage"):
                drive_network.update_developmental_stage(stage)
    
    def _record_interaction(self, mother_response: Any, child_response: str) -> None:
        """Record an interaction in the history"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "mother_response": {
                "verbal": mother_response.verbal.text,
                "emotion": mother_response.emotional.primary_emotion
            },
            "child_response": child_response,
            "child_emotion": self.mind_state.dominant_emotion,
            "developmental_metrics": {
                "age_days": self.metrics.age_days,
                "vocabulary_size": self.metrics.vocabulary_size,
                "grammar_complexity": self.metrics.grammar_complexity,
                "overall_development": self.metrics.overall_development
            }
        }
        
        self.interaction_history.append(interaction)
        
        # Keep history limited
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
    
    def advance_time(self, hours: float = 1.0) -> None:
        """Advance the simulation time
        
        Args:
            hours: Number of real hours to advance
        """
        # Calculate days to advance based on simulation speed
        days_to_advance = hours * self.simulation_speed / 24.0
        
        # Update metrics
        self.metrics.age_days += days_to_advance
        
        # Update networks based on development
        self._update_networks_development()
        
        # Apply memory decay
        self._apply_memory_decay(days_to_advance)
        
        # Update language development
        self._update_language_development(days_to_advance)
        
        logger.info(f"Advanced time by {days_to_advance:.2f} days, new age: {self.metrics.age_days:.2f} days")
    
    def _update_networks_development(self) -> None:
        """Update networks based on developmental stage"""
        age_days = self.metrics.age_days
        
        # Update emotional development
        emotions_network = self.networks[NetworkType.EMOTIONS]
        if hasattr(emotions_network, "update_development"):
            emotions_network.update_development(age_days)
        
        # Update consciousness development
        consciousness_network = self.networks[NetworkType.CONSCIOUSNESS]
        if hasattr(consciousness_network, "update_development"):
            consciousness_network.update_development(age_days, self.metrics.vocabulary_size)
        
        # Update unconsciousness development
        unconsciousness_network = self.networks[NetworkType.UNCONSCIOUSNESS]
        if hasattr(unconsciousness_network, "update_development"):
            unconsciousness_network.update_development(age_days)
        
        # Update archetypes development
        archetypes_network = self.networks[NetworkType.ARCHETYPES]
        if hasattr(archetypes_network, "update_development"):
            archetypes_network.update_development(age_days, self.mind_state.emotional_state)
        
        # Update instincts development
        instincts_network = self.networks[NetworkType.INSTINCTS]
        if hasattr(instincts_network, "update_development"):
            instincts_network.update_development(age_days)
        
        # Update perception development
        perception_network = self.networks[NetworkType.PERCEPTION]
        if hasattr(perception_network, "update_development"):
            perception_network.update_development(age_days, self.metrics.vocabulary_size)
        
        # Update attention development
        attention_network = self.networks[NetworkType.ATTENTION]
        if hasattr(attention_network, "update_development"):
            attention_network.update_development(age_days, self.metrics.interactions_count)
        
        # Update thoughts development
        thoughts_network = self.networks[NetworkType.THOUGHTS]
        if hasattr(thoughts_network, "update_development"):
            thoughts_network.update_development(age_days, self.metrics.vocabulary_size)
        
        # Update moods development
        moods_network = self.networks[NetworkType.MOODS]
        if hasattr(moods_network, "update_development"):
            moods_network.update_development(age_days)
    
    def _apply_memory_decay(self, days_elapsed: float) -> None:
        """Apply decay to memory systems"""
        # Apply decay to working memory
        self.working_memory.update()
        
        # Apply decay to long-term memory
        decay_rate = self.config.memory.long_term_decay_rate
        self.long_term_memory.apply_memory_decay(decay_rate * days_elapsed)
        
        # Apply decay to episodic memory
        self.episodic_memory.apply_decay(decay_rate * days_elapsed)
        
        # Apply decay to associative memory
        self.associative_memory.apply_decay(decay_rate * days_elapsed)
        
        # Apply decay to vocabulary
        self.vocabulary_manager.apply_memory_decay(days_elapsed)
    
    def _update_language_development(self, days_elapsed: float) -> None:
        """Update language development based on time"""
        # Update syntactic processor based on development
        self.syntactic_processor.update_rule_masteries(
            stage=self.development_tracker.get_capabilities().stage,
            grammar_complexity=self.metrics.grammar_complexity
        )
        
        # Update language development tracker
        self.development_tracker.update(self.metrics.age_days, self.metrics.vocabulary_size)
    
    def consolidate_memories(self) -> None:
        """Consolidate memories between memory systems"""
        # Working memory to long-term memory consolidation
        consolidation_candidates = self.working_memory.get_consolidation_candidates()
        
        for memory_id in consolidation_candidates:
            memory_item = self.memory_manager.retrieve(memory_id)
            if memory_item:
                # Store in long-term memory
                self.long_term_memory.store(
                    memory_item=memory_item,
                    domain=self._classify_memory_content(memory_item.content),
                    importance=0.6
                )
        
        # Episodic memory consolidation
        if self.metrics.age_days > 5:  # Only after sufficient development
            self.episodic_memory.consolidate_recent_episodes()
    
    def _classify_memory_content(self, content: Any) -> Optional[KnowledgeDomain]:
        """Classify memory content into a knowledge domain"""
        # Convert content to string representation for analysis
        content_str = str(content)
        
        # Use same classification logic as for concepts
        return self._classify_concept_domain("", content_str)
    
    def get_development_metrics(self) -> Dict[str, Any]:
        """Get current developmental metrics"""
        return self.metrics.model_dump()
    
    def get_mind_state(self) -> Dict[str, Any]:
        """Get current mind state"""
        return self.mind_state.model_dump()
    
    def get_network_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the state of all networks"""
        network_states = {}
        for network_type, network in self.networks.items():
            network_states[network_type.value] = network.get_state()
        return network_states
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the current state of the neural child
        
        Args:
            filepath: Path to save the state to, defaults to timestamp-based path
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.data_dir / f"neural_child_state_{timestamp}.json"
        
        # Save memory systems
        memory_dir = self.data_dir / "memory" / timestamp
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        memory_manager_path = memory_dir / "memory_manager.json"
        self.memory_manager.save_state(memory_manager_path)
        
        working_memory_path = memory_dir / "working_memory.json"
        # Working memory doesn't have a direct save_state method
        
        long_term_memory_path = memory_dir / "long_term_memory.json"
        self.long_term_memory.save_state(long_term_memory_path)
        
        episodic_memory_path = memory_dir / "episodic_memory.json"
        self.episodic_memory.save_state(episodic_memory_path)
        
        associative_memory_path = memory_dir / "associative_memory.json"
        self.associative_memory.save_state(associative_memory_path)
        
        # Save vocabulary
        vocabulary_path = self.data_dir / "vocabulary" / f"vocabulary_{timestamp}.json"
        self.vocabulary_manager.save_state(str(vocabulary_path))
        
        # Save metrics and state
        metrics_state = self.metrics.model_dump()
        mind_state = self.mind_state.model_dump()
        
        # Save network states
        network_states = {}
        for network_type, network in self.networks.items():
            network_states[network_type.value] = network.get_state()
        
        # Create combined state
        state = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics_state,
            "mind_state": mind_state,
            "network_states": network_states,
            "interaction_history": self.interaction_history[-20:],  # Last 20 interactions
            "memory_paths": {
                "memory_manager": str(memory_manager_path),
                "long_term_memory": str(long_term_memory_path),
                "episodic_memory": str(episodic_memory_path),
                "associative_memory": str(associative_memory_path)
            },
            "vocabulary_path": str(vocabulary_path),
            "simulation_speed": self.simulation_speed
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved neural child state to {filepath}")
        self.last_save_time = datetime.now()
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of the neural child
        
        Args:
            filepath: Path to the state file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load metrics
            if "metrics" in state:
                self.metrics = DevelopmentMetrics(**state["metrics"])
            
            # Load mind state
            if "mind_state" in state:
                self.mind_state = MindState(**state["mind_state"])
            
            # Load network states
            if "network_states" in state:
                for network_type_str, network_state in state["network_states"].items():
                    network_type = NetworkType(network_type_str)
                    if network_type in self.networks:
                        self.networks[network_type].load_state(network_state)
            
            # Load interaction history
            if "interaction_history" in state:
                self.interaction_history = state["interaction_history"]
            
            # Load memory systems
            memory_paths = state.get("memory_paths", {})
            
            if "memory_manager" in memory_paths:
                memory_manager_path = Path(memory_paths["memory_manager"])
                if os.path.exists(memory_manager_path):
                    self.memory_manager.load_state(memory_manager_path)
            
            if "long_term_memory" in memory_paths:
                long_term_memory_path = Path(memory_paths["long_term_memory"])
                if os.path.exists(long_term_memory_path):
                    self.long_term_memory.load_state(long_term_memory_path)
            
            if "episodic_memory" in memory_paths:
                episodic_memory_path = Path(memory_paths["episodic_memory"])
                if os.path.exists(episodic_memory_path):
                    self.episodic_memory.load_state(episodic_memory_path)
            
            if "associative_memory" in memory_paths:
                associative_memory_path = Path(memory_paths["associative_memory"])
                if os.path.exists(associative_memory_path):
                    self.associative_memory.load_state(associative_memory_path)
            
            # Load vocabulary
            if "vocabulary_path" in state:
                vocabulary_path = Path(state["vocabulary_path"])
                if os.path.exists(vocabulary_path):
                    # The vocabulary manager doesn't have a direct load_state method
                    pass
            
            # Load simulation speed
            if "simulation_speed" in state:
                self.simulation_speed = state["simulation_speed"]
            
            logger.info(f"Loaded neural child state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading neural child state: {str(e)}")
            return False
    
    def get_child_observation_for_mother(self) -> Dict[str, Any]:
        """Get an observation of the child that the mother can perceive
        
        Returns:
            Dictionary with observable child state
        """
        # Get the last response if available
        message = ""
        if self.interaction_history:
            last_interaction = self.interaction_history[-1]
            message = last_interaction.get("child_response", "")
        
        # Determine apparent emotion
        apparent_emotion = self.mind_state.dominant_emotion or "neutral"
        
        # Create a simplified observation that represents what a mother would see
        observation = {
            "message": message,
            "apparent_emotion": apparent_emotion,
            "vocabulary_size": self.metrics.vocabulary_size,
            "age_days": self.metrics.age_days,
            "recent_concepts_learned": self._get_recently_learned_concepts(),
            "attention_span": self.metrics.attention_span
        }
        
        return observation
    
    def _get_recently_learned_concepts(self) -> List[str]:
        """Get a list of recently learned concepts"""
        # Get recent learning events from vocabulary manager
        recent_events = self.vocabulary_manager.get_recent_learning_events(5)
        recent_words = [event.word for event in recent_events]
        
        # Get recent concepts from long-term memory
        memory_query = MemoryQuery(
            tags=["concept"],
            memory_types=[MemoryType.LONG_TERM],
            max_results=5,
            salience_threshold=0.5
        )
        search_result = self.memory_manager.search(memory_query)
        
        concept_items = []
        for item in search_result.items:
            if hasattr(item, "content") and isinstance(item.content, dict):
                concept = item.content.get("concept")
                if concept:
                    concept_items.append(concept)
        
        # Combine and return unique items
        all_concepts = list(set(recent_words + concept_items))
        return all_concepts[:5]  # Return at most 5 concepts

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the neural child's current status
        
        Returns:
            Dictionary with key status information
        """
        language_capabilities = self.development_tracker.get_capabilities()
        
        return {
            "age_days": self.metrics.age_days,
            "developmental_stage": self.metrics.developmental_stage,
            "vocabulary_size": self.metrics.vocabulary_size,
            "language_stage": language_capabilities.stage.value,
            "grammar_complexity": self.metrics.grammar_complexity,
            "emotional_development": self.metrics.emotional_development,
            "cognitive_development": self.metrics.cognitive_development,
            "social_development": self.metrics.social_development,
            "dominant_emotion": self.mind_state.dominant_emotion,
            "dominant_mood": self.mind_state.dominant_mood,
            "dominant_drive": self.mind_state.dominant_drive,
            "active_thoughts_count": len(self.mind_state.active_thoughts),
            "interactions_count": self.metrics.interactions_count,
            "overall_development": self.metrics.overall_development
        }

# Example of how to use the NeuralChild class
if __name__ == "__main__":
    # Initialize the neural child
    child = NeuralChild(simulation_speed=24.0)  # 1 day per hour
    
    # Output initial state
    status = child.get_status_summary()
    print(f"Neural child initialized at age {status['age_days']:.2f} days")
    print(f"Developmental stage: {status['developmental_stage']}")
    print(f"Language stage: {status['language_stage']}")
    
    # Initialize the LLM client for processing
    llm_client = LLMClient()
    
    # Example of mother's response (simplified)
    from mother import MotherResponse, VerbalResponse, EmotionalState, NonVerbalResponse, TeachingElements, ChildPerception, ParentingApproach, ContextAwareness
    
    # Create a simple mother response
    mother_response = MotherResponse(
        verbal=VerbalResponse(
            text="Hello, little one! How are you today?",
            tone="warm",
            complexity_level=0.3
        ),
        emotional=EmotionalState(
            primary_emotion="joy",
            intensity=0.7,
            secondary_emotion=None,
            patience_level=0.9
        ),
        non_verbal=NonVerbalResponse(
            physical_actions=["smiles", "leans closer"],
            facial_expression="warm smile",
            proximity="close"
        ),
        teaching=TeachingElements(
            vocabulary=[],
            concepts=[],
            values=[],
            corrections=[]
        ),
        perception=ChildPerception(
            child_emotion="curious",
            child_needs=["attention", "connection"],
            misinterpretations=None
        ),
        parenting=ParentingApproach(
            intention="establish connection",
            approach="warm and inviting",
            consistency=0.9,
            adaptation_to_development=0.8
        ),
        context_awareness=ContextAwareness(
            references_previous_interactions=None,
            environment_factors=None,
            recognizes_progress=None
        )
    )
    
    # Process the mother's response
    result = child.process_mother_response(mother_response)
    print(f"Child's response: {result['response']}")
    
    # Advance time
    child.advance_time(hours=2)
    
    # Check status after time advancement
    status = child.get_status_summary()
    print(f"Neural child is now {status['age_days']:.2f} days old")
    
    # Save the state
    child.save_state()