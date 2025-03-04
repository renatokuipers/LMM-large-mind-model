# neural_child.py
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import json
import os
import random
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np

# LLM module for communication
from llm_module import LLMClient, Message

# Mother component for interaction
from mother import Mother, MotherResponse, ChildState, EmotionalState as MotherEmotionalState

# Network imports - complete neural architecture
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

# Language and memory imports
from language.developmental_stages import DevelopmentTracker, LanguageDevelopmentStage, LanguageCapabilities
from language.lexical_memory import LexicalMemory
from language.production import LanguageProduction
from language.syntactic_processor import SyntacticProcessor
from language.vocabulary import VocabularyManager
from language.semantic_network import SemanticNetwork, SemanticRelation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralChild")

# ========== DEVELOPMENT ENUMS & MODELS ==========

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class HumanMindDevelopmentStage(str, Enum):
    """
    Stages of Human Mind Development

    This model integrates multiple perspectives to reflect the real progression of mental development
    """
    PRENATAL = "prenatal"
    INFANCY = "infancy"
    EARLY_CHILDHOOD = "early_childhood"
    MIDDLE_CHILDHOOD = "middle_childhood"
    ADOLESCENCE = "adolescence"
    YOUNG_ADULTHOOD = "young_adulthood"
    MIDDLE_ADULTHOOD = "middle_adulthood"
    LATE_ADULTHOOD = "late_adulthood"
    
    @property
    def age_range(self) -> str:
        """Return the age range for this development stage"""
        ranges = {
            self.PRENATAL: "Conception to Birth",
            self.INFANCY: "0-2 years",
            self.EARLY_CHILDHOOD: "2-7 years",
            self.MIDDLE_CHILDHOOD: "7-11 years",
            self.ADOLESCENCE: "11-18 years",
            self.YOUNG_ADULTHOOD: "18-40 years",
            self.MIDDLE_ADULTHOOD: "40-65 years",
            self.LATE_ADULTHOOD: "65+ years"
        }
        return ranges[self]
    
    @property
    def description(self) -> str:
        """Return the description for this development stage"""
        descriptions = {
            self.PRENATAL: "Rapid neural proliferation, migration, and synapse formation establishing the brain's core architecture.",
            self.INFANCY: "Sensorimotor exploration, early attachment formation, and the emergence of object permanence.",
            self.EARLY_CHILDHOOD: "Explosive language acquisition, symbolic thought, early self-awareness, and rudimentary emotional regulation.",
            self.MIDDLE_CHILDHOOD: "Development of logical reasoning, memory, social cognition, and foundational moral understanding.",
            self.ADOLESCENCE: "Emergence of abstract reasoning, identity exploration, emotional complexity, and maturation of the prefrontal cortex.",
            self.YOUNG_ADULTHOOD: "Integration of cognitive and emotional faculties, advanced problem solving, stable identity, and meaningful interpersonal relationships.",
            self.MIDDLE_ADULTHOOD: "Reflective thought, wisdom accumulation, and balancing emotional regulation with cognitive complexity amid life transitions.",
            self.LATE_ADULTHOOD: "Life review, synthesis of experiences, and potential cognitive shifts balanced by enduring knowledge and perspective."
        }
        return descriptions[self]
    
    @classmethod
    def get_stage_for_age(cls, age_days: float) -> 'HumanMindDevelopmentStage':
        """Determine development stage based on simulated age in days"""
        # Simplified mapping - in a real system this would be more nuanced
        # These thresholds are accelerated for simulation purposes
        if age_days < 1:
            return cls.PRENATAL
        elif age_days < 30:  # First month
            return cls.INFANCY
        elif age_days < 180:  # First 6 months
            return cls.EARLY_CHILDHOOD
        elif age_days < 365:  # First year
            return cls.MIDDLE_CHILDHOOD
        elif age_days < 730:  # Second year
            return cls.ADOLESCENCE
        elif age_days < 1095:  # Third year
            return cls.YOUNG_ADULTHOOD
        elif age_days < 1460:  # Fourth year
            return cls.MIDDLE_ADULTHOOD
        else:
            return cls.LATE_ADULTHOOD

class EmotionalState(BaseModel):
    """Child's emotional state, inspired by Plutchik's wheel of emotions"""
    joy: float = Field(0.0, ge=0.0, le=1.0)
    sadness: float = Field(0.0, ge=0.0, le=1.0)
    anger: float = Field(0.0, ge=0.0, le=1.0)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    disgust: float = Field(0.0, ge=0.0, le=1.0)
    trust: float = Field(0.0, ge=0.0, le=1.0)
    anticipation: float = Field(0.0, ge=0.0, le=1.0)
    
    def dominant_emotion(self) -> Tuple[str, float]:
        """Return the dominant emotion and its intensity"""
        emotions = {
            "joy": self.joy,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust,
            "trust": self.trust,
            "anticipation": self.anticipation
        }
        dominant = max(emotions, key=emotions.get)
        return dominant, emotions[dominant]
    
    def update_from_mother_response(self, mother_response: MotherResponse) -> None:
        """Update emotional state based on mother's response"""
        # Emotional contagion - child picks up on mother's emotions
        emotional_contagion_factor = 0.2
        
        # Influence based on mother's primary emotion
        primary_emotion = mother_response.emotional.primary_emotion.lower()
        intensity = mother_response.emotional.intensity * emotional_contagion_factor
        
        # Map mother's emotion to child's emotional state components
        if primary_emotion in ("joy", "happiness", "excitement"):
            self.joy = min(1.0, self.joy + intensity)
        elif primary_emotion in ("sadness", "sorrow", "grief"):
            self.sadness = min(1.0, self.sadness + intensity)
        elif primary_emotion in ("anger", "frustration", "irritation"):
            self.anger = min(1.0, self.anger + intensity)
        elif primary_emotion in ("fear", "anxiety", "worry"):
            self.fear = min(1.0, self.fear + intensity)
        elif primary_emotion in ("surprise", "amazement", "astonishment"):
            self.surprise = min(1.0, self.surprise + intensity)
        elif primary_emotion in ("disgust", "dislike", "aversion"):
            self.disgust = min(1.0, self.disgust + intensity)
        elif primary_emotion in ("trust", "acceptance", "admiration"):
            self.trust = min(1.0, self.trust + intensity)
        elif primary_emotion in ("anticipation", "interest", "vigilance"):
            self.anticipation = min(1.0, self.anticipation + intensity)
        
        # Apply natural decay to all emotions
        decay_factor = 0.95
        self.joy *= decay_factor
        self.sadness *= decay_factor
        self.anger *= decay_factor
        self.fear *= decay_factor
        self.surprise *= decay_factor
        self.disgust *= decay_factor
        self.trust *= decay_factor
        self.anticipation *= decay_factor

class DevelopmentalMetrics(BaseModel):
    """Tracking metrics for child's development"""
    age_days: float = Field(0.0, ge=0.0)
    vocabulary_size: int = Field(0, ge=0)
    development_stage: HumanMindDevelopmentStage = Field(HumanMindDevelopmentStage.PRENATAL)
    attention_span: float = Field(0.1, ge=0.0, le=1.0)
    emotional_stability: float = Field(0.1, ge=0.0, le=1.0)
    social_awareness: float = Field(0.0, ge=0.0, le=1.0)
    abstraction_capability: float = Field(0.0, ge=0.0, le=1.0)
    total_interactions: int = Field(0, ge=0)
    total_training_time: float = Field(0.0, ge=0.0)  # In hours
    
    @model_validator(mode='after')
    def validate_development_stage(self) -> 'DevelopmentalMetrics':
        """Ensure development stage matches age"""
        expected_stage = HumanMindDevelopmentStage.get_stage_for_age(self.age_days)
        if expected_stage != self.development_stage:
            logger.info(f"Development stage updated from {self.development_stage} to {expected_stage}")
            self.development_stage = expected_stage
        return self

class ChildConfig(BaseModel):
    """Configuration for the neural child"""
    # Learning parameters
    learning_rate_multiplier: float = Field(1.0, gt=0.0)
    emotional_sensitivity: float = Field(0.7, ge=0.0, le=1.0)
    curiosity_factor: float = Field(0.8, ge=0.0, le=1.0)
    memory_retention: float = Field(0.6, ge=0.0, le=1.0)
    
    # Development simulation
    simulated_time_ratio: float = Field(90.0, gt=0.0)  # How many simulated days per real day
    chat_readiness_threshold: float = Field(10.0, ge=0.0)  # Age in days when chat becomes available
    
    # Network weights - which aspects of personality are emphasized
    network_emphasis: Dict[NetworkType, float] = Field(default_factory=dict)
    
    @field_validator('network_emphasis')
    @classmethod
    def validate_network_emphasis(cls, v):
        """Ensure all network types have emphasis values and they sum to 1.0"""
        # Set default values for any missing network types
        for network_type in NetworkType:
            if network_type not in v:
                v[network_type] = 0.1
        
        # Normalize to ensure sum is 1.0
        total = sum(v.values())
        if total > 0:
            for network_type in v:
                v[network_type] /= total
        
        return v

# ========== NEURAL CHILD IMPLEMENTATION ==========

class NeuralChild:
    """Main coordinator for the neural child system"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        save_dir: Path = Path("./data/neural_child"),
        simulation_speed: float = 1.0,  # Multiplier for development speed
        llm_base_url: str = "http://192.168.2.12:1234"
    ):
        """Initialize the neural child with configuration"""
        # Load or create config
        self.config = self._load_config(config_path)
        self.save_dir = save_dir
        self.simulation_speed = simulation_speed
        
        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Initialize the child's state
        self.metrics = DevelopmentalMetrics()
        self.emotional_state = EmotionalState()
        
        # Initialize LLM client for Mother interactions
        self.llm_client = LLMClient(base_url=llm_base_url)
        
        # Initialize Mother component
        self.mother = Mother(
            llm_client=self.llm_client,
            model="qwen2.5-7b-instruct",
            temperature=0.75,
            max_tokens=1000,
            history_size=10
        )
        
        # Initialize neural networks
        self.networks = self._initialize_networks()
        
        # Connect networks to each other
        self._establish_network_connections()
        
        # Initialize language systems
        self.syntactic_processor = SyntacticProcessor()
        self.language_tracker = DevelopmentTracker(initial_age_days=self.metrics.age_days)
        self.vocabulary_manager = VocabularyManager(simulation_speed=simulation_speed)
        
        # Initialize language production component
        self.language_production = LanguageProduction(
            lexical_memory=self.vocabulary_manager.lexical_memory,
            syntactic_processor=self.syntactic_processor
        )
        
        # Memory for recent interactions
        self.recent_interactions = []
        
        logger.info("Neural child initialized with complete neural architecture")
    
    def _load_config(self, config_path: Optional[Path]) -> ChildConfig:
        """Load or create configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return ChildConfig(**config_data)
        
        # Default configuration
        return ChildConfig(
            network_emphasis={
                NetworkType.EMOTIONS: 0.2,
                NetworkType.PERCEPTION: 0.2,
                NetworkType.CONSCIOUSNESS: 0.15,
                NetworkType.INSTINCTS: 0.15,
                NetworkType.ATTENTION: 0.1,
                NetworkType.ARCHETYPES: 0.05,
                NetworkType.DRIVES: 0.05,
                NetworkType.THOUGHTS: 0.05,
                NetworkType.MOODS: 0.03,
                NetworkType.UNCONSCIOUSNESS: 0.02
            }
        )
    
    def _initialize_networks(self) -> Dict[NetworkType, Any]:
        """Initialize all neural networks with proper configurations"""
        networks = {}
        
        # Initialize each network with appropriate parameters
        networks[NetworkType.ARCHETYPES] = ArchetypesNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.15
        )
        
        networks[NetworkType.INSTINCTS] = InstinctsNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier * 0.5,
            activation_threshold=0.1
        )
        
        networks[NetworkType.UNCONSCIOUSNESS] = UnconsciousnessNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier * 0.8,
            activation_threshold=0.1
        )
        
        networks[NetworkType.DRIVES] = DrivesNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.2
        )
        
        networks[NetworkType.EMOTIONS] = EmotionsNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.15,
            emotional_sensitivity=self.config.emotional_sensitivity
        )
        
        networks[NetworkType.MOODS] = MoodsNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier * 0.8,
            activation_threshold=0.2,
            baseline_mood="neutral"
        )
        
        networks[NetworkType.ATTENTION] = AttentionNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.2,
            attention_span_base=0.2
        )
        
        networks[NetworkType.PERCEPTION] = PerceptionNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.1
        )
        
        networks[NetworkType.CONSCIOUSNESS] = ConsciousnessNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.3,
            self_awareness_base=0.1
        )
        
        networks[NetworkType.THOUGHTS] = ThoughtsNetwork(
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            activation_threshold=0.3
        )
        
        return networks
    
    def _establish_network_connections(self) -> None:
        """Establish connections between neural networks"""
        # Perception connects to many networks as it's the input gateway
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
        
        self.networks[NetworkType.PERCEPTION].connect_to(
            target_network=NetworkType.UNCONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.5
        )
        
        # Attention modulates consciousness and thoughts
        self.networks[NetworkType.ATTENTION].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.7
        )
        
        self.networks[NetworkType.ATTENTION].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.6
        )
        
        # Emotions influence many networks
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.MOODS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8
        )
        
        self.networks[NetworkType.EMOTIONS].connect_to(
            target_network=NetworkType.DRIVES.value,
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
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Unconsciousness connects to archetypes and consciousness
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.ARCHETYPES.value,
            connection_type=ConnectionType.ASSOCIATIVE,
            initial_strength=0.6
        )
        
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
        
        self.networks[NetworkType.UNCONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.ASSOCIATIVE,
            initial_strength=0.5
        )
        
        # Instincts connect to drives and emotions
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.DRIVES.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.7
        )
        
        self.networks[NetworkType.INSTINCTS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Drives connect to consciousness and thoughts
        self.networks[NetworkType.DRIVES].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.5
        )
        
        self.networks[NetworkType.DRIVES].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.6
        )
        
        # Moods influence emotions and thoughts
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5,
            bidirectional=True
        )
        
        self.networks[NetworkType.MOODS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
        
        # Consciousness connects to thoughts
        self.networks[NetworkType.CONSCIOUSNESS].connect_to(
            target_network=NetworkType.THOUGHTS.value,
            connection_type=ConnectionType.EXCITATORY,
            initial_strength=0.8,
            bidirectional=True
        )
        
        # Archetypes connect to consciousness and emotions
        self.networks[NetworkType.ARCHETYPES].connect_to(
            target_network=NetworkType.CONSCIOUSNESS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.4
        )
        
        self.networks[NetworkType.ARCHETYPES].connect_to(
            target_network=NetworkType.EMOTIONS.value,
            connection_type=ConnectionType.MODULATORY,
            initial_strength=0.5
        )
        
        logger.info("Established connections between neural networks")
    
    def process_mother_response(self, mother_response: MotherResponse) -> None:
        """Process a response from the mother, updating internal state through neural networks"""
        logger.info("Processing mother's response through neural architecture")
        
        # Update emotional state directly based on mother's response
        self.emotional_state.update_from_mother_response(mother_response)
        
        # Process through perception network first - this is the entry point
        perception_input = {
            "verbal": mother_response.verbal.text,
            "non_verbal": {
                "physical_actions": mother_response.non_verbal.physical_actions,
                "facial_expression": mother_response.non_verbal.facial_expression,
                "proximity": mother_response.non_verbal.proximity
            },
            "emotional_context": {
                "primary": mother_response.emotional.primary_emotion,
                "intensity": mother_response.emotional.intensity,
                "secondary": mother_response.emotional.secondary_emotion
            }
        }
        
        self.networks[NetworkType.PERCEPTION].receive_input(perception_input, "mother")
        perception_output = self.networks[NetworkType.PERCEPTION].process_inputs()
        
        # Process verbal content through language systems
        self._process_verbal_content(mother_response.verbal.text, mother_response.teaching)
        
        # Propagate perception output to connected networks
        self._propagate_network_outputs(NetworkType.PERCEPTION, perception_output)
        
        # Process through attention network
        attention_input = {
            "percepts": perception_output.get("percepts", []),
            "emotional_state": {
                emotion: getattr(self.emotional_state, emotion)
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
                if getattr(self.emotional_state, emotion) > 0.1
            }
        }
        
        self.networks[NetworkType.ATTENTION].receive_input(attention_input, NetworkType.PERCEPTION.value)
        attention_output = self.networks[NetworkType.ATTENTION].process_inputs()
        
        # Propagate attention output to connected networks
        self._propagate_network_outputs(NetworkType.ATTENTION, attention_output)
        
        # Process through emotions network
        emotions_input = {
            "stimuli": perception_output.get("percepts", []),
            "mother_emotion": {
                "emotion": mother_response.emotional.primary_emotion,
                "intensity": mother_response.emotional.intensity
            },
            "interpersonal": {
                "mother_emotion": mother_response.emotional.primary_emotion,
                "intensity": mother_response.emotional.intensity,
                "mother_actions": mother_response.non_verbal.physical_actions
            }
        }
        
        self.networks[NetworkType.EMOTIONS].receive_input(emotions_input, NetworkType.PERCEPTION.value)
        emotions_output = self.networks[NetworkType.EMOTIONS].process_inputs()
        
        # Propagate emotions output to connected networks
        self._propagate_network_outputs(NetworkType.EMOTIONS, emotions_output)
        
        # Process through instincts network
        instincts_input = {
            "stimuli": perception_output.get("percepts", []),
            "percepts": perception_output.get("percepts", [])
        }
        
        self.networks[NetworkType.INSTINCTS].receive_input(instincts_input, NetworkType.PERCEPTION.value)
        instincts_output = self.networks[NetworkType.INSTINCTS].process_inputs()
        
        # Propagate instincts output to connected networks
        self._propagate_network_outputs(NetworkType.INSTINCTS, instincts_output)
        
        # Process through drives network
        drives_input = {
            "mother_response": {
                "verbal": mother_response.verbal.text,
                "non_verbal": {
                    "physical_actions": mother_response.non_verbal.physical_actions
                }
            },
            "emotional_state": {
                emotion: getattr(self.emotional_state, emotion)
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
                if getattr(self.emotional_state, emotion) > 0.1
            }
        }
        
        self.networks[NetworkType.DRIVES].receive_input(drives_input, "mother")
        drives_output = self.networks[NetworkType.DRIVES].process_inputs()
        
        # Propagate drives output to connected networks
        self._propagate_network_outputs(NetworkType.DRIVES, drives_output)
        
        # Process through unconsciousness network
        unconscious_input = {
            "stimuli": perception_output.get("percepts", []),
            "emotional_state": {
                emotion: getattr(self.emotional_state, emotion)
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
                if getattr(self.emotional_state, emotion) > 0.1
            }
        }
        
        self.networks[NetworkType.UNCONSCIOUSNESS].receive_input(unconscious_input, NetworkType.PERCEPTION.value)
        unconscious_output = self.networks[NetworkType.UNCONSCIOUSNESS].process_inputs()
        
        # Propagate unconsciousness output to connected networks
        self._propagate_network_outputs(NetworkType.UNCONSCIOUSNESS, unconscious_output)
        
        # Process through archetypes network
        archetypes_input = {
            "stimuli": perception_output.get("percepts", []),
            "emotional_state": {
                emotion: getattr(self.emotional_state, emotion)
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
                if getattr(self.emotional_state, emotion) > 0.1
            },
            "triggered_concepts": unconscious_output.get("triggered_concepts", [])
        }
        
        self.networks[NetworkType.ARCHETYPES].receive_input(archetypes_input, NetworkType.UNCONSCIOUSNESS.value)
        archetypes_output = self.networks[NetworkType.ARCHETYPES].process_inputs()
        
        # Propagate archetypes output to connected networks
        self._propagate_network_outputs(NetworkType.ARCHETYPES, archetypes_output)
        
        # Process through moods network
        moods_input = {
            "emotional_state": emotions_output.get("emotional_state", {}),
            "environment": {
                "novelty": len(perception_output.get("percepts", [])) / 10.0
            }
        }
        
        self.networks[NetworkType.MOODS].receive_input(moods_input, NetworkType.EMOTIONS.value)
        moods_output = self.networks[NetworkType.MOODS].process_inputs()
        
        # Propagate moods output to connected networks
        self._propagate_network_outputs(NetworkType.MOODS, moods_output)
        
        # Process through consciousness network (integrates many inputs)
        consciousness_input = {
            "perceptions": perception_output.get("percepts", []),
            "emotions": emotions_output.get("emotional_state", {}),
            "attention_focus": attention_output.get("focus_objects", [])
        }
        
        self.networks[NetworkType.CONSCIOUSNESS].receive_input(consciousness_input, "integration")
        consciousness_output = self.networks[NetworkType.CONSCIOUSNESS].process_inputs()
        
        # Propagate consciousness output to connected networks
        self._propagate_network_outputs(NetworkType.CONSCIOUSNESS, consciousness_output)
        
        # Finally, process through thoughts network (final integration)
        thoughts_input = {
            "perceptions": perception_output.get("percepts", []),
            "consciousness": {
                "active_contents": consciousness_output.get("active_contents", {}),
                "self_representations": consciousness_output.get("self_representations", [])
            },
            "emotional_state": emotions_output.get("emotional_state", {}),
            "vocabulary": list(self.vocabulary_manager.lexical_memory.words.keys())
        }
        
        self.networks[NetworkType.THOUGHTS].receive_input(thoughts_input, NetworkType.CONSCIOUSNESS.value)
        thoughts_output = self.networks[NetworkType.THOUGHTS].process_inputs()
        
        # Add to recent interactions
        self.recent_interactions.append({
            "timestamp": datetime.now().isoformat(),
            "mother_verbal": mother_response.verbal.text,
            "mother_emotion": mother_response.emotional.primary_emotion,
            "child_emotion": self.emotional_state.dominant_emotion()[0],
            "thoughts": thoughts_output.get("thoughts", [])
        })
        
        if len(self.recent_interactions) > 20:
            self.recent_interactions.pop(0)
        
        # Apply language development updates
        self.language_tracker.update(
            age_days=self.metrics.age_days,
            vocabulary_size=len(self.vocabulary_manager.lexical_memory.words)
        )
        
        # Update syntactic processor based on language capabilities
        capabilities = self.language_tracker.get_capabilities()
        self.syntactic_processor.update_rule_masteries(
            capabilities.stage, 
            capabilities.grammar_complexity
        )
        
        # Update developmental metrics
        self._update_metrics(mother_response)
        
        # Update network development based on age
        self._update_network_development()
        
        logger.info(f"Processed mother's response through neural architecture")
        logger.info(f"Current emotional state: {self.emotional_state.dominant_emotion()[0]}")
        logger.info(f"Generated thoughts: {thoughts_output.get('thoughts', [])}")
    
    def _propagate_network_outputs(self, source_network: NetworkType, output_data: Dict[str, Any]) -> None:
        """Propagate outputs from one network to its connected networks"""
        # Get the source network instance
        source = self.networks[source_network]
        
        # For each connection from this network
        for target_name, connection in source.connections.items():
            # Find the target network by name
            target_network = None
            for network_type, network in self.networks.items():
                if network.name == target_name:
                    target_network = network
                    break
            
            if target_network:
                # Prepare data based on connection type
                signal_strength = connection.strength * source.state.activation
                
                propagated_data = {
                    "source_activation": source.state.activation,
                    "signal_strength": signal_strength,
                    "source_network": source.name,
                    "connection_type": connection.connection_type.value
                }
                
                # Add output data from source network
                propagated_data.update(output_data)
                
                # Send to target network
                target_network.receive_input(propagated_data, source.name)
    
    def _process_verbal_content(self, text: str, teaching_elements) -> None:
        """Extract and learn from verbal content through language systems"""
        # Process general speech with vocabulary manager
        emotional_state_dict = {
            emotion: getattr(self.emotional_state, emotion)
            for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
            if getattr(self.emotional_state, emotion) > 0.1
        }
        
        self.vocabulary_manager.process_heard_speech(
            text=text,
            context="mother's speech",
            emotional_state=emotional_state_dict
        )
        
        # Process explicit teaching elements
        for vocab_item in teaching_elements.vocabulary:
            self.vocabulary_manager.explicitly_learn_word(
                word=vocab_item.word,
                definition=vocab_item.simple_definition,
                example_usage=vocab_item.example_usage,
                emotional_state=emotional_state_dict
            )
            logger.info(f"Explicitly learned word: {vocab_item.word}")
        
        # Process concept teaching
        for concept_item in teaching_elements.concepts:
            # Add concept to semantic network
            self.vocabulary_manager.semantic_network.add_concept(
                word=concept_item.concept_name,
                properties={},
                emotions=emotional_state_dict
            )
            
            # If concept already exists as a word, update its definition
            if concept_item.concept_name.lower() in self.vocabulary_manager.lexical_memory.words:
                word_item = self.vocabulary_manager.lexical_memory.words[concept_item.concept_name.lower()]
                word_item.definition = concept_item.explanation
                
            # Otherwise, explicitly learn it
            else:
                self.vocabulary_manager.explicitly_learn_word(
                    word=concept_item.concept_name,
                    definition=concept_item.explanation,
                    example_usage=concept_item.relevance,
                    emotional_state=emotional_state_dict
                )
            
            logger.info(f"Learned new concept: {concept_item.concept_name}")
    
    def _update_metrics(self, mother_response: MotherResponse) -> None:
        """Update developmental metrics based on interaction"""
        # Update interaction count
        self.metrics.total_interactions += 1
        
        # Update vocabulary size
        self.metrics.vocabulary_size = len(self.vocabulary_manager.lexical_memory.words)
        
        # Update age (accelerated for simulation)
        time_increment = 0.1 * self.config.simulated_time_ratio * self.simulation_speed
        self.metrics.age_days += time_increment
        
        # Update training time (in hours)
        self.metrics.total_training_time += (time_increment / 24.0)
        
        # Update attention span based on interaction quality
        attention_increment = 0.001 * mother_response.parenting.adaptation_to_development
        self.metrics.attention_span = min(1.0, self.metrics.attention_span + attention_increment)
        
        # Update emotional stability based on emotional network state
        self.metrics.emotional_stability = min(1.0, self.networks[NetworkType.EMOTIONS].emotional_regulation)
        
        # Update social awareness if relevant teaching
        social_values = ["sharing", "empathy", "cooperation", "friendship"]
        for value in mother_response.teaching.values:
            if any(sv in value.lower() for sv in social_values):
                social_increment = 0.003
                self.metrics.social_awareness = min(1.0, self.metrics.social_awareness + social_increment)
                break
        
        # Update abstraction capability based on thoughts network
        self.metrics.abstraction_capability = min(1.0, self.networks[NetworkType.THOUGHTS].abstraction_level)
        
        # Validate development stage against age
        self.metrics = self.metrics.model_validate(self.metrics.model_dump())
    
    def _update_network_development(self) -> None:
        """Update all networks' developmental parameters based on age and vocabulary"""
        # Update each network with appropriate developmental parameters
        self.networks[NetworkType.ARCHETYPES].update_development(
            age_days=self.metrics.age_days,
            emotional_state={
                emotion: getattr(self.emotional_state, emotion)
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
                if getattr(self.emotional_state, emotion) > 0.1
            }
        )
        
        self.networks[NetworkType.INSTINCTS].update_development(age_days=self.metrics.age_days)
        
        self.networks[NetworkType.UNCONSCIOUSNESS].update_development(age_days=self.metrics.age_days)
        
        self.networks[NetworkType.DRIVES].update_developmental_stage(self.metrics.development_stage.value)
        
        self.networks[NetworkType.EMOTIONS].update_development(age_days=self.metrics.age_days)
        
        self.networks[NetworkType.MOODS].update_development(age_days=self.metrics.age_days)
        
        self.networks[NetworkType.ATTENTION].update_development(
            age_days=self.metrics.age_days,
            interactions_count=self.metrics.total_interactions
        )
        
        self.networks[NetworkType.PERCEPTION].update_development(
            age_days=self.metrics.age_days, 
            vocabulary_size=self.metrics.vocabulary_size
        )
        
        self.networks[NetworkType.CONSCIOUSNESS].update_development(
            age_days=self.metrics.age_days,
            vocabulary_size=self.metrics.vocabulary_size
        )
        
        self.networks[NetworkType.THOUGHTS].update_development(
            age_days=self.metrics.age_days,
            vocabulary_size=self.metrics.vocabulary_size
        )
    
    def generate_response(self) -> str:
        """Generate a verbal response based on current developmental state and neural activation"""
        # Get language capabilities
        capabilities = self.language_tracker.get_capabilities()
        
        # Get emotional state as dictionary for language production
        emotional_state_dict = {
            emotion: getattr(self.emotional_state, emotion)
            for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
        }
        
        # Get current thoughts that might influence response
        thoughts = []
        if NetworkType.THOUGHTS in self.networks:
            thought_output = self.networks[NetworkType.THOUGHTS]._prepare_output_data()
            thoughts = thought_output.get("thoughts", [])
        
        # Generate response using language production module
        response = self.language_production.generate_utterance(
            capabilities=capabilities,
            emotional_state=emotional_state_dict,
            context=", ".join(thoughts) if thoughts else None,
            response_to=self.recent_interactions[-1]["mother_verbal"] if self.recent_interactions else None
        )
        
        # Extract words used in response for updating statistics
        words_used = response.lower().split()
        self.vocabulary_manager.update_after_child_production(words_used, successful_communication=True)
        
        return response
    
    def get_child_state(self) -> ChildState:
        """Generate the observable state of the child for the mother"""
        # Generate a response
        message = self.generate_response()
        
        # Determine apparent emotion
        dominant_emotion, _ = self.emotional_state.dominant_emotion()
        
        # Get list of recently learned concepts
        recent_words = self.vocabulary_manager.get_vocabulary_statistics().recent_words[:5]
        
        return ChildState(
            message=message,
            apparent_emotion=dominant_emotion,
            vocabulary_size=len(self.vocabulary_manager.lexical_memory.words),
            age_days=self.metrics.age_days,
            recent_concepts_learned=recent_words,
            attention_span=self.metrics.attention_span
        )
    
    def interact_with_mother(self) -> Tuple[str, str]:
        """Complete interaction cycle with mother"""
        # Get current child state
        child_state = self.get_child_state()
        
        # Get mother's response
        mother_response = self.mother.respond_to_child(child_state)
        
        # Process mother's response
        self.process_mother_response(mother_response)
        
        # Return the interaction for display
        return child_state.message, mother_response.verbal.text
    
    def is_ready_for_chat(self) -> bool:
        """Check if the child is developed enough for chat interactions"""
        return self.metrics.age_days >= self.config.chat_readiness_threshold
    
    def get_active_networks(self) -> List[str]:
        """Get a list of currently active neural networks"""
        return [network_type.value for network_type, network in self.networks.items() 
                if network.state.activation > 0.3]
    
    def get_network_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all neural networks for visualization"""
        return {
            network_type.value: network.get_state() 
            for network_type, network in self.networks.items()
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the current state of the neural child"""
        if filepath is None:
            filepath = self.save_dir / f"neural_child_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save vocabulary manager state
        self.vocabulary_manager.save_state()
        
        # Prepare network states for serialization
        network_states = {}
        for network_type, network in self.networks.items():
            network_states[network_type.value] = {
                "network_type": network_type.value,
                "activation": network.state.activation,
                "confidence": network.state.confidence,
                "learning_rate": network.state.learning_rate,
                "training_progress": network.state.training_progress,
                "last_active": network.state.last_active,
                "error_rate": network.state.error_rate
            }
        
        # Prepare state for serialization
        state = {
            "metrics": self.metrics.model_dump(),
            "emotional_state": self.emotional_state.model_dump(),
            "networks": network_states,
            "config": self.config.model_dump(),
            "recent_interactions": self.recent_interactions,
            "language_capabilities": self.language_tracker.get_capabilities().model_dump()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Neural child state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> None:
        """Load the neural child state from a file"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore state using Pydantic validation
        self.metrics = DevelopmentalMetrics.model_validate(state["metrics"])
        self.emotional_state = EmotionalState.model_validate(state["emotional_state"])
        
        # Restore network states
        for network_type_str, network_data in state["networks"].items():
            network_type = NetworkType(network_type_str)
            if network_type in self.networks:
                # Update network state
                self.networks[network_type].state.activation = network_data["activation"]
                self.networks[network_type].state.confidence = network_data["confidence"]
                self.networks[network_type].state.learning_rate = network_data["learning_rate"]
                self.networks[network_type].state.training_progress = network_data["training_progress"]
                self.networks[network_type].state.error_rate = network_data["error_rate"]
        
        # Restore config
        self.config = ChildConfig(**state["config"])
        
        # Restore recent interactions
        self.recent_interactions = state["recent_interactions"]
        
        # Restore language capabilities if present
        if "language_capabilities" in state:
            self.language_tracker = DevelopmentTracker(initial_age_days=self.metrics.age_days)
            self.language_tracker.capabilities = LanguageCapabilities.model_validate(state["language_capabilities"])
        
        # Update network development parameters based on loaded age
        self._update_network_development()
        
        logger.info(f"Neural child state loaded from {filepath}")
        logger.info(f"Age: {self.metrics.age_days:.1f} days, Vocabulary: {self.metrics.vocabulary_size} words")

# ========== EXAMPLE USAGE ==========

def create_test_child() -> NeuralChild:
    """Create a test instance of the neural child"""
    return NeuralChild(simulation_speed=2.0)

if __name__ == "__main__":
    # Simple test to demonstrate usage
    child = create_test_child()
    
    # Interaction cycle
    print("\n=== Beginning Neural Child Interaction ===")
    
    for i in range(5):
        print(f"\n--- Interaction {i+1} ---")
        
        # Complete interaction cycle with mother
        child_message, mother_message = child.interact_with_mother()
        
        # Display the interaction
        print(f"Child age: {child.metrics.age_days:.1f} days")
        print(f"Child: '{child_message}'")
        print(f"Mother: '{mother_message}'")
        print(f"Child emotion: {child.emotional_state.dominant_emotion()[0]}")
        print(f"Vocabulary size: {child.metrics.vocabulary_size}")
        print(f"Active networks: {child.get_active_networks()}")
    
    # Save the final state
    child.save_state()
    print("\nNeural child state saved!")