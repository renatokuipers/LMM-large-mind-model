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

# Internal imports
from llm_module import LLMClient, Message
from mother import MotherResponse, ChildState, EmotionalState as MotherEmotionalState

# Network imports (these will be implemented later)
# from networks.perception import PerceptionNetwork
# from networks.emotions import EmotionNetwork
# from networks.consciousness import ConsciousnessNetwork
# from networks.attention import AttentionNetwork
# from networks.unconsciousness import UnconsciousnessNetwork
# from networks.instincts import InstinctsNetwork
# from networks.drives import DrivesNetwork
# from networks.archetypes import ArchetypesNetwork
# from networks.thoughts import ThoughtsNetwork
# from networks.moods import MoodsNetwork

# Language and memory imports (these will be implemented later)
from language.developmental_stages import DevelopmentTracker, LanguageDevelopmentStage, LanguageCapabilities
from language.lexical_memory import LexicalMemory
from language.production import LanguageProduction
from language.syntactic_processor import SyntacticProcessor
from language.vocabulary import VocabularyManager
from language.semantic_network import SemanticNetwork, SemanticRelation
# from memory.memory_manager import MemoryManager

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

class NetworkType(str, Enum):
    """Types of neural networks in the child's mind"""
    ARCHETYPES = "archetypes"
    INSTINCTS = "instincts"
    UNCONSCIOUSNESS = "unconsciousness"
    DRIVES = "drives"
    EMOTIONS = "emotions"
    MOODS = "moods"
    ATTENTION = "attention"
    PERCEPTION = "perception"
    CONSCIOUSNESS = "consciousness"
    THOUGHTS = "thoughts"

class NetworkActivationLevel(BaseModel):
    """Activation level and metrics for a neural network"""
    network_type: NetworkType
    activation: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    training_progress: float = Field(0.0, ge=0.0, le=1.0)
    last_active: datetime = Field(default_factory=datetime.now)
    error_rate: float = Field(0.0, ge=0.0)
    
    def update_activation(self, stimulation: float) -> None:
        """Update the activation level based on input stimulation"""
        # Simple activation function with decay
        decay_factor = 0.9  # Previous activation decays by this factor
        self.activation = min(1.0, max(0.0, self.activation * decay_factor + stimulation))
        self.last_active = datetime.now()

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

class VocabularyItem(BaseModel):
    """A word in the child's vocabulary"""
    word: str
    understanding: float = Field(0.1, ge=0.0, le=1.0)
    learned_at: datetime = Field(default_factory=datetime.now)
    contexts: List[str] = Field(default_factory=list)
    usage_count: int = Field(0, ge=0)
    emotional_associations: Dict[str, float] = Field(default_factory=dict)
    
    def update_understanding(self, context: str, emotion: Optional[str] = None, emotion_intensity: float = 0.0) -> None:
        """Update understanding of this word based on new exposure"""
        self.usage_count += 1
        
        # Add context if new
        if context not in self.contexts:
            self.contexts.append(context)
        
        # Update understanding (plateaus as it approaches 1.0)
        learning_rate = 0.05 * (1.0 - self.understanding)
        self.understanding = min(1.0, self.understanding + learning_rate)
        
        # Update emotional association if provided
        if emotion:
            if emotion not in self.emotional_associations:
                self.emotional_associations[emotion] = emotion_intensity
            else:
                # Blend existing and new emotional association
                current = self.emotional_associations[emotion]
                self.emotional_associations[emotion] = (current * 0.8) + (emotion_intensity * 0.2)

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
        simulation_speed: float = 1.0  # Multiplier for development speed
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
        
        # Initialize neural networks
        self.networks = self._initialize_networks()
        
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
        
        logger.info("Neural child initialized")
    
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
    
    def _initialize_networks(self) -> Dict[NetworkType, NetworkActivationLevel]:
        """Initialize all neural networks"""
        networks = {}
        for network_type in NetworkType:
            emphasis = self.config.network_emphasis.get(network_type, 0.1)
            networks[network_type] = NetworkActivationLevel(
                network_type=network_type,
                activation=0.0,
                confidence=0.0,
                learning_rate=0.01 * self.config.learning_rate_multiplier,
                training_progress=0.0,
                error_rate=random.uniform(0.1, 0.3)  # Initial random error rate
            )
        return networks
    
    def process_mother_response(self, mother_response: MotherResponse) -> None:
        """Process a response from the mother, updating internal state"""
        logger.info("Processing mother's response")
        
        # Update emotional state based on mother's response
        self.emotional_state.update_from_mother_response(mother_response)
        
        # Process verbal content - learn vocabulary
        self._process_verbal_content(mother_response.verbal.text, mother_response.teaching)
        
        # Update network activations based on response
        self._update_networks_from_response(mother_response)
        
        # Update developmental metrics
        self._update_metrics(mother_response)
        
        # Add to recent interactions
        self.recent_interactions.append({
            "timestamp": datetime.now().isoformat(),
            "mother_verbal": mother_response.verbal.text,
            "mother_emotion": mother_response.emotional.primary_emotion,
            "child_emotion": self.emotional_state.dominant_emotion()[0]
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
        
        logger.info(f"Updated state - vocabulary size: {len(self.vocabulary_manager.lexical_memory.words)}, "
                  f"emotion: {self.emotional_state.dominant_emotion()[0]}, "
                  f"language stage: {capabilities.stage}")
    
    def _process_verbal_content(self, text: str, teaching_elements) -> None:
        """Extract and learn from verbal content"""
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
    
    def _update_networks_from_response(self, mother_response: MotherResponse) -> None:
        """Update neural network activations based on mother's response"""
        # Perception is always activated when processing input
        self.networks[NetworkType.PERCEPTION].update_activation(0.8)
        
        # Activate emotional networks based on emotional content
        emotion_intensity = mother_response.emotional.intensity
        self.networks[NetworkType.EMOTIONS].update_activation(emotion_intensity * 0.9)
        self.networks[NetworkType.MOODS].update_activation(emotion_intensity * 0.3)
        
        # Activate cognitive networks based on teaching complexity
        teaching_complexity = len(mother_response.teaching.vocabulary) + len(mother_response.teaching.concepts)
        teaching_complexity = min(1.0, teaching_complexity * 0.1)  # Normalize
        
        self.networks[NetworkType.ATTENTION].update_activation(teaching_complexity * 0.7)
        self.networks[NetworkType.CONSCIOUSNESS].update_activation(teaching_complexity * 0.5)
        self.networks[NetworkType.THOUGHTS].update_activation(teaching_complexity * 0.4)
        
        # Subconscious processing is always happening at a low level
        self.networks[NetworkType.UNCONSCIOUSNESS].update_activation(0.2)
        self.networks[NetworkType.ARCHETYPES].update_activation(0.1)
        
        # Drive activation based on mother's approach
        if "nurture" in mother_response.parenting.approach.lower():
            self.networks[NetworkType.DRIVES].update_activation(0.3)
            self.networks[NetworkType.INSTINCTS].update_activation(0.4)
        elif "discipline" in mother_response.parenting.approach.lower():
            self.networks[NetworkType.DRIVES].update_activation(0.6)
            self.networks[NetworkType.INSTINCTS].update_activation(0.7)
        else:
            self.networks[NetworkType.DRIVES].update_activation(0.2)
            self.networks[NetworkType.INSTINCTS].update_activation(0.3)
        
        # Update training progress for all networks
        for network in self.networks.values():
            if network.activation > 0.1:  # Only networks with meaningful activation learn
                progress_increment = network.activation * network.learning_rate
                network.training_progress = min(1.0, network.training_progress + progress_increment)
                
                # As networks train more, error rate decreases
                network.error_rate = max(0.01, network.error_rate * 0.999)
                
                # Confidence increases with training progress
                network.confidence = min(1.0, network.training_progress * 0.8 + 0.1)
    
    def _update_metrics(self, mother_response: MotherResponse) -> None:
        """Update developmental metrics based on interaction"""
        # Update interaction count
        self.metrics.total_interactions += 1
        
        # Update vocabulary size
        self.metrics.vocabulary_size = len(self.vocabulary)
        
        # Update age (accelerated for simulation)
        time_increment = 0.1 * self.config.simulated_time_ratio * self.simulation_speed
        self.metrics.age_days += time_increment
        
        # Update training time (in hours)
        self.metrics.total_training_time += (time_increment / 24.0)
        
        # Update attention span based on interaction quality
        attention_increment = 0.001 * mother_response.parenting.adaptation_to_development
        self.metrics.attention_span = min(1.0, self.metrics.attention_span + attention_increment)
        
        # Update emotional stability
        if "regulate" in mother_response.parenting.intention.lower():
            stability_increment = 0.002
            self.metrics.emotional_stability = min(1.0, self.metrics.emotional_stability + stability_increment)
        
        # Update social awareness if relevant teaching
        social_values = ["sharing", "empathy", "cooperation", "friendship"]
        for value in mother_response.teaching.values:
            if any(sv in value.lower() for sv in social_values):
                social_increment = 0.003
                self.metrics.social_awareness = min(1.0, self.metrics.social_awareness + social_increment)
                break
        
        # Update abstraction capability based on concept teaching
        if mother_response.teaching.concepts:
            abstraction_increment = 0.002 * len(mother_response.teaching.concepts)
            self.metrics.abstraction_capability = min(1.0, self.metrics.abstraction_capability + abstraction_increment)
        
        # Validate development stage against age
        self.metrics = self.metrics.model_validate(self.metrics.model_dump())
    
    def generate_response(self) -> str:
        """Generate a verbal response based on current developmental state"""
        # Get language capabilities
        capabilities = self.language_tracker.get_capabilities()
        
        # Get emotional state as dictionary for language production
        emotional_state_dict = {
            emotion: getattr(self.emotional_state, emotion)
            for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
        }
        
        # Generate response using language production module
        response = self.language_production.generate_utterance(
            capabilities=capabilities,
            emotional_state=emotional_state_dict,
            context=None,  # Could use context from recent interactions
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
    
    def is_ready_for_chat(self) -> bool:
        """Check if the child is developed enough for chat interactions"""
        return self.metrics.age_days >= self.config.chat_readiness_threshold
    
    def get_active_networks(self) -> List[str]:
        """Get a list of currently active neural networks"""
        return [str(network_type.value) for network_type, network in self.networks.items() 
                if network.activation > 0.3]
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the current state of the neural child"""
        if filepath is None:
            filepath = self.save_dir / f"neural_child_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save vocabulary manager state
        self.vocabulary_manager.save_state()
        
        # Prepare state for serialization
        state = {
            "metrics": self.metrics.model_dump(),
            "emotional_state": self.emotional_state.model_dump(),
            "networks": {str(network_type.value): network.model_dump() 
                        for network_type, network in self.networks.items()},
            "config": self.config.model_dump(),
            "recent_interactions": self.recent_interactions,
            "language_capabilities": self.language_tracker.get_capabilities().model_dump()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, cls=DateTimeEncoder)  # Use our custom encoder here
        
        logger.info(f"Neural child state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> None:
        """Load the neural child state from a file"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore state using Pydantic validation (handles datetime strings automatically)
        self.metrics = DevelopmentalMetrics.model_validate(state["metrics"])
        self.emotional_state = EmotionalState.model_validate(state["emotional_state"])
        
        # Restore networks
        for network_type_str, network_data in state["networks"].items():
            network_type = NetworkType(network_type_str)
            self.networks[network_type] = NetworkActivationLevel(**network_data)
        
        # Restore config
        self.config = ChildConfig(**state["config"])
        
        # Restore recent interactions
        self.recent_interactions = state["recent_interactions"]
        
        # Restore language capabilities if present
        if "language_capabilities" in state:
            self.language_tracker = DevelopmentTracker(initial_age_days=self.metrics.age_days)
            self.language_tracker.capabilities = LanguageCapabilities.model_validate(state["language_capabilities"])
        
        # Note: Vocabulary state would be loaded separately
        
        logger.info(f"Neural child state loaded from {filepath}")
        logger.info(f"Age: {self.metrics.age_days:.1f} days, Vocabulary: {len(self.vocabulary_manager.lexical_memory.words)} words")

# ========== EXAMPLE USAGE ==========

def create_test_child() -> NeuralChild:
    """Create a test instance of the neural child"""
    return NeuralChild(simulation_speed=2.0)  # Faster development for testing

if __name__ == "__main__":
    # Simple test to demonstrate usage
    child = create_test_child()
    
    # Example of simple "development" through 5 interactions
    for i in range(5):
        print(f"\n--- Interaction {i+1} ---")
        
        # Create a simple mock mother response
        mock_response = MotherResponse(
            verbal={
                "text": "Hello my little one! Can you say 'ball'? Ball is round and fun to play with.",
                "tone": "gentle",
                "complexity_level": 0.3
            },
            emotional={
                "primary_emotion": "joy",
                "intensity": 0.8,
                "secondary_emotion": "love",
                "patience_level": 0.9
            },
            non_verbal={
                "physical_actions": ["smiles", "points to a ball"],
                "facial_expression": "warm smile",
                "proximity": "close"
            },
            teaching={
                "vocabulary": [
                    {
                        "word": "ball",
                        "simple_definition": "a round toy you can throw and catch",
                        "example_usage": "The ball is red."
                    }
                ],
                "concepts": [],
                "values": ["play", "learning"],
                "corrections": []
            },
            perception={
                "child_emotion": "curiosity",
                "child_needs": ["attention", "learning"],
                "misinterpretations": None
            },
            parenting={
                "intention": "teach new vocabulary",
                "approach": "nurturing guidance",
                "consistency": 0.9,
                "adaptation_to_development": 0.85
            },
            context_awareness={
                "references_previous_interactions": None,
                "environment_factors": ["playtime"],
                "recognizes_progress": True
            }
        )
        
        # Process the mother's response
        child.process_mother_response(mock_response)
        
        # Generate and print the child's response
        child_state = child.get_child_state()
        print(f"Child age: {child_state.age_days:.1f} days")
        print(f"Child emotional state: {child.emotional_state.dominant_emotion()[0]}")
        print(f"Child vocabulary size: {len(child.vocabulary)}")
        print(f"Child response: '{child_state.message}'")
        print(f"Active networks: {child.get_active_networks()}")
    
    # Save the state
    child.save_state()