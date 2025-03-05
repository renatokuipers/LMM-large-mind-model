"""
Configuration settings for the NeuralChild system.
Contains parameters for development rates, networks, memory, and integration settings.
"""
from pathlib import Path
from datetime import datetime
import os

# System paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# General system parameters
SYSTEM = {
    "random_seed": 42,
    "simulation_name": f"neural_child_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "cuda_enabled": True,
    "cuda_device": 0,
    "log_level": "INFO",
    "checkpoint_interval_minutes": 30,
    "enable_dashboard": True,
    "dashboard_port": 8050,
}

# Developmental parameters
DEVELOPMENT = {
    "start_date": datetime.now(),
    "time_acceleration": 720.0,  # 720x acceleration (1 hour = 30 days)
    "development_variability": 0.2,  # Random variation in development rate
    "sensitive_period_factor": 1.5,  # Learning boost during sensitive periods
    "environmental_richness": 0.8,  # Quality of environment (0-1)
    "caregiver_interaction_quality": 0.9,  # Quality of caregiver interactions (0-1)
    "milestone_progression_rate": 0.01,  # Base rate of milestone progress
    "domain_growth_rates": {
        "cognitive": 1.0,
        "language": 1.0,
        "emotional": 1.0,
        "social": 1.0,
        "memory": 1.0,
        "self_awareness": 1.0,
        "moral": 1.0,
    }
}

# Network parameters
NETWORKS = {
    "activation_decay_rate": 0.1,
    "learning_rates": {
        "archetypes": 0.001,  # Very slow-changing
        "instincts": 0.002,
        "unconsciousness": 0.005,
        "drives": 0.01,
        "emotions": 0.03,
        "moods": 0.02,
        "attention": 0.05,
        "perception": 0.05,
        "consciousness": 0.04,
        "thoughts": 0.05,
    },
    "connection_initial_weights": {
        "excitatory": 0.7,
        "inhibitory": -0.7,
        "modulatory": 0.5,
        "feedback": 0.3,
        "associative": 0.6,
    },
    "initial_drives": {
        "physiological": 0.3,
        "safety": 0.4,
        "belonging": 0.5,
        "esteem": 0.1,
        "self_actualization": 0.0,
        "curiosity": 0.6,
    },
    "emotional_intensity_cap": 0.9,
    "emotion_decay_rates": {
        "joy": 0.02,
        "sadness": 0.01,
        "anger": 0.03,
        "fear": 0.02,
        "disgust": 0.03,
        "surprise": 0.05,
        "trust": 0.01,
        "anticipation": 0.02,
    },
    "attention_capacity": 4,  # Max items in focus
    "consciousness_capacity": 7,  # Max items in consciousness
}

# Memory parameters
MEMORY = {
    "working_memory": {
        "capacity": 7,  # Miller's magical number
        "decay_rate": 0.1,
        "max_duration_seconds": 60.0,  # Without rehearsal
        "attention_boost": 0.3,
    },
    "episodic_memory": {
        "consolidation_threshold": 0.6,
        "emotional_weight": 0.7,
        "decay_rate": 0.01,
        "coherence_factor": 0.5,
    },
    "long_term_memory": {
        "consolidation_time_seconds": 3600.0,  # 1 hour real time
        "retrieval_difficulty_factor": 0.3,
        "forgetting_curve_factor": 0.1,
        "semantic_link_strength": 0.6,
        "emotional_persistence_factor": 0.8,
    },
    "consolidation": {
        "working_to_episodic_threshold": 0.5,
        "episodic_to_longterm_threshold": 0.7,
        "consolidation_rate": 0.1,
        "sleep_consolidation_boost": 0.3,
        "emotional_consolidation_factor": 0.4,
        "importance_weight": 0.6,
    },
    "enable_vector_storage": True,
    "embedding_dimensions": 384,
    "similarity_threshold": 0.75,
}

# Language development parameters
LANGUAGE = {
    "vocabulary_growth_rate": 0.05,
    "grammar_learning_rate": 0.02,
    "starting_vocabulary": [],
    "developmental_stages": {
        "prelinguistic": {
            "age_range": (0, 12),  # months
            "utterance_length": 0,
            "grammar_complexity": 0.0,
        },
        "holophrastic": {
            "age_range": (12, 18),
            "utterance_length": 1,
            "grammar_complexity": 0.1,
        },
        "telegraphic": {
            "age_range": (18, 24),
            "utterance_length": 2,
            "grammar_complexity": 0.2,
        },
        "simple_syntax": {
            "age_range": (24, 36),
            "utterance_length": 3,
            "grammar_complexity": 0.4,
        },
        "complex_syntax": {
            "age_range": (36, 60),
            "utterance_length": 5,
            "grammar_complexity": 0.6,
        },
        "advanced": {
            "age_range": (60, 96),
            "utterance_length": 7,
            "grammar_complexity": 0.8,
        },
    },
}

# Mother/caregiver parameters - USES LLM EXCLUSIVELY
MOTHER = {
    # Persona attributes
    "personality_traits": {
        "warmth": 0.85,  # How warm and affectionate the mother is
        "patience": 0.80,  # How patient with child's behavior
        "consistency": 0.75,  # How consistent in responses and rules
        "protectiveness": 0.70,  # How protective vs encouraging independence
        "playfulness": 0.65,  # How playful and fun-oriented
        "structure": 0.60,  # How much structure and routine is emphasized
    },
    
    # Interaction style
    "responsiveness": 0.9,  # How quickly the mother responds
    "sensitivity": 0.8,  # How well the mother interprets the child's signals
    "verbosity": 0.7,  # How much the mother talks to the child
    "elaboration": 0.6,  # How detailed the mother's speech is
    "correction_rate": 0.5,  # How often the mother corrects errors
    "encouragement_rate": 0.8,  # How often the mother gives encouragement
    
    # Teaching approach
    "teaching_style": "scaffolding",  # Options: direct, scaffolding, discovery
    "complexity_adjustment": 0.2,  # How much to adjust language complexity
    "repetition_factor": 3,  # How often concepts are repeated
    "question_frequency": 0.4,  # How often the mother asks questions
    
    # Relationship
    "attachment_security": 0.9,  # How secure the attachment relationship is
    "emotional_attunement": 0.85,  # How well mother attunes to child's emotions
    
    # Observational limitations (mother is not a mind-reader)
    "observable_signals": [
        "facial_expressions",
        "vocalizations",
        "body_language",
        "attention_focus",
        "emotional_state",
        "activity_level",
        "language_production"
    ],
    
    # Response schemas for structured mother outputs
    "response_schemas": {
        "verbal_interaction": {
            "type": "object",
            "properties": {
                "speech": {"type": "string", "description": "What the mother says to the child"},
                "tone": {"type": "string", "enum": ["soothing", "excited", "encouraging", "firm", "questioning", "playful"]},
                "complexity": {"type": "number", "minimum": 0, "maximum": 1, "description": "Language complexity level"},
                "volume": {"type": "number", "minimum": 0, "maximum": 1, "description": "How loudly mother speaks"},
                "repetition": {"type": "boolean", "description": "Whether this repeats a previous statement"}
            },
            "required": ["speech", "tone"]
        },
        "physical_interaction": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Physical action mother takes"},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "duration": {"type": "number", "description": "Duration in seconds"},
                "location": {"type": "string", "description": "Body part or area involved"}
            },
            "required": ["action", "intensity"]
        },
        "emotional_response": {
            "type": "object",
            "properties": {
                "primary_emotion": {"type": "string"},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "facial_expression": {"type": "string"},
                "body_language": {"type": "string"}
            },
            "required": ["primary_emotion", "intensity"]
        },
        "teaching_moment": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "Concept being taught"},
                "method": {"type": "string", "enum": ["explanation", "demonstration", "guided_practice", "correction", "reinforcement"]},
                "speech": {"type": "string", "description": "What the mother says"},
                "simplification_level": {"type": "number", "minimum": 0, "maximum": 1},
                "connection_to_previous": {"type": "string", "description": "How this connects to previously taught concepts"}
            },
            "required": ["concept", "method", "speech"]
        },
        "response_to_distress": {
            "type": "object",
            "properties": {
                "comfort_action": {"type": "string", "description": "What the mother does to comfort"},
                "comfort_speech": {"type": "string", "description": "What the mother says to comfort"},
                "distress_interpretation": {"type": "string", "description": "What mother thinks is causing distress"},
                "solution_offered": {"type": "string", "description": "Solution to the distress"}
            },
            "required": ["comfort_action", "comfort_speech"]
        },
        "boundaries_and_discipline": {
            "type": "object",
            "properties": {
                "rule_stated": {"type": "string", "description": "The rule or boundary being established"},
                "firmness": {"type": "number", "minimum": 0, "maximum": 1},
                "explanation": {"type": "string", "description": "Explanation for the rule"},
                "consequence": {"type": "string", "description": "Consequence if rule is broken"},
                "tone": {"type": "string", "enum": ["gentle", "firm", "stern", "understanding"]}
            },
            "required": ["rule_stated", "firmness", "explanation"]
        }
    }
}

# LLM Integration settings - ONLY FOR MOTHER MODULE
LLM = {
    "enabled": True,
    "base_url": "http://192.168.2.12:1234",
    "default_model": "qwen2.5-7b-instruct",
    "temperature": 0.7,
    "max_tokens": 400,
    "stream": False,
    "system_prompt": """You are a mother caring for and nurturing a developing child. 
You are NOT an AI assistant - you ARE a human mother with natural maternal instincts and behaviors.
Respond as this mother persona would, with appropriate emotional reactions, teaching moments, 
discipline when needed, and nurturing care. You can only observe the child's external behaviors
and signals - you cannot directly read their mind or internal state. Use your maternal intuition
to interpret their needs and development. Your responses should be natural, warm, and authentic
to how a real mother would interact with her child at each developmental stage.""",
    
    # Clarify LLM usage policy
    "usage_policy": {
        "mother_only": True,  # LLM is ONLY for mother module
        "child_never": True,  # NEVER use LLM for child's cognitive processes
    },
}

# Persistence and serialization settings
PERSISTENCE = {
    "save_format": "json",  # Options: json, pickle, both
    "compression": True,
    "use_incremental_save": True,
    "keep_backups": 3,
    "autosave_enabled": True,
    "autosave_interval_minutes": 60,
}

# Metrics and monitoring
METRICS = {
    "track_development_metrics": True,
    "metrics_logging_interval_seconds": 300,  # 5 minutes
    "development_plot_enabled": True,
    "network_activity_tracking": True,
    "memory_usage_tracking": True,
    "create_developmental_reports": True,
    "report_interval_simulated_months": 1,
}

# Load environment-specific overrides if available
try:
    from local_config import override_config
    override_config(globals())
except ImportError:
    pass