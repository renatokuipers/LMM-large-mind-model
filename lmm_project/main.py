#!/usr/bin/env python3
"""
Main entry point for the Large Mind Model (LMM) project.

This file implements the full integration of all major components:
- Mind (core cognitive architecture)
- Mother (nurturing LLM interface)
- Neural Substrate (biologically-inspired neural components)
- Learning Engines (reinforcement, hebbian, pruning, consolidation)
- Cognitive Modules (perception, attention, memory, etc.)

This implementation creates an authentic developmental journey where the Mother
nurtures and teaches the developing mind, which learns from zero with no
pre-programmed knowledge, gradually progressing through developmental stages.
"""

import os
import time
import logging
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta
import torch
import sqlite3

# Core components
from lmm_project.core.mind import Mind
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.state_manager import StateManager

# Neural substrate components
from lmm_project.neural_substrate import NeuralNetwork, NeuralCluster
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.neural_substrate.hebbian_learning import HebbianLearning

# Cognitive modules imports - updated for new perception system
from lmm_project.modules.perception import get_module as get_perception_module
from lmm_project.modules.perception.models import PerceptionResult

# Mother interface components
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.interfaces.mother.personality import PersonalityManager, PersonalityProfile, EmotionalValence
from lmm_project.interfaces.mother.teaching_strategies import TeachingStrategyManager, LearningGoalCategory, ComprehensionLevel
from lmm_project.interfaces.mother.interaction_patterns import InteractionPatternManager

# Learning engines
from lmm_project.learning_engines.models import LearningEngine

# Development components
from lmm_project.development.developmental_stages import DevelopmentalStageManager

# Utility components
from lmm_project.utils.llm_client import LLMClient, Message as LLMMessage
from lmm_project.utils.tts_client import TTSClient, GenerateAudioRequest
from lmm_project.utils.audio_player import play_audio_file
from lmm_project.utils.vector_store import VectorStore
from lmm_project.utils.visualization import visualize_development

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set higher log levels for noisy modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_device():
    """Get the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initialize_mind(event_bus: EventBus) -> Mind:
    """
    Initialize the Mind component with all cognitive modules
    
    Args:
        event_bus: Shared event bus for inter-module communication
        
    Returns:
        Initialized Mind instance
    """
    logger.info("Initializing Mind...")
    
    # Create state manager
    state_manager = StateManager()
    
    # Create mind with event bus and state manager
    mind = Mind(
        event_bus=event_bus,
        state_manager=state_manager
    )
    
    # Initialize all cognitive modules
    mind.initialize_modules()
    
    # Update the mind with the new perception module
    if "perception" in mind.modules:
        logger.info("Replacing default perception module with new implementation")
        perception_module = get_perception_module(
            module_id="perception",
            event_bus=event_bus
        )
        mind.modules["perception"] = perception_module
    
    logger.info(f"Mind initialized with modules: {list(mind.modules.keys())}")
    return mind


def initialize_mother(personality_profile: str = "nurturing") -> MotherLLM:
    """
    Initialize the Mother LLM interface
    
    Args:
        personality_profile: Personality profile to use for the Mother
        
    Returns:
        Initialized MotherLLM instance
    """
    logger.info(f"Initializing Mother LLM with {personality_profile} personality profile...")
    
    # Initialize LLM client
    llm_client = LLMClient(base_url="http://192.168.2.12:1234")
    
    # Initialize TTS client
    tts_client = TTSClient()
    
    # Create personality manager
    personality_manager = PersonalityManager(profile=personality_profile)
    
    # Create teaching strategy manager
    teaching_style = personality_manager._derive_teaching_style().value
    teaching_strategy_manager = TeachingStrategyManager(default_style=teaching_style)
    
    # Create interaction pattern manager
    interaction_pattern_manager = InteractionPatternManager()
    
    # Create Mother LLM interface
    mother = MotherLLM(
        llm_client=llm_client,
        tts_client=tts_client,
        teaching_style=teaching_style,
        voice="af_bella",
        personality_manager=personality_manager,
        teaching_strategy_manager=teaching_strategy_manager,
        interaction_pattern_manager=interaction_pattern_manager
    )
    
    logger.info("Mother LLM initialized")
    return mother


def initialize_neural_substrate(event_bus: EventBus) -> Dict[str, Any]:
    """
    Initialize the neural substrate components
    
    Args:
        event_bus: Shared event bus for neural events
        
    Returns:
        Dictionary of neural substrate components
    """
    logger.info("Initializing Neural Substrate...")
    logger.info(f"Using device: {get_device()}")
    
    # Create minimal neural network
    neural_network = NeuralNetwork(name="main_network")
    
    # Create input neurons (sensory inputs)
    for i in range(16):
        neuron = neural_network.create_neuron(
            activation_function="relu",
            activation_threshold=0.2  # Lower threshold from 0.5 to 0.2
        )
        neural_network.input_neurons.append(neuron.neuron_id)
    
    # Create hidden layer 1 (features)
    hidden_layer1 = []
    for i in range(12):
        neuron = neural_network.create_neuron(
            activation_function="relu",
            activation_threshold=0.2  # Lower threshold from 0.4 to 0.2
        )
        hidden_layer1.append(neuron.neuron_id)
    
    # Create hidden layer 2 (integration)
    hidden_layer2 = []
    for i in range(8):
        neuron = neural_network.create_neuron(
            activation_function="tanh",
            activation_threshold=0.2  # Lower threshold from 0.4 to 0.2
        )
        hidden_layer2.append(neuron.neuron_id)
    
    # Create output neurons (responses)
    for i in range(8):
        neuron = neural_network.create_neuron(
            activation_function="sigmoid",
            activation_threshold=0.3  # Lower threshold from 0.5 to 0.3
        )
        neural_network.output_neurons.append(neuron.neuron_id)
    
    # Connect input to hidden layer 1 (use stronger initial connections)
    for input_id in neural_network.input_neurons:
        for hidden_id in hidden_layer1:
            # Use stronger random weights
            weight = random.uniform(0.1, 0.3)  # Increased from (0.01, 0.1)
            neural_network.create_synapse(
                source_id=input_id,
                target_id=hidden_id,
                weight=weight
            )
    
    # Connect hidden layer 1 to hidden layer 2
    for hidden1_id in hidden_layer1:
        for hidden2_id in hidden_layer2:
            # Use stronger random weights
            weight = random.uniform(0.1, 0.3)  # Increased from (0.01, 0.1)
            neural_network.create_synapse(
                source_id=hidden1_id,
                target_id=hidden2_id,
                weight=weight
            )
    
    # Connect hidden layer 2 to output
    for hidden2_id in hidden_layer2:
        for output_id in neural_network.output_neurons:
            # Use stronger random weights
            weight = random.uniform(0.1, 0.3)  # Increased from (0.01, 0.1)
            neural_network.create_synapse(
                source_id=hidden2_id,
                target_id=output_id,
                weight=weight
            )
    
    # Create neural clusters for specific cognitive functions
    perception_cluster = NeuralCluster(name="perception_cluster")
    attention_cluster = NeuralCluster(name="attention_cluster")
    memory_cluster = NeuralCluster(name="memory_cluster")
    language_cluster = NeuralCluster(name="language_cluster")
    
    logger.info("Neural Substrate initialized")
    return {
        "neural_network": neural_network,
        "perception_cluster": perception_cluster,
        "attention_cluster": attention_cluster,
        "memory_cluster": memory_cluster,
        "language_cluster": language_cluster
    }


def initialize_learning_engines(event_bus: EventBus) -> Dict[str, Any]:
    """
    Initialize the learning engines
    
    This implementation uses a simplified approach to avoid compatibility issues
    with the existing implementation.
    
    Args:
        event_bus: Shared event bus for learning events
        
    Returns:
        Dictionary of learning engines
    """
    logger.info("Initializing Learning Engines (Simplified)...")
    
    # For now, we'll create a minimal placeholder for learning functions
    learning_engines = {
        "event_bus": event_bus,
        "hebbian_learning_enabled": True,
        "reinforcement_learning_enabled": True,
        "pruning_enabled": True,
        "consolidation_enabled": True
    }
    
    logger.info("Simplified learning mechanisms initialized")
    return learning_engines


def generate_lmm_thought(
    mind: Mind, 
    context: Dict[str, Any]
) -> str:
    """
    Generate a thought/statement from the LMM based on its current language capabilities
    
    Args:
        mind: The Mind instance
        context: The current context including previous interactions
        
    Returns:
        A string representing the LMM's current thought or statement
    """
    # Get the current state from language module
    language_module = mind.modules.get("language")
    
    # For pre-verbal stages, generate non-verbal or very simple responses
    if mind.developmental_stage == "prenatal":
        # Pre-verbal stage - just express basic sensory response
        return "..." if random.random() > 0.5 else ""  # Often no response or just pause
    
    elif mind.developmental_stage == "infant" and mind.age < 0.5:
        # Early infant stage - babbling and basic sounds
        sounds = ["ah", "ba", "ma", "ga", "da", "oh"]
        # Sometimes repeat sounds, sometimes single sounds
        return random.choice(sounds) * random.randint(1, 3) if random.random() > 0.3 else ""
    
    # For later stages, use the language model's capabilities
    # This is a simplified approach - a real implementation would use the actual
    # language capabilities that have developed through learning
    if language_module:
        # Generate thought based on language model state
        language_state = language_module.get_state()
        
        # Get vocabulary known to the model
        known_words = language_state.get("known_words", [])
        sentence_patterns = language_state.get("sentence_patterns", [])
        
        if not known_words:  # If no words are known yet
            # Still early - use primitive sounds or very basic words if any
            basic_sounds = ["ah", "oh", "eh"]
            return random.choice(basic_sounds)
        
        elif len(known_words) < 5:  # Very limited vocabulary 
            # Just use single words from what's been learned
            return random.choice(known_words)
        
        elif len(known_words) < 20:  # Growing vocabulary but limited structure
            # Simple two-word combinations
            if random.random() > 0.5 and len(known_words) >= 2:
                return f"{random.choice(known_words)} {random.choice(known_words)}"
            else:
                return random.choice(known_words)
        
        else:  # More developed language capabilities
            # If sentence patterns have been learned, use those
            if sentence_patterns:
                pattern = random.choice(sentence_patterns)
                # Fill in the pattern with known words
                # This is simplified - a real implementation would use more sophisticated
                # language generation based on the actual learned patterns
                words_needed = pattern.count("{}") 
                if words_needed > 0:
                    chosen_words = random.sample(known_words, min(words_needed, len(known_words)))
                    # Fill in the pattern with the words
                    return pattern.format(*chosen_words)
            
            # Fallback - just create a simple sentence from known words
            num_words = min(random.randint(2, 5), len(known_words))
            return " ".join(random.sample(known_words, num_words))
    
    # Fallback if language module isn't available
    return "..." 


def initialize_database() -> sqlite3.Connection:
    """
    Initialize the SQLite database for storing interactions and states
    
    Returns:
        SQLite connection object
    """
    # Create storage directory if it doesn't exist
    storage_dir = os.path.join("lmm_project", "storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    # Connect to SQLite database
    db_path = os.path.join(storage_dir, "lmm_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY,
        session_id TEXT,
        timestamp TEXT,
        lmm_input TEXT,
        mother_response TEXT,
        audio_path TEXT,
        developmental_stage TEXT,
        age REAL,
        development_increment REAL,
        perception_stats TEXT,
        interaction_details TEXT
    )
    ''')
    
    # Create states table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mind_states (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        session_id TEXT,
        age REAL,
        developmental_stage TEXT,
        module_states TEXT,
        emotional_state TEXT
    )
    ''')
    
    # Create sessions table to group related interactions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        start_time TEXT,
        end_time TEXT,
        initial_age REAL,
        final_age REAL,
        interactions_count INTEGER,
        description TEXT
    )
    ''')
    
    conn.commit()
    return conn


def save_interaction_to_db(
    conn: sqlite3.Connection,
    result: Dict[str, Any], 
    interaction_id: int,
    session_id: str
) -> None:
    """
    Save interaction details to the database
    
    Args:
        conn: SQLite connection
        result: The interaction result
        interaction_id: The ID of the interaction
        session_id: Current session ID
    """
    cursor = conn.cursor()
    
    # Get perception information if available
    perception_stats = {}
    if "perception" in result and "perception_result" in result["perception"]:
        pr = result["perception"]["perception_result"]
        perception_stats = {
            "pattern_count": len(pr.get("detected_patterns", [])),
            "novelty_score": pr.get("novelty_score"),
            "intensity_score": pr.get("intensity_score")
        }
    
    # Insert into database
    cursor.execute('''
    INSERT INTO interactions (
        session_id, timestamp, lmm_input, mother_response, audio_path,
        developmental_stage, age, development_increment, 
        perception_stats, interaction_details
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        datetime.now().isoformat(),
        result.get("lmm_input", ""),
        result.get("mother_response", {}).get("text", ""),
        result.get("mother_response", {}).get("audio_path", ""),
        result.get("mind_state", {}).get("developmental_stage", ""),
        result.get("mind_state", {}).get("age", 0),
        result.get("development_increment", 0),
        json.dumps(perception_stats),
        json.dumps(result.get("mother_response", {}).get("interaction_details", {}))
    ))
    
    conn.commit()


def save_state_to_db(
    conn: sqlite3.Connection,
    mind: Mind,
    session_id: str,
    emotional_state: str = "neutral"
) -> None:
    """
    Save mind state to the database
    
    Args:
        conn: SQLite connection
        mind: The Mind instance
        session_id: Current session ID
        emotional_state: Current emotional state
    """
    cursor = conn.cursor()
    
    # Gather module states
    module_states = {
        module_type: {
            "development_level": module.development_level,
            # Additional module-specific state information could be included here
        } for module_type, module in mind.modules.items()
    }
    
    # Insert into database
    cursor.execute('''
    INSERT INTO mind_states (
        timestamp, session_id, age, developmental_stage, 
        module_states, emotional_state
    ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        session_id,
        mind.age,
        mind.developmental_stage,
        json.dumps(module_states),
        emotional_state
    ))
    
    conn.commit()


def update_session_info(
    conn: sqlite3.Connection,
    session_id: str,
    mind: Mind,
    is_finished: bool = False,
    interaction_count: int = 0
) -> None:
    """
    Create or update session information
    
    Args:
        conn: SQLite connection
        session_id: Session identifier
        mind: Mind instance
        is_finished: Whether the session is finished
        interaction_count: Number of interactions in the session
    """
    cursor = conn.cursor()
    
    # Check if session exists
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    
    current_time = datetime.now().isoformat()
    
    if not session:
        # Create new session
        cursor.execute('''
        INSERT INTO sessions (
            id, start_time, end_time, initial_age, final_age, 
            interactions_count, description
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            current_time,
            current_time if is_finished else None,
            mind.age,
            mind.age,
            interaction_count,
            f"Development session from {mind.developmental_stage} stage"
        ))
    else:
        # Update existing session
        cursor.execute('''
        UPDATE sessions SET
            end_time = ?,
            final_age = ?,
            interactions_count = ?
        WHERE id = ?
        ''', (
            current_time if is_finished else None,
            mind.age,
            interaction_count,
            session_id
        ))
    
    conn.commit()


def autonomous_interaction_session(
    mind: Mind, 
    mother: MotherLLM, 
    neural_substrate: Dict[str, Any],
    learning_engines: Dict[str, Any],
    state_manager: StateManager,
    db_conn: sqlite3.Connection,
    max_interactions: int = 100,
    interaction_delay: float = 5.0
) -> None:
    """
    Run an autonomous interaction session between the LMM and Mother
    
    Args:
        mind: The Mind instance
        mother: The Mother LLM interface
        neural_substrate: Dictionary of neural substrate components
        learning_engines: Dictionary of learning engines
        state_manager: State manager for tracking system state
        db_conn: Database connection
        max_interactions: Maximum number of interactions to perform
        interaction_delay: Delay between interactions in seconds
    """
    # Generate session ID using timestamp
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting autonomous interaction session with ID: {session_id}")
    print(f"\n===== LMM Autonomous Development Session ({session_id}) =====")
    print("Press Ctrl+C to end the session\n")
    
    # Create development visualization directory
    viz_dir = os.path.join("lmm_project", "visualization", "output")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize session in database
    update_session_info(db_conn, session_id, mind)
    
    interaction_count = 0
    context = {"previous_interactions": []}
    
    # Variables for tracking development progress
    development_history = []
    module_development_history = {module_type: [] for module_type in mind.modules.keys()}
    
    try:
        while interaction_count < max_interactions:
            interaction_count += 1
            print(f"\n--- Interaction {interaction_count} ---")
            
            # Record development state before interaction
            development_history.append({
                "interaction": interaction_count,
                "age": mind.age,
                "stage": mind.developmental_stage,
                "timestamp": datetime.now().isoformat()
            })
            
            # Record module development
            for module_type, module in mind.modules.items():
                module_development_history[module_type].append({
                    "interaction": interaction_count,
                    "development_level": module.development_level,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Generate a thought from the LMM based on its current state
            lmm_thought = generate_lmm_thought(mind, context)
            print(f"LMM: {lmm_thought}")
            
            # Process the interaction
            result = process_interaction(
                mind=mind,
                mother=mother,
                neural_substrate=neural_substrate,
                learning_engines=learning_engines,
                lmm_input=lmm_thought,
                state_manager=state_manager
            )
            
            # Display the Mother's response
            mother_response = result["mother_response"]["text"]
            print(f"Mother: {mother_response}")
            
            # Display perception details if available
            if "perception" in result and "perception_result" in result["perception"]:
                pr = result["perception"]["perception_result"]
                print(f"\nPerception details:")
                print(f"- Detected patterns: {len(pr.get('detected_patterns', []))}")
                print(f"- Novelty score: {pr.get('novelty_score', 'N/A')}")
                print(f"- Intensity score: {pr.get('intensity_score', 'N/A')}")
                
                # Show some pattern information if available
                patterns = pr.get("detected_patterns", [])
                if patterns:
                    print("\nTop patterns:")
                    for i, pattern in enumerate(patterns[:3]):  # Show up to 3 patterns
                        print(f"- Pattern {i+1}: Type={pattern.get('pattern_type')}, "
                              f"Activation={pattern.get('activation', 0):.2f}")
            
            # Update context with this interaction
            context["previous_interactions"].append({
                "lmm": lmm_thought,
                "mother": mother_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save interaction to database
            save_interaction_to_db(db_conn, result, interaction_count, session_id)
            
            # Save state periodically (every 5 interactions)
            if interaction_count % 5 == 0:
                emotional_state = state_manager.get_state("emotional_state") or "neutral"
                save_state_to_db(db_conn, mind, session_id, emotional_state)
                
                # Update session info
                update_session_info(db_conn, session_id, mind, False, interaction_count)
            
            # Update and display developmental progress
            print(f"\nDevelopmental Stage: {mind.developmental_stage}")
            print(f"Age: {mind.age:.2f}")
            
            # Display learning details if available
            if "interaction_details" in result["mother_response"]:
                details = result["mother_response"]["interaction_details"]
                print(f"Learning Goal: {details.get('learning_goal', 'Unknown')}")
                print(f"Comprehension: {details.get('comprehension_level', 'Unknown')}")
                print(f"Pattern Used: {details.get('pattern_used', 'Unknown')}")
            
            # Periodically visualize development progress (every 10 interactions)
            if interaction_count % 10 == 0:
                visualize_development(
                    development_history=development_history,
                    module_history=module_development_history,
                    output_path=os.path.join(viz_dir, f"development_{session_id}_{interaction_count}.png")
                )
            
            # Check for stage transitions
            current_stage = mind.developmental_stage
            
            # Initialize developmental stage manager if not already in state_manager
            if not state_manager.get_state("developmental_stage_manager"):
                state_manager.update_state({"developmental_stage_manager": DevelopmentalStageManager()})
            
            # Get the stage manager from state
            dev_stage_manager = state_manager.get_state("developmental_stage_manager")
            
            # Get stage for current age
            updated_stage = dev_stage_manager.get_stage_by_age(mind.age)
            
            if updated_stage != current_stage:
                # Stage transition occurred
                mind.developmental_stage = updated_stage
                print(f"\n**** DEVELOPMENTAL MILESTONE REACHED ****")
                print(f"**** Transitioning from {current_stage} to {updated_stage} stage ****")
                
                # Adjust Mother's approach for new stage
                state_manager.update_state({"stage_transition": {
                    "from": current_stage,
                    "to": updated_stage,
                    "age": mind.age,
                    "interaction": interaction_count
                }})
                
                # Save state at milestone
                emotional_state = state_manager.get_state("emotional_state") or "neutral"
                save_state_to_db(db_conn, mind, session_id, emotional_state)
            
            # Wait before next interaction
            time.sleep(interaction_delay)
            
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        logger.error(f"Error in autonomous session: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")
    
    # Final visualization
    visualize_development(
        development_history=development_history,
        module_history=module_development_history,
        output_path=os.path.join(viz_dir, f"development_{session_id}_final.png")
    )
    
    # Save final state
    emotional_state = state_manager.get_state("emotional_state") or "neutral"
    save_state_to_db(db_conn, mind, session_id, emotional_state)
    
    # Mark session as complete
    update_session_info(db_conn, session_id, mind, True, interaction_count)
    
    print("\nSession ended. All data saved to database.")


def process_interaction(
    mind: Mind,
    mother: MotherLLM,
    neural_substrate: Dict[str, Any],
    learning_engines: Dict[str, Any],
    lmm_input: str,
    state_manager: StateManager
) -> Dict[str, Any]:
    """
    Process an interaction between the LMM and Mother
    
    Args:
        mind: The Mind instance
        mother: The Mother LLM interface
        neural_substrate: Dictionary of neural substrate components
        learning_engines: Dictionary of learning engines
        lmm_input: The LMM's thought or statement
        state_manager: State manager for tracking system state
        
    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing LMM input: '{lmm_input}'")
    
    interaction_start_time = time.time()
    
    # Keep track of activated neurons for learning
    activated_neurons = []
    
    # Process through perception module first
    perception_result = {}
    if "perception" in mind.modules:
        perception_input = {
            "text": lmm_input,
            "source": "self",
            "timestamp": time.time(),
            "context": {"stage": mind.developmental_stage}
        }
        perception_result = mind.modules["perception"].process_input(perception_input)
        logger.info(f"Perception processed input with result status: {perception_result.get('status', 'unknown')}")
        
        # Extract patterns and features if available
        patterns = []
        features = []
        if "perception_result" in perception_result:
            pr = perception_result["perception_result"]
            patterns = pr.get("detected_patterns", [])
            features = pr.get("feature_vector", [])
            
            # Log perception details
            logger.debug(f"Detected {len(patterns)} patterns with novelty score: {pr.get('novelty_score', 'N/A')}")
    
    # Process through attention module
    attention_result = {}
    if "attention" in mind.modules:
        attention_input = {
            "text": lmm_input,
            "source": "self",
            "timestamp": time.time(),
            "perception_data": perception_result
        }
        attention_result = mind.modules["attention"].process_input(attention_input)
    
    # Process through memory module
    memory_result = {}
    if "memory" in mind.modules:
        memory_input = {
            "text": lmm_input,
            "attention_data": attention_result,
            "perception_data": perception_result,
            "source": "self",
            "timestamp": time.time()
        }
        memory_result = mind.modules["memory"].process_input(memory_input)
    
    # 3. Get current mind state
    mind_state = {
        "developmental_stage": mind.developmental_stage,
        "age": mind.age,
        "modules": {module_type: module.get_state() 
                   for module_type, module in mind.modules.items()},
        "emotional_state": state_manager.get_state("emotional_state") or "neutral",
        "concept_comprehension": state_manager.get_state("concept_comprehension") or {},
    }
    
    # 4. Generate response from Mother
    mother_response = mother.generate_response(
        input_text=lmm_input,
        mind_state=mind_state
    )
    
    # 5. Play audio if available
    if mother_response.get("audio_path"):
        audio_path = mother_response.get("audio_path")
        logger.info(f"Playing audio response from: {audio_path}")
        try:
            play_audio_file(audio_path)
        except Exception as e:
            logger.warning(f"Failed to play audio: {e}")
    
    # 6. Process Mother's response through modules
    mother_text = mother_response.get("text", "")
    
    # Process through perception first
    mother_perception_result = {}
    if "perception" in mind.modules:
        mother_perception_input = {
            "text": mother_text,
            "source": "mother",
            "timestamp": time.time(),
            "context": {"stage": mind.developmental_stage}
        }
        mother_perception_result = mind.modules["perception"].process_input(mother_perception_input)
    
    # Process through attention
    attention_result_mother = {}
    if "attention" in mind.modules:
        attention_input = {
            "text": mother_text,
            "source": "mother",
            "timestamp": time.time(),
            "perception_data": mother_perception_result
        }
        attention_result_mother = mind.modules["attention"].process_input(attention_input)
    
    # Process through memory
    memory_result_mother = {}
    if "memory" in mind.modules:
        memory_input = {
            "text": mother_text,
            "attention_data": attention_result_mother,
            "perception_data": mother_perception_result,
            "source": "mother",
            "timestamp": time.time()
        }
        memory_result_mother = mind.modules["memory"].process_input(memory_input)
    
    # Process through language module if available
    language_result = {}
    if "language" in mind.modules:
        language_input = {
            "text": mother_text,
            "attention_data": attention_result_mother,
            "memory_data": memory_result_mother,
            "perception_data": mother_perception_result,
            "source": "mother",
            "timestamp": time.time()
        }
        language_result = mind.modules["language"].process_input(language_input)
    
    # 7. Apply simplified learning
    
    # Extract features for learning
    features_self = []
    if perception_result.get("perception_result", {}).get("feature_vector"):
        features_self = perception_result["perception_result"]["feature_vector"]
        
    features_mother = []
    if mother_perception_result.get("perception_result", {}).get("feature_vector"):
        features_mother = mother_perception_result["perception_result"]["feature_vector"]
    
    # Apply basic learning to strengthen relevant connections
    try:
        if learning_engines.get("hebbian_learning_enabled", False):
            # Process basic Hebbian learning directly with the neural network
            neural_network = neural_substrate["neural_network"]
            
            # Use the activated_neurons list that was calculated during neural network activation
            # If no neurons are activated, try to find some based on activation threshold
            if not activated_neurons:
                activation_threshold = 0.3  # Use same threshold as above
                for neuron_id, neuron in neural_network.neurons.items():
                    if neuron.activation > activation_threshold:
                        activated_neurons.append(neuron_id)
            
            # Strengthen connections between co-activated neurons
            strengthened_synapses = 0
            if len(activated_neurons) > 1:
                for i in range(len(activated_neurons)):
                    for j in range(i+1, len(activated_neurons)):
                        # Find if a synapse exists between these neurons
                        for synapse_id, synapse in neural_network.synapses.items():
                            if ((synapse.source_id == activated_neurons[i] and synapse.target_id == activated_neurons[j]) or
                                (synapse.source_id == activated_neurons[j] and synapse.target_id == activated_neurons[i])):
                                # Strengthen this connection
                                new_weight = min(1.0, synapse.weight + 0.01)
                                synapse.weight = new_weight
                                strengthened_synapses += 1
            
            logger.info(f"Applied simplified Hebbian learning to {len(activated_neurons)} neurons, strengthened {strengthened_synapses} synapses")
    except Exception as e:
        logger.warning(f"Simplified learning failed: {str(e)}")
    
    # Apply simple reinforcement based on emotional response
    try:
        if learning_engines.get("reinforcement_learning_enabled", False):
            # Extract emotional valence from mother's response
            interaction_details = mother_response.get("interaction_details", {})
            emotional_valence = interaction_details.get("emotional_valence", "neutral")
            
            # Map emotional valence to reward value
            reward_map = {
                "very_positive": 1.0,
                "positive": 0.7,
                "neutral": 0.2,
                "concerned": -0.2,
                "firm": -0.4
            }
            
            # Default to a small positive reward to encourage learning
            reward = reward_map.get(str(emotional_valence), 0.2)
            
            # Apply a simple reinforcement effect to the neural network
            strengthened_connections = 0
            if reward > 0:
                # Strengthen recently active connections
                neural_network = neural_substrate["neural_network"]
                for synapse_id, synapse in neural_network.synapses.items():
                    source_neuron = neural_network.neurons.get(synapse.source_id)
                    target_neuron = neural_network.neurons.get(synapse.target_id)
                    
                    if source_neuron and target_neuron and source_neuron.activation > 0.2 and target_neuron.activation > 0.2:
                        # This connection contributed to a positively reinforced outcome - use lower threshold (0.2)
                        new_weight = min(1.0, synapse.weight + (0.01 * reward))
                        synapse.weight = new_weight
                        strengthened_connections += 1
            
            logger.info(f"Applied simplified reinforcement with reward: {reward:.2f}, strengthened {strengthened_connections} connections")
    except Exception as e:
        logger.warning(f"Simplified reinforcement failed: {str(e)}")
    
    # 8. Update neural substrate
    try:
        # Get features from perception results if available
        feature_vector = []
        
        # Check if perception result contains feature data
        if "perception_result" in perception_result:
            pr = perception_result["perception_result"]
            feature_vector = pr.get("feature_vector", [])
        elif "features" in perception_result:
            # Try to get features directly if they exist
            feature_dict = perception_result.get("features", {})
            if "reduced_embedding" in feature_dict:
                # Use the reduced embedding as features
                feature_vector = feature_dict["reduced_embedding"].tolist() if hasattr(feature_dict["reduced_embedding"], "tolist") else list(feature_dict["reduced_embedding"])
            elif "embedding" in feature_dict:
                # Fall back to full embedding
                feature_vector = feature_dict["embedding"].tolist() if hasattr(feature_dict["embedding"], "tolist") else list(feature_dict["embedding"])
            elif "linguistic_features" in feature_dict:
                # Fall back to linguistic features
                linguistic_features = feature_dict["linguistic_features"]
                feature_vector = [v for k, v in sorted(linguistic_features.items())]
        
        # Ensure we have some features to work with
        if not feature_vector:
            # Create some random features for neural activation
            feature_vector = [random.uniform(0.3, 0.5) for _ in range(16)]  # Increased values
            logger.info("Using random feature vector for neural activation")
        
        # Limit to first 16 features for neural input
        feature_vector = feature_vector[:16]
        # Pad if necessary to have exactly 16 features
        if len(feature_vector) < 16:
            feature_vector.extend([0.0] * (16 - len(feature_vector)))
            
        # Boost the feature values to ensure activation
        boosted_features = []
        for val in feature_vector:
            # Scale values to increase activation potential
            boosted_val = min(1.0, val * 1.5)  # Boost by 50%
            # Ensure each value has at least 0.3 to activate neurons
            boosted_val = max(0.3, boosted_val)
            boosted_features.append(boosted_val)
            
        # Create input data for neural network
        input_data = {f"input_{i}": float(val) for i, val in enumerate(boosted_features)}
        
        # Log input data for debugging
        logger.info(f"Neural network input data: min={min(boosted_features):.2f}, max={max(boosted_features):.2f}, avg={sum(boosted_features)/len(boosted_features):.2f}")
            
        # Use the activate method
        network_outputs = neural_substrate["neural_network"].activate(
            inputs=input_data,
            steps=3
        )
        
        # Log network outputs for debugging
        logger.info(f"Neural network outputs: {', '.join([f'{k}={v:.2f}' for k, v in network_outputs.items()])}")
        
        # Count activated neurons for learning - use a lower threshold to ensure some activation
        activation_threshold = 0.2  # Lower threshold to 0.2
        activated_neurons = []
        
        # Log neuron activations for debugging
        neuron_activations = []
        for neuron_id, neuron in neural_substrate["neural_network"].neurons.items():
            neuron_activations.append(neuron.activation)
            if neuron.activation > activation_threshold:  # Lower threshold for considering a neuron "activated"
                activated_neurons.append(neuron_id)
        
        # Log activation statistics
        if neuron_activations:
            logger.info(f"Neuron activations: min={min(neuron_activations):.2f}, max={max(neuron_activations):.2f}, avg={sum(neuron_activations)/len(neuron_activations):.2f}")
        
        # If no neurons are activated even with the lower threshold, activate the neurons with the highest activations
        if not activated_neurons:
            # Get the top 3 most activated neurons
            top_neurons = sorted(
                neural_substrate["neural_network"].neurons.items(),
                key=lambda x: x[1].activation,
                reverse=True
            )[:3]
            
            for neuron_id, neuron in top_neurons:
                activated_neurons.append(neuron_id)
                logger.info(f"Forcing activation of top neuron {neuron_id} with activation {neuron.activation:.3f}")
        
        # Log the actual activation count for debugging
        logger.info(f"Neural network activated with {len(activated_neurons)} neurons above threshold {activation_threshold}.")
        
    except Exception as e:
        logger.warning(f"Neural network activation failed: {str(e)}")
        activated_neurons = []  # Ensure this is defined in case of exception
    
    # 9. Update development
    # Calculate development increment based on interaction quality
    interaction_quality = 0.5  # Base quality
    
    # Adjust based on comprehension level
    comprehension_level = mother_response.get("interaction_details", {}).get("comprehension_level", "partial")
    comprehension_bonus = {
        "none": 0.0,
        "minimal": 0.1,
        "partial": 0.2,
        "functional": 0.3,
        "solid": 0.4,
        "mastery": 0.5
    }.get(str(comprehension_level), 0.2)
    
    # Add a small bonus based on perception novelty
    novelty_bonus = 0.0
    if perception_result.get("perception_result", {}).get("novelty_score"):
        novelty_score = perception_result["perception_result"]["novelty_score"]
        novelty_bonus = novelty_score * 0.2  # Scale the bonus
    
    interaction_quality += comprehension_bonus + novelty_bonus
    
    # Calculate time spent on this interaction
    interaction_duration = time.time() - interaction_start_time
    time_factor = min(1.0, interaction_duration / 10.0)  # Cap at 1.0
    
    # Calculate final development increment
    development_increment = 0.01 * interaction_quality * time_factor
    
    # Update mind development
    mind.update_development(delta_time=development_increment)
    
    # 10. Save concept comprehension to state manager
    concept_comprehension = state_manager.get_state("concept_comprehension") or {}
    if "interaction_details" in mother_response and "comprehension_level" in mother_response["interaction_details"]:
        concept = mother_response.get("interaction_details", {}).get("learning_goal", "general concept")
        concept_comprehension[concept] = mother_response["interaction_details"]["comprehension_level"]
        state_manager.update_state({"concept_comprehension": concept_comprehension})
    
    # Update interaction count
    interaction_count = state_manager.get_state("interaction_count") or 0
    interaction_count += 1
    state_manager.update_state({"interaction_count": interaction_count})
    
    return {
        "lmm_input": lmm_input,
        "mother_response": mother_response,
        "perception": perception_result,
        "attention": attention_result,
        "memory": memory_result,
        "language": language_result,
        "mind_state": mind_state,
        "development_increment": development_increment
    }


def main():
    """
    Main entry point
    """
    try:
        # Create necessary directories
        os.makedirs(os.path.join("lmm_project", "logs"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "storage"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "visualization", "output"), exist_ok=True)
        
        # Initialize database
        db_conn = initialize_database()
        
        # Create shared event bus for component communication
        event_bus = EventBus()
        
        # Create state manager
        state_manager = StateManager()
        
        # Initialize components
        mind = initialize_mind(event_bus)
        mother = initialize_mother(personality_profile="nurturing")  # Can be changed to any profile
        neural_substrate = initialize_neural_substrate(event_bus)
        learning_engines = initialize_learning_engines(event_bus)
        
        # Display system information
        print("\n===== Large Mind Model (LMM) System Information =====")
        print(f"Neural Network: {neural_substrate['neural_network'].network_id}")
        print(f"Number of Neurons: {len(neural_substrate['neural_network'].neurons)}")
        print(f"Number of Synapses: {len(neural_substrate['neural_network'].synapses)}")
        print(f"Cognitive Modules: {list(mind.modules.keys())}")
        print(f"Learning Engines: {list(learning_engines.keys())}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Mother Personality: {mother.personality_manager.profile}")
        print(f"Mother Teaching Style: {mother.teaching_style}")
        print(f"Storage: SQLite Database")
        print("=" * 60)
        
        # Run autonomous interaction session
        autonomous_interaction_session(
            mind=mind,
            mother=mother,
            neural_substrate=neural_substrate,
            learning_engines=learning_engines,
            state_manager=state_manager,
            db_conn=db_conn,
            max_interactions=100,  # Adjust as needed
            interaction_delay=3.0  # Adjust as needed
        )
        
        # Close database connection
        db_conn.close()
        logger.info("Database connection closed")
        print("\nDatabase connection closed.")
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")


if __name__ == "__main__":
    main()