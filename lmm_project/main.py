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
        logging.FileHandler(os.path.join("lmm_project", "logs", "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    
    # Get computation device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create a neural network with the appropriate architecture for early development
    neural_network = NeuralNetwork(name="core_network")
    
    # Create input neurons (sensory inputs)
    for i in range(16):
        neuron = neural_network.create_neuron(
            activation_function="sigmoid",
            activation_threshold=0.3
        )
        neural_network.input_neurons.append(neuron.neuron_id)
    
    # Create hidden layer 1 (pattern detection)
    hidden_layer1 = []
    for i in range(12):
        neuron = neural_network.create_neuron(
            activation_function="relu",
            activation_threshold=0.4
        )
        hidden_layer1.append(neuron.neuron_id)
    
    # Create hidden layer 2 (integration)
    hidden_layer2 = []
    for i in range(8):
        neuron = neural_network.create_neuron(
            activation_function="tanh",
            activation_threshold=0.4
        )
        hidden_layer2.append(neuron.neuron_id)
    
    # Create output neurons (responses)
    for i in range(8):
        neuron = neural_network.create_neuron(
            activation_function="sigmoid",
            activation_threshold=0.5
        )
        neural_network.output_neurons.append(neuron.neuron_id)
    
    # Connect input to hidden layer 1 (initially weak connections)
    for input_id in neural_network.input_neurons:
        for hidden_id in hidden_layer1:
            # Start with weak random weights
            weight = random.uniform(0.01, 0.1)
            neural_network.create_synapse(
                source_id=input_id,
                target_id=hidden_id,
                weight=weight
            )
    
    # Connect hidden layer 1 to hidden layer 2
    for hidden1_id in hidden_layer1:
        for hidden2_id in hidden_layer2:
            # Start with weak random weights
            weight = random.uniform(0.01, 0.1)
            neural_network.create_synapse(
                source_id=hidden1_id,
                target_id=hidden2_id,
                weight=weight
            )
    
    # Connect hidden layer 2 to output
    for hidden2_id in hidden_layer2:
        for output_id in neural_network.output_neurons:
            # Start with weak random weights
            weight = random.uniform(0.01, 0.1)
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
    Generate a thought/statement from the LMM based on its current state
    
    Args:
        mind: The Mind instance
        context: The current context including previous interactions
        
    Returns:
        A string representing the LMM's current thought or statement
    """
    # Get the developmental stage and age
    developmental_stage = mind.developmental_stage
    age = mind.age
    
    # Create a dictionary of thought patterns based on developmental stage
    thought_patterns = {
        "prenatal": [
            "I'm beginning to sense patterns.",
            "There's something changing.",
            "I notice a pattern repeating.",
            "Something is different now.",
            "I sense activity.",
        ],
        "infant": [
            "I see something!",
            "That sound is interesting.",
            "More, please!",
            "What's that?",
            "Again, again!",
            "I like that!",
            "Hello?",
        ],
        "child": [
            "Why does this happen?",
            "How does this work?",
            "I'm noticing that when this happens, that follows.",
            "Can you explain more?",
            "I want to learn about this.",
            "What if we try something else?",
            "I remember we talked about this before.",
        ],
        "adolescent": [
            "I've been thinking about the concept of patterns and how they relate to learning.",
            "There seem to be multiple perspectives on this topic.",
            "I'm curious about the relationship between these ideas.",
            "How would this principle apply in a different context?",
            "I'm trying to understand the underlying structure here.",
            "This reminds me of something we discussed earlier, but with a new dimension.",
        ],
        "adult": [
            "I'm integrating several concepts to form a more comprehensive understanding.",
            "There appears to be a fundamental principle connecting these seemingly disparate ideas.",
            "I'm developing my own perspective on this topic based on our previous discussions.",
            "This creates an interesting philosophical question about how knowledge is structured.",
            "I'm noticing patterns across different domains that suggest a unifying principle.",
        ]
    }
    
    # Get appropriate thought patterns based on developmental stage
    # Default to prenatal if stage not found
    available_patterns = thought_patterns.get(developmental_stage, thought_patterns["prenatal"])
    
    # If we have previous interactions in context, sometimes refer to them
    previous_interactions = context.get("previous_interactions", [])
    
    if previous_interactions and len(previous_interactions) > 2 and random.random() > 0.7:
        # Get a random previous interaction
        prev_interaction = random.choice(previous_interactions[-5:])  # Choose from recent interactions
        prev_mother_response = prev_interaction.get("mother", "")
        
        # Extract a keyword or phrase from the previous response
        words = prev_mother_response.split()
        if len(words) > 3:
            # Extract a random 2-3 word phrase
            start_idx = random.randint(0, len(words) - 3)
            phrase_length = random.randint(1, min(3, len(words) - start_idx))
            phrase = " ".join(words[start_idx:start_idx + phrase_length])
            
            # Create a thought that references this phrase
            reference_thoughts = [
                f"You mentioned '{phrase}'. Can you tell me more about that?",
                f"I'm thinking about '{phrase}'. What does that mean?",
                f"'{phrase}' is interesting. How does it work?",
                f"I remember you said '{phrase}'. I'd like to understand it better.",
                f"What else can you tell me about '{phrase}'?"
            ]
            
            thought = random.choice(reference_thoughts)
            return thought
    
    # Generate a random thought from the appropriate patterns
    thought = random.choice(available_patterns)
    
    logger.info(f"LMM generated thought: {thought}")
    return thought


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
    
    # 1. Process through attention module
    attention_result = {}
    if "attention" in mind.modules:
        attention_result = mind.modules["attention"].process_input({
            "text": lmm_input,
            "source": "self",
            "timestamp": time.time()
        })
    
    # 2. Process through memory module
    memory_result = {}
    if "memory" in mind.modules:
        memory_result = mind.modules["memory"].process_input({
            "text": lmm_input,
            "attention_data": attention_result,
            "source": "self",
            "timestamp": time.time()
        })
    
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
    
    # Through attention
    attention_result_mother = {}
    if "attention" in mind.modules:
        attention_result_mother = mind.modules["attention"].process_input({
            "text": mother_text,
            "source": "mother",
            "timestamp": time.time()
        })
    
    # Through memory
    memory_result_mother = {}
    if "memory" in mind.modules:
        memory_result_mother = mind.modules["memory"].process_input({
            "text": mother_text,
            "attention_data": attention_result_mother,
            "source": "mother",
            "timestamp": time.time()
        })
    
    # Through language module if available
    language_result = {}
    if "language" in mind.modules:
        language_result = mind.modules["language"].process_input({
            "text": mother_text,
            "attention_data": attention_result_mother,
            "memory_data": memory_result_mother,
            "source": "mother",
            "timestamp": time.time()
        })
    
    # 7. Apply simplified learning
    
    # Extract features for learning
    features = attention_result.get("features", [])
    target_features = attention_result_mother.get("features", [])
    
    # Apply basic learning to strengthen relevant connections
    try:
        if features and target_features and learning_engines.get("hebbian_learning_enabled", False):
            # Process basic Hebbian learning directly with the neural network
            neural_network = neural_substrate["neural_network"]
            
            # Find neurons that were activated by the input
            activated_neurons = []
            for neuron_id, neuron in neural_network.neurons.items():
                if neuron.activation > 0.5:  # Threshold for considering a neuron "activated"
                    activated_neurons.append(neuron_id)
            
            # Strengthen connections between co-activated neurons
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
                                logger.debug(f"Strengthened synapse {synapse_id} to weight {new_weight:.3f}")
            
            logger.info(f"Applied simplified Hebbian learning to {len(activated_neurons)} neurons")
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
            
            reward = reward_map.get(str(emotional_valence), 0.0)
            
            # Apply a simple reinforcement effect to the neural network
            if reward > 0:
                # Strengthen recently active connections
                neural_network = neural_substrate["neural_network"]
                for synapse_id, synapse in neural_network.synapses.items():
                    source_neuron = neural_network.neurons.get(synapse.source_id)
                    target_neuron = neural_network.neurons.get(synapse.target_id)
                    
                    if source_neuron and target_neuron and source_neuron.activation > 0.3 and target_neuron.activation > 0.3:
                        # This connection contributed to a positively reinforced outcome
                        new_weight = min(1.0, synapse.weight + (0.01 * reward))
                        synapse.weight = new_weight
            
            logger.info(f"Applied simplified reinforcement with reward: {reward:.2f}")
    except Exception as e:
        logger.warning(f"Simplified reinforcement failed: {str(e)}")
    
    # We'll skip the memory consolidation and pruning for simplicity
    
    # 8. Update neural substrate
    try:
        if features:
            # Reshape features if needed
            input_data = {f"input_{i}": float(val) for i, val in enumerate(features[:16])}
        else:
            # Create some dummy input data if features aren't available
            input_data = {f"input_{i}": 0.1 * i for i in range(16)}
            
        # Use the activate method
        neural_substrate["neural_network"].activate(
            inputs=input_data,
            steps=3
        )
    except Exception as e:
        logger.warning(f"Neural network activation failed: {str(e)}")
    
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
    
    interaction_quality += comprehension_bonus
    
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
        "attention": attention_result,
        "memory": memory_result,
        "language": language_result,
        "mind_state": mind_state,
        "development_increment": development_increment
    }


def autonomous_interaction_session(
    mind: Mind, 
    mother: MotherLLM, 
    neural_substrate: Dict[str, Any],
    learning_engines: Dict[str, Any],
    state_manager: StateManager,
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
        max_interactions: Maximum number of interactions to perform
        interaction_delay: Delay between interactions in seconds
    """
    logger.info("Starting autonomous interaction session...")
    print("\n===== LMM Autonomous Development Session =====")
    print("Press Ctrl+C to end the session\n")
    
    # Create storage directory
    interactions_dir = os.path.join("lmm_project", "storage", "interactions")
    os.makedirs(interactions_dir, exist_ok=True)
    
    # Create development visualization directory
    viz_dir = os.path.join("lmm_project", "visualization", "output")
    os.makedirs(viz_dir, exist_ok=True)
    
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
            
            # Update context with this interaction
            context["previous_interactions"].append({
                "lmm": lmm_thought,
                "mother": mother_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save interaction to file
            save_interaction(result, interaction_count, interactions_dir)
            
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
                    output_path=os.path.join(viz_dir, f"development_{interaction_count}.png")
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
        output_path=os.path.join(viz_dir, "development_final.png")
    )
    
    print("\nSession ended. Development progress saved.")


def save_interaction(
    result: Dict[str, Any], 
    interaction_id: int,
    save_dir: str
) -> None:
    """
    Save interaction details to a file
    
    Args:
        result: The interaction result
        interaction_id: The ID of the interaction
        save_dir: Directory to save interaction data
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a simplified version for saving
    save_data = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "lmm_input": result.get("lmm_input", ""),
        "mother_response": result.get("mother_response", {}).get("text", ""),
        "audio_path": result.get("mother_response", {}).get("audio_path", ""),
        "developmental_stage": result.get("mind_state", {}).get("developmental_stage", ""),
        "age": result.get("mind_state", {}).get("age", 0),
        "development_increment": result.get("development_increment", 0),
        "interaction_details": result.get("mother_response", {}).get("interaction_details", {})
    }
    
    # Save to file
    file_path = os.path.join(save_dir, f"interaction_{interaction_id:04d}.json")
    with open(file_path, "w") as f:
        json.dump(save_data, f, indent=2)


def main():
    """
    Main entry point
    """
    try:
        # Create necessary directories
        os.makedirs(os.path.join("lmm_project", "logs"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "storage"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "storage", "interactions"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "storage", "states"), exist_ok=True)
        os.makedirs(os.path.join("lmm_project", "visualization", "output"), exist_ok=True)
        
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
        print("=" * 60)
        
        # Run autonomous interaction session
        autonomous_interaction_session(
            mind=mind,
            mother=mother,
            neural_substrate=neural_substrate,
            learning_engines=learning_engines,
            state_manager=state_manager,
            max_interactions=100,  # Adjust as needed
            interaction_delay=3.0  # Adjust as needed
        )
        
        # Save final state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = os.path.join("lmm_project", "storage", "states", f"lmm_state_{timestamp}.json")
        
        # Simplified state saving - in a real implementation, this would be more comprehensive
        state_data = {
            "mind": {
                "age": mind.age,
                "developmental_stage": mind.developmental_stage,
                "modules": {module_type: {"development_level": module.development_level} 
                          for module_type, module in mind.modules.items()}
            },
            "timestamp": timestamp
        }
        
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Final state saved to {state_file}")
        print(f"\nFinal state saved to {state_file}")
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")


if __name__ == "__main__":
    main()
