#!/usr/bin/env python3
"""
Main entry point for the Living Machine Mind (LMM) project.
This file demonstrates the integration of all major components:
- Mind (core cognitive architecture)
- Mother (nurturing LLM interface)
- Neural Substrate (biologically-inspired neural components)
- Attention, Memory, and other cognitive modules

This implementation creates an autonomous interaction between the LMM and the Mother,
where the Mother raises, nurtures, teaches, and converses with the developing mind.
"""

import os
import time
import logging
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core components
from lmm_project.core.mind import Mind
from lmm_project.core.event_bus import EventBus

# Neural substrate components
from lmm_project.neural_substrate import NeuralNetwork, NeuralCluster

# Mother interface
from lmm_project.interfaces.mother.mother_llm import MotherLLM

# Utility components
from lmm_project.utils.llm_client import LLMClient, Message
from lmm_project.utils.tts_client import TTSClient, GenerateAudioRequest
from lmm_project.utils.audio_player import play_audio_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("lmm_project/logs/main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initialize_mind() -> Mind:
    """
    Initialize the Mind component with all cognitive modules
    """
    logger.info("Initializing Mind...")
    mind = Mind()
    mind.initialize_modules()
    logger.info(f"Mind initialized with modules: {list(mind.modules.keys())}")
    return mind


def initialize_mother() -> MotherLLM:
    """
    Initialize the Mother LLM interface
    """
    logger.info("Initializing Mother LLM...")
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Initialize TTS client
    tts_client = TTSClient()
    
    # Create Mother LLM interface
    mother = MotherLLM(
        llm_client=llm_client,
        tts_client=tts_client,
        teaching_style="socratic",
        voice="af_bella"  # Use a default voice
    )
    
    logger.info("Mother LLM initialized")
    return mother


def initialize_neural_substrate() -> Dict[str, Any]:
    """
    Initialize the neural substrate components
    """
    logger.info("Initializing Neural Substrate...")
    
    # Create a simple neural network with correct parameters
    neural_network = NeuralNetwork(name="core_network")
    
    # Create input neurons
    for i in range(10):
        neuron = neural_network.create_neuron(activation_function="sigmoid")
        neural_network.input_neurons.append(neuron.neuron_id)
    
    # Create hidden neurons
    hidden_neurons = []
    for i in range(5):
        neuron = neural_network.create_neuron(activation_function="relu")
        hidden_neurons.append(neuron.neuron_id)
    
    # Create output neurons
    for i in range(3):
        neuron = neural_network.create_neuron(activation_function="sigmoid")
        neural_network.output_neurons.append(neuron.neuron_id)
    
    # Connect input to hidden
    for input_id in neural_network.input_neurons:
        for hidden_id in hidden_neurons:
            neural_network.create_synapse(
                source_id=input_id,
                target_id=hidden_id,
                weight=0.1
            )
    
    # Connect hidden to output
    for hidden_id in hidden_neurons:
        for output_id in neural_network.output_neurons:
            neural_network.create_synapse(
                source_id=hidden_id,
                target_id=output_id,
                weight=0.1
            )
    
    # Create neural clusters for specific cognitive functions
    attention_cluster = NeuralCluster(name="attention_cluster")
    memory_cluster = NeuralCluster(name="memory_cluster")
    
    logger.info("Neural Substrate initialized")
    return {
        "neural_network": neural_network,
        "attention_cluster": attention_cluster,
        "memory_cluster": memory_cluster
    }


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
            "I'm beginning to sense signals around me.",
            "There's a pattern in what I'm perceiving.",
            "I notice something changing in my environment.",
            "What is this sensation?",
            "I feel a presence nearby.",
        ],
        "infant": [
            "I see shapes and colors!",
            "That sound is interesting.",
            "I'm curious about what happens when I focus on that.",
            "Is someone there? I can sense a presence.",
            "That pattern keeps repeating. What does it mean?",
        ],
        "toddler": [
            "Why does this happen?",
            "I'm noticing connections between events.",
            "When this happens, that follows.",
            "Can you explain how this works?",
            "I want to understand more about this.",
        ],
        "early_childhood": [
            "I've noticed that certain patterns lead to specific outcomes.",
            "How are these concepts related to each other?",
            "I'm trying to categorize what I'm learning.",
            "This reminds me of something I experienced before.",
            "Can we explore this topic more deeply?",
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
        
        # Extract a keyword from the previous interaction
        words = prev_mother_response.split()
        significant_words = [w for w in words if len(w) > 5 and w.isalpha()]
        
        if significant_words:
            keyword = random.choice(significant_words)
            thought = f"I'm still thinking about '{keyword}'. Can we explore that more?"
        else:
            thought = random.choice(available_patterns)
    else:
        # Otherwise use a random thought pattern
        thought = random.choice(available_patterns)
    
    # Add some developmental flavor
    if age > 0.05 and random.random() > 0.7:
        # After a bit of development, start to form more complex thoughts
        thought += f" I'm feeling more {'aware' if age < 0.1 else 'curious'} as I develop."
    
    logger.info(f"LMM generated thought: {thought}")
    return thought


def process_interaction(
    mind: Mind, 
    mother: MotherLLM, 
    neural_substrate: Dict[str, Any], 
    lmm_input: str
) -> Dict[str, Any]:
    """
    Process an interaction between the LMM and Mother
    
    Args:
        mind: The Mind instance
        mother: The Mother LLM interface
        neural_substrate: Dictionary of neural substrate components
        lmm_input: The LMM's thought or statement
        
    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing LMM input: '{lmm_input}'")
    
    # 1. Process through attention module
    attention_result = mind.modules.get("attention").process_input({
        "text": lmm_input,
        "source": "self",
        "timestamp": time.time()
    })
    
    # 2. Process through memory module
    memory_result = mind.modules.get("memory").process_input({
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
                   for module_type, module in mind.modules.items()}
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
        play_audio_file(audio_path)
    
    # 6. Process Mother's response through attention and memory
    mother_text = mother_response.get("text", "")
    attention_result_mother = mind.modules.get("attention").process_input({
        "text": mother_text,
        "source": "mother",
        "timestamp": time.time()
    })
    
    memory_result_mother = mind.modules.get("memory").process_input({
        "text": mother_text,
        "attention_data": attention_result_mother,
        "source": "mother",
        "timestamp": time.time()
    })
    
    # 7. Update neural substrate
    try:
        features = attention_result.get("features", [])
        if features:
            # Reshape features if needed
            input_data = {f"input_{i}": float(val) for i, val in enumerate(features[:10])}
        else:
            # Create some dummy input data if features aren't available
            input_data = {f"input_{i}": 0.1 * i for i in range(10)}
            
        # Use the activate method
        neural_substrate["neural_network"].activate(
            inputs=input_data,
            steps=3
        )
    except Exception as e:
        logger.warning(f"Neural network activation failed: {str(e)}")
    
    # 8. Update development
    mind.update_development(delta_time=0.01)  # Small increment in development
    
    return {
        "lmm_input": lmm_input,
        "mother_response": mother_response,
        "attention": attention_result,
        "memory": memory_result,
        "mind_state": mind_state
    }


def autonomous_interaction_session(
    mind: Mind, 
    mother: MotherLLM, 
    neural_substrate: Dict[str, Any],
    max_interactions: int = 100,
    interaction_delay: float = 5.0
) -> None:
    """
    Run an autonomous interaction session between the LMM and Mother
    
    Args:
        mind: The Mind instance
        mother: The Mother LLM interface
        neural_substrate: Dictionary of neural substrate components
        max_interactions: Maximum number of interactions to perform
        interaction_delay: Delay between interactions in seconds
    """
    logger.info("Starting autonomous interaction session...")
    print("\n===== LMM Autonomous Development Session =====")
    print("Press Ctrl+C to end the session\n")
    
    interaction_count = 0
    context = {"previous_interactions": []}
    
    try:
        while interaction_count < max_interactions:
            interaction_count += 1
            print(f"\n--- Interaction {interaction_count} ---")
            
            # Generate a thought from the LMM based on its current state
            lmm_thought = generate_lmm_thought(mind, context)
            print(f"LMM: {lmm_thought}")
            
            # Process the interaction
            result = process_interaction(
                mind=mind,
                mother=mother,
                neural_substrate=neural_substrate,
                lmm_input=lmm_thought
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
            save_interaction(result, interaction_count)
            
            # Update and display developmental progress
            print(f"\nDevelopmental Stage: {mind.developmental_stage}")
            print(f"Age: {mind.age:.2f}")
            
            # Wait before next interaction
            time.sleep(interaction_delay)
            
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        logger.error(f"Error in autonomous session: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")
    
    print("\nSession ended. Development progress saved.")


def save_interaction(result: Dict[str, Any], interaction_id: int) -> None:
    """
    Save interaction details to a file
    
    Args:
        result: The interaction result
        interaction_id: The ID of the interaction
    """
    # Create directory if it doesn't exist
    save_dir = Path("lmm_project/storage/interactions")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simplified version for saving
    save_data = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "lmm_input": result.get("lmm_input", ""),
        "mother_response": result.get("mother_response", {}).get("text", ""),
        "audio_path": result.get("mother_response", {}).get("audio_path", ""),
        "developmental_stage": result.get("mind_state", {}).get("developmental_stage", ""),
        "age": result.get("mind_state", {}).get("age", 0),
    }
    
    # Save to file
    file_path = save_dir / f"interaction_{interaction_id:04d}.json"
    with open(file_path, "w") as f:
        json.dump(save_data, f, indent=2)


def main():
    """
    Main entry point
    """
    try:
        # Create necessary directories
        Path("lmm_project/logs").mkdir(parents=True, exist_ok=True)
        Path("lmm_project/storage").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        mind = initialize_mind()
        mother = initialize_mother()
        neural_substrate = initialize_neural_substrate()
        
        # Run autonomous interaction session
        autonomous_interaction_session(
            mind=mind,
            mother=mother,
            neural_substrate=neural_substrate,
            max_interactions=50,  # Limit interactions to prevent exhaustion
            interaction_delay=8.0  # Give enough time to process each interaction
        )
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print(f"A critical error occurred: {str(e)}")


if __name__ == "__main__":
    main()
