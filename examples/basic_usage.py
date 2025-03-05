#!/usr/bin/env python3
"""
Basic usage example for the NeuralChild framework.

This example demonstrates how to create a NeuralChild instance,
interact with it, and monitor its development.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the parent directory to the path so we can import NeuralChild
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralChild import NeuralChild, DevelopmentalStage
from NeuralChild.core.mother import Mother


def basic_interaction_example():
    """Example of basic interaction with the Neural Child."""
    print("Creating a new Neural Child...\n")
    
    # Create a neural child with default settings
    child = NeuralChild()
    
    # Get initial metrics
    metrics = child.get_developmental_metrics()
    print("Initial state:")
    print(f"Developmental stage: {DevelopmentalStage(metrics['developmental_stage']).name}")
    print(f"Age: {metrics['age_days']} days")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"Emotional stability: {metrics['emotional_stability']:.2f}")
    print("\n" + "-"*50 + "\n")
    
    # Perform a few interactions
    print("Performing 5 interactions...\n")
    for i in range(5):
        print(f"Interaction {i+1}:")
        result = child.interact_with_mother()
        
        # Extract child and mother content
        child_input = result.get('child_input', {})
        mother_response = result.get('mother_response', {})
        
        # Print interaction
        child_content = child_input.get('content', '')
        mother_content = mother_response.get('verbal_response', '')
        
        if child_content:
            print(f"Child: {child_content}")
        
        if mother_content:
            print(f"Mother: {mother_content}")
        
        print()
    
    # Get updated metrics
    metrics = child.get_developmental_metrics()
    print("Updated state:")
    print(f"Developmental stage: {DevelopmentalStage(metrics['developmental_stage']).name}")
    print(f"Age: {metrics['age_days']} days")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"Emotional stability: {metrics['emotional_stability']:.2f}")
    print("\n" + "-"*50 + "\n")
    
    return child


def save_load_example(child):
    """Example of saving and loading Neural Child state."""
    # Create a temporary file for the state
    state_file = "temp_child_state.json"
    
    # Save the state
    print(f"Saving child state to {state_file}...")
    success = child.save_state(state_file)
    
    if success:
        print("State saved successfully.")
        
        # Create a new child and load the state
        print("\nCreating a new child and loading the saved state...")
        new_child = NeuralChild()
        
        # Get initial metrics of the new child
        metrics_before = new_child.get_developmental_metrics()
        
        # Load the state
        success = new_child.load_state(state_file)
        
        if success:
            # Get metrics after loading
            metrics_after = new_child.get_developmental_metrics()
            
            print("\nBefore loading state:")
            print(f"Developmental stage: {DevelopmentalStage(metrics_before['developmental_stage']).name}")
            print(f"Age: {metrics_before['age_days']} days")
            print(f"Vocabulary size: {metrics_before['vocabulary_size']}")
            
            print("\nAfter loading state:")
            print(f"Developmental stage: {DevelopmentalStage(metrics_after['developmental_stage']).name}")
            print(f"Age: {metrics_after['age_days']} days")
            print(f"Vocabulary size: {metrics_after['vocabulary_size']}")
            
            # Clean up
            try:
                os.remove(state_file)
                print(f"\nRemoved temporary state file: {state_file}")
            except:
                pass
        else:
            print("Failed to load state.")
    else:
        print("Failed to save state.")
    
    print("\n" + "-"*50 + "\n")


def custom_config_example():
    """Example of using a custom configuration."""
    print("Creating a Neural Child with custom configuration...\n")
    
    # Define a custom configuration
    custom_config = {
        "mother": {
            "personality_traits": {
                "openness": 0.9,       # More creative and curious
                "conscientiousness": 0.7,
                "extraversion": 0.8,    # More talkative
                "agreeableness": 0.9,   # Very nurturing
                "neuroticism": 0.2      # Emotionally stable
            },
            "parenting_style": "authoritative",
            "teaching_style": "socratic"
        },
        "development": {
            "time_acceleration": 2000,  # Faster time progression
            "learning_rate_multiplier": 1.5,  # Faster learning
        }
    }
    
    # Create mother with custom personality
    mother = Mother(
        personality_traits=custom_config["mother"]["personality_traits"],
        parenting_style=custom_config["mother"]["parenting_style"],
        teaching_style=custom_config["mother"]["teaching_style"]
    )
    
    # Create child with custom config and mother
    child = NeuralChild(config=custom_config, mother=mother)
    
    # Simulate development
    print("Simulating accelerated development (10 interactions)...")
    
    def progress_callback(info):
        print(f"Interaction {info['interaction']}/{info['total']}, " 
              f"Age: {info['metrics']['age_days']} days, "
              f"Stage: {info['stage']}")
    
    result = child.simulate_development(10, callback=progress_callback)
    
    # Print final results
    print("\nSimulation complete!")
    print(f"Final age: {result['final_metrics']['age_days']} days")
    print(f"Final developmental stage: {result['developmental_stage']}")
    print(f"Vocabulary size: {result['final_metrics']['vocabulary_size']}")
    print(f"Emotional stability: {result['final_metrics']['emotional_stability']:.2f}")
    print(f"Self-awareness: {result['final_metrics']['self_awareness']:.2f}")
    
    print("\n" + "-"*50 + "\n")


def run_all_examples():
    """Run all examples."""
    print("=" * 50)
    print("NeuralChild Basic Usage Examples")
    print("=" * 50)
    
    # Basic interaction example
    child = basic_interaction_example()
    
    # Save and load example
    save_load_example(child)
    
    # Custom configuration example
    custom_config_example()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    run_all_examples()