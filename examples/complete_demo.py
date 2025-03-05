#!/usr/bin/env python3
"""
Complete demonstration script for the NeuralChild project.

This script provides a comprehensive demonstration of the NeuralChild 
framework, showing all major features including development tracking,
component visualization, and mother-child interactions.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

# Add the parent directory to the path so we can import NeuralChild
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralChild import NeuralChild, DashboardApp, DevelopmentalStage
from NeuralChild.core.mother import Mother


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='NeuralChild Complete Demonstration')
    
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no dashboard)')
    parser.add_argument('--interactions', type=int, default=20, help='Number of interactions to simulate')
    parser.add_argument('--save-state', type=str, help='Save final state to specified file')
    parser.add_argument('--load-state', type=str, help='Load initial state from specified file')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port (default: 8050)')
    
    return parser.parse_args()


def load_custom_config(config_path):
    """Load custom configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"Loaded custom configuration from {config_path}")
        return config_data
    except Exception as e:
        print(f"Error loading custom config: {e}")
        print("Using default configuration.")
        return {}


def run_headless_demonstration(neural_child, num_interactions, save_path=None):
    """
    Run a headless demonstration without the dashboard.
    
    Args:
        neural_child: NeuralChild instance
        num_interactions: Number of interactions to simulate
        save_path: Optional path to save state after simulation
    """
    print("\n" + "=" * 50)
    print(f"Running headless demonstration for {num_interactions} interactions...")
    print("=" * 50 + "\n")
    
    start_time = time.time()
    
    # Get initial metrics
    initial_metrics = neural_child.get_developmental_metrics()
    print("Initial state:")
    print(f"Developmental stage: {DevelopmentalStage(initial_metrics['developmental_stage']).name}")
    print(f"Age: {initial_metrics['age_days']} days")
    print(f"Vocabulary size: {initial_metrics['vocabulary_size']}")
    print(f"Emotional stability: {initial_metrics['emotional_stability']:.2f}")
    print(f"Self-awareness: {initial_metrics['self_awareness']:.2f}")
    print("\n" + "-" * 50 + "\n")
    
    # Run simulation with detailed output
    print("Beginning mother-child interactions:")
    for i in range(num_interactions):
        print(f"\nInteraction {i+1}/{num_interactions}:")
        
        # Interact with mother
        result = neural_child.interact_with_mother()
        
        # Extract child and mother messages
        child_input = result.get('child_input', {})
        mother_response = result.get('mother_response', {})
        
        # Print interaction
        child_content = child_input.get('content', '')
        mother_content = mother_response.get('verbal_response', '')
        
        if child_content:
            print(f"Child: {child_content}")
        
        if mother_content:
            print(f"Mother: {mother_content}")
        
        # Get current metrics every 5 interactions
        if (i + 1) % 5 == 0:
            metrics = neural_child.get_developmental_metrics()
            print("\nCurrent developmental status:")
            print(f"Age: {metrics['age_days']} days")
            print(f"Stage: {DevelopmentalStage(metrics['developmental_stage']).name}")
            print(f"Vocabulary size: {metrics['vocabulary_size']}")
            print(f"Emotional stability: {metrics['emotional_stability']:.2f}")
            print(f"Self-awareness: {metrics['self_awareness']:.2f}")
            print("-" * 30)
    
    # Get final metrics
    final_metrics = neural_child.get_developmental_metrics()
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print("=" * 50)
    print(f"\nCompleted {num_interactions} interactions in {elapsed_time:.2f} seconds")
    print("\nFinal state:")
    print(f"Developmental stage: {DevelopmentalStage(final_metrics['developmental_stage']).name}")
    print(f"Age: {final_metrics['age_days']} days (from initial {initial_metrics['age_days']} days)")
    print(f"Vocabulary size: {final_metrics['vocabulary_size']} (from initial {initial_metrics['vocabulary_size']})")
    print(f"Emotional stability: {final_metrics['emotional_stability']:.2f} (from initial {initial_metrics['emotional_stability']:.2f})")
    print(f"Self-awareness: {final_metrics['self_awareness']:.2f} (from initial {initial_metrics['self_awareness']:.2f})")
    print(f"Language complexity: {final_metrics['language_complexity']:.2f} (from initial {initial_metrics['language_complexity']:.2f})")
    
    # Print component states
    print("\nNeural component states:")
    component_states = neural_child.get_component_states()
    for component, state in component_states.items():
        print(f"- {component}: Activation {state.get('activation', 0):.2f}, Confidence {state.get('confidence', 0):.2f}")
    
    # Save state if requested
    if save_path:
        print(f"\nSaving neural child state to {save_path}...")
        success = neural_child.save_state(save_path)
        if success:
            print("State saved successfully.")
        else:
            print("Failed to save state.")
    
    print("\nHeadless demonstration complete.")


def run_dashboard_demonstration(neural_child, port=8050):
    """
    Run the dashboard for an interactive demonstration.
    
    Args:
        neural_child: NeuralChild instance
        port: Port to run the dashboard on
    """
    print("\n" + "=" * 50)
    print("Starting interactive dashboard demonstration...")
    print("=" * 50)
    
    # Print initial state
    metrics = neural_child.get_developmental_metrics()
    print("\nInitial state:")
    print(f"Developmental stage: {DevelopmentalStage(metrics['developmental_stage']).name}")
    print(f"Age: {metrics['age_days']} days")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    
    # Create and launch dashboard
    print(f"\nLaunching dashboard on port {port}...")
    print("Use the dashboard to interact with the child, monitor development,")
    print("and visualize neural component activations.")
    print("\nPress Ctrl+C in the terminal to stop the dashboard.")
    
    dashboard = DashboardApp(neural_child)
    dashboard_url = f"http://localhost:{port}/"
    print(f"\nDashboard running at {dashboard_url}")
    print(f"Open {dashboard_url} in your browser to interact with NeuralChild")
    
    try:
        dashboard.run_server(debug=False, port=port)
    except KeyboardInterrupt:
        print("\nShutting down NeuralChild dashboard...")


def main():
    """Main entry point for the demo."""
    print("🧠 NeuralChild: Complete Demonstration")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_custom_config(args.config)
    
    # Set up mother with config
    mother_params = config.get('mother', {})
    mother = Mother(
        personality_traits=mother_params.get('personality_traits'),
        parenting_style=mother_params.get('parenting_style'),
        teaching_style=mother_params.get('teaching_style')
    )
    
    # Create neural child
    print("Initializing neural child...")
    neural_child = NeuralChild(
        config=config,
        mother=mother,
        load_state_path=args.load_state
    )
    
    if args.load_state:
        print(f"Loaded neural child state from {args.load_state}")
    
    # Run demonstration
    if args.headless:
        run_headless_demonstration(
            neural_child,
            args.interactions,
            args.save_state
        )
    else:
        run_dashboard_demonstration(
            neural_child,
            args.port
        )


if __name__ == "__main__":
    main()