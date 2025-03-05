"""
Main entry point for the NeuralChild project.

This module initializes the NeuralChild system and launches the
dashboard interface for interaction and visualization.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

from .neural_child import NeuralChild
from .dashboard import DashboardApp
  # Import directly from dashboard.py instead of dashboard/ package
from .config import CONFIG
from .core.mother import Mother


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='NeuralChild: Psychological Mind Simulation')
    
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--load-state', type=str, help='Path to load state from')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port (default: 8050)')
    parser.add_argument('--no-dashboard', action='store_true', help='Run without dashboard')
    parser.add_argument('--simulate', type=int, help='Run simulation for N interactions then exit')
    parser.add_argument('--save-after-simulate', type=str, help='Save state after simulation')
    
    return parser.parse_args()


def load_custom_config(config_path):
    """Load custom configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update default config with custom values
        custom_config = CONFIG.copy()
        custom_config.update(config_data)
        
        return custom_config
    except Exception as e:
        print(f"Error loading custom config: {e}")
        print("Using default configuration.")
        return CONFIG


def setup_mother(config):
    """Set up the mother component with appropriate configuration."""
    # Create mother with config
    # Mother will use CONFIG.mother_personality internally
    mother = Mother()
    
    return mother


def run_headless_simulation(neural_child, interactions, save_path=None):
    """
    Run a headless simulation without the dashboard.
    
    Args:
        neural_child: NeuralChild instance
        interactions: Number of interactions to simulate
        save_path: Optional path to save state after simulation
    """
    print(f"Running headless simulation for {interactions} interactions...")
    
    start_time = time.time()
    
    # Define callback for progress updates
    def progress_callback(info):
        current = info['interaction']
        total = info['total']
        percentage = (current / total) * 100
        
        # Print progress update every 10% or for every interaction if total <= 10
        if total <= 10 or current % max(1, total // 10) == 0:
            print(f"Progress: {current}/{total} interactions ({percentage:.1f}%)")
            print(f"Current age: {info['metrics']['age_days']} days")
            print(f"Developmental stage: {info['stage']}")
            print(f"Vocabulary size: {info['metrics']['vocabulary_size']}")
            print("-" * 40)
    
    # Run simulation
    result = neural_child.simulate_development(interactions, callback=progress_callback)
    
    # Print results
    elapsed_time = time.time() - start_time
    print("\nSimulation complete!")
    print(f"Completed {result['interactions_completed']} interactions")
    print(f"Final age: {result['final_metrics']['age_days']} days")
    print(f"Final developmental stage: {result['developmental_stage']}")
    print(f"Vocabulary size: {result['final_metrics']['vocabulary_size']}")
    print(f"Emotional stability: {result['final_metrics']['emotional_stability']:.2f}")
    print(f"Self-awareness: {result['final_metrics']['self_awareness']:.2f}")
    print(f"Language complexity: {result['final_metrics']['language_complexity']:.2f}")
    print(f"Simulation time: {elapsed_time:.2f} seconds")
    
    # Save state if requested
    if save_path:
        print(f"Saving state to {save_path}...")
        success = neural_child.save_state(save_path)
        if success:
            print("State saved successfully.")
        else:
            print("Failed to save state.")


def main():
    """Main entry point."""
    print("🧠 NeuralChild: Psychological Mind Simulation")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = CONFIG
    if args.config:
        config = load_custom_config(args.config)
    
    # Set up mother
    mother = setup_mother(config)
    
    # Create neural child
    neural_child = NeuralChild(
        config=config,
        mother=mother,
        load_state_path=args.load_state
    )
    
    # If running headless simulation
    if args.simulate:
        run_headless_simulation(
            neural_child, 
            args.simulate, 
            args.save_after_simulate
        )
        return
    
    # If dashboard disabled, just print status and exit
    if args.no_dashboard:
        print("Neural child initialized successfully.")
        print("Dashboard disabled. Exiting.")
        return
    
    # Create and run dashboard
    print("Initializing dashboard...")
    dashboard = DashboardApp(neural_child)
    
    # Run the dashboard server
    print(f"Dashboard running at http://localhost:{args.port}/")
    print("Press Ctrl+C to exit")
    dashboard.run_server(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()