"""
Application entry point for the Large Mind Model (LMM) project.

This script provides a user-friendly interface for starting and
controlling the LMM simulation with various configuration options.
"""

import sys
import argparse
from pathlib import Path
from lmm_project.main import LMMSimulation

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the Large Mind Model (LMM) simulation"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yml",
        help="Path to configuration file (default: config.yml)"
    )
    
    parser.add_argument(
        "--tts-disabled",
        action="store_true",
        help="Disable text-to-speech for Mother interactions"
    )
    
    parser.add_argument(
        "--visualization-disabled",
        action="store_true",
        help="Disable visualization dashboard"
    )
    
    parser.add_argument(
        "--max-age",
        type=float,
        help="Maximum developmental age to simulate to (overrides config)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of simulation steps (overrides config)"
    )
    
    parser.add_argument(
        "--initial-age",
        type=float,
        default=0.0,
        help="Initial developmental age (default: 0.0)"
    )
    
    parser.add_argument(
        "--acceleration",
        type=float,
        help="Time acceleration factor (overrides config)"
    )
    
    parser.add_argument(
        "--mother-personality",
        type=str,
        choices=["nurturing", "analytical", "creative", "balanced"],
        help="Mother's personality type (overrides config)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point with command-line argument handling."""
    # Parse arguments
    args = parse_arguments()
    
    # Print welcome message
    print("""
    ┌─────────────────────────────────────────────┐
    │                                             │
    │        Large Mind Model Simulation          │
    │        -------------------------           │
    │                                             │
    │ A developmental cognitive architecture      │
    │ that learns through nurturing interaction.  │
    │                                             │
    │ This is a watch-only experience during      │
    │ development. Press Ctrl+C to stop.          │
    │                                             │
    └─────────────────────────────────────────────┘
    """)
    
    # Load the base configuration
    simulation = LMMSimulation(args.config)
    
    # Apply command-line overrides to configuration
    if args.tts_disabled:
        simulation.config["tts"]["enabled"] = False
        print("* Text-to-speech disabled")
    
    if args.visualization_disabled:
        simulation.config["visualization"]["enable_dashboard"] = False
        print("* Visualization dashboard disabled")
    
    if args.max_age is not None:
        simulation.config["development"]["max_age"] = args.max_age
        print(f"* Maximum developmental age set to: {args.max_age}")
    
    if args.max_steps is not None:
        simulation.config["system"]["max_steps"] = args.max_steps
        print(f"* Maximum steps set to: {args.max_steps}")
    
    if args.initial_age != 0.0:
        simulation.config["development"]["initial_age"] = args.initial_age
        # Need to reset the development manager with new age
        simulation.development_manager.set_age(args.initial_age)
        print(f"* Initial developmental age set to: {args.initial_age}")
    
    if args.acceleration is not None:
        simulation.config["development"]["time_acceleration"] = args.acceleration
        simulation.development_manager.set_time_acceleration(args.acceleration)
        print(f"* Time acceleration factor set to: {args.acceleration}")
    
    if args.mother_personality is not None:
        simulation.config["mother"]["personality"] = args.mother_personality
        # Update mother personality
        simulation.mother.set_personality(args.mother_personality)
        print(f"* Mother's personality set to: {args.mother_personality}")
    
    print("\nStarting simulation... (Press Ctrl+C to stop)")
    print("─" * 50)
    
    # Start the simulation
    simulation.start()

if __name__ == "__main__":
    main()
