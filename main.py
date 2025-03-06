"""
Main Entry Point

This is the main entry point for the NeuralChild system.
It provides a command-line interface for running different aspects of the system:
1. Running the dashboard
2. Accelerated development mode
3. Memory integration
4. Component registration
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("neuralchild")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import core components
from neuralchild.core.child import Child
from neuralchild.core.mother import Mother
from neuralchild.core.development import Development

# Import data types
from neuralchild.utils.data_types import (
    DevelopmentalStage, MotherPersonality, DevelopmentConfig,
    ChildState, SystemState
)

# Import components
from neuralchild.components.memory import MemorySystem
from neuralchild.components.language import LanguageComponent
from neuralchild.components.emotional import EmotionalComponent
from neuralchild.components.consciousness import ConsciousnessComponent
from neuralchild.components.social import SocialComponent
from neuralchild.components.cognitive import CognitiveComponent


def setup_system(
    load_state: Optional[str] = None,
    mother_personality: MotherPersonality = MotherPersonality.BALANCED,
    acceleration_factor: int = 100,
    random_seed: Optional[int] = None,
    start_age_months: int = 0
) -> Development:
    """
    Set up the NeuralChild development system.
    
    Args:
        load_state: Optional path to a saved state file
        mother_personality: Personality type for the Mother component
        acceleration_factor: Time acceleration factor
        random_seed: Optional random seed for reproducibility
        start_age_months: Starting age in months (only used for new systems)
        
    Returns:
        Initialized Development system
    """
    if load_state:
        logger.info(f"Loading system state from {load_state}")
        development_system = Development.load_system_state(
            filepath=load_state,
            mother_personality=mother_personality
        )
    else:
        logger.info("Creating new development system")
        
        # Create config
        config = DevelopmentConfig(
            time_acceleration_factor=acceleration_factor,
            random_seed=random_seed,
            mother_personality=mother_personality,
            start_age_months=start_age_months,
            enable_random_factors=True
        )
        
        # Create child and mother
        child = Child()
        mother = Mother(personality=mother_personality)
        
        # Create development system
        development_system = Development(
            child=child,
            mother=mother,
            config=config
        )
        
        # Register components
        register_components(development_system)
    
    return development_system


def register_components(development_system: Development):
    """
    Register all neural components with the Child's mind.
    
    Args:
        development_system: The Development system
    """
    child = development_system.child
    
    # Register memory system
    memory_system = MemorySystem(
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_indexes"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
        use_gpu="CUDA_AVAILABLE" in os.environ
    )
    child.register_component("memory_system", "memory", memory_system)
    
    # Register language component
    language_component = LanguageComponent()
    child.register_component("language_component", "language", language_component)
    
    # Register emotional component
    emotional_component = EmotionalComponent()
    child.register_component("emotional_component", "emotional", emotional_component)
    
    # Register consciousness component
    consciousness_component = ConsciousnessComponent()
    child.register_component("consciousness_component", "consciousness", consciousness_component)
    
    # Register social component
    social_component = SocialComponent()
    child.register_component("social_component", "social", social_component)
    
    # Register cognitive component
    cognitive_component = CognitiveComponent()
    child.register_component("cognitive_component", "cognitive", cognitive_component)
    
    logger.info("All components registered")


def run_dashboard():
    """Run the dashboard interface."""
    try:
        from neuralchild.dashboard.app import app, initialize_system
        
        host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
        port = int(os.getenv("DASHBOARD_PORT", "8050"))
        debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        
        # Initialize the system
        initialize_system()
        
        # Run the dashboard
        logger.info(f"Starting dashboard on {host}:{port}")
        app.run_server(host=host, port=port, debug=debug)
        
    except ImportError as e:
        logger.error(f"Failed to import dashboard components: {e}")
        logger.error("Make sure dash and its dependencies are installed")
        sys.exit(1)


def run_accelerated_development(
    months: int,
    load_state: Optional[str] = None,
    save_state: bool = True,
    mother_personality: str = "balanced"
):
    """
    Run accelerated development for a specified number of months.
    
    Args:
        months: Number of months to simulate
        load_state: Optional path to load initial state
        save_state: Whether to save the final state
        mother_personality: Personality type for the Mother
    """
    # Convert string personality to enum
    try:
        personality = MotherPersonality(mother_personality.lower())
    except ValueError:
        logger.error(f"Invalid mother personality: {mother_personality}")
        logger.error(f"Valid options: {[p.value for p in MotherPersonality]}")
        sys.exit(1)
    
    # Set up the system
    system = setup_system(
        load_state=load_state,
        mother_personality=personality,
        acceleration_factor=int(os.getenv("TIME_ACCELERATION_FACTOR", "100")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")) if os.getenv("RANDOM_SEED") else None
    )
    
    # Record initial state
    initial_stage = system.child.state.developmental_stage
    initial_age = int(system.child.state.simulated_age_months)
    logger.info(f"Initial state: {initial_stage.value} stage, {initial_age} months old")
    
    # Run accelerated development
    logger.info(f"Starting accelerated development for {months} months...")
    stages_progressed = system.accelerate_development(months)
    
    # Record final state
    final_stage = system.child.state.developmental_stage
    final_age = int(system.child.state.simulated_age_months)
    logger.info(f"Final state: {final_stage.value} stage, {final_age} months old")
    
    # Save the state if requested
    if save_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neuralchild_{final_stage.value}_{final_age}m_{timestamp}.json"
        save_path = os.path.join(os.getenv("STATES_PATH", "./data/states"), filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save state
        system.save_system_state(save_path)
        logger.info(f"State saved to {save_path}")


def run_interactive_mode(
    load_state: Optional[str] = None,
    mother_personality: str = "balanced"
):
    """
    Run an interactive console session with the NeuralChild.
    
    Args:
        load_state: Optional path to load initial state
        mother_personality: Personality type for the Mother
    """
    # Convert string personality to enum
    try:
        personality = MotherPersonality(mother_personality.lower())
    except ValueError:
        logger.error(f"Invalid mother personality: {mother_personality}")
        logger.error(f"Valid options: {[p.value for p in MotherPersonality]}")
        sys.exit(1)
    
    # Set up the system
    system = setup_system(
        load_state=load_state,
        mother_personality=personality
    )
    
    # Display initial state
    child = system.child
    stage = child.state.developmental_stage
    age_months = int(child.state.simulated_age_months)
    logger.info(f"Interactive mode started with child at {stage.value} stage ({age_months} months old)")
    
    # Welcome message
    print("\n===== NeuralChild Interactive Console =====")
    print(f"Child's developmental stage: {stage.value.replace('_', ' ').title()}")
    print(f"Child's age: {age_months} months")
    print("Type 'exit' to quit, 'help' for commands")
    print("===========================================\n")
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Check for special commands
            if user_input.lower() == 'exit':
                print("Exiting interactive mode.")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit - Exit interactive mode")
                print("  help - Show this help message")
                print("  save - Save current state")
                print("  info - Show information about the child")
                print("  accel <months> - Accelerate development by <months> months")
                print("  Any other input will be sent to the Mother and Child\n")
                continue
            elif user_input.lower() == 'save':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stage_value = child.state.developmental_stage.value
                age = int(child.state.simulated_age_months)
                filename = f"neuralchild_{stage_value}_{age}m_{timestamp}.json"
                save_path = os.path.join(os.getenv("STATES_PATH", "./data/states"), filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save state
                system.save_system_state(save_path)
                print(f"State saved to {save_path}")
                continue
            elif user_input.lower() == 'info':
                stage = child.state.developmental_stage
                age_months = int(child.state.simulated_age_months)
                vocab_size = child.state.metrics.vocabulary_size
                emotional_reg = child.state.metrics.emotional_regulation
                
                print("\nChild Information:")
                print(f"  Developmental Stage: {stage.value.replace('_', ' ').title()}")
                print(f"  Age: {age_months} months")
                print(f"  Vocabulary Size: {vocab_size}")
                print(f"  Emotional Regulation: {emotional_reg:.2f}")
                print(f"  Current Emotions: {', '.join([f'{e.type.value} ({e.intensity:.2f})' for e in child.state.current_emotional_state])}")
                continue
            elif user_input.lower().startswith('accel '):
                try:
                    months = int(user_input.split(' ')[1])
                    if months < 1:
                        print("Months must be at least 1")
                        continue
                        
                    print(f"Accelerating development by {months} months...")
                    initial_stage = child.state.developmental_stage
                    initial_age = int(child.state.simulated_age_months)
                    
                    system.accelerate_development(months)
                    
                    final_stage = child.state.developmental_stage
                    final_age = int(child.state.simulated_age_months)
                    
                    print(f"Development accelerated:")
                    print(f"  Age: {initial_age} months → {final_age} months")
                    print(f"  Stage: {initial_stage.value.replace('_', ' ').title()} → {final_stage.value.replace('_', ' ').title()}")
                    
                except (ValueError, IndexError):
                    print("Invalid format. Use 'accel <number>'")
                continue
            
            # Process regular input through the system
            if child.state.developmental_stage == DevelopmentalStage.INFANCY:
                # For infants, determine an initial vocalization
                initial_vocalization = child.state.current_emotional_state[0].type.value[:3].lower()
                child_response, mother_response = system.simulate_interaction(initial_vocalization=initial_vocalization)
            else:
                # For older children, use the text
                child_response, mother_response = system.simulate_interaction(initial_text=user_input)
            
            # Display responses
            print(f"\nMother: {mother_response.text}")
            
            if child_response.text:
                print(f"\nChild: {child_response.text}")
            else:
                print(f"\nChild: *{child_response.vocalization}*")
            
            # Update system time (in case of prolonged interaction)
            system.update_simulated_time()
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"An error occurred: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NeuralChild - Psychological Mind Simulation")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Dashboard mode
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the dashboard interface")
    
    # Development acceleration mode
    accel_parser = subparsers.add_parser("accelerate", help="Run accelerated development")
    accel_parser.add_argument("months", type=int, help="Number of months to simulate")
    accel_parser.add_argument("--load", type=str, help="Path to load initial state")
    accel_parser.add_argument("--no-save", action="store_true", help="Don't save the final state")
    accel_parser.add_argument("--personality", type=str, default="balanced",
                             help="Mother personality type (balanced, nurturing, authoritarian, permissive, neglectful)")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive console mode")
    interactive_parser.add_argument("--load", type=str, help="Path to load initial state")
    interactive_parser.add_argument("--personality", type=str, default="balanced",
                                  help="Mother personality type (balanced, nurturing, authoritarian, permissive, neglectful)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure data directories exist
    os.makedirs(os.getenv("STATES_PATH", "./data/states"), exist_ok=True)
    os.makedirs(os.getenv("FAISS_INDEX_PATH", "./data/faiss_indexes"), exist_ok=True)
    os.makedirs(os.getenv("VECTOR_DB_PATH", "./data/vector_db"), exist_ok=True)
    
    # Run the appropriate mode
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "accelerate":
        run_accelerated_development(
            months=args.months,
            load_state=args.load,
            save_state=not args.no_save,
            mother_personality=args.personality
        )
    elif args.mode == "interactive":
        run_interactive_mode(
            load_state=args.load,
            mother_personality=args.personality
        )
    else:
        # Default to dashboard if no mode specified
        run_dashboard()


if __name__ == "__main__":
    main() 