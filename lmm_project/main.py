#!/usr/bin/env python
"""
LMM Project - Main Entry Point

This script initializes and runs the Large Mind Model system with all cognitive modules.
It sets up the autonomous interaction between the Mother LLM and the developing mind.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style

# Initialize colorama for colored terminal output
colorama.init()

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import LMM components
from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message
from lmm_project.core.types import DevelopmentalStage
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.utils.llm_client import LLMClient, Message as LLMMessage
from lmm_project.utils.tts_client import TTSClient, play_audio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("lmm_main")

def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        sys.exit(1)

def init_mind(config, state_path=None):
    """Initialize the Mind with all modules"""
    # Create event bus and state manager
    event_bus = EventBus()
    state_manager = StateManager()
    
    # Load state if provided
    if state_path and os.path.exists(state_path):
        with open(state_path, "r") as f:
            state_data = json.load(f)
            state_manager.load_state(state_data)
            logger.info(f"Loaded state from {state_path}")
            
            # Extract age and stage from loaded state
            age = state_data.get("age", 0.0)
            stage = state_data.get("developmental_stage", "prenatal")
    else:
        # Start fresh
        age = 0.0
        stage = "prenatal"
        
    # Create Mind instance
    from lmm_project.core.mind import Mind
    mind = Mind(
        event_bus=event_bus,
        state_manager=state_manager,
        initial_age=age,
        developmental_stage=stage
    )
    
    # Initialize all cognitive modules
    mind.initialize_modules()
    
    return mind, event_bus, state_manager

def create_mother_interface(config, event_bus):
    """Create Mother LLM interface with TTS capabilities"""
    # Create LLM client
    llm_api_url = os.getenv("LLM_API_URL", config["apis"]["llm_api_url"])
    llm_client = LLMClient(base_url=llm_api_url)
    
    # Create TTS client
    tts_api_url = os.getenv("TTS_API_URL", config["apis"]["tts_api_url"])
    tts_client = TTSClient(base_url=tts_api_url)
    
    # Create Mother LLM
    mother_config = config["mother"]
    mother = MotherLLM(
        llm_client=llm_client,
        tts_client=tts_client,
        personality_traits=mother_config["personality"],
        teaching_style=mother_config["teaching_style"],
        voice=mother_config["voice"]
    )
    
    return mother

def save_state(mind, state_manager, config):
    """Save current state to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = mind.developmental_stage
    save_dir = Path(config["storage"]["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"mind_state_{stage}_{timestamp}.json"
    save_path = save_dir / filename
    
    # Get current state
    state = state_manager.get_state()
    
    # Add mind attributes
    state["age"] = mind.age
    state["developmental_stage"] = mind.developmental_stage
    state["cycle_count"] = mind.cycle_count
    
    # Save to file
    with open(save_path, "w") as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"Saved state to {save_path}")
    return save_path

def log_developmental_info(mind):
    """Log information about current developmental stage"""
    age = mind.age
    stage = mind.developmental_stage
    cycle = mind.cycle_count
    
    print(f"\n{Fore.CYAN}======= DEVELOPMENTAL INFO ======={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Age:{Style.RESET_ALL} {age:.3f}")
    print(f"{Fore.YELLOW}Stage:{Style.RESET_ALL} {stage}")
    print(f"{Fore.YELLOW}Cycle:{Style.RESET_ALL} {cycle}")
    
    # Log module development status
    print(f"\n{Fore.YELLOW}Module Development Status:{Style.RESET_ALL}")
    for module_name, module in mind.modules.items():
        dev_level = getattr(module, "development_level", 0.0)
        print(f"  - {module_name}: {dev_level:.2f}")
    print()

def autonomous_interaction_cycle(mind, mother, event_bus, state_manager, config, cycles=100):
    """Run autonomous interaction between Mother and LMM without user input"""
    last_save_cycle = 0
    save_interval = config["development"]["save_interval"]
    dev_rate = float(os.getenv("DEVELOPMENT_RATE", config["development"]["default_rate"]))
    
    print(f"\n{Fore.GREEN}Starting autonomous interaction cycle...{Style.RESET_ALL}")
    print(f"Development rate: {dev_rate} per cycle")
    print(f"Total cycles: {cycles}")
    print(f"Save interval: {save_interval} cycles")
    print(f"Mother personality: {config['mother']['personality']}")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        for cycle in range(cycles):
            # Log cycle information
            print(f"\n{Fore.MAGENTA}===== CYCLE {cycle+1}/{cycles} ====={Style.RESET_ALL}")
            log_developmental_info(mind)
            
            # Get mind state for Mother
            mind_state = {
                "age": mind.age,
                "developmental_stage": mind.developmental_stage,
                "modules": {name: {"development_level": getattr(module, "development_level", 0.0)} 
                           for name, module in mind.modules.items()},
                "emotional_state": state_manager.get_state("emotional_state") or "neutral",
                "concept_comprehension": state_manager.get_state("concept_comprehension") or {}
            }
            
            # Generate Mother's message based on current state
            current_focus = get_appropriate_focus(mind)
            
            # Create a message from the mind to the mother
            mind_message = get_mind_output(mind, event_bus, current_focus)
            
            # Mother processes and responds
            print(f"\n{Fore.CYAN}Mind:{Style.RESET_ALL} {mind_message}")
            
            # Generate response from Mother
            mother_response = mother.generate_response(mind_message, mind_state)
            
            # Display Mother's response
            print(f"{Fore.YELLOW}Mother:{Style.RESET_ALL} {mother_response['text']}")

            # # Play audio if available
            # if mother_response.get("audio_path") and os.path.exists(mother_response["audio_path"]):
            #     play_audio(mother_response["audio_path"])
            
            # Process Mother's response in the mind
            process_mother_response(mind, event_bus, mother_response["text"])
            
            # Update mind development
            mind.update_development(dev_rate)
            
            # Run a full mind cycle
            run_mind_cycle(mind, event_bus)
            
            # Save state at intervals
            if cycle % save_interval == 0 and cycle != last_save_cycle:
                save_state(mind, state_manager, config)
                last_save_cycle = cycle
            
            # Pause to allow reading
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interaction stopped by user{Style.RESET_ALL}")
    finally:
        # Save final state
        save_state(mind, state_manager, config)

def get_appropriate_focus(mind):
    """Determine appropriate focus for current developmental stage"""
    age = mind.age
    stage = mind.developmental_stage
    
    # Topics based on developmental stage
    focus_topics = {
        "prenatal": ["basic patterns", "sensory input", "neural activity"],
        "infant": ["objects", "colors", "sounds", "simple language"],
        "child": ["concepts", "relationships", "simple reasoning", "language"],
        "adolescent": ["abstract thinking", "identity", "complex reasoning"],
        "adult": ["philosophy", "creativity", "integration", "deep reasoning"]
    }
    
    # Select appropriate focus
    import random
    if stage in focus_topics:
        return random.choice(focus_topics[stage])
    return "general development"

def get_mind_output(mind, event_bus, current_focus):
    """Generate output from the mind based on current state"""
    # Early stages have simple, limited output
    age = mind.age
    stage = mind.developmental_stage
    
    # Very early stages just have simple pattern recognition
    if stage == "prenatal":
        return "..." if age < 0.05 else "? ..."
    
    # Generate message based on activated modules
    if stage == "infant":
        # Simple single word or short phrases
        simple_outputs = [
            "see", "hear", "feel", "what?", "more", "again", "light", "dark",
            "sound nice", "see thing", "hear voice", "feel good"
        ]
        import random
        return random.choice(simple_outputs)
    
    # More complex output as development progresses
    message = Message(
        sender="main",
        message_type="output_request",
        content={"focus": current_focus}
    )
    event_bus.publish(message)
    
    # For demonstration, simulate mind output
    # In a complete implementation, we would wait for a response from language module
    if stage == "child":
        return get_child_output(mind, current_focus)
    elif stage == "adolescent":
        return get_adolescent_output(mind, current_focus)
    else:
        return get_adult_output(mind, current_focus)

def get_child_output(mind, focus):
    """Generate child-like output"""
    import random
    outputs = [
        f"What is {focus}?",
        f"Tell me about {focus}.",
        f"I want to learn {focus}.",
        f"How does {focus} work?",
        f"I see {focus}. What is it?",
        f"Can you explain {focus}?",
        f"I like {focus}. More please."
    ]
    return random.choice(outputs)

def get_adolescent_output(mind, focus):
    """Generate adolescent-like output"""
    import random
    outputs = [
        f"I've been thinking about {focus}. How does it relate to other concepts?",
        f"What are the principles behind {focus}?",
        f"Can you explain how {focus} connects to the bigger picture?",
        f"I'm curious about the different aspects of {focus}.",
        f"What's your perspective on {focus}?",
        f"How do different people view {focus}?",
        f"I want to understand {focus} more deeply."
    ]
    return random.choice(outputs)

def get_adult_output(mind, focus):
    """Generate adult-like output"""
    import random
    outputs = [
        f"I'm exploring the philosophical implications of {focus}. What perspectives should I consider?",
        f"How does {focus} integrate with other domains of knowledge?",
        f"I'm analyzing the nuances of {focus}. What subtle aspects might I be missing?",
        f"What are the most complex aspects of {focus} that we should discuss?",
        f"I'm synthesizing information about {focus}. What connections might be valuable to explore?",
        f"Let's discuss the deeper implications of {focus} and how it shapes understanding.",
        f"I'm interested in the theoretical frameworks that best explain {focus}."
    ]
    return random.choice(outputs)

def process_mother_response(mind, event_bus, response_text):
    """Process Mother's response in the mind"""
    # Create perception message
    perception_msg = Message(
        sender="mother_interface",
        message_type="perception_input",
        content={
            "modality": "language",
            "content": response_text,
            "source": "mother",
            "timestamp": datetime.now().isoformat()
        }
    )
    event_bus.publish(perception_msg)
    
    # Create emotional response (simplified)
    import random
    valence = random.uniform(0.3, 0.8)  # Positive bias for mother's responses
    arousal = random.uniform(0.2, 0.6)
    
    emotion_msg = Message(
        sender="emotion_processor",
        message_type="emotional_response",
        content={
            "valence": valence,
            "arousal": arousal,
            "source_event": "mother_communication",
            "timestamp": datetime.now().isoformat()
        }
    )
    event_bus.publish(emotion_msg)

def run_mind_cycle(mind, event_bus):
    """Run a full processing cycle in the mind"""
    # Signal the start of a cycle
    cycle_start_msg = Message(
        sender="main",
        message_type="system_cycle_start",
        content={
            "cycle_number": mind.cycle_count + 1,
            "timestamp": datetime.now().isoformat()
        }
    )
    event_bus.publish(cycle_start_msg)
    
    # Allow modules to process
    time.sleep(0.5)  # Simulate processing time
    
    # Signal the end of a cycle
    cycle_end_msg = Message(
        sender="main",
        message_type="system_cycle_complete",
        content={
            "cycle_number": mind.cycle_count + 1,
            "timestamp": datetime.now().isoformat()
        }
    )
    event_bus.publish(cycle_end_msg)

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LMM Project - Large Mind Model")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--load-state", type=str, help="Path to state file to load")
    parser.add_argument("--cycles", type=int, help="Number of development cycles to run")
    parser.add_argument("--development-rate", type=float, help="Development rate per cycle")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.cycles:
        config["development"]["default_cycles"] = args.cycles
    if args.development_rate:
        config["development"]["default_rate"] = args.development_rate
        
    # Initialize Mind with all modules
    mind, event_bus, state_manager = init_mind(config, args.load_state)
    
    # Create Mother interface
    mother = create_mother_interface(config, event_bus)
    
    # Print header
    print(f"{Fore.GREEN}=========================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}  Large Mind Model - System Initialized  {Style.RESET_ALL}")
    print(f"{Fore.GREEN}=========================================={Style.RESET_ALL}")
    print(f"Development Stage: {mind.developmental_stage}")
    print(f"Active Modules: {', '.join(mind.modules.keys())}")
    
    # Run autonomous interaction cycle
    cycles = config["development"]["default_cycles"]
    autonomous_interaction_cycle(mind, mother, event_bus, state_manager, config, cycles)
    
    # Print footer
    print(f"\n{Fore.GREEN}=========================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}  LMM Session Complete  {Style.RESET_ALL}")
    print(f"{Fore.GREEN}=========================================={Style.RESET_ALL}")
    print(f"Final Age: {mind.age:.3f}")
    print(f"Final Stage: {mind.developmental_stage}")
    print(f"Total Cycles: {mind.cycle_count}")

if __name__ == "__main__":
    main()
