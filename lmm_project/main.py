"""
Main entry point for the Large Mind Model (LMM) project.

This script initializes and runs the LMM simulation with all implemented modules.
It provides a "run and watch" experience where the Mother LLM nurtures and teaches
the developing mind, with no user interaction during the developmental process.
"""
import os
import time
import logging
import threading
import sys
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, Any, Optional, List

# Core components
from lmm_project.core.event_bus import get_event_bus, EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.mind import Mind
from lmm_project.core.message import Message, MessageType

# Development components
from lmm_project.development import DevelopmentManager, create_development_manager
from lmm_project.development.models import DevelopmentConfig, StageDefinition, StageRange, MilestoneDefinition, CriticalPeriodDefinition, DevelopmentalStage, CriticalPeriodType

# Memory components (now improved)
from lmm_project.modules.memory import initialize as initialize_memory
from lmm_project.modules.memory.models import MemoryConfig

# Mother LLM interface
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.interfaces.mother.models import TeachingMethod

# Storage and Utilities
from lmm_project.storage.state_persistence import StatePersistence
from lmm_project.utils.logging_utils import setup_system_logging, get_module_logger
from lmm_project.utils.tts_client import text_to_speech
from lmm_project.utils.config_manager import load_config, ConfigManager, get_config

# Setup logging
setup_system_logging()
logger = get_module_logger("main")

class LMMSimulation:
    """
    Main controller for the LMM simulation.
    
    Manages the initialization, running, and coordination of all components
    of the Large Mind Model simulation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the LMM simulation.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = get_module_logger("lmm.main")
        self.logger.info("Initializing LMM simulation")
        self.logger.info(f"Using configuration from: {config_path}")
        
        # Store config path
        self.config_path = config_path
        
        # Use absolute path for config file
        if not os.path.isabs(config_path):
            # Try different locations for the config file
            possible_paths = [
                # Current directory
                config_path,
                # lmm_project directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path),
                # Parent directory of lmm_project
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path),
                # lmm_project/lmm_project directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lmm_project", config_path)
            ]
            
            # Find the first path that exists
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize core components
        self.event_bus = get_event_bus()
        
        # Setup storage
        storage_dir = self.config.get("storage_dir", "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize state manager with storage directory
        self.state_manager = StateManager(os.path.join(storage_dir, "states"))
        self.state_persistence = StatePersistence(storage_dir)
        
        # Initialize development
        dev_config = self._create_development_config()
        self.development_manager = create_development_manager(dev_config, self.event_bus)
        
        # Initialize memory with developmental integration
        memory_config = self._create_memory_config()
        initialize_memory(
            event_bus=self.event_bus,
            config=memory_config,
            developmental_age=self.development_manager.get_age()
        )
        
        # Initialize mother LLM
        self.mother = MotherLLM(
            personality_preset=self.config.get("mother", {}).get("personality", "nurturing"),
            teaching_preset="scaffolding",
            use_tts=self.config.get("mother", {}).get("use_tts", True),
            tts_voice=self.config.get("mother", {}).get("voice", "en-US-JennyNeural")
        )
        
        # Initialize the mind itself
        self.mind = Mind(
            config_path=self.config_path,
            storage_dir=self.config.get("storage", {}).get("directory", "storage")
        )
        
        # Control variables
        self.running = False
        self.paused = False
        self.current_step = 0
        self.last_update_time = time.time()
        self.last_save_time = time.time()
        self.last_interaction_time = time.time()
        
        # Initialize visualization if enabled
        self.visualization_enabled = self.config.get("visualization", {}).get("enabled", True)
        if self.visualization_enabled:
            self.logger.info("Visualization enabled, initializing dashboard")
            # Dashboard initialization would go here
            # initialize_dashboard()
        else:
            self.logger.info("Visualization disabled")
        
        # Setup event handlers
        self._register_event_handlers()
        
        logger.info("LMM simulation initialized successfully")
    
    def _create_development_config(self) -> DevelopmentConfig:
        """Create development configuration from loaded config."""
        dev_conf = self.config.get("development", {})
        
        # Create default stage definitions
        stage_definitions = [
            StageDefinition(
                stage=DevelopmentalStage.PRENATAL,
                range=StageRange(min_age=0.0, max_age=0.1),
                description="Neural foundation formation stage",
                learning_rate_multiplier=0.8,
                key_capabilities=["basic pattern recognition", "sensory processing"]
            ),
            StageDefinition(
                stage=DevelopmentalStage.INFANT,
                range=StageRange(min_age=0.1, max_age=1.0),
                description="Rapid early learning stage",
                learning_rate_multiplier=1.5,
                key_capabilities=["object permanence", "basic communication", "motor control"]
            ),
            StageDefinition(
                stage=DevelopmentalStage.CHILD,
                range=StageRange(min_age=1.0, max_age=3.0),
                description="Exploratory learning stage",
                learning_rate_multiplier=1.2,
                key_capabilities=["language acquisition", "symbolic thinking", "social interaction"]
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADOLESCENT,
                range=StageRange(min_age=3.0, max_age=5.0),
                description="Abstract reasoning development stage",
                learning_rate_multiplier=1.0,
                key_capabilities=["abstract reasoning", "complex problem solving", "identity formation"]
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADULT,
                range=StageRange(min_age=5.0, max_age=None),
                description="Mature cognitive stage",
                learning_rate_multiplier=0.7,
                key_capabilities=["expertise development", "wisdom", "metacognition"]
            )
        ]
        
        # Create minimal milestone and critical period definitions
        milestone_definitions = [
            MilestoneDefinition(
                id="first_memory",
                name="First Memory Formation",
                description="Formation of the first persistent memory",
                typical_stage=DevelopmentalStage.INFANT,
                typical_age=0.2,
                importance=1.0
            )
        ]
        
        critical_period_definitions = [
            CriticalPeriodDefinition(
                id="language_acquisition",
                name="Language Acquisition Period",
                period_type=CriticalPeriodType.LANGUAGE,
                description="Critical period for language acquisition",
                begin_age=0.5,
                end_age=2.0,
                learning_multiplier=2.0,
                affected_modules=["language"],
                affected_capabilities=["vocabulary", "grammar", "syntax"]
            )
        ]
        
        return DevelopmentConfig(
            initial_age=dev_conf.get("initial_age", 0.0),
            time_acceleration=dev_conf.get("time_acceleration", 1000.0),
            stage_definitions=stage_definitions,
            milestone_definitions=milestone_definitions,
            critical_period_definitions=critical_period_definitions
        )
    
    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration from loaded config."""
        mem_conf = self.config.get("memory", {})
        return MemoryConfig(
            base_working_memory_capacity=mem_conf.get("base_working_memory_capacity", 5),
            consolidation_threshold=mem_conf.get("consolidation_threshold", 0.6),
            base_decay_rate=mem_conf.get("base_decay_rate", 0.05),
            embedding_dimension=mem_conf.get("embedding_dimension", 768),
            use_neural_networks=mem_conf.get("use_neural_networks", True),
            vector_store_gpu_enabled=mem_conf.get("vector_store_gpu_enabled", True)
        )
    
    def _register_event_handlers(self) -> None:
        """Register handlers for simulation events."""
        self.event_bus.subscribe("developmental_milestone_reached", self._handle_milestone)
        self.event_bus.subscribe("message", self._handle_message)
        self.event_bus.subscribe("critical_period_activated", self._handle_critical_period)
        self.event_bus.subscribe("stage_transition", self._handle_stage_transition)
    
    def _handle_milestone(self, event: Dict[str, Any]) -> None:
        """Handle developmental milestone events."""
        milestone_data = event.get("data", {})
        milestone_id = milestone_data.get("milestone_id")
        milestone_name = milestone_data.get("name", "Unknown milestone")
        
        logger.info(f"Milestone reached: {milestone_name}")
        
        # Generate mother response to milestone
        response = self.mother.respond_to_milestone(milestone_id, milestone_data)
        if response:
            logger.info(f"Mother's response: {response}")
            # Speak the response if TTS is enabled
            if self.config.get("tts", {}).get("enabled", True):
                self._speak_text(response, "milestone")
    
    def _handle_message(self, event: Dict[str, Any]) -> None:
        """Handle message events."""
        message_data = event.get("data", {}).get("message", {})
        if not message_data:
            return
            
        message_type = message_data.get("type")
        source = message_data.get("source", "")
        content = message_data.get("content", {})
        
        # Only process certain message types
        if message_type == MessageType.DEVELOPMENT or message_type == MessageType.MEMORY:
            if "mind" in source:
                # Process mind messages specially - these might trigger mother interaction
                self._process_mind_message(content)
    
    def _handle_critical_period(self, event: Dict[str, Any]) -> None:
        """Handle critical period activation."""
        period_data = event.get("data", {})
        period_name = period_data.get("period_name", "Unknown period")
        period_type = period_data.get("period_type", "Unknown type")
        
        logger.info(f"Critical period activated: {period_name} ({period_type})")
        
        # Have mother respond to the critical period
        response = self.mother.respond_to_critical_period(period_data)
        if response:
            logger.info(f"Mother's response to critical period: {response}")
            if self.config.get("tts", {}).get("enabled", True):
                self._speak_text(response, "critical_period")
    
    def _handle_stage_transition(self, event: Dict[str, Any]) -> None:
        """Handle developmental stage transitions."""
        stage_data = event.get("data", {})
        new_stage = stage_data.get("new_stage", "Unknown stage")
        old_stage = stage_data.get("old_stage", "Unknown stage")
        
        logger.info(f"Stage transition: {old_stage} -> {new_stage}")
        
        # Save state on stage transition
        self._save_state()
        
        # Generate mother response
        response = self.mother.respond_to_stage_transition(old_stage, new_stage)
        if response:
            logger.info(f"Mother's response to stage transition: {response}")
            if self.config.get("tts", {}).get("enabled", True):
                self._speak_text(response, "stage_transition")
    
    def _process_mind_message(self, content: Dict[str, Any]) -> None:
        """Process messages from the mind that might trigger mother interaction."""
        # Check if we should interact based on time since last interaction
        current_time = time.time()
        min_interaction_interval = self.config.get("mother", {}).get("min_interaction_interval_sec", 300)
        
        if current_time - self.last_interaction_time < min_interaction_interval:
            return
            
        # Decide if the mother should spontaneously interact
        interaction_probability = self.config.get("mother", {}).get("spontaneous_interaction_probability", 0.1)
        if np.random.random() < interaction_probability:
            # Generate mother interaction
            interaction = self.mother.generate_spontaneous_interaction(self.development_manager.get_age())
            if interaction:
                logger.info(f"Mother spontaneous interaction: {interaction}")
                if self.config.get("tts", {}).get("enabled", True):
                    self._speak_text(interaction, "spontaneous")
                
                self.last_interaction_time = current_time
    
    def _speak_text(self, text: str, interaction_type: str) -> None:
        """Speak text using TTS service."""
        try:
            tts_config = self.config.get("tts", {})
            voice = tts_config.get("voice", "af_nicole")
            speed = tts_config.get("speed", 1.0)
            
            # Different voices/speeds for different types of interactions
            if interaction_type == "milestone":
                speed = 0.9  # Slightly slower for important milestones
            elif interaction_type == "critical_period":
                voice = tts_config.get("emphasis_voice", voice)
            
            # Actually call TTS
            result = text_to_speech(
                text=text,
                voice=voice,
                speed=speed,
                auto_play=True
            )
            
            logger.debug(f"TTS output: {result.get('audio_path', 'No audio path')}")
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
    
    def _save_state(self) -> None:
        """Save the current state of the system."""
        try:
            # Get state from components
            mind_state = self.mind.get_state()
            developmental_state = self.development_manager.get_state()
            
            # Combine states
            state = {
                "timestamp": datetime.now().isoformat(),
                "developmental_age": self.development_manager.get_age(),
                "mind_state": mind_state,
                "developmental_state": developmental_state,
                "simulation_step": self.current_step
            }
            
            # Save to file
            filename = f"state_{self.current_step}_{self.development_manager.get_age():.2f}.json"
            self.state_persistence.save_state(state, filename)
            
            logger.info(f"State saved to {filename}")
            self.last_save_time = time.time()
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _should_save_state(self) -> bool:
        """Determine if state should be saved based on config and time elapsed."""
        current_time = time.time()
        save_interval = self.config.get("system", {}).get("save_state_interval_sec", 300)
        return (current_time - self.last_save_time) >= save_interval
    
    def _update_visualization(self) -> None:
        """Update visualization dashboard if enabled."""
        if not self.visualization_enabled:
            return
            
        try:
            # Gather data for visualization
            data = {
                "developmental_age": self.development_manager.get_age(),
                "current_stage": self.development_manager.get_current_stage(),
                "active_milestones": self.development_manager.get_active_milestones(),
                "completed_milestones": self.development_manager.get_completed_milestones(),
                "mind_state": self.mind.get_state(),
                "simulation_step": self.current_step,
                "last_mother_interaction": self.last_interaction_time
            }
            
            # Log visualization data instead of updating dashboard
            logger.info(f"Visualization data: {data}")
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
    
    def start(self) -> None:
        """Start the LMM simulation."""
        logger.info("Starting LMM simulation")
        self.running = True
        self.event_bus.start()
        
        # Welcome message from mother
        welcome = self.mother.generate_welcome()
        logger.info(f"Mother's welcome: {welcome}")
        if self.config.get("tts", {}).get("enabled", True):
            self._speak_text(welcome, "welcome")
        
        try:
            # Main simulation loop
            while self.running:
                self._run_step()
                
                # Check if we should exit based on max age or steps
                max_age = self.config.get("development", {}).get("max_age", float('inf'))
                max_steps = self.config.get("system", {}).get("max_steps", float('inf'))
                
                if (self.development_manager.get_age() >= max_age or 
                    self.current_step >= max_steps):
                    logger.info(f"Simulation complete: Age={self.development_manager.get_age():.2f}, Steps={self.current_step}")
                    break
                
                # Sleep to control simulation speed
                step_interval = self.config.get("system", {}).get("step_interval_sec", 0.1)
                time.sleep(step_interval)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self._cleanup()
    
    def _run_step(self) -> None:
        """Run a single simulation step."""
        if self.paused:
            return
            
        # Update development
        self.development_manager.update()
        
        # Check for time to save state
        if self._should_save_state():
            self._save_state()
        
        # Update visualization
        self._update_visualization()
        
        # Count step
        self.current_step += 1
        
        # Log occasional status
        if self.current_step % 100 == 0:
            logger.info(f"Simulation step {self.current_step}, Age: {self.development_manager.get_age():.2f}")
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.paused = True
        logger.info("Simulation paused")
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.paused = False
        logger.info("Simulation resumed")
    
    def _cleanup(self) -> None:
        """Clean up resources before exiting."""
        # Save final state
        self._save_state()
        
        # Stop event bus
        self.event_bus.stop()
        
        logger.info("Simulation cleaned up and ready to exit")

def main():
    """Main entry point for the LMM simulation."""
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
    
    # Check command line arguments
    config_path = "config.yml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and start simulation
    simulation = LMMSimulation(config_path)
    simulation.start()

if __name__ == "__main__":
    main()
