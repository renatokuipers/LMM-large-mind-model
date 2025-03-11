"""
Main entry point for the LMM system.
Provides autonomous interaction between Mother and LMM cognitive system.
"""
import os
import sys
import time
import threading
import signal
from pathlib import Path
from typing import Optional

from lmm_project.core.event_bus import get_event_bus
from lmm_project.core.state_manager import get_state_manager
from lmm_project.core.mind import get_mind
from lmm_project.utils.config_manager import get_config
from lmm_project.utils.logging_utils import setup_system_logging, get_module_logger
from lmm_project.interfaces.mother import get_mother, create_mother_input
from lmm_project.modules import memory, perception, attention
from lmm_project.development import (
    create_development_manager,
    get_development_manager,
    DevelopmentConfig
)
from lmm_project.utils.tts_client import text_to_speech, get_output_path
from lmm_project.utils.audio_extraction import extract_features_from_file
from lmm_project.storage.experience_logger import ExperienceLogger
from lmm_project.development.models import (
    StageDefinition,
    MilestoneDefinition,
    CriticalPeriodDefinition,
    StageRange,
    DevelopmentalStage,
    CriticalPeriodType,
    GrowthRateModel
)
from lmm_project.core.message import TextContent, Message
from lmm_project.core.types import MessageType
from lmm_project.modules.module_type import ModuleType

# Initialize logger
logger = get_module_logger("main")

class LMMSystem:
    """Main system coordinator for autonomous LMM-Mother interaction."""
    
    def __init__(self):
        """Initialize the LMM system components."""
        # Load configuration
        self.config = get_config()
        self.config_path = self.config.config_path
        
        # Initialize core components first
        self._initialize_core_systems()
        
        # Initialize development system
        self._initialize_development_system()
        
        # Initialize mind after development system
        self._initialize_mind()
        
        # Initialize cognitive modules before mother
        self._initialize_cognitive_modules()
        
        # Initialize mother interface after cognitive modules
        self._initialize_mother_interface()
        
        # Initialize support systems
        self._initialize_support_systems()
        
        # System state
        self.running = False
        self.interaction_count = 0
        self.development_thread = None
        self.state_saving_thread = None
        self.monitoring_thread = None
        
        logger.info("LMM System initialized")
    
    def _initialize_core_systems(self) -> None:
        """Initialize and start core system components."""
        # Set up event bus first and ensure it's running
        self.event_bus = get_event_bus()
        self.event_bus.start()
        time.sleep(0.1)  # Small delay to ensure event bus is fully started
        
        # Set up state manager
        self.state_manager = get_state_manager()
        
        logger.info("Core systems initialized and started")
    
    def _initialize_development_system(self) -> None:
        """Initialize the development tracking system."""
        # Create growth rate model with appropriate scaling
        growth_rate_model = GrowthRateModel(
            base_rate=self.config.get_float("development.base_rate", 0.001),  # Slower base rate
            stage_multipliers={
                DevelopmentalStage.PRENATAL: 1.0,
                DevelopmentalStage.INFANT: 1.2,
                DevelopmentalStage.CHILD: 1.5,
                DevelopmentalStage.ADOLESCENT: 1.3,
                DevelopmentalStage.ADULT: 1.0
            },
            critical_period_boost=1.5,
            practice_effect=1.2,
            plateau_threshold=0.9,
            plateau_factor=0.5
        )

        # Define developmental stages with appropriate age ranges
        stage_definitions = [
            StageDefinition(
                stage=DevelopmentalStage.PRENATAL,
                range=StageRange(min_age=0.0, max_age=0.1),
                description="Initial formation stage",
                learning_rate_multiplier=0.5,
                key_capabilities=["basic_perception", "reflexive_response"],
                neural_characteristics={"plasticity": 0.9, "connection_density": 0.3}
            ),
            StageDefinition(
                stage=DevelopmentalStage.INFANT,
                range=StageRange(min_age=0.1, max_age=1.0),
                description="Early learning stage",
                learning_rate_multiplier=1.0,
                key_capabilities=["pattern_recognition", "simple_memory"],
                neural_characteristics={"plasticity": 0.8, "connection_density": 0.4}
            ),
            StageDefinition(
                stage=DevelopmentalStage.CHILD,
                range=StageRange(min_age=1.0, max_age=3.0),
                description="Active learning stage",
                learning_rate_multiplier=1.5,
                key_capabilities=["language_processing", "working_memory"],
                neural_characteristics={"plasticity": 0.7, "connection_density": 0.5}
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADOLESCENT,
                range=StageRange(min_age=3.0, max_age=6.0),
                description="Advanced development stage",
                learning_rate_multiplier=1.2,
                key_capabilities=["abstract_thinking", "complex_memory"],
                neural_characteristics={"plasticity": 0.6, "connection_density": 0.6}
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADULT,
                range=StageRange(min_age=6.0, max_age=None),
                description="Mature operation stage",
                learning_rate_multiplier=1.0,
                key_capabilities=["self_regulation", "meta_learning"],
                neural_characteristics={"plasticity": 0.5, "connection_density": 0.7}
            )
        ]

        # Define milestones
        milestone_definitions = [
            MilestoneDefinition(
                id="basic_perception",
                name="Basic Perception",
                description="Ability to process basic sensory input",
                typical_stage=DevelopmentalStage.PRENATAL,
                typical_age=0.05,
                prerequisite_milestones=[],
                module_dependencies=["perception"],
                importance=1.0
            ),
            MilestoneDefinition(
                id="pattern_recognition",
                name="Pattern Recognition",
                description="Recognition of simple patterns in input",
                typical_stage=DevelopmentalStage.INFANT,
                typical_age=0.5,
                prerequisite_milestones=["basic_perception"],
                module_dependencies=["perception", "attention"],
                importance=1.2
            ),
            # Add more milestones as needed
        ]

        # Define critical periods
        critical_period_definitions = [
            CriticalPeriodDefinition(
                id="early_perception",
                name="Early Perception Development",
                period_type=CriticalPeriodType.SENSORY,
                description="Critical period for sensory processing development",
                begin_age=0.0,
                end_age=0.5,
                learning_multiplier=2.0,
                affected_modules=["perception"],
                affected_capabilities=["pattern_recognition", "sensory_processing"]
            ),
            CriticalPeriodDefinition(
                id="language_acquisition",
                name="Language Acquisition Period",
                period_type=CriticalPeriodType.LANGUAGE,
                description="Critical period for language learning",
                begin_age=0.5,
                end_age=2.0,
                learning_multiplier=2.5,
                affected_modules=["language", "memory"],
                affected_capabilities=["language_processing", "semantic_memory"]
            ),
            # Add more critical periods as needed
        ]

        # Create development config with adjusted timing
        dev_config = DevelopmentConfig(
            initial_age=self.config.get_float("development.initial_age", 0.0),
            time_acceleration=self.config.get_float("development.time_acceleration", 1000.0),
            stage_definitions=stage_definitions,
            milestone_definitions=milestone_definitions,
            critical_period_definitions=critical_period_definitions,
            growth_rate_model=growth_rate_model,
            enable_variability=self.config.get_boolean("development.enable_variability", True),
            variability_factor=self.config.get_float("development.variability_factor", 0.2)
        )

        # Create the development manager instance
        self.development_manager = create_development_manager(
            config=dev_config,
            event_bus=self.event_bus
        )

        logger.info(f"Development system initialized with age {dev_config.initial_age}")
    
    def _initialize_cognitive_modules(self) -> None:
        """Initialize cognitive modules in the correct order."""
        # Get initial developmental age
        initial_age = self.config.get_float("development.initial_age", 0.0)
        
        # Create initialization message
        init_message = Message(
            sender="system",
            sender_type=ModuleType.EXECUTIVE,
            message_type=MessageType.SYSTEM_STATUS,
            content=TextContent(
                content_type="text",
                data={
                    "action": "initialize",
                    "developmental_age": initial_age,
                    "config": self.config.config
                }
            )
        )
        
        # Initialize modules with event bus and developmental age
        # Ensure event bus is running first
        if not self.event_bus.running:
            self.event_bus.start()
            time.sleep(0.1)  # Small delay to ensure event bus is running
        
        # Initialize modules in order
        perception.initialize(self.event_bus, developmental_age=initial_age)
        self.event_bus.publish(init_message)
        time.sleep(0.1)
        
        attention.initialize(self.event_bus, developmental_age=initial_age)
        self.event_bus.publish(init_message)
        time.sleep(0.1)
        
        memory.initialize(self.event_bus, developmental_age=initial_age)
        self.event_bus.publish(init_message)
        time.sleep(0.1)
        
        logger.info("Cognitive modules initialized")
    
    def _initialize_mother_interface(self) -> None:
        """Initialize the mother interface."""
        self.mother = get_mother(
            personality_preset=self.config.get_string("interfaces.mother.personality", "nurturing"),
            teaching_preset=self.config.get_string("interfaces.mother.teaching_preset", "balanced"),
            use_tts=True,
            tts_voice=self.config.get_string("interfaces.mother.tts_voice", "af_bella")
        )
    
    def _initialize_support_systems(self) -> None:
        """Initialize support systems like logging and monitoring."""
        # Set up experience logging
        self.experience_logger = ExperienceLogger()
        
        # Initialize state saving and monitoring
        self._initialize_state_saving()
        self._monitor_system_status()
        
        # Set up signal handlers
        self._setup_signal_handlers()
    
    def run(self) -> None:
        """Run the autonomous interaction loop."""
        self.running = True
        logger.info("Starting autonomous interaction loop")
        
        try:
            # Start development thread first
            self.development_thread = threading.Thread(
                target=self._development_timer,
                daemon=True,
                name="DevelopmentThread"
            )
            self.development_thread.start()
            
            # Small delay to allow development system to initialize
            time.sleep(0.2)
            
            # Initial mother greeting
            self._process_mother_response(self.mother.generate_welcome())
            
            # Main interaction loop
            while self.running:
                self._interaction_cycle()
                time.sleep(0.1)  # Prevent tight loop
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        finally:
            self._cleanup()
    
    def _interaction_cycle(self) -> None:
        """Execute one full interaction cycle."""
        try:
            # Get current developmental age
            current_age = self.development_manager.get_age()
            
            # Get system state
            system_state = self.mind.get_state()
            
            # Create mother input from system state
            mother_input = create_mother_input(
                content=str(system_state),
                age=current_age
            )
            
            # Get mother's response
            response = self.mother.respond(mother_input)
            
            # Process the response
            self._process_mother_response(response.content)
            
            # Update interaction count
            self.interaction_count += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in interaction cycle: {str(e)}", exc_info=True)
    
    def _process_mother_response(self, text: str) -> None:
        """Process mother's text response through TTS and perception."""
        from lmm_project.core.message import TextContent, Message
        from lmm_project.core.types import MessageType
        
        try:
            # Print the response
            print(f"\nMother: {text}\n")
            
            # Generate and play audio with blocking
            audio_result = text_to_speech(
                text=text,
                voice=self.config.get_string("interfaces.mother.tts_voice", "af_bella"),
                auto_play=False  # Don't auto-play, we'll handle playback manually
            )
            
            # Get the audio path
            audio_path = audio_result.get("file_path")
            if audio_path:
                # Play audio and block until complete
                from lmm_project.utils.audio_player import play_audio
                logger.info("Playing audio (blocking)...")
                play_audio(audio_path, blocking=True)  # This will block until playback is complete
                logger.info("Audio playback completed")
                
                # Now that audio is complete, extract features
                features = extract_features_from_file(
                    audio_path,
                    developmental_age=self.development_manager.get_age()
                )
                
                # Create proper message content
                content = TextContent(
                    content_type="text",
                    data={
                        "text": text,
                        "audio_features": features,
                        "audio_path": audio_path,
                        "is_mother_speech": True
                    }
                )
                
                # Create and publish perception message
                perception_message = Message(
                    sender="mother",
                    sender_type=ModuleType.SOCIAL,
                    message_type=MessageType.PERCEPTION_INPUT,
                    content=content
                )
                
                # Process through perception
                self.event_bus.publish(perception_message)
                
                # Log experience with proper structure
                self.experience_logger.log_experience(
                    content={"text": text, "audio_path": audio_path},
                    experience_type="mother_interaction",
                    source="mother",
                    emotional_valence="positive",
                    embedding_text=text
                )
                
                logger.info("Mother response processing completed")
                
        except Exception as e:
            logger.error(f"Error processing mother response: {str(e)}", exc_info=True)
    
    def _cleanup(self) -> None:
        """Clean up system resources."""
        logger.info("Cleaning up system resources")
        self.running = False
        
        # Stop event bus
        self.event_bus.stop()
        
        # Wait for development thread to finish
        if self.development_thread and self.development_thread.is_alive():
            self.development_thread.join(timeout=2.0)
        
        # Save final state
        self.state_manager.save_state(
            description=f"Final state at age {self.development_manager.get_age():.2f}"
        )
        
        # Save experience log
        self.experience_logger.save()
        
        logger.info(f"System shutdown complete. Total interactions: {self.interaction_count}")

    def _development_timer(self) -> None:
        """Update development based on elapsed time."""
        try:
            last_update = time.time()
            update_interval = 0.1  # 100ms update interval
            
            while self.running:
                current_time = time.time()
                elapsed = current_time - last_update
                
                if elapsed >= update_interval:
                    # Update development age
                    self.development_manager.update()
                    
                    # Get current developmental state
                    current_age = self.development_manager.get_age()
                    current_stage = self.development_manager.get_stage()
                    
                    # Update cognitive modules with new age
                    perception.update_age(current_age)
                    attention.update_age(current_age)
                    memory.update_age(current_age)
                    
                    # Log development progress periodically (every 10 seconds)
                    if self.interaction_count % 100 == 0:
                        logger.info(
                            f"Development Progress - Age: {current_age:.3f}, "
                            f"Stage: {current_stage.name}"
                        )
                    
                    last_update = current_time
                
                # Sleep for a short time to prevent CPU overuse
                time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in development timer: {str(e)}", exc_info=True)
            self.running = False  # Signal main loop to stop

    def _initialize_state_saving(self) -> None:
        """Initialize periodic state saving."""
        def state_saver():
            while self.running:
                try:
                    # Save state every hour
                    time.sleep(3600)
                    if self.running:
                        self.state_manager.save_state(
                            description=f"Automatic save at age {self.development_manager.get_age():.2f}"
                        )
                except Exception as e:
                    logger.error(f"Error in state saving: {str(e)}")

        self.state_saving_thread = threading.Thread(
            target=state_saver,
            daemon=True
        )
        self.state_saving_thread.start()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _monitor_system_status(self) -> None:
        """Monitor and log system status periodically."""
        def status_monitor():
            while self.running:
                try:
                    # Log status every 5 minutes
                    time.sleep(300)
                    if self.running:
                        age = self.development_manager.get_age()
                        stage = self.development_manager.get_stage()
                        logger.info(
                            f"System Status - Age: {age:.2f}, Stage: {stage.name}, "
                            f"Interactions: {self.interaction_count}"
                        )
                except Exception as e:
                    logger.error(f"Error in status monitoring: {str(e)}")

        self.monitoring_thread = threading.Thread(
            target=status_monitor,
            daemon=True
        )
        self.monitoring_thread.start()

    def _track_metrics(self) -> None:
        """Track system metrics for monitoring."""
        metrics = {
            "developmental_age": self.development_manager.get_age(),
            "developmental_stage": self.development_manager.get_stage().name,
            "interaction_count": self.interaction_count,
            "memory_usage": memory.get_state(),
            "attention_state": attention.get_state(),
            "perception_state": perception.get_state()
        }
        
        self.state_manager.update_state("system_metrics", metrics)

    def _initialize_mind(self) -> None:
        """Initialize the mind system with proper configuration."""
        # Get mind configuration from config file
        mind_config = {
            "storage_dir": self.config.get_string("storage_dir", "storage"),
            "vector_dimension": self.config.get_int("vector_dimension", 768),
            "cuda_enabled": self.config.get_boolean("system.cuda_enabled", True),
            "cuda_fallback": self.config.get_boolean("system.cuda_fallback", True)
        }

        # Initialize mind with event bus and development manager
        self.mind = get_mind(
            config_path=self.config_path,
            storage_dir=mind_config["storage_dir"]
        )

        # Register the mind with the state manager
        self.state_manager.register_module(
            module_id="mind",
            module_state={
                "config": mind_config,
                "development_level": self.development_manager.get_age(),
                "is_active": True
            }
        )

        logger.info("Mind system initialized")


def main():
    """Main entry point."""
    try:
        # Set up system logging
        setup_system_logging(
            log_dir="logs",
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )
        
        # Create and run system
        system = LMMSystem()
        system.run()
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
