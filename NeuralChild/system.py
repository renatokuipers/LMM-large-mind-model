# system.py - System orchestrator for the Neural Child system
from datetime import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
import threading
import json

from config import get_config
from neural_child import NeuralChild, ChildState
from mother import Mother, MotherResponse
from training.session_manager import SessionManager
from utils.serialization import save_state, load_state
from utils.logging_utils import setup_logging, log_interaction

logger = logging.getLogger("NeuralSystem")

class NeuralSystem:
    """Main system orchestrator for the Neural Child simulation"""
    
    def __init__(
        self, 
        config_path: Optional[Path] = None,
        save_dir: Path = Path("./data"),
        simulation_speed: float = 1.0,
        auto_save_interval: int = 100,
        dashboard_enabled: bool = True
    ):
        """Initialize the Neural Child system
        
        Args:
            config_path: Path to configuration file
            save_dir: Directory for saving data
            simulation_speed: Multiplier for development speed
            auto_save_interval: Save state every N interactions
            dashboard_enabled: Whether to enable the dashboard
        """
        # Initialize logging
        setup_logging(save_dir / "logs")
        
        # Load configuration
        self.config = get_config()
        if config_path and os.path.exists(config_path):
            self.config = get_config().load_from_file(config_path)
        
        self.save_dir = save_dir
        self.simulation_speed = simulation_speed
        self.auto_save_interval = auto_save_interval
        self.dashboard_enabled = dashboard_enabled
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir / "checkpoints", exist_ok=True)
        
        # Initialize components
        self.child = None
        self.mother = None
        self.session_manager = None
        
        # System state
        self.interaction_count = 0
        self.running = False
        self.paused = False
        self.last_save_time = datetime.now()
        
        # Dashboard data
        self.dashboard_data = {
            "system_status": {
                "is_training": False,
                "total_training_time": 0.0,
                "total_interactions": 0,
                "child_age": 0.0,
                "chat_ready": False,
                "vocabulary_size": 0,
                "active_networks": [],
                "system_load": 0.0,
                "last_updated": datetime.now().isoformat()
            },
            "interactions": []
        }
        
        logger.info("Neural system initialized")
    
    def initialize_components(self) -> None:
        """Initialize the neural child, mother, and session manager"""
        # Initialize neural child
        self.child = NeuralChild(
            save_dir=self.save_dir / "neural_child",
            simulation_speed=self.simulation_speed,
            llm_base_url=self.config.system.llm_base_url
        )
        
        # Initialize mother
        self.mother = self.child.mother  # Use the mother already created in neural_child
        
        # Initialize session manager
        self.session_manager = SessionManager(
            save_dir=self.save_dir / "training"
        )
        
        logger.info("Neural system components initialized")
    
    def start(self) -> None:
        """Start the neural child system"""
        if self.running:
            logger.warning("System already running")
            return
        
        if not self.child or not self.mother or not self.session_manager:
            self.initialize_components()
        
        self.running = True
        self.paused = False
        
        # Start in a separate thread
        threading.Thread(target=self._run_loop, daemon=True).start()
        
        logger.info("Neural system started")
    
    def pause(self) -> None:
        """Pause the system"""
        self.paused = True
        logger.info("Neural system paused")
    
    def resume(self) -> None:
        """Resume the system"""
        self.paused = False
        logger.info("Neural system resumed")
    
    def stop(self) -> None:
        """Stop the system"""
        self.running = False
        
        # Save state before stopping
        self.save_state()
        
        logger.info("Neural system stopped")
    
    def _run_loop(self) -> None:
        """Main interaction loop"""
        logger.info("Starting interaction loop")
        
        # Mark system as training
        self.dashboard_data["system_status"]["is_training"] = True
        
        # Start session
        session_id = self.session_manager.start_session()
        start_time = datetime.now()
        
        try:
            while self.running:
                # Check if paused
                if self.paused:
                    time.sleep(0.5)
                    continue
                
                # Run one interaction cycle
                self._run_interaction_cycle(session_id)
                
                # Auto-save periodically
                if self.interaction_count % self.auto_save_interval == 0:
                    self.save_state()
                
                # Update dashboard data
                self._update_dashboard_data()
                
                # Brief pause to prevent CPU overload
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in interaction loop: {str(e)}", exc_info=True)
        finally:
            # End session
            session_duration = (datetime.now() - start_time).total_seconds() / 3600  # hours
            self.session_manager.end_session(
                session_id=session_id,
                metrics={
                    "duration_hours": session_duration,
                    "interactions_count": self.interaction_count,
                    "final_vocabulary_size": self.child.metrics.vocabulary_size if self.child else 0,
                    "final_age_days": self.child.metrics.age_days if self.child else 0
                }
            )
            
            # Mark system as not training
            self.dashboard_data["system_status"]["is_training"] = False
            logger.info("Interaction loop ended")
    
    def _run_interaction_cycle(self, session_id: str) -> None:
        """Run a single interaction cycle"""
        # Get child state
        child_state = self.child.get_child_state()
        
        # Get mother's response
        mother_response = self.mother.respond_to_child(child_state)
        
        # Process mother's response
        self.child.process_mother_response(mother_response)
        
        # Log the interaction
        log_interaction(
            child_message=child_state.message,
            mother_message=mother_response.verbal.text,
            child_emotion=child_state.apparent_emotion,
            mother_emotion=mother_response.emotional.primary_emotion
        )
        
        # Record in session manager
        self.session_manager.record_interaction(
            session_id=session_id,
            child_message=child_state.message,
            mother_message=mother_response.verbal.text,
            child_emotion=child_state.apparent_emotion,
            mother_emotion=mother_response.emotional.primary_emotion,
            vocabulary_size=child_state.vocabulary_size,
            age_days=child_state.age_days
        )
        
        # Add to dashboard data
        self.dashboard_data["interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "child_message": child_state.message,
            "mother_message": mother_response.verbal.text,
            "child_emotion": child_state.apparent_emotion,
            "mother_emotion": mother_response.emotional.primary_emotion,
            "child_age_days": child_state.age_days,
            "vocabulary_size": child_state.vocabulary_size
        })
        
        # Keep only recent interactions in memory
        if len(self.dashboard_data["interactions"]) > 20:
            self.dashboard_data["interactions"] = self.dashboard_data["interactions"][-20:]
        
        # Increment counter
        self.interaction_count += 1
    
    def _update_dashboard_data(self) -> None:
        """Update the dashboard data"""
        if not self.child:
            return
        
        self.dashboard_data["system_status"] = {
            "is_training": self.running and not self.paused,
            "total_training_time": self.child.metrics.total_training_time,
            "total_interactions": self.child.metrics.total_interactions,
            "child_age": self.child.metrics.age_days,
            "chat_ready": self.child.is_ready_for_chat(),
            "vocabulary_size": self.child.metrics.vocabulary_size,
            "active_networks": self.child.get_active_networks(),
            "system_load": min(1.0, self.interaction_count / 5000),  # Simulated load
            "last_updated": datetime.now().isoformat()
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the system state"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"system_state_{timestamp}.json"
        
        try:
            # Save child state
            if self.child:
                self.child.save_state()
            
            # Save session manager state
            if self.session_manager:
                self.session_manager.save_state()
            
            # Save system state
            system_state = {
                "interaction_count": self.interaction_count,
                "simulation_speed": self.simulation_speed,
                "last_save_time": datetime.now().isoformat(),
                "config": self.config.model_dump(),
                "dashboard_data": self.dashboard_data
            }
            
            # Use serialization utility
            save_state(system_state, filepath)
            
            self.last_save_time = datetime.now()
            logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}", exc_info=True)
    
    def load_state(self, filepath: Path) -> bool:
        """Load the system state"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            # Load system state
            system_state = load_state(filepath)
            
            if not system_state:
                logger.error("Failed to load system state")
                return False
            
            # Initialize components if needed
            if not self.child or not self.mother or not self.session_manager:
                self.initialize_components()
            
            # Update system state
            self.interaction_count = system_state.get("interaction_count", 0)
            self.simulation_speed = system_state.get("simulation_speed", 1.0)
            self.dashboard_data = system_state.get("dashboard_data", self.dashboard_data)
            
            # Look for child state file in the same directory
            dirname = os.path.dirname(filepath)
            child_state_files = [f for f in os.listdir(dirname) if f.startswith("neural_child_state_")]
            
            if child_state_files:
                # Load the most recent child state
                child_state_files.sort(reverse=True)
                child_state_path = Path(dirname) / child_state_files[0]
                if self.child:
                    self.child.load_state(child_state_path)
            
            # Look for session manager state file
            session_files = [f for f in os.listdir(dirname) if f.startswith("session_manager_")]
            
            if session_files:
                # Load the most recent session state
                session_files.sort(reverse=True)
                session_path = Path(dirname) / session_files[0]
                if self.session_manager:
                    self.session_manager.load_state(session_path)
            
            logger.info(f"System state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}", exc_info=True)
            return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        self._update_dashboard_data()
        
        # Get additional data
        if self.child:
            network_states = self.child.get_network_states()
            vocabulary_stats = self.child.vocabulary_manager.get_vocabulary_statistics().model_dump()
            
            extended_data = {
                "network_states": network_states,
                "vocabulary_stats": vocabulary_stats,
                "development_metrics": self.child.metrics.model_dump()
            }
            
            return {**self.dashboard_data, **extended_data}
        
        return self.dashboard_data
    
    def perform_interaction(self) -> Tuple[str, str]:
        """Manually trigger one interaction and return the results"""
        if not self.child or not self.mother:
            self.initialize_components()
        
        # Create a temporary session if needed
        if not self.session_manager:
            self.session_manager = SessionManager(self.save_dir / "training")
        
        session_id = self.session_manager.get_current_session_id()
        if not session_id:
            session_id = self.session_manager.start_session()
        
        # Run interaction
        child_state = self.child.get_child_state()
        mother_response = self.mother.respond_to_child(child_state)
        self.child.process_mother_response(mother_response)
        
        # Record interaction
        self.session_manager.record_interaction(
            session_id=session_id,
            child_message=child_state.message,
            mother_message=mother_response.verbal.text,
            child_emotion=child_state.apparent_emotion,
            mother_emotion=mother_response.emotional.primary_emotion,
            vocabulary_size=child_state.vocabulary_size,
            age_days=child_state.age_days
        )
        
        # Log the interaction
        log_interaction(
            child_message=child_state.message,
            mother_message=mother_response.verbal.text,
            child_emotion=child_state.apparent_emotion,
            mother_emotion=mother_response.emotional.primary_emotion
        )
        
        # Update dashboard
        self._update_dashboard_data()
        
        self.interaction_count += 1
        
        return child_state.message, mother_response.verbal.text

# Main execution function
def run_neural_system(config_path: Optional[Path] = None, headless: bool = False) -> NeuralSystem:
    """Run the neural child system
    
    Args:
        config_path: Path to configuration file
        headless: Whether to run in headless mode (no dashboard)
    
    Returns:
        The NeuralSystem instance
    """
    system = NeuralSystem(
        config_path=config_path,
        dashboard_enabled=not headless
    )
    system.initialize_components()
    system.start()
    
    logger.info(f"Neural system running in {'headless' if headless else 'dashboard'} mode")
    return system

if __name__ == "__main__":
    # Run the system when executed directly
    system = run_neural_system()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
        logger.info("Neural system terminated by user")