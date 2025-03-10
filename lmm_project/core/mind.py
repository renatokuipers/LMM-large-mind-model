"""
Mind class that serves as the central coordinator for all cognitive modules.
"""
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Type, Set
import os
import json
from datetime import datetime

from .types import ModuleType, ModuleID, DevelopmentalStage, Age, current_timestamp, generate_id
from .exceptions import ModuleError, StateError
from .message import Message, TextContent
from .event_bus import get_event_bus, EventBus
from .state_manager import get_state_manager, StateManager


class Mind:
    """
    Central coordinator for the cognitive system.
    Manages all cognitive modules and their interactions.
    """
    def __init__(self, 
                config_path: Optional[str] = None,
                storage_dir: str = "storage"):
        self.id = generate_id()
        self.logger = logging.getLogger(__name__)
        self.modules: Dict[ModuleID, Any] = {}
        self.module_types: Dict[ModuleID, ModuleType] = {}
        self.config = self._load_config(config_path)
        
        # Set up storage directory
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Get event bus and state manager
        self.event_bus = get_event_bus()
        self.state_manager = get_state_manager(os.path.join(storage_dir, "states"))
        
        # Runtime tracking
        self.start_time: Optional[float] = None
        self.running = False
        self.development_thread: Optional[threading.Thread] = None
        
        # Development rate (age units per real second)
        self.development_rate = self.config.get("development", {}).get("rate", 0.001)
        
        self.logger.info(f"Mind initialized with ID: {self.id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file, or None for defaults
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "system": {
                "version": "0.1.0",
                "log_level": "INFO",
            },
            "development": {
                "rate": 0.001,  # Age units per real second
                "checkpoint_interval": 3600,  # Seconds between state checkpoints
            },
            "modules": {
                # Default module configurations will go here
            },
            "interfaces": {
                "mother": {
                    "enabled": True,
                    "llm_url": "http://192.168.2.12:1234",
                    "personality": "nurturing",
                },
                "researcher": {
                    "enabled": True,
                },
            },
        }
        
        if not config_path or not os.path.exists(config_path):
            self.logger.warning(f"Configuration file not found, using defaults")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return default_config
    
    def register_module(self, module_id: ModuleID, module: Any, module_type: ModuleType) -> None:
        """
        Register a cognitive module with the mind.
        
        Args:
            module_id: Unique ID for the module
            module: Module instance
            module_type: Type of cognitive module
            
        Raises:
            ModuleError: If module is already registered
        """
        if module_id in self.modules:
            raise ModuleError(module_id, "Module already registered")
        
        self.modules[module_id] = module
        self.module_types[module_id] = module_type
        
        # Register module state if it has a get_state method
        if hasattr(module, 'get_state') and callable(module.get_state):
            try:
                module_state = module.get_state()
                self.state_manager.register_module(module_id, module_state)
            except Exception as e:
                self.logger.error(f"Error registering module state: {str(e)}")
        
        self.logger.info(f"Registered module: {module_id} (Type: {module_type.name})")
    
    def unregister_module(self, module_id: ModuleID) -> None:
        """
        Unregister a module from the mind.
        
        Args:
            module_id: ID of the module to unregister
            
        Raises:
            ModuleError: If module is not registered
        """
        if module_id not in self.modules:
            raise ModuleError(module_id, "Module not registered")
        
        # Get module instance
        module = self.modules[module_id]
        
        # Call shutdown method if available
        if hasattr(module, 'shutdown') and callable(module.shutdown):
            try:
                module.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down module {module_id}: {str(e)}")
        
        # Unsubscribe from event bus
        self.event_bus.unsubscribe(module_id)
        
        # Remove from tracking
        del self.modules[module_id]
        del self.module_types[module_id]
        
        self.logger.info(f"Unregistered module: {module_id}")
    
    def get_module(self, module_id: ModuleID) -> Any:
        """
        Get a module by ID.
        
        Args:
            module_id: ID of the module to get
            
        Returns:
            Module instance
            
        Raises:
            ModuleError: If module is not registered
        """
        if module_id not in self.modules:
            raise ModuleError(module_id, "Module not registered")
        
        return self.modules[module_id]
    
    def get_modules_by_type(self, module_type: ModuleType) -> Dict[ModuleID, Any]:
        """
        Get all modules of a specific type.
        
        Args:
            module_type: Type of modules to get
            
        Returns:
            Dictionary of module ID to module instance
        """
        result = {}
        for module_id, module in self.modules.items():
            if self.module_types[module_id] == module_type:
                result[module_id] = module
        
        return result
    
    def start(self) -> None:
        """
        Start the mind system.
        """
        if self.running:
            self.logger.warning("Mind is already running")
            return
        
        # Start the event bus
        self.event_bus.start()
        
        # Start all modules
        for module_id, module in self.modules.items():
            if hasattr(module, 'start') and callable(module.start):
                try:
                    module.start()
                except Exception as e:
                    self.logger.error(f"Error starting module {module_id}: {str(e)}")
        
        # Start development thread
        self.running = True
        self.start_time = current_timestamp()
        self.development_thread = threading.Thread(
            target=self._development_loop,
            daemon=True,
            name="DevelopmentThread"
        )
        self.development_thread.start()
        
        self.logger.info("Mind system started")
    
    def stop(self) -> None:
        """
        Stop the mind system.
        """
        if not self.running:
            self.logger.warning("Mind is not running")
            return
        
        # Stop development thread
        self.running = False
        if self.development_thread:
            self.development_thread.join(timeout=2.0)
            self.development_thread = None
        
        # Stop all modules
        for module_id, module in self.modules.items():
            if hasattr(module, 'stop') and callable(module.stop):
                try:
                    module.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping module {module_id}: {str(e)}")
        
        # Stop the event bus
        self.event_bus.stop()
        
        # Save final state
        try:
            self.state_manager.save_state("Final state at shutdown")
        except Exception as e:
            self.logger.error(f"Error saving final state: {str(e)}")
        
        self.logger.info("Mind system stopped")
    
    def _development_loop(self) -> None:
        """
        Main development loop that runs in a separate thread.
        Updates developmental age and handles periodic tasks.
        """
        last_checkpoint_time = current_timestamp()
        checkpoint_interval = self.config.get("development", {}).get("checkpoint_interval", 3600)
        
        while self.running:
            try:
                # Update age based on development rate
                current_age = self.state_manager.get_state_value("development.age")
                elapsed_seconds = current_timestamp() - self.start_time
                new_age = current_age + (self.development_rate * 1.0)  # 1.0 = one second of development
                
                # Update developmental age
                self.state_manager.update_developmental_age(new_age)
                
                # Update system runtime
                self.state_manager.update_system_runtime()
                
                # Check if it's time for a checkpoint
                current_time = current_timestamp()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    try:
                        self.state_manager.save_state("Periodic checkpoint")
                        last_checkpoint_time = current_time
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {str(e)}")
                
                # Sleep for a short time to avoid consuming too much CPU
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in development loop: {str(e)}")
                time.sleep(1.0)  # Sleep to avoid error spam
    
    def get_developmental_stage(self) -> DevelopmentalStage:
        """
        Get the current developmental stage.
        
        Returns:
            Current developmental stage
        """
        stage_name = self.state_manager.get_state_value("development.stage")
        return DevelopmentalStage[stage_name]
    
    def get_age(self) -> Age:
        """
        Get the current developmental age.
        
        Returns:
            Current developmental age
        """
        return self.state_manager.get_state_value("development.age")
    
    def save_state(self, description: Optional[str] = None) -> str:
        """
        Save the current state.
        
        Args:
            description: Optional description of the state
            
        Returns:
            Path to the saved state file
        """
        return self.state_manager.save_state(description)
    
    def load_state(self, filepath: str) -> None:
        """
        Load state from a file.
        
        Args:
            filepath: Path to the state file
            
        Raises:
            StateError: If the file is not found or invalid
        """
        # Stop the system if it's running
        was_running = self.running
        if was_running:
            self.stop()
        
        # Load state
        self.state_manager.load_state(filepath)
        
        # Restart if it was running
        if was_running:
            self.start()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current system state.
        
        Returns:
            Current system state
        """
        return self.state_manager.get_state()


# Singleton instance
_mind_instance = None
_mind_lock = threading.RLock()

def get_mind(config_path: Optional[str] = None, storage_dir: str = "storage") -> Mind:
    """
    Get the global Mind instance (singleton).
    
    Args:
        config_path: Path to configuration file, or None for defaults
        storage_dir: Directory for storing data
        
    Returns:
        Global Mind instance
    """
    global _mind_instance
    
    with _mind_lock:
        if _mind_instance is None:
            _mind_instance = Mind(
                config_path=config_path,
                storage_dir=storage_dir
            )
            
    return _mind_instance
