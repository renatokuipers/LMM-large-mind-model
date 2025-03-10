"""
State Manager for tracking, updating and persisting the system state.
"""
import os
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
import shutil

from .types import StateDict, Age, DevelopmentalStage, ModuleID, Timestamp, current_timestamp
from .exceptions import StateError


class StateManager:
    """
    Manages the state of the entire mind system.
    Provides functionality for tracking, updating, and persisting state.
    """
    def __init__(self, storage_dir: str = "storage/states"):
        self.state: StateDict = self._create_initial_state()
        self.storage_dir = storage_dir
        self.state_lock = threading.RLock()
        self.state_observers: List[Callable[[StateDict], None]] = []
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        self.logger.info("StateManager initialized")
    
    def _create_initial_state(self) -> StateDict:
        """Create the initial system state."""
        return {
            "system": {
                "created_at": current_timestamp(),
                "last_updated": current_timestamp(),
                "version": "0.1.0",
                "runtime": 0.0,  # Runtime in seconds
            },
            "development": {
                "age": 0.0,  # Age in developmental units
                "stage": DevelopmentalStage.PRENATAL.name,
                "milestones_achieved": [],
            },
            "modules": {},  # Module-specific states
            "homeostasis": {
                "energy": 1.0,
                "arousal": 0.5,
                "cognitive_load": 0.0,
                "coherence": 1.0,
                "social_need": 0.5,
            },
            "interfaces": {},  # Interface-specific states
            "metrics": {
                "message_count": 0,
                "learning_events": 0,
                "errors": 0,
            }
        }
    
    def get_state(self, deep_copy: bool = True) -> StateDict:
        """
        Get the current system state.
        
        Args:
            deep_copy: Whether to return a deep copy of the state
            
        Returns:
            The current system state
        """
        with self.state_lock:
            if deep_copy:
                # Use json to create a deep copy
                return json.loads(json.dumps(self.state))
            else:
                return self.state
    
    def update_system_runtime(self) -> None:
        """Update the system runtime based on creation timestamp."""
        with self.state_lock:
            created_at = self.state["system"]["created_at"]
            runtime = current_timestamp() - created_at
            self.state["system"]["runtime"] = runtime
            self.state["system"]["last_updated"] = current_timestamp()
    
    def update_state(self, path: str, value: Any) -> None:
        """
        Update a specific part of the state.
        
        Args:
            path: Dot-separated path to the state element (e.g., "development.age")
            value: New value to set
            
        Raises:
            StateError: If the path is invalid
        """
        parts = path.split('.')
        
        with self.state_lock:
            # Navigate to the target dictionary
            current = self.state
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = value
            
            # Update last_updated timestamp
            self.state["system"]["last_updated"] = current_timestamp()
            
            # Notify observers
            self._notify_observers()
    
    def get_state_value(self, path: str) -> Any:
        """
        Get a specific value from the state.
        
        Args:
            path: Dot-separated path to the state element
            
        Returns:
            The requested state value
            
        Raises:
            StateError: If the path is invalid
        """
        parts = path.split('.')
        
        with self.state_lock:
            # Navigate to the target value
            current = self.state
            for part in parts:
                if part not in current:
                    raise StateError(f"Invalid state path: {path}")
                current = current[part]
            
            return current
    
    def register_module(self, module_id: ModuleID, module_state: Dict[str, Any]) -> None:
        """
        Register a module's state.
        
        Args:
            module_id: ID of the module
            module_state: Initial state for the module
        """
        with self.state_lock:
            self.state["modules"][module_id] = module_state
            self.state["system"]["last_updated"] = current_timestamp()
            self._notify_observers()
    
    def update_module_state(self, module_id: ModuleID, state_updates: Dict[str, Any]) -> None:
        """
        Update a module's state.
        
        Args:
            module_id: ID of the module
            state_updates: State values to update
            
        Raises:
            StateError: If the module is not registered
        """
        with self.state_lock:
            if module_id not in self.state["modules"]:
                raise StateError(f"Module not registered: {module_id}")
            
            # Update the module state
            for key, value in state_updates.items():
                self.state["modules"][module_id][key] = value
            
            self.state["system"]["last_updated"] = current_timestamp()
            self._notify_observers()
    
    def update_developmental_age(self, new_age: Age) -> None:
        """
        Update the developmental age and stage.
        
        Args:
            new_age: New developmental age
        """
        with self.state_lock:
            self.state["development"]["age"] = new_age
            
            # Update developmental stage based on age
            if 0.0 <= new_age < 0.1:
                stage = DevelopmentalStage.PRENATAL
            elif 0.1 <= new_age < 1.0:
                stage = DevelopmentalStage.INFANT
            elif 1.0 <= new_age < 3.0:
                stage = DevelopmentalStage.CHILD
            elif 3.0 <= new_age < 6.0:
                stage = DevelopmentalStage.ADOLESCENT
            else:  # new_age >= 6.0
                stage = DevelopmentalStage.ADULT
            
            self.state["development"]["stage"] = stage.name
            self.state["system"]["last_updated"] = current_timestamp()
            self._notify_observers()
    
    def add_milestone(self, milestone: str) -> None:
        """
        Add an achieved developmental milestone.
        
        Args:
            milestone: Name of the milestone achieved
        """
        with self.state_lock:
            milestones = self.state["development"]["milestones_achieved"]
            if milestone not in milestones:
                milestones.append(milestone)
                self.state["system"]["last_updated"] = current_timestamp()
                self._notify_observers()
    
    def add_observer(self, observer: Callable[[StateDict], None]) -> None:
        """
        Add a state observer function.
        
        Args:
            observer: Function to call when state changes
        """
        with self.state_lock:
            if observer not in self.state_observers:
                self.state_observers.append(observer)
    
    def remove_observer(self, observer: Callable[[StateDict], None]) -> None:
        """
        Remove a state observer.
        
        Args:
            observer: Observer function to remove
        """
        with self.state_lock:
            if observer in self.state_observers:
                self.state_observers.remove(observer)
    
    def _notify_observers(self) -> None:
        """Notify all state observers of a state change."""
        state_copy = self.get_state(deep_copy=True)
        for observer in self.state_observers:
            try:
                observer(state_copy)
            except Exception as e:
                self.logger.error(f"Error in state observer: {str(e)}")
    
    def save_state(self, description: Optional[str] = None) -> str:
        """
        Save the current state to disk.
        
        Args:
            description: Optional description of the state
            
        Returns:
            Path to the saved state file
        """
        with self.state_lock:
            # Create state to save with metadata
            save_state = self.get_state(deep_copy=True)
            save_state["_meta"] = {
                "saved_at": current_timestamp(),
                "description": description or "Automatic state save",
            }
            
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"state_{timestamp}.json"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Save state to file
            with open(filepath, 'w') as f:
                json.dump(save_state, f, indent=2)
            
            self.logger.info(f"State saved to {filepath}")
            return filepath
    
    def load_state(self, filepath: str) -> None:
        """
        Load state from disk.
        
        Args:
            filepath: Path to the state file
            
        Raises:
            StateError: If the file is not found or invalid
        """
        if not os.path.exists(filepath):
            raise StateError(f"State file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
            
            # Remove metadata
            if "_meta" in loaded_state:
                del loaded_state["_meta"]
            
            with self.state_lock:
                self.state = loaded_state
                self.state["system"]["last_updated"] = current_timestamp()
                self._notify_observers()
            
            self.logger.info(f"State loaded from {filepath}")
        
        except Exception as e:
            raise StateError(f"Failed to load state: {str(e)}")
    
    def list_saved_states(self) -> List[Dict[str, Any]]:
        """
        List all saved states.
        
        Returns:
            List of saved state metadata
        """
        result = []
        
        if not os.path.exists(self.storage_dir):
            return result
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json') and filename.startswith('state_'):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        state_data = json.load(f)
                    
                    # Extract metadata
                    meta = state_data.get("_meta", {})
                    saved_at = meta.get("saved_at", 0)
                    description = meta.get("description", "Unknown")
                    
                    # Add to result
                    result.append({
                        "filename": filename,
                        "filepath": filepath,
                        "saved_at": saved_at,
                        "saved_at_human": datetime.fromtimestamp(saved_at).strftime("%Y-%m-%d %H:%M:%S"),
                        "description": description,
                        "age": state_data.get("development", {}).get("age", 0),
                        "stage": state_data.get("development", {}).get("stage", "UNKNOWN"),
                    })
                
                except Exception as e:
                    self.logger.error(f"Error reading state file {filename}: {str(e)}")
        
        # Sort by saved_at, newest first
        result.sort(key=lambda x: x["saved_at"], reverse=True)
        return result
    
    def reset_state(self) -> None:
        """Reset the state to initial values."""
        with self.state_lock:
            self.state = self._create_initial_state()
            self._notify_observers()
            self.logger.info("State reset to initial values")


# Singleton instance
_state_manager_instance = None
_state_manager_lock = threading.RLock()

def get_state_manager(storage_dir: str = "storage/states") -> StateManager:
    """
    Get the global StateManager instance (singleton).
    
    Args:
        storage_dir: Directory for storing state files
        
    Returns:
        Global StateManager instance
    """
    global _state_manager_instance
    
    with _state_manager_lock:
        if _state_manager_instance is None:
            _state_manager_instance = StateManager(storage_dir=storage_dir)
            
    return _state_manager_instance
