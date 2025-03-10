from typing import Dict, List, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
import gzip
import hashlib
import logging
import threading
from pathlib import Path

from .exceptions import StateManagerError

logger = logging.getLogger(__name__)

class StateManager(BaseModel):
    """
    Manages the state of the cognitive system
    
    The StateManager tracks the current state of the mind and its modules,
    provides methods to update the state, and handles state persistence.
    
    Features:
    - Thread-safe state updates
    - State history with configurable size
    - Compressed state storage for large states
    - State diffing to track changes
    - Automatic state backup
    """
    current_state: Dict[str, Any] = Field(default_factory=dict)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history_size: int = Field(default=100)
    last_updated: datetime = Field(default_factory=datetime.now)
    save_directory: Path = Field(default=Path("storage/states"))
    backup_directory: Path = Field(default=Path("storage/backups"))
    compression_threshold: int = Field(default=1024 * 1024)  # 1MB
    _lock: threading.RLock = Field(default_factory=threading.RLock, exclude=True)
    _state_hashes: Dict[str, str] = Field(default_factory=dict, exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
    
    def update_state(self, state_update: Dict[str, Any], track_changes: bool = True) -> Dict[str, Any]:
        """
        Update the current state with new information
        
        Args:
            state_update: Dictionary containing state updates
            track_changes: Whether to track changes for diffing
        
        Returns:
            Updated complete state
        
        Raises:
            StateManagerError: If there's an error updating the state
        """
        try:
            with self._lock:
                # Track changes if requested
                changed_keys = set()
                if track_changes:
                    for key, value in state_update.items():
                        if key in self.current_state:
                            if self._has_changed(key, value):
                                changed_keys.add(key)
                        else:
                            changed_keys.add(key)
                
                # Update current state
                self.current_state.update(state_update)
                
                # Update hashes for changed keys
                for key in changed_keys:
                    self._update_hash(key, self.current_state[key])
                
                # Record timestamp
                self.last_updated = datetime.now()
                self.current_state["last_updated"] = self.last_updated.isoformat()
                
                # Add to history
                history_entry = self.current_state.copy()
                if changed_keys:
                    history_entry["_changed_keys"] = list(changed_keys)
                self.state_history.append(history_entry)
                
                # Trim history if needed
                if len(self.state_history) > self.max_history_size:
                    self.state_history = self.state_history[-self.max_history_size:]
                    
                return self.current_state
        except Exception as e:
            logger.error(f"Error updating state: {str(e)}")
            raise StateManagerError(f"Error updating state: {str(e)}")
    
    def get_state(self, key: Optional[str] = None) -> Any:
        """
        Get current state or a specific state value
        
        Args:
            key: Optional key to retrieve specific state value
        
        Returns:
            Current state or specific state value
        """
        with self._lock:
            if key:
                return self.current_state.get(key)
            
            return self.current_state.copy()
    
    def get_state_history(self, limit: int = 10, keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get state history, optionally filtered by keys
        
        Args:
            limit: Maximum number of historical states to return
            keys: Optional list of keys to include in the result
        
        Returns:
            List of historical states, most recent first
        """
        with self._lock:
            if keys:
                filtered_history = []
                for state in self.state_history[-limit:]:
                    filtered_state = {k: state.get(k) for k in keys if k in state}
                    if "_changed_keys" in state:
                        filtered_state["_changed_keys"] = [k for k in state["_changed_keys"] if k in keys]
                    filtered_history.append(filtered_state)
                return filtered_history
            
            return [state.copy() for state in self.state_history[-limit:]]
    
    def get_changes_since(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get all state changes since a specific timestamp
        
        Args:
            timestamp: The timestamp to get changes since
        
        Returns:
            Dictionary of changed state values
        """
        with self._lock:
            changes = {}
            for state in reversed(self.state_history):
                state_time = datetime.fromisoformat(state["last_updated"]) if "last_updated" in state else self.last_updated
                if state_time <= timestamp:
                    break
                
                if "_changed_keys" in state:
                    for key in state["_changed_keys"]:
                        if key not in changes and key in state:
                            changes[key] = state[key]
            
            return changes
    
    def save_state(self, filename: Optional[str] = None, compress: Optional[bool] = None) -> Path:
        """
        Save current state to file
        
        Args:
            filename: Optional filename to save state to
            compress: Whether to compress the state (auto-determined by size if None)
        
        Returns:
            Path to saved state file
        
        Raises:
            StateManagerError: If there's an error saving the state
        """
        try:
            with self._lock:
                # Create directory if it doesn't exist
                self.save_directory.mkdir(parents=True, exist_ok=True)
                
                # Generate filename if not provided
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    stage = self.current_state.get("developmental_stage", "unknown")
                    filename = f"mind_state_{stage}_{timestamp}.json"
                
                # Determine if compression should be used
                state_json = json.dumps(self.current_state, indent=2, default=str)
                if compress is None:
                    compress = len(state_json) > self.compression_threshold
                
                # Save to file
                file_path = self.save_directory / filename
                if compress:
                    if not filename.endswith('.gz'):
                        file_path = file_path.with_suffix('.json.gz')
                    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                        f.write(state_json)
                else:
                    with open(file_path, "w", encoding='utf-8') as f:
                        f.write(state_json)
                
                # Create backup
                self._create_backup(file_path)
                
                logger.info(f"Saved state to {file_path}")
                return file_path
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise StateManagerError(f"Error saving state: {str(e)}")
    
    def load_state(self, filepath: str) -> Dict[str, Any]:
        """
        Load state from file
        
        Args:
            filepath: Path to state file
        
        Returns:
            Loaded state
        
        Raises:
            StateManagerError: If there's an error loading the state
            FileNotFoundError: If the state file doesn't exist
        """
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                raise FileNotFoundError(f"State file not found: {filepath}")
            
            # Determine if file is compressed
            is_compressed = filepath.endswith('.gz')
            
            # Load state
            if is_compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    loaded_state = json.load(f)
            else:
                with open(file_path, "r", encoding='utf-8') as f:
                    loaded_state = json.load(f)
            
            with self._lock:
                # Update current state
                self.current_state = loaded_state
                
                # Reset hashes
                self._state_hashes = {}
                for key, value in self.current_state.items():
                    self._update_hash(key, value)
                
                # Add to history
                history_entry = self.current_state.copy()
                history_entry["_loaded_from"] = str(file_path)
                self.state_history.append(history_entry)
                
                # Trim history if needed
                if len(self.state_history) > self.max_history_size:
                    self.state_history = self.state_history[-self.max_history_size:]
                
                logger.info(f"Loaded state from {file_path}")
                return self.current_state
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise StateManagerError(f"Error loading state: {str(e)}")
    
    def clear_state(self) -> None:
        """
        Clear the current state
        
        This resets the state to an empty dictionary but preserves history.
        """
        with self._lock:
            self.current_state = {}
            self.last_updated = datetime.now()
            self._state_hashes = {}
    
    def _has_changed(self, key: str, value: Any) -> bool:
        """Check if a state value has changed using hash comparison"""
        current_hash = self._calculate_hash(value)
        previous_hash = self._state_hashes.get(key)
        return previous_hash != current_hash
    
    def _update_hash(self, key: str, value: Any) -> None:
        """Update the hash for a state value"""
        self._state_hashes[key] = self._calculate_hash(value)
    
    def _calculate_hash(self, value: Any) -> str:
        """Calculate a hash for a value"""
        try:
            value_str = json.dumps(value, sort_keys=True, default=str)
            return hashlib.md5(value_str.encode()).hexdigest()
        except:
            return hashlib.md5(str(value).encode()).hexdigest()
    
    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the state file"""
        try:
            backup_path = self.backup_directory / file_path.name
            if file_path.exists():
                with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                    dst.write(src.read())
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")
