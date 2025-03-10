"""
State persistence implementation for the LMM project.

This module provides mechanisms for saving and loading system state,
including versioning, metadata tracking, and efficient serialization.
"""
import os
import json
import pickle
import shutil
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from lmm_project.core.types import StateDict
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("storage.state_persistence")


class StatePersistence:
    """
    System state persistence manager.
    
    Handles saving, loading, and managing system state snapshots with
    versioning, compression, and metadata tracking.
    """
    
    def __init__(self, storage_dir: str = "storage/states"):
        """
        Initialize the state persistence manager.
        
        Parameters:
        storage_dir: Directory to store state snapshots
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_path = self.storage_dir / "metadata.json"
        self.state_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"Initialized StatePersistence in {self.storage_dir}")
    
    def _load_metadata(self) -> None:
        """Load state metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    self.state_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.state_metadata)} saved states")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load state metadata: {e}")
                self.state_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save state metadata to disk."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.state_metadata, f)
            logger.debug(f"Saved metadata for {len(self.state_metadata)} states")
        except IOError as e:
            logger.error(f"Failed to save state metadata: {e}")
    
    def save_state(
        self, 
        state: StateDict, 
        description: Optional[str] = None, 
        labels: Optional[List[str]] = None,
        is_checkpoint: bool = False
    ) -> str:
        """
        Save a system state snapshot.
        
        Parameters:
        state: System state dictionary
        description: Optional description of the state
        labels: Optional list of labels for categorization
        is_checkpoint: Whether this is an important checkpoint
        
        Returns:
        ID of the saved state
        """
        # Generate timestamp and ID
        timestamp = datetime.now().isoformat()
        state_id = f"state_{timestamp.replace(':', '-')}"
        
        # Create state directory
        state_dir = self.storage_dir / state_id
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state data with compression
        state_path = state_dir / "state.pickle.gz"
        try:
            with gzip.open(state_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to save state {state_id}: {e}")
            return ""
        
        # Save metadata
        meta = {
            "id": state_id,
            "timestamp": timestamp,
            "description": description or "System state snapshot",
            "labels": labels or [],
            "is_checkpoint": is_checkpoint,
            "size_bytes": state_path.stat().st_size,
            "developmental_age": state.get("developmental_age", 0),
            "developmental_stage": state.get("developmental_stage", "PRENATAL"),
            "modules": list(state.get("modules", {}).keys())
        }
        
        self.state_metadata[state_id] = meta
        self._save_metadata()
        
        logger.info(f"Saved state {state_id} ({meta['size_bytes']} bytes)")
        return state_id
    
    def load_state(self, state_id: str) -> Optional[StateDict]:
        """
        Load a system state snapshot.
        
        Parameters:
        state_id: ID of the state to load
        
        Returns:
        Loaded state dictionary or None if not found
        """
        if state_id not in self.state_metadata:
            logger.warning(f"State {state_id} not found")
            return None
            
        state_path = self.storage_dir / state_id / "state.pickle.gz"
        
        if not state_path.exists():
            logger.error(f"State file for {state_id} not found at {state_path}")
            return None
            
        try:
            with gzip.open(state_path, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"Loaded state {state_id}")
            return state
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to load state {state_id}: {e}")
            return None
    
    def get_latest_state(self, 
                       checkpoint_only: bool = False, 
                       with_label: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the ID of the latest state.
        
        Parameters:
        checkpoint_only: Whether to only consider checkpoints
        with_label: Optional label to filter by
        
        Returns:
        ID of the latest state or None if none found
        """
        if not self.state_metadata:
            return None
            
        # Filter and sort states
        filtered_states = []
        for state_id, meta in self.state_metadata.items():
            if checkpoint_only and not meta.get("is_checkpoint", False):
                continue
                
            if with_label and with_label not in meta.get("labels", []):
                continue
                
            filtered_states.append((state_id, meta["timestamp"]))
            
        if not filtered_states:
            return None
            
        # Sort by timestamp and return the latest
        latest_id, _ = sorted(filtered_states, key=lambda x: x[1], reverse=True)[0]
        return latest_id
    
    def list_states(self, 
                  limit: int = 100, 
                  checkpoint_only: bool = False,
                  with_label: Optional[str] = None,
                  sort_by: str = "timestamp",
                  reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List saved states with metadata.
        
        Parameters:
        limit: Maximum number of states to return
        checkpoint_only: Whether to only list checkpoints
        with_label: Optional label to filter by
        sort_by: Field to sort by
        reverse: Whether to reverse the sort order
        
        Returns:
        List of state metadata dictionaries
        """
        # Filter states
        filtered_states = []
        for state_id, meta in self.state_metadata.items():
            if checkpoint_only and not meta.get("is_checkpoint", False):
                continue
                
            if with_label and with_label not in meta.get("labels", []):
                continue
                
            filtered_states.append(meta)
            
        # Sort states
        if sort_by not in filtered_states[0] if filtered_states else {}:
            sort_by = "timestamp"
            
        sorted_states = sorted(
            filtered_states, 
            key=lambda x: x.get(sort_by, ""), 
            reverse=reverse
        )
        
        # Apply limit
        return sorted_states[:limit]
    
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a saved state.
        
        Parameters:
        state_id: ID of the state to delete
        
        Returns:
        True if successfully deleted
        """
        if state_id not in self.state_metadata:
            logger.warning(f"State {state_id} not found")
            return False
            
        state_dir = self.storage_dir / state_id
        
        try:
            if state_dir.exists():
                shutil.rmtree(state_dir)
                
            # Remove from metadata
            del self.state_metadata[state_id]
            self._save_metadata()
            
            logger.info(f"Deleted state {state_id}")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False
    
    def prune_states(self, 
                   max_states: int = 100, 
                   keep_checkpoints: bool = True,
                   keep_labeled: bool = True
    ) -> int:
        """
        Prune old states to limit storage usage.
        
        Parameters:
        max_states: Maximum number of states to keep
        keep_checkpoints: Whether to always keep checkpoints
        keep_labeled: Whether to always keep labeled states
        
        Returns:
        Number of states deleted
        """
        if len(self.state_metadata) <= max_states:
            return 0
            
        # Separate states to keep and candidates for deletion
        states_to_keep = []
        deletion_candidates = []
        
        for state_id, meta in self.state_metadata.items():
            if (keep_checkpoints and meta.get("is_checkpoint", False)) or \
               (keep_labeled and meta.get("labels", [])):
                states_to_keep.append((state_id, meta["timestamp"]))
            else:
                deletion_candidates.append((state_id, meta["timestamp"]))
                
        # Sort deletion candidates by timestamp (oldest first)
        deletion_candidates.sort(key=lambda x: x[1])
        
        # Calculate how many to delete
        to_delete_count = len(self.state_metadata) - max_states
        to_delete_count = min(to_delete_count, len(deletion_candidates))
        
        if to_delete_count <= 0:
            return 0
            
        # Delete oldest states
        deleted_count = 0
        for state_id, _ in deletion_candidates[:to_delete_count]:
            if self.delete_state(state_id):
                deleted_count += 1
                
        return deleted_count
    
    def add_label(self, state_id: str, label: str) -> bool:
        """
        Add a label to a state.
        
        Parameters:
        state_id: ID of the state
        label: Label to add
        
        Returns:
        True if successfully added
        """
        if state_id not in self.state_metadata:
            logger.warning(f"State {state_id} not found")
            return False
            
        # Add label if not already present
        if label not in self.state_metadata[state_id].get("labels", []):
            if "labels" not in self.state_metadata[state_id]:
                self.state_metadata[state_id]["labels"] = []
                
            self.state_metadata[state_id]["labels"].append(label)
            self._save_metadata()
            
        return True
    
    def remove_label(self, state_id: str, label: str) -> bool:
        """
        Remove a label from a state.
        
        Parameters:
        state_id: ID of the state
        label: Label to remove
        
        Returns:
        True if successfully removed
        """
        if state_id not in self.state_metadata:
            logger.warning(f"State {state_id} not found")
            return False
            
        # Remove label if present
        if "labels" in self.state_metadata[state_id] and \
           label in self.state_metadata[state_id]["labels"]:
            self.state_metadata[state_id]["labels"].remove(label)
            self._save_metadata()
            
        return True
