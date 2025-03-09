"""
State Persistence Module

This module provides functionality for saving and loading state information
for the LMM system. It supports versioning, differential backups, and
restoration of mind states from various points in the developmental timeline.

State persistence is crucial for long-running developmental processes and
allows for exploring different developmental trajectories.
"""

import os
import json
import shutil
import logging
import sqlite3
import gzip
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class StatePersistence:
    """
    Handles saving and loading state for the LMM system
    
    This class provides functionality for:
    - Saving complete mind states
    - Creating checkpoints at important developmental milestones
    - Loading from previously saved states
    - Managing state versions and developmental branches
    """
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize the state persistence system
        
        Args:
            storage_dir: Base directory for state storage
        """
        # Ensure storage directories exist
        self.base_dir = Path(storage_dir)
        self.states_dir = self.base_dir / "states"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.backups_dir = self.base_dir / "backups"
        
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for state metadata
        self.db_path = self.base_dir / "states.db"
        self.conn = self._initialize_database()
        
        # Keep track of current state
        self.current_state_id = None
        
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize the states database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create states table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            label TEXT,
            description TEXT,
            developmental_stage TEXT,
            age REAL,
            version TEXT,
            branch TEXT,
            is_checkpoint BOOLEAN,
            parent_state_id TEXT,
            file_path TEXT,
            metadata TEXT
        )
        ''')
        
        # Create index on timestamp
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON states(timestamp)
        ''')
        
        # Create index on developmental_stage
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stage ON states(developmental_stage)
        ''')
        
        # Create index on branch
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_branch ON states(branch)
        ''')
        
        conn.commit()
        return conn
    
    def save_state(
        self,
        mind_state: Dict[str, Any],
        label: Optional[str] = None,
        description: Optional[str] = None,
        is_checkpoint: bool = False,
        branch: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ) -> str:
        """
        Save a mind state
        
        Args:
            mind_state: The state to save
            label: Optional label for this state
            description: Optional description
            is_checkpoint: Whether this is a development checkpoint
            branch: Branch name for this state
            metadata: Additional metadata
            compress: Whether to compress the state
            
        Returns:
            ID of the saved state
        """
        try:
            # Generate a unique ID based on timestamp
            timestamp = datetime.now().isoformat()
            state_id = f"state_{timestamp.replace(':', '-').replace('.', '-')}"
            
            # Generate label if not provided
            if not label:
                developmental_stage = mind_state.get("developmental_stage", "unknown")
                age = mind_state.get("age", 0.0)
                label = f"{developmental_stage.capitalize()} ({age:.2f})"
                
            # Get version number
            version = self._get_next_version(branch)
            
            # Determine save directory
            target_dir = self.checkpoints_dir if is_checkpoint else self.states_dir
            
            # Create file path
            file_name = f"{state_id}.{'gz' if compress else 'json'}"
            file_path = target_dir / file_name
            
            # Create backup of current state if this is a checkpoint
            if is_checkpoint:
                self._create_backup()
                
            # Extract development info
            developmental_stage = mind_state.get("developmental_stage", "unknown")
            age = mind_state.get("age", 0.0)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata_str = json.dumps(metadata)
            
            # Save state to file
            if compress:
                with gzip.open(file_path, 'wb') as f:
                    # Use pickle for compressed storage
                    pickle.dump(mind_state, f, protocol=4)
            else:
                with open(file_path, 'w') as f:
                    json.dump(mind_state, f, indent=2)
            
            # Store metadata in database
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO states (
                id, timestamp, label, description, developmental_stage, age,
                version, branch, is_checkpoint, parent_state_id, file_path, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                state_id,
                timestamp,
                label,
                description or "",
                developmental_stage,
                age,
                version,
                branch,
                1 if is_checkpoint else 0,
                self.current_state_id,  # Track parent state
                str(file_path),
                metadata_str
            ))
            
            self.conn.commit()
            
            # Update current state ID
            self.current_state_id = state_id
            
            logger.info(f"Saved mind state {state_id} (version {version}, branch {branch})")
            
            # Create a symbolic link to latest state
            latest_link = target_dir / f"latest_{branch}.{'gz' if compress else 'json'}"
            if os.path.exists(latest_link):
                os.remove(latest_link)
            
            # Create relative link to the actual file
            shutil.copy2(file_path, latest_link)  # Use copy instead of symlink for Windows compatibility
            
            return state_id
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            self.conn.rollback()
            return ""
    
    def load_state(
        self,
        state_id: Optional[str] = None,
        version: Optional[str] = None,
        branch: str = "main",
        use_latest: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load a saved state
        
        Args:
            state_id: ID of the state to load
            version: Version to load
            branch: Branch to load from
            use_latest: Whether to load the latest state
            
        Returns:
            The loaded state, or None if state not found
        """
        try:
            file_path = None
            
            if state_id:
                # Get specific state by ID
                cursor = self.conn.cursor()
                cursor.execute("SELECT file_path FROM states WHERE id = ?", (state_id,))
                result = cursor.fetchone()
                
                if result:
                    file_path = result[0]
                else:
                    logger.warning(f"State {state_id} not found")
                    return None
                    
            elif version:
                # Get specific version on branch
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT file_path FROM states WHERE version = ? AND branch = ?",
                    (version, branch)
                )
                result = cursor.fetchone()
                
                if result:
                    file_path = result[0]
                else:
                    logger.warning(f"Version {version} on branch {branch} not found")
                    return None
                    
            elif use_latest:
                # Get latest version on branch
                target_dir = self.states_dir
                latest_link = target_dir / f"latest_{branch}.gz"
                
                if os.path.exists(latest_link):
                    file_path = str(latest_link)
                else:
                    # Try non-compressed version
                    latest_link = target_dir / f"latest_{branch}.json"
                    if os.path.exists(latest_link):
                        file_path = str(latest_link)
                    else:
                        logger.warning(f"No latest state found for branch {branch}")
                        return None
            
            if not file_path:
                logger.warning("No valid state identifier provided")
                return None
                
            # Load state from file
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    state = pickle.load(f)
            else:
                with open(file_path, 'r') as f:
                    state = json.load(f)
                    
            # Update current state ID
            if state_id:
                self.current_state_id = state_id
                
            logger.info(f"Loaded mind state from {file_path}")
            return state
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return None
    
    def _get_next_version(self, branch: str) -> str:
        """Get the next version number for a branch"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT version FROM states WHERE branch = ? ORDER BY timestamp DESC LIMIT 1",
                (branch,)
            )
            result = cursor.fetchone()
            
            if not result:
                # First version on this branch
                return "0.1.0"
                
            # Parse existing version
            current_version = result[0]
            major, minor, patch = [int(x) for x in current_version.split('.')]
            
            # Increment patch version
            patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except Exception as e:
            logger.error(f"Error getting next version: {e}")
            return "0.1.0"  # Fallback to initial version
    
    def _create_backup(self) -> bool:
        """Create a backup of all states"""
        try:
            # Create timestamp for backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backups_dir / f"state_backup_{timestamp}.zip"
            
            # Create zip archive
            shutil.make_archive(
                str(backup_file).replace('.zip', ''),
                'zip',
                self.states_dir
            )
            
            logger.info(f"Created backup at {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def list_states(
        self,
        branch: Optional[str] = None,
        developmental_stage: Optional[str] = None,
        checkpoints_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List saved states
        
        Args:
            branch: Filter by branch
            developmental_stage: Filter by developmental stage
            checkpoints_only: Show only checkpoints
            limit: Maximum number of states to return
            
        Returns:
            List of state metadata
        """
        try:
            query_parts = ["SELECT * FROM states WHERE 1=1"]
            params = []
            
            if branch:
                query_parts.append("AND branch = ?")
                params.append(branch)
                
            if developmental_stage:
                query_parts.append("AND developmental_stage = ?")
                params.append(developmental_stage)
                
            if checkpoints_only:
                query_parts.append("AND is_checkpoint = 1")
                
            query_parts.append("ORDER BY timestamp DESC LIMIT ?")
            params.append(limit)
            
            cursor = self.conn.cursor()
            cursor.execute(" ".join(query_parts), tuple(params))
            results = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert to list of dictionaries
            states = []
            for row in results:
                state_dict = dict(zip(column_names, row))
                
                # Parse metadata
                try:
                    state_dict["metadata"] = json.loads(state_dict["metadata"])
                except:
                    state_dict["metadata"] = {}
                    
                states.append(state_dict)
                
            return states
            
        except Exception as e:
            logger.error(f"Error listing states: {e}")
            return []
    
    def create_branch(
        self,
        new_branch: str,
        from_state_id: Optional[str] = None,
        label: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Create a new development branch
        
        Args:
            new_branch: Name of the new branch
            from_state_id: State to branch from (default: current state)
            label: Label for the new branch
            description: Description of the branch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the state to branch from
            source_state_id = from_state_id or self.current_state_id
            
            if not source_state_id:
                logger.error("No state to branch from")
                return False
                
            # Load the source state
            source_state = self.load_state(state_id=source_state_id)
            if not source_state:
                return False
                
            # Save state to new branch
            result = self.save_state(
                mind_state=source_state,
                label=label or f"Branch from {source_state_id}",
                description=description or f"Created branch {new_branch}",
                is_checkpoint=True,
                branch=new_branch
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return False
    
    def get_state_info(self, state_id: str) -> Dict[str, Any]:
        """
        Get information about a specific state
        
        Args:
            state_id: ID of the state
            
        Returns:
            Dictionary with state metadata
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM states WHERE id = ?", (state_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"State {state_id} not found")
                return {}
                
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert to dictionary
            state_info = dict(zip(column_names, result))
            
            # Parse metadata
            try:
                state_info["metadata"] = json.loads(state_info["metadata"])
            except:
                state_info["metadata"] = {}
                
            return state_info
            
        except Exception as e:
            logger.error(f"Error getting state info: {e}")
            return {}
    
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a state
        
        Args:
            state_id: ID of the state to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get file path
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path, is_checkpoint FROM states WHERE id = ?", (state_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"State {state_id} not found for deletion")
                return False
                
            file_path, is_checkpoint = result
            
            # Don't allow deleting checkpoints
            if is_checkpoint:
                logger.warning(f"Cannot delete checkpoint state {state_id}")
                return False
                
            # Delete from database
            cursor.execute("DELETE FROM states WHERE id = ?", (state_id,))
            self.conn.commit()
            
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            logger.info(f"Deleted state {state_id}")
            
            # Update current state ID if necessary
            if self.current_state_id == state_id:
                # Set to parent state if possible
                cursor.execute("SELECT parent_state_id FROM states WHERE id = ?", (state_id,))
                parent_result = cursor.fetchone()
                
                if parent_result and parent_result[0]:
                    self.current_state_id = parent_result[0]
                else:
                    self.current_state_id = None
                    
            return True
            
        except Exception as e:
            logger.error(f"Error deleting state {state_id}: {e}")
            self.conn.rollback()
            return False
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """
        Restore from a backup
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup of current state
            self._create_backup()
            
            # Create temporary restoration directory
            restore_dir = self.base_dir / "restore_temp"
            if restore_dir.exists():
                shutil.rmtree(restore_dir)
            restore_dir.mkdir()
            
            # Extract backup
            shutil.unpack_archive(backup_file, restore_dir, 'zip')
            
            # Copy files to states directory
            for file in restore_dir.iterdir():
                shutil.copy2(file, self.states_dir)
                
            # Clean up
            shutil.rmtree(restore_dir)
            
            logger.info(f"Restored from backup {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed on deletion"""
        self.close() 
