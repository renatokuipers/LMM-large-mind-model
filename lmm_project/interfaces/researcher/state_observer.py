"""
State Observer Module

This module provides functionality for observing, recording, and analyzing
the internal state of cognitive modules in the LMM system. It enables researchers
to monitor activation patterns, connection strengths, and processing flows to
better understand the system's internal operations.
"""

import os
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
import sqlite3
import numpy as np
from collections import defaultdict

from lmm_project.interfaces.researcher.models import (
    CognitiveModuleState,
    SystemStateSnapshot,
    NeuralActivitySnapshot,
    DevelopmentalStage
)
from lmm_project.storage.state_persistence import StatePersistence

# Set up logging
logger = logging.getLogger(__name__)

class StateObserver:
    """
    Observes and analyzes the internal state of cognitive modules.
    
    This class provides functionality for:
    - Monitoring activation levels in cognitive modules
    - Recording snapshots of system state for analysis
    - Visualizing neural activity patterns
    - Detecting significant state changes 
    - Comparing states across developmental stages
    """
    
    def __init__(
        self, 
        storage_dir: str = "storage/observations",
        state_persistence: Optional[StatePersistence] = None,
        snapshot_interval: int = 3600  # One hour default interval (seconds)
    ):
        """
        Initialize the StateObserver.
        
        Args:
            storage_dir: Directory to store state observations
            state_persistence: Optional StatePersistence instance to use
            snapshot_interval: Default interval for automatic snapshots (seconds)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to state persistence system
        self.state_persistence = state_persistence or StatePersistence()
        
        # Configuration
        self.snapshot_interval = snapshot_interval
        
        # Initialize database for state observations
        self.db_path = self.storage_dir / "state_observations.db"
        self.conn = self._initialize_database()
        
        # In-memory cache of most recent module states
        self.module_states: Dict[str, CognitiveModuleState] = {}
        
        # Registered observers and callbacks
        self.state_observers: Dict[str, Dict[str, Callable]] = {}
        
        # Snapshot scheduling
        self.last_snapshot_time = None
        self.is_auto_snapshot_enabled = False
        
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for state observations."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create system snapshots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            developmental_stage TEXT NOT NULL,
            global_metrics TEXT,
            active_processes TEXT,
            system_load TEXT,
            recent_experiences TEXT,
            notes TEXT
        )
        ''')
        
        # Create module states table (linked to snapshots)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS module_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            module_name TEXT NOT NULL,
            active INTEGER NOT NULL,
            activation_level REAL NOT NULL,
            last_update TEXT NOT NULL,
            internal_state TEXT,
            connections TEXT,
            performance_metrics TEXT,
            developmental_metrics TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES system_snapshots (snapshot_id)
        )
        ''')
        
        # Create neural activity snapshots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS neural_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            activity_level REAL NOT NULL,
            activation_map TEXT NOT NULL,
            context TEXT,
            duration_ms REAL NOT NULL,
            related_stimulus TEXT
        )
        ''')
        
        # Create state change events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS state_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            change_type TEXT NOT NULL,
            previous_value TEXT,
            new_value TEXT,
            magnitude REAL,
            importance REAL
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON system_snapshots (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_module_states_module ON module_states (module_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_neural_activity_module ON neural_activity (module)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_changes_module ON state_changes (module)')
        
        conn.commit()
        return conn
    
    def register_module(
        self, 
        module_name: str,
        module_instance: Any,
        state_extractor: Callable[[Any], Dict[str, Any]]
    ) -> bool:
        """
        Register a cognitive module for state observation.
        
        Args:
            module_name: Name of the module to register
            module_instance: Instance of the module
            state_extractor: Function to extract state from the module
            
        Returns:
            success: Whether the module was successfully registered
        """
        if not callable(state_extractor):
            logger.error(f"State extractor for {module_name} must be callable")
            return False
            
        self.state_observers[module_name] = {
            "instance": module_instance,
            "extractor": state_extractor,
            "last_observation": None
        }
        
        logger.info(f"Registered module {module_name} for state observation")
        return True
    
    def unregister_module(self, module_name: str) -> bool:
        """
        Unregister a module from state observation.
        
        Args:
            module_name: Name of the module to unregister
            
        Returns:
            success: Whether the module was successfully unregistered
        """
        if module_name in self.state_observers:
            del self.state_observers[module_name]
            logger.info(f"Unregistered module {module_name} from state observation")
            return True
        
        logger.warning(f"Module {module_name} was not registered for observation")
        return False
    
    def observe_module_state(self, module_name: str) -> Optional[CognitiveModuleState]:
        """
        Observe the current state of a specific cognitive module.
        
        Args:
            module_name: Name of the module to observe
            
        Returns:
            module_state: Current state of the module or None if not available
        """
        if module_name not in self.state_observers:
            logger.warning(f"Module {module_name} is not registered for observation")
            return None
            
        try:
            # Get module instance and state extractor
            observer = self.state_observers[module_name]
            module_instance = observer["instance"]
            state_extractor = observer["extractor"]
            
            # Extract raw state data
            raw_state = state_extractor(module_instance)
            
            # Convert to CognitiveModuleState
            module_state = CognitiveModuleState(
                module_name=module_name,
                active=raw_state.get("active", True),
                activation_level=raw_state.get("activation_level", 0.0),
                last_update=raw_state.get("last_update", datetime.now()),
                internal_state=raw_state.get("internal_state", {}),
                connections=raw_state.get("connections", {}),
                performance_metrics=raw_state.get("performance_metrics", {}),
                developmental_metrics=raw_state.get("developmental_metrics", {})
            )
            
            # Check for significant changes
            last_observation = observer.get("last_observation")
            if last_observation:
                self._detect_state_changes(module_name, last_observation, module_state)
                
            # Update cache
            self.module_states[module_name] = module_state
            self.state_observers[module_name]["last_observation"] = module_state
            
            return module_state
            
        except Exception as e:
            logger.error(f"Error observing state of module {module_name}: {str(e)}")
            return None
    
    def observe_all_modules(self) -> Dict[str, CognitiveModuleState]:
        """
        Observe the current state of all registered cognitive modules.
        
        Returns:
            module_states: Dictionary of module states by name
        """
        results = {}
        
        for module_name in self.state_observers.keys():
            module_state = self.observe_module_state(module_name)
            if module_state:
                results[module_name] = module_state
                
        return results
    
    def _detect_state_changes(
        self, 
        module_name: str,
        previous_state: CognitiveModuleState,
        current_state: CognitiveModuleState,
        threshold: float = 0.1
    ) -> None:
        """
        Detect and record significant changes in module state.
        
        Args:
            module_name: Name of the module
            previous_state: Previous observed state
            current_state: Current observed state
            threshold: Threshold for significance (0.0-1.0)
        """
        # Check activation level change
        activation_change = abs(current_state.activation_level - previous_state.activation_level)
        if activation_change > threshold:
            self.record_state_change(
                module=module_name,
                change_type="activation_level",
                previous_value=previous_state.activation_level,
                new_value=current_state.activation_level,
                magnitude=activation_change,
                importance=activation_change
            )
            
        # Check active state change
        if current_state.active != previous_state.active:
            self.record_state_change(
                module=module_name,
                change_type="active_state",
                previous_value=previous_state.active,
                new_value=current_state.active,
                magnitude=1.0,
                importance=0.8
            )
            
        # Check for new connections
        prev_connections = set(previous_state.connections.keys())
        curr_connections = set(current_state.connections.keys())
        
        new_connections = curr_connections - prev_connections
        for conn in new_connections:
            self.record_state_change(
                module=module_name,
                change_type="new_connection",
                previous_value=None,
                new_value=conn,
                magnitude=current_state.connections.get(conn, 0.0),
                importance=0.7
            )
            
        # Check for significant changes in developmental metrics
        for metric, value in current_state.developmental_metrics.items():
            if metric in previous_state.developmental_metrics:
                prev_value = previous_state.developmental_metrics[metric]
                change = abs(value - prev_value)
                
                if change > threshold:
                    self.record_state_change(
                        module=module_name,
                        change_type=f"developmental_metric_{metric}",
                        previous_value=prev_value,
                        new_value=value,
                        magnitude=change,
                        importance=change * 0.5 + 0.3  # Scale importance
                    )
    
    def record_state_change(
        self,
        module: str,
        change_type: str,
        previous_value: Any,
        new_value: Any,
        magnitude: float = 0.0,
        importance: float = 0.5,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Record a significant state change event.
        
        Args:
            module: Module where the change occurred
            change_type: Type of state change
            previous_value: Value before the change
            new_value: Value after the change
            magnitude: Magnitude of the change (0.0-1.0)
            importance: Importance of the change (0.0-1.0)
            timestamp: When the change occurred (defaults to now)
            
        Returns:
            change_id: ID of the recorded change
        """
        timestamp = timestamp or datetime.now()
        
        # Convert values to JSON strings if they're complex
        prev_value_str = json.dumps(previous_value) if not isinstance(previous_value, (str, int, float, bool, type(None))) else str(previous_value)
        new_value_str = json.dumps(new_value) if not isinstance(new_value, (str, int, float, bool, type(None))) else str(new_value)
        
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO state_changes (
                module, timestamp, change_type, previous_value, new_value, magnitude, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                module,
                timestamp.isoformat(),
                change_type,
                prev_value_str,
                new_value_str,
                magnitude,
                importance
            )
        )
        self.conn.commit()
        
        # Get the ID of the inserted row
        change_id = cursor.lastrowid
        
        # Log significant changes
        if importance > 0.7:
            logger.info(f"Significant state change in {module}: {change_type} ({prev_value_str} -> {new_value_str})")
            
        return change_id
    
    def record_neural_activity(self, activity: NeuralActivitySnapshot) -> int:
        """
        Record neural activity pattern.
        
        Args:
            activity: NeuralActivitySnapshot object
            
        Returns:
            activity_id: ID of the recorded activity
        """
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO neural_activity (
                module, timestamp, pattern_type, activity_level, activation_map,
                context, duration_ms, related_stimulus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                activity.module,
                activity.timestamp.isoformat(),
                activity.pattern_type,
                activity.activity_level,
                json.dumps(activity.activation_map),
                activity.context,
                activity.duration_ms,
                activity.related_stimulus
            )
        )
        self.conn.commit()
        
        # Get the ID of the inserted row
        activity_id = cursor.lastrowid
        
        return activity_id
    
    def take_system_snapshot(
        self,
        developmental_stage: Union[DevelopmentalStage, str],
        global_metrics: Optional[Dict[str, Any]] = None,
        active_processes: Optional[List[str]] = None,
        system_load: Optional[Dict[str, float]] = None,
        recent_experiences: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Take a snapshot of the entire system state.
        
        Args:
            developmental_stage: Current developmental stage
            global_metrics: System-wide metrics
            active_processes: Currently active cognitive processes
            system_load: System load metrics
            recent_experiences: Recent experiences
            notes: Additional notes about this snapshot
            
        Returns:
            snapshot_id: ID of the created snapshot
        """
        # Generate snapshot ID
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Ensure we have the latest module states
        self.observe_all_modules()
        
        # Handle stage if it's a string
        if isinstance(developmental_stage, str):
            try:
                stage_value = DevelopmentalStage(developmental_stage).value
            except ValueError:
                logger.warning(f"Invalid developmental stage: {developmental_stage}")
                stage_value = developmental_stage
        else:
            stage_value = developmental_stage.value
            
        # Store system snapshot
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO system_snapshots (
                snapshot_id, timestamp, developmental_stage, global_metrics,
                active_processes, system_load, recent_experiences, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                snapshot_id,
                timestamp.isoformat(),
                stage_value,
                json.dumps(global_metrics) if global_metrics else None,
                json.dumps(active_processes) if active_processes else None,
                json.dumps(system_load) if system_load else None,
                json.dumps(recent_experiences) if recent_experiences else None,
                notes
            )
        )
        
        # Store module states for this snapshot
        for module_name, module_state in self.module_states.items():
            cursor.execute(
                '''
                INSERT INTO module_states (
                    snapshot_id, module_name, active, activation_level, last_update,
                    internal_state, connections, performance_metrics, developmental_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    snapshot_id,
                    module_name,
                    int(module_state.active),
                    module_state.activation_level,
                    module_state.last_update.isoformat(),
                    json.dumps(module_state.internal_state),
                    json.dumps(module_state.connections),
                    json.dumps(module_state.performance_metrics),
                    json.dumps(module_state.developmental_metrics)
                )
            )
            
        self.conn.commit()
        
        # Update last snapshot time
        self.last_snapshot_time = timestamp
        
        logger.info(f"Took system snapshot {snapshot_id} at developmental stage {stage_value}")
        return snapshot_id
    
    def get_snapshot(self, snapshot_id: str) -> Optional[SystemStateSnapshot]:
        """
        Retrieve a system snapshot by ID.
        
        Args:
            snapshot_id: ID of the snapshot to retrieve
            
        Returns:
            snapshot: SystemStateSnapshot object or None if not found
        """
        cursor = self.conn.cursor()
        
        # Get system snapshot
        cursor.execute(
            "SELECT * FROM system_snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        snapshot_row = cursor.fetchone()
        
        if not snapshot_row:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return None
            
        # Get column names
        snapshot_columns = [col[0] for col in cursor.description]
        snapshot_data = dict(zip(snapshot_columns, snapshot_row))
        
        # Get module states for this snapshot
        cursor.execute(
            "SELECT * FROM module_states WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        module_rows = cursor.fetchall()
        module_columns = [col[0] for col in cursor.description]
        
        # Build module states dictionary
        module_states = {}
        for row in module_rows:
            module_data = dict(zip(module_columns, row))
            module_name = module_data["module_name"]
            
            # Parse JSON fields
            internal_state = json.loads(module_data["internal_state"] or "{}")
            connections = json.loads(module_data["connections"] or "{}")
            performance_metrics = json.loads(module_data["performance_metrics"] or "{}")
            developmental_metrics = json.loads(module_data["developmental_metrics"] or "{}")
            
            # Create CognitiveModuleState
            module_states[module_name] = CognitiveModuleState(
                module_name=module_name,
                active=bool(module_data["active"]),
                activation_level=module_data["activation_level"],
                last_update=datetime.fromisoformat(module_data["last_update"]),
                internal_state=internal_state,
                connections=connections,
                performance_metrics=performance_metrics,
                developmental_metrics=developmental_metrics
            )
            
        # Parse JSON fields in snapshot data
        global_metrics = json.loads(snapshot_data["global_metrics"] or "{}")
        active_processes = json.loads(snapshot_data["active_processes"] or "[]")
        system_load = json.loads(snapshot_data["system_load"] or "{}")
        recent_experiences = json.loads(snapshot_data["recent_experiences"] or "[]")
        
        # Create the SystemStateSnapshot
        try:
            # Try to convert stage string to enum
            developmental_stage = DevelopmentalStage(snapshot_data["developmental_stage"])
        except ValueError:
            # Use a default if the value isn't a valid enum
            logger.warning(f"Unknown developmental stage: {snapshot_data['developmental_stage']}")
            developmental_stage = DevelopmentalStage.PRENATAL
            
        snapshot = SystemStateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
            developmental_stage=developmental_stage,
            global_metrics=global_metrics,
            module_states=module_states,
            active_processes=active_processes,
            system_load=system_load,
            recent_experiences=recent_experiences,
            notes=snapshot_data["notes"]
        )
        
        return snapshot
    
    def get_recent_snapshots(
        self,
        limit: int = 10,
        since: Optional[datetime] = None,
        developmental_stage: Optional[Union[DevelopmentalStage, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent system snapshots.
        
        Args:
            limit: Maximum number of snapshots to return
            since: Only return snapshots after this time
            developmental_stage: Filter by developmental stage
            
        Returns:
            snapshots: List of snapshot summaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM system_snapshots"
        params = []
        
        # Add filters
        filters = []
        
        if since:
            filters.append("timestamp >= ?")
            params.append(since.isoformat())
            
        if developmental_stage:
            stage_value = developmental_stage.value if isinstance(developmental_stage, DevelopmentalStage) else developmental_stage
            filters.append("developmental_stage = ?")
            params.append(stage_value)
            
        if filters:
            query += " WHERE " + " AND ".join(filters)
            
        # Add sorting and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        
        # Convert to list of dictionaries
        snapshots = []
        for row in rows:
            snapshot = dict(zip(columns, row))
            
            # Add module count
            cursor.execute(
                "SELECT COUNT(*) FROM module_states WHERE snapshot_id = ?",
                (snapshot["snapshot_id"],)
            )
            module_count = cursor.fetchone()[0]
            snapshot["module_count"] = module_count
            
            # Parse JSON fields
            snapshot["global_metrics"] = json.loads(snapshot["global_metrics"] or "{}")
            snapshot["active_processes"] = json.loads(snapshot["active_processes"] or "[]")
            snapshot["system_load"] = json.loads(snapshot["system_load"] or "{}")
            snapshot["recent_experiences"] = json.loads(snapshot["recent_experiences"] or "[]")
            
            snapshots.append(snapshot)
            
        return snapshots
    
    def compare_snapshots(
        self,
        snapshot_id1: str,
        snapshot_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two system snapshots.
        
        Args:
            snapshot_id1: ID of first snapshot
            snapshot_id2: ID of second snapshot
            
        Returns:
            comparison: Snapshot comparison results
        """
        # Get snapshots
        snapshot1 = self.get_snapshot(snapshot_id1)
        snapshot2 = self.get_snapshot(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
            
        # Ensure snapshot1 is the earlier one
        if snapshot1.timestamp > snapshot2.timestamp:
            snapshot1, snapshot2 = snapshot2, snapshot1
            snapshot_id1, snapshot_id2 = snapshot_id2, snapshot_id1
            
        # Calculate time difference
        time_diff = (snapshot2.timestamp - snapshot1.timestamp).total_seconds()
        
        # Compare global metrics
        global_metrics_diff = {}
        for key in set(snapshot1.global_metrics.keys()).union(snapshot2.global_metrics.keys()):
            val1 = snapshot1.global_metrics.get(key)
            val2 = snapshot2.global_metrics.get(key)
            
            if val1 is not None and val2 is not None and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                global_metrics_diff[key] = {
                    "before": val1,
                    "after": val2,
                    "change": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                }
            else:
                global_metrics_diff[key] = {
                    "before": val1,
                    "after": val2,
                    "changed": val1 != val2
                }
                
        # Compare module states
        module_diffs = {}
        all_modules = set(snapshot1.module_states.keys()).union(snapshot2.module_states.keys())
        
        for module_name in all_modules:
            module_diff = {}
            
            # Handle modules that exist in only one snapshot
            if module_name not in snapshot1.module_states:
                module_diff["status"] = "new"
                module_diff["active"] = snapshot2.module_states[module_name].active
                module_diff["activation_level"] = snapshot2.module_states[module_name].activation_level
                module_diff["developmental_metrics"] = snapshot2.module_states[module_name].developmental_metrics
            elif module_name not in snapshot2.module_states:
                module_diff["status"] = "removed"
                module_diff["active"] = snapshot1.module_states[module_name].active
                module_diff["activation_level"] = snapshot1.module_states[module_name].activation_level
                module_diff["developmental_metrics"] = snapshot1.module_states[module_name].developmental_metrics
            else:
                # Module exists in both snapshots
                module1 = snapshot1.module_states[module_name]
                module2 = snapshot2.module_states[module_name]
                
                module_diff["status"] = "changed"
                module_diff["active"] = {
                    "before": module1.active,
                    "after": module2.active,
                    "changed": module1.active != module2.active
                }
                
                module_diff["activation_level"] = {
                    "before": module1.activation_level,
                    "after": module2.activation_level,
                    "change": module2.activation_level - module1.activation_level
                }
                
                # Compare connections
                conn_before = set(module1.connections.keys())
                conn_after = set(module2.connections.keys())
                
                module_diff["connections"] = {
                    "added": list(conn_after - conn_before),
                    "removed": list(conn_before - conn_after),
                    "changed": [
                        {
                            "name": c,
                            "before": module1.connections[c],
                            "after": module2.connections[c],
                            "change": module2.connections[c] - module1.connections[c]
                        }
                        for c in conn_before.intersection(conn_after)
                        if module1.connections[c] != module2.connections[c]
                    ],
                    "total_before": len(conn_before),
                    "total_after": len(conn_after)
                }
                
                # Compare developmental metrics
                dev_metrics_diff = {}
                for key in set(module1.developmental_metrics.keys()).union(module2.developmental_metrics.keys()):
                    val1 = module1.developmental_metrics.get(key)
                    val2 = module2.developmental_metrics.get(key)
                    
                    if val1 is not None and val2 is not None and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        dev_metrics_diff[key] = {
                            "before": val1,
                            "after": val2,
                            "change": val2 - val1,
                            "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                        }
                    else:
                        dev_metrics_diff[key] = {
                            "before": val1,
                            "after": val2,
                            "changed": val1 != val2
                        }
                        
                module_diff["developmental_metrics"] = dev_metrics_diff
                
            module_diffs[module_name] = module_diff
            
        # Create comparison result
        comparison = {
            "snapshot1_id": snapshot_id1,
            "snapshot2_id": snapshot_id2,
            "timestamp1": snapshot1.timestamp.isoformat(),
            "timestamp2": snapshot2.timestamp.isoformat(),
            "time_difference_seconds": time_diff,
            "developmental_stage1": snapshot1.developmental_stage.value,
            "developmental_stage2": snapshot2.developmental_stage.value,
            "stage_changed": snapshot1.developmental_stage != snapshot2.developmental_stage,
            "global_metrics_diff": global_metrics_diff,
            "module_diffs": module_diffs,
            "modules_added": [m for m in all_modules if m not in snapshot1.module_states],
            "modules_removed": [m for m in all_modules if m not in snapshot2.module_states]
        }
        
        return comparison
    
    def get_module_timeline(
        self,
        module_name: str,
        metric: str,
        timeframe_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of a specific metric for a module.
        
        Args:
            module_name: Name of the module
            metric: Name of the metric to track
            timeframe_days: Number of days to include
            
        Returns:
            timeline: Timeline data for the specified metric
        """
        since = datetime.now() - timedelta(days=timeframe_days)
        cursor = self.conn.cursor()
        
        # Get snapshots within the timeframe
        cursor.execute(
            '''
            SELECT snapshot_id, timestamp
            FROM system_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            ''',
            (since.isoformat(),)
        )
        
        snapshots = cursor.fetchall()
        results = []
        
        for snapshot_id, timestamp_str in snapshots:
            # Get module state for this snapshot
            cursor.execute(
                '''
                SELECT developmental_metrics
                FROM module_states
                WHERE snapshot_id = ? AND module_name = ?
                ''',
                (snapshot_id, module_name)
            )
            
            row = cursor.fetchone()
            if not row:
                continue
                
            # Parse metrics
            developmental_metrics = json.loads(row[0] or "{}")
            
            # Extract the requested metric
            if metric in developmental_metrics:
                results.append({
                    "timestamp": timestamp_str,
                    "value": developmental_metrics[metric],
                    "snapshot_id": snapshot_id
                })
                
        return results
    
    def analyze_activation_patterns(
        self,
        module: Optional[str] = None,
        pattern_type: Optional[str] = None,
        timeframe_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze neural activation patterns.
        
        Args:
            module: Optional specific module to analyze
            pattern_type: Optional specific pattern type to analyze
            timeframe_days: Number of days to include in analysis
            
        Returns:
            analysis: Analysis of activation patterns
        """
        since = datetime.now() - timedelta(days=timeframe_days)
        cursor = self.conn.cursor()
        
        # Build query
        query = "SELECT * FROM neural_activity WHERE timestamp >= ?"
        params = [since.isoformat()]
        
        if module:
            query += " AND module = ?"
            params.append(module)
            
        if pattern_type:
            query += " AND pattern_type = ?"
            params.append(pattern_type)
            
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        
        # Group by module and pattern type
        patterns_by_module: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        
        for row in rows:
            data = dict(zip(columns, row))
            
            # Parse activation map
            data["activation_map"] = json.loads(data["activation_map"] or "{}")
            
            module_name = data["module"]
            pattern = data["pattern_type"]
            
            patterns_by_module[module_name][pattern].append(data)
            
        # Analyze each group
        results = {}
        
        for mod, patterns in patterns_by_module.items():
            mod_results = {}
            
            for pat, instances in patterns.items():
                # Calculate statistics
                activity_levels = [inst["activity_level"] for inst in instances]
                durations = [inst["duration_ms"] for inst in instances]
                
                # Basic statistics
                stats = {
                    "count": len(instances),
                    "avg_activity_level": sum(activity_levels) / len(activity_levels) if activity_levels else 0,
                    "max_activity_level": max(activity_levels) if activity_levels else 0,
                    "min_activity_level": min(activity_levels) if activity_levels else 0,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                }
                
                # Analyze most frequent activation patterns
                all_regions = set()
                for inst in instances:
                    all_regions.update(inst["activation_map"].keys())
                    
                region_stats = {}
                for region in all_regions:
                    # Get activation values for this region
                    values = [inst["activation_map"].get(region, 0.0) for inst in instances]
                    
                    region_stats[region] = {
                        "avg_activation": sum(values) / len(values),
                        "max_activation": max(values),
                        "frequency": sum(1 for v in values if v > 0.1) / len(values)
                    }
                    
                # Sort regions by average activation
                top_regions = sorted(
                    region_stats.items(), 
                    key=lambda x: x[1]["avg_activation"], 
                    reverse=True
                )[:10]  # Top 10 regions
                
                mod_results[pat] = {
                    "statistics": stats,
                    "top_regions": dict(top_regions),
                    "first_timestamp": instances[0]["timestamp"],
                    "last_timestamp": instances[-1]["timestamp"],
                }
                
            results[mod] = mod_results
            
        return results
    
    def enable_auto_snapshots(
        self, 
        interval_seconds: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Enable automatic system snapshots at regular intervals.
        
        Note: This method sets up the configuration but actual scheduling
        would require an external scheduler or event loop to call check_auto_snapshot
        
        Args:
            interval_seconds: Interval between snapshots
            callback: Optional callback function receiving the snapshot_id
            
        Returns:
            success: Whether auto-snapshots were successfully enabled
        """
        if interval_seconds:
            self.snapshot_interval = interval_seconds
            
        self.is_auto_snapshot_enabled = True
        self.last_snapshot_time = datetime.now()
        self.snapshot_callback = callback
        
        logger.info(f"Enabled automatic system snapshots every {self.snapshot_interval} seconds")
        return True
    
    def disable_auto_snapshots(self) -> bool:
        """
        Disable automatic system snapshots.
        
        Returns:
            success: Whether auto-snapshots were successfully disabled
        """
        self.is_auto_snapshot_enabled = False
        logger.info("Disabled automatic system snapshots")
        return True
    
    def check_auto_snapshot(
        self, 
        developmental_stage: Union[DevelopmentalStage, str],
        global_metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Check if it's time for an automatic snapshot and take one if needed.
        
        Args:
            developmental_stage: Current developmental stage
            global_metrics: Current global metrics
            
        Returns:
            snapshot_id: ID of the snapshot taken, or None if no snapshot was taken
        """
        if not self.is_auto_snapshot_enabled:
            return None
            
        current_time = datetime.now()
        
        if not self.last_snapshot_time or (current_time - self.last_snapshot_time).total_seconds() >= self.snapshot_interval:
            # Time for a snapshot
            snapshot_id = self.take_system_snapshot(developmental_stage, global_metrics)
            if self.snapshot_callback:
                self.snapshot_callback(snapshot_id)
            return snapshot_id
        return None 
