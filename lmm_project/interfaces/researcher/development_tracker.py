"""
Development Tracker Module

This module provides functionality for tracking, recording, and analyzing
the developmental progress of the LMM system across different cognitive modules.
It records milestones, analyzes growth trajectories, and provides insights
into the overall cognitive development process.
"""

import os
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import sqlite3

from lmm_project.interfaces.researcher.models import (
    DevelopmentalStage,
    DevelopmentalMilestone,
    DevelopmentalEvent,
    DevelopmentalTrajectory,
    ResearchMetrics
)
from lmm_project.storage.experience_logger import ExperienceLogger
from lmm_project.storage.state_persistence import StatePersistence

# Set up logging
logger = logging.getLogger(__name__)

class DevelopmentTracker:
    """
    Tracks and analyzes the developmental progress of the LMM system.
    
    This class provides functionality for:
    - Recording and tracking developmental milestones
    - Analyzing growth trajectories across modules
    - Detecting developmental plateaus and accelerations
    - Comparing development across different cognitive domains
    - Predicting future developmental trajectories
    """
    
    def __init__(
        self, 
        storage_dir: str = "storage/development",
        experience_logger: Optional[ExperienceLogger] = None,
        state_persistence: Optional[StatePersistence] = None
    ):
        """
        Initialize the DevelopmentTracker.
        
        Args:
            storage_dir: Directory to store development tracking data
            experience_logger: Optional ExperienceLogger instance to use
            state_persistence: Optional StatePersistence instance to use
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to external systems if provided, or create new ones
        self.experience_logger = experience_logger or ExperienceLogger()
        self.state_persistence = state_persistence or StatePersistence()
        
        # Initialize database for milestone and event tracking
        self.db_path = self.storage_dir / "development_tracking.db"
        self.conn = self._initialize_database()
        
        # Load milestone definitions
        self.milestones_path = self.storage_dir / "milestone_definitions.json"
        self.milestones = self._load_milestones()
        
        # Metrics history for trajectory analysis
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for tracking developmental milestones and events."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create milestones table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS milestones (
            milestone_id TEXT PRIMARY KEY,
            module TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            achieved INTEGER DEFAULT 0,
            timestamp TEXT,
            developmental_stage TEXT NOT NULL,
            difficulty REAL DEFAULT 0.5,
            importance REAL DEFAULT 0.5,
            prerequisites TEXT,
            metrics_snapshot TEXT
        )
        ''')
        
        # Create developmental events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS developmental_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            module TEXT,
            developmental_stage TEXT,
            importance REAL DEFAULT 0.5,
            related_milestones TEXT,
            metrics_before TEXT,
            metrics_after TEXT,
            notes TEXT
        )
        ''')
        
        # Create metrics history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp TEXT NOT NULL,
            developmental_stage TEXT,
            session_id TEXT
        )
        ''')
        
        # Create trajectories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trajectories (
            trajectory_id TEXT PRIMARY KEY,
            module TEXT NOT NULL,
            metric TEXT NOT NULL,
            timeframe_start TEXT NOT NULL,
            timeframe_end TEXT NOT NULL,
            trend_type TEXT NOT NULL,
            trend_strength REAL NOT NULL,
            growth_rate REAL NOT NULL,
            data_points TEXT NOT NULL,
            plateaus TEXT,
            milestones_achieved TEXT,
            predicted_trajectory TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        return conn
    
    def _load_milestones(self) -> Dict[str, Dict[str, Any]]:
        """Load milestone definitions from JSON file."""
        if self.milestones_path.exists():
            with open(self.milestones_path, 'r') as f:
                return json.load(f)
        else:
            # Create empty milestone definitions file
            milestones = {}
            with open(self.milestones_path, 'w') as f:
                json.dump(milestones, f, indent=2)
            return milestones
        
    def define_milestone(self, milestone: DevelopmentalMilestone) -> str:
        """
        Define a new developmental milestone.
        
        Args:
            milestone: DevelopmentalMilestone object defining the milestone
            
        Returns:
            milestone_id: ID of the created milestone
        """
        milestone_dict = milestone.model_dump()
        
        # Add to database
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT OR REPLACE INTO milestones (
                milestone_id, module, name, description, achieved, timestamp,
                developmental_stage, difficulty, importance, prerequisites, metrics_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                milestone.milestone_id,
                milestone.module,
                milestone.name,
                milestone.description,
                int(milestone.achieved),
                milestone.timestamp.isoformat() if milestone.timestamp else None,
                milestone.developmental_stage.value,
                milestone.difficulty,
                milestone.importance,
                json.dumps(milestone.prerequisites),
                json.dumps(milestone.metrics_snapshot)
            )
        )
        self.conn.commit()
        
        # Also add to JSON file for easier querying
        self.milestones[milestone.milestone_id] = milestone_dict
        with open(self.milestones_path, 'w') as f:
            json.dump(self.milestones, f, indent=2)
            
        return milestone.milestone_id
        
    def record_milestone_achievement(
        self, 
        milestone_id: str,
        timestamp: Optional[datetime] = None,
        metrics_snapshot: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record the achievement of a developmental milestone.
        
        Args:
            milestone_id: ID of the milestone that was achieved
            timestamp: When the milestone was achieved (defaults to now)
            metrics_snapshot: Snapshot of relevant metrics at achievement time
            
        Returns:
            success: Whether the milestone was successfully recorded
        """
        timestamp = timestamp or datetime.now()
        
        # Update milestone in database
        cursor = self.conn.cursor()
        
        # First check if milestone exists
        cursor.execute("SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,))
        if cursor.fetchone() is None:
            logger.error(f"Cannot record achievement for undefined milestone: {milestone_id}")
            return False
        
        # Update milestone
        cursor.execute(
            '''
            UPDATE milestones 
            SET achieved = 1, timestamp = ?, metrics_snapshot = ?
            WHERE milestone_id = ?
            ''',
            (
                timestamp.isoformat(),
                json.dumps(metrics_snapshot) if metrics_snapshot else None,
                milestone_id
            )
        )
        self.conn.commit()
        
        # Update in memory cache as well
        if milestone_id in self.milestones:
            self.milestones[milestone_id]["achieved"] = True
            self.milestones[milestone_id]["timestamp"] = timestamp.isoformat()
            if metrics_snapshot:
                self.milestones[milestone_id]["metrics_snapshot"] = metrics_snapshot
            
            with open(self.milestones_path, 'w') as f:
                json.dump(self.milestones, f, indent=2)
        
        # Create a developmental event for this achievement
        milestone_data = self.get_milestone(milestone_id)
        if milestone_data:
            self.record_developmental_event(
                event_type="milestone_achieved",
                description=f"Achieved milestone: {milestone_data.get('name')}",
                module=milestone_data.get('module'),
                developmental_stage=milestone_data.get('developmental_stage'),
                importance=milestone_data.get('importance', 0.5),
                related_milestones=[milestone_id],
                metrics_snapshot=metrics_snapshot
            )
        
        logger.info(f"Recorded milestone achievement: {milestone_id}")
        return True
    
    def record_developmental_event(
        self,
        event_type: str,
        description: str,
        module: Optional[str] = None,
        developmental_stage: Optional[Union[DevelopmentalStage, str]] = None,
        importance: float = 0.5,
        related_milestones: Optional[List[str]] = None,
        metrics_before: Optional[Dict[str, Any]] = None,
        metrics_after: Optional[Dict[str, Any]] = None,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Record a significant developmental event.
        
        Args:
            event_type: Type of developmental event
            description: Description of the event
            module: Related cognitive module
            developmental_stage: Current developmental stage
            importance: Importance of the event (0.0-1.0)
            related_milestones: List of related milestone IDs
            metrics_before: Metrics before the event
            metrics_after: Metrics after the event
            metrics_snapshot: Combined metrics snapshot
            notes: Additional notes
            timestamp: When the event occurred (defaults to now)
            
        Returns:
            event_id: ID of the created event
        """
        timestamp = timestamp or datetime.now()
        event_id = str(uuid.uuid4())
        
        # If only metrics_snapshot is provided, use it for both before/after
        if metrics_snapshot and not metrics_before and not metrics_after:
            metrics_after = metrics_snapshot
        
        # Handle stage if it's a string
        if isinstance(developmental_stage, str):
            try:
                developmental_stage = DevelopmentalStage(developmental_stage)
            except ValueError:
                logger.warning(f"Invalid developmental stage: {developmental_stage}")
                developmental_stage = None
        
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO developmental_events (
                event_id, event_type, description, timestamp, module,
                developmental_stage, importance, related_milestones,
                metrics_before, metrics_after, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                event_id,
                event_type,
                description,
                timestamp.isoformat(),
                module,
                developmental_stage.value if developmental_stage else None,
                importance,
                json.dumps(related_milestones) if related_milestones else None,
                json.dumps(metrics_before) if metrics_before else None,
                json.dumps(metrics_after) if metrics_after else None,
                notes
            )
        )
        self.conn.commit()
        
        # Also log to the experience logger for timeline integration
        self.experience_logger.log_experience(
            experience_data={
                "event_id": event_id,
                "event_type": event_type,
                "description": description,
                "module": module,
                "developmental_stage": developmental_stage.value if developmental_stage else None,
                "related_milestones": related_milestones,
                "notes": notes
            },
            experience_type="developmental_event",
            source="development_tracker",
            emotional_valence="positive" if importance > 0.7 else "neutral",
            emotional_intensity=importance,
            importance_score=importance,
            tags=["development", event_type, module] if module else ["development", event_type],
            metadata={
                "metrics_before": metrics_before,
                "metrics_after": metrics_after
            }
        )
        
        logger.info(f"Recorded developmental event: {event_type} - {description}")
        return event_id
    
    def record_metrics(
        self,
        metrics: Union[ResearchMetrics, Dict[str, Any]],
        module: Optional[str] = None,
        developmental_stage: Optional[Union[DevelopmentalStage, str]] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record research metrics for analysis.
        
        Args:
            metrics: ResearchMetrics object or dictionary of metric values
            module: Related cognitive module
            developmental_stage: Current developmental stage
            session_id: Current session ID
            timestamp: When metrics were collected (defaults to now)
            
        Returns:
            success: Whether metrics were successfully recorded
        """
        timestamp = timestamp or datetime.now()
        
        # Convert to ResearchMetrics if needed
        if isinstance(metrics, dict):
            if not module:
                logger.error("Module name required when providing metrics as dictionary")
                return False
            
            metrics_obj = ResearchMetrics(
                category=module,
                metrics=metrics,
                timestamp=timestamp,
                developmental_stage=developmental_stage,
                session_id=session_id
            )
        else:
            metrics_obj = metrics
            
        # Override module if provided
        if module:
            metrics_obj.category = module
            
        # Override stage if provided
        if developmental_stage:
            if isinstance(developmental_stage, str):
                try:
                    metrics_obj.developmental_stage = DevelopmentalStage(developmental_stage)
                except ValueError:
                    logger.warning(f"Invalid developmental stage: {developmental_stage}")
            else:
                metrics_obj.developmental_stage = developmental_stage
                
        # Override session if provided
        if session_id:
            metrics_obj.session_id = session_id
            
        # Override timestamp if provided
        if timestamp:
            metrics_obj.timestamp = timestamp
            
        # Store each individual metric in the database
        cursor = self.conn.cursor()
        for metric_name, value in metrics_obj.metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
                
            cursor.execute(
                '''
                INSERT INTO metrics_history (
                    module, metric_name, value, timestamp, developmental_stage, session_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    metrics_obj.category,
                    metric_name,
                    float(value),
                    metrics_obj.timestamp.isoformat(),
                    metrics_obj.developmental_stage.value if metrics_obj.developmental_stage else None,
                    metrics_obj.session_id
                )
            )
            
        self.conn.commit()
        
        # Add to in-memory cache for trajectory analysis
        module_key = metrics_obj.category
        if module_key not in self.metrics_history:
            self.metrics_history[module_key] = []
            
        self.metrics_history[module_key].append({
            "timestamp": metrics_obj.timestamp,
            "metrics": metrics_obj.metrics,
            "developmental_stage": metrics_obj.developmental_stage.value if metrics_obj.developmental_stage else None,
            "session_id": metrics_obj.session_id
        })
        
        # Trim in-memory cache to last 1000 entries per module
        if len(self.metrics_history[module_key]) > 1000:
            self.metrics_history[module_key] = self.metrics_history[module_key][-1000:]
            
        return True
        
    def analyze_trajectory(
        self,
        module: str,
        metric: str,
        timeframe_days: int = 30,
        end_time: Optional[datetime] = None
    ) -> Optional[DevelopmentalTrajectory]:
        """
        Analyze the developmental trajectory for a specific metric.
        
        Args:
            module: Cognitive module to analyze
            metric: Specific metric to analyze
            timeframe_days: Number of days to analyze
            end_time: End of analysis period (defaults to now)
            
        Returns:
            trajectory: DevelopmentalTrajectory analysis or None if insufficient data
        """
        end_time = end_time or datetime.now()
        start_time = end_time - timedelta(days=timeframe_days)
        
        # Query the database for metric history
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT value, timestamp, developmental_stage
            FROM metrics_history
            WHERE module = ? AND metric_name = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            ''',
            (
                module,
                metric,
                start_time.isoformat(),
                end_time.isoformat()
            )
        )
        
        rows = cursor.fetchall()
        if len(rows) < 5:  # Require at least 5 data points for analysis
            logger.warning(f"Insufficient data for trajectory analysis of {module}.{metric}")
            return None
            
        # Convert to data points
        data_points = []
        values = []
        timestamps = []
        
        for value, timestamp_str, stage in rows:
            timestamp = datetime.fromisoformat(timestamp_str)
            values.append(value)
            timestamps.append(timestamp)
            data_points.append({
                "value": value,
                "timestamp": timestamp_str,
                "developmental_stage": stage
            })
            
        # Simple linear regression for trend analysis
        x = np.array([(t - start_time).total_seconds() for t in timestamps])
        y = np.array(values)
        
        # Normalize x to avoid numerical issues
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else x
        
        # Calculate trend
        if len(x) > 1:
            slope, intercept = np.polyfit(x_norm, y, 1)
            
            # Calculate R^2 (coefficient of determination)
            y_pred = slope * x_norm + intercept
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Determine trend type
            if abs(slope) < 0.01:
                trend_type = "stable"
            elif slope > 0:
                trend_type = "increasing"
            else:
                trend_type = "decreasing"
                
            # Normalized growth rate (percent change over the period)
            if values[0] != 0:
                growth_rate = (values[-1] - values[0]) / values[0]
            else:
                growth_rate = 0 if values[-1] == 0 else float('inf')
        else:
            trend_type = "insufficient_data"
            r_squared = 0
            growth_rate = 0
            
        # Detect plateaus (periods of little change)
        plateaus = []
        if len(values) > 10:
            plateau_threshold = 0.05  # 5% change threshold
            window_size = max(3, len(values) // 10)
            
            for i in range(0, len(values) - window_size):
                window_values = values[i:i+window_size]
                max_change = (max(window_values) - min(window_values)) / (max(window_values) + 1e-10)
                
                if max_change < plateau_threshold:
                    plateau = {
                        "start_index": i,
                        "end_index": i + window_size - 1,
                        "start_time": timestamps[i].isoformat(),
                        "end_time": timestamps[i + window_size - 1].isoformat(),
                        "avg_value": sum(window_values) / len(window_values),
                        "duration_hours": (timestamps[i + window_size - 1] - timestamps[i]).total_seconds() / 3600
                    }
                    plateaus.append(plateau)
        
        # Get milestones achieved during this timeframe
        cursor.execute(
            '''
            SELECT milestone_id
            FROM milestones
            WHERE module = ? AND achieved = 1 AND timestamp BETWEEN ? AND ?
            ''',
            (
                module,
                start_time.isoformat(),
                end_time.isoformat()
            )
        )
        milestones_achieved = [row[0] for row in cursor.fetchall()]
        
        # Create a simple prediction based on the trend
        predicted_trajectory = None
        if len(values) > 10 and trend_type != "stable":
            # Project forward 20% of the current timeframe
            prediction_days = timeframe_days * 0.2
            prediction_seconds = prediction_days * 24 * 60 * 60
            
            # Use last timestamp and value as starting point
            last_timestamp = timestamps[-1]
            last_value = values[-1]
            
            # Create prediction points
            prediction_points = []
            num_points = 5  # Number of prediction points
            
            for i in range(1, num_points + 1):
                # Calculate predicted timestamp and value
                point_seconds = (i / num_points) * prediction_seconds
                pred_timestamp = last_timestamp + timedelta(seconds=point_seconds)
                
                # Simple linear prediction
                if trend_type == "increasing":
                    # Apply some damping to avoid unrealistic growth
                    damping = 0.9 ** i
                    pred_value = last_value + (slope * point_seconds * damping)
                elif trend_type == "decreasing":
                    # Apply floor to avoid negative values
                    pred_value = max(0, last_value + (slope * point_seconds))
                else:
                    pred_value = last_value
                    
                prediction_points.append({
                    "timestamp": pred_timestamp.isoformat(),
                    "value": pred_value,
                    "is_prediction": True
                })
                
            predicted_trajectory = prediction_points
        
        # Create trajectory object
        trajectory_id = str(uuid.uuid4())
        trajectory = DevelopmentalTrajectory(
            module=module,
            metric=metric,
            timeframe_start=start_time,
            timeframe_end=end_time,
            data_points=data_points,
            trend_type=trend_type,
            trend_strength=float(r_squared),
            growth_rate=float(growth_rate),
            plateaus=plateaus,
            milestones_achieved=milestones_achieved,
            predicted_trajectory=predicted_trajectory
        )
        
        # Store in database
        cursor.execute(
            '''
            INSERT INTO trajectories (
                trajectory_id, module, metric, timeframe_start, timeframe_end,
                trend_type, trend_strength, growth_rate, data_points,
                plateaus, milestones_achieved, predicted_trajectory, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                trajectory_id,
                module,
                metric,
                start_time.isoformat(),
                end_time.isoformat(),
                trend_type,
                float(r_squared),
                float(growth_rate),
                json.dumps(data_points),
                json.dumps(plateaus),
                json.dumps(milestones_achieved),
                json.dumps(predicted_trajectory) if predicted_trajectory else None,
                datetime.now().isoformat()
            )
        )
        self.conn.commit()
        
        return trajectory
        
    def detect_developmental_plateaus(
        self,
        module: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        timeframe_days: int = 30,
        plateau_threshold: float = 0.05
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect developmental plateaus across modules and metrics.
        
        Args:
            module: Optional specific module to analyze
            metrics: Optional list of specific metrics to analyze
            timeframe_days: Number of days to analyze
            plateau_threshold: Threshold for plateau detection (lower = more sensitive)
            
        Returns:
            plateaus: Dictionary of detected plateaus by module and metric
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=timeframe_days)
        
        # Query modules if not specified
        cursor = self.conn.cursor()
        if not module:
            cursor.execute("SELECT DISTINCT module FROM metrics_history")
            modules = [row[0] for row in cursor.fetchall()]
        else:
            modules = [module]
            
        results = {}
        
        for mod in modules:
            # Query metrics if not specified
            if not metrics:
                cursor.execute(
                    "SELECT DISTINCT metric_name FROM metrics_history WHERE module = ?",
                    (mod,)
                )
                mod_metrics = [row[0] for row in cursor.fetchall()]
            else:
                mod_metrics = metrics
                
            mod_plateaus = []
            
            for metric in mod_metrics:
                # Analyze trajectory and extract plateaus
                trajectory = self.analyze_trajectory(
                    module=mod,
                    metric=metric,
                    timeframe_days=timeframe_days,
                    end_time=end_time
                )
                
                if trajectory and trajectory.plateaus:
                    # Filter plateaus by threshold and minimum duration
                    significant_plateaus = [
                        {**p, "metric": metric}
                        for p in trajectory.plateaus
                        if p.get("duration_hours", 0) > 24  # At least one day
                    ]
                    
                    if significant_plateaus:
                        mod_plateaus.extend(significant_plateaus)
            
            if mod_plateaus:
                results[mod] = mod_plateaus
                
        return results
        
    def get_milestone(self, milestone_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific milestone.
        
        Args:
            milestone_id: ID of the milestone to retrieve
            
        Returns:
            milestone: Milestone information or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,))
        columns = [col[0] for col in cursor.description]
        row = cursor.fetchone()
        
        if not row:
            return None
            
        milestone = dict(zip(columns, row))
        
        # Parse JSON fields
        milestone["prerequisites"] = json.loads(milestone["prerequisites"] or "[]")
        milestone["metrics_snapshot"] = json.loads(milestone["metrics_snapshot"] or "{}")
        milestone["achieved"] = bool(milestone["achieved"])
        
        return milestone
        
    def get_milestones_by_module(
        self,
        module: str,
        include_achieved: bool = True,
        include_pending: bool = True,
        developmental_stage: Optional[Union[DevelopmentalStage, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get milestones for a specific module.
        
        Args:
            module: Module to get milestones for
            include_achieved: Whether to include achieved milestones
            include_pending: Whether to include pending milestones
            developmental_stage: Optional filter by developmental stage
            
        Returns:
            milestones: List of milestone information
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM milestones WHERE module = ?"
        params = [module]
        
        # Filter by achievement status
        if include_achieved and not include_pending:
            query += " AND achieved = 1"
        elif include_pending and not include_achieved:
            query += " AND achieved = 0"
            
        # Filter by developmental stage
        if developmental_stage:
            stage_value = developmental_stage.value if isinstance(developmental_stage, DevelopmentalStage) else developmental_stage
            query += " AND developmental_stage = ?"
            params.append(stage_value)
            
        query += " ORDER BY importance DESC, difficulty ASC"
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            milestone = dict(zip(columns, row))
            
            # Parse JSON fields
            milestone["prerequisites"] = json.loads(milestone["prerequisites"] or "[]")
            milestone["metrics_snapshot"] = json.loads(milestone["metrics_snapshot"] or "{}")
            milestone["achieved"] = bool(milestone["achieved"])
            
            results.append(milestone)
            
        return results
        
    def get_recent_developmental_events(
        self, 
        limit: int = 20,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        importance_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get recent developmental events.
        
        Args:
            limit: Maximum number of events to return
            module: Optional filter by module
            event_type: Optional filter by event type
            importance_threshold: Minimum importance level
            
        Returns:
            events: List of recent developmental events
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM developmental_events WHERE importance >= ?"
        params = [importance_threshold]
        
        if module:
            query += " AND module = ?"
            params.append(module)
            
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            event = dict(zip(columns, row))
            
            # Parse JSON fields
            event["related_milestones"] = json.loads(event["related_milestones"] or "[]")
            event["metrics_before"] = json.loads(event["metrics_before"] or "{}")
            event["metrics_after"] = json.loads(event["metrics_after"] or "{}")
            
            results.append(event)
            
        return results
        
    def get_developmental_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current developmental status across all modules.
        
        Returns:
            summary: Developmental summary information
        """
        cursor = self.conn.cursor()
        
        # Get milestone stats
        cursor.execute(
            '''
            SELECT 
                module,
                COUNT(*) as total_milestones,
                SUM(achieved) as achieved_milestones,
                developmental_stage
            FROM milestones
            GROUP BY module, developmental_stage
            '''
        )
        
        milestone_stats = {}
        for module, total, achieved, stage in cursor.fetchall():
            if module not in milestone_stats:
                milestone_stats[module] = {"stages": {}}
                
            milestone_stats[module]["stages"][stage] = {
                "total": total,
                "achieved": achieved,
                "progress": achieved / total if total > 0 else 0
            }
            
            # Calculate overall stats per module
            if "overall" not in milestone_stats[module]:
                milestone_stats[module]["overall"] = {
                    "total": 0,
                    "achieved": 0,
                    "progress": 0
                }
                
            milestone_stats[module]["overall"]["total"] += total
            milestone_stats[module]["overall"]["achieved"] += achieved
            
        # Calculate overall progress percentages
        for module in milestone_stats:
            total = milestone_stats[module]["overall"]["total"]
            achieved = milestone_stats[module]["overall"]["achieved"]
            milestone_stats[module]["overall"]["progress"] = achieved / total if total > 0 else 0
        
        # Get most recent metrics for each module
        cursor.execute(
            '''
            SELECT m1.module, m1.metric_name, m1.value, m1.timestamp
            FROM metrics_history m1
            INNER JOIN (
                SELECT module, metric_name, MAX(timestamp) as max_time
                FROM metrics_history
                GROUP BY module, metric_name
            ) m2 ON m1.module = m2.module AND m1.metric_name = m2.metric_name AND m1.timestamp = m2.max_time
            '''
        )
        
        recent_metrics = {}
        for module, metric, value, timestamp in cursor.fetchall():
            if module not in recent_metrics:
                recent_metrics[module] = {}
                
            recent_metrics[module][metric] = {
                "value": value,
                "timestamp": timestamp
            }
            
        # Get developmental events from last 7 days
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute(
            '''
            SELECT module, event_type, COUNT(*) as count
            FROM developmental_events
            WHERE timestamp > ?
            GROUP BY module, event_type
            ''',
            (week_ago,)
        )
        
        recent_events = {}
        for module, event_type, count in cursor.fetchall():
            if module not in recent_events:
                recent_events[module] = {}
                
            recent_events[module][event_type] = count
            
        # Get active developmental plateaus
        plateaus = self.detect_developmental_plateaus(timeframe_days=30)
        
        # Compile summary
        summary = {
            "milestone_stats": milestone_stats,
            "recent_metrics": recent_metrics,
            "recent_events": recent_events,
            "developmental_plateaus": plateaus,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
        
    def close(self):
        """Close database connection and save state."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()