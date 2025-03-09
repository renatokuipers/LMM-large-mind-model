"""
Metrics Collector Module

This module provides functionality for collecting, aggregating, and analyzing
performance metrics from various cognitive modules and the overall system.
It supports both real-time collection and historical analysis of metrics.
"""

import os
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
import sqlite3
import numpy as np
from collections import defaultdict

from lmm_project.interfaces.researcher.models import (
    ResearchMetrics,
    MetricCategory,
    DevelopmentalStage,
    LearningAnalysis
)

# Set up logging
logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects and analyzes performance metrics from cognitive modules.
    
    This class provides functionality for:
    - Real-time collection of metrics from cognitive modules
    - Periodic sampling of system state
    - Statistical analysis of performance trends
    - Learning rate calculations
    - Performance anomaly detection
    - Integration with the development tracker
    """
    
    def __init__(
        self, 
        storage_dir: str = "storage/metrics",
        collection_interval: int = 60,  # seconds
        retention_days: int = 90
    ):
        """
        Initialize the MetricsCollector.
        
        Args:
            storage_dir: Directory to store collected metrics
            collection_interval: Default interval for automatic collection (seconds)
            retention_days: How long to retain detailed metrics history
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        
        # Initialize database for metrics storage
        self.db_path = self.storage_dir / "metrics.db"
        self.conn = self._initialize_database()
        
        # In-memory cache for recent metrics (for quick access)
        self.recent_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        
        # Registered modules for automatic collection
        self.registered_modules: Dict[str, Dict[str, Any]] = {}
        
        # Collection status
        self.is_collecting = False
        self.last_collection_time = None
        self.collection_count = 0
        
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            developmental_stage TEXT,
            session_id TEXT,
            source TEXT,
            metadata TEXT
        )
        ''')
        
        # Create learning analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_analysis (
            analysis_id TEXT PRIMARY KEY,
            module TEXT NOT NULL,
            learning_type TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            improvement_metrics TEXT NOT NULL,
            learning_rate REAL NOT NULL,
            plateau_detected INTEGER DEFAULT 0,
            efficiency REAL DEFAULT 0.5,
            correlated_experiences TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics (category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics (metric_name)')
        
        conn.commit()
        return conn
        
    def register_module(
        self, 
        module_name: str,
        metrics_extractor: Callable[[Any], Dict[str, Any]],
        module_instance: Any,
        collection_interval: Optional[int] = None
    ) -> bool:
        """
        Register a cognitive module for automatic metrics collection.
        
        Args:
            module_name: Name of the module to register
            metrics_extractor: Function to extract metrics from the module
            module_instance: Instance of the module to collect from
            collection_interval: Custom collection interval (seconds)
            
        Returns:
            success: Whether the module was successfully registered
        """
        if not callable(metrics_extractor):
            logger.error(f"Metrics extractor for {module_name} must be callable")
            return False
            
        self.registered_modules[module_name] = {
            "extractor": metrics_extractor,
            "instance": module_instance,
            "interval": collection_interval or self.collection_interval,
            "last_collection": None
        }
        
        logger.info(f"Registered module {module_name} for metrics collection")
        return True
        
    def unregister_module(self, module_name: str) -> bool:
        """
        Unregister a module from automatic metrics collection.
        
        Args:
            module_name: Name of the module to unregister
            
        Returns:
            success: Whether the module was successfully unregistered
        """
        if module_name in self.registered_modules:
            del self.registered_modules[module_name]
            logger.info(f"Unregistered module {module_name} from metrics collection")
            return True
        
        logger.warning(f"Module {module_name} was not registered")
        return False
        
    def collect_metrics(
        self, 
        modules: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect metrics from registered modules.
        
        Args:
            modules: Optional list of specific modules to collect from
            force: Whether to force collection regardless of interval
            
        Returns:
            collected_metrics: Dictionary of collected metrics by module
        """
        current_time = datetime.now()
        modules_to_collect = modules or list(self.registered_modules.keys())
        collected_metrics = {}
        
        for module_name in modules_to_collect:
            if module_name not in self.registered_modules:
                logger.warning(f"Module {module_name} is not registered for collection")
                continue
                
            module_info = self.registered_modules[module_name]
            
            # Check if it's time to collect from this module
            last_collection = module_info.get("last_collection")
            interval = module_info.get("interval", self.collection_interval)
            
            if not force and last_collection and (current_time - last_collection).total_seconds() < interval:
                logger.debug(f"Skipping collection for {module_name}: collection interval not reached")
                continue
                
            # Extract metrics from the module
            try:
                module_instance = module_info.get("instance")
                metrics_extractor = module_info.get("extractor")
                
                metrics = metrics_extractor(module_instance)
                
                # Update collection timestamp
                self.registered_modules[module_name]["last_collection"] = current_time
                
                # Store the collected metrics
                self.store_metrics(
                    category=module_name,
                    metrics=metrics,
                    timestamp=current_time
                )
                
                collected_metrics[module_name] = metrics
                logger.debug(f"Collected {len(metrics)} metrics from {module_name}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics from {module_name}: {str(e)}")
                
        self.last_collection_time = current_time
        self.collection_count += 1
        
        return collected_metrics
        
    def store_metrics(
        self,
        category: str,
        metrics: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        developmental_stage: Optional[Union[DevelopmentalStage, str]] = None,
        session_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store metrics in the database.
        
        Args:
            category: Category (usually module name) for these metrics
            metrics: Dictionary of metric values to store
            timestamp: When these metrics were collected
            developmental_stage: Current developmental stage
            session_id: Current session ID
            source: Source of these metrics
            metadata: Additional contextual information
            
        Returns:
            success: Whether metrics were successfully stored
        """
        timestamp = timestamp or datetime.now()
        source = source or "metrics_collector"
        
        # Handle stage if it's a string
        if isinstance(developmental_stage, str):
            try:
                stage_value = DevelopmentalStage(developmental_stage).value
            except ValueError:
                logger.warning(f"Invalid developmental stage: {developmental_stage}")
                stage_value = developmental_stage
        elif isinstance(developmental_stage, DevelopmentalStage):
            stage_value = developmental_stage.value
        else:
            stage_value = None
            
        # Store in database
        cursor = self.conn.cursor()
        
        try:
            for metric_name, metric_value in metrics.items():
                # Skip non-numeric and None values
                if metric_value is None or not isinstance(metric_value, (int, float, bool)):
                    continue
                    
                # Convert boolean to int
                if isinstance(metric_value, bool):
                    metric_value = 1 if metric_value else 0
                    
                cursor.execute(
                    '''
                    INSERT INTO metrics (
                        timestamp, category, metric_name, metric_value,
                        developmental_stage, session_id, source, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        timestamp.isoformat(),
                        category,
                        metric_name,
                        float(metric_value),
                        stage_value,
                        session_id,
                        source,
                        json.dumps(metadata) if metadata else None
                    )
                )
                
            self.conn.commit()
            
            # Also store in memory cache for quick access
            if category not in self.recent_metrics:
                self.recent_metrics[category] = {}
                
            cache_metrics = {}
            for metric_name, metric_value in metrics.items():
                if metric_value is not None and isinstance(metric_value, (int, float, bool)):
                    cache_metrics[metric_name] = float(metric_value) if isinstance(metric_value, bool) else metric_value
                    
            self.recent_metrics[category] = {
                "timestamp": timestamp,
                "metrics": cache_metrics,
                "developmental_stage": stage_value,
                "session_id": session_id
            }
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing metrics: {str(e)}")
            return False
            
    def get_recent_metrics(
        self,
        category: Optional[str] = None,
        metric_name: Optional[str] = None,
        timeframe_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get recent metrics from memory cache or database.
        
        Args:
            category: Optional category to filter by
            metric_name: Optional specific metric to retrieve
            timeframe_minutes: How far back to look for recent metrics
            
        Returns:
            metrics: Dictionary of recent metrics
        """
        # First try in-memory cache
        if category and category in self.recent_metrics:
            cache_entry = self.recent_metrics[category]
            cache_time = cache_entry.get("timestamp")
            
            if cache_time and (datetime.now() - cache_time).total_seconds() < timeframe_minutes * 60:
                cache_metrics = cache_entry.get("metrics", {})
                
                if metric_name:
                    if metric_name in cache_metrics:
                        return {
                            "category": category,
                            "metric_name": metric_name,
                            "value": cache_metrics[metric_name],
                            "timestamp": cache_time.isoformat()
                        }
                    else:
                        logger.debug(f"Metric {metric_name} not found in cache for {category}")
                else:
                    return {
                        "category": category,
                        "metrics": cache_metrics,
                        "timestamp": cache_time.isoformat()
                    }
                    
        # If not in cache or we need a wider timeframe, query the database
        since_time = (datetime.now() - timedelta(minutes=timeframe_minutes)).isoformat()
        cursor = self.conn.cursor()
        
        query = "SELECT category, metric_name, metric_value, timestamp FROM metrics WHERE timestamp > ?"
        params = [since_time]
        
        if category:
            query += " AND category = ?"
            params.append(category)
            
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
            
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return {}
            
        if category and metric_name:
            # Just return the most recent value for this specific metric
            if rows:
                cat, name, value, timestamp = rows[0]
                return {
                    "category": cat,
                    "metric_name": name,
                    "value": value,
                    "timestamp": timestamp
                }
            else:
                return {}
        elif category:
            # Return all recent metrics for this category
            result = {"category": category, "metrics": {}}
            latest_timestamp = None
            
            for cat, name, value, timestamp in rows:
                if name not in result["metrics"]:
                    result["metrics"][name] = value
                    
                    if not latest_timestamp or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        
            result["timestamp"] = latest_timestamp
            return result
        else:
            # Return metrics grouped by category
            result = {}
            
            for cat, name, value, timestamp in rows:
                if cat not in result:
                    result[cat] = {"metrics": {}, "timestamp": timestamp}
                    
                result[cat]["metrics"][name] = value
                
            return result
    
    def get_metric_history(
        self,
        category: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: Optional[str] = None,
        aggregation: str = "avg"
    ) -> List[Dict[str, Any]]:
        """
        Get historical values for a specific metric.
        
        Args:
            category: Category to retrieve metrics for
            metric_name: Name of the metric to retrieve
            start_time: Start of the time range
            end_time: End of the time range
            interval: Optional interval for aggregation (hour, day, week)
            aggregation: Aggregation function (avg, min, max, sum)
            
        Returns:
            history: List of historical metric values
        """
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=7))
        
        cursor = self.conn.cursor()
        
        if interval:
            # Use SQLite date functions for aggregation
            if interval == "hour":
                time_group = "strftime('%Y-%m-%d %H:00:00', timestamp)"
            elif interval == "day":
                time_group = "strftime('%Y-%m-%d', timestamp)"
            elif interval == "week":
                time_group = "strftime('%Y-%W', timestamp)"
            else:
                logger.warning(f"Unknown interval: {interval}, using raw data")
                time_group = None
                
            if time_group:
                # Select appropriate aggregation function
                if aggregation == "min":
                    agg_func = "MIN"
                elif aggregation == "max":
                    agg_func = "MAX"
                elif aggregation == "sum":
                    agg_func = "SUM"
                else:
                    agg_func = "AVG"
                    
                query = f"""
                SELECT {time_group} as period, {agg_func}(metric_value) as value
                FROM metrics
                WHERE category = ? AND metric_name = ? AND timestamp BETWEEN ? AND ?
                GROUP BY period
                ORDER BY period ASC
                """
                
                cursor.execute(
                    query,
                    (category, metric_name, start_time.isoformat(), end_time.isoformat())
                )
                
                return [
                    {"timestamp": period, "value": value}
                    for period, value in cursor.fetchall()
                ]
        else:
            # Return raw data points
            query = """
            SELECT timestamp, metric_value
            FROM metrics
            WHERE category = ? AND metric_name = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """
            
            cursor.execute(
                query,
                (category, metric_name, start_time.isoformat(), end_time.isoformat())
            )
            
            return [
                {"timestamp": timestamp, "value": value}
                for timestamp, value in cursor.fetchall()
            ]
            
    def analyze_learning(
        self,
        module: str,
        metric_name: str,
        learning_type: str,
        period_days: int = 7,
        smoothing_window: int = 5
    ) -> LearningAnalysis:
        """
        Analyze learning patterns for a specific metric.
        
        Args:
            module: Module to analyze
            metric_name: Metric to analyze
            learning_type: Type of learning being analyzed
            period_days: Number of days to analyze
            smoothing_window: Window size for smoothing
            
        Returns:
            analysis: LearningAnalysis object with findings
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_days)
        
        # Get metric history
        history = self.get_metric_history(
            category=module,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        if len(history) < 5:
            logger.warning(f"Insufficient data for learning analysis of {module}.{metric_name}")
            return LearningAnalysis(
                analysis_id=str(uuid.uuid4()),
                module=module,
                learning_type=learning_type,
                period_start=start_time,
                period_end=end_time,
                duration_seconds=0,
                improvement_metrics={},
                learning_rate=0.0,
                plateau_detected=False,
                efficiency=0.0,
                notes="Insufficient data for analysis"
            )
            
        # Extract timestamps and values
        timestamps = [datetime.fromisoformat(point["timestamp"]) for point in history]
        values = [point["value"] for point in history]
        
        # Calculate duration
        duration_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        
        # Apply smoothing if needed
        if len(values) >= smoothing_window and smoothing_window > 1:
            smoothed_values = []
            for i in range(len(values)):
                start_idx = max(0, i - smoothing_window // 2)
                end_idx = min(len(values), i + smoothing_window // 2 + 1)
                window = values[start_idx:end_idx]
                smoothed_values.append(sum(window) / len(window))
            values = smoothed_values
            
        # Calculate improvement metrics
        first_value = values[0]
        last_value = values[-1]
        min_value = min(values)
        max_value = max(values)
        
        improvement_metrics = {
            "first_value": first_value,
            "last_value": last_value,
            "min_value": min_value,
            "max_value": max_value,
            "absolute_change": last_value - first_value,
            "percent_change": ((last_value - first_value) / abs(first_value)) * 100 if first_value != 0 else 0
        }
        
        # Calculate learning rate
        # Simple approach: change per unit time
        if duration_seconds > 0:
            learning_rate = (last_value - first_value) / duration_seconds
        else:
            learning_rate = 0
            
        # Detect plateaus
        plateau_detected = False
        
        if len(values) >= 10:
            # Look at the last 25% of values
            plateau_window = values[-len(values)//4:]
            plateau_range = max(plateau_window) - min(plateau_window)
            max_range = max_value - min_value
            
            # If range in recent values is small compared to overall range, it's a plateau
            if max_range > 0 and plateau_range / max_range < 0.1:
                plateau_detected = True
                
        # Calculate efficiency
        # Efficiency is higher if learning occurred quickly with few fluctuations
        if max_value > min_value:
            # Calculate area under the curve using simple trapezoidal rule
            auc = 0
            for i in range(1, len(values)):
                auc += (values[i] + values[i-1]) / 2
                
            # Perfect learning would be a straight line to the maximum
            perfect_auc = (first_value + max_value) / 2 * len(values)
            
            # Efficiency is ratio of actual vs perfect (capped at 1.0)
            efficiency = min(1.0, auc / perfect_auc) if perfect_auc > 0 else 0.5
        else:
            efficiency = 0.5
            
        # Create analysis object
        analysis = LearningAnalysis(
            analysis_id=str(uuid.uuid4()),
            module=module,
            learning_type=learning_type,
            period_start=start_time,
            period_end=end_time,
            duration_seconds=duration_seconds,
            improvement_metrics=improvement_metrics,
            learning_rate=learning_rate,
            plateau_detected=plateau_detected,
            efficiency=efficiency
        )
        
        # Store analysis in database
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            INSERT INTO learning_analysis (
                analysis_id, module, learning_type, period_start, period_end,
                duration_seconds, improvement_metrics, learning_rate,
                plateau_detected, efficiency, correlated_experiences, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                analysis.analysis_id,
                analysis.module,
                analysis.learning_type,
                analysis.period_start.isoformat(),
                analysis.period_end.isoformat(),
                analysis.duration_seconds,
                json.dumps(analysis.improvement_metrics),
                analysis.learning_rate,
                int(analysis.plateau_detected),
                analysis.efficiency,
                json.dumps(analysis.correlated_experiences),
                analysis.notes,
                datetime.now().isoformat()
            )
        )
        self.conn.commit()
        
        return analysis
        
    def compare_metrics(
        self,
        metrics: List[Tuple[str, str]],  # List of (category, metric_name) pairs
        timeframe_days: int = 7,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple metrics over the same timeframe.
        
        Args:
            metrics: List of (category, metric_name) pairs to compare
            timeframe_days: Number of days to analyze
            normalize: Whether to normalize values for comparison
            
        Returns:
            comparison: Comparison results
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=timeframe_days)
        
        # Get history for each metric
        metric_histories = {}
        
        for category, metric_name in metrics:
            metric_key = f"{category}:{metric_name}"
            history = self.get_metric_history(
                category=category,
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time
            )
            
            if history:
                metric_histories[metric_key] = history
                
        if not metric_histories:
            logger.warning("No data available for metric comparison")
            return {"error": "No data available for specified metrics"}
            
        # Calculate correlation and other comparative statistics
        results = {
            "metrics": {},
            "correlations": {},
            "trends": {},
            "timeframe": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": timeframe_days
            }
        }
        
        # Calculate basic statistics for each metric
        for metric_key, history in metric_histories.items():
            values = [point["value"] for point in history]
            
            if not values:
                continue
                
            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "first": values[0],
                "last": values[-1],
                "change": values[-1] - values[0],
                "percent_change": ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
            }
            
            results["metrics"][metric_key] = stats
            
            # Calculate trend (simple linear regression)
            if len(values) > 1:
                x = np.arange(len(values))
                y = np.array(values)
                
                # Normalize x to avoid numerical issues
                x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else x
                
                # Linear regression
                slope, intercept = np.polyfit(x_norm, y, 1)
                
                results["trends"][metric_key] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                }
                
        # Calculate correlations between metrics
        # Note: This is a simple correlation and doesn't account for time alignment
        for i, (key1, history1) in enumerate(metric_histories.items()):
            values1 = [point["value"] for point in history1]
            
            for key2, history2 in list(metric_histories.items())[i+1:]:
                values2 = [point["value"] for point in history2]
                
                # Need to align time series (use simple approach - truncate to shorter length)
                min_length = min(len(values1), len(values2))
                
                if min_length < 3:
                    continue
                    
                aligned_values1 = values1[:min_length]
                aligned_values2 = values2[:min_length]
                
                # Normalize if requested
                if normalize:
                    v1_min, v1_max = min(aligned_values1), max(aligned_values1)
                    v2_min, v2_max = min(aligned_values2), max(aligned_values2)
                    
                    if v1_max > v1_min and v2_max > v2_min:
                        aligned_values1 = [(v - v1_min) / (v1_max - v1_min) for v in aligned_values1]
                        aligned_values2 = [(v - v2_min) / (v2_max - v2_min) for v in aligned_values2]
                        
                # Calculate correlation
                corr = np.corrcoef(aligned_values1, aligned_values2)[0, 1]
                
                # Add to results
                corr_key = f"{key1}|{key2}"
                results["correlations"][corr_key] = float(corr)
                
        return results
        
    def detect_anomalies(
        self,
        category: Optional[str] = None,
        timeframe_days: int = 7,
        threshold: float = 2.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies in metrics using simple statistical methods.
        
        Args:
            category: Optional category to focus on
            timeframe_days: Number of days to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            anomalies: Dictionary of detected anomalies by metric
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=timeframe_days)
        
        # Query for distinct categories and metrics
        cursor = self.conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT DISTINCT metric_name FROM metrics WHERE category = ? AND timestamp BETWEEN ? AND ?",
                (category, start_time.isoformat(), end_time.isoformat())
            )
            
            metrics_to_check = [(category, row[0]) for row in cursor.fetchall()]
        else:
            cursor.execute(
                "SELECT DISTINCT category, metric_name FROM metrics WHERE timestamp BETWEEN ? AND ?",
                (start_time.isoformat(), end_time.isoformat())
            )
            
            metrics_to_check = [(row[0], row[1]) for row in cursor.fetchall()]
            
        anomalies = {}
        
        for cat, metric_name in metrics_to_check:
            # Get metric history
            history = self.get_metric_history(
                category=cat,
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time
            )
            
            if len(history) < 10:  # Need sufficient data for meaningful detection
                continue
                
            values = [point["value"] for point in history]
            timestamps = [point["timestamp"] for point in history]
            
            # Calculate mean and standard deviation
            mean = sum(values) / len(values)
            std_dev = np.std(values) if len(values) > 1 else 0
            
            if std_dev == 0:
                continue  # Skip if no variation
                
            # Find values that deviate significantly
            metric_anomalies = []
            
            for i, value in enumerate(values):
                z_score = abs(value - mean) / std_dev
                
                if z_score > threshold:
                    anomaly = {
                        "timestamp": timestamps[i],
                        "value": value,
                        "z_score": float(z_score),
                        "mean": float(mean),
                        "std_dev": float(std_dev),
                        "deviation": float(value - mean)
                    }
                    metric_anomalies.append(anomaly)
                    
            if metric_anomalies:
                key = f"{cat}:{metric_name}"
                anomalies[key] = metric_anomalies
                
        return anomalies
        
    def clean_old_data(self, days_to_keep: Optional[int] = None) -> int:
        """
        Clean up old metrics data.
        
        Args:
            days_to_keep: Number of days of data to retain (defaults to retention_days)
            
        Returns:
            deleted_count: Number of records deleted
        """
        days_to_keep = days_to_keep or self.retention_days
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old metrics records")
        return deleted_count
        
    def close(self):
        """Close database connection and clean up resources."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()