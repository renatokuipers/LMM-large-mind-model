"""
Metrics tracking module for the Large Mind Model (LMM).

This module implements metrics tracking and visualization for the LMM,
including developmental metrics, emotional state, and memory statistics.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import os
import threading
import time

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger

logger = get_logger("lmm.visualization.metrics")

class MetricsTracker:
    """
    Tracks and records metrics for the LMM.
    
    This class provides methods for tracking, recording, and visualizing
    various metrics related to the LMM's development and state.
    """
    
    def __init__(self, lmm_instance=None, metrics_dir: str = "./metrics"):
        """
        Initialize the Metrics Tracker.
        
        Args:
            lmm_instance: Instance of the LargeMindsModel
            metrics_dir: Directory to store metrics data
        """
        self.lmm = lmm_instance
        self.metrics_dir = os.path.normpath(metrics_dir)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.developmental_metrics: List[Dict[str, Any]] = []
        self.emotional_metrics: List[Dict[str, Any]] = []
        self.memory_metrics: List[Dict[str, Any]] = []
        self.interaction_metrics: List[Dict[str, Any]] = []
        
        # Initialize tracking thread
        self.tracking_thread = None
        self.stop_tracking = threading.Event()
        self.tracking_interval = 60  # seconds
        
        logger.info("Initialized Metrics Tracker")
    
    def start_tracking(self):
        """Start tracking metrics in a background thread."""
        if not self.lmm:
            logger.warning("No LMM instance connected, metrics tracking not started")
            return
        
        self.stop_tracking.clear()
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        logger.info("Started metrics tracking")
    
    def stop_tracking(self):
        """Stop tracking metrics."""
        if self.tracking_thread:
            self.stop_tracking.set()
            self.tracking_thread.join(timeout=1.0)
            logger.info("Stopped metrics tracking")
    
    def _tracking_loop(self):
        """Background thread for tracking metrics."""
        while not self.stop_tracking.is_set():
            try:
                self.record_metrics()
            except Exception as e:
                logger.error(f"Error recording metrics: {e}")
            
            # Sleep for the tracking interval
            time.sleep(self.tracking_interval)
    
    def record_metrics(self):
        """Record current metrics."""
        if not self.lmm:
            return
        
        timestamp = datetime.now()
        
        # Record developmental metrics
        try:
            dev_status = self.lmm.get_development_status()
            dev_metrics = {
                "timestamp": timestamp.isoformat(),
                "stage": dev_status["stage"],
                "progress": dev_status["progress_percentage"],
                "metrics": dev_status["metrics"]
            }
            self.developmental_metrics.append(dev_metrics)
            
            # Save to file periodically
            if len(self.developmental_metrics) % 10 == 0:
                self._save_metrics("developmental", self.developmental_metrics)
        except Exception as e:
            logger.error(f"Error recording developmental metrics: {e}")
        
        # Record memory metrics
        try:
            memory_stats = self.lmm.get_memory_status()
            memory_metrics = {
                "timestamp": timestamp.isoformat(),
                "total_memories": memory_stats["total_memories"],
                "memory_counts_by_type": memory_stats["memory_counts_by_type"]
            }
            self.memory_metrics.append(memory_metrics)
            
            # Save to file periodically
            if len(self.memory_metrics) % 10 == 0:
                self._save_metrics("memory", self.memory_metrics)
        except Exception as e:
            logger.error(f"Error recording memory metrics: {e}")
        
        # Trim metrics lists if they get too long
        self._trim_metrics()
        
        logger.debug("Recorded metrics")
    
    def record_interaction(self, message: str, response: str, metrics: Dict[str, float]):
        """
        Record metrics for an interaction.
        
        Args:
            message: Message from the LMM
            response: Response from the Mother
            metrics: Learning metrics for the interaction
        """
        timestamp = datetime.now()
        
        # Record interaction metrics
        interaction_metrics = {
            "timestamp": timestamp.isoformat(),
            "message_length": len(message),
            "response_length": len(response),
            "learning_metrics": metrics
        }
        self.interaction_metrics.append(interaction_metrics)
        
        # Save to file periodically
        if len(self.interaction_metrics) % 10 == 0:
            self._save_metrics("interaction", self.interaction_metrics)
        
        logger.debug("Recorded interaction metrics")
    
    def _save_metrics(self, metrics_type: str, metrics_data: List[Dict[str, Any]]):
        """
        Save metrics to a file.
        
        Args:
            metrics_type: Type of metrics
            metrics_data: Metrics data to save
        """
        try:
            # Create metrics file path
            file_path = os.path.join(self.metrics_dir, f"{metrics_type}_metrics.json")
            
            # Save metrics to file
            with open(file_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.debug(f"Saved {metrics_type} metrics to {file_path}")
        except Exception as e:
            logger.error(f"Error saving {metrics_type} metrics: {e}")
    
    def _trim_metrics(self):
        """Trim metrics lists if they get too long."""
        max_metrics = 1000
        
        if len(self.developmental_metrics) > max_metrics:
            self.developmental_metrics = self.developmental_metrics[-max_metrics:]
        
        if len(self.emotional_metrics) > max_metrics:
            self.emotional_metrics = self.emotional_metrics[-max_metrics:]
        
        if len(self.memory_metrics) > max_metrics:
            self.memory_metrics = self.memory_metrics[-max_metrics:]
        
        if len(self.interaction_metrics) > max_metrics:
            self.interaction_metrics = self.interaction_metrics[-max_metrics:]
    
    def load_metrics(self):
        """Load metrics from files."""
        try:
            # Load developmental metrics
            dev_file = os.path.join(self.metrics_dir, "developmental_metrics.json")
            if os.path.exists(dev_file):
                with open(dev_file, "r") as f:
                    self.developmental_metrics = json.load(f)
            
            # Load memory metrics
            memory_file = os.path.join(self.metrics_dir, "memory_metrics.json")
            if os.path.exists(memory_file):
                with open(memory_file, "r") as f:
                    self.memory_metrics = json.load(f)
            
            # Load interaction metrics
            interaction_file = os.path.join(self.metrics_dir, "interaction_metrics.json")
            if os.path.exists(interaction_file):
                with open(interaction_file, "r") as f:
                    self.interaction_metrics = json.load(f)
            
            logger.info("Loaded metrics from files")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def get_developmental_metrics(self) -> List[Dict[str, Any]]:
        """
        Get developmental metrics.
        
        Returns:
            List of developmental metrics
        """
        return self.developmental_metrics
    
    def get_memory_metrics(self) -> List[Dict[str, Any]]:
        """
        Get memory metrics.
        
        Returns:
            List of memory metrics
        """
        return self.memory_metrics
    
    def get_interaction_metrics(self) -> List[Dict[str, Any]]:
        """
        Get interaction metrics.
        
        Returns:
            List of interaction metrics
        """
        return self.interaction_metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        summary = {
            "developmental": {
                "count": len(self.developmental_metrics),
                "latest": self.developmental_metrics[-1] if self.developmental_metrics else None
            },
            "memory": {
                "count": len(self.memory_metrics),
                "latest": self.memory_metrics[-1] if self.memory_metrics else None
            },
            "interaction": {
                "count": len(self.interaction_metrics),
                "latest": self.interaction_metrics[-1] if self.interaction_metrics else None
            }
        }
        
        return summary 