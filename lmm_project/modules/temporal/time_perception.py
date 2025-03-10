# TODO: Implement the TimePerception class to track and estimate time intervals
# This component should be able to:
# - Track the passage of time
# - Estimate durations of events and intervals
# - Synchronize internal processes with temporal rhythms
# - Develop a sense of past, present, and future

# TODO: Implement developmental progression in time perception:
# - Basic rhythmic awareness in early stages
# - Growing time interval discrimination in childhood
# - Extended time horizons in adolescence
# - Sophisticated temporal cognition in adulthood

# TODO: Create mechanisms for:
# - Time tracking: Monitor the passage of time
# - Duration estimation: Judge the length of intervals
# - Temporal integration: Connect events across time
# - Temporal organization: Structure experiences in time

# TODO: Implement different temporal scales:
# - Millisecond timing: For perceptual processes
# - Second-to-minute timing: For immediate action
# - Hour-to-day timing: For activity planning
# - Extended time perception: Past history and future projection

# TODO: Connect to memory and consciousness modules
# Time perception should interact with memory processes
# and contribute to conscious awareness of time

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
import uuid
import time

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient

from lmm_project.modules.temporal.models import TimeInterval, TemporalRhythm, TemporalContext
from lmm_project.modules.temporal.neural_net import TimePerceptionNetwork

logger = logging.getLogger(__name__)

class TimePerception(BaseModule):
    """
    Tracks and estimates time intervals
    
    This module monitors the passage of time, estimates
    durations, synchronizes with temporal rhythms, and
    develops awareness of past, present, and future.
    """
    
    # Override developmental milestones with time perception-specific milestones
    development_milestones = {
        0.0: "Basic time awareness",
        0.2: "Simple interval discrimination",
        0.4: "Temporal pattern recognition",
        0.6: "Extended temporal horizon",
        0.8: "Multiple timescale integration",
        1.0: "Sophisticated temporal cognition"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the time perception module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="time_perception", event_bus=event_bus)
        
        # Initialize time tracking mechanisms
        self.time_intervals: Dict[str, TimeInterval] = {}
        self.active_intervals: Dict[str, TimeInterval] = {}  # Currently ongoing intervals
        self.system_start_time = datetime.now()
        self.last_update_time = self.system_start_time
        
        # Set up duration estimation capability
        self.duration_estimates: Dict[str, Dict[str, float]] = defaultdict(dict)  # context -> event_type -> duration
        self.estimation_errors: Dict[str, List[float]] = defaultdict(list)  # event_type -> error_history
        
        # Create temporal integration processes
        self.event_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.event_timestamps: Dict[str, List[datetime]] = defaultdict(list)  # event_type -> timestamps
        
        # Initialize temporal organization structures
        self.temporal_rhythms: Dict[str, TemporalRhythm] = {}
        self.temporal_context = TemporalContext()
        
        # Neural network for time estimation
        self.time_perception_network = TimePerceptionNetwork()
        
        # Time scales and perception ranges (seconds)
        # The development level affects which scales are active
        self.time_scales = {
            "millisecond": {"range": (0.001, 1.0), "min_development": 0.0},  # Always active
            "second": {"range": (1.0, 60.0), "min_development": 0.0},  # Always active
            "minute": {"range": (60.0, 3600.0), "min_development": 0.2},  # Requires development
            "hour": {"range": (3600.0, 86400.0), "min_development": 0.4},  # Requires development
            "day": {"range": (86400.0, 604800.0), "min_development": 0.6},  # Requires development
            "week": {"range": (604800.0, 2592000.0), "min_development": 0.8},  # Requires development
            "month": {"range": (2592000.0, 31536000.0), "min_development": 0.9}  # Requires development
        }
        
        # Embedding client for semantic processing
        self.embedding_client = LLMClient()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self.subscribe_to_message("event_start", self._handle_event_start)
            self.subscribe_to_message("event_end", self._handle_event_end)
            self.subscribe_to_message("heartbeat", self._handle_heartbeat)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to track and estimate time
        
        Args:
            input_data: Dictionary containing temporal information
            
        Returns:
            Dictionary with time perception results
        """
        # Determine what type of input we're processing
        input_type = input_data.get("input_type", "")
        
        if input_type == "start_interval":
            return self._process_start_interval(input_data)
        elif input_type == "end_interval":
            return self._process_end_interval(input_data)
        elif input_type == "estimate_duration":
            return self._process_estimate_duration(input_data)
        elif input_type == "detect_rhythm":
            return self._process_detect_rhythm(input_data)
        elif input_type == "update_context":
            return self._process_update_context(input_data)
        else:
            # Default to system update
            self._update_system_time()
            return {
                "current_time": datetime.now().isoformat(),
                "system_uptime": (datetime.now() - self.system_start_time).total_seconds(),
                "active_intervals": len(self.active_intervals),
                "detected_rhythms": len(self.temporal_rhythms)
            }
    
    def _process_start_interval(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start tracking a time interval"""
        event_type = input_data.get("event_type")
        context = input_data.get("context", {})
        start_time = input_data.get("start_time", datetime.now())
        
        if not event_type:
            return {"error": "Event type is required"}
        
        # Create interval object
        interval_id = str(uuid.uuid4())
        interval = TimeInterval(
            id=interval_id,
            start_time=start_time,
            context=context,
            events=[event_type]
        )
        
        # Store interval and mark as active
        self.time_intervals[interval_id] = interval
        self.active_intervals[interval_id] = interval
        
        # Record event
        self._record_event({
            "type": "interval_start",
            "event_type": event_type,
            "interval_id": interval_id,
            "timestamp": start_time,
            "context": context
        })
        
        # Update the temporal context
        self.temporal_context.active_intervals.append(interval_id)
        
        return {
            "interval_id": interval_id,
            "start_time": start_time.isoformat(),
            "event_type": event_type
        }
    
    def _process_end_interval(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """End tracking a time interval"""
        interval_id = input_data.get("interval_id")
        end_time = input_data.get("end_time", datetime.now())
        
        if not interval_id or interval_id not in self.active_intervals:
            return {"error": "Valid active interval ID is required"}
        
        # Get interval and update end time
        interval = self.active_intervals[interval_id]
        interval.end_time = end_time
        
        # Calculate actual duration
        start_time = interval.start_time
        duration_seconds = (end_time - start_time).total_seconds()
        interval.actual_duration = duration_seconds
        
        # Record the interval duration for this event type
        event_type = interval.events[0] if interval.events else "unknown"
        context_key = str(interval.context.get("type", "general"))
        
        # Update duration estimates
        if event_type in self.duration_estimates[context_key]:
            # Update existing estimate with weighted average
            old_estimate = self.duration_estimates[context_key][event_type]
            # Give more weight to past estimates as development increases
            weight = min(0.9, 0.5 + self.development_level * 0.4)
            new_estimate = old_estimate * weight + duration_seconds * (1.0 - weight)
            self.duration_estimates[context_key][event_type] = new_estimate
        else:
            # First occurrence of this event type in this context
            self.duration_estimates[context_key][event_type] = duration_seconds
        
        # Remove from active intervals
        del self.active_intervals[interval_id]
        
        # Remove from temporal context
        if interval_id in self.temporal_context.active_intervals:
            self.temporal_context.active_intervals.remove(interval_id)
        
        # Record event
        self._record_event({
            "type": "interval_end",
            "event_type": event_type,
            "interval_id": interval_id,
            "timestamp": end_time,
            "duration": duration_seconds,
            "context": interval.context
        })
        
        # Track event timestamp
        self.event_timestamps[event_type].append(end_time)
        if len(self.event_timestamps[event_type]) > 100:
            self.event_timestamps[event_type] = self.event_timestamps[event_type][-100:]
        
        # For higher development levels, update subjective duration
        if self.development_level >= 0.4:
            # Subjective duration is affected by context and complexity
            complexity = interval.context.get("complexity", 0.5)
            attention = interval.context.get("attention", 0.5)
            
            # Higher complexity and less attention make time seem longer
            subjective_multiplier = 1.0 + (complexity - 0.5) * 0.4 - (attention - 0.5) * 0.6
            interval.subjective_duration = duration_seconds * subjective_multiplier
        
        return {
            "interval_id": interval_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration_seconds,
            "event_type": event_type,
            "subjective_duration": interval.subjective_duration
        }
    
    def _process_estimate_duration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the duration of an event"""
        event_type = input_data.get("event_type")
        context_type = input_data.get("context_type", "general")
        
        if not event_type:
            return {"error": "Event type is required"}
        
        # Base duration on historical data if available
        if event_type in self.duration_estimates[context_type]:
            estimated_duration = self.duration_estimates[context_type][event_type]
            
            # Get confidence based on consistency of past estimates
            confidence = 0.5  # Default confidence
            
            if event_type in self.estimation_errors and len(self.estimation_errors[event_type]) > 0:
                # Calculate coefficient of variation (normalized std dev)
                errors = self.estimation_errors[event_type]
                if len(errors) >= 3:
                    std_dev = np.std(errors)
                    mean_error = np.mean(errors)
                    if abs(mean_error) > 0.001:  # Avoid division by very small numbers
                        cv = std_dev / abs(mean_error)
                        # Lower variability = higher confidence
                        confidence = max(0.1, min(0.9, 1.0 - cv))
                    else:
                        confidence = 0.8  # Very low mean error = high confidence
            
            basis = "historical_data"
        else:
            # No historical data, estimate based on development level
            if self.development_level < 0.3:
                # Very basic estimation - assume medium duration
                estimated_duration = 60.0  # Default to 1 minute
                confidence = 0.2
                basis = "default_assumption"
            elif self.development_level < 0.6:
                # Try to infer from semantically similar events
                similar_types = self._find_similar_event_types(event_type)
                if similar_types:
                    # Average durations of similar events
                    similar_durations = []
                    for similar_type in similar_types:
                        if similar_type in self.duration_estimates[context_type]:
                            similar_durations.append(self.duration_estimates[context_type][similar_type])
                    
                    if similar_durations:
                        estimated_duration = np.mean(similar_durations)
                        confidence = 0.4 * (len(similar_durations) / len(similar_types))
                        basis = "semantic_similarity"
                    else:
                        estimated_duration = 60.0
                        confidence = 0.2
                        basis = "default_assumption"
                else:
                    estimated_duration = 60.0
                    confidence = 0.2
                    basis = "default_assumption"
            else:
                # Use neural network for estimation
                try:
                    # This is a simplified approach - a real implementation would use
                    # proper embeddings and trained models
                    # Get embedding for event type
                    estimate, _ = self._neural_duration_estimate(event_type, context_type)
                    estimated_duration = estimate
                    confidence = 0.6
                    basis = "neural_model"
                except Exception as e:
                    logger.warning(f"Neural duration estimation failed: {str(e)}")
                    estimated_duration = 60.0
                    confidence = 0.2
                    basis = "default_assumption"
        
        return {
            "event_type": event_type,
            "context_type": context_type,
            "estimated_duration": estimated_duration,
            "confidence": confidence,
            "basis": basis
        }
    
    def _process_detect_rhythm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect rhythms and cycles in event occurrences"""
        event_type = input_data.get("event_type")
        min_occurrences = input_data.get("min_occurrences", 3)
        
        # Check development level required for rhythm detection
        if self.development_level < 0.4:
            return {
                "error": "Rhythm detection not available at current development level",
                "development_needed": "This capability requires development level of at least 0.4"
            }
        
        if not event_type:
            return {"error": "Event type is required"}
        
        # Check if we have enough data points
        if event_type not in self.event_timestamps or len(self.event_timestamps[event_type]) < min_occurrences:
            return {
                "event_type": event_type,
                "detected": False,
                "reason": "Insufficient occurrences",
                "available_occurrences": len(self.event_timestamps[event_type]) if event_type in self.event_timestamps else 0,
                "required_occurrences": min_occurrences
            }
        
        # Get timestamps for this event type
        timestamps = sorted(self.event_timestamps[event_type])
        
        # Calculate intervals between occurrences
        intervals = []
        for i in range(len(timestamps) - 1):
            interval_seconds = (timestamps[i+1] - timestamps[i]).total_seconds()
            intervals.append(interval_seconds)
        
        # Check for rhythmicity
        rhythm_detected = False
        period = 0.0
        stability = 0.0
        
        if len(intervals) >= min_occurrences - 1:
            # Calculate mean and standard deviation of intervals
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Calculate coefficient of variation as a measure of stability
            # Lower CV = more stable rhythm
            if mean_interval > 0:
                cv = std_interval / mean_interval
                stability = max(0.0, min(1.0, 1.0 - cv))
                
                # Consider a rhythm detected if stability is above threshold
                # Higher development levels can detect more subtle rhythms
                stability_threshold = max(0.1, 0.7 - self.development_level * 0.4)
                if stability > stability_threshold:
                    rhythm_detected = True
                    period = mean_interval
        
        # If rhythm detected, create or update rhythm object
        if rhythm_detected:
            # Check if we already have a rhythm for this event type
            existing_rhythm = None
            for rhythm_id, rhythm in self.temporal_rhythms.items():
                if rhythm.domain == event_type:
                    existing_rhythm = rhythm
                    break
            
            if existing_rhythm:
                # Update existing rhythm
                existing_rhythm.period = period
                existing_rhythm.stability = stability
                existing_rhythm.last_updated = datetime.now()
                rhythm_id = existing_rhythm.id
            else:
                # Create new rhythm
                new_rhythm = TemporalRhythm(
                    period=period,
                    stability=stability,
                    domain=event_type
                )
                rhythm_id = new_rhythm.id
                self.temporal_rhythms[rhythm_id] = new_rhythm
                
                # Add to active rhythms in temporal context
                self.temporal_context.active_rhythms.append(rhythm_id)
            
            return {
                "event_type": event_type,
                "detected": True,
                "rhythm_id": rhythm_id,
                "period": period,
                "stability": stability,
                "occurrences": len(timestamps),
                "intervals": intervals
            }
        else:
            return {
                "event_type": event_type,
                "detected": False,
                "reason": "No stable rhythm detected",
                "occurrences": len(timestamps),
                "mean_interval": np.mean(intervals) if intervals else 0,
                "interval_variability": np.std(intervals) if intervals else 0
            }
    
    def _process_update_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update the temporal context"""
        if "temporal_focus" in input_data:
            self.temporal_context.temporal_focus = input_data["temporal_focus"]
            
        if "subjective_time_rate" in input_data:
            self.temporal_context.subjective_time_rate = input_data["subjective_time_rate"]
            
        if "time_horizon" in input_data and isinstance(input_data["time_horizon"], dict):
            self.temporal_context.time_horizon.update(input_data["time_horizon"])
            
        # Update current time
        self.temporal_context.current_time = datetime.now()
        
        return {
            "context_updated": True,
            "temporal_context": self.temporal_context.model_dump()
        }
    
    def _update_system_time(self) -> None:
        """Update internal time tracking"""
        current_time = datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time
        
        # Update temporal context
        self.temporal_context.current_time = current_time
        
        # Update rhythms
        for rhythm_id, rhythm in self.temporal_rhythms.items():
            if rhythm.period > 0:
                # Update phase
                elapsed_seconds = (current_time - rhythm.detected_at).total_seconds()
                rhythm.phase = (elapsed_seconds % rhythm.period) / rhythm.period
    
    def _record_event(self, event: Dict[str, Any]) -> None:
        """Record an event in the time perception history"""
        if "timestamp" not in event:
            event["timestamp"] = datetime.now()
            
        self.event_history.append(event)
        
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    def _find_similar_event_types(self, event_type: str) -> List[str]:
        """Find semantically similar event types"""
        # Simple string matching approach
        # In a real implementation, would use embeddings for semantic similarity
        similar_types = []
        
        for existing_type in self.duration_estimates["general"].keys():
            # Check if strings share words
            if self._string_similarity(event_type, existing_type) > 0.3:
                similar_types.append(existing_type)
                
        return similar_types
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        # Split strings into words
        words1 = set(str1.lower().split('_'))
        words2 = set(str2.lower().split('_'))
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _neural_duration_estimate(self, event_type: str, context_type: str) -> Tuple[float, float]:
        """Use neural network to estimate duration"""
        # This is a simplified implementation
        # In a real system, would convert event type to embedding and use proper neural processing
        
        # Create a simple feature vector
        # In a real implementation, would use embeddings
        feature_vec = np.zeros(self.time_perception_network.input_dim)
        
        # Fill the first few elements with hash of strings
        hash_val = hash(event_type) % 1000
        feature_vec[0] = hash_val / 1000.0
        
        hash_val = hash(context_type) % 1000
        feature_vec[1] = hash_val / 1000.0
        
        # Create proper input format
        input_tensor = torch.tensor(feature_vec).float().reshape(1, 1, -1)
        
        # Get estimates from network
        with torch.no_grad():
            duration, rhythm = self.time_perception_network(input_tensor)
            
        return duration.item(), rhythm[0].norm().item()
    
    def _handle_event_start(self, message: Message) -> None:
        """Handle event start messages from the event bus"""
        content = message.content
        
        if "event_type" in content:
            self._process_start_interval({
                "event_type": content["event_type"],
                "context": content.get("context", {}),
                "start_time": content.get("timestamp", datetime.now())
            })
    
    def _handle_event_end(self, message: Message) -> None:
        """Handle event end messages from the event bus"""
        content = message.content
        
        if "interval_id" in content:
            self._process_end_interval({
                "interval_id": content["interval_id"],
                "end_time": content.get("timestamp", datetime.now())
            })
    
    def _handle_heartbeat(self, message: Message) -> None:
        """Handle system heartbeat messages"""
        self._update_system_time()
    
    def get_interval_by_id(self, interval_id: str) -> Optional[TimeInterval]:
        """Get a time interval by ID"""
        return self.time_intervals.get(interval_id)
    
    def get_rhythm_by_id(self, rhythm_id: str) -> Optional[TemporalRhythm]:
        """Get a temporal rhythm by ID"""
        return self.temporal_rhythms.get(rhythm_id)
    
    def get_current_context(self) -> TemporalContext:
        """Get the current temporal context"""
        return self.temporal_context
    
    def get_active_intervals(self) -> List[TimeInterval]:
        """Get all currently active time intervals"""
        return list(self.active_intervals.values())
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history of temporal events, optionally filtered by type"""
        if event_type:
            filtered_history = [event for event in self.event_history 
                                if event.get("event_type") == event_type]
            return filtered_history[-limit:]
        else:
            return self.event_history[-limit:]
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        old_level = self.development_level
        new_level = super().update_development(amount)
        
        # If crossing thresholds, update capabilities
        if old_level < 0.6 and new_level >= 0.6:
            # Expand time horizons at higher development levels
            self.temporal_context.time_horizon = {
                "past": 86400.0,  # 1 day
                "future": 604800.0  # 1 week
            }
        
        if old_level < 0.8 and new_level >= 0.8:
            # Further expand time horizons at very high development
            self.temporal_context.time_horizon = {
                "past": 2592000.0,  # 30 days
                "future": 7776000.0  # 90 days
            }
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add time perception-specific state information
        state.update({
            "interval_count": len(self.time_intervals),
            "active_interval_count": len(self.active_intervals),
            "rhythm_count": len(self.temporal_rhythms),
            "event_count": len(self.event_history),
            "temporal_focus": self.temporal_context.temporal_focus
        })
        
        return state
