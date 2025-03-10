# TODO: Implement the SelfMonitoring class to track internal states and behaviors
# This component should be able to:
# - Monitor internal states (emotions, thoughts, goals)
# - Track behavioral responses and their outcomes
# - Detect discrepancies between goals and current states
# - Provide feedback for regulatory processes

# TODO: Implement developmental progression in self-monitoring:
# - Basic state awareness in early stages
# - Growing behavior tracking in childhood
# - Increased metacognitive monitoring in adolescence
# - Sophisticated self-awareness in adulthood

# TODO: Create mechanisms for:
# - State detection: Identify current internal conditions
# - Discrepancy detection: Notice gaps between goals and reality
# - Progress tracking: Monitor advancement toward goals
# - Error detection: Identify mistakes and suboptimal responses

# TODO: Implement different monitoring types:
# - Emotional monitoring: Track affective states
# - Cognitive monitoring: Observe thoughts and beliefs
# - Behavioral monitoring: Track actions and responses
# - Social monitoring: Observe interpersonal impacts

# TODO: Connect to consciousness and identity modules
# Self-monitoring should utilize conscious awareness
# and contribute to self-concept development

"""
Self Monitoring Module

This module implements self-monitoring capabilities, which develop
from basic state awareness in early stages to sophisticated 
introspection and error-detection in later developmental stages.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import os
import json
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.self_regulation.models import MonitoringEvent
from lmm_project.modules.self_regulation.neural_net import SelfMonitoringNetwork

class SelfMonitoring(BaseModule):
    """
    Handles self-monitoring functionality
    
    This module implements mechanisms for monitoring internal states,
    detecting discrepancies, and triggering corrective processes,
    developing from basic awareness to sophisticated introspection.
    """
    
    # Developmental milestones for self-monitoring
    development_milestones = {
        0.1: "Basic state awareness",
        0.2: "Simple error detection",
        0.3: "Recognition of behavioral outcomes",
        0.4: "Goal-state discrepancy monitoring",
        0.5: "Simple performance monitoring",
        0.6: "Error prediction based on past experience",
        0.7: "Integrated monitoring across domains",
        0.8: "Advanced error anticipation",
        0.9: "Metacognitive monitoring",
        1.0: "Sophisticated introspection and self-correction"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the self-monitoring module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id,
            module_type="self_monitoring",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.self_regulation.self_monitoring.{module_id}")
        
        # Initialize neural network
        self.network = SelfMonitoringNetwork(
            developmental_level=development_level
        )
        
        # Recent monitoring events
        self.recent_events: List[MonitoringEvent] = []
        self.max_event_history = 50
        
        # Monitoring statistics
        self.monitoring_attempts = 0
        self.detected_discrepancies = 0
        self.corrections_triggered = 0
        
        # Current monitoring focus
        self.current_focus: Optional[str] = None
        
        # Detection thresholds (decrease with development)
        self.discrepancy_threshold = self._calculate_discrepancy_threshold()
        self.error_threshold = self._calculate_error_threshold()
        
        # Monitoring frequencies (increase with development)
        self.monitoring_frequency = self._calculate_monitoring_frequency()
        
        # Subscribe to relevant messages
        if event_bus:
            self.subscribe_to_message("state_update")
            self.subscribe_to_message("monitoring_request")
            self.subscribe_to_message("monitoring_query")
        
        self.logger.info(f"Self-monitoring module initialized at development level {development_level:.2f}")
    
    def _calculate_discrepancy_threshold(self) -> float:
        """Calculate discrepancy detection threshold based on development level"""
        # Starts high (0.7), decreases to low (0.15) with development
        max_threshold = 0.7
        min_threshold = 0.15
        return max_threshold - (max_threshold - min_threshold) * self.development_level
        
    def _calculate_error_threshold(self) -> float:
        """Calculate error detection threshold based on development level"""
        # Starts high (0.8), decreases to low (0.1) with development
        max_threshold = 0.8
        min_threshold = 0.1
        return max_threshold - (max_threshold - min_threshold) * self.development_level
    
    def _calculate_monitoring_frequency(self) -> float:
        """Calculate monitoring frequency based on development level"""
        # Starts low (0.2), increases to high (0.9) with development
        min_frequency = 0.2
        max_frequency = 0.9
        return min_frequency + (max_frequency - min_frequency) * self.development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for self-monitoring
        
        Args:
            input_data: Dictionary with input data
                Required keys depend on input type:
                - "type": Type of input ("state", "monitoring_request", "query")
                - For state input: "current_state", "goal_state", "domain"
                - For monitoring request: "focus", "params"
                - For query: "query_type"
                
        Returns:
            Dictionary with process results
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input")
        
        result = {
            "success": False,
            "message": f"Unknown input type: {input_type}"
        }
        
        # Process different input types
        if input_type == "state":
            result = self._process_state(input_data)
            
        elif input_type == "monitoring_request":
            result = self._process_monitoring_request(input_data)
            
        elif input_type == "query":
            result = self._process_query(input_data)
            
        return result
    
    def _process_state(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a state update for monitoring"""
        try:
            # Extract required fields
            current_state = state_data.get("current_state")
            if not current_state:
                return {"success": False, "message": "Missing current_state"}
                
            goal_state = state_data.get("goal_state")
            domain = state_data.get("domain", "general")
            
            # Create the monitoring event
            monitoring_event = MonitoringEvent(
                monitoring_type="state_comparison",
                domain=domain,
                current_state=current_state,
                goal_state=goal_state,
                context=state_data.get("context", {})
            )
            
            # Add to recent events
            self.recent_events.append(monitoring_event)
            if len(self.recent_events) > self.max_event_history:
                self.recent_events = self.recent_events[-self.max_event_history:]
                
            # Update monitoring focus
            self.current_focus = domain
            
            # Increment monitoring counter
            self.monitoring_attempts += 1
            
            # Should we actually monitor?
            # Early developmental stages don't monitor every event
            should_monitor = np.random.random() < self.monitoring_frequency
            
            if should_monitor and self.development_level >= 0.1:
                monitoring_result = self._monitor_state(current_state, goal_state, domain)
                return {
                    "success": True,
                    "event_id": monitoring_event.id,
                    "event": monitoring_event.dict(),
                    "was_monitored": True,
                    "monitoring_result": monitoring_result
                }
            else:
                return {
                    "success": True,
                    "event_id": monitoring_event.id,
                    "event": monitoring_event.dict(),
                    "was_monitored": False
                }
                
        except Exception as e:
            self.logger.error(f"Error processing state: {e}")
            return {"success": False, "message": f"Error processing state: {str(e)}"}
    
    def _process_monitoring_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific monitoring request"""
        focus = request_data.get("focus")
        if not focus:
            return {"success": False, "message": "Missing monitoring focus"}
            
        params = request_data.get("params", {})
        
        # Update monitoring focus
        self.current_focus = focus
        
        # Get current and goal states from params or request other modules
        current_state = params.get("current_state")
        goal_state = params.get("goal_state")
        
        # If states not provided, try to get them from other modules
        if (not current_state or not goal_state) and self.event_bus:
            states = self._request_states(focus)
            current_state = states.get("current_state", current_state)
            goal_state = states.get("goal_state", goal_state)
            
        if not current_state:
            return {"success": False, "message": "Could not obtain current state"}
            
        # Now monitor the state
        monitoring_result = self._monitor_state(current_state, goal_state, focus)
        
        # Create and save the monitoring event
        monitoring_event = MonitoringEvent(
            monitoring_type="requested",
            domain=focus,
            current_state=current_state,
            goal_state=goal_state,
            context=params.get("context", {})
        )
        
        self.recent_events.append(monitoring_event)
        if len(self.recent_events) > self.max_event_history:
            self.recent_events = self.recent_events[-self.max_event_history:]
            
        return {
            "success": True,
            "event_id": monitoring_event.id,
            "event": monitoring_event.dict(),
            "monitoring_result": monitoring_result
        }
    
    def _process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query about self-monitoring"""
        query_type = query_data.get("query_type", "status")
        
        if query_type == "status":
            return {
                "success": True,
                "current_focus": self.current_focus,
                "monitoring_frequency": self.monitoring_frequency,
                "discrepancy_threshold": self.discrepancy_threshold,
                "error_threshold": self.error_threshold,
                "development_level": self.development_level
            }
                
        elif query_type == "recent":
            limit = query_data.get("limit", 10)
            recent = self.recent_events[-limit:] if self.recent_events else []
            return {
                "success": True,
                "recent_events": [e.dict() for e in recent]
            }
            
        elif query_type == "statistics":
            discrepancy_rate = 0.0
            if self.monitoring_attempts > 0:
                discrepancy_rate = self.detected_discrepancies / self.monitoring_attempts
                
            correction_rate = 0.0
            if self.detected_discrepancies > 0:
                correction_rate = self.corrections_triggered / self.detected_discrepancies
                
            return {
                "success": True,
                "monitoring_attempts": self.monitoring_attempts,
                "detected_discrepancies": self.detected_discrepancies,
                "corrections_triggered": self.corrections_triggered,
                "discrepancy_rate": discrepancy_rate,
                "correction_rate": correction_rate
            }
            
        elif query_type == "domain":
            domain = query_data.get("domain")
            if not domain:
                return {"success": False, "message": "Missing domain for query"}
                
            # Filter events by domain
            domain_events = [e for e in self.recent_events if e.domain == domain]
            return {
                "success": True,
                "domain": domain,
                "event_count": len(domain_events),
                "recent_events": [e.dict() for e in domain_events[-5:]]
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown query type: {query_type}"
            }
    
    def _monitor_state(self, current_state: Any, goal_state: Any, domain: str) -> Dict[str, Any]:
        """
        Monitor the discrepancy between current and goal states
        
        Args:
            current_state: The current state to monitor
            goal_state: The goal state to compare against
            domain: The domain being monitored
            
        Returns:
            Dictionary with monitoring results
        """
        self.logger.debug(f"Monitoring {domain} state")
        
        # Prepare states for neural network
        try:
            # Convert states to vectors if they aren't already
            current_vector = self._state_to_vector(current_state, domain)
            goal_vector = self._state_to_vector(goal_state, domain) if goal_state else None
            
            # Process with neural network
            detection_results = self.network.forward(
                current_state=current_vector,
                goal_state=goal_vector
            )
            
            # Extract results
            discrepancy_detected = detection_results["discrepancy_detected"]
            discrepancy_magnitude = float(detection_results["discrepancy_magnitude"])
            error_probability = float(detection_results["error_probability"])
            
            # Apply thresholds based on development level
            significant_discrepancy = discrepancy_magnitude > self.discrepancy_threshold
            significant_error = error_probability > self.error_threshold
            
            # Update counters
            if significant_discrepancy:
                self.detected_discrepancies += 1
                
            # Determine if correction is needed
            needs_correction = significant_discrepancy or significant_error
            
            if needs_correction:
                self.corrections_triggered += 1
                
                # Publish event to trigger correction if needed
                if self.event_bus:
                    self.publish_message("discrepancy_detected", {
                        "domain": domain,
                        "discrepancy_magnitude": discrepancy_magnitude,
                        "error_probability": error_probability,
                        "current_state": current_state,
                        "goal_state": goal_state
                    })
            
            # Return monitoring results
            return {
                "success": True,
                "domain": domain,
                "discrepancy_detected": bool(discrepancy_detected),
                "discrepancy_magnitude": discrepancy_magnitude,
                "error_probability": error_probability,
                "significant_discrepancy": significant_discrepancy,
                "significant_error": significant_error,
                "needs_correction": needs_correction
            }
            
        except Exception as e:
            self.logger.error(f"Error in state monitoring: {e}")
            return {
                "success": False,
                "message": f"Monitoring error: {str(e)}"
            }
    
    def _state_to_vector(self, state: Any, domain: str) -> np.ndarray:
        """
        Convert a state to a vector representation
        
        Args:
            state: The state to convert
            domain: The domain of the state
            
        Returns:
            Numpy array representation of the state
        """
        # If already a numpy array, return as is
        if isinstance(state, np.ndarray):
            return state
            
        # If a list or tuple of numbers, convert to numpy array
        if isinstance(state, (list, tuple)) and all(isinstance(x, (int, float)) for x in state):
            return np.array(state, dtype=np.float32)
            
        # If a dictionary, handle based on domain
        if isinstance(state, dict):
            if domain == "emotion":
                # Handle emotional states
                intensity = state.get("intensity", 0.5)
                valence = state.get("valence", 0.0)
                arousal = state.get("arousal", 0.5)
                return np.array([intensity, valence, arousal], dtype=np.float32)
                
            elif domain == "impulse":
                # Handle impulse states
                strength = state.get("strength", 0.5)
                urgency = state.get("urgency", 0.5)
                return np.array([strength, urgency], dtype=np.float32)
                
            elif domain == "goal":
                # Handle goal states
                importance = state.get("importance", 0.5)
                progress = state.get("progress", 0.0)
                deadline_proximity = state.get("deadline_proximity", 0.0)
                return np.array([importance, progress, deadline_proximity], dtype=np.float32)
                
            elif domain == "attention":
                # Handle attention states
                focus = state.get("focus", 0.5)
                stability = state.get("stability", 0.5)
                distraction = state.get("distraction", 0.5)
                return np.array([focus, stability, distraction], dtype=np.float32)
                
            else:
                # Generic handling for other domains
                # Extract numeric values and use them
                numeric_values = [v for k, v in state.items() if isinstance(v, (int, float))]
                if numeric_values:
                    return np.array(numeric_values, dtype=np.float32)
        
        # For scalar values, convert to single-element array
        if isinstance(state, (int, float)):
            return np.array([state], dtype=np.float32)
            
        # If we can't convert, use a simple fallback
        self.logger.warning(f"Could not convert state to vector: {state}")
        return np.array([0.5], dtype=np.float32)
    
    def _request_states(self, domain: str) -> Dict[str, Any]:
        """
        Request current and goal states from other modules
        
        Args:
            domain: The domain to request states for
            
        Returns:
            Dictionary with current and goal states
        """
        result = {
            "current_state": None,
            "goal_state": None
        }
        
        if not self.event_bus:
            return result
            
        # Determine which module to query based on domain
        target_module = None
        if domain == "emotion":
            target_module = "emotional_regulation"
        elif domain == "impulse":
            target_module = "impulse_control"
        elif domain == "goal":
            target_module = "goal_setting"
        elif domain == "attention":
            target_module = "attention"
        else:
            # No specific module for this domain
            return result
            
        # Request current state
        reply = self.send_message_and_wait_for_reply(
            recipient=target_module,
            message_type=f"{domain}_query",
            content={"query_type": "current"},
            timeout=1.0
        )
        
        if reply and reply.content and reply.content.get("success"):
            # Extract current state from reply
            if domain == "emotion":
                result["current_state"] = reply.content.get("current_emotion")
            elif domain == "impulse":
                result["current_state"] = reply.content.get("current_impulse")
            elif domain == "goal":
                result["current_state"] = reply.content.get("current_goal")
            elif domain == "attention":
                result["current_state"] = reply.content.get("current_focus")
                
        # Request goal state
        reply = self.send_message_and_wait_for_reply(
            recipient=target_module,
            message_type=f"{domain}_query",
            content={"query_type": "goal"},
            timeout=1.0
        )
        
        if reply and reply.content and reply.content.get("success"):
            # Extract goal state from reply
            result["goal_state"] = reply.content.get("goal_state")
            
        return result
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "state_update":
            # Process state update
            result = self.process_input({
                "type": "state",
                **content
            })
            
            # Send response if requested
            if message.reply_to and self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="state_update_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "monitoring_request":
            # Process monitoring request
            result = self.process_input({
                "type": "monitoring_request",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="monitoring_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "monitoring_query":
            # Process query
            result = self.process_input({
                "type": "query",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="monitoring_query_response",
                    content=result,
                    reply_to=message.id
                ))
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        super().update_development(amount)
        
        # Update neural network development
        self.network.update_developmental_level(self.development_level)
        
        # Update development-dependent parameters
        self.discrepancy_threshold = self._calculate_discrepancy_threshold()
        self.error_threshold = self._calculate_error_threshold()
        self.monitoring_frequency = self._calculate_monitoring_frequency()
        
        # Check for developmental milestones
        self._check_development_milestones(old_level)
        
        self.logger.info(f"Updated self-monitoring development to {self.development_level:.2f}")
        return self.development_level
    
    def _check_development_milestones(self, previous_level: float) -> None:
        """
        Check if any developmental milestones have been reached
        
        Args:
            previous_level: The previous development level
        """
        # Check each milestone to see if we've crossed the threshold
        for level, description in self.development_milestones.items():
            # If we've crossed a milestone threshold
            if previous_level < level <= self.development_level:
                self.logger.info(f"Self-monitoring milestone reached at {level:.1f}: {description}")
                
                # Adjust monitoring capabilities based on the new milestone
                if level == 0.1:
                    self.logger.info("Now capable of basic state awareness")
                elif level == 0.2:
                    self.logger.info("Now capable of simple error detection")
                elif level == 0.3:
                    self.logger.info("Now capable of recognizing behavioral outcomes")
                elif level == 0.4:
                    self.logger.info("Now capable of goal-state discrepancy monitoring")
                elif level == 0.5:
                    self.logger.info("Now capable of simple performance monitoring")
                elif level == 0.6:
                    self.logger.info("Now capable of error prediction based on past experience")
                elif level == 0.7:
                    self.logger.info("Now capable of integrated monitoring across domains")
                elif level == 0.8:
                    self.logger.info("Now capable of advanced error anticipation")
                elif level == 0.9:
                    self.logger.info("Now capable of metacognitive monitoring")
                elif level == 1.0:
                    self.logger.info("Now capable of sophisticated introspection and self-correction")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with the module state
        """
        base_state = super().get_state()
        
        # Add self-monitoring specific state
        monitoring_state = {
            "current_focus": self.current_focus,
            "recent_events_count": len(self.recent_events),
            "monitoring_attempts": self.monitoring_attempts,
            "detected_discrepancies": self.detected_discrepancies,
            "corrections_triggered": self.corrections_triggered,
            "discrepancy_threshold": self.discrepancy_threshold,
            "error_threshold": self.error_threshold,
            "monitoring_frequency": self.monitoring_frequency
        }
        
        return {**base_state, **monitoring_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save basic module state
        state_path = os.path.join(module_dir, "module_state.json")
        with open(state_path, 'w') as f:
            # Get state with serializable event data
            state = self.get_state()
            state["recent_events"] = [e.dict() for e in self.recent_events[-20:]] # Save last 20
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Saved self-monitoring state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update base state
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Update self-monitoring specific state
            self.current_focus = state.get("current_focus")
            self.monitoring_attempts = state.get("monitoring_attempts", 0)
            self.detected_discrepancies = state.get("detected_discrepancies", 0)
            self.corrections_triggered = state.get("corrections_triggered", 0)
            self.discrepancy_threshold = state.get("discrepancy_threshold", self._calculate_discrepancy_threshold())
            self.error_threshold = state.get("error_threshold", self._calculate_error_threshold())
            self.monitoring_frequency = state.get("monitoring_frequency", self._calculate_monitoring_frequency())
            
            # Recreate recent events
            from lmm_project.modules.self_regulation.models import MonitoringEvent
            self.recent_events = []
            for event_data in state.get("recent_events", []):
                try:
                    self.recent_events.append(MonitoringEvent(**event_data))
                except Exception as e:
                    self.logger.warning(f"Could not recreate monitoring event: {e}")
            
            # Update neural network
            self.network.update_developmental_level(self.development_level)
            
            self.logger.info(f"Loaded self-monitoring state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False