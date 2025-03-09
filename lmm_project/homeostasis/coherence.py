from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import logging
import math
import random

from lmm_project.core.message import Message
from lmm_project.core.event_bus import EventBus
from lmm_project.core.types import DevelopmentalStage, StateDict
from .models import HomeostaticSystem, HomeostaticNeedType, HomeostaticResponse, NeedState

logger = logging.getLogger(__name__)

class CoherenceManager:
    """
    Manages the coherence and internal consistency of the cognitive system.
    
    The Coherence Manager:
    - Tracks contradictions between beliefs and experiences
    - Detects cognitive dissonance situations
    - Manages consistency between memory and current beliefs
    - Signals when coherence is compromised
    - Facilitates resolution of contradictions
    
    Coherence is analogous to a mind's need for internal consistency and making sense of the world.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        initial_coherence: float = 0.8,
        tolerance_threshold: float = 0.3,
        resolution_rate: float = 0.05
    ):
        self.event_bus = event_bus
        self.homeostatic_system = HomeostaticSystem()
        self.homeostatic_system.initialize_needs()
        
        # Initialize coherence need state
        coherence_need = self.homeostatic_system.needs.get(HomeostaticNeedType.COHERENCE)
        if coherence_need:
            coherence_need.current_value = initial_coherence
            coherence_need.last_updated = datetime.now()
        
        # Coherence parameters
        self.tolerance_threshold = tolerance_threshold  # How much contradiction is tolerable
        self.resolution_rate = resolution_rate  # How quickly contradictions are resolved
        self.last_update_time = datetime.now()
        
        # Contradiction tracking
        self.contradictions: List[Dict[str, Any]] = []
        self.belief_confidence: Dict[str, float] = {}  # Confidence in different beliefs
        self.recent_dissonance_events: List[Dict[str, Any]] = []
        
        # Coherence metrics
        self.knowledge_consistency_score = 0.9  # Overall consistency of knowledge
        self.belief_stability_score = 0.8  # Stability of beliefs over time
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register handlers for coherence-related events"""
        self.event_bus.subscribe("belief_update", self._handle_belief_update)
        self.event_bus.subscribe("perception_input", self._handle_perception)
        self.event_bus.subscribe("memory_retrieval", self._handle_memory_retrieval)
        self.event_bus.subscribe("system_cycle", self._handle_system_cycle)
        self.event_bus.subscribe("development_update", self._handle_development_update)
    
    def _handle_belief_update(self, message: Message):
        """Handle updates to beliefs that might affect coherence"""
        belief_type = message.content.get("belief_type", "unknown")
        belief_content = message.content.get("content", {})
        confidence = message.content.get("confidence", 0.5)
        prior_belief = message.content.get("prior_belief", None)
        
        # Generate belief ID for tracking
        belief_id = f"{belief_type}:{str(hash(str(belief_content)))[-8:]}"
        
        # Track confidence in this belief
        self.belief_confidence[belief_id] = confidence
        
        # Check for contradiction with prior beliefs
        if prior_belief and prior_belief != belief_content:
            # This is a belief change that might indicate contradiction
            contradiction_severity = self._calculate_contradiction_severity(
                prior_belief, belief_content, confidence
            )
            
            if contradiction_severity > self.tolerance_threshold:
                # Record contradiction
                self._record_contradiction(
                    belief_id=belief_id,
                    belief_type=belief_type, 
                    old_belief=prior_belief,
                    new_belief=belief_content,
                    severity=contradiction_severity,
                    source="belief_update"
                )
                
                # Update coherence based on contradiction severity
                coherence_impact = -contradiction_severity * 0.2
                self.homeostatic_system.update_need(
                    HomeostaticNeedType.COHERENCE,
                    coherence_impact,
                    f"Belief contradiction detected: {belief_type}"
                )
        
        # Check for changes in belief stability
        self._update_belief_stability()
        
        # Publish current coherence state
        self._publish_coherence_state()
    
    def _handle_perception(self, message: Message):
        """Handle perception input that might contradict existing beliefs"""
        perceptions = message.content.get("perceptions", {})
        
        # Check each perception against relevant beliefs
        for perception_type, perception_data in perceptions.items():
            belief_matches = self._find_relevant_beliefs(perception_type, perception_data)
            
            for belief_id, relevance in belief_matches:
                # Get belief details
                if belief_id in self.belief_confidence:
                    # Check for contradiction
                    contradiction_severity = self._check_perception_contradiction(
                        perception_type, perception_data, belief_id, relevance
                    )
                    
                    if contradiction_severity > self.tolerance_threshold:
                        # Record contradiction
                        self._record_contradiction(
                            belief_id=belief_id,
                            belief_type=perception_type, 
                            old_belief=f"Existing belief {belief_id}",
                            new_belief=f"Perception {perception_type}",
                            severity=contradiction_severity,
                            source="perception_contradiction"
                        )
                        
                        # Update coherence based on contradiction severity
                        coherence_impact = -contradiction_severity * 0.15
                        self.homeostatic_system.update_need(
                            HomeostaticNeedType.COHERENCE,
                            coherence_impact,
                            f"Perception contradiction: {perception_type}"
                        )
    
    def _handle_memory_retrieval(self, message: Message):
        """Handle memory retrievals that might contradict current beliefs"""
        memory_type = message.content.get("memory_type", "unknown")
        memory_content = message.content.get("content", {})
        
        # Only check episodic or semantic memories for contradictions
        if memory_type not in ["episodic", "semantic"]:
            return
            
        # Check memory against current beliefs
        belief_matches = self._find_relevant_beliefs(memory_type, memory_content)
        
        for belief_id, relevance in belief_matches:
            # Check for contradiction
            contradiction_severity = self._check_memory_contradiction(
                memory_type, memory_content, belief_id, relevance
            )
            
            if contradiction_severity > self.tolerance_threshold:
                # Record contradiction
                self._record_contradiction(
                    belief_id=belief_id,
                    belief_type=memory_type, 
                    old_belief=f"Existing belief {belief_id}",
                    new_belief=f"Memory {memory_type}",
                    severity=contradiction_severity,
                    source="memory_contradiction"
                )
                
                # Update coherence based on contradiction severity
                coherence_impact = -contradiction_severity * 0.1
                self.homeostatic_system.update_need(
                    HomeostaticNeedType.COHERENCE,
                    coherence_impact,
                    f"Memory contradiction: {memory_type}"
                )
    
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle events to update coherence naturally"""
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update_time).total_seconds()
        
        # Natural resolution of contradictions over time
        if self.contradictions and time_delta > 60:  # Only process periodically
            self._process_contradictions()
            self.last_update_time = now
            
            # Publish current coherence state
            self._publish_coherence_state()
    
    def _handle_development_update(self, message: Message):
        """Adapt coherence parameters based on developmental stage"""
        development_level = message.content.get("development_level", 0.0)
        
        # Update homeostatic setpoints based on development
        self.homeostatic_system.adapt_to_development(development_level)
        
        # Adjust coherence parameters based on development
        # Young minds have lower coherence requirements and higher tolerance for contradictions
        if development_level < 0.3:  # Infant/early child
            self.tolerance_threshold = 0.5  # Higher tolerance for contradictions
            self.resolution_rate = 0.03  # Slower resolution
        elif development_level < 0.6:  # Child
            self.tolerance_threshold = 0.4
            self.resolution_rate = 0.04
        elif development_level < 0.8:  # Adolescent
            self.tolerance_threshold = 0.3
            self.resolution_rate = 0.05
        else:  # Adult
            self.tolerance_threshold = 0.2  # Lower tolerance for contradictions
            self.resolution_rate = 0.06  # Faster resolution
            
        logger.info(
            f"Coherence parameters adapted to development level {development_level:.2f}: "
            f"tolerance={self.tolerance_threshold:.2f}, resolution_rate={self.resolution_rate:.2f}"
        )
    
    def _calculate_contradiction_severity(self, old_belief: Any, new_belief: Any, confidence: float) -> float:
        """
        Calculate the severity of a contradiction between beliefs
        
        Args:
            old_belief: The prior belief
            new_belief: The new belief
            confidence: Confidence in the new belief
            
        Returns:
            Severity score (0.0-1.0) where higher means more severe contradiction
        """
        # This is a simplified implementation - a real one would do semantic comparison
        # For now, we'll use a random value weighted by confidence
        base_severity = random.uniform(0.1, 0.9)
        return base_severity * confidence
    
    def _find_relevant_beliefs(self, context_type: str, context_data: Any) -> List[Tuple[str, float]]:
        """
        Find beliefs relevant to the given context
        
        Args:
            context_type: Type of context (perception, memory type, etc.)
            context_data: The context data
            
        Returns:
            List of (belief_id, relevance) pairs
        """
        # This is a simplified implementation - a real one would use semantic matching
        # For now, we'll return a random subset of beliefs with random relevance
        belief_ids = list(self.belief_confidence.keys())
        if not belief_ids:
            return []
            
        num_matches = min(3, len(belief_ids))
        matches = random.sample(belief_ids, num_matches)
        
        return [(belief_id, random.uniform(0.3, 0.9)) for belief_id in matches]
    
    def _check_perception_contradiction(
        self, 
        perception_type: str, 
        perception_data: Any, 
        belief_id: str, 
        relevance: float
    ) -> float:
        """
        Check if a perception contradicts a belief
        
        Args:
            perception_type: Type of perception
            perception_data: The perception data
            belief_id: ID of the belief to check against
            relevance: Relevance of the belief to this perception
            
        Returns:
            Contradiction severity (0.0-1.0)
        """
        # This is a simplified implementation
        # A real implementation would do semantic analysis of the contradiction
        if random.random() < 0.2:  # 20% chance of contradiction
            return random.uniform(0.3, 0.8) * relevance
        return 0.0
    
    def _check_memory_contradiction(
        self, 
        memory_type: str, 
        memory_content: Any, 
        belief_id: str, 
        relevance: float
    ) -> float:
        """
        Check if a memory contradicts a belief
        
        Args:
            memory_type: Type of memory
            memory_content: The memory content
            belief_id: ID of the belief to check against
            relevance: Relevance of the belief to this memory
            
        Returns:
            Contradiction severity (0.0-1.0)
        """
        # This is a simplified implementation
        # A real implementation would do semantic analysis of the contradiction
        if random.random() < 0.15:  # 15% chance of contradiction
            return random.uniform(0.2, 0.7) * relevance
        return 0.0
    
    def _record_contradiction(
        self, 
        belief_id: str,
        belief_type: str,
        old_belief: Any,
        new_belief: Any,
        severity: float,
        source: str
    ):
        """Record a contradiction for processing"""
        contradiction = {
            "id": f"contra_{len(self.contradictions) + 1}",
            "belief_id": belief_id,
            "belief_type": belief_type,
            "old_belief": old_belief,
            "new_belief": new_belief,
            "severity": severity,
            "source": source,
            "timestamp": datetime.now(),
            "status": "active",
            "resolution_attempts": 0
        }
        
        self.contradictions.append(contradiction)
        
        # Log the contradiction
        logger.info(
            f"Recorded contradiction: {belief_type} (severity: {severity:.2f}, source: {source})"
        )
        
        # Create a dissonance event if severe enough
        if severity > self.tolerance_threshold * 1.5:
            self._create_cognitive_dissonance_event(contradiction)
    
    def _create_cognitive_dissonance_event(self, contradiction: Dict[str, Any]):
        """Create a cognitive dissonance event for significant contradictions"""
        dissonance_event = {
            "contradiction_id": contradiction["id"],
            "belief_type": contradiction["belief_type"],
            "severity": contradiction["severity"],
            "timestamp": datetime.now()
        }
        
        self.recent_dissonance_events.append(dissonance_event)
        
        # Keep only recent events
        cutoff_time = datetime.now() - timedelta(minutes=30)
        self.recent_dissonance_events = [
            event for event in self.recent_dissonance_events
            if event["timestamp"] > cutoff_time
        ]
        
        # Publish dissonance event
        dissonance_message = Message(
            sender="coherence_manager",
            message_type="cognitive_dissonance",
            content={
                "contradiction": contradiction,
                "dissonance_level": contradiction["severity"],
                "active_dissonance_count": len(self.recent_dissonance_events)
            }
        )
        self.event_bus.publish(dissonance_message)
        
        logger.warning(
            f"Cognitive dissonance event: {contradiction['belief_type']} "
            f"(severity: {contradiction['severity']:.2f})"
        )
    
    def _process_contradictions(self):
        """Process and attempt to resolve active contradictions"""
        # Track how many were resolved
        resolved_count = 0
        
        for contradiction in self.contradictions:
            if contradiction["status"] != "active":
                continue
                
            # Attempt resolution
            resolution_chance = self.resolution_rate * (1 + contradiction["resolution_attempts"] * 0.2)
            if random.random() < resolution_chance:
                # Successfully resolved
                contradiction["status"] = "resolved"
                contradiction["resolution_time"] = datetime.now()
                
                # Improve coherence
                coherence_improvement = contradiction["severity"] * 0.3
                self.homeostatic_system.update_need(
                    HomeostaticNeedType.COHERENCE,
                    coherence_improvement,
                    f"Contradiction resolved: {contradiction['belief_type']}"
                )
                
                resolved_count += 1
                
                # Publish resolution event
                resolution_message = Message(
                    sender="coherence_manager",
                    message_type="contradiction_resolved",
                    content={
                        "contradiction_id": contradiction["id"],
                        "belief_type": contradiction["belief_type"],
                        "resolution_time": contradiction["resolution_time"].isoformat(),
                        "attempts": contradiction["resolution_attempts"]
                    }
                )
                self.event_bus.publish(resolution_message)
            else:
                # Increment resolution attempts
                contradiction["resolution_attempts"] += 1
                
                # If many failed attempts, consider it unresolvable
                if contradiction["resolution_attempts"] > 5:
                    contradiction["status"] = "unresolvable"
                    
                    # This permanently affects coherence negatively
                    self.homeostatic_system.update_need(
                        HomeostaticNeedType.COHERENCE,
                        -contradiction["severity"] * 0.1,
                        f"Unresolvable contradiction: {contradiction['belief_type']}"
                    )
                    
                    # Update knowledge consistency score
                    self.knowledge_consistency_score = max(
                        0.1, 
                        self.knowledge_consistency_score - (contradiction["severity"] * 0.05)
                    )
        
        # Clean up old resolved contradictions
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.contradictions = [
            c for c in self.contradictions
            if c["status"] != "resolved" or 
               (c["status"] == "resolved" and c.get("resolution_time", datetime.now()) > cutoff_time)
        ]
        
        # If resolved contradictions, log it
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} contradictions")
            
    def _update_belief_stability(self):
        """Update the belief stability score based on recent changes"""
        # Count active contradictions
        active_contradictions = sum(1 for c in self.contradictions if c["status"] == "active")
        
        # Update stability score
        if active_contradictions > 5:
            # Many contradictions reduce stability
            self.belief_stability_score = max(0.1, self.belief_stability_score - 0.1)
        elif active_contradictions < 2:
            # Few contradictions improve stability
            self.belief_stability_score = min(1.0, self.belief_stability_score + 0.05)
    
    def _publish_coherence_state(self):
        """Publish current coherence state to the event bus"""
        coherence_need = self.homeostatic_system.needs[HomeostaticNeedType.COHERENCE]
        
        # Count contradiction states
        active_count = sum(1 for c in self.contradictions if c["status"] == "active")
        resolved_count = sum(1 for c in self.contradictions if c["status"] == "resolved")
        unresolvable_count = sum(1 for c in self.contradictions if c["status"] == "unresolvable")
        
        coherence_message = Message(
            sender="coherence_manager",
            message_type="coherence_state_update",
            content={
                "coherence_level": coherence_need.current_value,
                "setpoint": coherence_need.setpoint,
                "is_deficient": coherence_need.is_deficient,
                "urgency": coherence_need.urgency,
                "active_contradictions": active_count,
                "resolved_contradictions": resolved_count,
                "unresolvable_contradictions": unresolvable_count,
                "knowledge_consistency": self.knowledge_consistency_score,
                "belief_stability": self.belief_stability_score,
                "tolerance_threshold": self.tolerance_threshold
            }
        )
        self.event_bus.publish(coherence_message)
    
    def report_contradiction(
        self, 
        belief_type: str, 
        description: str, 
        severity: float, 
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Report a contradiction from an external source
        
        Args:
            belief_type: Type or category of the contradiction
            description: Description of the contradiction
            severity: Severity of the contradiction (0.0-1.0)
            details: Optional additional details
            
        Returns:
            bool: True if the contradiction was recorded
        """
        if severity < 0.1:
            return False  # Too minor to track
            
        # Record contradiction
        details = details or {}
        self._record_contradiction(
            belief_id=f"external_{int(time.time())}",
            belief_type=belief_type,
            old_belief=details.get("old_belief", "Unknown prior belief"),
            new_belief=details.get("new_belief", description),
            severity=severity,
            source="external_report"
        )
        
        # Update coherence based on contradiction severity
        coherence_impact = -severity * 0.2
        self.homeostatic_system.update_need(
            HomeostaticNeedType.COHERENCE,
            coherence_impact,
            f"Externally reported contradiction: {belief_type}"
        )
        
        # Publish updated state
        self._publish_coherence_state()
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the coherence manager"""
        coherence_need = self.homeostatic_system.needs[HomeostaticNeedType.COHERENCE]
        return {
            "coherence_level": coherence_need.current_value,
            "coherence_setpoint": coherence_need.setpoint,
            "is_deficient": coherence_need.is_deficient,
            "is_excessive": coherence_need.is_excessive,
            "urgency": coherence_need.urgency,
            "active_contradictions": sum(1 for c in self.contradictions if c["status"] == "active"),
            "knowledge_consistency": self.knowledge_consistency_score,
            "belief_stability": self.belief_stability_score,
            "tolerance_threshold": self.tolerance_threshold,
            "resolution_rate": self.resolution_rate,
            "recent_contradictions": [
                {
                    "id": c["id"],
                    "type": c["belief_type"],
                    "severity": c["severity"],
                    "status": c["status"]
                }
                for c in self.contradictions[-5:] if c["status"] == "active"
            ]
        }
    
    def load_state(self, state_dict: StateDict) -> None:
        """Load state from the provided state dictionary"""
        if "coherence_level" in state_dict:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.COHERENCE,
                state_dict["coherence_level"] - 
                self.homeostatic_system.needs[HomeostaticNeedType.COHERENCE].current_value,
                "State loaded"
            )
            
        if "knowledge_consistency" in state_dict:
            self.knowledge_consistency_score = state_dict["knowledge_consistency"]
            
        if "belief_stability" in state_dict:
            self.belief_stability_score = state_dict["belief_stability"]
            
        if "tolerance_threshold" in state_dict:
            self.tolerance_threshold = state_dict["tolerance_threshold"]
            
        if "resolution_rate" in state_dict:
            self.resolution_rate = state_dict["resolution_rate"] 