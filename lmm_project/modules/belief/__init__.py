"""
Belief Module

This module is responsible for the formation, evaluation, updating, and resolution
of contradictions in the mind's belief system. It includes mechanisms for forming
new beliefs, evaluating evidence, updating existing beliefs, and resolving conflicts
between contradictory beliefs.

The belief system evolves developmentally from simple, rigid beliefs in early stages
to more nuanced, flexible, and evidence-based beliefs in later stages.
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.base_module import BaseModule
from lmm_project.modules.belief.models import Belief, Evidence, BeliefSystem
from lmm_project.modules.belief.belief_formation import BeliefFormation
from lmm_project.modules.belief.evidence_evaluation import EvidenceEvaluation
from lmm_project.modules.belief.belief_updating import BeliefUpdating
from lmm_project.modules.belief.contradiction_resolution import ContradictionResolution
from lmm_project.modules.belief.neural_net import BeliefNetwork

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "belief",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "BeliefSystemModule":
    """
    Factory function to create a belief module.
    
    The belief system is responsible for:
    - Forming beliefs based on experiences and evidence
    - Evaluating the reliability of evidence
    - Updating beliefs in response to new information
    - Resolving contradictions between beliefs
    
    Args:
        module_id: Identifier for this module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level (0.0-1.0)
    
    Returns:
        An integrated BeliefSystemModule instance
    """
    return BeliefSystemModule(module_id, event_bus, development_level)

class BeliefSystemModule(BaseModule):
    """
    Integrated belief system that coordinates belief formation, evidence evaluation,
    belief updating, and contradiction resolution.
    
    This module evolves developmentally from simple, rigid beliefs with minimal
    evidence requirements to sophisticated, nuanced beliefs with complex
    evidential standards and uncertainty handling.
    """
    # Development milestones for the belief system
    development_milestones = {
        0.0: "Basic association-based beliefs",
        0.2: "Simple causal beliefs emerging",
        0.4: "Basic evidence evaluation capabilities",
        0.6: "Consideration of multiple evidence sources",
        0.8: "Handling of uncertainty and probability",
        1.0: "Complex belief networks with nuanced confidence"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """Initialize the belief system with its components"""
        super().__init__(module_id, "belief", event_bus)
        self.development_level = development_level
        
        # Initialize belief system state
        self.belief_system = BeliefSystem(developmental_level=development_level)
        
        # Initialize sub-components
        self.belief_formation = BeliefFormation(f"{module_id}_formation", event_bus)
        self.evidence_evaluation = EvidenceEvaluation(f"{module_id}_evidence", event_bus)
        self.belief_updating = BeliefUpdating(f"{module_id}_updating", event_bus)
        self.contradiction_resolution = ContradictionResolution(f"{module_id}_resolution", event_bus)
        
        # Neural network for belief processing
        self.neural_net = BeliefNetwork()
        
        # Activation tracking for beliefs
        self.active_beliefs = {}  # Maps belief_id to activation level
        self.activation_decay_rate = 0.1  # How quickly activations decay
        self.last_activation_update = datetime.now()
        
        # Update component development levels
        self._sync_development_levels()
        
        # Subscribe to relevant message types
        if event_bus:
            event_bus.subscribe("perception_input", self._handle_message)
            event_bus.subscribe("memory_retrieval", self._handle_message)
            event_bus.subscribe("language_comprehension", self._handle_message)
            event_bus.subscribe("consciousness_broadcast", self._handle_message)
            event_bus.subscribe("belief_query", self._handle_message)
            
        logger.info(f"Belief system initialized at development level {development_level:.2f}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to update the belief system
        
        Args:
            input_data: Data to process, may include:
                - new_evidence: Evidence to evaluate
                - belief_query: Query about existing beliefs
                - contradiction_check: Check for conflicting beliefs
                - context: Contextual information for processing
        
        Returns:
            Results of belief processing
        """
        results = {"processed": False, "module_id": self.module_id}
        
        # Extract context if available
        context = input_data.get("context", {})
        
        # Handle different input types
        if "new_evidence" in input_data:
            # Get prior evidence for consistency evaluation
            prior_evidence = []
            for belief in self.belief_system.beliefs.values():
                prior_evidence.extend(belief.evidence_for)
                prior_evidence.extend(belief.evidence_against)
            
            # Evaluate the evidence
            eval_results = self.evidence_evaluation.process_input({
                "evidence": input_data["new_evidence"],
                "context": context,
                "prior_evidence": prior_evidence
            })
            
            # Use evaluated evidence for belief formation/updating
            if "source" in input_data and input_data["source"] == "memory":
                # Check if this relates to existing beliefs
                update_results = self.belief_updating.process_input({
                    "evidence": eval_results["evaluated_evidence"],
                    "belief_system": self.belief_system,
                    "context": context
                })
                results.update(update_results)
            else:
                # Form new beliefs from the evidence
                formation_results = self.belief_formation.process_input({
                    "evidence": eval_results["evaluated_evidence"],
                    "belief_system": self.belief_system,
                    "context": context
                })
                results.update(formation_results)
                
            # Recalculate belief system consistency
            self.belief_system._calculate_consistency()
                
            # Check for contradictions after updating beliefs
            contradiction_results = self.contradiction_resolution.process_input({
                "belief_system": self.belief_system,
                "context": context
            })
            results.update(contradiction_results)
            
            results["processed"] = True
            results["consistency_score"] = self.belief_system.consistency_score
            
        elif "belief_query" in input_data:
            query = input_data["belief_query"]
            matching_beliefs = self._find_matching_beliefs(query)
            results["beliefs"] = matching_beliefs
            results["processed"] = True
            
        elif "contradiction_check" in input_data:
            # Explicit contradiction check
            contradiction_results = self.contradiction_resolution.process_input({
                "belief_system": self.belief_system,
                "context": context
            })
            results.update(contradiction_results)
            results["processed"] = True
            
        return results
    
    def _find_matching_beliefs(self, query: Dict[str, Any]) -> List[Belief]:
        """Find beliefs that match the provided query criteria"""
        matches = []
        
        # Basic matching logic - can be enhanced with semantic matching
        for belief_id, belief in self.belief_system.beliefs.items():
            match = True
            for key, value in query.items():
                if key not in belief.content or belief.content[key] != value:
                    match = False
                    break
            
            if match:
                matches.append(belief)
                
        return matches
    
    def _handle_message(self, message: Message) -> None:
        """Process messages from other modules"""
        # Extract context if available in the message
        context = message.content.get("context", {}) if isinstance(message.content, dict) else {}
        
        if message.msg_type == "perception_input":
            # Extract perception data, preserving context if available
            perception_data = message.content if not isinstance(message.content, dict) else {
                k: v for k, v in message.content.items() if k != "context"
            }
            
            # Convert perception into potential evidence
            evidence = self._create_evidence_from_perception(perception_data)
            self.process_input({
                "new_evidence": evidence, 
                "source": "perception",
                "context": context
            })
            
        elif message.msg_type == "memory_retrieval":
            # Extract memory data, preserving context if available
            memory_data = message.content if not isinstance(message.content, dict) else {
                k: v for k, v in message.content.items() if k != "context"
            }
            
            # Process memories as evidence for beliefs
            evidence = self._create_evidence_from_memory(memory_data)
            self.process_input({
                "new_evidence": evidence, 
                "source": "memory",
                "context": context
            })
            
        elif message.msg_type == "language_comprehension":
            # Extract language data, preserving context if available
            language_data = message.content if not isinstance(message.content, dict) else {
                k: v for k, v in message.content.items() if k != "context"
            }
            
            # Process language as evidence for beliefs
            evidence = self._create_evidence_from_language(language_data)
            self.process_input({
                "new_evidence": evidence, 
                "source": "language",
                "context": context
            })
            
        elif message.msg_type == "belief_query":
            # Handle queries about beliefs
            query_params = {"belief_query": message.content}
            if context:
                query_params["context"] = context
                
            results = self.process_input(query_params)
            
            # Reply with results
            if self.event_bus and message.reply_to:
                reply = Message(
                    msg_type="belief_query_response",
                    sender=self.module_id,
                    recipient=message.sender,
                    content=results,
                    reply_to=message.id
                )
                self.event_bus.publish(reply)
    
    def _create_evidence_from_perception(self, perception_data: Dict[str, Any]) -> Evidence:
        """Convert perception data into evidence format"""
        return Evidence(
            source="perception",
            content=perception_data,
            reliability=self._calculate_perception_reliability(perception_data),
            relevance=1.0,  # Can be refined based on current context
            timestamp=datetime.now()
        )
    
    def _calculate_perception_reliability(self, perception_data: Dict[str, Any]) -> float:
        """Calculate the reliability of perceptual evidence"""
        # Base reliability on perception clarity if available
        if "clarity" in perception_data:
            return max(0.0, min(1.0, perception_data["clarity"]))
        
        # Default medium-high reliability for perception
        return 0.8
    
    def _create_evidence_from_memory(self, memory_data: Dict[str, Any]) -> Evidence:
        """Convert memory data into evidence format"""
        # Memory reliability decreases with age and increases with emotional significance
        age_factor = 1.0
        if "age" in memory_data:
            age_factor = max(0.3, 1.0 - (memory_data.get("age", 0) * 0.1))
            
        emotional_factor = 1.0
        if "emotional_intensity" in memory_data:
            emotional_factor = 1.0 + (memory_data.get("emotional_intensity", 0) * 0.2)
        
        reliability = min(1.0, age_factor * emotional_factor * 0.7)
        
        return Evidence(
            source="memory",
            content=memory_data,
            reliability=reliability,
            relevance=memory_data.get("relevance", 0.5),
            timestamp=datetime.now()
        )
    
    def _create_evidence_from_language(self, language_data: Dict[str, Any]) -> Evidence:
        """Convert language comprehension data into evidence format"""
        # Language reliability based on source trustworthiness if available
        reliability = language_data.get("source_trustworthiness", 0.5)
        
        return Evidence(
            source="language",
            content=language_data,
            reliability=reliability,
            relevance=language_data.get("relevance", 0.5),
            timestamp=datetime.now()
        )
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the belief system
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update belief system's developmental level
        self.belief_system.developmental_level = self.development_level
        
        # Synchronize development levels across components
        self._sync_development_levels()
        
        if int(prev_level * 10) != int(self.development_level * 10):
            milestone = self._get_current_milestone()
            logger.info(f"Belief system reached development level {self.development_level:.2f}: {milestone}")
        
        return self.development_level
    
    def _sync_development_levels(self) -> None:
        """Synchronize development levels across all belief components"""
        self.belief_formation.development_level = self.development_level
        self.evidence_evaluation.development_level = self.development_level
        self.belief_updating.development_level = self.development_level
        self.contradiction_resolution.development_level = self.development_level
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone description"""
        # Find the highest milestone that's less than or equal to current level
        milestone_levels = sorted(self.development_milestones.keys())
        
        for i in range(len(milestone_levels) - 1, -1, -1):
            if milestone_levels[i] <= self.development_level:
                return self.development_milestones[milestone_levels[i]]
        
        return self.development_milestones[0]
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the belief system"""
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "belief_count": len(self.belief_system.beliefs),
            "beliefs": {
                belief_id: {
                    "content": belief.content,
                    "confidence": belief.confidence,
                    "stability": belief.stability,
                    "evidence_count": len(belief.evidence_for) + len(belief.evidence_against)
                }
                for belief_id, belief in self.belief_system.beliefs.items()
            },
            "consistency_score": self.belief_system.consistency_score,
            "current_milestone": self._get_current_milestone()
        }

    def activate_belief(self, belief_id: str, activation_level: float = 1.0) -> None:
        """
        Activate a belief, making it more prominent in processing
        
        Args:
            belief_id: ID of the belief to activate
            activation_level: Level of activation (0.0-1.0)
        """
        # Ensure belief exists
        if belief_id not in self.belief_system.beliefs:
            return
            
        # Update activation
        current_level = self.active_beliefs.get(belief_id, 0.0)
        new_level = min(1.0, current_level + activation_level)
        self.active_beliefs[belief_id] = new_level
        
        # Activate related beliefs with diminishing activation
        if self.development_level > 0.4:  # Only at higher development levels
            related_beliefs = self.belief_system.find_related_beliefs(belief_id)
            for related_id, relatedness in related_beliefs.items():
                if related_id in self.belief_system.beliefs:
                    related_activation = min(0.5, activation_level * relatedness)
                    current = self.active_beliefs.get(related_id, 0.0)
                    self.active_beliefs[related_id] = min(1.0, current + related_activation)
    
    def decay_activations(self) -> None:
        """Decay belief activations over time"""
        now = datetime.now()
        time_diff = (now - self.last_activation_update).total_seconds()
        
        # Only decay if significant time has passed
        if time_diff < 0.1:
            return
            
        # Calculate decay amount
        decay_amount = self.activation_decay_rate * time_diff
        
        # Apply decay to all active beliefs
        for belief_id in list(self.active_beliefs.keys()):
            self.active_beliefs[belief_id] = max(0.0, self.active_beliefs[belief_id] - decay_amount)
            
            # Remove beliefs with negligible activation
            if self.active_beliefs[belief_id] < 0.01:
                del self.active_beliefs[belief_id]
                
        self.last_activation_update = now
    
    def get_active_beliefs(self, threshold: float = 0.1) -> Dict[str, float]:
        """
        Get currently active beliefs
        
        Args:
            threshold: Minimum activation level to include
            
        Returns:
            Dictionary mapping belief IDs to activation levels
        """
        # First decay existing activations
        self.decay_activations()
        
        # Filter and return active beliefs above threshold
        return {
            belief_id: activation 
            for belief_id, activation in self.active_beliefs.items() 
            if activation >= threshold
        }
