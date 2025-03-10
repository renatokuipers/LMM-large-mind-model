"""
Contradiction Resolution Module

This module detects and resolves contradictions between beliefs in the belief system.
It implements strategies for handling conflicts that vary based on developmental level,
from simple binary contradiction handling to nuanced contextual resolution.

The module ensures the overall consistency of the belief system while allowing for
appropriate levels of ambiguity and nuance at higher developmental stages.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import random
import numpy as np
from datetime import datetime

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.belief.models import Belief, Evidence, BeliefSystem

logger = logging.getLogger(__name__)

class ContradictionParameters:
    """Parameters controlling contradiction resolution"""
    def __init__(self):
        # Threshold for detecting contradictions
        self.contradiction_threshold = 0.6
        # Influence of context on contradiction evaluation
        self.context_sensitivity = 0.5
        # Tolerance for ambiguity in beliefs
        self.ambiguity_tolerance = 0.3
        # Strategy for resolving contradictions
        self.resolution_strategy = "binary"  # "binary", "weighted", or "contextual"
        # Whether to merge contradictory beliefs
        self.merge_capability = False
        # Number of different models/perspectives that can be maintained
        self.multiple_model_capacity = 1
        # Whether to detect higher-order contradictions
        self.higher_order_detection = False

class ContradictionResolution(BaseModule):
    """
    Responsible for detecting and resolving contradictions between beliefs
    
    This module ensures consistency in the belief system by identifying and
    resolving conflicts between beliefs using developmentally appropriate
    strategies.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the contradiction resolution module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="contradiction_resolution", event_bus=event_bus)
        self.parameters = ContradictionParameters()
        self.resolution_history = []  # Track contradiction resolutions
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to detect and resolve contradictions
        
        Args:
            input_data: Data including belief system to check
        
        Returns:
            Results including any resolved contradictions
        """
        if "belief_system" not in input_data:
            return {
                "processed": False,
                "error": "Missing required input: belief_system",
                "module_id": self.module_id
            }
            
        belief_system = input_data["belief_system"]
        context = input_data.get("context", {})
        
        # Detect contradictions
        contradictions = self._detect_contradictions(belief_system, context)
        
        # If no contradictions found, return early
        if not contradictions:
            return {
                "processed": True,
                "contradictions_found": 0,
                "contradictions_resolved": 0,
                "module_id": self.module_id
            }
            
        # Resolve each contradiction
        resolutions = []
        for contradiction in contradictions:
            belief1_id, belief2_id, strength = contradiction
            
            # Get the beliefs
            belief1 = belief_system.get_belief(belief1_id)
            belief2 = belief_system.get_belief(belief2_id)
            
            if not belief1 or not belief2:
                continue
                
            # Apply resolution strategy
            resolution_result = self._resolve_contradiction(
                belief1, belief2, strength, belief_system, context
            )
            
            if resolution_result["resolved"]:
                resolutions.append({
                    "belief1_id": belief1_id,
                    "belief2_id": belief2_id,
                    "contradiction_strength": strength,
                    "resolution_type": resolution_result["resolution_type"],
                    "resolution_details": resolution_result["details"]
                })
                
                # Track resolution history
                self.resolution_history.append({
                    "timestamp": datetime.now(),
                    "contradiction": (belief1_id, belief2_id, strength),
                    "resolution": resolution_result
                })
                
                # Limit history size
                if len(self.resolution_history) > 50:
                    self.resolution_history = self.resolution_history[-50:]
                    
        # Return results
        return {
            "processed": True,
            "contradictions_found": len(contradictions),
            "contradictions_resolved": len(resolutions),
            "resolutions": resolutions,
            "module_id": self.module_id
        }
        
    def _detect_contradictions(
        self, 
        belief_system: BeliefSystem, 
        context: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        """
        Detect contradictions between beliefs
        
        Args:
            belief_system: The belief system to check
            context: Current context information
            
        Returns:
            List of contradictions as (belief1_id, belief2_id, strength) tuples
        """
        # Use the system's built-in contradiction detection 
        contradictions = belief_system.find_contradictions()
        
        # Apply additional developmental filters based on current level
        if self.development_level < 0.3:
            # Early development: only detect strong contradictions
            filtered = [
                (id1, id2, strength) for id1, id2, strength in contradictions
                if strength > 0.7
            ]
            return filtered
            
        elif self.development_level < 0.6:
            # Middle development: detect moderate contradictions
            filtered = [
                (id1, id2, strength) for id1, id2, strength in contradictions
                if strength > self.parameters.contradiction_threshold
            ]
            return filtered
            
        else:
            # Advanced development: context-sensitive contradiction detection
            filtered = []
            
            for id1, id2, strength in contradictions:
                # Skip if below threshold
                if strength < self.parameters.contradiction_threshold:
                    continue
                    
                # Get the beliefs
                belief1 = belief_system.get_belief(id1)
                belief2 = belief_system.get_belief(id2)
                
                if not belief1 or not belief2:
                    continue
                    
                # Check if context reduces contradiction
                if self.parameters.context_sensitivity > 0 and context:
                    # Context might show these aren't actually contradictory
                    context_adjustment = self._evaluate_contextual_contradiction(
                        belief1, belief2, context
                    )
                    
                    # Adjust contradiction strength by context
                    adjusted_strength = strength * (1.0 - context_adjustment * self.parameters.context_sensitivity)
                    
                    # Only include if still significant
                    if adjusted_strength > self.parameters.contradiction_threshold:
                        filtered.append((id1, id2, adjusted_strength))
                else:
                    filtered.append((id1, id2, strength))
                    
            return filtered
    
    def _evaluate_contextual_contradiction(
        self, 
        belief1: Belief, 
        belief2: Belief, 
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate how much context reduces contradiction between beliefs
        
        Returns:
            Reduction factor (0.0-1.0) where higher values indicate more reduction
        """
        # Early development: no context sensitivity
        if self.development_level < 0.4:
            return 0.0
            
        # Check if context has keys that would differentiate these beliefs
        context_differentiation = 0.0
        relevant_context_keys = set()
        
        # Find contradictory content between beliefs
        contradictory_keys = set()
        for key in set(belief1.content.keys()) & set(belief2.content.keys()):
            if belief1.content[key] != belief2.content[key]:
                contradictory_keys.add(key)
                
        # Look for contextual factors that might explain the difference
        for key in context:
            # Check if context is related to contradictory content
            for c_key in contradictory_keys:
                # Simple partial string matching - could be more sophisticated
                if key in c_key or c_key in key:
                    relevant_context_keys.add(key)
                    break
                    
        # More relevant context factors reduce contradiction more
        if relevant_context_keys:
            context_differentiation = min(0.8, len(relevant_context_keys) * 0.2)
            
        return context_differentiation
    
    def _resolve_contradiction(
        self, 
        belief1: Belief, 
        belief2: Belief, 
        contradiction_strength: float,
        belief_system: BeliefSystem,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve a contradiction between beliefs
        
        Args:
            belief1: First contradictory belief
            belief2: Second contradictory belief
            contradiction_strength: Strength of the contradiction
            belief_system: The belief system
            context: Current context
            
        Returns:
            Resolution result information
        """
        # Early development (binary strategy)
        if self.parameters.resolution_strategy == "binary" or self.development_level < 0.3:
            return self._binary_resolution(belief1, belief2, belief_system)
            
        # Middle development (weighted strategy)
        elif self.parameters.resolution_strategy == "weighted" or self.development_level < 0.6:
            return self._weighted_resolution(belief1, belief2, belief_system)
            
        # Advanced development (contextual strategy)
        else:
            return self._contextual_resolution(belief1, belief2, contradiction_strength, belief_system, context)
    
    def _binary_resolution(
        self, 
        belief1: Belief, 
        belief2: Belief, 
        belief_system: BeliefSystem
    ) -> Dict[str, Any]:
        """
        Simple binary resolution - keep stronger belief, discard weaker one
        
        Args:
            belief1: First contradictory belief
            belief2: Second contradictory belief
            belief_system: The belief system
            
        Returns:
            Resolution result information
        """
        # Determine which belief is stronger
        if belief1.confidence > belief2.confidence:
            stronger = belief1
            weaker = belief2
            stronger_id = belief1.belief_id
            weaker_id = belief2.belief_id
        else:
            stronger = belief2
            weaker = belief1
            stronger_id = belief2.belief_id
            weaker_id = belief1.belief_id
            
        # Only remove weaker belief if the difference is significant
        confidence_diff = abs(belief1.confidence - belief2.confidence)
        
        if confidence_diff < 0.1:
            # Too close to call - for early development, random choice
            if random.random() < 0.5:
                stronger, weaker = weaker, stronger
                stronger_id, weaker_id = weaker_id, stronger_id
        
        # Remove the weaker belief
        belief_system.remove_belief(weaker_id)
        
        # Return resolution information
        return {
            "resolved": True,
            "resolution_type": "binary_removal",
            "details": {
                "kept_belief": stronger_id,
                "removed_belief": weaker_id,
                "confidence_difference": confidence_diff
            }
        }
    
    def _weighted_resolution(
        self, 
        belief1: Belief, 
        belief2: Belief, 
        belief_system: BeliefSystem
    ) -> Dict[str, Any]:
        """
        Weighted resolution - potentially merge beliefs or adjust confidences
        
        Args:
            belief1: First contradictory belief
            belief2: Second contradictory belief
            belief_system: The belief system
            
        Returns:
            Resolution result information
        """
        # Determine which belief is stronger
        if belief1.confidence > belief2.confidence:
            stronger = belief1
            weaker = belief2
            stronger_id = belief1.belief_id
            weaker_id = belief2.belief_id
        else:
            stronger = belief2
            weaker = belief1
            stronger_id = belief2.belief_id
            weaker_id = belief1.belief_id
            
        confidence_diff = abs(belief1.confidence - belief2.confidence)
        
        # If merge capability is available and confidences are close
        if self.parameters.merge_capability and confidence_diff < 0.2:
            # Create a merged belief
            merged_belief = stronger.model_copy(deep=True)
            
            # Resolve contradictory content by taking from stronger belief
            # and adjusting confidence
            confidence_adjustment = 0.1  # Confidence reduction due to contradiction
            
            # Take non-contradictory content from weaker belief
            for key, value in weaker.content.items():
                if key not in merged_belief.content:
                    merged_belief.content[key] = value
            
            # Reduce confidence due to contradiction
            merged_belief.confidence = max(0.1, stronger.confidence - confidence_adjustment)
            
            # Add evidence from both beliefs
            for evidence in weaker.evidence_for:
                if not any(e.evidence_id == evidence.evidence_id for e in merged_belief.evidence_for):
                    merged_belief.evidence_for.append(evidence)
                    
            for evidence in weaker.evidence_against:
                if not any(e.evidence_id == evidence.evidence_id for e in merged_belief.evidence_against):
                    merged_belief.evidence_against.append(evidence)
            
            # Update merged belief in system
            belief_system.update_belief(stronger_id, merged_belief)
            
            # Remove weaker belief
            belief_system.remove_belief(weaker_id)
            
            return {
                "resolved": True,
                "resolution_type": "merge",
                "details": {
                    "merged_into": stronger_id,
                    "removed_belief": weaker_id,
                    "confidence_adjustment": confidence_adjustment
                }
            }
            
        else:
            # Adjust confidence of both beliefs
            confidence_penalty1 = min(0.2, belief2.confidence * 0.3)
            confidence_penalty2 = min(0.2, belief1.confidence * 0.3)
            
            updated1 = belief1.model_copy(deep=True)
            updated1.confidence = max(0.1, belief1.confidence - confidence_penalty1)
            belief_system.update_belief(belief1.belief_id, updated1)
            
            updated2 = belief2.model_copy(deep=True)
            updated2.confidence = max(0.1, belief2.confidence - confidence_penalty2)
            belief_system.update_belief(belief2.belief_id, updated2)
            
            return {
                "resolved": True,
                "resolution_type": "mutual_confidence_adjustment",
                "details": {
                    "belief1_adjustment": confidence_penalty1,
                    "belief2_adjustment": confidence_penalty2
                }
            }
    
    def _contextual_resolution(
        self, 
        belief1: Belief, 
        belief2: Belief, 
        contradiction_strength: float,
        belief_system: BeliefSystem,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Contextual resolution - sophisticated handling with qualifiers and context
        
        Args:
            belief1: First contradictory belief
            belief2: Second contradictory belief
            contradiction_strength: Strength of the contradiction
            belief_system: The belief system
            context: Current context
            
        Returns:
            Resolution result information
        """
        # If contradiction is weak enough, don't resolve
        if contradiction_strength < 0.4 + self.parameters.ambiguity_tolerance:
            return {
                "resolved": False,
                "resolution_type": "tolerated_ambiguity",
                "details": {
                    "contradiction_strength": contradiction_strength,
                    "ambiguity_tolerance": self.parameters.ambiguity_tolerance
                }
            }
            
        # Check if these could be contextually distinct beliefs
        context_differentiation = self._evaluate_contextual_contradiction(belief1, belief2, context)
        
        # If context strongly differentiates them, add contextual qualifiers
        if context_differentiation > 0.5:
            # Add contextual qualifiers to both beliefs
            updated1 = belief1.model_copy(deep=True)
            updated2 = belief2.model_copy(deep=True)
            
            # Add context metadata
            updated1.metadata["contextual_scope"] = {
                key: value for key, value in context.items()
                if key in belief1.content or random.random() < 0.3  # Some randomness in selection
            }
            
            updated2.metadata["contextual_scope"] = {
                key: value for key, value in context.items()
                if key in belief2.content or random.random() < 0.3  # Some randomness in selection
            }
            
            # Update both beliefs
            belief_system.update_belief(belief1.belief_id, updated1)
            belief_system.update_belief(belief2.belief_id, updated2)
            
            return {
                "resolved": True,
                "resolution_type": "contextual_qualification",
                "details": {
                    "context_differentiation": context_differentiation,
                    "belief1_qualifiers": list(updated1.metadata["contextual_scope"].keys()),
                    "belief2_qualifiers": list(updated2.metadata["contextual_scope"].keys())
                }
            }
            
        # If multiple models are supported, maintain both beliefs
        if self.parameters.multiple_model_capacity > 1:
            # Tag both beliefs as belonging to different perspectives
            updated1 = belief1.model_copy(deep=True)
            updated2 = belief2.model_copy(deep=True)
            
            # Create or update perspective metadata
            if "perspective" not in updated1.metadata:
                updated1.metadata["perspective"] = f"perspective_{random.randint(1, self.parameters.multiple_model_capacity)}"
                
            if "perspective" not in updated2.metadata:
                # Ensure different perspective
                perspective2 = f"perspective_{random.randint(1, self.parameters.multiple_model_capacity)}"
                while perspective2 == updated1.metadata.get("perspective", ""):
                    perspective2 = f"perspective_{random.randint(1, self.parameters.multiple_model_capacity)}"
                updated2.metadata["perspective"] = perspective2
            
            # Update both beliefs
            belief_system.update_belief(belief1.belief_id, updated1)
            belief_system.update_belief(belief2.belief_id, updated2)
            
            return {
                "resolved": True,
                "resolution_type": "multiple_perspectives",
                "details": {
                    "belief1_perspective": updated1.metadata["perspective"],
                    "belief2_perspective": updated2.metadata["perspective"]
                }
            }
            
        # Fall back to weighted resolution for other cases
        return self._weighted_resolution(belief1, belief2, belief_system)
