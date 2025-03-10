"""
Belief Updating Module

This module is responsible for modifying existing beliefs based on new evidence.
It handles the evolution of beliefs over time, including confidence adjustments,
temporal decay, and integration of contradictory information.

The module's developmental progression moves from simple overwriting of beliefs
to more nuanced integration of new information with preservation of existing
belief structures when appropriate.
"""

from typing import Dict, List, Any, Optional
import logging
import time
import numpy as np
from datetime import datetime, timedelta

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.belief.models import Belief, Evidence, BeliefSystem

logger = logging.getLogger(__name__)

class BeliefUpdatingParameters:
    """Parameters controlling belief updating"""
    def __init__(self):
        # Threshold for evidence to trigger update
        self.update_threshold = 0.5
        # Rate of updating (how much new evidence affects beliefs)
        self.update_rate = 0.7
        # Weight given to conflicting evidence
        self.conflicting_evidence_weight = 0.5
        # Depth of source weighting (how many factors considered)
        self.source_weighting_depth = 1
        # How stable beliefs are (resistance to change)
        self.stability_factor = 0.3
        # Influence of update history on current updates
        self.history_influence = 0.2
        # Whether to perform partial updating vs. wholesale replacement
        self.partial_updating = False
        # Influence of related beliefs on this belief's updates
        self.related_belief_influence = 0.1

class BeliefUpdating(BaseModule):
    """
    Responsible for updating existing beliefs based on new evidence
    
    This module modifies existing beliefs in response to new information,
    with strategies that evolve as cognitive development progresses.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the belief updating module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="belief_updating", event_bus=event_bus)
        self.parameters = BeliefUpdatingParameters()
        self.update_history = []  # Track belief updates
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to update beliefs
        
        Args:
            input_data: Data including evidence and belief system
        
        Returns:
            Results including updated beliefs
        """
        if "evidence" not in input_data or "belief_system" not in input_data:
            return {
                "processed": False,
                "error": "Missing required input: evidence and belief_system",
                "module_id": self.module_id
            }
            
        evidence = input_data["evidence"]
        belief_system = input_data["belief_system"]
        belief_id = input_data.get("belief_id")  # Optional specific belief to update
        
        # Handle different evidence formats
        if isinstance(evidence, list):
            evidence_list = evidence
        else:
            evidence_list = [evidence]
            
        updated_beliefs = []
        
        # If specific belief ID provided, update only that belief
        if belief_id:
            belief = belief_system.get_belief(belief_id)
            if belief:
                updated = self._update_belief(belief, evidence_list, belief_system)
                belief_system.update_belief(belief_id, updated)
                
                updated_beliefs.append({
                    "belief_id": belief_id,
                    "content": updated.content,
                    "confidence": updated.confidence,
                    "previous_confidence": belief.confidence
                })
                
        # Otherwise, find and update all relevant beliefs
        else:
            # Find all beliefs potentially affected by this evidence
            beliefs_to_update = self._find_relevant_beliefs(evidence_list, belief_system)
            
            for belief_id in beliefs_to_update:
                belief = belief_system.get_belief(belief_id)
                if not belief:
                    continue
                    
                updated = self._update_belief(belief, evidence_list, belief_system)
                belief_system.update_belief(belief_id, updated)
                
                updated_beliefs.append({
                    "belief_id": belief_id,
                    "content": updated.content,
                    "confidence": updated.confidence,
                    "previous_confidence": belief.confidence
                })
                
        # Handle temporal decay for all beliefs if sufficient time has passed
        self._apply_temporal_decay(belief_system)
        
        # Return results
        return {
            "processed": True,
            "updated_beliefs": updated_beliefs,
            "module_id": self.module_id,
            "message": f"Updated {len(updated_beliefs)} belief(s)"
        }
        
    def _find_relevant_beliefs(self, evidence_list: List[Evidence], belief_system: BeliefSystem) -> List[str]:
        """Find beliefs that are relevant to the provided evidence"""
        relevant_beliefs = set()
        
        for evidence in evidence_list:
            # Find directly relevant beliefs based on content
            for belief_id, belief in belief_system.beliefs.items():
                # Check content overlap
                content_match = False
                for key in evidence.content:
                    if key in belief.content:
                        content_match = True
                        break
                        
                if content_match:
                    relevant_beliefs.add(belief_id)
        
        return list(relevant_beliefs)
    
    def _update_belief(self, belief: Belief, evidence_list: List[Evidence], belief_system: BeliefSystem) -> Belief:
        """Update a belief based on new evidence"""
        # Create a working copy of the belief
        updated_belief = belief.model_copy(deep=True)
        updated_belief.last_updated = datetime.now()
        
        # Sort evidence into supporting and contradicting
        supporting = []
        contradicting = []
        
        for evidence in evidence_list:
            is_supporting = self._is_evidence_supporting(evidence, belief)
            if is_supporting:
                # Add to supporting evidence if not already present
                if not any(e.evidence_id == evidence.evidence_id for e in updated_belief.evidence_for):
                    updated_belief.evidence_for.append(evidence)
                supporting.append(evidence)
            else:
                # Add to contradicting evidence if not already present
                if not any(e.evidence_id == evidence.evidence_id for e in updated_belief.evidence_against):
                    updated_belief.evidence_against.append(evidence)
                contradicting.append(evidence)
        
        # Early development: Simple overwriting or reinforcement
        if self.development_level < 0.3:
            # If strong contradicting evidence, simply adopt its content
            if contradicting and sum(e.reliability for e in contradicting) > belief.confidence * 1.5:
                # Take content from the most reliable contradicting evidence
                strongest = max(contradicting, key=lambda e: e.reliability)
                # Simple overwrite while preserving some structure
                for key, value in strongest.content.items():
                    updated_belief.content[key] = value
                    
        # Middle development: Partial integration
        elif self.development_level < 0.7 and self.parameters.partial_updating:
            # Selective content updating based on evidence reliability
            for evidence in supporting + contradicting:
                # Only update if evidence is reliable enough
                if evidence.reliability > self.parameters.update_threshold:
                    # Update specific content elements
                    for key, value in evidence.content.items():
                        # Calculate change factor based on evidence quality
                        change_factor = evidence.reliability * evidence.relevance
                        # Adjust by belief stability
                        change_factor *= (1.0 - updated_belief.stability)
                        
                        # If key already exists, weighted update
                        if key in updated_belief.content:
                            if isinstance(value, (int, float)) and isinstance(updated_belief.content[key], (int, float)):
                                # Numerical blending
                                updated_belief.content[key] = (
                                    (1.0 - change_factor) * updated_belief.content[key] + 
                                    change_factor * value
                                )
                            else:
                                # For non-numeric, update if strong enough evidence
                                if change_factor > 0.6:
                                    updated_belief.content[key] = value
                        else:
                            # Add new content
                            updated_belief.content[key] = value
        
        # Advanced development: Sophisticated integration
        elif self.development_level >= 0.7:
            # Consider related beliefs
            if self.parameters.related_belief_influence > 0:
                related_beliefs = {}
                
                # Find related beliefs with their relationship strength
                if updated_belief.related_beliefs:
                    for related_id in updated_belief.related_beliefs:
                        related = belief_system.get_belief(related_id)
                        if related:
                            relationship_strength = self._calculate_relationship_strength(
                                updated_belief, related, belief_system
                            )
                            related_beliefs[related_id] = (related, relationship_strength)
                
                # Integrate influence from related beliefs
                for related_id, (related, strength) in related_beliefs.items():
                    influence = strength * self.parameters.related_belief_influence
                    # Apply influence to update
                    for key, value in related.content.items():
                        if key in updated_belief.content:
                            if isinstance(value, (int, float)) and isinstance(updated_belief.content[key], (int, float)):
                                # Subtle influence for numeric values
                                updated_belief.content[key] = (
                                    (1.0 - influence) * updated_belief.content[key] + 
                                    influence * value
                                )
            
            # Complex content integration from new evidence
            for evidence in supporting + contradicting:
                # Calculate evidence influence
                influence = evidence.reliability * evidence.relevance
                if evidence in contradicting:
                    influence *= self.parameters.conflicting_evidence_weight
                
                # Adjust by belief stability and history
                influence *= (1.0 - updated_belief.stability)
                
                # Apply Bayesian-inspired content updating
                for key, value in evidence.content.items():
                    # Different handling for numeric vs categorical
                    if key in updated_belief.content:
                        if isinstance(value, (int, float)) and isinstance(updated_belief.content[key], (int, float)):
                            # Weighted average for numeric values
                            updated_belief.content[key] = (
                                (1.0 - influence) * updated_belief.content[key] + 
                                influence * value
                            )
                        else:
                            # For non-numeric, probabilistic update
                            if influence > 1.0 - updated_belief.confidence:
                                updated_belief.content[key] = value
                    elif influence > 0.4:  # Threshold for adding new content
                        # Add new content
                        updated_belief.content[key] = value
                    
        # Record the update
        self.update_history.append({
            "belief_id": updated_belief.belief_id,
            "timestamp": datetime.now(),
            "previous_confidence": belief.confidence,
            "new_confidence": updated_belief.confidence,
            "supporting_evidence": len(supporting),
            "contradicting_evidence": len(contradicting)
        })
        
        # Limit history size
        if len(self.update_history) > 100:
            self.update_history = self.update_history[-100:]
        
        # Confidence will be updated by model validator
        return updated_belief
    
    def _is_evidence_supporting(self, evidence: Evidence, belief: Belief) -> bool:
        """Determine if evidence supports or contradicts the belief"""
        matches = 0
        mismatches = 0
        
        for key, value in belief.content.items():
            if key in evidence.content:
                if evidence.content[key] == value:
                    matches += 1
                else:
                    mismatches += 1
        
        # Early development: Simple majority voting
        if self.development_level < 0.3:
            return matches >= mismatches
            
        # More sophisticated reasoning with development
        if matches + mismatches == 0:
            # No direct comparison possible
            return evidence.reliability > 0.5  # Default based on reliability
            
        support_ratio = matches / (matches + mismatches)
        
        # Adjust by evidence quality
        weighted_support = support_ratio * evidence.reliability * evidence.relevance
        
        return weighted_support >= 0.5
    
    def _calculate_relationship_strength(
        self, 
        belief1: Belief, 
        belief2: Belief,
        belief_system: BeliefSystem
    ) -> float:
        """Calculate relationship strength between two beliefs"""
        # Content similarity
        similarity = 0.0
        shared_keys = set(belief1.content.keys()) & set(belief2.content.keys())
        
        if shared_keys:
            matches = sum(1 for k in shared_keys if belief1.content[k] == belief2.content[k])
            similarity = matches / len(shared_keys)
        
        # Evidence overlap
        evidence1 = {e.evidence_id for e in belief1.evidence_for + belief1.evidence_against}
        evidence2 = {e.evidence_id for e in belief2.evidence_for + belief2.evidence_against}
        
        overlap = 0.0
        if evidence1 and evidence2:
            intersection = evidence1 & evidence2
            overlap = len(intersection) / min(len(evidence1), len(evidence2))
        
        # Combine factors
        return 0.7 * similarity + 0.3 * overlap
    
    def _apply_temporal_decay(self, belief_system: BeliefSystem) -> None:
        """Apply time-based decay to belief confidences"""
        # Only apply decay periodically
        current_time = datetime.now()
        
        # Skip if not enough time has passed since last decay
        if hasattr(self, 'last_decay_time') and (current_time - self.last_decay_time).total_seconds() < 3600:
            return
            
        self.last_decay_time = current_time
        
        # Apply decay to each belief
        for belief_id, belief in belief_system.beliefs.items():
            # Calculate time since last update
            if not belief.last_updated:
                continue
                
            time_diff = (current_time - belief.last_updated).total_seconds() / 86400.0  # Days
            
            # Different decay rates based on development level
            if self.development_level < 0.3:
                # Early development: faster forgetting
                decay_rate = 0.1
            elif self.development_level < 0.7:
                # Middle development: more stable
                decay_rate = 0.05
            else:
                # Advanced: very stable long-term beliefs
                decay_rate = 0.02
                
            # Adjust by belief stability
            decay_rate *= (1.0 - belief.stability)
            
            # Calculate confidence decay
            decay_amount = min(0.5, decay_rate * time_diff)  # Cap at 50% decay
            
            # Only apply significant decay
            if decay_amount > 0.01:
                # Create updated belief
                updated = belief.model_copy(deep=True)
                
                # Apply decay to confidence
                updated.confidence = max(0.1, updated.confidence - decay_amount)
                
                # Update the belief
                belief_system.update_belief(belief_id, updated)
                
                logger.debug(f"Applied temporal decay to belief {belief_id}: {decay_amount:.2f}")
                
    def _adjust_belief_properties(self, belief: Belief) -> None:
        """Adjust belief properties based on development level"""
        # Stability increases with development level
        base_stability = 0.1 + (self.development_level * 0.4)
        
        # Evidence quantity increases stability
        evidence_count = len(belief.evidence_for) + len(belief.evidence_against)
        count_factor = min(0.4, evidence_count * 0.05)
        
        # Time stability - older beliefs are more stable
        age_hours = (datetime.now() - belief.creation_time).total_seconds() / 3600
        age_factor = min(0.3, (age_hours / 240) * 0.3)  # Max effect after 10 days
        
        # Update stability
        belief.stability = min(0.9, base_stability + count_factor + age_factor)
