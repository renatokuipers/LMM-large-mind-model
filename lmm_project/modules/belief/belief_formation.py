"""
Belief Formation Module

This module is responsible for forming new beliefs based on evidence.
It evaluates how pieces of evidence support potential beliefs and creates
new belief structures with appropriate confidence levels.

The module's developmental progression moves from simple, direct evidence-based
beliefs to more sophisticated beliefs that incorporate contextual understanding,
uncertainty, and integration with existing beliefs.
"""

from typing import Dict, List, Any, Optional
import logging
import time
import numpy as np
from datetime import datetime

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.belief.models import Belief, Evidence, BeliefSystem

logger = logging.getLogger(__name__)

class BeliefFormationParameters:
    """Parameters controlling belief formation"""
    def __init__(self):
        # Minimum evidence threshold for forming beliefs
        self.min_evidence_threshold = 0.5
        # Minimum number of evidence pieces required
        self.min_evidence_count = 1
        # Confidence threshold for belief formation
        self.confidence_threshold = 0.5
        # How much to factor in indirect evidence
        self.indirect_evidence_factor = 0.3
        # Whether to weigh sources differently
        self.source_weighting = False
        # Sensitivity to context when forming beliefs
        self.context_sensitivity = 0.5
        # Whether to handle uncertainty in belief formation
        self.uncertainty_handling = False

class BeliefFormation(BaseModule):
    """
    Responsible for forming new beliefs based on evidence
    
    This module evaluates evidence and forms beliefs based on developmental
    appropriate strategies.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the belief formation module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="belief_formation", event_bus=event_bus)
        self.parameters = BeliefFormationParameters()
        self.recent_formations = []  # Track recently formed beliefs
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to form beliefs
        
        Args:
            input_data: Data including evidence and belief system
        
        Returns:
            Results including any formed beliefs
        """
        if "evidence" not in input_data or "belief_system" not in input_data:
            return {
                "processed": False,
                "error": "Missing required input: evidence and belief_system",
                "module_id": self.module_id
            }
            
        evidence = input_data["evidence"]
        belief_system = input_data["belief_system"]
        context = input_data.get("context", {})
        
        # Form beliefs from the evidence
        formed_beliefs = []
        
        # Check evidence quality
        if isinstance(evidence, list):
            # Multiple evidence pieces
            evidence_list = evidence
        else:
            # Single evidence piece
            evidence_list = [evidence]
            
        # Only proceed if sufficient evidence quality and quantity
        if len(evidence_list) < self.parameters.min_evidence_count:
            return {
                "processed": True,
                "formed_beliefs": [],
                "message": "Insufficient evidence quantity",
                "module_id": self.module_id
            }
            
        # Group evidence by potential belief content
        belief_candidates = self._group_evidence_by_content(evidence_list, context)
        
        # Evaluate each candidate belief
        for content_key, evidence_groups in belief_candidates.items():
            # Check if this content already exists in a belief
            existing_belief = self._find_matching_belief(belief_system, evidence_groups["content"])
            
            if existing_belief:
                # Update existing belief with new evidence
                if self.event_bus:
                    # Request belief update via event bus
                    self._request_belief_update(existing_belief.belief_id, evidence_groups)
                continue
                
            # Calculate overall evidence quality
            evidence_strength = self._calculate_evidence_strength(
                evidence_groups["supporting"], 
                evidence_groups["contradicting"]
            )
            
            # Only form belief if evidence is strong enough
            if evidence_strength >= self.parameters.min_evidence_threshold:
                # Create the new belief
                belief = self._form_belief(evidence_groups, context)
                
                # Add to belief system
                belief_id = belief_system.add_belief(belief)
                
                # Record formation
                formed_beliefs.append({
                    "belief_id": belief_id,
                    "content": belief.content,
                    "confidence": belief.confidence,
                    "evidence_count": len(belief.evidence_for) + len(belief.evidence_against)
                })
                
                # Track recently formed beliefs
                self.recent_formations.append({
                    "belief_id": belief_id,
                    "timestamp": datetime.now(),
                    "context": context
                })
                
                # Limit history size
                if len(self.recent_formations) > 50:
                    self.recent_formations = self.recent_formations[-50:]
                    
                logger.info(f"Formed belief: {belief.content} with confidence {belief.confidence:.2f}")
                
        # Return results
        return {
            "processed": True,
            "formed_beliefs": formed_beliefs,
            "module_id": self.module_id,
            "message": f"Formed {len(formed_beliefs)} belief(s)"
        }
    
    def _request_belief_update(self, belief_id: str, evidence_groups: Dict[str, Any]) -> None:
        """Request update of an existing belief with new evidence"""
        # Send message to belief updating component
        if self.event_bus:
            message = Message(
                msg_type="belief_update_request",
                sender=self.module_id,
                content={
                    "belief_id": belief_id,
                    "supporting_evidence": evidence_groups["supporting"],
                    "contradicting_evidence": evidence_groups["contradicting"]
                }
            )
            self.event_bus.publish(message)
    
    def _find_matching_belief(self, belief_system: BeliefSystem, content: Dict[str, Any]) -> Optional[Belief]:
        """Find a belief with matching content"""
        # Check each belief for matching content
        for belief_id, belief in belief_system.beliefs.items():
            matches = True
            
            # Check each key in the content
            for key, value in content.items():
                if key not in belief.content or belief.content[key] != value:
                    matches = False
                    break
                    
            if matches and len(content) > 0:
                return belief
                
        return None
    
    def _group_evidence_by_content(
        self, 
        evidence_list: List[Evidence], 
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Group evidence by potential belief content"""
        belief_candidates = {}
        
        for evidence in evidence_list:
            # Generate a content key based on the evidence
            content = self._extract_belief_content(evidence, context)
            content_key = self._generate_content_key(content)
            
            # Initialize if new content
            if content_key not in belief_candidates:
                belief_candidates[content_key] = {
                    "content": content,
                    "supporting": [],
                    "contradicting": []
                }
                
            # Determine if this evidence supports or contradicts
            is_supporting = self._is_supporting_evidence(evidence, content)
            
            # Add to appropriate list
            if is_supporting:
                belief_candidates[content_key]["supporting"].append(evidence)
            else:
                belief_candidates[content_key]["contradicting"].append(evidence)
                
        return belief_candidates
    
    def _extract_belief_content(self, evidence: Evidence, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract potential belief content from evidence"""
        # Basic content extraction - directly from evidence
        # Later development levels could use more sophisticated extraction
        
        content = {}
        
        # Early development: simple mapping from evidence to belief
        if self.development_level < 0.3:
            # Direct content copy with minimal processing
            return dict(evidence.content)
            
        # Middle development: consider context and source
        elif self.development_level < 0.7:
            # Enhance with context if enabled
            if self.parameters.context_sensitivity > 0:
                # Integrate some context elements
                for key, value in context.items():
                    if key in evidence.content and evidence.content[key] != value:
                        # Resolve conflicts based on reliability
                        if evidence.reliability > 0.7:
                            content[key] = evidence.content[key]
                        else:
                            content[key] = value
                    elif key not in evidence.content:
                        content[key] = value
                        
            # Add remaining evidence content
            for key, value in evidence.content.items():
                if key not in content:
                    content[key] = value
                    
            return content
                    
        # Advanced development: sophisticated content extraction
        else:
            # Use evidence content as base
            content = dict(evidence.content)
            
            # Enhance with context using weighted integration
            context_weight = self.parameters.context_sensitivity
            
            for key, value in context.items():
                if key in content:
                    # Weighted blend of evidence and context
                    if isinstance(value, (int, float)) and isinstance(content[key], (int, float)):
                        # Numerical blending
                        content[key] = (1 - context_weight) * content[key] + context_weight * value
                    elif evidence.reliability < 0.6:
                        # For non-numerical, use context if evidence reliability is low
                        content[key] = value
                else:
                    # Add context if not in evidence
                    content[key] = value
                    
            return content
    
    def _generate_content_key(self, content: Dict[str, Any]) -> str:
        """Generate a unique key for belief content"""
        # Simple string representation of sorted content items
        items = sorted((str(k), str(v)) for k, v in content.items())
        return "|".join(f"{k}:{v}" for k, v in items)
    
    def _is_supporting_evidence(self, evidence: Evidence, content: Dict[str, Any]) -> bool:
        """Determine if evidence supports or contradicts the belief content"""
        # Simple implementation: if most content matches, it's supporting
        matches = 0
        mismatches = 0
        
        for key, value in content.items():
            if key in evidence.content:
                if evidence.content[key] == value:
                    matches += 1
                else:
                    mismatches += 1
                    
        # Early development: binary support/contradict
        if self.development_level < 0.4:
            return matches > mismatches
            
        # More developed: weighted by matches and evidence quality
        support_score = matches / (matches + mismatches) if (matches + mismatches) > 0 else 0.5
        
        # Adjust by evidence reliability and relevance
        weighted_score = support_score * evidence.reliability * evidence.relevance
        
        return weighted_score >= 0.5
    
    def _calculate_evidence_strength(
        self, 
        supporting_evidence: List[Evidence], 
        contradicting_evidence: List[Evidence]
    ) -> float:
        """Calculate overall evidence strength for belief formation"""
        if not supporting_evidence:
            return 0.0
            
        # Calculate weighted strength of supporting evidence
        supporting_strength = sum(
            e.reliability * e.relevance for e in supporting_evidence
        )
        
        # Calculate weighted strength of contradicting evidence
        contradicting_strength = sum(
            e.reliability * e.relevance for e in contradicting_evidence
        )
        
        # Early development: simple ratio
        if self.development_level < 0.3:
            total_strength = supporting_strength + contradicting_strength
            if total_strength == 0:
                return 0.0
                
            return supporting_strength / total_strength
            
        # More developed: non-linear combination with thresholds
        else:
            # If contradicting evidence is substantial, require stronger support
            if contradicting_strength > 0:
                ratio = supporting_strength / contradicting_strength
                
                # Non-linear scaling that favors evidence consensus
                adjusted_strength = np.tanh(ratio) * 0.5 + 0.5  # Range: [0, 1]
                
                # Scale by total evidence quantity
                total_count = len(supporting_evidence) + len(contradicting_evidence)
                quantity_factor = min(1.0, total_count / 3.0)  # More evidence increases strength
                
                return adjusted_strength * quantity_factor
            else:
                # No contradicting evidence
                return min(1.0, supporting_strength / len(supporting_evidence))
    
    def _form_belief(self, evidence_groups: Dict[str, Any], context: Dict[str, Any]) -> Belief:
        """Form a new belief based on evidence"""
        # Create the belief
        belief = Belief(
            content=evidence_groups["content"],
            confidence=0.5,  # Initial confidence will be updated
            evidence_for=evidence_groups["supporting"],
            evidence_against=evidence_groups["contradicting"],
            source="evidence"
        )
        
        # Early development: stability is low (easily changed)
        if self.development_level < 0.3:
            belief.stability = 0.1
        # Medium development: moderate stability
        elif self.development_level < 0.7:
            belief.stability = 0.3
        # Advanced development: higher stability for well-supported beliefs
        else:
            evidence_count = len(evidence_groups["supporting"]) + len(evidence_groups["contradicting"])
            belief.stability = min(0.7, 0.2 + (evidence_count * 0.1))
            
        # Update confidence based on evidence
        # The model_validator will handle this automatically
        
        # Set metadata from context if available
        if context:
            belief.metadata["formation_context"] = {
                key: value for key, value in context.items()
                if key in ["task", "emotional_state", "attention_focus"]
            }
            
        # Add timestamp
        belief.metadata["formation_time"] = datetime.now().isoformat()
        belief.metadata["developmental_level"] = self.development_level
        
        return belief
