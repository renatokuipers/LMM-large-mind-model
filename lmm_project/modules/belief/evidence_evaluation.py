"""
Evidence Evaluation Module

This module assesses the quality, reliability, and relevance of evidence used
in belief formation and updating. It evaluates evidence based on source credibility,
consistency with other evidence, and contextual factors.

The module's developmental progression moves from simple acceptance of evidence
to sophisticated critical analysis with consideration of source reliability,
methodology, and potential biases.
"""

from typing import Dict, List, Any, Optional
from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.belief.models import Evidence
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EvidenceEvaluationParameters:
    """Parameters controlling evidence evaluation"""
    def __init__(self):
        # Default reliability for unknown sources
        self.default_reliability = 0.5
        # Weight given to source credibility
        self.source_credibility_weight = 0.7
        # Weight given to evidence consistency with prior evidence
        self.consistency_weight = 0.5
        # Depth of analysis (how many factors considered)
        self.analysis_depth = 2
        # Sensitivity to conflicting details
        self.conflict_sensitivity = 0.6
        # Whether to differentiate between source types
        self.source_differentiation = False
        # Number of source types recognized
        self.source_types_recognized = 2
        # Whether to consider methodology in evaluation
        self.methodology_awareness = False
        # Weight given to methodological rigor
        self.methodology_weight = 0.0
        # Whether to apply statistical reasoning
        self.statistical_reasoning = False

class EvidenceEvaluation(BaseModule):
    """
    Responsible for evaluating the quality and reliability of evidence
    
    This module assesses evidence used in belief formation and updating,
    determining how strongly it should influence the belief system.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the evidence evaluation module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="evidence_evaluation", event_bus=event_bus)
        self.parameters = EvidenceEvaluationParameters()
        self.source_reliability_cache = {}  # Cache source reliability assessments
        self.evaluation_history = []  # Track evidence evaluations
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs to evaluate evidence
        
        Args:
            input_data: Data including evidence to evaluate
        
        Returns:
            Results including evaluated evidence
        """
        if "evidence" not in input_data:
            return {
                "processed": False,
                "error": "Missing required input: evidence",
                "module_id": self.module_id
            }
            
        # Get evidence and context
        evidence = input_data["evidence"]
        context = input_data.get("context", {})
        prior_evidence = input_data.get("prior_evidence", [])
        
        # Handle different evidence formats
        if isinstance(evidence, list):
            evidence_list = evidence
        else:
            evidence_list = [evidence]
            
        # Evaluate each piece of evidence
        evaluated_evidence = []
        
        for evidence_item in evidence_list:
            # Create a working copy
            evaluated = self._evaluate_evidence(evidence_item, context, prior_evidence)
            evaluated_evidence.append(evaluated)
            
            # Record evaluation
            self.evaluation_history.append({
                "evidence_id": evaluated.evidence_id,
                "source": evaluated.source,
                "reliability": evaluated.reliability,
                "relevance": evaluated.relevance,
                "timestamp": datetime.now()
            })
            
            # Limit history size
            if len(self.evaluation_history) > 100:
                self.evaluation_history = self.evaluation_history[-100:]
                
        # Return results
        return {
            "processed": True,
            "evaluated_evidence": evaluated_evidence if len(evaluated_evidence) > 1 else evaluated_evidence[0],
            "module_id": self.module_id
        }
        
    def _evaluate_evidence(
        self, 
        evidence: Evidence, 
        context: Dict[str, Any],
        prior_evidence: List[Evidence]
    ) -> Evidence:
        """
        Evaluate a single piece of evidence
        
        Args:
            evidence: The evidence to evaluate
            context: Current context information
            prior_evidence: Previously evaluated evidence
            
        Returns:
            Evaluated evidence with updated reliability and relevance
        """
        # Create a working copy
        evaluated = evidence.model_copy(deep=True)
        
        # Evaluate source reliability
        source_reliability = self._evaluate_source_reliability(evidence.source, evidence.content)
        
        # Evaluate consistency with prior evidence
        consistency_score = self._evaluate_consistency(evidence, prior_evidence)
        
        # Evaluate relevance to context
        relevance_score = self._evaluate_relevance(evidence, context)
        
        # Early development: Simple source-based reliability
        if self.development_level < 0.3:
            # Just use source reliability with minimal adjustment
            evaluated.reliability = source_reliability
            evaluated.relevance = max(0.5, relevance_score)
            
        # Middle development: Consider consistency and relevance
        elif self.development_level < 0.7:
            # Combine source reliability and consistency
            source_weight = self.parameters.source_credibility_weight
            consistency_weight = self.parameters.consistency_weight
            
            # Normalize weights
            total_weight = source_weight + consistency_weight
            source_weight /= total_weight
            consistency_weight /= total_weight
            
            # Calculate combined reliability
            evaluated.reliability = (
                source_weight * source_reliability +
                consistency_weight * consistency_score
            )
            
            # Set relevance
            evaluated.relevance = relevance_score
            
        # Advanced development: Sophisticated evaluation
        else:
            # Consider methodology if enabled
            methodology_score = 0.0
            if self.parameters.methodology_awareness:
                methodology_score = self._evaluate_methodology(evidence)
                
            # Consider statistical properties if enabled
            statistical_score = 0.0
            if self.parameters.statistical_reasoning:
                statistical_score = self._evaluate_statistical_properties(evidence)
                
            # Calculate weights
            source_weight = self.parameters.source_credibility_weight
            consistency_weight = self.parameters.consistency_weight
            methodology_weight = self.parameters.methodology_weight
            statistical_weight = 0.3 if self.parameters.statistical_reasoning else 0.0
            
            # Normalize weights
            total_weight = source_weight + consistency_weight + methodology_weight + statistical_weight
            if total_weight > 0:
                source_weight /= total_weight
                consistency_weight /= total_weight
                methodology_weight /= total_weight
                statistical_weight /= total_weight
            
            # Calculate combined reliability
            evaluated.reliability = (
                source_weight * source_reliability +
                consistency_weight * consistency_score +
                methodology_weight * methodology_score +
                statistical_weight * statistical_score
            )
            
            # Set relevance with context-sensitivity
            evaluated.relevance = relevance_score
            
        # Ensure values are in valid range
        evaluated.reliability = max(0.0, min(1.0, evaluated.reliability))
        evaluated.relevance = max(0.0, min(1.0, evaluated.relevance))
        
        return evaluated
    
    def _evaluate_source_reliability(self, source: str, content: Dict[str, Any]) -> float:
        """
        Evaluate the reliability of an evidence source
        
        Args:
            source: Source identifier
            content: Evidence content
            
        Returns:
            Source reliability score (0.0-1.0)
        """
        # Check if we have cached reliability for this source
        if source in self.source_reliability_cache:
            return self.source_reliability_cache[source]
            
        reliability = self.parameters.default_reliability
        
        # Early development: Simple fixed reliabilities
        if self.development_level < 0.3:
            if source == "perception":
                reliability = 0.8  # Perception is considered highly reliable
            elif source == "memory":
                reliability = 0.6  # Memory is moderately reliable
            elif source == "language":
                reliability = 0.5  # Language input is variable
            elif source == "reasoning":
                reliability = 0.7  # Internal reasoning is fairly reliable
                
        # Middle development: More source types with quality assessment
        elif self.development_level < 0.7:
            if source == "perception":
                # Adjust based on clarity if available
                clarity = content.get("clarity", 0.8)
                reliability = 0.6 + (clarity * 0.4)
            elif source == "memory":
                # Adjust based on recency if available
                age = content.get("age", 0.5)
                age_factor = max(0.0, 1.0 - age)
                reliability = 0.4 + (age_factor * 0.5)
            elif source == "language":
                # Adjust based on source trustworthiness
                trustworthiness = content.get("source_trustworthiness", 0.5)
                reliability = trustworthiness
            elif source == "reasoning":
                # Adjust based on confidence if available
                confidence = content.get("confidence", 0.7)
                reliability = 0.5 + (confidence * 0.4)
            else:
                # Unknown source types get lower reliability
                reliability = 0.4
                
        # Advanced development: Sophisticated source evaluation
        else:
            # Base reliability factors
            if source == "perception":
                base_reliability = 0.8
                
                # Modulate by perceptual factors
                clarity = content.get("clarity", 0.8)
                attention = content.get("attention_level", 0.7)
                
                # Adjust base reliability
                reliability = base_reliability * (0.6 + (clarity * 0.2) + (attention * 0.2))
                
            elif source == "memory":
                base_reliability = 0.7
                
                # Modulate by memory factors
                age = content.get("age", 0.5)
                emotional_significance = content.get("emotional_significance", 0.5)
                
                # Adjust for age (newer memories more reliable)
                age_factor = max(0.3, 1.0 - (age * 0.6))
                
                # Emotional memories are more reliable but can be distorted
                emotional_factor = 0.5 + (emotional_significance * 0.5)
                
                # Combine factors
                reliability = base_reliability * (age_factor * 0.6 + emotional_factor * 0.4)
                
            elif source == "language":
                base_reliability = 0.6
                
                # Consider source factors
                trustworthiness = content.get("source_trustworthiness", 0.5)
                expertise = content.get("source_expertise", 0.5)
                
                # Consider content factors
                verifiability = content.get("verifiability", 0.5)
                consistency = content.get("internal_consistency", 0.7)
                
                # Combine factors with appropriate weights
                reliability = base_reliability * (
                    trustworthiness * 0.3 +
                    expertise * 0.3 +
                    verifiability * 0.2 +
                    consistency * 0.2
                )
                
            elif source == "reasoning":
                base_reliability = 0.7
                
                # Consider reasoning quality factors
                logic_quality = content.get("logic_quality", 0.7)
                evidence_quality = content.get("evidence_quality", 0.6)
                confidence = content.get("confidence", 0.6)
                
                # Combine factors
                reliability = base_reliability * (
                    logic_quality * 0.4 +
                    evidence_quality * 0.4 +
                    confidence * 0.2
                )
                
            else:
                # More sophisticated unknown source handling
                reliability = 0.3 + (random.random() * 0.2)  # Some randomness in evaluation
        
        # Cache the result for future use
        self.source_reliability_cache[source] = reliability
        
        # Ensure value is in valid range
        return max(0.0, min(1.0, reliability))
    
    def _evaluate_consistency(self, evidence: Evidence, prior_evidence: List[Evidence]) -> float:
        """
        Evaluate consistency of evidence with prior evidence
        
        Args:
            evidence: Evidence to evaluate
            prior_evidence: Previously evaluated evidence
            
        Returns:
            Consistency score (0.0-1.0)
        """
        if not prior_evidence:
            return 1.0  # No prior evidence to check consistency against
            
        # Early development: Simple agreement counting
        if self.development_level < 0.3:
            agreements = 0
            disagreements = 0
            
            for prior in prior_evidence:
                # Check for overlapping content
                overlap = False
                
                for key in evidence.content:
                    if key in prior.content:
                        overlap = True
                        if evidence.content[key] == prior.content[key]:
                            agreements += 1
                        else:
                            disagreements += 1
            
            total = agreements + disagreements
            if total == 0:
                return 0.7  # No direct overlaps, neutral consistency
                
            return agreements / total
            
        # More advanced: Weighted agreement based on source and recency
        else:
            total_score = 0.0
            total_weight = 0.0
            
            for prior in prior_evidence:
                # Calculate temporal recency (more recent evidence counts more)
                time_diff = (evidence.timestamp - prior.timestamp).total_seconds() if evidence.timestamp and prior.timestamp else 86400
                recency = 1.0 / (1.0 + (time_diff / 86400))  # Normalize to 0-1 range
                
                # Calculate content overlap
                overlap_keys = set(evidence.content.keys()) & set(prior.content.keys())
                if not overlap_keys:
                    continue
                    
                # Count agreements and disagreements
                agreements = sum(1 for k in overlap_keys if evidence.content[k] == prior.content[k])
                disagreements = len(overlap_keys) - agreements
                
                # Calculate agreement ratio
                if len(overlap_keys) > 0:
                    agreement_ratio = agreements / len(overlap_keys)
                    
                    # Weight by prior evidence reliability and recency
                    weight = prior.reliability * (0.5 + (recency * 0.5))
                    
                    total_score += agreement_ratio * weight
                    total_weight += weight
            
            if total_weight == 0:
                return 0.7  # No overlapping content, neutral consistency
                
            return total_score / total_weight
    
    def _evaluate_relevance(self, evidence: Evidence, context: Dict[str, Any]) -> float:
        """
        Evaluate relevance of evidence to current context
        
        Args:
            evidence: Evidence to evaluate
            context: Current context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        if not context:
            return 0.7  # No context, assume moderate relevance
            
        # Early development: Simple key matching
        if self.development_level < 0.3:
            # Count matching keys between evidence and context
            evidence_keys = set(evidence.content.keys())
            context_keys = set(context.keys())
            
            matching_keys = evidence_keys.intersection(context_keys)
            
            if not evidence_keys:
                return 0.5  # Empty evidence, neutral relevance
                
            return len(matching_keys) / len(evidence_keys)
            
        # More advanced: Content-based relevance
        else:
            # Look for content overlap
            direct_matches = 0
            total_keys = len(evidence.content)
            
            if total_keys == 0:
                return 0.5  # Empty evidence, neutral relevance
            
            # Check for direct content matches
            for key, value in evidence.content.items():
                if key in context and context[key] == value:
                    direct_matches += 1
            
            # Check for related content
            related_matches = 0
            for key in evidence.content:
                # Look for related keys in context
                for context_key in context:
                    # Simple relatedness check using string overlap
                    if key in context_key or context_key in key:
                        related_matches += 0.5
                        break
            
            # Combine direct and related matches
            weighted_matches = direct_matches + (related_matches * 0.5)
            relevance = min(1.0, weighted_matches / total_keys)
            
            return max(0.1, relevance)  # Ensure minimum relevance
    
    def _evaluate_methodology(self, evidence: Evidence) -> float:
        """
        Evaluate methodological quality of evidence
        
        Args:
            evidence: Evidence to evaluate
            
        Returns:
            Methodology quality score (0.0-1.0)
        """
        # Only available at higher development levels
        if self.development_level < 0.6:
            return 0.5
            
        # Look for methodology indicators in evidence
        methodology_indicators = {
            'sample_size': evidence.content.get('sample_size', 0),
            'control_group': evidence.content.get('control_group', False),
            'randomization': evidence.content.get('randomization', False),
            'peer_reviewed': evidence.content.get('peer_reviewed', False),
            'replication': evidence.content.get('replication', False)
        }
        
        # Calculate methodology score
        score = 0.5  # Default score
        
        # Sample size effect
        if methodology_indicators['sample_size'] > 0:
            # Logarithmic scaling of sample size benefit
            sample_size_score = min(0.3, 0.1 + (np.log10(methodology_indicators['sample_size']) * 0.05))
            score += sample_size_score
            
        # Control group effect
        if methodology_indicators['control_group']:
            score += 0.15
            
        # Randomization effect
        if methodology_indicators['randomization']:
            score += 0.1
            
        # Peer review effect
        if methodology_indicators['peer_reviewed']:
            score += 0.15
            
        # Replication effect
        if methodology_indicators['replication']:
            score += 0.1
            
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def _evaluate_statistical_properties(self, evidence: Evidence) -> float:
        """
        Evaluate statistical properties of evidence
        
        Args:
            evidence: Evidence to evaluate
            
        Returns:
            Statistical quality score (0.0-1.0)
        """
        # Only available at higher development levels
        if self.development_level < 0.8:
            return 0.5
            
        # Look for statistical indicators in evidence
        statistical_indicators = {
            'p_value': evidence.content.get('p_value', 1.0),
            'confidence_interval': evidence.content.get('confidence_interval', 0.0),
            'effect_size': evidence.content.get('effect_size', 0.0),
            'statistical_power': evidence.content.get('statistical_power', 0.0)
        }
        
        # Calculate statistical quality score
        score = 0.5  # Default score
        
        # P-value effect (lower is better)
        if statistical_indicators['p_value'] < 1.0:
            p_value = statistical_indicators['p_value']
            if p_value < 0.01:
                score += 0.2
            elif p_value < 0.05:
                score += 0.15
            elif p_value < 0.1:
                score += 0.05
            else:
                score -= 0.1  # Penalize high p-values
                
        # Effect size effect
        effect_size = statistical_indicators['effect_size']
        if effect_size > 0:
            score += min(0.2, effect_size * 0.5)
            
        # Statistical power effect
        power = statistical_indicators['statistical_power']
        if power > 0:
            if power > 0.8:
                score += 0.15
            elif power > 0.5:
                score += 0.1
            else:
                score += 0.05
                
        # Confidence interval effect
        ci = statistical_indicators['confidence_interval']
        if ci > 0:
            if ci >= 0.95:
                score += 0.15
            elif ci >= 0.9:
                score += 0.1
            else:
                score += 0.05
                
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
