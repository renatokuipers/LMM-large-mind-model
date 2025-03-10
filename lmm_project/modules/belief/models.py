# models.py
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Any, Optional, Union, Literal, Tuple
from datetime import datetime
import uuid
import numpy as np
import random

class Evidence(BaseModel):
    """Evidence supporting or contradicting a belief"""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    content: Dict[str, Any]
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class Belief(BaseModel):
    """Representation of a belief with confidence and supporting evidence"""
    belief_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    creation_time: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    stability: float = Field(default=0.2, ge=0.0, le=1.0)  # How resistant to change
    evidence_for: List[Evidence] = Field(default_factory=list)
    evidence_against: List[Evidence] = Field(default_factory=list)
    related_beliefs: List[str] = Field(default_factory=list)  # IDs of related beliefs
    source: str = "reasoning"  # Where this belief came from
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @model_validator(mode='after')
    def update_confidence_from_evidence(self):
        """Update confidence based on supporting and contradicting evidence"""
        if not self.evidence_for and not self.evidence_against:
            return self
            
        # Implement Bayesian confidence calculation
        # We'll use a simplified Bayesian update based on evidence reliability and relevance
        
        # Start with prior confidence
        prior = self.confidence
        
        # Calculate likelihood ratios for evidence
        likelihood_for = 1.0
        for evidence in self.evidence_for:
            # Weight by reliability and relevance
            weighted_support = evidence.reliability * evidence.relevance
            # Convert to likelihood ratio (values > 1.0 increase confidence)
            evidence_factor = 1.0 + weighted_support
            likelihood_for *= evidence_factor
            
        likelihood_against = 1.0
        for evidence in self.evidence_against:
            # Weight by reliability and relevance
            weighted_contradiction = evidence.reliability * evidence.relevance
            # Convert to likelihood ratio (values > 1.0 increase confidence)
            evidence_factor = 1.0 + weighted_contradiction
            likelihood_against *= evidence_factor
        
        # Apply Bayes' rule
        # posterior_odds = prior_odds * likelihood_ratio
        prior_odds = prior / (1.0 - prior) if prior < 1.0 else 100.0  # Prevent division by zero
        likelihood_ratio = likelihood_for / likelihood_against if likelihood_against > 0 else likelihood_for
        posterior_odds = prior_odds * likelihood_ratio
        
        # Convert back to probability
        posterior = posterior_odds / (1.0 + posterior_odds)
        
        # Apply stability as a damping factor on changes
        confidence_change = posterior - prior
        damped_change = confidence_change * (1.0 - self.stability)
        
        # Update confidence with damped change
        self.confidence = max(0.0, min(1.0, prior + damped_change))
        self.last_updated = datetime.now()
            
        return self

class BeliefSystem(BaseModel):
    """Collection of interrelated beliefs"""
    beliefs: Dict[str, Belief] = Field(default_factory=dict)
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    belief_network: Dict[str, List[str]] = Field(default_factory=dict)  # Graph of belief relations
    developmental_level: float = Field(default=0.0, ge=0.0, le=1.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def add_belief(self, belief: Belief) -> str:
        """
        Add a new belief to the system
        
        Args:
            belief: The belief to add
            
        Returns:
            ID of the added belief
        """
        self.beliefs[belief.belief_id] = belief
        
        # Initialize in network graph if new
        if belief.belief_id not in self.belief_network:
            self.belief_network[belief.belief_id] = []
            
        # Update relationships
        self._update_belief_relationships(belief)
        
        # Recalculate consistency
        self._calculate_consistency()
        
        return belief.belief_id
    
    def update_belief(self, belief_id: str, updated_belief: Belief) -> bool:
        """
        Update an existing belief
        
        Args:
            belief_id: ID of belief to update
            updated_belief: New belief data
            
        Returns:
            True if updated, False if belief not found
        """
        if belief_id not in self.beliefs:
            return False
            
        # Preserve the original ID
        updated_belief.belief_id = belief_id
        self.beliefs[belief_id] = updated_belief
        
        # Update relationships
        self._update_belief_relationships(updated_belief)
        
        # Recalculate consistency
        self._calculate_consistency()
        
        return True
    
    def remove_belief(self, belief_id: str) -> bool:
        """
        Remove a belief from the system
        
        Args:
            belief_id: ID of belief to remove
            
        Returns:
            True if removed, False if not found
        """
        if belief_id not in self.beliefs:
            return False
            
        # Remove from beliefs dictionary
        del self.beliefs[belief_id]
        
        # Remove from network
        if belief_id in self.belief_network:
            del self.belief_network[belief_id]
            
        # Remove references from other beliefs' relationships
        for other_id in self.belief_network:
            if belief_id in self.belief_network[other_id]:
                self.belief_network[other_id].remove(belief_id)
                
        # Update related_beliefs lists in all beliefs
        for other_belief in self.beliefs.values():
            if belief_id in other_belief.related_beliefs:
                other_belief.related_beliefs.remove(belief_id)
                
        # Recalculate consistency
        self._calculate_consistency()
        
        return True
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """
        Get a belief by ID
        
        Args:
            belief_id: ID of the belief to retrieve
            
        Returns:
            The belief if found, None otherwise
        """
        return self.beliefs.get(belief_id)
    
    def find_related_beliefs(self, belief_id: str, max_depth: int = 1) -> Dict[str, float]:
        """
        Find beliefs related to the given belief
        
        Args:
            belief_id: ID of the belief to find relations for
            max_depth: Maximum network traversal depth
            
        Returns:
            Dictionary mapping related belief IDs to relatedness scores
        """
        if belief_id not in self.beliefs:
            return {}
            
        related = {}
        visited = set()
        
        # Helper function for recursive traversal
        def traverse(current_id: str, depth: int, path_strength: float = 1.0):
            if depth > max_depth or current_id in visited:
                return
                
            visited.add(current_id)
            
            # Get direct neighbors
            for neighbor_id in self.belief_network.get(current_id, []):
                # Calculate relatedness as product of path strengths
                relationship_strength = self._calculate_relationship_strength(current_id, neighbor_id)
                new_strength = path_strength * relationship_strength
                
                # Store or update relatedness score
                if neighbor_id not in related or new_strength > related[neighbor_id]:
                    related[neighbor_id] = new_strength
                    
                # Recurse to next level
                traverse(neighbor_id, depth + 1, new_strength)
                
        # Start traversal
        traverse(belief_id, 0)
        
        # Remove the starting belief itself
        if belief_id in related:
            del related[belief_id]
            
        return related
    
    def find_contradictions(self, belief_id: str = None) -> List[Tuple[str, str, float]]:
        """
        Find contradictory beliefs in the system
        
        Args:
            belief_id: Optional ID to check contradictions for a specific belief
            
        Returns:
            List of tuples (belief_id1, belief_id2, contradiction_strength)
        """
        contradictions = []
        
        # If specific belief provided, only check that one
        if belief_id:
            if belief_id not in self.beliefs:
                return []
                
            beliefs_to_check = [belief_id]
        else:
            # Otherwise check all beliefs
            beliefs_to_check = list(self.beliefs.keys())
            
        # Check each belief against others
        for i, id1 in enumerate(beliefs_to_check):
            # Only compare with beliefs we haven't checked yet
            other_beliefs = beliefs_to_check[i+1:] if belief_id is None else list(self.beliefs.keys())
            
            for id2 in other_beliefs:
                if id1 == id2:
                    continue
                    
                contradiction_strength = self._calculate_contradiction(id1, id2)
                if contradiction_strength > 0.2:  # Only report significant contradictions
                    contradictions.append((id1, id2, contradiction_strength))
                    
        return contradictions
    
    def _update_belief_relationships(self, belief: Belief) -> None:
        """Update relationships between this belief and others"""
        # Clear existing relationships for this belief
        belief.related_beliefs = []
        
        # Find content-based relationships with other beliefs
        for other_id, other_belief in self.beliefs.items():
            if other_id == belief.belief_id:
                continue
                
            # Calculate relationship strength
            relatedness = self._calculate_content_similarity(belief, other_belief)
            
            # If sufficiently related, add to relationship network
            if relatedness > 0.3:  # Threshold for relationship
                # Update belief network (bidirectional)
                if other_id not in self.belief_network[belief.belief_id]:
                    self.belief_network[belief.belief_id].append(other_id)
                
                if belief.belief_id not in self.belief_network.get(other_id, []):
                    if other_id not in self.belief_network:
                        self.belief_network[other_id] = []
                    self.belief_network[other_id].append(belief.belief_id)
                
                # Update related_beliefs lists
                if other_id not in belief.related_beliefs:
                    belief.related_beliefs.append(other_id)
                
                if belief.belief_id not in other_belief.related_beliefs:
                    other_belief.related_beliefs.append(belief.belief_id)
    
    def _calculate_content_similarity(self, belief1: Belief, belief2: Belief) -> float:
        """Calculate similarity between belief contents"""
        # Simple content overlap calculation
        # In a real implementation, this could use semantic similarity
        
        # Collect keys from both beliefs
        all_keys = set(belief1.content.keys()) | set(belief2.content.keys())
        if not all_keys:
            return 0.0
            
        # Count matching values
        matching = 0
        for key in all_keys:
            if key in belief1.content and key in belief2.content:
                if belief1.content[key] == belief2.content[key]:
                    matching += 1
                    
        # Calculate similarity score
        return matching / len(all_keys)
    
    def _calculate_relationship_strength(self, belief_id1: str, belief_id2: str) -> float:
        """Calculate relationship strength between two beliefs"""
        belief1 = self.beliefs.get(belief_id1)
        belief2 = self.beliefs.get(belief_id2)
        
        if not belief1 or not belief2:
            return 0.0
            
        # Content similarity
        content_similarity = self._calculate_content_similarity(belief1, belief2)
        
        # Evidence overlap
        evidence_overlap = self._calculate_evidence_overlap(belief1, belief2)
        
        # Combine factors
        return 0.7 * content_similarity + 0.3 * evidence_overlap
    
    def _calculate_evidence_overlap(self, belief1: Belief, belief2: Belief) -> float:
        """Calculate overlap in evidence between beliefs"""
        # Extract evidence IDs
        evidence1_ids = {e.evidence_id for e in belief1.evidence_for + belief1.evidence_against}
        evidence2_ids = {e.evidence_id for e in belief2.evidence_for + belief2.evidence_against}
        
        # Calculate overlap
        if not evidence1_ids or not evidence2_ids:
            return 0.0
            
        intersection = evidence1_ids.intersection(evidence2_ids)
        union = evidence1_ids.union(evidence2_ids)
        
        return len(intersection) / len(union)
    
    def _calculate_contradiction(self, belief_id1: str, belief_id2: str) -> float:
        """Calculate contradiction strength between two beliefs"""
        belief1 = self.beliefs.get(belief_id1)
        belief2 = self.beliefs.get(belief_id2)
        
        if not belief1 or not belief2:
            return 0.0
            
        # Simple contradiction detection based on content
        contradictions = 0
        total_comparisons = 0
        
        # Compare shared keys for direct contradictions
        for key in set(belief1.content.keys()) & set(belief2.content.keys()):
            if belief1.content[key] != belief2.content[key]:
                contradictions += 1
            total_comparisons += 1
                
        # If no shared keys, check for evidence that directly opposes
        if total_comparisons == 0:
            # Evidence for belief1 that's against belief2
            for evidence in belief1.evidence_for:
                if any(e.evidence_id == evidence.evidence_id for e in belief2.evidence_against):
                    contradictions += 1
                    total_comparisons += 1
                    
            # Evidence for belief2 that's against belief1
            for evidence in belief2.evidence_for:
                if any(e.evidence_id == evidence.evidence_id for e in belief1.evidence_against):
                    contradictions += 1
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
            
        return contradictions / total_comparisons
    
    def _calculate_consistency(self) -> None:
        """Calculate overall consistency score for the belief system"""
        if len(self.beliefs) <= 1:
            self.consistency_score = 1.0
            return
            
        # Find all contradictions
        contradictions = self.find_contradictions()
        
        # If no contradictions, perfect consistency
        if not contradictions:
            self.consistency_score = 1.0
            return
            
        # Calculate weighted contradiction score
        total_possible_pairs = len(self.beliefs) * (len(self.beliefs) - 1) / 2
        contradiction_sum = sum(strength for _, _, strength in contradictions)
        
        # Scale by number of beliefs and max possible contradictions
        raw_consistency = 1.0 - (contradiction_sum / total_possible_pairs)
        
        # Ensure in valid range
        self.consistency_score = max(0.0, min(1.0, raw_consistency))