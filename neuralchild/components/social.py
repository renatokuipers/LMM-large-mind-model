"""
Social Component

This module implements the social awareness and interaction capabilities
of the child's mind. It models how social understanding, relationships,
and social cognition develop over time.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np

from ..utils.data_types import (
    DevelopmentalStage, Emotion, EmotionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialComponent:
    """
    The SocialComponent handles social awareness and interaction development.
    
    It models how a child develops understanding of social norms, relationships,
    and social cognition abilities.
    """
    
    def __init__(self):
        """Initialize the social component."""
        # Core social abilities
        self.social_recognition = 0.1
        self.social_reciprocity = 0.1
        self.social_communication = 0.1
        
        # Test compatibility attributes
        self.social_awareness = 0.1
        self.peer_relationships = 0.0
        self.attachment_security = 0.3
        
        # Social development tracking
        self.joint_attention = 0.1
        
        # Social awareness development
        self.relationship_understanding = 0.0  # Understanding of relationships
        self.social_norm_comprehension = 0.0  # Understanding of social norms
        
        # Social skills development
        self.communication_skills = 0.1  # Ability to communicate effectively
        self.cooperation_skills = 0.0  # Ability to cooperate with others
        self.conflict_resolution = 0.0  # Ability to resolve conflicts
        
        # Social cognition development
        self.empathy = 0.1  # Ability to understand others' emotions
        self.social_perspective_taking = 0.0  # Taking others' perspectives
        self.social_problem_solving = 0.0  # Solving social problems
        
        # Social relationships
        self.social_confidence = 0.1  # Confidence in social situations
        
        # Social interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        self.max_history_size = 20
        
        # Known social norms (develops over time)
        self.social_norms: Dict[str, float] = {
            # Format: "norm description": understanding_level
            "greeting others": 0.3,
            "saying please and thank you": 0.2,
            "taking turns": 0.1,
            "sharing toys": 0.1,
            "not interrupting": 0.0,
            "respecting personal space": 0.0,
            "using indoor voice": 0.1,
            "waiting patiently": 0.0,
            "apologizing when wrong": 0.0,
            "helping others": 0.1,
        }
        
        logger.info("Social component initialized")
    
    def update_social_development_for_tests(self, developmental_stage: DevelopmentalStage, substage: str = None) -> Dict[str, float]:
        """
        Update social development metrics based on the developmental stage.
        Test compatibility method.
        
        Args:
            developmental_stage: The current developmental stage
            substage: The specific developmental substage (optional)
            
        Returns:
            Dict of updated social metrics
        """
        # Use existing method to update metrics
        self.update_social_metrics(developmental_stage)
        
        # Return current metrics
        return {
            "social_awareness": self.social_awareness,
            "relationship_understanding": self.relationship_understanding,
            "empathy": self.empathy,
        }
    
    def get_social_awareness(self) -> float:
        """Return the current social awareness level for test compatibility."""
        return self.social_awareness
        
    def get_social_metrics(self) -> Dict[str, float]:
        """
        Get the current social development metrics.
        
        Returns:
            Dictionary of social metrics
        """
        return {
            "social_awareness": self.social_awareness,
            "relationship_understanding": self.relationship_understanding,
            "social_norm_comprehension": self.social_norm_comprehension,
            "communication_skills": self.communication_skills,
            "cooperation_skills": self.cooperation_skills,
            "conflict_resolution": self.conflict_resolution,
            "empathy": self.empathy,
            "social_perspective_taking": self.social_perspective_taking,
            "social_problem_solving": self.social_problem_solving,
            "attachment_security": self.attachment_security,
            "peer_relationships": self.peer_relationships,
            "social_confidence": self.social_confidence,
            "overall_social_development": self._calculate_overall_social_development()
        }
    
    def _calculate_overall_social_development(self) -> float:
        """
        Calculate overall social development score.
        
        Returns:
            Overall social development score (0-1)
        """
        # Average of all social metrics
        metrics = [
            self.social_awareness,
            self.relationship_understanding,
            self.social_norm_comprehension,
            self.communication_skills,
            self.cooperation_skills,
            self.conflict_resolution,
            self.empathy,
            self.social_perspective_taking,
            self.social_problem_solving,
            self.attachment_security,
            self.peer_relationships,
            self.social_confidence
        ]
        
        return sum(metrics) / len(metrics)
    
    def update_social_development(self, developmental_stage: DevelopmentalStage) -> Dict[str, float]:
        """
        Update social development metrics based on developmental stage.
        
        Args:
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary of updated social development metrics
        """
        # Social development by stage
        if developmental_stage == DevelopmentalStage.INFANCY:
            # Infants develop basic social awareness and attachment
            self._update_social_awareness(0.3, 0.1, 0.0)
            self._update_social_skills(0.2, 0.1, 0.0)
            self._update_social_cognition(0.2, 0.0, 0.0)
            self._update_social_relationships(0.6, 0.0, 0.1)
            
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Early childhood: developing basic social skills and awareness
            self._update_social_awareness(0.5, 0.3, 0.2)
            self._update_social_skills(0.4, 0.3, 0.1)
            self._update_social_cognition(0.4, 0.2, 0.1)
            self._update_social_relationships(0.7, 0.3, 0.3)
            
        elif developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            # Middle childhood: more complex social understanding and skills
            self._update_social_awareness(0.7, 0.5, 0.5)
            self._update_social_skills(0.6, 0.5, 0.4)
            self._update_social_cognition(0.6, 0.5, 0.4)
            self._update_social_relationships(0.8, 0.6, 0.5)
            
        elif developmental_stage == DevelopmentalStage.ADOLESCENCE:
            # Adolescence: advanced social cognition and peer relationships
            self._update_social_awareness(0.8, 0.7, 0.7)
            self._update_social_skills(0.8, 0.7, 0.6)
            self._update_social_cognition(0.8, 0.7, 0.7)
            self._update_social_relationships(0.9, 0.8, 0.7)
            
        elif developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
            # Early adulthood: mature social understanding and skills
            self._update_social_awareness(0.9, 0.9, 0.9)
            self._update_social_skills(0.9, 0.9, 0.8)
            self._update_social_cognition(0.9, 0.9, 0.8)
            self._update_social_relationships(1.0, 0.9, 0.9)
        
        # Update social norms understanding based on stage
        self._update_social_norms(developmental_stage)
        
        # Return current metrics
        return self.get_social_metrics()
    
    def _update_social_awareness(
        self, 
        max_awareness: float, 
        max_relationship: float,
        max_norm: float
    ):
        """
        Update social awareness metrics with gradual improvement.
        
        Args:
            max_awareness: Maximum social awareness for current stage
            max_relationship: Maximum relationship understanding for current stage
            max_norm: Maximum social norm comprehension for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.social_awareness = min(
            max_awareness,
            self.social_awareness + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.relationship_understanding = min(
            max_relationship,
            self.relationship_understanding + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.social_norm_comprehension = min(
            max_norm,
            self.social_norm_comprehension + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_social_skills(
        self,
        max_communication: float,
        max_cooperation: float,
        max_conflict: float
    ):
        """
        Update social skills metrics with gradual improvement.
        
        Args:
            max_communication: Maximum communication skills for current stage
            max_cooperation: Maximum cooperation skills for current stage
            max_conflict: Maximum conflict resolution for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.communication_skills = min(
            max_communication,
            self.communication_skills + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.cooperation_skills = min(
            max_cooperation,
            self.cooperation_skills + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.conflict_resolution = min(
            max_conflict,
            self.conflict_resolution + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_social_cognition(
        self,
        max_empathy: float,
        max_perspective: float,
        max_problem: float
    ):
        """
        Update social cognition metrics with gradual improvement.
        
        Args:
            max_empathy: Maximum empathy for current stage
            max_perspective: Maximum social perspective taking for current stage
            max_problem: Maximum social problem solving for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.empathy = min(
            max_empathy,
            self.empathy + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.social_perspective_taking = min(
            max_perspective,
            self.social_perspective_taking + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.social_problem_solving = min(
            max_problem,
            self.social_problem_solving + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_social_relationships(
        self,
        max_attachment: float,
        max_peer: float,
        max_confidence: float
    ):
        """
        Update social relationship metrics with gradual improvement.
        
        Args:
            max_attachment: Maximum attachment security for current stage
            max_peer: Maximum peer relationships for current stage
            max_confidence: Maximum social confidence for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.attachment_security = min(
            max_attachment,
            self.attachment_security + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.peer_relationships = min(
            max_peer,
            self.peer_relationships + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.social_confidence = min(
            max_confidence,
            self.social_confidence + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_social_norms(self, stage: DevelopmentalStage):
        """
        Update understanding of social norms based on developmental stage.
        
        Args:
            stage: Current developmental stage
        """
        # Define maximum understanding by stage for each norm
        stage_maximums = {
            DevelopmentalStage.INFANCY: {
                "greeting others": 0.2,
                "saying please and thank you": 0.1,
                "taking turns": 0.1,
                "sharing toys": 0.1,
                "not interrupting": 0.0,
                "respecting personal space": 0.0,
                "using indoor voice": 0.1,
                "waiting patiently": 0.0,
                "apologizing when wrong": 0.0,
                "helping others": 0.1,
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                "greeting others": 0.6,
                "saying please and thank you": 0.5,
                "taking turns": 0.4,
                "sharing toys": 0.4,
                "not interrupting": 0.3,
                "respecting personal space": 0.3,
                "using indoor voice": 0.5,
                "waiting patiently": 0.3,
                "apologizing when wrong": 0.3,
                "helping others": 0.4,
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                "greeting others": 0.8,
                "saying please and thank you": 0.8,
                "taking turns": 0.7,
                "sharing toys": 0.7,
                "not interrupting": 0.6,
                "respecting personal space": 0.6,
                "using indoor voice": 0.7,
                "waiting patiently": 0.6,
                "apologizing when wrong": 0.6,
                "helping others": 0.7,
            },
            DevelopmentalStage.ADOLESCENCE: {
                "greeting others": 0.9,
                "saying please and thank you": 0.9,
                "taking turns": 0.9,
                "sharing toys": 0.9,
                "not interrupting": 0.8,
                "respecting personal space": 0.8,
                "using indoor voice": 0.9,
                "waiting patiently": 0.8,
                "apologizing when wrong": 0.8,
                "helping others": 0.9,
            },
            DevelopmentalStage.EARLY_ADULTHOOD: {
                "greeting others": 1.0,
                "saying please and thank you": 1.0,
                "taking turns": 1.0,
                "sharing toys": 1.0,
                "not interrupting": 1.0,
                "respecting personal space": 1.0,
                "using indoor voice": 1.0,
                "waiting patiently": 1.0,
                "apologizing when wrong": 1.0,
                "helping others": 1.0,
            }
        }
        
        # Get maximum values for current stage
        current_maximums = stage_maximums.get(stage, {})
        
        # Update each norm with gradual improvement
        improvement_rate = 0.01  # Small incremental improvements
        
        for norm, current_level in self.social_norms.items():
            max_level = current_maximums.get(norm, 0.0)
            
            # Only update if below maximum for stage
            if current_level < max_level:
                self.social_norms[norm] = min(
                    max_level,
                    current_level + improvement_rate * random.uniform(0.8, 1.2)
                )
    
    def process_social_interaction(
        self, 
        interaction_text: str, 
        emotional_state: List[Emotion],
        developmental_stage: DevelopmentalStage
    ) -> Dict[str, Any]:
        """
        Process a social interaction to update social development.
        
        Args:
            interaction_text: Text of the interaction
            emotional_state: Current emotional state
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary with social processing results
        """
        # Update social development based on stage
        self.update_social_development(developmental_stage)
        
        # Process social elements in text
        social_elements = self._identify_social_elements(interaction_text)
        
        # Process emotional elements related to social interaction
        social_emotions = self._identify_social_emotions(emotional_state)
        
        # Process social norms in interaction
        norms_identified = self._identify_social_norms(interaction_text)
        
        # Small boost to metrics based on interaction content
        if social_elements["prosocial_behaviors"] > 0:
            self.cooperation_skills = min(1.0, self.cooperation_skills + 0.005 * social_elements["prosocial_behaviors"])
            
        if social_elements["social_references"] > 0:
            self.social_awareness = min(1.0, self.social_awareness + 0.005 * social_elements["social_references"])
            
        if social_emotions["empathic_emotions"] > 0:
            self.empathy = min(1.0, self.empathy + 0.005 * social_emotions["empathic_emotions"])
        
        # Boost understanding of identified norms
        for norm in norms_identified:
            if norm in self.social_norms:
                self.social_norms[norm] = min(1.0, self.social_norms[norm] + 0.01)
        
        # Record interaction in history
        interaction_record = {
            "social_elements": social_elements,
            "social_emotions": social_emotions,
            "norms_identified": norms_identified,
            "developmental_stage": developmental_stage.value
        }
        
        self.interaction_history.append(interaction_record)
        
        # Limit history size
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history.pop(0)
        
        # Return processing results
        return {
            "social_elements": social_elements,
            "social_emotions": social_emotions,
            "norms_identified": norms_identified,
            "social_awareness": self.social_awareness,
            "empathy": self.empathy,
            "cooperation_skills": self.cooperation_skills
        }
    
    def _identify_social_elements(self, text: str) -> Dict[str, int]:
        """
        Identify social elements in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of social elements and their counts
        """
        # Initialize counts
        social_elements = {
            "prosocial_behaviors": 0,  # Helping, sharing, cooperating
            "social_references": 0,  # References to other people
            "relationship_terms": 0,  # Terms describing relationships
            "social_rules": 0,  # References to social rules or norms
        }
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for prosocial behaviors
        prosocial_indicators = [
            "help", "share", "cooperate", "together", "please", "thank you",
            "sorry", "excuse me", "give", "support", "assist"
        ]
        social_elements["prosocial_behaviors"] = sum(
            1 for indicator in prosocial_indicators if indicator in text_lower
        )
        
        # Check for social references
        social_reference_indicators = [
            "people", "person", "friend", "friends", "they", "them", "others",
            "everyone", "anybody", "somebody", "group", "team"
        ]
        social_elements["social_references"] = sum(
            1 for indicator in social_reference_indicators if indicator in text_lower
        )
        
        # Check for relationship terms
        relationship_indicators = [
            "friend", "mother", "father", "parent", "brother", "sister", "family",
            "teacher", "classmate", "neighbor", "relationship", "partner"
        ]
        social_elements["relationship_terms"] = sum(
            1 for indicator in relationship_indicators if indicator in text_lower
        )
        
        # Check for social rules
        social_rule_indicators = [
            "should", "must", "rule", "allowed", "not allowed", "supposed to",
            "expected", "polite", "rude", "appropriate", "inappropriate"
        ]
        social_elements["social_rules"] = sum(
            1 for indicator in social_rule_indicators if indicator in text_lower
        )
        
        return social_elements
    
    def _identify_social_emotions(self, emotions: List[Emotion]) -> Dict[str, int]:
        """
        Identify social emotions in emotional state.
        
        Args:
            emotions: List of emotions
            
        Returns:
            Dictionary of social emotion categories and their counts
        """
        # Initialize counts
        social_emotions = {
            "empathic_emotions": 0,  # Emotions related to empathy
            "attachment_emotions": 0,  # Emotions related to attachment
            "social_anxiety": 0,  # Emotions related to social anxiety
            "prosocial_emotions": 0,  # Emotions promoting prosocial behavior
        }
        
        # Categorize emotions
        for emotion in emotions:
            # Empathic emotions
            if emotion.type in [EmotionType.TRUST, EmotionType.LOVE]:
                social_emotions["empathic_emotions"] += 1
                
            # Attachment emotions
            if emotion.type in [EmotionType.LOVE, EmotionType.TRUST]:
                social_emotions["attachment_emotions"] += 1
                
            # Social anxiety
            if emotion.type in [EmotionType.FEAR, EmotionType.SHAME]:
                social_emotions["social_anxiety"] += 1
                
            # Prosocial emotions
            if emotion.type in [EmotionType.JOY, EmotionType.TRUST, EmotionType.LOVE]:
                social_emotions["prosocial_emotions"] += 1
        
        return social_emotions
    
    def _identify_social_norms(self, text: str) -> List[str]:
        """
        Identify social norms referenced in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified social norms
        """
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for each known norm
        identified_norms = []
        
        for norm in self.social_norms.keys():
            # Simple check if norm is mentioned in text
            if norm in text_lower:
                identified_norms.append(norm)
                
            # Check for related terms
            if norm == "greeting others" and any(term in text_lower for term in ["hello", "hi", "greet", "wave"]):
                identified_norms.append(norm)
                
            elif norm == "saying please and thank you" and any(term in text_lower for term in ["please", "thank you", "thanks"]):
                identified_norms.append(norm)
                
            elif norm == "taking turns" and any(term in text_lower for term in ["turn", "wait", "next"]):
                identified_norms.append(norm)
                
            elif norm == "sharing toys" and any(term in text_lower for term in ["share", "sharing", "give", "let you have"]):
                identified_norms.append(norm)
                
            elif norm == "not interrupting" and any(term in text_lower for term in ["interrupt", "wait", "let finish"]):
                identified_norms.append(norm)
        
        # Remove duplicates
        return list(set(identified_norms))
    
    def get_known_social_norms(self, min_understanding: float = 0.3) -> Dict[str, float]:
        """
        Get social norms that the child understands at a minimum level.
        
        Args:
            min_understanding: Minimum understanding level to include
            
        Returns:
            Dictionary of understood social norms and their understanding levels
        """
        return {
            norm: level for norm, level in self.social_norms.items()
            if level >= min_understanding
        }
    
    def get_social_interaction_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent social interaction history.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent social interactions
        """
        return self.interaction_history[-limit:] if self.interaction_history else [] 