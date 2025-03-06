"""
Cognitive Component

This module implements the cognitive capabilities of the child's mind,
including reasoning, problem-solving, decision-making, and learning.
It models how cognitive abilities develop over time.
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


class CognitiveComponent:
    """
    The CognitiveComponent handles cognitive development and processing.
    
    It models how a child develops reasoning, problem-solving, decision-making,
    and learning abilities over time.
    """
    
    def __init__(self):
        """Initialize the cognitive component."""
        # Core cognitive abilities
        self.logical_reasoning = 0.1
        self.causal_reasoning = 0.1
        self.problem_recognition = 0.1
        self.decision_making = 0.1
        
        # Piaget's cognitive development stages - for test compatibility
        self.object_permanence = 0.1  # Sensorimotor stage
        self.conservation = 0.0       # Concrete operational stage
        self.abstract_thinking = 0.0  # Formal operational stage
        self.problem_solving = 0.1    # General ability
        
        # Attention and memory integration
        self.attention_span = 0.1
        self.working_memory_capacity = 0.1
        
        # Reasoning development
        self.analogical_reasoning = 0.0  # Ability to see patterns and analogies
        
        # Problem-solving development
        self.solution_generation = 0.1  # Ability to generate solutions
        self.planning_ability = 0.0  # Ability to plan ahead
        
        # Decision-making development
        self.decision_speed = 0.3  # Speed of decision-making (starts fast but impulsive)
        self.decision_quality = 0.1  # Quality of decisions
        self.risk_assessment = 0.0  # Ability to assess risks
        
        # Learning development
        self.learning_speed = 0.5  # Speed of learning (high in early childhood)
        self.knowledge_integration = 0.1  # Ability to integrate new knowledge
        self.cognitive_flexibility = 0.1  # Ability to adapt thinking
        
        # Cognitive biases (start high, decrease with development)
        self.confirmation_bias = 0.8  # Tendency to confirm existing beliefs
        self.recency_bias = 0.8  # Overweighting recent information
        self.availability_bias = 0.8  # Overweighting easily recalled information
        
        # Cognitive history
        self.problem_history: List[Dict[str, Any]] = []  # History of problems encountered
        self.decision_history: List[Dict[str, Any]] = []  # History of decisions made
        self.max_history_size = 20
        
        logger.info("Cognitive component initialized")
    
    def update_cognitive_development(self, developmental_stage: DevelopmentalStage) -> Dict[str, float]:
        """
        Update cognitive development metrics based on developmental stage.
        
        Args:
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary of updated cognitive development metrics
        """
        # Cognitive development by stage
        if developmental_stage == DevelopmentalStage.INFANCY:
            # Infants develop basic cognitive abilities
            self._update_reasoning(0.2, 0.2, 0.0)
            self._update_problem_solving(0.2, 0.1, 0.0)
            self._update_decision_making(0.3, 0.1, 0.0)
            self._update_learning(0.5, 0.2, 0.1)
            self._update_biases(0.8, 0.8, 0.8)
            
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Early childhood: rapid cognitive development
            self._update_reasoning(0.4, 0.4, 0.2)
            self._update_problem_solving(0.4, 0.3, 0.2)
            self._update_decision_making(0.4, 0.3, 0.2)
            self._update_learning(0.7, 0.4, 0.3)
            self._update_biases(0.7, 0.7, 0.7)
            
        elif developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            # Middle childhood: more complex cognitive abilities
            self._update_reasoning(0.6, 0.6, 0.4)
            self._update_problem_solving(0.6, 0.5, 0.4)
            self._update_decision_making(0.5, 0.5, 0.4)
            self._update_learning(0.6, 0.6, 0.5)
            self._update_biases(0.6, 0.6, 0.6)
            
        elif developmental_stage == DevelopmentalStage.ADOLESCENCE:
            # Adolescence: advanced cognitive abilities
            self._update_reasoning(0.8, 0.8, 0.7)
            self._update_problem_solving(0.8, 0.7, 0.7)
            self._update_decision_making(0.7, 0.7, 0.6)
            self._update_learning(0.5, 0.8, 0.7)  # Learning speed decreases but integration improves
            self._update_biases(0.4, 0.4, 0.4)
            
        elif developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
            # Early adulthood: mature cognitive abilities
            self._update_reasoning(0.9, 0.9, 0.9)
            self._update_problem_solving(0.9, 0.9, 0.8)
            self._update_decision_making(0.8, 0.9, 0.8)
            self._update_learning(0.4, 0.9, 0.9)  # Learning speed decreases but integration is high
            self._update_biases(0.3, 0.3, 0.3)
        
        # Return current metrics
        return self.get_cognitive_metrics()
    
    def _update_reasoning(
        self, 
        max_logical: float, 
        max_causal: float,
        max_analogical: float
    ):
        """
        Update reasoning metrics with gradual improvement.
        
        Args:
            max_logical: Maximum logical reasoning for current stage
            max_causal: Maximum causal reasoning for current stage
            max_analogical: Maximum analogical reasoning for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.logical_reasoning = min(
            max_logical,
            self.logical_reasoning + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.causal_reasoning = min(
            max_causal,
            self.causal_reasoning + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.analogical_reasoning = min(
            max_analogical,
            self.analogical_reasoning + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_problem_solving(
        self,
        max_recognition: float,
        max_generation: float,
        max_planning: float
    ):
        """
        Update problem-solving metrics with gradual improvement.
        
        Args:
            max_recognition: Maximum problem recognition for current stage
            max_generation: Maximum solution generation for current stage
            max_planning: Maximum planning ability for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.problem_recognition = min(
            max_recognition,
            self.problem_recognition + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.solution_generation = min(
            max_generation,
            self.solution_generation + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.planning_ability = min(
            max_planning,
            self.planning_ability + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_decision_making(
        self,
        max_speed: float,
        max_quality: float,
        max_risk: float
    ):
        """
        Update decision-making metrics with gradual improvement.
        
        Args:
            max_speed: Maximum decision speed for current stage
            max_quality: Maximum decision quality for current stage
            max_risk: Maximum risk assessment for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.decision_speed = min(
            max_speed,
            self.decision_speed + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.decision_quality = min(
            max_quality,
            self.decision_quality + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.risk_assessment = min(
            max_risk,
            self.risk_assessment + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_learning(
        self,
        max_speed: float,
        max_integration: float,
        max_flexibility: float
    ):
        """
        Update learning metrics with gradual improvement.
        
        Args:
            max_speed: Maximum learning speed for current stage
            max_integration: Maximum knowledge integration for current stage
            max_flexibility: Maximum cognitive flexibility for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.learning_speed = min(
            max_speed,
            self.learning_speed + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.knowledge_integration = min(
            max_integration,
            self.knowledge_integration + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.cognitive_flexibility = min(
            max_flexibility,
            self.cognitive_flexibility + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_biases(
        self,
        max_confirmation: float,
        max_recency: float,
        max_availability: float
    ):
        """
        Update cognitive bias metrics with gradual improvement.
        
        Args:
            max_confirmation: Maximum confirmation bias for current stage
            max_recency: Maximum recency bias for current stage
            max_availability: Maximum availability bias for current stage
        """
        # Gradual improvement toward stage maximum (biases decrease with development)
        improvement_rate = 0.01  # Small incremental improvements
        
        # For biases, we want them to decrease, so we approach the maximum from above
        if self.confirmation_bias > max_confirmation:
            self.confirmation_bias = max(
                max_confirmation,
                self.confirmation_bias - improvement_rate * random.uniform(0.8, 1.2)
            )
        
        if self.recency_bias > max_recency:
            self.recency_bias = max(
                max_recency,
                self.recency_bias - improvement_rate * random.uniform(0.8, 1.2)
            )
        
        if self.availability_bias > max_availability:
            self.availability_bias = max(
                max_availability,
                self.availability_bias - improvement_rate * random.uniform(0.8, 1.2)
            )
    
    def process_cognitive_input(
        self, 
        input_text: str, 
        emotional_state: List[Emotion],
        developmental_stage: DevelopmentalStage
    ) -> Dict[str, Any]:
        """
        Process cognitive input to update cognitive development.
        
        Args:
            input_text: Text to process
            emotional_state: Current emotional state
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary with cognitive processing results
        """
        # Update cognitive development based on stage
        self.update_cognitive_development(developmental_stage)
        
        # Process reasoning elements in text
        reasoning_elements = self._identify_reasoning_elements(input_text, developmental_stage)
        
        # Process problem-solving elements in text
        problem_elements = self._identify_problem_elements(input_text, developmental_stage)
        
        # Process decision-making elements in text
        decision_elements = self._identify_decision_elements(input_text, developmental_stage)
        
        # Process learning elements in text
        learning_elements = self._identify_learning_elements(input_text, developmental_stage)
        
        # Small boost to metrics based on interaction content
        if reasoning_elements["logical_elements"] > 0:
            self.logical_reasoning = min(1.0, self.logical_reasoning + 0.005 * reasoning_elements["logical_elements"])
            
        if problem_elements["problem_statements"] > 0:
            self.problem_recognition = min(1.0, self.problem_recognition + 0.005 * problem_elements["problem_statements"])
            
        if decision_elements["decision_points"] > 0:
            self.decision_quality = min(1.0, self.decision_quality + 0.005 * decision_elements["decision_points"])
            
        if learning_elements["new_concepts"] > 0:
            self.knowledge_integration = min(1.0, self.knowledge_integration + 0.005 * learning_elements["new_concepts"])
        
        # Record problem in history if identified
        if problem_elements["problem_statements"] > 0:
            problem_record = {
                "problem_elements": problem_elements,
                "developmental_stage": developmental_stage.value,
                "emotional_context": [e.type.value for e in emotional_state]
            }
            
            self.problem_history.append(problem_record)
            
            # Limit history size
            if len(self.problem_history) > self.max_history_size:
                self.problem_history.pop(0)
        
        # Record decision in history if identified
        if decision_elements["decision_points"] > 0:
            decision_record = {
                "decision_elements": decision_elements,
                "developmental_stage": developmental_stage.value,
                "emotional_context": [e.type.value for e in emotional_state]
            }
            
            self.decision_history.append(decision_record)
            
            # Limit history size
            if len(self.decision_history) > self.max_history_size:
                self.decision_history.pop(0)
        
        # Return processing results
        return {
            "reasoning_elements": reasoning_elements,
            "problem_elements": problem_elements,
            "decision_elements": decision_elements,
            "learning_elements": learning_elements,
            "logical_reasoning": self.logical_reasoning,
            "problem_recognition": self.problem_recognition,
            "decision_quality": self.decision_quality,
            "knowledge_integration": self.knowledge_integration
        }
    
    def _identify_reasoning_elements(self, text: str, stage: DevelopmentalStage) -> Dict[str, int]:
        """
        Identify reasoning elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Dictionary of reasoning elements and their counts
        """
        # Initialize counts
        reasoning_elements = {
            "logical_elements": 0,  # Logical reasoning elements
            "causal_elements": 0,  # Causal reasoning elements
            "analogical_elements": 0,  # Analogical reasoning elements
        }
        
        # Only process if stage is appropriate
        if stage == DevelopmentalStage.INFANCY:
            return reasoning_elements  # Infants don't process reasoning elements
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for logical reasoning elements
        logical_indicators = [
            "if", "then", "because", "so", "therefore",
            "must be", "can't be", "has to be", "always", "never",
            "all", "none", "some", "most", "few"
        ]
        reasoning_elements["logical_elements"] = sum(
            1 for indicator in logical_indicators if indicator in text_lower
        )
        
        # Check for causal reasoning elements
        causal_indicators = [
            "because", "since", "as a result", "due to", "caused by",
            "leads to", "results in", "effect of", "impact of", "influence"
        ]
        reasoning_elements["causal_elements"] = sum(
            1 for indicator in causal_indicators if indicator in text_lower
        )
        
        # Check for analogical reasoning elements (more advanced)
        if stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
            analogical_indicators = [
                "like", "similar to", "just as", "compared to", "resembles",
                "same as", "different from", "pattern", "analogy", "metaphor"
            ]
            reasoning_elements["analogical_elements"] = sum(
                1 for indicator in analogical_indicators if indicator in text_lower
            )
        
        return reasoning_elements
    
    def _identify_problem_elements(self, text: str, stage: DevelopmentalStage) -> Dict[str, int]:
        """
        Identify problem-solving elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Dictionary of problem-solving elements and their counts
        """
        # Initialize counts
        problem_elements = {
            "problem_statements": 0,  # Problem statements
            "solution_proposals": 0,  # Solution proposals
            "planning_elements": 0,  # Planning elements
        }
        
        # Only process if stage is appropriate
        if stage == DevelopmentalStage.INFANCY:
            return problem_elements  # Infants don't process problem elements
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for problem statements
        problem_indicators = [
            "problem", "issue", "trouble", "difficult", "challenge",
            "can't", "won't", "doesn't work", "broken", "wrong",
            "how do i", "how can i", "what should i do"
        ]
        problem_elements["problem_statements"] = sum(
            1 for indicator in problem_indicators if indicator in text_lower
        )
        
        # Check for solution proposals
        solution_indicators = [
            "solution", "solve", "fix", "resolve", "answer",
            "try", "attempt", "maybe", "perhaps", "could",
            "let's", "we should", "i should", "would work"
        ]
        problem_elements["solution_proposals"] = sum(
            1 for indicator in solution_indicators if indicator in text_lower
        )
        
        # Check for planning elements (more advanced)
        if stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
            planning_indicators = [
                "plan", "step", "first", "then", "next",
                "after", "before", "finally", "process", "procedure",
                "strategy", "approach", "method", "way to"
            ]
            problem_elements["planning_elements"] = sum(
                1 for indicator in planning_indicators if indicator in text_lower
            )
        
        return problem_elements
    
    def _identify_decision_elements(self, text: str, stage: DevelopmentalStage) -> Dict[str, int]:
        """
        Identify decision-making elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Dictionary of decision-making elements and their counts
        """
        # Initialize counts
        decision_elements = {
            "decision_points": 0,  # Decision points
            "choice_considerations": 0,  # Choice considerations
            "risk_assessments": 0,  # Risk assessments
        }
        
        # Only process if stage is appropriate
        if stage == DevelopmentalStage.INFANCY:
            return decision_elements  # Infants don't process decision elements
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for decision points
        decision_indicators = [
            "decide", "decision", "choose", "choice", "select",
            "option", "alternative", "pick", "prefer", "rather",
            "should i", "would be better", "go with"
        ]
        decision_elements["decision_points"] = sum(
            1 for indicator in decision_indicators if indicator in text_lower
        )
        
        # Check for choice considerations
        choice_indicators = [
            "consider", "think about", "weigh", "compare", "contrast",
            "pros and cons", "advantages", "disadvantages", "benefits", "drawbacks",
            "better", "worse", "best", "worst"
        ]
        decision_elements["choice_considerations"] = sum(
            1 for indicator in choice_indicators if indicator in text_lower
        )
        
        # Check for risk assessments (more advanced)
        if stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
            risk_indicators = [
                "risk", "danger", "safe", "unsafe", "careful",
                "caution", "warning", "might", "could", "possibly",
                "chance", "likelihood", "probability", "uncertain"
            ]
            decision_elements["risk_assessments"] = sum(
                1 for indicator in risk_indicators if indicator in text_lower
            )
        
        return decision_elements
    
    def _identify_learning_elements(self, text: str, stage: DevelopmentalStage) -> Dict[str, int]:
        """
        Identify learning elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Dictionary of learning elements and their counts
        """
        # Initialize counts
        learning_elements = {
            "new_concepts": 0,  # New concepts
            "knowledge_connections": 0,  # Knowledge connections
            "cognitive_adaptations": 0,  # Cognitive adaptations
        }
        
        # Only process if stage is appropriate
        if stage == DevelopmentalStage.INFANCY:
            # For infants, any noun could be a new concept
            words = text.lower().split()
            learning_elements["new_concepts"] = min(3, len(words))  # Cap at 3
            return learning_elements
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Check for new concepts
        concept_indicators = [
            "new", "learn", "understand", "concept", "idea",
            "meaning", "definition", "what is", "called", "known as",
            "refers to", "defined as", "explanation"
        ]
        learning_elements["new_concepts"] = sum(
            1 for indicator in concept_indicators if indicator in text_lower
        )
        
        # Check for knowledge connections
        connection_indicators = [
            "related to", "connected to", "associated with", "linked to", "part of",
            "similar to", "different from", "same as", "like", "unlike",
            "remember", "recall", "reminds me", "familiar"
        ]
        learning_elements["knowledge_connections"] = sum(
            1 for indicator in connection_indicators if indicator in text_lower
        )
        
        # Check for cognitive adaptations (more advanced)
        if stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
            adaptation_indicators = [
                "change mind", "rethink", "reconsider", "new perspective", "different view",
                "instead", "alternatively", "on second thought", "actually", "realized",
                "now i see", "understand better", "changed my thinking"
            ]
            learning_elements["cognitive_adaptations"] = sum(
                1 for indicator in adaptation_indicators if indicator in text_lower
            )
        
        return learning_elements
    
    def get_cognitive_metrics(self) -> Dict[str, float]:
        """
        Get the current cognitive development metrics.
        
        Returns:
            Dictionary of cognitive metrics
        """
        return {
            "logical_reasoning": self.logical_reasoning,
            "causal_reasoning": self.causal_reasoning,
            "analogical_reasoning": self.analogical_reasoning,
            "problem_recognition": self.problem_recognition,
            "solution_generation": self.solution_generation,
            "planning_ability": self.planning_ability,
            "decision_speed": self.decision_speed,
            "decision_quality": self.decision_quality,
            "risk_assessment": self.risk_assessment,
            "learning_speed": self.learning_speed,
            "knowledge_integration": self.knowledge_integration,
            "cognitive_flexibility": self.cognitive_flexibility,
            "confirmation_bias": self.confirmation_bias,
            "recency_bias": self.recency_bias,
            "availability_bias": self.availability_bias,
            "overall_cognitive_development": self._calculate_overall_cognitive_development()
        }
    
    def get_object_permanence(self) -> float:
        """Return the current object permanence level for test compatibility."""
        return self.object_permanence
    
    def get_abstract_thinking(self) -> float:
        """Return the current abstract thinking level for test compatibility."""
        return self.abstract_thinking
    
    def _calculate_overall_cognitive_development(self) -> float:
        """
        Calculate overall cognitive development score.
        
        Returns:
            Overall cognitive development score (0-1)
        """
        # Average of all cognitive metrics (excluding biases)
        metrics = [
            self.logical_reasoning,
            self.causal_reasoning,
            self.analogical_reasoning,
            self.problem_recognition,
            self.solution_generation,
            self.planning_ability,
            self.decision_speed,
            self.decision_quality,
            self.risk_assessment,
            self.learning_speed,
            self.knowledge_integration,
            self.cognitive_flexibility,
            # Biases are inverted (lower is better)
            1 - self.confirmation_bias,
            1 - self.recency_bias,
            1 - self.availability_bias
        ]
        
        return sum(metrics) / len(metrics)
    
    def get_problem_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent problem history.
        
        Args:
            limit: Maximum number of problems to return
            
        Returns:
            List of recent problems
        """
        return self.problem_history[-limit:] if self.problem_history else []
    
    def get_decision_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent decision history.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent decisions
        """
        return self.decision_history[-limit:] if self.decision_history else [] 