"""
Consciousness Component

This module implements the consciousness and self-awareness capabilities
of the child's mind. It models how self-awareness, theory of mind, and
metacognition develop over time.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..utils.data_types import (
    DevelopmentalStage, Emotion, EmotionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsciousnessComponent:
    """
    The ConsciousnessComponent handles the development of self-awareness and theory of mind.
    
    It models how a child develops a sense of self, awareness of others' mental states,
    and metacognitive abilities (thinking about thinking).
    """
    
    def __init__(self):
        """Initialize the consciousness component."""
        # Self-awareness development
        self.self_awareness = 0.1  # Starts low, develops over time
        self.self_concept_complexity = 0.0  # Complexity of self-representation
        self.autobiographical_memory_access = 0.0  # Access to memories about self
        
        # Theory of mind development
        self.theory_of_mind = 0.0  # Understanding others' mental states
        self.perspective_taking = 0.0  # Ability to take others' perspectives
        self.false_belief_understanding = 0.0  # Understanding others can have false beliefs
        
        # Metacognition development
        self.metacognition = 0.0  # Awareness of own cognitive processes
        self.introspection = 0.0  # Ability to examine own thoughts
        self.cognitive_control = 0.0  # Ability to control own thinking
        
        # Self-reflection log
        self.reflections = []  # List of self-reflections
        self.max_reflections = 20  # Maximum number of reflections to store
        
        # Test compatibility attributes
        self.reflective_thinking = self.metacognition
        
        logger.info("Consciousness component initialized")
    
    def update_consciousness(self, developmental_stage: DevelopmentalStage) -> Dict[str, float]:
        """
        Update consciousness metrics based on developmental stage.
        
        Args:
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary of updated consciousness metrics
        """
        # Self-awareness development by stage
        if developmental_stage == DevelopmentalStage.INFANCY:
            # Infants develop basic self-awareness (recognizing self as separate from others)
            self._update_self_awareness(0.3, 0.1, 0.1)
            self._update_theory_of_mind(0.1, 0.1, 0.0)
            self._update_metacognition(0.0, 0.0, 0.1)
            
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Early childhood: developing self-concept, beginning theory of mind
            self._update_self_awareness(0.5, 0.3, 0.3)
            self._update_theory_of_mind(0.3, 0.3, 0.1)
            self._update_metacognition(0.1, 0.1, 0.2)
            
        elif developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            # Middle childhood: more complex self-concept, developing theory of mind
            self._update_self_awareness(0.7, 0.5, 0.5)
            self._update_theory_of_mind(0.6, 0.5, 0.4)
            self._update_metacognition(0.4, 0.3, 0.4)
            
        elif developmental_stage == DevelopmentalStage.ADOLESCENCE:
            # Adolescence: identity formation, advanced theory of mind
            self._update_self_awareness(0.9, 0.8, 0.7)
            self._update_theory_of_mind(0.8, 0.7, 0.7)
            self._update_metacognition(0.7, 0.6, 0.6)
            
        elif developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
            # Early adulthood: mature self-awareness and theory of mind
            self._update_self_awareness(1.0, 0.9, 0.9)
            self._update_theory_of_mind(1.0, 0.9, 0.9)
            self._update_metacognition(0.9, 0.8, 0.8)
        
        # Generate a self-reflection based on current development
        self._generate_self_reflection(developmental_stage)
        
        # Return current metrics
        return self.get_consciousness_metrics()
    
    def _update_self_awareness(
        self, 
        max_self_awareness: float, 
        max_self_concept: float,
        max_autobiographical: float
    ):
        """
        Update self-awareness metrics with gradual improvement.
        
        Args:
            max_self_awareness: Maximum self-awareness for current stage
            max_self_concept: Maximum self-concept complexity for current stage
            max_autobiographical: Maximum autobiographical memory access for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.self_awareness = min(
            max_self_awareness,
            self.self_awareness + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.self_concept_complexity = min(
            max_self_concept,
            self.self_concept_complexity + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.autobiographical_memory_access = min(
            max_autobiographical,
            self.autobiographical_memory_access + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_theory_of_mind(
        self,
        max_theory_of_mind: float,
        max_perspective: float,
        max_false_belief: float
    ):
        """
        Update theory of mind metrics with gradual improvement.
        
        Args:
            max_theory_of_mind: Maximum theory of mind for current stage
            max_perspective: Maximum perspective taking for current stage
            max_false_belief: Maximum false belief understanding for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.theory_of_mind = min(
            max_theory_of_mind,
            self.theory_of_mind + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.perspective_taking = min(
            max_perspective,
            self.perspective_taking + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.false_belief_understanding = min(
            max_false_belief,
            self.false_belief_understanding + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _update_metacognition(
        self,
        max_metacognition: float,
        max_introspection: float,
        max_control: float
    ):
        """
        Update metacognition metrics with gradual improvement.
        
        Args:
            max_metacognition: Maximum metacognition for current stage
            max_introspection: Maximum introspection for current stage
            max_control: Maximum cognitive control for current stage
        """
        # Gradual improvement toward stage maximum
        improvement_rate = 0.01  # Small incremental improvements
        
        self.metacognition = min(
            max_metacognition,
            self.metacognition + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.introspection = min(
            max_introspection,
            self.introspection + improvement_rate * random.uniform(0.8, 1.2)
        )
        
        self.cognitive_control = min(
            max_control,
            self.cognitive_control + improvement_rate * random.uniform(0.8, 1.2)
        )
    
    def _generate_self_reflection(self, stage: DevelopmentalStage):
        """
        Generate a self-reflection based on current development level.
        
        Args:
            stage: Current developmental stage
        """
        # Only generate reflections at appropriate developmental stages
        if stage == DevelopmentalStage.INFANCY:
            # Infants don't have verbal self-reflection
            return
        
        # Determine reflection complexity based on development
        reflection_complexity = 0.0
        
        if stage == DevelopmentalStage.EARLY_CHILDHOOD:
            reflection_complexity = 0.3  # Simple self-descriptions
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            reflection_complexity = 0.6  # More complex self-understanding
        elif stage == DevelopmentalStage.ADOLESCENCE:
            reflection_complexity = 0.8  # Identity exploration
        elif stage == DevelopmentalStage.EARLY_ADULTHOOD:
            reflection_complexity = 1.0  # Mature self-reflection
        
        # Generate reflection based on complexity
        reflection = ""
        
        if reflection_complexity <= 0.3:
            # Simple self-descriptions
            descriptors = ["happy", "sad", "big", "small", "good", "bad"]
            activities = ["play", "eat", "sleep", "run", "jump", "draw"]
            
            reflection = f"I am {random.choice(descriptors)}. I like to {random.choice(activities)}."
            
        elif reflection_complexity <= 0.6:
            # More complex self-understanding
            traits = ["kind", "smart", "funny", "curious", "brave", "careful"]
            comparisons = ["better at", "not as good at", "different in", "similar in"]
            domains = ["reading", "math", "sports", "art", "making friends", "following rules"]
            
            reflection = f"I am {random.choice(traits)}. I am {random.choice(comparisons)} {random.choice(domains)} than some other kids."
            
        elif reflection_complexity <= 0.8:
            # Identity exploration
            identity_aspects = [
                "who I want to be in the future",
                "what makes me different from others",
                "my values and beliefs",
                "how others see me",
                "what is important to me"
            ]
            
            emotions = ["confused about", "interested in", "worried about", "excited about"]
            
            reflection = f"I've been thinking about {random.choice(identity_aspects)}. I feel {random.choice(emotions)} figuring out who I really am."
            
        else:
            # Mature self-reflection
            insights = [
                "I notice patterns in how I respond to challenges",
                "I understand my strengths and weaknesses better now",
                "I can see how my past experiences have shaped who I am",
                "I recognize when my emotions are affecting my thinking",
                "I'm aware of how my actions impact others"
            ]
            
            developments = [
                "becoming more comfortable with uncertainty",
                "learning to balance different aspects of myself",
                "integrating different perspectives into my worldview",
                "developing a more stable sense of identity",
                "understanding the complexity of my own motivations"
            ]
            
            reflection = f"{random.choice(insights)}. I'm {random.choice(developments)}."
        
        # Add to reflections list
        self.reflections.append(reflection)
        
        # Limit size of reflections list
        if len(self.reflections) > self.max_reflections:
            self.reflections.pop(0)  # Remove oldest reflection
    
    def process_interaction(
        self, 
        interaction_text: str, 
        emotional_state: List[Emotion],
        developmental_stage: DevelopmentalStage
    ) -> Dict[str, Any]:
        """
        Process an interaction to update consciousness development.
        
        Args:
            interaction_text: Text of the interaction
            emotional_state: Current emotional state
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary with consciousness processing results
        """
        # Update consciousness based on stage
        self.update_consciousness(developmental_stage)
        
        # Process self-references in text
        self_references = self._count_self_references(interaction_text)
        
        # Process perspective-taking opportunities
        perspective_taking = self._evaluate_perspective_taking(interaction_text, developmental_stage)
        
        # Process metacognitive elements
        metacognitive_elements = self._identify_metacognitive_elements(interaction_text, developmental_stage)
        
        # Small boost to metrics based on interaction content
        if self_references > 0:
            self.self_awareness = min(1.0, self.self_awareness + 0.005 * self_references)
            
        if perspective_taking > 0:
            self.theory_of_mind = min(1.0, self.theory_of_mind + 0.005 * perspective_taking)
            
        if metacognitive_elements > 0:
            self.metacognition = min(1.0, self.metacognition + 0.005 * metacognitive_elements)
        
        # Return processing results
        return {
            "self_references": self_references,
            "perspective_taking": perspective_taking,
            "metacognitive_elements": metacognitive_elements,
            "current_self_awareness": self.self_awareness,
            "current_theory_of_mind": self.theory_of_mind,
            "current_metacognition": self.metacognition
        }
    
    def _count_self_references(self, text: str) -> int:
        """
        Count references to self in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Number of self-references
        """
        # Simple count of first-person pronouns
        self_words = ["i", "me", "my", "mine", "myself"]
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Count occurrences
        count = sum(1 for word in words if word.strip(".,!?;:\"'()[]{}") in self_words)
        
        return count
    
    def _evaluate_perspective_taking(self, text: str, stage: DevelopmentalStage) -> int:
        """
        Evaluate perspective-taking elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Score for perspective-taking elements
        """
        # Only relevant for later stages
        if stage < DevelopmentalStage.MIDDLE_CHILDHOOD:
            return 0
        
        # Look for perspective-taking indicators
        perspective_indicators = [
            "you think", "you feel", "you believe",
            "they think", "they feel", "they believe",
            "understand why", "point of view", "perspective",
            "in your shoes", "from your perspective"
        ]
        
        # Count occurrences
        text_lower = text.lower()
        count = sum(1 for indicator in perspective_indicators if indicator in text_lower)
        
        return count
    
    def _identify_metacognitive_elements(self, text: str, stage: DevelopmentalStage) -> int:
        """
        Identify metacognitive elements in text.
        
        Args:
            text: Text to analyze
            stage: Current developmental stage
            
        Returns:
            Score for metacognitive elements
        """
        # Only relevant for later stages
        if stage < DevelopmentalStage.MIDDLE_CHILDHOOD:
            return 0
        
        # Look for metacognitive indicators
        metacognitive_indicators = [
            "i think", "i believe", "i know", "i wonder",
            "thinking about", "trying to understand",
            "realize", "remember", "forget", "confused",
            "understand", "figure out", "learning"
        ]
        
        # Count occurrences
        text_lower = text.lower()
        count = sum(1 for indicator in metacognitive_indicators if indicator in text_lower)
        
        return count
    
    def get_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get the current consciousness development metrics.
        
        Returns:
            Dictionary of consciousness metrics
        """
        return {
            "self_awareness": self.self_awareness,
            "self_concept_complexity": self.self_concept_complexity,
            "autobiographical_memory_access": self.autobiographical_memory_access,
            "theory_of_mind": self.theory_of_mind,
            "perspective_taking": self.perspective_taking,
            "false_belief_understanding": self.false_belief_understanding,
            "metacognition": self.metacognition,
            "introspection": self.introspection,
            "cognitive_control": self.cognitive_control
        }
    
    def get_recent_reflections(self, limit: int = 5) -> List[str]:
        """
        Get the most recent self-reflections.
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of recent self-reflections
        """
        return self.reflections[-limit:] if self.reflections else []
    
    def update_consciousness_development(self, developmental_stage: DevelopmentalStage) -> Dict[str, float]:
        """
        Update consciousness development based on the developmental stage.
        Test compatibility method.
        
        Args:
            developmental_stage: The current developmental stage
            
        Returns:
            Dict of updated consciousness metrics
        """
        # Call update_consciousness to handle the actual update
        result = self.update_consciousness(developmental_stage)
        
        # Update test compatibility attribute
        self.reflective_thinking = self.metacognition
        
        return result
        
    def get_self_awareness(self) -> float:
        """Return the current self-awareness level for test compatibility."""
        return self.self_awareness 