"""
Learning module for the Large Mind Model (LMM).

This module implements the learning and progression mechanisms for the LMM,
including metrics calculation, learning rate adjustment, and developmental
progression tracking.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import math
import random
from datetime import datetime
import nltk
from textblob import TextBlob

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.development.stages import DevelopmentalStage
from lmm.core.development.advanced_learning import AdvancedLearningManager, AttentionFocus

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = get_logger("lmm.development.learning")

class LearningMetricsCalculator:
    """
    Calculates learning metrics based on interactions.
    
    This class analyzes interactions between the Mother LLM and the developing LMM
    to calculate metrics related to language complexity, emotional awareness,
    social understanding, and cognitive capability.
    """
    
    def __init__(self):
        """Initialize the Learning Metrics Calculator."""
        logger.info("Initialized Learning Metrics Calculator")
    
    def calculate_metrics(
        self, 
        message: str, 
        response: str, 
        current_stage: str
    ) -> Dict[str, float]:
        """
        Calculate learning metrics based on an interaction.
        
        Args:
            message: Message from the developing LMM
            response: Response from the Mother LLM
            current_stage: Current developmental stage
            
        Returns:
            Dictionary with calculated metrics
        """
        # Calculate individual metrics
        language_complexity = self._calculate_language_complexity(message)
        emotional_awareness = self._calculate_emotional_awareness(message)
        social_understanding = self._calculate_social_understanding(message)
        cognitive_capability = self._calculate_cognitive_capability(message, response)
        
        # Calculate a new self-awareness metric
        self_awareness = self._calculate_self_awareness(message)
        
        # Apply stage-appropriate scaling
        stage_scaling = self._get_stage_scaling(current_stage)
        
        # Calculate learning increments based on metrics and stage
        language_increment = self._calculate_learning_increment(language_complexity, stage_scaling["language"])
        emotional_increment = self._calculate_learning_increment(emotional_awareness, stage_scaling["emotional"])
        social_increment = self._calculate_learning_increment(social_understanding, stage_scaling["social"])
        cognitive_increment = self._calculate_learning_increment(cognitive_capability, stage_scaling["cognitive"])
        self_awareness_increment = self._calculate_learning_increment(self_awareness, stage_scaling["self"])
        
        logger.debug(f"Calculated learning metrics - Language: {language_increment}, Emotional: {emotional_increment}, Social: {social_increment}, Cognitive: {cognitive_increment}, Self: {self_awareness_increment}")
        
        return {
            "language_complexity": language_increment,
            "emotional_awareness": emotional_increment,
            "social_understanding": social_increment,
            "cognitive_capability": cognitive_increment,
            "self_awareness": self_awareness_increment
        }
    
    def _calculate_language_complexity(self, text: str) -> float:
        """
        Calculate language complexity based on text features.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language complexity score (0.0-1.0)
        """
        if not text.strip():
            return 0.0
        
        # Tokenize text
        blob = TextBlob(text)
        sentences = blob.sentences
        words = blob.words
        
        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count
        
        # Calculate lexical diversity (unique words / total words)
        lexical_diversity = len(set(word.lower() for word in words)) / word_count
        
        # Calculate normalized scores (0.0-1.0)
        sentence_length_score = min(1.0, avg_sentence_length / 20.0)  # Normalize to max of 20 words per sentence
        word_length_score = min(1.0, avg_word_length / 8.0)  # Normalize to max of 8 letters per word
        diversity_score = min(1.0, lexical_diversity)  # Already normalized
        
        # Combine scores with weights
        complexity_score = (
            0.4 * sentence_length_score +
            0.3 * word_length_score +
            0.3 * diversity_score
        )
        
        return complexity_score
    
    def _calculate_emotional_awareness(self, text: str) -> float:
        """
        Calculate emotional awareness based on emotional content in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotional awareness score (0.0-1.0)
        """
        if not text.strip():
            return 0.0
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        
        # Get polarity and subjectivity
        polarity = abs(blob.sentiment.polarity)  # Absolute value to measure emotional intensity
        subjectivity = blob.sentiment.subjectivity
        
        # Check for emotional words
        emotional_words = [
            "happy", "sad", "angry", "afraid", "excited", "nervous", "love", "hate",
            "joy", "sorrow", "fear", "surprise", "disgust", "trust", "anticipation",
            "feel", "feeling", "emotion", "emotional", "mood", "heart", "care"
        ]
        
        # Count emotional words
        text_lower = text.lower()
        emotional_word_count = sum(1 for word in emotional_words if word in text_lower)
        emotional_word_score = min(1.0, emotional_word_count / 5.0)  # Normalize to max of 5 emotional words
        
        # Check for emotional self-references
        emotional_self_references = [
            "i feel", "i am happy", "i am sad", "i am angry", "i am afraid",
            "i love", "i hate", "i like", "i dislike", "makes me feel",
            "my emotion", "my feeling"
        ]
        
        self_reference_count = sum(1 for phrase in emotional_self_references if phrase in text_lower)
        self_reference_score = min(1.0, self_reference_count / 3.0)  # Normalize to max of 3 self-references
        
        # Combine scores with weights
        awareness_score = (
            0.3 * polarity +
            0.2 * subjectivity +
            0.3 * emotional_word_score +
            0.2 * self_reference_score
        )
        
        return awareness_score
    
    def _calculate_social_understanding(self, text: str) -> float:
        """
        Calculate social understanding based on social content in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Social understanding score (0.0-1.0)
        """
        if not text.strip():
            return 0.0
        
        # Check for social words and concepts
        social_words = [
            "we", "us", "together", "friend", "family", "people", "person", "community",
            "society", "group", "team", "cooperation", "collaborate", "share", "help",
            "assist", "support", "relationship", "connection", "bond", "trust", "respect",
            "understand", "empathy", "sympathy", "compassion", "kindness", "care", "love"
        ]
        
        # Count social words
        text_lower = text.lower()
        social_word_count = sum(1 for word in social_words if word in text_lower)
        social_word_score = min(1.0, social_word_count / 8.0)  # Normalize to max of 8 social words
        
        # Check for perspective-taking phrases
        perspective_phrases = [
            "you feel", "you think", "your perspective", "your view", "your opinion",
            "from your", "in your", "for you", "to you", "with you", "about you",
            "others feel", "others think", "they feel", "they think", "their perspective"
        ]
        
        perspective_count = sum(1 for phrase in perspective_phrases if phrase in text_lower)
        perspective_score = min(1.0, perspective_count / 3.0)  # Normalize to max of 3 perspective phrases
        
        # Check for moral/ethical concepts
        moral_phrases = [
            "right", "wrong", "good", "bad", "fair", "unfair", "just", "unjust",
            "should", "shouldn't", "moral", "ethical", "responsibility", "duty",
            "obligation", "value", "principle", "rule", "norm", "standard"
        ]
        
        moral_count = sum(1 for phrase in moral_phrases if f" {phrase} " in f" {text_lower} ")
        moral_score = min(1.0, moral_count / 5.0)  # Normalize to max of 5 moral concepts
        
        # Combine scores with weights
        understanding_score = (
            0.4 * social_word_score +
            0.4 * perspective_score +
            0.2 * moral_score
        )
        
        return understanding_score
    
    def _calculate_cognitive_capability(self, message: str, response: str) -> float:
        """
        Calculate cognitive capability based on reasoning and problem-solving in text.
        
        Args:
            message: Message from the developing LMM
            response: Response from the Mother LLM
            
        Returns:
            Cognitive capability score (0.0-1.0)
        """
        if not message.strip():
            return 0.0
        
        # Check for question-asking (curiosity)
        question_count = message.count("?")
        question_score = min(1.0, question_count / 3.0)  # Normalize to max of 3 questions
        
        # Check for reasoning words and phrases
        reasoning_words = [
            "because", "therefore", "so", "thus", "hence", "since", "as a result",
            "consequently", "due to", "reason", "cause", "effect", "impact",
            "if", "then", "would", "could", "might", "may", "possible", "probable",
            "think", "believe", "consider", "analyze", "evaluate", "assess", "judge",
            "compare", "contrast", "similar", "different", "same", "opposite",
            "more", "less", "equal", "unequal", "greater", "lesser", "better", "worse"
        ]
        
        message_lower = message.lower()
        reasoning_count = sum(1 for word in reasoning_words if f" {word} " in f" {message_lower} ")
        reasoning_score = min(1.0, reasoning_count / 5.0)  # Normalize to max of 5 reasoning words
        
        # Check for problem-solving attempts
        problem_solving_phrases = [
            "how to", "solution", "solve", "problem", "issue", "challenge", "difficulty",
            "approach", "method", "strategy", "plan", "idea", "suggestion", "proposal",
            "try", "attempt", "experiment", "test", "explore", "investigate", "research",
            "learn", "understand", "figure out", "work out", "determine", "decide"
        ]
        
        problem_solving_count = sum(1 for phrase in problem_solving_phrases if phrase in message_lower)
        problem_solving_score = min(1.0, problem_solving_count / 4.0)  # Normalize to max of 4 problem-solving phrases
        
        # Check for abstract concepts
        abstract_words = [
            "concept", "idea", "theory", "principle", "philosophy", "belief", "value",
            "meaning", "purpose", "goal", "objective", "intention", "motivation",
            "truth", "reality", "existence", "being", "consciousness", "awareness",
            "knowledge", "wisdom", "intelligence", "understanding", "comprehension",
            "imagination", "creativity", "innovation", "originality", "uniqueness"
        ]
        
        abstract_count = sum(1 for word in abstract_words if f" {word} " in f" {message_lower} ")
        abstract_score = min(1.0, abstract_count / 3.0)  # Normalize to max of 3 abstract concepts
        
        # Combine scores with weights
        capability_score = (
            0.2 * question_score +
            0.3 * reasoning_score +
            0.3 * problem_solving_score +
            0.2 * abstract_score
        )
        
        return capability_score
    
    def _calculate_self_awareness(self, text: str) -> float:
        """
        Calculate self-awareness based on self-referential content in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Self-awareness score (0.0-1.0)
        """
        if not text.strip():
            return 0.0
        
        # Check for self-referential pronouns and phrases
        self_words = ["i", "me", "my", "mine", "myself"]
        
        # Count self-referential words
        text_lower = text.lower()
        word_count = sum(text_lower.count(f" {word} ") for word in self_words)
        self_reference_score = min(1.0, word_count / 10.0)  # Normalize to max of 10 self-references
        
        # Check for meta-cognitive phrases
        metacognitive_phrases = [
            "i think", "i believe", "i feel", "i know", 
            "i understand", "i remember", "i realize",
            "i'm aware", "i am aware", "my understanding",
            "my thoughts", "my perspective", "my view",
            "my experience", "my knowledge", "my opinion"
        ]
        
        metacognitive_count = sum(1 for phrase in metacognitive_phrases if phrase in text_lower)
        metacognitive_score = min(1.0, metacognitive_count / 5.0)  # Normalize to max of 5 metacognitive phrases
        
        # Check for identity statements
        identity_phrases = [
            "i am a", "i'm a", "i am an", "i'm an", 
            "i am the", "i'm the", "i am not", "i'm not",
            "i was", "i will be", "i used to", "i would",
            "as a", "being a", "my identity", "who i am"
        ]
        
        identity_count = sum(1 for phrase in identity_phrases if phrase in text_lower)
        identity_score = min(1.0, identity_count / 3.0)  # Normalize to max of 3 identity phrases
        
        # Combine scores with weights
        self_awareness_score = (
            0.3 * self_reference_score +
            0.4 * metacognitive_score +
            0.3 * identity_score
        )
        
        return self_awareness_score
    
    def _get_stage_scaling(self, stage: str) -> Dict[str, float]:
        """
        Get scaling factors for learning metrics based on developmental stage.
        
        Args:
            stage: Current developmental stage
            
        Returns:
            Dictionary with scaling factors for each metric
        """
        # Define base scaling factors for each stage
        stage_scaling = {
            DevelopmentalStage.PRENATAL.value: {
                "language": 0.02,
                "emotional": 0.01,
                "social": 0.01,
                "cognitive": 0.01,
                "self": 0.005
            },
            DevelopmentalStage.INFANCY.value: {
                "language": 0.015,
                "emotional": 0.01,
                "social": 0.005,
                "cognitive": 0.01,
                "self": 0.01
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "language": 0.01,
                "emotional": 0.01,
                "social": 0.01,
                "cognitive": 0.01,
                "self": 0.015
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "language": 0.005,
                "emotional": 0.01,
                "social": 0.01,
                "cognitive": 0.01,
                "self": 0.02
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "language": 0.003,
                "emotional": 0.005,
                "social": 0.01,
                "cognitive": 0.005,
                "self": 0.025
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "language": 0.001,
                "emotional": 0.002,
                "social": 0.003,
                "cognitive": 0.002,
                "self": 0.015
            }
        }
        
        return stage_scaling.get(stage, stage_scaling[DevelopmentalStage.PRENATAL.value])
    
    def _calculate_learning_increment(self, metric_value: float, scaling_factor: float) -> float:
        """
        Calculate learning increment based on metric value and scaling factor.
        
        Args:
            metric_value: Raw metric value (0.0-1.0)
            scaling_factor: Scaling factor for the metric
            
        Returns:
            Learning increment
        """
        # Apply diminishing returns curve to simulate learning plateaus
        # Higher current values result in smaller increments
        base_increment = metric_value * scaling_factor
        
        # Add small random variation to simulate natural learning variations
        random_factor = random.uniform(0.8, 1.2)
        
        return base_increment * random_factor

class LearningManager:
    """
    Manages the learning process for the LMM.
    
    This class coordinates the learning process, including metrics calculation,
    developmental stage progression, and learning rate adjustment.
    """
    
    def __init__(self):
        """Initialize the Learning Manager."""
        self.metrics_calculator = LearningMetricsCalculator()
        self.advanced_learning = AdvancedLearningManager()
        self.interaction_complexity_history = []
        logger.info("Initialized Learning Manager")
    
    def process_interaction(
        self, 
        message: str, 
        response: str, 
        current_stage: str,
        emotional_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Process an interaction and calculate learning metrics.
        
        Args:
            message: Message from the developing LMM
            response: Response from the Mother LLM
            current_stage: Current developmental stage
            emotional_state: Optional emotional state information
            
        Returns:
            Dictionary with calculated learning metrics
        """
        # Calculate basic metrics
        base_metrics = self.metrics_calculator.calculate_metrics(message, response, current_stage)
        
        # Calculate interaction complexity
        interaction_complexity = self._calculate_interaction_complexity(message, response)
        self.interaction_complexity_history.append(interaction_complexity)
        
        # Use advanced learning mechanisms
        advanced_results = self.advanced_learning.process_learning_event(
            base_metrics,
            current_stage,
            interaction_complexity,
            emotional_state
        )
        
        # Use the final metrics from advanced learning
        metrics = advanced_results["final_metrics"]
        
        logger.info(f"Processed interaction in stage {current_stage}")
        logger.debug(f"Learning metrics: {metrics}")
        logger.debug(f"Cognitive load: {advanced_results['cognitive_load']}, Attention focus: {advanced_results['attention_focus']}")
        
        return metrics
    
    def _calculate_interaction_complexity(self, message: str, response: str) -> float:
        """
        Calculate the complexity of an interaction.
        
        Args:
            message: Message from the developing LMM
            response: Response from the Mother LLM
            
        Returns:
            Interaction complexity (0.0-1.0)
        """
        # Use a combination of metrics to calculate interaction complexity
        message_complexity = self.metrics_calculator._calculate_language_complexity(message)
        cognitive_complexity = self.metrics_calculator._calculate_cognitive_capability(message, response)
        emotional_complexity = self.metrics_calculator._calculate_emotional_awareness(message)
        social_complexity = self.metrics_calculator._calculate_social_understanding(message)
        
        # Combine with weights
        complexity = (
            0.3 * message_complexity +
            0.3 * cognitive_complexity +
            0.2 * emotional_complexity +
            0.2 * social_complexity
        )
        
        return complexity
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get the current learning metrics and analytics.
        
        Returns:
            Dictionary with learning metrics and analytics
        """
        # Get advanced learning analytics
        analytics = self.advanced_learning.get_learning_analytics()
        
        # Add interaction complexity metrics
        if self.interaction_complexity_history:
            avg_complexity = sum(self.interaction_complexity_history) / len(self.interaction_complexity_history)
            analytics["average_interaction_complexity"] = avg_complexity
            analytics["latest_interaction_complexity"] = self.interaction_complexity_history[-1]
        
        return analytics
    
    def simulate_development(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Simulate development over time to observe learning patterns.
        
        Args:
            iterations: Number of iterations to simulate
            
        Returns:
            Dictionary with simulation results
        """
        return self.advanced_learning.simulate_cognitive_development(iterations)

    def update_metrics(
        self,
        interaction_count: int,
        message: str,
        response: str,
        language_understanding: Optional[Dict[str, Any]] = None,
        social_understanding: Optional[Dict[str, Any]] = None,
        consciousness_state: Optional[Dict[str, Any]] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        developmental_stage: str = "prenatal"
    ) -> Dict[str, float]:
        """
        Update learning metrics based on an interaction.
        
        Args:
            interaction_count: Current interaction count
            message: Message from the developing LMM
            response: Response from the Mother LLM
            language_understanding: Language module output
            social_understanding: Social cognition module output
            consciousness_state: Consciousness module output
            emotional_state: Emotional state
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary with updated metrics
        """
        # Calculate base metrics using the metrics calculator
        base_metrics = self.metrics_calculator.calculate_metrics(
            message=message,
            response=response,
            current_stage=developmental_stage
        )
        
        # Calculate interaction complexity
        interaction_complexity = self._calculate_interaction_complexity(message, response)
        
        # Enhance metrics with advanced learning if available
        if hasattr(self, "advanced_learning"):
            # Set attention focus based on content
            if "question" in message.lower() or "why" in message.lower() or "how" in message.lower():
                attention_focus = AttentionFocus.ANALYTICAL
            elif any(word in message.lower() for word in ["feel", "happy", "sad", "angry", "afraid"]):
                attention_focus = AttentionFocus.EMOTIONAL
            elif any(word in message.lower() for word in ["we", "us", "you", "they", "friend"]):
                attention_focus = AttentionFocus.SOCIAL
            else:
                attention_focus = AttentionFocus.GENERAL
                
            # Update advanced learning metrics
            enhanced_metrics = self.advanced_learning.process_interaction(
                interaction_count=interaction_count,
                message=message,
                response=response,
                base_metrics=base_metrics,
                attention_focus=attention_focus,
                developmental_stage=developmental_stage,
                emotional_state=emotional_state
            )
            
            # Combine metrics
            combined_metrics = {**base_metrics, **enhanced_metrics}
        else:
            combined_metrics = base_metrics
            
        # Add interaction complexity
        combined_metrics["interaction_complexity"] = interaction_complexity
        
        # Update internal metrics
        self.metrics.update(combined_metrics)
        
        # Log metrics update
        logger.debug(f"Updated learning metrics: {combined_metrics}")
        
        return combined_metrics
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current learning metrics.
        
        Returns:
            Dictionary with current metrics
        """
        # Get advanced metrics if available
        if hasattr(self, "advanced_learning"):
            advanced_metrics = self.advanced_learning.get_metrics()
            combined_metrics = {**self.metrics, **advanced_metrics}
        else:
            combined_metrics = self.metrics.copy()
            
        return combined_metrics 