"""
Social cognition module for the Large Mind Model (LMM).

This module handles social understanding, empathy, and theory of mind
for the LMM, enabling it to reason about social interactions.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import random

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.mind_modules.base import MindModule
from lmm.core.development.stages import DevelopmentalStage

logger = get_logger("lmm.mind_modules.social")

class SocialCognitionModule(MindModule):
    """
    Handles social understanding and empathy for the LMM.
    
    This module manages social cognition, including theory of mind,
    empathy, and understanding of social norms and interactions.
    """
    
    def __init__(self):
        """Initialize the Social Cognition Module."""
        super().__init__("Social Cognition")
        
        # Initialize social cognition parameters
        self.empathy_level = 0.3  # Starts low, increases with development
        self.theory_of_mind = 0.2  # Ability to understand others' mental states
        self.social_norm_understanding = 0.2  # Understanding of social rules
        
        # Social interaction history
        self.interaction_history = []
        
        logger.info("Initialized Social Cognition Module")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for social cognition operations.
        
        Args:
            input_data: Dictionary containing input data
                - input: Input text
                - language_understanding: Results from language module
                - relevant_memories: Relevant memories
                - emotional_state: Current emotional state
                - developmental_stage: Current developmental stage
                
        Returns:
            Dictionary with social cognition results
        """
        # Extract input parameters
        text = input_data.get("input", "")
        language_understanding = input_data.get("language_understanding", {})
        relevant_memories = input_data.get("relevant_memories", [])
        emotional_state = input_data.get("emotional_state", {})
        stage = input_data.get("developmental_stage", DevelopmentalStage.PRENATAL.value)
        
        # Update developmental parameters
        self._update_developmental_parameters(stage)
        
        # Analyze social content in input
        social_analysis = self._analyze_social_content(text)
        
        # Apply empathy to understand emotional content
        empathy_analysis = self._apply_empathy(text, emotional_state)
        
        # Record interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": text[:100] if len(text) > 100 else text,
            "social_analysis": social_analysis,
            "empathy_analysis": empathy_analysis,
            "developmental_stage": stage
        })
        
        # Limit history size
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-50:]
        
        # Return combined results
        return {
            "social_understanding": social_analysis,
            "empathy": empathy_analysis,
            "theory_of_mind": self._apply_theory_of_mind(text, language_understanding),
            "social_norm_awareness": self._check_social_norms(text),
            "empathy_level": self.empathy_level,
            "theory_of_mind_level": self.theory_of_mind,
            "social_norm_understanding": self.social_norm_understanding
        }
    
    def _analyze_social_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze social aspects of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with social analysis
        """
        # Check for social keywords
        social_keywords = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "farewell": ["goodbye", "bye", "see you", "farewell", "take care", "until next time"],
            "gratitude": ["thank", "appreciate", "grateful", "thanks"],
            "apology": ["sorry", "apologize", "regret", "apology", "forgive"],
            "agreement": ["agree", "yes", "indeed", "absolutely", "exactly", "correct"],
            "disagreement": ["disagree", "no", "not really", "incorrect", "wrong", "nope"]
        }
        
        social_signals = {}
        text_lower = text.lower()
        
        for category, keywords in social_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    social_signals[category] = True
                    break
        
        # Calculate social interaction level (0.0-1.0)
        interaction_level = min(1.0, len(social_signals) * 0.2)
        
        return {
            "social_signals": social_signals,
            "interaction_level": interaction_level
        }
    
    def _apply_empathy(self, text: str, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply empathy to understand the emotional aspects of text.
        
        Args:
            text: Input text
            emotional_state: Current emotional state
            
        Returns:
            Dictionary with empathy analysis
        """
        # Simple emotion detection
        emotion_keywords = {
            "joy": ["happy", "glad", "joy", "delight", "pleased", "excited"],
            "sadness": ["sad", "unhappy", "sorrow", "grief", "disappointed"],
            "anger": ["angry", "mad", "furious", "annoyed", "frustrated"],
            "fear": ["afraid", "scared", "worried", "anxious", "frightened"],
            "surprise": ["surprised", "amazed", "astonished", "shocked"]
        }
        
        detected_emotions = {}
        text_lower = text.lower()
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_emotions[emotion] = True
                    break
        
        # Apply empathy level to determine empathic response
        empathic_understanding = min(1.0, len(detected_emotions) * self.empathy_level)
        
        return {
            "detected_emotions": list(detected_emotions.keys()),
            "empathic_understanding": empathic_understanding,
            "empathy_limited_by": 1.0 - self.empathy_level if self.empathy_level < 0.8 else None
        }
    
    def _apply_theory_of_mind(self, text: str, language_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply theory of mind to understand perspectives and beliefs.
        
        Args:
            text: Input text
            language_understanding: Results from language module
            
        Returns:
            Dictionary with theory of mind analysis
        """
        # Check for perspective-taking words
        perspective_keywords = ["think", "believe", "know", "understand", "feel", "want", "need", "hope"]
        
        perspective_indicators = []
        text_lower = text.lower()
        
        for keyword in perspective_keywords:
            if f"you {keyword}" in text_lower:
                perspective_indicators.append(f"you_{keyword}")
            if f"i {keyword}" in text_lower:
                perspective_indicators.append(f"i_{keyword}")
        
        # Calculate perspective understanding
        perspective_understanding = min(1.0, len(perspective_indicators) * 0.2 * self.theory_of_mind)
        
        return {
            "perspective_indicators": perspective_indicators,
            "perspective_understanding": perspective_understanding,
            "theory_of_mind_limited_by": 1.0 - self.theory_of_mind if self.theory_of_mind < 0.8 else None
        }
    
    def _check_social_norms(self, text: str) -> Dict[str, Any]:
        """
        Check for adherence to social norms in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with social norm analysis
        """
        # Check for politeness indicators
        politeness_indicators = ["please", "thank you", "thanks", "excuse me", "sorry", "may i", "would you"]
        
        politeness_score = 0.0
        text_lower = text.lower()
        
        for indicator in politeness_indicators:
            if indicator in text_lower:
                politeness_score += 0.2
        
        politeness_score = min(1.0, politeness_score)
        
        # Limited by social norm understanding
        effective_politeness = politeness_score * self.social_norm_understanding
        
        return {
            "politeness_score": politeness_score,
            "effective_politeness": effective_politeness,
            "social_norm_understanding_limited_by": 1.0 - self.social_norm_understanding if self.social_norm_understanding < 0.8 else None
        }
    
    def _update_developmental_parameters(self, stage: str) -> None:
        """
        Update social cognition parameters based on developmental stage.
        
        Args:
            stage: Current developmental stage
        """
        # Define social cognition development by stage
        stage_params = {
            DevelopmentalStage.PRENATAL.value: {
                "empathy_level": 0.1,
                "theory_of_mind": 0.0,
                "social_norm_understanding": 0.0
            },
            DevelopmentalStage.INFANCY.value: {
                "empathy_level": 0.3,
                "theory_of_mind": 0.1,
                "social_norm_understanding": 0.1
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "empathy_level": 0.5,
                "theory_of_mind": 0.4,
                "social_norm_understanding": 0.3
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "empathy_level": 0.7,
                "theory_of_mind": 0.6,
                "social_norm_understanding": 0.6
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "empathy_level": 0.8,
                "theory_of_mind": 0.8,
                "social_norm_understanding": 0.7
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "empathy_level": 0.9,
                "theory_of_mind": 0.9,
                "social_norm_understanding": 0.9
            }
        }
        
        # Get parameters for current stage
        params = stage_params.get(stage, stage_params[DevelopmentalStage.PRENATAL.value])
        
        # Update parameters
        self.empathy_level = params["empathy_level"]
        self.theory_of_mind = params["theory_of_mind"]
        self.social_norm_understanding = params["social_norm_understanding"]
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get the current status of the social cognition module.
        
        Returns:
            Dictionary with module status
        """
        # Get the base status
        status = super().get_module_status()
        
        # Add social cognition-specific status
        status.update({
            "empathy_level": self.empathy_level,
            "theory_of_mind": self.theory_of_mind,
            "social_norm_understanding": self.social_norm_understanding,
            "interactions_processed": len(self.interaction_history),
            "recent_social_signals": [
                interaction["social_analysis"]["social_signals"]
                for interaction in self.interaction_history[-5:]
            ] if self.interaction_history else []
        })
        
        return status 