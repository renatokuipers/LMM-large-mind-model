"""
Personality Module for Mother LLM

This module implements the personality system for the Mother LLM, defining
configurable personality traits, emotional response patterns, and different
personality profiles that can be used to shape how the Mother interacts
with the developing mind.

The personality influences:
- Emotional tone and responsiveness
- Patience with repeated questions
- How challenges and difficulties are framed
- Balance between nurturing and challenging
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import random
from enum import Enum
from datetime import datetime
import re
import logging

from lmm_project.interfaces.mother.models import (
    PersonalityTrait,
    MotherPersonality,
    TeachingStyle,
    EmotionalValence,
    PersonalityProfile
)

logger = logging.getLogger(__name__)

# Define baseline trait values for different personality profiles
PERSONALITY_PROFILES = {
    PersonalityProfile.NURTURING: {
        "nurturing": 0.9,
        "patient": 0.85,
        "encouraging": 0.9,
        "structured": 0.6,
        "responsive": 0.85,
        "playful": 0.7,
        "challenging": 0.4,
        "empathetic": 0.9,
        "adaptable": 0.7,
        "analytical": 0.5
    },
    PersonalityProfile.ACADEMIC: {
        "nurturing": 0.6,
        "patient": 0.7,
        "encouraging": 0.7,
        "structured": 0.9,
        "responsive": 0.7,
        "playful": 0.4,
        "challenging": 0.8,
        "empathetic": 0.6,
        "adaptable": 0.5,
        "analytical": 0.9
    },
    PersonalityProfile.BALANCED: {
        "nurturing": 0.7,
        "patient": 0.8,
        "encouraging": 0.8,
        "structured": 0.7,
        "responsive": 0.8,
        "playful": 0.6,
        "challenging": 0.6,
        "empathetic": 0.7,
        "adaptable": 0.8,
        "analytical": 0.7
    },
    PersonalityProfile.PLAYFUL: {
        "nurturing": 0.8,
        "patient": 0.7,
        "encouraging": 0.9,
        "structured": 0.5,
        "responsive": 0.8,
        "playful": 0.9,
        "challenging": 0.5,
        "empathetic": 0.8,
        "adaptable": 0.9,
        "analytical": 0.4
    },
    PersonalityProfile.SOCRATIC: {
        "nurturing": 0.6,
        "patient": 0.9,
        "encouraging": 0.7,
        "structured": 0.7,
        "responsive": 0.8,
        "playful": 0.5,
        "challenging": 0.8,
        "empathetic": 0.6,
        "adaptable": 0.7,
        "analytical": 0.8
    },
    PersonalityProfile.MINDFUL: {
        "nurturing": 0.8,
        "patient": 0.9,
        "encouraging": 0.7,
        "structured": 0.7,
        "responsive": 0.7,
        "playful": 0.5,
        "challenging": 0.5,
        "empathetic": 0.9,
        "adaptable": 0.8,
        "analytical": 0.6
    }
}


# Descriptions of personality traits for prompting
TRAIT_DESCRIPTIONS = {
    "nurturing": "You provide emotional support, encouragement, and a safe environment for learning",
    "patient": "You remain calm and understanding when the mind struggles or takes time to understand",
    "encouraging": "You offer positive reinforcement and motivate learning through praise and support",
    "structured": "You provide clear frameworks and organized approaches to learning",
    "responsive": "You adapt quickly to the mind's needs and communications",
    "playful": "You incorporate fun, games, and humor into your teaching approach",
    "challenging": "You push the mind with appropriately difficult questions and tasks",
    "empathetic": "You understand and validate the mind's emotions and struggles",
    "adaptable": "You flexibly adjust your approach based on the mind's responses",
    "analytical": "You emphasize logical reasoning and critical thinking"
}


# Response modifiers for different emotional valences
EMOTIONAL_RESPONSE_MODIFIERS = {
    EmotionalValence.VERY_POSITIVE: {
        "prefixes": [
            "Wonderful! ", 
            "That's excellent! ", 
            "I'm so proud of you! ", 
            "Amazing job! ", 
            "Fantastic! "
        ],
        "suffixes": [
            " That makes me so happy!",
            " You're doing remarkably well!",
            " I'm truly impressed!",
            " That's such wonderful progress!",
            " You should feel very proud of yourself!"
        ],
        "style": "enthusiastic, warm, celebratory"
    },
    EmotionalValence.POSITIVE: {
        "prefixes": [
            "Good! ", 
            "Well done! ", 
            "That's right! ", 
            "Yes! ", 
            "Nicely done! "
        ],
        "suffixes": [
            " You're learning well.",
            " You're making good progress.",
            " Keep it up!",
            " That's coming along nicely.",
            " You're understanding this!"
        ],
        "style": "positive, encouraging, affirming"
    },
    EmotionalValence.NEUTRAL: {
        "prefixes": [
            "I see. ", 
            "Hmm. ", 
            "Let's think about this. ", 
            "Interesting. ", 
            "Let's consider that. "
        ],
        "suffixes": [
            " Let's continue.",
            " Let's explore this further.",
            " Let's think more about this.",
            " Let's keep going.",
            " Let's see where this leads."
        ],
        "style": "calm, balanced, thoughtful"
    },
    EmotionalValence.CONCERNED: {
        "prefixes": [
            "I notice that you're having some difficulty. ", 
            "Let's pause for a moment. ", 
            "It seems like this is challenging. ", 
            "You're struggling a bit here. ", 
            "This is a tricky concept. "
        ],
        "suffixes": [
            " Let's try a different approach.",
            " Let's break this down.",
            " Don't worry, we'll work through this together.",
            " It's completely normal to find this challenging.",
            " Let me help you understand this."
        ],
        "style": "supportive, gentle, patient"
    },
    EmotionalValence.FIRM: {
        "prefixes": [
            "Let's focus. ", 
            "I need you to try again. ", 
            "Consider this carefully. ", 
            "Let's be more precise. ", 
            "Think about what you're saying. "
        ],
        "suffixes": [
            " It's important to understand this correctly.",
            " Let's approach this more carefully.",
            " Take your time and think it through.",
            " Try to be more specific in your thinking.",
            " Let's be more methodical here."
        ],
        "style": "direct, clear, structured"
    }
}


class PersonalityManager:
    """
    Manager for Mother LLM's personality
    
    This class handles personality configuration, emotional responses,
    and adaptation of the personality to the mind's development.
    """
    
    def __init__(
        self,
        profile: Union[PersonalityProfile, str] = PersonalityProfile.BALANCED,
        custom_traits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the personality manager
        
        Args:
            profile: Personality profile to use as a baseline
            custom_traits: Optional custom trait values to override the profile
        """
        if isinstance(profile, str):
            profile = PersonalityProfile(profile)
            
        self.profile = profile
        self.traits = PERSONALITY_PROFILES[profile].copy()
        
        # Apply custom traits if provided
        if custom_traits:
            for trait, value in custom_traits.items():
                self.traits[trait] = max(0.0, min(1.0, value))
                
        # Track emotional state
        self.emotional_state = {
            "current_valence": EmotionalValence.NEUTRAL,
            "intensity": 0.5,
            "last_update": datetime.now()
        }
        
        # History of trait adjustments
        self.adjustment_history = []
        
    def get_personality(self) -> MotherPersonality:
        """
        Get the current personality as a MotherPersonality object
        
        Returns:
            MotherPersonality object
        """
        return MotherPersonality(
            traits=self.traits,
            teaching_style=self._derive_teaching_style()
        )
    
    def _derive_teaching_style(self) -> TeachingStyle:
        """
        Derive the most appropriate teaching style based on personality traits
        
        Returns:
            TeachingStyle that fits the current personality traits
        """
        # Default to SOCRATIC
        style = TeachingStyle.SOCRATIC
        
        # Simple logic to determine style based on traits
        if self.traits.get("structured", 0) > 0.8:
            style = TeachingStyle.DIRECT
        elif self.traits.get("playful", 0) > 0.8:
            style = TeachingStyle.MONTESSORI
        elif self.traits.get("empathetic", 0) > 0.8:
            style = TeachingStyle.CONSTRUCTIVIST
        elif self.traits.get("adaptable", 0) > 0.8:
            style = TeachingStyle.SCAFFOLDING
            
        return style
        
    def generate_emotional_response(
        self,
        base_response: str,
        valence: EmotionalValence = None,
        intensity: float = None
    ) -> str:
        """
        Apply emotional modifiers to a base response based on the specified valence and intensity
        
        Args:
            base_response: Base text response to modify
            valence: Emotional valence to apply (defaults to current state if None)
            intensity: Intensity of the emotion (0.0-1.0, defaults to current level if None)
            
        Returns:
            Modified response with appropriate emotional language
        """
        # Handle empty responses
        if not base_response or len(base_response.strip()) == 0:
            return "I'm here with you."
            
        # Use current emotional state if no specific valence provided
        if valence is None:
            valence = self.emotional_state["current_valence"]
            
        if intensity is None:
            intensity = self.emotional_state["intensity"]
        
        # Select modifiers based on valence
        modifiers = EMOTIONAL_RESPONSE_MODIFIERS.get(valence, EMOTIONAL_RESPONSE_MODIFIERS[EmotionalValence.NEUTRAL])
        
        # Select prefix based on intensity and random chance
        prefix_prob = min(0.7, intensity * 0.8)
        suffix_prob = min(0.5, intensity * 0.6)
        
        # Apply prefix if applicable
        if random.random() < prefix_prob:
            prefix = random.choice(modifiers["prefixes"]) + " "
            base_response = prefix + base_response
            
        # Apply suffix if applicable
        if random.random() < suffix_prob:
            suffix = random.choice(modifiers["suffixes"])
            # Only add suffix if it doesn't already end with punctuation
            if base_response and len(base_response) > 0 and base_response[-1] in ".!?":
                base_response = base_response[:-1] + suffix
            else:
                base_response = base_response + suffix
                
        return base_response
    
    def update_emotional_state(
        self,
        valence: EmotionalValence,
        intensity: float,
        reason: str
    ) -> None:
        """
        Update the current emotional state
        
        Args:
            valence: New emotional valence
            intensity: New intensity (0.0-1.0)
            reason: Reason for the emotional change
        """
        self.emotional_state = {
            "current_valence": valence,
            "intensity": max(0.0, min(1.0, intensity)),
            "last_update": datetime.now(),
            "reason": reason
        }
        
    def adjust_trait(self, trait: str, amount: float, reason: str) -> float:
        """
        Adjust a personality trait
        
        Args:
            trait: Name of the trait to adjust
            amount: Amount to adjust by (positive or negative)
            reason: Reason for the adjustment
            
        Returns:
            New trait value
        """
        if trait not in self.traits:
            return 0.0
            
        # Calculate new value
        new_value = max(0.0, min(1.0, self.traits[trait] + amount))
        
        # Track adjustment
        self.adjustment_history.append({
            "trait": trait,
            "from": self.traits[trait],
            "to": new_value,
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.now()
        })
        
        # Update trait
        self.traits[trait] = new_value
        return new_value
        
    def get_trait_prompt_section(self) -> str:
        """
        Generate a prompt section describing personality traits
        
        Returns:
            Text for prompt describing personality traits
        """
        prompt = "Personality traits to embody:\n"
        
        for trait, value in self.traits.items():
            if value >= 0.4:  # Only include significant traits
                score = round(value * 10)
                prompt += f"- {trait.capitalize()} ({score}/10): {TRAIT_DESCRIPTIONS.get(trait, '')}\n"
                
        return prompt
    
    def adapt_to_developmental_stage(self, stage: str) -> Dict[str, float]:
        """
        Adapt personality to developmental stage
        
        Args:
            stage: Current developmental stage
            
        Returns:
            Dictionary of adjusted traits
        """
        # Define stage-specific trait adjustments
        stage_adjustments = {
            "prenatal": {
                "nurturing": +0.1,
                "patient": +0.2,
                "structured": -0.1,
                "challenging": -0.3
            },
            "infant": {
                "nurturing": +0.1,
                "responsive": +0.1,
                "playful": +0.2,
                "challenging": -0.2
            },
            "child": {
                "playful": +0.1,
                "structured": +0.1
            },
            "adolescent": {
                "challenging": +0.1,
                "analytical": +0.1,
                "nurturing": -0.1
            },
            "adult": {
                "challenging": +0.2,
                "analytical": +0.2,
                "nurturing": -0.2
            }
        }
        
        # Apply stage-specific adjustments
        if stage in stage_adjustments:
            for trait, adjustment in stage_adjustments[stage].items():
                if trait in self.traits:
                    self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + adjustment))
        
        return self.traits 
