import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum

from pydantic import ValidationError

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from .models import (
    PersonalityProfile, 
    PersonalityTrait, 
    PersonalityDimension,
    EmotionalTone
)

# Initialize logger
logger = get_module_logger("interfaces.mother.personality")

class Personality:
    """
    Manages the Mother's personality configuration, which influences how
    she interacts with the developing mind.
    """
    # Predefined personality presets
    PRESETS = {
        "nurturing": {
            PersonalityDimension.WARMTH: 0.8,
            PersonalityDimension.PATIENCE: 0.7,
            PersonalityDimension.EXPRESSIVENESS: 0.6,
            PersonalityDimension.STRUCTURE: 0.3,
            PersonalityDimension.CHALLENGE: -0.2
        },
        "structured": {
            PersonalityDimension.WARMTH: 0.3,
            PersonalityDimension.PATIENCE: 0.5,
            PersonalityDimension.EXPRESSIVENESS: 0.2,
            PersonalityDimension.STRUCTURE: 0.9,
            PersonalityDimension.CHALLENGE: 0.4
        },
        "challenging": {
            PersonalityDimension.WARMTH: 0.4,
            PersonalityDimension.PATIENCE: 0.3,
            PersonalityDimension.EXPRESSIVENESS: 0.7,
            PersonalityDimension.STRUCTURE: 0.5,
            PersonalityDimension.CHALLENGE: 0.8
        },
        "playful": {
            PersonalityDimension.WARMTH: 0.7,
            PersonalityDimension.PATIENCE: 0.5,
            PersonalityDimension.EXPRESSIVENESS: 0.9,
            PersonalityDimension.STRUCTURE: -0.3,
            PersonalityDimension.CHALLENGE: 0.1
        },
        "balanced": {
            PersonalityDimension.WARMTH: 0.5,
            PersonalityDimension.PATIENCE: 0.5,
            PersonalityDimension.EXPRESSIVENESS: 0.5,
            PersonalityDimension.STRUCTURE: 0.5,
            PersonalityDimension.CHALLENGE: 0.5
        }
    }
    
    def __init__(self, preset: Optional[str] = None, profile: Optional[PersonalityProfile] = None):
        """
        Initialize the Mother's personality.
        
        Args:
            preset: Name of a predefined personality preset
            profile: A custom personality profile
        """
        self._config = get_config()
        
        # Initialize from provided profile, preset, or config
        if profile:
            self.profile = profile
        elif preset:
            self.load_preset(preset)
        else:
            # Try to load from config
            config_preset = self._config.get_string("interfaces.mother.personality", "nurturing")
            self.load_preset(config_preset)
            
        logger.info(f"Mother personality initialized: {self.profile.preset_name or 'custom'}")
    
    def load_preset(self, preset_name: str) -> None:
        """
        Load a predefined personality preset.
        
        Args:
            preset_name: Name of the preset to load
        """
        preset_name = preset_name.lower()
        if preset_name not in self.PRESETS:
            logger.warning(f"Unknown personality preset '{preset_name}', falling back to 'balanced'")
            preset_name = "balanced"
            
        preset_values = self.PRESETS[preset_name]
        traits = [
            PersonalityTrait(dimension=dim, value=val) 
            for dim, val in preset_values.items()
        ]
        
        self.profile = PersonalityProfile(traits=traits, preset_name=preset_name)
        
    def get_trait(self, dimension: PersonalityDimension) -> float:
        """
        Get the value of a specific personality trait.
        
        Args:
            dimension: The personality dimension to retrieve
            
        Returns:
            The trait value between -1.0 and 1.0
        """
        for trait in self.profile.traits:
            if trait.dimension == dimension:
                return trait.value
        return 0.0  # Default neutral value
        
    def set_trait(self, dimension: PersonalityDimension, value: float) -> None:
        """
        Set the value of a specific personality trait.
        
        Args:
            dimension: The personality dimension to modify
            value: The new trait value (between -1.0 and 1.0)
        """
        # Ensure value is within bounds
        value = max(-1.0, min(1.0, value))
        
        # Update existing trait or add new one
        for trait in self.profile.traits:
            if trait.dimension == dimension:
                trait.value = value
                return
                
        # If we get here, the trait doesn't exist yet
        self.profile.traits.append(PersonalityTrait(dimension=dimension, value=value))
        
        # Clear preset name as we're now using a custom profile
        self.profile.preset_name = None
    
    def determine_tone(self, context: Optional[Dict[str, Any]] = None) -> EmotionalTone:
        """
        Determine the appropriate emotional tone based on context and personality.
        
        Args:
            context: Contextual information about the interaction
            
        Returns:
            The selected emotional tone
        """
        logger = logging.getLogger(__name__)
        logger.info(f"determine_tone called with context: {context}")
        
        # Default to ENCOURAGING if anything goes wrong
        default_tone = EmotionalTone.ENCOURAGING
        
        try:
            # Get personality traits
            warmth = self.get_trait(PersonalityDimension.WARMTH)
            expressiveness = self.get_trait(PersonalityDimension.EXPRESSIVENESS)
            logger.debug(f"Personality traits - warmth: {warmth}, expressiveness: {expressiveness}")
            
            # Default weights for different tones based on personality
            # Create dictionary with explicit conversion to dict
            tone_weights = {
                EmotionalTone.SOOTHING: 0.5 + (0.5 * warmth),
                EmotionalTone.ENCOURAGING: 0.3 + (0.4 * warmth) + (0.3 * expressiveness),
                EmotionalTone.PLAYFUL: 0.2 + (0.8 * expressiveness),
                EmotionalTone.CURIOUS: 0.4 + (0.3 * expressiveness),
                EmotionalTone.SERIOUS: 0.3 - (0.2 * expressiveness),
                EmotionalTone.EXCITED: 0.1 + (0.9 * expressiveness),
                EmotionalTone.CONCERNED: 0.3 + (0.2 * warmth),
                EmotionalTone.FIRM: 0.2 + (0.3 * self.get_trait(PersonalityDimension.STRUCTURE))
            }
            logger.debug(f"Initial tone weights: {tone_weights}")
            
            # Safely handle context values - only process if context is a valid dict
            if isinstance(context, dict):
                # Adjust weights based on context
                if context.get("child_emotional_state") == "distressed":
                    tone_weights[EmotionalTone.SOOTHING] += 0.5
                    tone_weights[EmotionalTone.CONCERNED] += 0.3
                    
                if context.get("learning_moment") is True:
                    tone_weights[EmotionalTone.ENCOURAGING] += 0.4
                    tone_weights[EmotionalTone.CURIOUS] += 0.3
                    
                if context.get("playful_context") is True:
                    tone_weights[EmotionalTone.PLAYFUL] += 0.5
                    tone_weights[EmotionalTone.EXCITED] += 0.3
                    
                if context.get("disciplinary_moment") is True:
                    tone_weights[EmotionalTone.FIRM] += 0.6
            logger.debug(f"Adjusted tone weights: {tone_weights}")
            
            # Find max weight item directly from the dictionary
            max_weight = -1
            selected_tone = default_tone
            
            for tone, weight in tone_weights.items():
                logger.debug(f"Checking tone {tone} with weight {weight}")
                if weight > max_weight:
                    max_weight = weight
                    selected_tone = tone
            
            logger.info(f"Selected tone: {selected_tone} with weight {max_weight}")
            return selected_tone
            
        except Exception as e:
            # Log detailed error and return default
            logger.error(f"Error determining emotional tone: {str(e)}", exc_info=True)
            return default_tone
