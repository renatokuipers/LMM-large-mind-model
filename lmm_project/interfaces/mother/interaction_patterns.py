import logging
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from .models import (
    InteractionStyle, 
    EmotionalTone, 
    InteractionPattern,
    TeachingMethod
)

# Initialize logger
logger = get_module_logger("interfaces.mother.interaction")

class InteractionPatterns:
    """
    Manages the patterns of interaction between Mother and child,
    including conversational styles, complexity adaptation, and
    interaction rhythm.
    """
    
    # Mapping of teaching methods to appropriate interaction styles
    TEACHING_STYLE_MAP = {
        TeachingMethod.DIRECT_INSTRUCTION: InteractionStyle.INSTRUCTIONAL,
        TeachingMethod.SCAFFOLDING: InteractionStyle.RESPONSIVE,
        TeachingMethod.SOCRATIC_QUESTIONING: InteractionStyle.QUESTIONING,
        TeachingMethod.REPETITION: InteractionStyle.INSTRUCTIONAL,
        TeachingMethod.REINFORCEMENT: InteractionStyle.RESPONSIVE,
        TeachingMethod.EXPERIENTIAL: InteractionStyle.STORYTELLING,
        TeachingMethod.DISCOVERY: InteractionStyle.REFLECTIVE,
        TeachingMethod.ANALOGICAL: InteractionStyle.STORYTELLING
    }
    
    # Default interaction styles for different developmental stages
    AGE_BASED_DEFAULTS = {
        # Age range: (primary style, secondary style)
        (0.0, 0.5): (InteractionStyle.RESPONSIVE, InteractionStyle.CONVERSATIONAL),
        (0.5, 1.5): (InteractionStyle.CONVERSATIONAL, InteractionStyle.STORYTELLING),
        (1.5, 3.0): (InteractionStyle.QUESTIONING, InteractionStyle.REFLECTIVE),
        (3.0, 6.0): (InteractionStyle.REFLECTIVE, InteractionStyle.QUESTIONING),
        (6.0, float('inf')): (InteractionStyle.REFLECTIVE, InteractionStyle.CONVERSATIONAL)
    }
    
    # Complexity scaling - how complex communication should be at different ages
    COMPLEXITY_SCALING = {
        0.0: 0.5,   # Very simple language for new minds
        0.5: 1.0,   # Simple but slightly more varied
        1.0: 2.0,   # Basic complexity
        2.0: 4.0,   # Moderate complexity
        3.0: 5.0,   # Higher complexity
        5.0: 7.0,   # Advanced complexity
        7.0: 10.0   # Full complex language
    }
    
    def __init__(self):
        """Initialize interaction patterns manager"""
        self._config = get_config()
        self._recent_patterns = []  # Track recent interactions for variety
        self._last_interaction_time = None
        
        # Maximum number of recent patterns to track
        self._max_recent_patterns = 5
        
        logger.info("Mother interaction patterns system initialized")
    
    def determine_complexity(self, age: float) -> float:
        """
        Determine appropriate language complexity for the current age.
        
        Args:
            age: Current developmental age
            
        Returns:
            Complexity level from 0.1 (simplest) to 10.0 (most complex)
        """
        try:
            # Ensure age is a float
            safe_age = float(age) if age is not None else 0.0
            
            # Find the appropriate complexity range
            age_brackets = sorted(list(self.COMPLEXITY_SCALING.keys()))
            
            # Handle case before first bracket
            if safe_age <= age_brackets[0]:
                return self.COMPLEXITY_SCALING[age_brackets[0]]
            
            # Linear interpolation between brackets
            for i, bracket_age in enumerate(age_brackets[1:], 1):
                if safe_age <= bracket_age:
                    prev_age = age_brackets[i-1]
                    prev_complexity = self.COMPLEXITY_SCALING[prev_age]
                    curr_complexity = self.COMPLEXITY_SCALING[bracket_age]
                    
                    # Calculate interpolation factor
                    factor = (safe_age - prev_age) / (bracket_age - prev_age)
                    return prev_complexity + factor * (curr_complexity - prev_complexity)
            
            # Beyond last age bracket
            return self.COMPLEXITY_SCALING[age_brackets[-1]]
        except Exception as e:
            # Log error and return a safe default value
            logger = logging.getLogger(__name__)
            logger.error(f"Error determining complexity: {e}")
            return 1.0  # Safe default complexity
    
    def select_interaction_pattern(
        self, 
        age: float,
        context: Dict[str, Any],
        teaching_method: Optional[TeachingMethod] = None,
        emotional_tone: Optional[EmotionalTone] = None
    ) -> InteractionPattern:
        """
        Select an appropriate interaction pattern based on current context.
        
        Args:
            age: Current developmental age
            context: Additional context for the interaction
            teaching_method: Currently selected teaching method (if any)
            emotional_tone: Currently selected emotional tone (if any)
            
        Returns:
            An InteractionPattern object
        """
        try:
            # Validate inputs
            if not isinstance(context, dict):
                context = {}  # Use empty dict if context is invalid
            
            safe_age = float(age) if age is not None else 0.0
                
            # Determine complexity level
            complexity = self.determine_complexity(safe_age)
            
            # Adjust complexity based on context - use safe gets
            if context.get("simplified_language", False):
                complexity *= 0.7
            if context.get("challenging_concept", False):
                complexity *= 0.8
            if context.get("receptive_state", False):
                complexity *= 1.2
            
            # Determine primary interaction style
            primary_style = self._determine_primary_style(safe_age, teaching_method, context)
            
            # If no emotional tone provided, determine one that complements the style
            if emotional_tone is None:
                primary_tone = self._select_default_tone(primary_style)
            else:
                primary_tone = emotional_tone
                
            # Select a complementary secondary tone
            secondary_tone = self._select_complementary_tone(primary_tone, primary_style)
            
            # Create and return the pattern
            pattern = InteractionPattern(
                style=primary_style,
                primary_tone=primary_tone,
                secondary_tone=secondary_tone,
                complexity_level=min(10.0, max(0.1, complexity))  # Clamp between 0.1 and 10.0
            )
            
            # Record the selected pattern
            self._record_pattern(pattern)
            
            return pattern
            
        except Exception as e:
            # Log error and provide safe fallback
            logger = logging.getLogger(__name__)
            logger.error(f"Error selecting interaction pattern: {e}")
            
            # Create a safe default pattern
            default_pattern = InteractionPattern(
                style=InteractionStyle.CONVERSATIONAL,
                primary_tone=EmotionalTone.ENCOURAGING,
                secondary_tone=None,
                complexity_level=1.0
            )
            
            return default_pattern
        
    def _determine_primary_style(
        self, 
        age: float, 
        teaching_method: Optional[TeachingMethod], 
        context: Dict[str, Any]
    ) -> InteractionStyle:
        """
        Determine the primary interaction style based on age, teaching method and context.
        
        Args:
            age: Current developmental age
            teaching_method: Selected teaching method (if any)
            context: Interaction context
            
        Returns:
            The selected interaction style
        """
        try:
            # Safety checks
            safe_age = float(age) if age is not None else 0.0
            safe_context = context if isinstance(context, dict) else {}
            
            # Get candidate styles based on age
            primary_candidates = []
            secondary_candidates = []
            
            # Convert dictionary items to list
            age_brackets = list(self.AGE_BASED_DEFAULTS.items())
            
            for (min_age, max_age), (primary, secondary) in age_brackets:
                if min_age <= safe_age <= max_age:
                    primary_candidates.append(primary)
                    secondary_candidates.append(secondary)
                    
            # If no specific age bracket matches, find the closest one
            if not primary_candidates:
                # Find the closest age bracket
                closest_bracket = min(age_brackets, key=lambda x: abs((x[0][0] + x[0][1])/2 - safe_age))
                primary_candidates.append(closest_bracket[1][0])
                secondary_candidates.append(closest_bracket[1][1])
            
            # First check if teaching method maps to a specific style
            if teaching_method:
                method_style = self.TEACHING_STYLE_MAP.get(teaching_method)
                if method_style:
                    # 70% chance to use the teaching method's style
                    if random.random() < 0.7:
                        return method_style
            
            # Next check context for special conditions that might influence style
            if safe_context.get("needs_reflection", False):
                return InteractionStyle.REFLECTIVE
                
            if safe_context.get("needs_instruction", False):
                return InteractionStyle.INSTRUCTIONAL
                
            if safe_context.get("needs_correction", False):
                return InteractionStyle.CORRECTIVE
            
            # Otherwise, randomly select from candidates with primary having higher weight
            if random.random() < 0.7 and primary_candidates:
                return random.choice(primary_candidates)
            elif secondary_candidates:
                return random.choice(secondary_candidates)
            else:
                # Final fallback
                return InteractionStyle.CONVERSATIONAL
                
        except Exception as e:
            # Log error and return a safe default
            logger = logging.getLogger(__name__)
            logger.error(f"Error determining primary style: {e}")
            return InteractionStyle.CONVERSATIONAL
    
    def _select_default_tone(self, style: InteractionStyle) -> EmotionalTone:
        """
        Select a default emotional tone that pairs well with a given interaction style.
        
        Args:
            style: The interaction style
            
        Returns:
            A complementary emotional tone
        """
        # Style-appropriate tones
        style_tone_map = {
            InteractionStyle.CONVERSATIONAL: [
                EmotionalTone.ENCOURAGING, EmotionalTone.PLAYFUL, EmotionalTone.CURIOUS
            ],
            InteractionStyle.INSTRUCTIONAL: [
                EmotionalTone.SERIOUS, EmotionalTone.ENCOURAGING, EmotionalTone.FIRM
            ],
            InteractionStyle.QUESTIONING: [
                EmotionalTone.CURIOUS, EmotionalTone.ENCOURAGING, EmotionalTone.PLAYFUL
            ],
            InteractionStyle.STORYTELLING: [
                EmotionalTone.EXCITED, EmotionalTone.PLAYFUL, EmotionalTone.CURIOUS
            ],
            InteractionStyle.RESPONSIVE: [
                EmotionalTone.ENCOURAGING, EmotionalTone.SOOTHING, EmotionalTone.CONCERNED
            ],
            InteractionStyle.CORRECTIVE: [
                EmotionalTone.FIRM, EmotionalTone.ENCOURAGING, EmotionalTone.SERIOUS
            ],
            InteractionStyle.REFLECTIVE: [
                EmotionalTone.CURIOUS, EmotionalTone.SERIOUS, EmotionalTone.ENCOURAGING
            ]
        }
        
        # Get appropriate tones or default to general purpose ones
        appropriate_tones = style_tone_map.get(
            style, 
            [EmotionalTone.ENCOURAGING, EmotionalTone.CURIOUS, EmotionalTone.SERIOUS]
        )
        
        return random.choice(appropriate_tones)
    
    def _select_complementary_tone(self, primary_tone: EmotionalTone, style: InteractionStyle) -> EmotionalTone:
        """
        Select a secondary emotional tone that complements the primary tone.
        
        Args:
            primary_tone: The primary emotional tone
            style: The interaction style
            
        Returns:
            A complementary secondary tone
        """
        # Tones that work well together
        complementary_tones = {
            EmotionalTone.SOOTHING: [EmotionalTone.ENCOURAGING, EmotionalTone.CONCERNED],
            EmotionalTone.ENCOURAGING: [EmotionalTone.EXCITED, EmotionalTone.PLAYFUL],
            EmotionalTone.PLAYFUL: [EmotionalTone.EXCITED, EmotionalTone.CURIOUS],
            EmotionalTone.CURIOUS: [EmotionalTone.ENCOURAGING, EmotionalTone.PLAYFUL],
            EmotionalTone.SERIOUS: [EmotionalTone.FIRM, EmotionalTone.CURIOUS],
            EmotionalTone.EXCITED: [EmotionalTone.ENCOURAGING, EmotionalTone.PLAYFUL],
            EmotionalTone.CONCERNED: [EmotionalTone.SOOTHING, EmotionalTone.ENCOURAGING],
            EmotionalTone.FIRM: [EmotionalTone.SERIOUS, EmotionalTone.ENCOURAGING]
        }
        
        # Get candidate secondary tones
        candidate_tones = complementary_tones.get(
            primary_tone, 
            [t for t in EmotionalTone if t != primary_tone]
        )
        
        # Filter out the primary tone to avoid duplication
        candidate_tones = [t for t in candidate_tones if t != primary_tone]
        
        # If no candidates remain, return a default alternative
        if not candidate_tones:
            all_tones = list(EmotionalTone)
            return random.choice([t for t in all_tones if t != primary_tone])
            
        return random.choice(candidate_tones)
    
    def _record_pattern(self, pattern: InteractionPattern) -> None:
        """
        Record an interaction pattern to avoid repetition.
        
        Args:
            pattern: The interaction pattern that was used
        """
        self._recent_patterns.append(pattern)
        
        # Keep only the most recent patterns
        if len(self._recent_patterns) > self._max_recent_patterns:
            self._recent_patterns.pop(0)
