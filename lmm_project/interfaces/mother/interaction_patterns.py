import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import random
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
        # Find the appropriate complexity range
        age_brackets = sorted(self.COMPLEXITY_SCALING.keys())
        
        # Handle case before first bracket
        if age <= age_brackets[0]:
            return self.COMPLEXITY_SCALING[age_brackets[0]]
        
        # Linear interpolation between brackets
        for i, bracket_age in enumerate(age_brackets[1:], 1):
            if age <= bracket_age:
                prev_age = age_brackets[i-1]
                prev_complexity = self.COMPLEXITY_SCALING[prev_age]
                curr_complexity = self.COMPLEXITY_SCALING[bracket_age]
                
                # Calculate interpolation factor
                factor = (age - prev_age) / (bracket_age - prev_age)
                
                # Interpolate complexity
                return prev_complexity + factor * (curr_complexity - prev_complexity)
        
        # Beyond the last bracket - use maximum complexity
        return self.COMPLEXITY_SCALING[age_brackets[-1]]
    
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
        # Determine complexity level
        complexity = self.determine_complexity(age)
        
        # Adjust complexity based on context
        if context.get("simplified_language", False):
            complexity *= 0.7
        if context.get("challenging_concept", False):
            complexity *= 0.8
        if context.get("receptive_state", False):
            complexity *= 1.2
            
        # Determine primary interaction style
        primary_style = self._determine_primary_style(age, teaching_method, context)
        
        # Ensure we have an emotional tone
        if emotional_tone is None:
            emotional_tone = self._select_default_tone(primary_style)
        
        # Create the interaction pattern
        pattern = InteractionPattern(
            style=primary_style,
            primary_tone=emotional_tone,
            complexity_level=min(10.0, max(0.1, complexity))  # Ensure within bounds
        )
        
        # Add secondary tone for some interactions
        if random.random() < 0.3:  # 30% chance of secondary tone
            pattern.secondary_tone = self._select_complementary_tone(
                pattern.primary_tone, 
                primary_style
            )
            
        # Record this pattern to avoid repetition
        self._record_pattern(pattern)
        
        # Update interaction timing
        self._last_interaction_time = datetime.now()
        
        return pattern
        
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
        # Get candidate styles based on age
        primary_candidates = []
        secondary_candidates = []
        
        for (min_age, max_age), (primary, secondary) in self.AGE_BASED_DEFAULTS.items():
            if min_age <= age <= max_age:
                primary_candidates.append(primary)
                secondary_candidates.append(secondary)
                
        # If we have a teaching method, consider its preferred style
        if teaching_method and teaching_method in self.TEACHING_STYLE_MAP:
            preferred_style = self.TEACHING_STYLE_MAP[teaching_method]
            primary_candidates.append(preferred_style)
            
        # Special context handling
        if context.get("correction_needed", False):
            primary_candidates.append(InteractionStyle.CORRECTIVE)
            
        if context.get("storytelling_moment", False):
            primary_candidates.append(InteractionStyle.STORYTELLING)
            
        if context.get("questions_encouraged", False):
            primary_candidates.append(InteractionStyle.QUESTIONING)
            
        # Avoid recently used patterns for variety
        recent_styles = {p.style for p in self._recent_patterns}
        fresh_candidates = [s for s in primary_candidates if s not in recent_styles]
        
        # Select from fresh candidates if possible, otherwise from all candidates
        if fresh_candidates:
            return random.choice(fresh_candidates)
        elif primary_candidates:
            return random.choice(primary_candidates)
        elif secondary_candidates:
            return random.choice(secondary_candidates)
        else:
            # Fallback to conversational
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
