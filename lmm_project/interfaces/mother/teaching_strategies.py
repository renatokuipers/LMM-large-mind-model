import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import random

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from .models import TeachingMethod, TeachingStrategy, TeachingProfile

# Initialize logger
logger = get_module_logger("interfaces.mother.teaching")

class TeachingStrategies:
    """
    Manages the Mother's teaching strategies, adapting approaches based on
    the child's developmental level and learning needs.
    """
    
    # Predefined teaching strategy presets
    PRESETS = {
        "balanced": [
            TeachingStrategy(method=TeachingMethod.DIRECT_INSTRUCTION, priority=5.0),
            TeachingStrategy(method=TeachingMethod.SCAFFOLDING, priority=6.0),
            TeachingStrategy(method=TeachingMethod.SOCRATIC_QUESTIONING, priority=5.0),
            TeachingStrategy(method=TeachingMethod.REPETITION, priority=4.0),
            TeachingStrategy(method=TeachingMethod.REINFORCEMENT, priority=6.0),
            TeachingStrategy(method=TeachingMethod.EXPERIENTIAL, priority=4.0),
            TeachingStrategy(method=TeachingMethod.DISCOVERY, priority=3.0),
            TeachingStrategy(method=TeachingMethod.ANALOGICAL, priority=4.0),
        ],
        "nurturing": [
            TeachingStrategy(method=TeachingMethod.SCAFFOLDING, priority=8.0),
            TeachingStrategy(method=TeachingMethod.REINFORCEMENT, priority=7.0),
            TeachingStrategy(method=TeachingMethod.REPETITION, priority=6.0),
            TeachingStrategy(method=TeachingMethod.DIRECT_INSTRUCTION, priority=4.0),
            TeachingStrategy(method=TeachingMethod.EXPERIENTIAL, priority=5.0),
            TeachingStrategy(method=TeachingMethod.SOCRATIC_QUESTIONING, priority=3.0),
            TeachingStrategy(method=TeachingMethod.DISCOVERY, priority=4.0),
            TeachingStrategy(method=TeachingMethod.ANALOGICAL, priority=5.0),
        ],
        "academic": [
            TeachingStrategy(method=TeachingMethod.DIRECT_INSTRUCTION, priority=8.0),
            TeachingStrategy(method=TeachingMethod.SOCRATIC_QUESTIONING, priority=7.0),
            TeachingStrategy(method=TeachingMethod.ANALOGICAL, priority=6.0),
            TeachingStrategy(method=TeachingMethod.REPETITION, priority=5.0),
            TeachingStrategy(method=TeachingMethod.SCAFFOLDING, priority=4.0),
            TeachingStrategy(method=TeachingMethod.REINFORCEMENT, priority=4.0),
            TeachingStrategy(method=TeachingMethod.DISCOVERY, priority=3.0),
            TeachingStrategy(method=TeachingMethod.EXPERIENTIAL, priority=2.0),
        ],
        "discovery": [
            TeachingStrategy(method=TeachingMethod.DISCOVERY, priority=8.0),
            TeachingStrategy(method=TeachingMethod.EXPERIENTIAL, priority=7.0),
            TeachingStrategy(method=TeachingMethod.SOCRATIC_QUESTIONING, priority=6.0),
            TeachingStrategy(method=TeachingMethod.SCAFFOLDING, priority=5.0),
            TeachingStrategy(method=TeachingMethod.ANALOGICAL, priority=4.0),
            TeachingStrategy(method=TeachingMethod.REINFORCEMENT, priority=4.0),
            TeachingStrategy(method=TeachingMethod.REPETITION, priority=2.0),
            TeachingStrategy(method=TeachingMethod.DIRECT_INSTRUCTION, priority=1.0),
        ]
    }
    
    # Developmental applicability maps - which strategies work best at different ages
    DEVELOPMENTAL_WEIGHTS = {
        TeachingMethod.DIRECT_INSTRUCTION: {
            0.0: 0.2,  # Less effective for very young minds
            0.5: 0.4,
            1.0: 0.6,
            2.0: 0.8,
            3.0: 1.0,  # Most effective for older, more developed minds
        },
        TeachingMethod.REPETITION: {
            0.0: 1.0,  # Very effective for young minds
            1.0: 0.9,
            2.0: 0.7,
            3.0: 0.5,  # Less necessary for older minds
        },
        TeachingMethod.SCAFFOLDING: {
            0.0: 0.7,  # Useful early but requires some minimal capacity
            0.5: 0.9,
            1.0: 1.0,  # Peak effectiveness during early language development
            2.0: 0.9,
            3.0: 0.8,  # Still valuable but less critical for advanced minds
        },
        TeachingMethod.SOCRATIC_QUESTIONING: {
            0.0: 0.1,  # Nearly ineffective for very young minds
            0.5: 0.3,
            1.0: 0.5,
            2.0: 0.8,
            3.0: 1.0,  # Ideal for developed minds capable of reflection
        },
        TeachingMethod.REINFORCEMENT: {
            0.0: 0.9,  # Highly effective from the beginning
            0.5: 1.0,
            1.0: 0.9,
            2.0: 0.8,
            3.0: 0.7,  # Still valuable but less critical as intrinsic motivation develops
        },
        TeachingMethod.EXPERIENTIAL: {
            0.0: 0.5,  # Moderately effective even early
            0.5: 0.7,
            1.0: 0.9,
            2.0: 1.0,  # Peak effectiveness during concrete operational stage
            3.0: 0.9,
        },
        TeachingMethod.DISCOVERY: {
            0.0: 0.2,  # Limited effectiveness for very young minds
            0.5: 0.4,
            1.0: 0.6,
            2.0: 0.8,
            3.0: 1.0,  # Most effective for minds capable of independent exploration
        },
        TeachingMethod.ANALOGICAL: {
            0.0: 0.1,  # Minimal effectiveness for very young minds
            0.5: 0.2,
            1.0: 0.5,
            2.0: 0.8,
            3.0: 1.0,  # Highly effective when abstract thinking is developed
        },
    }
    
    def __init__(self, preset: Optional[str] = None, profile: Optional[TeachingProfile] = None):
        """
        Initialize teaching strategies.
        
        Args:
            preset: Name of a predefined teaching preset
            profile: A custom teaching profile
        """
        self._config = get_config()
        
        # Initialize from provided profile, preset, or config
        if profile:
            self.profile = profile
        elif preset:
            self.load_preset(preset)
        else:
            # Try to load from config
            config_strategies = self._config.get_list("interfaces.mother.teaching_strategies", None)
            if config_strategies:
                self.create_profile_from_config(config_strategies)
            else:
                # Default to balanced preset
                self.load_preset("balanced")
                
        logger.info(f"Mother teaching strategies initialized: {self.profile.preset_name or 'custom'}")
    
    def load_preset(self, preset_name: str) -> None:
        """
        Load a predefined teaching strategies preset.
        
        Args:
            preset_name: Name of the preset to load
        """
        preset_name = preset_name.lower()
        if preset_name not in self.PRESETS:
            logger.warning(f"Unknown teaching preset '{preset_name}', falling back to 'balanced'")
            preset_name = "balanced"
            
        strategies = self.PRESETS[preset_name]
        self.profile = TeachingProfile(strategies=strategies, preset_name=preset_name)
    
    def create_profile_from_config(self, strategy_names: List[str]) -> None:
        """
        Create a teaching profile from strategy names in config.
        
        Args:
            strategy_names: List of strategy names
        """
        strategies = []
        
        for name in strategy_names:
            try:
                method = TeachingMethod(name)
                # Assign a default priority of 5.0
                strategies.append(TeachingStrategy(method=method, priority=5.0))
            except ValueError:
                logger.warning(f"Unknown teaching method '{name}', skipping")
                
        if not strategies:
            logger.warning("No valid teaching strategies found in config, using balanced preset")
            self.load_preset("balanced")
            return
            
        self.profile = TeachingProfile(strategies=strategies)
    
    def select_strategy(self, age: float, context: Dict[str, Any]) -> TeachingMethod:
        """
        Select the most appropriate teaching strategy based on current context.
        
        Args:
            age: Current developmental age of the mind
            context: Additional context about the learning situation
            
        Returns:
            The selected teaching method
        """
        # Get all strategies and their base priorities
        weighted_strategies = {
            strategy.method: strategy.priority 
            for strategy in self.profile.strategies
        }
        
        # Apply developmental adjustments
        for method, weights in self.DEVELOPMENTAL_WEIGHTS.items():
            if method in weighted_strategies:
                # Find the applicable age bracket
                applicable_ages = sorted(weights.keys())
                weight_adjustment = 1.0
                
                # Find the closest age brackets and interpolate
                for i, bracket_age in enumerate(applicable_ages):
                    if age <= bracket_age:
                        if i == 0:
                            weight_adjustment = weights[bracket_age]
                        else:
                            # Linear interpolation between brackets
                            prev_age = applicable_ages[i-1]
                            prev_weight = weights[prev_age]
                            curr_weight = weights[bracket_age]
                            
                            # Calculate interpolation factor
                            factor = (age - prev_age) / (bracket_age - prev_age)
                            weight_adjustment = prev_weight + factor * (curr_weight - prev_weight)
                        break
                else:
                    # If we're beyond the last bracket, use the last weight
                    weight_adjustment = weights[applicable_ages[-1]]
                
                # Apply the adjustment
                weighted_strategies[method] *= weight_adjustment
        
        # Context-based adjustments
        if context.get("attention_span_low", False):
            # Favor direct and experiential approaches when attention is low
            self._adjust_weight(weighted_strategies, TeachingMethod.DIRECT_INSTRUCTION, 1.5)
            self._adjust_weight(weighted_strategies, TeachingMethod.EXPERIENTIAL, 1.7)
            self._adjust_weight(weighted_strategies, TeachingMethod.SOCRATIC_QUESTIONING, 0.6)
            
        if context.get("confusion_level", 0) > 0.7:
            # When confusion is high, use more scaffolding and repetition
            self._adjust_weight(weighted_strategies, TeachingMethod.SCAFFOLDING, 1.8)
            self._adjust_weight(weighted_strategies, TeachingMethod.REPETITION, 1.6)
            self._adjust_weight(weighted_strategies, TeachingMethod.DISCOVERY, 0.5)
            
        if context.get("learning_area") == "abstract_concept":
            # For abstract concepts, favor analogical and socratic approaches
            self._adjust_weight(weighted_strategies, TeachingMethod.ANALOGICAL, 1.9)
            self._adjust_weight(weighted_strategies, TeachingMethod.SOCRATIC_QUESTIONING, 1.7)
        
        # Select strategy (weighted random selection)
        total_weight = sum(weighted_strategies.values())
        if total_weight <= 0:
            # Fallback if all weights are zero or negative
            return TeachingMethod.DIRECT_INSTRUCTION
            
        # Normalize weights to probabilities
        threshold = random.random() * total_weight
        current_sum = 0
        
        for method, weight in weighted_strategies.items():
            current_sum += weight
            if current_sum >= threshold:
                return method
                
        # Fallback in case of rounding errors
        return list(weighted_strategies.keys())[-1]
    
    def _adjust_weight(self, weights: Dict[TeachingMethod, float], method: TeachingMethod, factor: float) -> None:
        """Helper method to adjust weights for a specific teaching method"""
        if method in weights:
            weights[method] *= factor
