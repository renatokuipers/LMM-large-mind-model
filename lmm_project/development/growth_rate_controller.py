"""
Growth Rate Controller Module

This module manages the rate at which cognitive capabilities develop over time.
It implements various factors that influence development speed, including:
- Base growth rates for different capabilities
- Natural variation in development rates
- Developmental stage-appropriate rates
- Critical period accelerations
- Environmental influences
- Individual differences

The growth rate controller ensures realistic, variable development that follows
patterns similar to human cognitive development.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import random
import math
from datetime import datetime

from lmm_project.development.models import GrowthRateParameters, DevelopmentalEvent

logger = logging.getLogger(__name__)

class GrowthRateController:
    """
    Controls the rate at which different capabilities develop
    
    This class manages development speed with natural variation,
    critical period effects, and other influences on growth rates.
    """
    
    def __init__(self, 
                 base_rate: float = 0.01, 
                 variability: float = 0.2):
        """
        Initialize the growth rate controller
        
        Args:
            base_rate: The default growth rate for capabilities
            variability: How much natural variation to apply (0.0-1.0)
        """
        self.parameters = GrowthRateParameters(
            base_rate=base_rate,
            variability=variability,
            # Standard factors that can accelerate development
            acceleration_factors={
                "critical_period": 2.0,      # During critical periods
                "focused_learning": 1.5,     # When specifically focused on learning
                "emotional_engagement": 1.3, # When emotionally engaged
                "repetition": 1.2,           # With repetitive practice
                "mother_guidance": 1.4       # When guided by the Mother
            },
            # Standard factors that can slow development
            inhibition_factors={
                "cognitive_overload": 0.7,   # When cognitive load is too high
                "emotional_distress": 0.6,   # During emotional distress
                "attention_deficit": 0.8,    # When attention is divided
                "conflicting_inputs": 0.7,   # When receiving conflicting information
                "developmental_plateau": 0.5 # During developmental plateaus
            },
            # How age affects development rate (age, multiplier)
            age_modifiers=[
                (0.0, 1.5),    # Early development is faster
                (1.0, 1.3),    # Still accelerated in infancy
                (3.0, 1.0),    # Normal rate in early childhood
                (7.0, 0.8),    # Slows in middle childhood
                (14.0, 0.7),   # Slower in adolescence
                (21.0, 0.5)    # Much slower in adulthood
            ]
        )
        
        # Default capability-specific base rates
        self.capability_base_rates = {
            # Core capabilities
            "neural_formation": 0.012,
            "pattern_recognition": 0.015,
            "sensory_processing": 0.014,
            "association_formation": 0.013,
            
            # Cognitive capabilities
            "attention": 0.011,
            "working_memory": 0.010,
            "episodic_memory": 0.009,
            "semantic_memory": 0.008,
            "logical_reasoning": 0.007,
            "abstract_thinking": 0.006,
            "metacognition": 0.005,
            
            # Linguistic capabilities
            "language_comprehension": 0.013,
            "language_production": 0.011,
            
            # Emotional capabilities
            "emotional_response": 0.014,
            "emotional_understanding": 0.009,
            "emotional_regulation": 0.007,
            
            # Social capabilities
            "self_awareness": 0.008,
            "identity_formation": 0.006,
            "social_understanding": 0.009,
            "moral_reasoning": 0.007,
            
            # Creative capabilities
            "creativity": 0.008,
            "imagination": 0.010,
            
            # Advanced capabilities
            "wisdom": 0.004
        }
        
        # Module-specific growth rates
        self.module_growth_rates = {
            "perception": 1.2,
            "attention": 1.1,
            "memory": 1.0,
            "language": 1.1,
            "emotion": 1.1,
            "consciousness": 0.9,
            "executive": 0.9,
            "social": 1.0,
            "temporal": 0.9,
            "creativity": 1.0,
            "self_regulation": 0.9,
            "learning": 1.0,
            "identity": 0.8,
            "belief": 0.9
        }
        
        # Natural growth variation over time
        self.growth_cycle_state = {}
        # Initialize random phases for cyclical development
        for capability in self.capability_base_rates:
            self.growth_cycle_state[capability] = {
                "phase": random.random() * 2 * math.pi,
                "frequency": 0.1 + random.random() * 0.2  # Cycles per time unit
            }
    
    def get_growth_rate(self, 
                        capability: str, 
                        module: str, 
                        age: float,
                        active_factors: Dict[str, bool] = None,
                        critical_period_multiplier: float = 1.0) -> float:
        """
        Calculate the growth rate for a specific capability
        
        Args:
            capability: The capability to calculate growth for
            module: The module this capability belongs to
            age: Current developmental age
            active_factors: Dict of factors that are currently active
            critical_period_multiplier: Multiplier from any active critical periods
            
        Returns:
            The growth rate to apply
        """
        # Start with the base rate for this capability
        base_rate = self.capability_base_rates.get(capability, self.parameters.base_rate)
        
        # Apply module-specific adjustments
        module_multiplier = self.module_growth_rates.get(module, 1.0)
        rate = base_rate * module_multiplier
        
        # Apply age modifiers
        age_multiplier = self._get_age_multiplier(age)
        rate *= age_multiplier
        
        # Apply natural cyclical variation
        cycle_multiplier = self._get_cycle_multiplier(capability, age)
        rate *= cycle_multiplier
        
        # Apply critical period effects
        rate *= critical_period_multiplier
        
        # Apply active factors
        if active_factors:
            factor_multiplier = self._calculate_factor_multiplier(active_factors)
            rate *= factor_multiplier
        
        # Apply random variation
        if self.parameters.variability > 0:
            variation = 1.0 + (random.random() * 2 - 1) * self.parameters.variability
            rate *= variation
            
        return max(0.001, rate)  # Ensure minimum growth rate
    
    def _get_age_multiplier(self, age: float) -> float:
        """Get the age-based multiplier for the given developmental age"""
        if not self.parameters.age_modifiers:
            return 1.0
            
        # Find the appropriate age bracket
        for i, (bracket_age, multiplier) in enumerate(sorted(self.parameters.age_modifiers)):
            if age < bracket_age:
                if i == 0:
                    return multiplier
                
                # Interpolate between brackets
                prev_age, prev_mult = self.parameters.age_modifiers[i-1]
                weight = (age - prev_age) / (bracket_age - prev_age)
                return prev_mult + weight * (multiplier - prev_mult)
        
        # If beyond the last bracket, use the last multiplier
        return self.parameters.age_modifiers[-1][1]
    
    def _get_cycle_multiplier(self, capability: str, age: float) -> float:
        """
        Calculate natural cyclical variations in growth rate
        
        This simulates natural periods of faster and slower growth
        that occur during development.
        """
        if capability not in self.growth_cycle_state:
            return 1.0
            
        state = self.growth_cycle_state[capability]
        phase = state["phase"]
        frequency = state["frequency"]
        
        # Calculate position in the cycle based on age
        cycle_position = phase + (age * frequency * 2 * math.pi)
        
        # Sinusoidal variation centered on 1.0 with +/- 20% variation
        return 1.0 + 0.2 * math.sin(cycle_position)
    
    def _calculate_factor_multiplier(self, active_factors: Dict[str, bool]) -> float:
        """
        Calculate the combined effect of all active factors
        
        Args:
            active_factors: Dictionary of factor name -> is_active
            
        Returns:
            Combined multiplier for all active factors
        """
        combined_multiplier = 1.0
        
        for factor, is_active in active_factors.items():
            if not is_active:
                continue
                
            # Check if it's an acceleration factor
            if factor in self.parameters.acceleration_factors:
                combined_multiplier *= self.parameters.acceleration_factors[factor]
                
            # Check if it's an inhibition factor
            elif factor in self.parameters.inhibition_factors:
                combined_multiplier *= self.parameters.inhibition_factors[factor]
        
        return combined_multiplier
    
    def register_acceleration_factor(self, name: str, multiplier: float) -> None:
        """Register a new acceleration factor"""
        self.parameters.acceleration_factors[name] = max(1.0, multiplier)
        logger.info(f"Registered acceleration factor: {name} with multiplier {multiplier}")
    
    def register_inhibition_factor(self, name: str, multiplier: float) -> None:
        """Register a new inhibition factor"""
        self.parameters.inhibition_factors[name] = min(1.0, max(0.1, multiplier))
        logger.info(f"Registered inhibition factor: {name} with multiplier {multiplier}")
    
    def update_capability_base_rate(self, capability: str, base_rate: float) -> None:
        """Update the base rate for a specific capability"""
        self.capability_base_rates[capability] = max(0.001, base_rate)
        logger.info(f"Updated base rate for capability {capability}: {base_rate}")
    
    def update_module_growth_rate(self, module: str, multiplier: float) -> None:
        """Update the growth rate multiplier for a specific module"""
        self.module_growth_rates[module] = max(0.1, multiplier)
        logger.info(f"Updated growth rate for module {module}: {multiplier}")
    
    def detect_developmental_plateau(self, 
                                    capability: str, 
                                    recent_growth: List[float],
                                    threshold: float = 0.01) -> bool:
        """
        Detect if development has plateaued for a capability
        
        Args:
            capability: The capability to check
            recent_growth: List of recent growth increments
            threshold: Minimum growth rate to avoid plateau detection
            
        Returns:
            True if a plateau is detected, False otherwise
        """
        if len(recent_growth) < 5:
            return False
            
        # Calculate average recent growth
        avg_growth = sum(recent_growth) / len(recent_growth)
        
        # Check if growth has slowed below threshold
        return avg_growth < threshold
    
    def get_plateau_intervention(self, 
                               capability: str, 
                               module: str, 
                               age: float) -> Dict[str, Any]:
        """
        Generate intervention recommendations for breaking through a plateau
        
        Args:
            capability: The plateaued capability
            module: The module this capability belongs to
            age: Current developmental age
            
        Returns:
            Dictionary with intervention recommendations
        """
        interventions = {
            "change_learning_approach": True,
            "increase_challenge_level": True,
            "introduce_novel_stimuli": True,
            "focus_on_prerequisites": False,
            "recommended_experiences": []
        }
        
        # Recommended experiences depend on capability type
        if "language" in capability:
            interventions["recommended_experiences"] = [
                "vocabulary_expansion_exercises",
                "complex_grammar_exposure",
                "creative_language_use"
            ]
        elif "memory" in capability:
            interventions["recommended_experiences"] = [
                "memory_games",
                "association_exercises",
                "retrieval_practice"
            ]
        elif "emotion" in capability:
            interventions["recommended_experiences"] = [
                "emotional_scenario_exploration",
                "emotional_vocabulary_building",
                "self-regulation_practice"
            ]
        # Add more capability-specific recommendations as needed
        
        return interventions
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the growth rate controller"""
        return {
            "parameters": self.parameters.dict(),
            "capability_base_rates": self.capability_base_rates.copy(),
            "module_growth_rates": self.module_growth_rates.copy(),
            "growth_cycle_state": self.growth_cycle_state.copy()
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state"""
        if "parameters" in state:
            self.parameters = GrowthRateParameters(**state["parameters"])
            
        if "capability_base_rates" in state:
            self.capability_base_rates = state["capability_base_rates"]
            
        if "module_growth_rates" in state:
            self.module_growth_rates = state["module_growth_rates"]
            
        if "growth_cycle_state" in state:
            self.growth_cycle_state = state["growth_cycle_state"] 
