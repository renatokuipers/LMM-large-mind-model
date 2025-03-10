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

from typing import Dict, List, Optional, Any, Tuple, Set, Union
import logging
import random
import math
import threading
import json
import os
from datetime import datetime
from pathlib import Path
import traceback

from lmm_project.development.models import GrowthRateParameters, DevelopmentalEvent
from lmm_project.core.exceptions import DevelopmentError

logger = logging.getLogger(__name__)

class GrowthRateController:
    """
    Controls the rate at which different capabilities develop
    
    This class manages development speed with natural variation,
    critical period effects, and other influences on growth rates.
    
    Features:
    - Thread-safe growth rate calculations
    - Caching of frequently accessed values
    - Adaptive growth rate adjustments
    - Plateau detection and intervention
    - Persistence of growth rate parameters
    """
    
    def __init__(self, 
                 base_rate: float = 0.01, 
                 variability: float = 0.2):
        """
        Initialize the growth rate controller
        
        Args:
            base_rate: The default growth rate for capabilities (0.0-1.0)
            variability: How much natural variation to apply (0.0-1.0)
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        if not 0.0 <= base_rate <= 1.0:
            raise ValueError("base_rate must be between 0.0 and 1.0")
            
        if not 0.0 <= variability <= 1.0:
            raise ValueError("variability must be between 0.0 and 1.0")
            
        self._lock = threading.RLock()
        
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
            
        # Cache for frequently calculated values
        self._cache = {
            "age_multipliers": {},  # age -> multiplier
            "cycle_multipliers": {},  # (capability, age) -> multiplier
            "last_cache_update": datetime.now()
        }
        self._cache_ttl = 5.0  # Cache time-to-live in seconds
        
        # Growth history for plateau detection
        self.growth_history = {}  # capability -> List[growth_rates]
        self.max_history_length = 20  # Maximum number of historical growth rates to keep
        
        # Adaptive growth parameters
        self.adaptive_adjustments = {}  # capability -> adjustment_factor
        
        logger.info("Growth rate controller initialized with base rate %.3f", base_rate)
    
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
            The growth rate to apply (0.0-1.0)
            
        Raises:
            ValueError: If parameters are invalid
            DevelopmentError: If growth rate calculation fails
        """
        if age < 0:
            raise ValueError("Age cannot be negative")
            
        if critical_period_multiplier < 0:
            raise ValueError("Critical period multiplier cannot be negative")
            
        try:
            with self._lock:
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
                
                # Apply adaptive adjustments if any
                if capability in self.adaptive_adjustments:
                    rate *= self.adaptive_adjustments[capability]
                
                # Apply random variation
                if self.parameters.variability > 0:
                    variation = 1.0 + (random.random() * 2 - 1) * self.parameters.variability
                    rate *= variation
                    
                # Ensure rate is within valid bounds
                rate = max(0.001, min(1.0, rate))
                
                # Record growth rate in history for plateau detection
                self._record_growth_rate(capability, rate)
                
                return rate
                
        except Exception as e:
            error_msg = f"Failed to calculate growth rate for {capability} in {module}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={
                "capability": capability,
                "module": module,
                "age": age,
                "original_error": str(e)
            })
    
    def _get_age_multiplier(self, age: float) -> float:
        """
        Get the age-based multiplier for the given developmental age
        
        Args:
            age: The developmental age
            
        Returns:
            The age-based multiplier
        """
        # Check cache first
        cache_key = round(age, 2)  # Round to 2 decimal places for cache
        if cache_key in self._cache["age_multipliers"] and self._is_cache_valid():
            return self._cache["age_multipliers"][cache_key]
            
        if not self.parameters.age_modifiers:
            return 1.0
            
        # Find the appropriate age bracket
        for i, (bracket_age, multiplier) in enumerate(sorted(self.parameters.age_modifiers)):
            if age < bracket_age:
                if i == 0:
                    result = multiplier
                    break
                
                # Interpolate between brackets
                prev_age, prev_mult = self.parameters.age_modifiers[i-1]
                weight = (age - prev_age) / (bracket_age - prev_age)
                result = prev_mult + weight * (multiplier - prev_mult)
                break
        else:
            # If beyond the last bracket, use the last multiplier
            result = self.parameters.age_modifiers[-1][1]
        
        # Cache the result
        self._cache["age_multipliers"][cache_key] = result
        return result
    
    def _get_cycle_multiplier(self, capability: str, age: float) -> float:
        """
        Calculate natural cyclical variations in growth rate
        
        This simulates natural periods of faster and slower growth
        that occur during development.
        
        Args:
            capability: The capability to calculate for
            age: The developmental age
            
        Returns:
            The cycle-based multiplier
        """
        # Check cache first
        cache_key = (capability, round(age, 2))  # Round to 2 decimal places for cache
        if cache_key in self._cache["cycle_multipliers"] and self._is_cache_valid():
            return self._cache["cycle_multipliers"][cache_key]
            
        if capability not in self.growth_cycle_state:
            return 1.0
            
        state = self.growth_cycle_state[capability]
        phase = state["phase"]
        frequency = state["frequency"]
        
        # Calculate position in the cycle based on age
        cycle_position = phase + (age * frequency * 2 * math.pi)
        
        # Sinusoidal variation centered on 1.0 with +/- 20% variation
        result = 1.0 + 0.2 * math.sin(cycle_position)
        
        # Cache the result
        self._cache["cycle_multipliers"][cache_key] = result
        return result
    
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
        """
        Register a new acceleration factor
        
        Args:
            name: The name of the factor
            multiplier: The multiplier to apply (must be > 1.0)
            
        Raises:
            ValueError: If multiplier is invalid
        """
        if multiplier <= 1.0:
            raise ValueError(f"Acceleration factor multiplier must be greater than 1.0, got {multiplier}")
            
        with self._lock:
            self.parameters.acceleration_factors[name] = multiplier
            logger.info(f"Registered acceleration factor '{name}' with multiplier {multiplier}")
    
    def register_inhibition_factor(self, name: str, multiplier: float) -> None:
        """
        Register a new inhibition factor
        
        Args:
            name: The name of the factor
            multiplier: The multiplier to apply (must be < 1.0 and > 0)
            
        Raises:
            ValueError: If multiplier is invalid
        """
        if not 0.0 < multiplier < 1.0:
            raise ValueError(f"Inhibition factor multiplier must be between 0.0 and 1.0, got {multiplier}")
            
        with self._lock:
            self.parameters.inhibition_factors[name] = multiplier
            logger.info(f"Registered inhibition factor '{name}' with multiplier {multiplier}")
    
    def update_capability_base_rate(self, capability: str, base_rate: float) -> None:
        """
        Update the base growth rate for a specific capability
        
        Args:
            capability: The capability to update
            base_rate: The new base rate (0.0-1.0)
            
        Raises:
            ValueError: If base_rate is outside valid range
        """
        if not 0.0 <= base_rate <= 1.0:
            raise ValueError(f"Base rate must be between 0.0 and 1.0, got {base_rate}")
            
        with self._lock:
            self.capability_base_rates[capability] = base_rate
            logger.info(f"Updated base rate for '{capability}' to {base_rate}")
    
    def update_module_growth_rate(self, module: str, multiplier: float) -> None:
        """
        Update the growth rate multiplier for a specific module
        
        Args:
            module: The module to update
            multiplier: The new multiplier (must be positive)
            
        Raises:
            ValueError: If multiplier is invalid
        """
        if multiplier <= 0:
            raise ValueError(f"Module growth rate multiplier must be positive, got {multiplier}")
            
        with self._lock:
            self.module_growth_rates[module] = multiplier
            logger.info(f"Updated growth rate multiplier for '{module}' to {multiplier}")
    
    def detect_developmental_plateau(self, 
                                    capability: str, 
                                    recent_growth: List[float],
                                    threshold: float = 0.01) -> bool:
        """
        Detect if development has plateaued for a capability
        
        Args:
            capability: The capability to check
            recent_growth: List of recent growth amounts
            threshold: Threshold for plateau detection
            
        Returns:
            True if a plateau is detected, False otherwise
        """
        if len(recent_growth) < 5:
            return False  # Need at least 5 data points
            
        # Calculate average growth
        avg_growth = sum(recent_growth) / len(recent_growth)
        
        # Calculate variance
        variance = sum((g - avg_growth) ** 2 for g in recent_growth) / len(recent_growth)
        std_dev = variance ** 0.5
        
        # Check if growth is consistently low with low variance
        return avg_growth < threshold and std_dev < threshold / 2
    
    def get_plateau_intervention(self, 
                               capability: str, 
                               module: str, 
                               age: float) -> Dict[str, Any]:
        """
        Get intervention recommendations for a developmental plateau
        
        Args:
            capability: The capability that has plateaued
            module: The module this capability belongs to
            age: Current developmental age
            
        Returns:
            Dictionary with intervention recommendations
        """
        with self._lock:
            # Check if we have growth history for this capability
            if capability not in self.growth_history or len(self.growth_history[capability]) < 5:
                return {"has_plateau": False}
                
            # Check for plateau
            recent_growth = self.growth_history[capability][-5:]
            has_plateau = self.detect_developmental_plateau(capability, recent_growth)
            
            if not has_plateau:
                return {"has_plateau": False}
                
            # Generate intervention recommendations
            interventions = []
            
            # Recommend focused learning
            interventions.append({
                "type": "focused_learning",
                "description": f"Focus specifically on {capability} development",
                "expected_impact": "high"
            })
            
            # Recommend new experiences
            interventions.append({
                "type": "new_experiences",
                "description": f"Introduce novel stimuli related to {capability}",
                "expected_impact": "medium"
            })
            
            # Recommend temporary growth rate boost
            if capability in self.adaptive_adjustments:
                current_adjustment = self.adaptive_adjustments[capability]
                new_adjustment = min(2.0, current_adjustment * 1.2)  # Increase by 20%, max 2x
            else:
                new_adjustment = 1.2  # Start with 20% boost
                
            self.adaptive_adjustments[capability] = new_adjustment
            
            interventions.append({
                "type": "growth_rate_adjustment",
                "description": f"Temporary growth rate boost applied: {new_adjustment:.2f}x",
                "expected_impact": "high",
                "automatic": True
            })
            
            return {
                "has_plateau": True,
                "capability": capability,
                "module": module,
                "age": age,
                "recent_growth_rates": recent_growth,
                "interventions": interventions
            }
    
    def _record_growth_rate(self, capability: str, rate: float) -> None:
        """
        Record a growth rate in history for plateau detection
        
        Args:
            capability: The capability
            rate: The growth rate
        """
        if capability not in self.growth_history:
            self.growth_history[capability] = []
            
        history = self.growth_history[capability]
        history.append(rate)
        
        # Limit history size
        if len(history) > self.max_history_length:
            self.growth_history[capability] = history[-self.max_history_length:]
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        time_diff = (datetime.now() - self._cache["last_cache_update"]).total_seconds()
        return time_diff < self._cache_ttl
    
    def _invalidate_cache(self) -> None:
        """Invalidate the cache"""
        self._cache["age_multipliers"] = {}
        self._cache["cycle_multipliers"] = {}
        self._cache["last_cache_update"] = datetime.min
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state for persistence
        
        Returns:
            Dictionary with the complete state
        """
        with self._lock:
            return {
                "parameters": self.parameters.to_dict(),
                "capability_base_rates": self.capability_base_rates,
                "module_growth_rates": self.module_growth_rates,
                "growth_cycle_state": self.growth_cycle_state,
                "adaptive_adjustments": self.adaptive_adjustments
            }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state dictionary to load
            
        Raises:
            ValueError: If state dictionary is invalid
        """
        if not state:
            raise ValueError("State dictionary cannot be empty")
            
        with self._lock:
            if "parameters" in state:
                params = state["parameters"]
                self.parameters = GrowthRateParameters(
                    base_rate=params.get("base_rate", 0.01),
                    variability=params.get("variability", 0.2),
                    acceleration_factors=params.get("acceleration_factors", {}),
                    inhibition_factors=params.get("inhibition_factors", {}),
                    age_modifiers=params.get("age_modifiers", [])
                )
                
            if "capability_base_rates" in state:
                self.capability_base_rates = state["capability_base_rates"]
                
            if "module_growth_rates" in state:
                self.module_growth_rates = state["module_growth_rates"]
                
            if "growth_cycle_state" in state:
                self.growth_cycle_state = state["growth_cycle_state"]
                
            if "adaptive_adjustments" in state:
                self.adaptive_adjustments = state["adaptive_adjustments"]
                
            # Invalidate cache after state load
            self._invalidate_cache()
            
            logger.info("Growth rate controller state loaded")
    
    def save_state_to_file(self, filepath: str) -> None:
        """
        Save the current state to a file
        
        Args:
            filepath: Path to save the state file
            
        Raises:
            IOError: If file cannot be written
        """
        state = self.get_state()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        logger.info(f"Growth rate controller state saved to {filepath}")
    
    def load_state_from_file(self, filepath: str) -> None:
        """
        Load state from a file
        
        Args:
            filepath: Path to the state file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If file cannot be read
            ValueError: If file contains invalid state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.load_state(state)
        logger.info(f"Growth rate controller state loaded from {filepath}")
    
    def reset_adaptive_adjustments(self) -> None:
        """Reset all adaptive growth rate adjustments"""
        with self._lock:
            self.adaptive_adjustments = {}
            logger.info("Adaptive growth rate adjustments reset")
    
    def get_capability_growth_history(self, capability: str) -> List[float]:
        """
        Get the growth history for a specific capability
        
        Args:
            capability: The capability to get history for
            
        Returns:
            List of recent growth rates, or empty list if no history
        """
        with self._lock:
            return self.growth_history.get(capability, [])[:]  # Return a copy 
