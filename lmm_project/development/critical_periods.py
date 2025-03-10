"""
Critical periods management for the LMM project.

This module implements a system for managing critical periods - specific
developmental windows where certain capabilities are more readily learned,
modeled after similar periods in human development.
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple

import numpy as np

from lmm_project.core.event_system import EventSystem, Event
from lmm_project.core.types import StateDict
from lmm_project.development.models import (
    CriticalPeriodDefinition,
    CriticalPeriodType,
    DevelopmentConfig
)
from lmm_project.development.developmental_stages import DevelopmentalStages
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

class CriticalPeriods:
    """
    Manages critical periods for enhanced learning of specific capabilities.
    
    Critical periods are specific age ranges where certain capabilities are 
    more readily learned, similar to sensitive periods in human development.
    """
    
    def __init__(
        self, 
        dev_stages: DevelopmentalStages,
        config: Optional[DevelopmentConfig] = None
    ):
        """
        Initialize the critical periods manager.
        
        Parameters:
        -----------
        dev_stages : DevelopmentalStages
            Reference to the developmental stages manager for age tracking
        config : Optional[DevelopmentConfig]
            Configuration containing critical period definitions. If None, default settings will be loaded.
        """
        self.event_system = EventSystem()
        self.dev_stages = dev_stages
        self._config = config or self._load_default_config()
        
        # Store critical period definitions for easy lookup
        self._critical_periods: List[CriticalPeriodDefinition] = self._config.critical_period_definitions
        
        # Track active critical periods
        self._active_periods: Dict[str, CriticalPeriodDefinition] = {}
        
        # Track history of critical periods
        self._period_history: List[Dict[str, Any]] = []
        
        # Track when critical periods have been activated and ended
        self._period_activation_history: Dict[str, List[Tuple[datetime, float, str]]] = {}
        
        # Last update time
        self._last_update_time = time.time()
        
        logger.info(f"Critical periods system initialized with {len(self._critical_periods)} defined periods")
    
    def _load_default_config(self) -> DevelopmentConfig:
        """
        Load default configuration for critical periods.
        
        Returns:
        --------
        DevelopmentConfig
            Default configuration
        """
        # Define some basic critical periods based on developmental psychology
        critical_periods = [
            CriticalPeriodDefinition(
                id="language_acquisition",
                name="Language Acquisition Period",
                period_type=CriticalPeriodType.LANGUAGE,
                description="Period of enhanced language learning ability",
                begin_age=0.2,
                end_age=2.0,
                learning_multiplier=2.5,
                affected_modules=["language", "communication"],
                affected_capabilities=["vocabulary", "grammar", "syntax"]
            ),
            CriticalPeriodDefinition(
                id="sensory_integration",
                name="Sensory Integration Period",
                period_type=CriticalPeriodType.SENSORY,
                description="Period when sensory systems are calibrated and integrated",
                begin_age=0.0,
                end_age=0.5,
                learning_multiplier=2.0,
                affected_modules=["perception", "attention"],
                affected_capabilities=["visual_processing", "auditory_processing", "sensory_fusion"]
            ),
            CriticalPeriodDefinition(
                id="social_bonding",
                name="Social Bonding Period",
                period_type=CriticalPeriodType.SOCIAL,
                description="Period of enhanced social relationship formation",
                begin_age=0.3,
                end_age=1.5,
                learning_multiplier=2.2,
                affected_modules=["social", "emotional"],
                affected_capabilities=["attachment", "empathy", "social_norms"]
            ),
            CriticalPeriodDefinition(
                id="abstract_reasoning",
                name="Abstract Reasoning Development",
                period_type=CriticalPeriodType.COGNITIVE,
                description="Period when abstract thinking abilities develop rapidly",
                begin_age=2.0,
                end_age=4.0,
                learning_multiplier=1.8,
                affected_modules=["reasoning", "metacognition"],
                affected_capabilities=["logical_reasoning", "abstract_concepts", "hypothesis_testing"]
            ),
            CriticalPeriodDefinition(
                id="emotional_regulation",
                name="Emotional Regulation Development",
                period_type=CriticalPeriodType.EMOTIONAL,
                description="Period when emotional regulation capabilities develop",
                begin_age=1.0,
                end_age=3.0,
                learning_multiplier=1.7,
                affected_modules=["emotional", "executive"],
                affected_capabilities=["emotion_recognition", "self_regulation", "impulse_control"]
            )
        ]
        
        # Return minimal config with just critical periods
        return DevelopmentConfig(
            initial_age=0.0,
            time_acceleration=1000.0,
            stage_definitions=[],
            milestone_definitions=[],
            critical_period_definitions=critical_periods
        )
    
    def update(self) -> None:
        """
        Update the critical periods based on current developmental age.
        
        Activates or deactivates critical periods based on age thresholds.
        """
        current_time = time.time()
        current_age = self.dev_stages.get_age()
        
        # Check all defined critical periods
        for period in self._critical_periods:
            period_id = period.id
            in_period = period.begin_age <= current_age < period.end_age
            
            # If period should be active but isn't currently active
            if in_period and period_id not in self._active_periods:
                self._active_periods[period_id] = period
                
                # Record activation in history
                activation_record = (datetime.now(), current_age, "activated")
                if period_id not in self._period_activation_history:
                    self._period_activation_history[period_id] = []
                self._period_activation_history[period_id].append(activation_record)
                
                # Emit event for period activation
                self.event_system.emit(Event(
                    name="critical_period_activated",
                    data={
                        "period_id": period_id,
                        "period_name": period.name,
                        "period_type": period.period_type,
                        "age": current_age,
                        "learning_multiplier": period.learning_multiplier,
                        "affected_modules": period.affected_modules,
                        "affected_capabilities": period.affected_capabilities
                    }
                ))
                
                logger.info(f"Critical period activated: {period.name} at age {current_age:.3f}")
            
            # If period should not be active but is currently active
            elif not in_period and period_id in self._active_periods:
                # Remove from active periods
                del self._active_periods[period_id]
                
                # Record deactivation in history
                deactivation_record = (datetime.now(), current_age, "deactivated")
                if period_id not in self._period_activation_history:
                    self._period_activation_history[period_id] = []
                self._period_activation_history[period_id].append(deactivation_record)
                
                # Emit event for period deactivation
                self.event_system.emit(Event(
                    name="critical_period_deactivated",
                    data={
                        "period_id": period_id,
                        "period_name": period.name,
                        "period_type": period.period_type,
                        "age": current_age
                    }
                ))
                
                logger.info(f"Critical period deactivated: {period.name} at age {current_age:.3f}")
        
        self._last_update_time = current_time
    
    def is_active(self, period_id: str) -> bool:
        """
        Check if a specific critical period is currently active.
        
        Parameters:
        -----------
        period_id : str
            ID of the critical period to check
            
        Returns:
        --------
        bool
            True if the period is active, False otherwise
        """
        return period_id in self._active_periods
    
    def get_active_periods(self) -> Dict[str, CriticalPeriodDefinition]:
        """
        Get all currently active critical periods.
        
        Returns:
        --------
        Dict[str, CriticalPeriodDefinition]
            Dictionary of active critical periods, keyed by period ID
        """
        return self._active_periods.copy()
    
    def get_learning_multiplier(
        self, 
        module_name: str = None, 
        capability: str = None
    ) -> float:
        """
        Get the learning rate multiplier for a specific module or capability.
        
        If both module_name and capability are specified, returns the highest
        multiplier that applies to both.
        
        Parameters:
        -----------
        module_name : Optional[str]
            Module name to check for multipliers
        capability : Optional[str]
            Capability to check for multipliers
            
        Returns:
        --------
        float
            Learning rate multiplier (default 1.0 if no active periods apply)
        """
        # If neither is specified, return default multiplier
        if module_name is None and capability is None:
            return 1.0
            
        multipliers = []
        
        # Check all active periods
        for period in self._active_periods.values():
            applies_to_module = module_name is None or module_name in period.affected_modules
            applies_to_capability = capability is None or capability in period.affected_capabilities
            
            # If period applies to both (or to the one that was specified), add its multiplier
            if (module_name is None or applies_to_module) and (capability is None or applies_to_capability):
                multipliers.append(period.learning_multiplier)
        
        # Return highest multiplier, or default 1.0 if none apply
        return max(multipliers) if multipliers else 1.0
    
    def get_upcoming_periods(self, lookahead_age: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get list of upcoming critical periods within a specified age range.
        
        Parameters:
        -----------
        lookahead_age : float
            Age range to look ahead for upcoming periods
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of upcoming critical periods with metadata
        """
        current_age = self.dev_stages.get_age()
        max_age = current_age + lookahead_age
        
        upcoming = []
        for period in self._critical_periods:
            # If period starts in the future and within lookahead range
            if period.begin_age > current_age and period.begin_age <= max_age:
                time_until = period.begin_age - current_age
                
                upcoming.append({
                    "period_id": period.id,
                    "name": period.name,
                    "type": period.period_type,
                    "begin_age": period.begin_age,
                    "time_until": time_until,
                    "learning_multiplier": period.learning_multiplier,
                    "affected_modules": period.affected_modules,
                    "affected_capabilities": period.affected_capabilities
                })
        
        # Sort by begin_age
        upcoming.sort(key=lambda p: p["begin_age"])
        return upcoming
    
    def get_period_history(self, period_id: Optional[str] = None) -> List[Tuple[datetime, float, str]]:
        """
        Get history of activations and deactivations for a critical period.
        
        Parameters:
        -----------
        period_id : Optional[str]
            ID of the critical period to get history for.
            If None, returns history for all periods (flattened).
            
        Returns:
        --------
        List[Tuple[datetime, float, str]]
            List of (timestamp, age, event) records
        """
        if period_id is not None:
            return self._period_activation_history.get(period_id, []).copy()
            
        # Flatten history for all periods
        all_history = []
        for period_history in self._period_activation_history.values():
            all_history.extend(period_history)
            
        # Sort by timestamp
        all_history.sort(key=lambda record: record[0])
        return all_history
    
    def add_custom_period(self, period: CriticalPeriodDefinition) -> str:
        """
        Add a custom critical period definition.
        
        Parameters:
        -----------
        period : CriticalPeriodDefinition
            Critical period definition to add
            
        Returns:
        --------
        str
            ID of the added critical period
        """
        # Check if period with this ID already exists
        existing_ids = {p.id for p in self._critical_periods}
        if period.id in existing_ids:
            raise ValueError(f"Critical period with ID {period.id} already exists")
            
        # Add to critical periods list
        self._critical_periods.append(period)
        
        # Check if it should be active immediately
        current_age = self.dev_stages.get_age()
        if period.begin_age <= current_age < period.end_age:
            self._active_periods[period.id] = period
            
            # Record activation in history
            activation_record = (datetime.now(), current_age, "activated")
            self._period_activation_history[period.id] = [activation_record]
            
            logger.info(f"Custom critical period added and activated: {period.name}")
        else:
            logger.info(f"Custom critical period added: {period.name}")
            
        return period.id
    
    def get_state(self) -> StateDict:
        """
        Get the current state as a dictionary for saving.
        
        Returns:
        --------
        StateDict
            Current state dictionary
        """
        return {
            "active_periods": list(self._active_periods.keys()),
            "period_activation_history": self._period_activation_history,
            "last_update_time": self._last_update_time
        }
    
    def load_state(self, state: StateDict) -> None:
        """
        Load state from a state dictionary.
        
        Parameters:
        -----------
        state : StateDict
            State dictionary to load from
        """
        # Reset active periods
        self._active_periods = {}
        
        # Activate periods based on state
        for period_id in state["active_periods"]:
            for period in self._critical_periods:
                if period.id == period_id:
                    self._active_periods[period_id] = period
                    break
        
        self._period_activation_history = state["period_activation_history"]
        self._last_update_time = state["last_update_time"]
        
        logger.info(f"Critical periods state loaded with {len(self._active_periods)} active periods")
