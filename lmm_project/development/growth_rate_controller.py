"""
Growth rate controller for the LMM project.

This module provides mechanisms to control the rate of growth and development
across different capabilities and modules within the LMM system, implementing
variable growth trajectories and non-linear developmental patterns.
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.types import StateDict
from lmm_project.development.models import (
    DevelopmentalStage, 
    GrowthRateModel,
    DevelopmentConfig
)
from lmm_project.development.developmental_stages import DevelopmentalStages
from lmm_project.development.critical_periods import CriticalPeriods
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

class GrowthRateController:
    """
    Controls growth rates across different capabilities and modules.
    
    This class manages differential growth rates across the cognitive system,
    implementing principles like variable growth trajectories, development plateaus,
    and practice effects to create authentic cognitive development.
    """
    
    def __init__(
        self,
        dev_stages: DevelopmentalStages,
        critical_periods: CriticalPeriods,
        config: Optional[DevelopmentConfig] = None
    ):
        """
        Initialize the growth rate controller.
        
        Parameters:
        -----------
        dev_stages : DevelopmentalStages
            Reference to the developmental stages manager
        critical_periods : CriticalPeriods
            Reference to the critical periods manager
        config : Optional[DevelopmentConfig]
            Configuration containing growth rate model. If None, uses default settings.
        """
        self.event_system = EventBus()
        self.dev_stages = dev_stages
        self.critical_periods = critical_periods
        self._config = config or self._load_default_config()
        
        # Store growth rate model parameters
        self._growth_model = self._config.growth_rate_model
        
        # Track usage of capabilities to model practice effects
        self._capability_usage: Dict[str, Dict[str, float]] = {}
        
        # Track growth progress for different capabilities
        self._growth_progress: Dict[str, Dict[str, float]] = {}
        
        # Track growth rates history for analysis
        self._growth_rate_history: List[Dict[str, Any]] = []
        
        # Record last update time
        self._last_update_time = time.time()
        
        # Random number generator for variability
        self._rng = np.random.RandomState(int(time.time()))
        
        logger.info("Growth rate controller initialized")
    
    def _load_default_config(self) -> DevelopmentConfig:
        """
        Load default configuration for growth rate controller.
        
        Returns:
        --------
        DevelopmentConfig
            Default configuration with growth rate model
        """
        # Create default growth rate model
        growth_model = GrowthRateModel(
            base_rate=1.0,
            stage_multipliers={
                DevelopmentalStage.PRENATAL: 0.8,
                DevelopmentalStage.INFANT: 1.6,
                DevelopmentalStage.CHILD: 1.2,
                DevelopmentalStage.ADOLESCENT: 0.9,
                DevelopmentalStage.ADULT: 0.6
            },
            critical_period_boost=2.0,
            practice_effect=1.3,
            plateau_threshold=0.85,
            plateau_factor=0.4
        )
        
        # Return minimal config with just growth rate model
        return DevelopmentConfig(
            initial_age=0.0,
            time_acceleration=1000.0,
            stage_definitions=[],
            milestone_definitions=[],
            critical_period_definitions=[],
            growth_rate_model=growth_model
        )
    
    def register_module(self, module_name: str, capabilities: List[str]) -> None:
        """
        Register a module and its capabilities for growth rate tracking.
        
        Parameters:
        -----------
        module_name : str
            Name of the module to register
        capabilities : List[str]
            List of capabilities provided by the module
        """
        # Initialize usage tracking if not already present
        if module_name not in self._capability_usage:
            self._capability_usage[module_name] = {}
            
        if module_name not in self._growth_progress:
            self._growth_progress[module_name] = {}
            
        # Register each capability
        for capability in capabilities:
            if capability not in self._capability_usage[module_name]:
                self._capability_usage[module_name][capability] = 0.0
                
            if capability not in self._growth_progress[module_name]:
                self._growth_progress[module_name][capability] = 0.0
                
        logger.info(f"Registered module {module_name} with {len(capabilities)} capabilities")
    
    def record_capability_usage(
        self, 
        module_name: str, 
        capability: str, 
        usage_intensity: float = 1.0
    ) -> None:
        """
        Record usage of a capability to model practice effects.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability being used
        usage_intensity : float
            Intensity of usage, higher values mean stronger practice effect (0.0-2.0)
        """
        # Ensure module and capability are registered
        if module_name not in self._capability_usage:
            self._capability_usage[module_name] = {}
            self._growth_progress[module_name] = {}
            
        if capability not in self._capability_usage[module_name]:
            self._capability_usage[module_name][capability] = 0.0
            self._growth_progress[module_name][capability] = 0.0
            
        # Update usage tracker with time decay (recent usage matters more)
        current_time = time.time()
        time_since_update = current_time - self._last_update_time
        
        # Apply time decay to existing usage value
        decay_factor = np.exp(-0.1 * time_since_update)  # Exponential decay
        current_usage = self._capability_usage[module_name][capability]
        decayed_usage = current_usage * decay_factor
        
        # Add new usage with intensity factor
        new_usage = decayed_usage + usage_intensity
        
        # Cap at reasonable maximum to prevent runaway effects
        self._capability_usage[module_name][capability] = min(new_usage, 5.0)
    
    def calculate_growth_rate(
        self, 
        module_name: str, 
        capability: str, 
        current_progress: Optional[float] = None
    ) -> float:
        """
        Calculate the current growth rate for a specific capability.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability to calculate growth rate for
        current_progress : Optional[float]
            Current progress level (0.0-1.0). If None, uses tracked progress.
            
        Returns:
        --------
        float
            Growth rate multiplier for the capability
        """
        # Get current stage for base multiplier
        current_stage = self.dev_stages.get_current_stage()
        stage_multiplier = self._growth_model.stage_multipliers.get(
            current_stage, 1.0
        )
        
        # Get critical period multiplier if applicable
        critical_multiplier = self.critical_periods.get_learning_multiplier(
            module_name=module_name,
            capability=capability
        )
        
        # Get practice effect multiplier
        usage_level = self._capability_usage.get(module_name, {}).get(capability, 0.0)
        practice_multiplier = 1.0 + (
            (self._growth_model.practice_effect - 1.0) * 
            min(1.0, usage_level / 3.0)  # Scale by usage, cap at 1.0
        )
        
        # Apply plateau effect for advanced progress
        if current_progress is None:
            current_progress = self._growth_progress.get(module_name, {}).get(capability, 0.0)
            
        plateau_effect = 1.0
        if current_progress >= self._growth_model.plateau_threshold:
            # Calculate how far into the plateau region we are (0.0-1.0)
            plateau_progress = (current_progress - self._growth_model.plateau_threshold) / (
                1.0 - self._growth_model.plateau_threshold
            )
            # Apply diminishing returns
            plateau_effect = 1.0 - (plateau_progress * (1.0 - self._growth_model.plateau_factor))
            
        # Calculate variability factor if enabled
        variability = 1.0
        if self._config.enable_variability:
            # Generate random variability around 1.0
            variability_range = self._config.variability_factor
            variability = 1.0 + self._rng.uniform(-variability_range, variability_range)
            
        # Combine all factors with base rate
        growth_rate = (
            self._growth_model.base_rate *
            stage_multiplier *
            critical_multiplier *
            practice_multiplier *
            plateau_effect *
            variability
        )
        
        return growth_rate
    
    def update_growth_progress(
        self, 
        module_name: str, 
        capability: str, 
        delta_time: Optional[float] = None
    ) -> float:
        """
        Update growth progress for a capability based on current growth rate.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability to update
        delta_time : Optional[float]
            Time elapsed for growth. If None, calculated automatically.
            
        Returns:
        --------
        float
            New progress value
        """
        # Ensure module and capability are registered
        if module_name not in self._growth_progress:
            self._growth_progress[module_name] = {}
            
        if capability not in self._growth_progress[module_name]:
            self._growth_progress[module_name][capability] = 0.0
            
        # Get current progress
        current_progress = self._growth_progress[module_name][capability]
        
        # Calculate growth rate
        growth_rate = self.calculate_growth_rate(
            module_name, capability, current_progress
        )
        
        # Determine time elapsed
        current_time = time.time()
        if delta_time is None:
            delta_time = current_time - self._last_update_time
            
        # Convert real time to developmental time
        dev_delta_time = delta_time * self._config.time_acceleration
        
        # Calculate progress increment
        progress_increment = growth_rate * dev_delta_time / 3600.0  # Scale to reasonable units
        
        # Update progress, capped at 1.0
        new_progress = min(1.0, current_progress + progress_increment)
        self._growth_progress[module_name][capability] = new_progress
        
        # Record history for analysis
        self._growth_rate_history.append({
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "capability": capability,
            "growth_rate": growth_rate,
            "progress": new_progress,
            "stage": self.dev_stages.get_current_stage(),
            "age": self.dev_stages.get_age()
        })
        
        # Emit event if significant progress made
        if int(new_progress * 100) > int(current_progress * 100):
            self.event_system.publish(Message(
                name="capability_progress_updated",
                data={
                    "module": module_name,
                    "capability": capability,
                    "previous_progress": current_progress,
                    "new_progress": new_progress,
                    "growth_rate": growth_rate
                }
            ))
            
        return new_progress
    
    def update_all(self) -> None:
        """
        Update growth progress for all registered capabilities.
        
        This method updates all capabilities based on their current growth rates.
        """
        current_time = time.time()
        delta_time = current_time - self._last_update_time
        
        # Update each registered module and capability
        for module_name, capabilities in self._growth_progress.items():
            for capability in capabilities:
                self.update_growth_progress(module_name, capability, delta_time)
                
        # Update last update time
        self._last_update_time = current_time
    
    def get_capability_progress(self, module_name: str, capability: str) -> float:
        """
        Get the current progress level for a capability.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability to get progress for
            
        Returns:
        --------
        float
            Current progress level (0.0-1.0)
        """
        return self._growth_progress.get(module_name, {}).get(capability, 0.0)
    
    def get_module_progress(self, module_name: str) -> Dict[str, float]:
        """
        Get progress levels for all capabilities in a module.
        
        Parameters:
        -----------
        module_name : str
            Name of the module to get progress for
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping capability names to progress levels
        """
        return self._growth_progress.get(module_name, {}).copy()
    
    def get_overall_progress(self) -> float:
        """
        Get overall progress across all capabilities.
        
        Returns:
        --------
        float
            Average progress across all capabilities (0.0-1.0)
        """
        total_progress = 0.0
        capability_count = 0
        
        for module_progress in self._growth_progress.values():
            for progress in module_progress.values():
                total_progress += progress
                capability_count += 1
                
        return total_progress / max(1, capability_count)
    
    def set_capability_progress(
        self, 
        module_name: str, 
        capability: str, 
        progress: float
    ) -> None:
        """
        Manually set progress for a capability.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability to set progress for
        progress : float
            Progress level to set (0.0-1.0)
        """
        if progress < 0.0 or progress > 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
            
        # Ensure module and capability are registered
        if module_name not in self._growth_progress:
            self._growth_progress[module_name] = {}
            
        # Get current progress for event
        current_progress = self._growth_progress.get(module_name, {}).get(capability, 0.0)
        
        # Set new progress
        self._growth_progress[module_name][capability] = progress
        
        # Emit event for progress change
        self.event_system.publish(Message(
            name="capability_progress_set",
            data={
                "module": module_name,
                "capability": capability,
                "previous_progress": current_progress,
                "new_progress": progress
            }
        ))
        
        logger.info(f"Manually set {module_name}.{capability} progress to {progress:.2f}")
    
    def get_growth_rate_history(
        self, 
        module_name: Optional[str] = None, 
        capability: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get history of growth rate changes for analysis.
        
        Parameters:
        -----------
        module_name : Optional[str]
            Filter by module name
        capability : Optional[str]
            Filter by capability name
        limit : int
            Maximum number of records to return
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of growth rate history records
        """
        filtered_history = self._growth_rate_history
        
        if module_name is not None:
            filtered_history = [
                record for record in filtered_history 
                if record["module"] == module_name
            ]
            
        if capability is not None:
            filtered_history = [
                record for record in filtered_history
                if record["capability"] == capability
            ]
            
        # Return most recent records first
        return list(reversed(filtered_history))[:limit]
    
    def get_state(self) -> StateDict:
        """
        Get the current state as a dictionary for saving.
        
        Returns:
        --------
        StateDict
            Current state dictionary
        """
        return {
            "capability_usage": self._capability_usage,
            "growth_progress": self._growth_progress,
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
        self._capability_usage = state["capability_usage"]
        self._growth_progress = state["growth_progress"]
        self._last_update_time = state["last_update_time"]
        
        logger.info("Growth rate controller state loaded")
