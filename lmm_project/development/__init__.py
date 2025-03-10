"""
Development module for the LMM project.

This module implements the developmental framework for the LMM system,
managing cognitive growth, developmental stages, critical periods,
and developmental milestones to create authentic cognitive development.
"""

from typing import Dict, Optional, Any, List

# Import all components from submodules
from lmm_project.development.models import (
    DevelopmentalStage,
    StageRange,
    StageDefinition,
    MilestoneStatus,
    MilestoneDefinition,
    MilestoneRecord,
    CriticalPeriodType,
    CriticalPeriodDefinition,
    GrowthRateModel,
    DevelopmentConfig
)

from lmm_project.development.developmental_stages import DevelopmentalStages
from lmm_project.development.critical_periods import CriticalPeriods
from lmm_project.development.growth_rate_controller import GrowthRateController
from lmm_project.development.milestone_tracker import MilestoneTracker

# Make all imported components available
__all__ = [
    # Model classes
    "DevelopmentalStage",
    "StageRange",
    "StageDefinition",
    "MilestoneStatus",
    "MilestoneDefinition",
    "MilestoneRecord",
    "CriticalPeriodType",
    "CriticalPeriodDefinition",
    "GrowthRateModel",
    "DevelopmentConfig",
    
    # Core components
    "DevelopmentalStages",
    "CriticalPeriods",
    "GrowthRateController",
    "MilestoneTracker",
    
    # Main development manager
    "DevelopmentManager",
    "create_development_manager",
    "get_development_manager"
]

# Singleton instance
_development_manager = None

class DevelopmentManager:
    """
    Central manager for all developmental components.
    
    This class provides a unified interface to all developmental subsystems,
    managing cognitive growth, developmental stages, critical periods, and milestones.
    """
    
    def __init__(self, config: Optional[DevelopmentConfig] = None):
        """
        Initialize the development manager.
        
        Parameters:
        -----------
        config : Optional[DevelopmentConfig]
            Configuration for all developmental subsystems.
            If None, default settings will be used.
        """
        # Create all subsystems
        self.dev_stages = DevelopmentalStages(config)
        self.critical_periods = CriticalPeriods(self.dev_stages, config)
        self.milestone_tracker = MilestoneTracker(self.dev_stages, config)
        self.growth_controller = GrowthRateController(
            self.dev_stages, 
            self.critical_periods,
            config
        )
        
        # Store config
        self._config = config
    
    def update(self) -> None:
        """
        Update all developmental subsystems.
        
        This method should be called regularly to update the developmental state.
        """
        # Update in dependency order
        self.dev_stages.update()
        self.critical_periods.update()
        self.milestone_tracker.update()
        self.growth_controller.update_all()
    
    def get_age(self) -> float:
        """
        Get the current developmental age.
        
        Returns:
        --------
        float
            Current age in age units
        """
        return self.dev_stages.get_age()
    
    def get_stage(self) -> DevelopmentalStage:
        """
        Get the current developmental stage.
        
        Returns:
        --------
        DevelopmentalStage
            Current developmental stage
        """
        return self.dev_stages.get_current_stage()
    
    def set_age(self, age: float) -> None:
        """
        Set the developmental age manually.
        
        Parameters:
        -----------
        age : float
            New developmental age in age units
        """
        self.dev_stages.set_age(age)
        
        # Update dependent systems to reflect new age
        self.critical_periods.update()
        self.milestone_tracker.update()
    
    def register_module(self, module_name: str, capabilities: List[str]) -> None:
        """
        Register a module and its capabilities for growth tracking.
        
        Parameters:
        -----------
        module_name : str
            Name of the module to register
        capabilities : List[str]
            List of capabilities provided by the module
        """
        self.growth_controller.register_module(module_name, capabilities)
    
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
            Intensity of usage (0.0-2.0)
        """
        self.growth_controller.record_capability_usage(
            module_name, capability, usage_intensity
        )
    
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
        return self.growth_controller.get_capability_progress(module_name, capability)
    
    def get_growth_rate(self, module_name: str, capability: str) -> float:
        """
        Get the current growth rate for a capability.
        
        Parameters:
        -----------
        module_name : str
            Name of the module containing the capability
        capability : str
            Name of the capability to calculate growth rate for
            
        Returns:
        --------
        float
            Growth rate multiplier for the capability
        """
        return self.growth_controller.calculate_growth_rate(module_name, capability)
    
    def evaluate_milestone(
        self, 
        milestone_id: str, 
        evaluation_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate progress on a specific milestone.
        
        Parameters:
        -----------
        milestone_id : str
            ID of the milestone to evaluate
        evaluation_data : Optional[Dict[str, Any]]
            Data to use for evaluation
            
        Returns:
        --------
        float
            Current progress on milestone (0.0-1.0)
        """
        return self.milestone_tracker.evaluate_milestone(milestone_id, evaluation_data)
    
    def get_active_milestones(self) -> List[str]:
        """
        Get list of currently active milestones.
        
        Returns:
        --------
        List[str]
            List of milestone IDs that are currently in progress
        """
        return self.milestone_tracker.get_active_milestones()
    
    def get_developmental_status(self) -> Dict[str, Any]:
        """
        Get comprehensive developmental status report.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with all developmental status metrics
        """
        # Get status from milestone tracker
        milestone_status = self.milestone_tracker.get_developmental_status()
        
        # Add critical period information
        active_periods = self.critical_periods.get_active_periods()
        upcoming_periods = self.critical_periods.get_upcoming_periods()
        
        # Add growth controller information
        overall_progress = self.growth_controller.get_overall_progress()
        
        # Combine all information
        return {
            "age": self.dev_stages.get_age(),
            "stage": self.dev_stages.get_current_stage(),
            "milestones": milestone_status,
            "active_critical_periods": [
                {"id": period_id, "name": period.name, "type": period.period_type}
                for period_id, period in active_periods.items()
            ],
            "upcoming_critical_periods": upcoming_periods,
            "overall_capability_progress": overall_progress,
            "time_to_next_stage": self.dev_stages.estimate_time_to_next_stage()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get combined state from all developmental subsystems.
        
        Returns:
        --------
        Dict[str, Any]
            Combined state dictionary
        """
        return {
            "dev_stages": self.dev_stages.get_state(),
            "critical_periods": self.critical_periods.get_state(),
            "milestone_tracker": self.milestone_tracker.get_state(),
            "growth_controller": self.growth_controller.get_state()
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state for all developmental subsystems.
        
        Parameters:
        -----------
        state : Dict[str, Any]
            Combined state dictionary
        """
        if "dev_stages" in state:
            self.dev_stages.load_state(state["dev_stages"])
            
        if "critical_periods" in state:
            self.critical_periods.load_state(state["critical_periods"])
            
        if "milestone_tracker" in state:
            self.milestone_tracker.load_state(state["milestone_tracker"])
            
        if "growth_controller" in state:
            self.growth_controller.load_state(state["growth_controller"])

def create_development_manager(config: Optional[DevelopmentConfig] = None) -> DevelopmentManager:
    """
    Create the global development manager.
    
    Parameters:
    -----------
    config : Optional[DevelopmentConfig]
        Configuration for all developmental subsystems
        
    Returns:
    --------
    DevelopmentManager
        The created development manager instance
    """
    global _development_manager
    
    if _development_manager is not None:
        raise RuntimeError("Development manager has already been created")
        
    _development_manager = DevelopmentManager(config)
    return _development_manager

def get_development_manager() -> DevelopmentManager:
    """
    Get the global development manager instance.
    
    Returns:
    --------
    DevelopmentManager
        The global development manager instance
        
    Raises:
    -------
    RuntimeError
        If the development manager has not been created yet
    """
    global _development_manager
    
    if _development_manager is None:
        raise RuntimeError(
            "Development manager has not been created yet. "
            "Call create_development_manager() first."
        )
        
    return _development_manager
