"""
Development Package

This package contains the modules that manage the developmental processes in the LMM.
The development system handles stages, critical periods, milestones, and growth rates
to create a realistic and psychologically-grounded cognitive development process.

Main Components:
- DevelopmentalStageManager: Manages progression through developmental stages
- CriticalPeriodManager: Handles critical/sensitive periods for capabilities
- MilestoneTracker: Tracks developmental milestones across various domains
- GrowthRateController: Controls the rate of development for different capabilities
- DevelopmentSystem: Integrates all components into a cohesive system
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.development.models import (
    DevelopmentalStage, CriticalPeriod, Milestone, DevelopmentalTrajectory,
    DevelopmentalEvent, GrowthRateParameters
)
from lmm_project.development.developmental_stages import DevelopmentalStageManager
from lmm_project.development.critical_periods import CriticalPeriodManager
from lmm_project.development.milestone_tracker import MilestoneTracker
from lmm_project.development.growth_rate_controller import GrowthRateController

logger = logging.getLogger(__name__)

def get_development_system(event_bus: Optional[EventBus] = None) -> "DevelopmentSystem":
    """
    Factory function to create a complete development system
    
    Returns:
        A fully configured DevelopmentSystem instance
    """
    return DevelopmentSystem(event_bus=event_bus)

class DevelopmentSystem:
    """
    Integrated development system for the LMM
    
    This class integrates all developmental components:
    - Stage management
    - Critical period tracking
    - Milestone achievement
    - Growth rate control
    
    It provides a unified interface for managing the mind's developmental progression.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the development system with all components"""
        self.event_bus = event_bus
        
        # Initialize all development components
        self.stage_manager = DevelopmentalStageManager(event_bus=event_bus)
        self.critical_period_manager = CriticalPeriodManager(event_bus=event_bus)
        self.milestone_tracker = MilestoneTracker(event_bus=event_bus)
        self.growth_controller = GrowthRateController()
        
        # Track developmental events
        self.developmental_events: List[DevelopmentalEvent] = []
        
        logger.info("Development system initialized")
    
    def update_development(self, 
                          capabilities: Dict[str, float], 
                          module_capabilities: Dict[str, Dict[str, float]],
                          delta_age: float) -> Dict[str, Any]:
        """
        Update the developmental state based on the current capabilities
        
        Args:
            capabilities: The overall system capabilities
            module_capabilities: Module-specific capabilities
            delta_age: How much to increase the developmental age
            
        Returns:
            Dictionary with development update results and events
        """
        events = []
        development_updates = {}
        
        # Update the developmental age
        current_age = self.stage_manager.trajectory.current_age
        self.stage_manager.update_age(delta_age)
        new_age = self.stage_manager.trajectory.current_age
        
        # Update critical periods based on new age
        critical_period_events = self.critical_period_manager.update_periods_for_age(new_age)
        events.extend(critical_period_events)
        
        # Update age in events
        for event in critical_period_events:
            event.age = new_age
        
        # Check for milestone achievements
        milestone_events = self.milestone_tracker.evaluate_milestones(capabilities)
        events.extend(milestone_events)
        
        # Update age in events
        for event in milestone_events:
            event.age = new_age
        
        # Evaluate stage transitions
        next_stage = self.stage_manager.evaluate_stage_transition(capabilities)
        if next_stage:
            # Transition to the next stage
            self.stage_manager.transition_to_stage(next_stage)
            
            # Record transition in results
            development_updates["stage_transition"] = {
                "from": self.stage_manager.current_stage,
                "to": next_stage,
                "age": new_age
            }
        
        # Store all developmental events
        self.developmental_events.extend(events)
        
        # Prepare result dictionary
        result = {
            "age": {
                "previous": current_age,
                "current": new_age,
                "delta": delta_age
            },
            "stage": self.stage_manager.current_stage,
            "events": [event.dict() for event in events],
            "active_critical_periods": [period.dict() for period in self.critical_period_manager.get_active_periods()],
            "achieved_milestones": [m.dict() for m in self.milestone_tracker.get_achieved_milestones()],
            "updates": development_updates
        }
        
        return result
    
    def get_growth_rates(self, 
                        module: str, 
                        capabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Get growth rates for all capabilities in a module
        
        Args:
            module: The module to get growth rates for
            capabilities: Current capability levels
            
        Returns:
            Dictionary mapping capabilities to growth rates
        """
        growth_rates = {}
        current_age = self.stage_manager.trajectory.current_age
        
        for capability, level in capabilities.items():
            # Get critical period multiplier if any
            critical_multiplier = self.critical_period_manager.get_development_multiplier(
                capability, module
            )
            
            # Get capability limitation factor from missed critical periods
            limitation = self.critical_period_manager.get_capability_limitation_factor(capability)
            
            # Get stage-based capability ceiling
            ceiling = self.stage_manager.get_capability_ceiling(capability)
            
            # Adjust ceiling based on limitations from missed critical periods
            adjusted_ceiling = ceiling * limitation
            
            # Calculate proximity to ceiling (slows growth as approaching ceiling)
            ceiling_proximity = level / adjusted_ceiling if adjusted_ceiling > 0 else 1.0
            ceiling_factor = 1.0 - (ceiling_proximity ** 2) * 0.9
            
            # Calculate growth rate
            growth_rate = self.growth_controller.get_growth_rate(
                capability=capability,
                module=module,
                age=current_age,
                critical_period_multiplier=critical_multiplier
            )
            
            # Apply ceiling factor
            growth_rate *= max(0.1, ceiling_factor)
            
            growth_rates[capability] = growth_rate
        
        return growth_rates
    
    def get_recommended_experiences(self) -> Dict[str, Any]:
        """
        Get recommended experiences based on current developmental state
        
        This includes recommendations from critical periods, milestone
        requirements, and other developmental considerations.
        
        Returns:
            Dictionary with experience recommendations
        """
        current_age = self.stage_manager.trajectory.current_age
        
        # Get recommendations from critical periods
        critical_period_recommendations = self.critical_period_manager.get_recommended_experiences(current_age)
        
        # Get pending milestones to focus on
        pending_milestones = self.milestone_tracker.get_pending_milestones()
        achievable_milestones = [
            m for m in pending_milestones
            if all(prereq in self.milestone_tracker.achieved_milestones 
                  for prereq in m.prerequisite_milestones)
        ]
        
        # Prepare milestone recommendations
        milestone_recommendations = []
        for milestone in achievable_milestones[:5]:  # Limit to 5 most relevant
            # Generate recommendations based on milestone capabilities
            capabilities_needed = []
            for capability, required in milestone.capability_requirements.items():
                capabilities_needed.append({
                    "capability": capability,
                    "required_level": required,
                    "priority": "high" if milestone.is_essential else "medium"
                })
            
            milestone_recommendations.append({
                "milestone_name": milestone.name,
                "description": milestone.description,
                "category": milestone.category,
                "is_essential": milestone.is_essential,
                "capabilities_needed": capabilities_needed
            })
        
        return {
            "critical_period_recommendations": critical_period_recommendations,
            "milestone_recommendations": milestone_recommendations,
            "current_stage": self.stage_manager.current_stage,
            "current_age": current_age
        }
    
    def get_developmental_status(self) -> Dict[str, Any]:
        """
        Get the complete developmental status
        
        Returns:
            Dictionary with comprehensive developmental information
        """
        return {
            "age": self.stage_manager.trajectory.current_age,
            "stage": {
                "current": self.stage_manager.current_stage,
                "name": self.stage_manager.get_current_stage().name,
                "description": self.stage_manager.get_current_stage().description
            },
            "critical_periods": {
                "active": [p.dict() for p in self.critical_period_manager.get_active_periods()],
                "completed": len(self.critical_period_manager.completed_periods),
                "missed": len(self.critical_period_manager.missed_periods)
            },
            "milestones": {
                "achieved": len(self.milestone_tracker.achieved_milestones),
                "pending": len(self.milestone_tracker.pending_milestones),
                "recent": [
                    m.dict() for m in self.milestone_tracker.get_achieved_milestones()[-5:]
                ] if self.milestone_tracker.get_achieved_milestones() else []
            },
            "capabilities": {
                # Expected capability levels for current stage
                "expected": self.stage_manager.get_current_capabilities()
            },
            "trajectory": self.stage_manager.trajectory.dict()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the developmental system"""
        return {
            "stage_manager": self.stage_manager.get_state(),
            "critical_period_manager": self.critical_period_manager.get_state(),
            "milestone_tracker": self.milestone_tracker.get_state(),
            "growth_controller": self.growth_controller.get_state(),
            "developmental_events": [event.dict() for event in self.developmental_events[-100:]]  # Last 100 events
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state"""
        if "stage_manager" in state:
            self.stage_manager.load_state(state["stage_manager"])
            
        if "critical_period_manager" in state:
            self.critical_period_manager.load_state(state["critical_period_manager"])
            
        if "milestone_tracker" in state:
            self.milestone_tracker.load_state(state["milestone_tracker"])
            
        if "growth_controller" in state:
            self.growth_controller.load_state(state["growth_controller"])
            
        if "developmental_events" in state:
            self.developmental_events = [
                DevelopmentalEvent(**event_data) 
                for event_data in state["developmental_events"]
            ]
            
        logger.info("Development system state loaded") 