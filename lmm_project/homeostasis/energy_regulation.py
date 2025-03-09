from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import math
import logging

from lmm_project.core.message import Message
from lmm_project.core.event_bus import EventBus
from lmm_project.core.types import DevelopmentalStage, StateDict
from .models import HomeostaticSystem, HomeostaticNeedType, HomeostaticResponse, NeedState

logger = logging.getLogger(__name__)

class EnergyRegulator:
    """
    Manages the energy level of the cognitive system.
    
    The Energy Regulator:
    - Tracks overall energy consumption across modules
    - Implements fatigue and recovery mechanisms
    - Signals when energy conservation is needed
    - Allocates energy based on priorities during low-energy states
    
    Energy is analogous to cognitive resources and attention capacity.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        initial_energy: float = 0.8,
        recovery_rate: float = 0.05,
        consumption_rate: float = 0.02
    ):
        self.event_bus = event_bus
        self.homeostatic_system = HomeostaticSystem()
        self.homeostatic_system.initialize_needs()
        
        # Initialize energy state
        energy_need = self.homeostatic_system.needs.get(HomeostaticNeedType.ENERGY)
        if energy_need:
            energy_need.current_value = initial_energy
            energy_need.last_updated = datetime.now()
        
        # Energy regulation parameters
        self.recovery_rate = recovery_rate
        self.consumption_rate = consumption_rate
        self.module_energy_usage: Dict[str, float] = {}
        self.last_recovery_time = datetime.now()
        self.resting_state = False
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register handlers for energy-related events"""
        self.event_bus.subscribe("module_activity", self._handle_module_activity)
        self.event_bus.subscribe("system_cycle", self._handle_system_cycle)
        self.event_bus.subscribe("development_update", self._handle_development_update)
        self.event_bus.subscribe("rest_state_changed", self._handle_rest_state_changed)
    
    def _handle_module_activity(self, message: Message):
        """Handle module activity events to track energy consumption"""
        module_name = message.content.get("module_name", "unknown")
        activity_level = message.content.get("activity_level", 0.0)
        
        # Record module energy usage
        self.module_energy_usage[module_name] = activity_level
        
        # Calculate energy consumption based on activity level
        energy_consumption = activity_level * self.consumption_rate
        
        # Update energy level
        self.homeostatic_system.update_need(
            HomeostaticNeedType.ENERGY,
            -energy_consumption,
            f"Activity in {module_name} module"
        )
        
        # Check if energy is critically low
        energy_need = self.homeostatic_system.needs[HomeostaticNeedType.ENERGY]
        if energy_need.is_deficient and energy_need.urgency > 0.8:
            self._signal_energy_conservation()
    
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle events to update energy levels naturally"""
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_recovery_time).total_seconds()
        
        # Natural energy recovery (when resting) or decay (when active)
        energy_change = 0.0
        
        if self.resting_state:
            # Faster recovery during rest
            energy_change = self.recovery_rate * time_delta / 60.0
        else:
            # Gradual decay during activity
            energy_change = -self.consumption_rate * 0.5 * time_delta / 60.0
        
        # Apply change
        if abs(energy_change) > 0.001:  # Only if meaningful change
            self.homeostatic_system.update_need(
                HomeostaticNeedType.ENERGY,
                energy_change,
                "Natural recovery/consumption cycle"
            )
            self.last_recovery_time = now
            
            # Publish current energy state
            self._publish_energy_state()
    
    def _handle_development_update(self, message: Message):
        """Adapt energy parameters based on developmental stage"""
        development_level = message.content.get("development_level", 0.0)
        
        # Update homeostatic setpoints based on development
        self.homeostatic_system.adapt_to_development(development_level)
        
        # Adjust energy parameters based on development
        # Younger minds have faster recovery but also faster consumption
        if development_level < 0.3:  # Infant/early child
            self.recovery_rate = 0.08
            self.consumption_rate = 0.03
        elif development_level < 0.6:  # Child/adolescent
            self.recovery_rate = 0.06
            self.consumption_rate = 0.025
        else:  # Adult
            self.recovery_rate = 0.04
            self.consumption_rate = 0.015
            
        logger.info(f"Energy regulation adapted to development level {development_level:.2f}")
    
    def _handle_rest_state_changed(self, message: Message):
        """Handle changes in rest state"""
        self.resting_state = message.content.get("is_resting", False)
        logger.info(f"Rest state changed to: {self.resting_state}")
    
    def _signal_energy_conservation(self):
        """Signal to other modules that energy conservation is needed"""
        urgent_message = Message(
            sender="energy_regulator",
            message_type="energy_conservation_needed",
            content={
                "current_energy": self.homeostatic_system.needs[HomeostaticNeedType.ENERGY].current_value,
                "urgency": self.homeostatic_system.needs[HomeostaticNeedType.ENERGY].urgency,
                "high_usage_modules": self._get_high_energy_consumers()
            },
            priority=5  # High priority
        )
        self.event_bus.publish(urgent_message)
        logger.warning("Energy conservation signal sent - energy critically low")
    
    def _get_high_energy_consumers(self) -> List[Tuple[str, float]]:
        """Identify modules with highest energy consumption"""
        return sorted(
            [(module, usage) for module, usage in self.module_energy_usage.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 consumers
    
    def _publish_energy_state(self):
        """Publish current energy state to the event bus"""
        energy_need = self.homeostatic_system.needs[HomeostaticNeedType.ENERGY]
        energy_message = Message(
            sender="energy_regulator",
            message_type="energy_state_update",
            content={
                "current_energy": energy_need.current_value,
                "setpoint": energy_need.setpoint,
                "is_deficient": energy_need.is_deficient,
                "urgency": energy_need.urgency,
                "resting_state": self.resting_state
            }
        )
        self.event_bus.publish(energy_message)
    
    def request_energy_boost(self, amount: float, reason: str) -> bool:
        """
        Request an energy boost (from external intervention or stimulation)
        
        Returns:
            bool: True if successful, False if already at maximum energy
        """
        energy_need = self.homeostatic_system.needs[HomeostaticNeedType.ENERGY]
        
        if energy_need.current_value >= 0.95:
            logger.info(f"Energy boost request denied: already at maximum - {reason}")
            return False
        
        self.homeostatic_system.update_need(
            HomeostaticNeedType.ENERGY,
            min(amount, 1.0 - energy_need.current_value),
            f"Energy boost: {reason}"
        )
        
        self._publish_energy_state()
        logger.info(f"Energy boosted by {amount:.2f}: {reason}")
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the energy regulator"""
        energy_need = self.homeostatic_system.needs[HomeostaticNeedType.ENERGY]
        return {
            "energy_level": energy_need.current_value,
            "energy_setpoint": energy_need.setpoint,
            "energy_deficit": max(0, energy_need.setpoint - energy_need.current_value),
            "is_deficient": energy_need.is_deficient,
            "is_excessive": energy_need.is_excessive,
            "urgency": energy_need.urgency,
            "resting_state": self.resting_state,
            "module_usage": self.module_energy_usage,
            "recovery_rate": self.recovery_rate,
            "consumption_rate": self.consumption_rate
        }
    
    def load_state(self, state_dict: StateDict) -> None:
        """Load state from the provided state dictionary"""
        if "energy_level" in state_dict:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.ENERGY,
                state_dict["energy_level"] - 
                self.homeostatic_system.needs[HomeostaticNeedType.ENERGY].current_value,
                "State loaded"
            )
        
        if "resting_state" in state_dict:
            self.resting_state = state_dict["resting_state"]
            
        if "module_usage" in state_dict:
            self.module_energy_usage = state_dict["module_usage"]
            
        if "recovery_rate" in state_dict:
            self.recovery_rate = state_dict["recovery_rate"]
            
        if "consumption_rate" in state_dict:
            self.consumption_rate = state_dict["consumption_rate"]
