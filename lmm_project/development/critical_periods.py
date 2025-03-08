"""
Critical Periods Module

This module implements critical/sensitive periods in development where certain
capabilities develop more rapidly and with higher plasticity. Missing these periods
may result in limited development of the relevant capabilities.

Critical periods are an essential concept in developmental psychology and are
implemented here to model realistic cognitive development pathways.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime
import uuid

from lmm_project.development.models import CriticalPeriod, DevelopmentalEvent
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class CriticalPeriodManager:
    """
    Manages critical and sensitive periods for developmental capabilities
    
    Critical periods are time windows where specific capabilities develop more
    rapidly and with greater plasticity. Missing these periods may result in
    permanent limitations to development.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.critical_periods: Dict[str, CriticalPeriod] = {}
        self.active_periods: Set[str] = set()
        self.completed_periods: Set[str] = set()
        self.missed_periods: Set[str] = set()
        
        # Initialize with predefined critical periods
        self._define_critical_periods()
        
    def _define_critical_periods(self) -> None:
        """Define the critical periods for various capabilities"""
        
        # Language acquisition critical period
        language_period = CriticalPeriod(
            name="Language Acquisition Period",
            capability="language_acquisition",
            age_range=(0.5, 7.0),
            plasticity_multiplier=3.0,
            importance=0.9,
            module_targets=["language"],
            causes_permanent_limitation=True,
            limitation_factor=0.7,
            recommended_experiences=[
                "mother_conversation",
                "language_exposure",
                "word_learning",
                "grammar_acquisition"
            ]
        )
        self.critical_periods[language_period.id] = language_period
        
        # Basic sensory processing period
        sensory_period = CriticalPeriod(
            name="Sensory Processing Period",
            capability="sensory_processing",
            age_range=(0.0, 2.0),
            plasticity_multiplier=2.5,
            importance=0.8,
            module_targets=["perception", "attention"],
            causes_permanent_limitation=True,
            limitation_factor=0.8,
            recommended_experiences=[
                "sensory_stimulation",
                "pattern_recognition_training",
                "sensory_discrimination"
            ]
        )
        self.critical_periods[sensory_period.id] = sensory_period
        
        # Basic emotional development period
        emotional_period = CriticalPeriod(
            name="Emotional Development Period",
            capability="emotional_processing",
            age_range=(0.2, 3.0),
            plasticity_multiplier=2.0,
            importance=0.8,
            module_targets=["emotion"],
            causes_permanent_limitation=True,
            limitation_factor=0.75,
            recommended_experiences=[
                "emotional_mirroring",
                "affective_interaction",
                "emotion_naming",
                "emotional_response_training"
            ]
        )
        self.critical_periods[emotional_period.id] = emotional_period
        
        # Social understanding period
        social_period = CriticalPeriod(
            name="Social Understanding Period",
            capability="social_cognition",
            age_range=(1.0, 8.0),
            plasticity_multiplier=1.8,
            importance=0.7,
            module_targets=["social"],
            causes_permanent_limitation=False,
            limitation_factor=0.85,
            recommended_experiences=[
                "social_interaction",
                "perspective_taking",
                "social_rules_learning",
                "empathy_development"
            ]
        )
        self.critical_periods[social_period.id] = social_period
        
        # Working memory development
        memory_period = CriticalPeriod(
            name="Working Memory Development",
            capability="working_memory",
            age_range=(2.0, 10.0),
            plasticity_multiplier=1.5,
            importance=0.6,
            module_targets=["memory", "executive"],
            causes_permanent_limitation=False,
            limitation_factor=0.9,
            recommended_experiences=[
                "memory_games",
                "sequential_tasks",
                "information_holding",
                "task_switching"
            ]
        )
        self.critical_periods[memory_period.id] = memory_period
        
        # Self-awareness formation period
        self_awareness_period = CriticalPeriod(
            name="Self-Awareness Formation",
            capability="self_model",
            age_range=(1.5, 6.0),
            plasticity_multiplier=1.7,
            importance=0.7,
            module_targets=["consciousness", "identity"],
            causes_permanent_limitation=False,
            limitation_factor=0.85,
            recommended_experiences=[
                "self_recognition",
                "autobiographical_memory_formation",
                "self_other_distinction",
                "preference_formation"
            ]
        )
        self.critical_periods[self_awareness_period.id] = self_awareness_period
        
        # Abstract reasoning development
        reasoning_period = CriticalPeriod(
            name="Abstract Reasoning Development",
            capability="abstract_thinking",
            age_range=(5.0, 14.0),
            plasticity_multiplier=1.6,
            importance=0.6,
            module_targets=["executive", "creativity", "belief"],
            causes_permanent_limitation=False,
            limitation_factor=0.9,
            recommended_experiences=[
                "abstract_concept_learning",
                "hypothetical_reasoning",
                "category_formation",
                "symbolic_thinking"
            ]
        )
        self.critical_periods[reasoning_period.id] = reasoning_period
        
        # Moral reasoning development
        moral_period = CriticalPeriod(
            name="Moral Development Period",
            capability="moral_reasoning",
            age_range=(3.0, 12.0),
            plasticity_multiplier=1.5,
            importance=0.6,
            module_targets=["social", "belief"],
            causes_permanent_limitation=False,
            limitation_factor=0.9,
            recommended_experiences=[
                "moral_dilemmas",
                "rule_learning",
                "fairness_discussions",
                "empathy_building"
            ]
        )
        self.critical_periods[moral_period.id] = moral_period
    
    def update_periods_for_age(self, current_age: float) -> List[DevelopmentalEvent]:
        """
        Update critical periods based on the current developmental age
        
        Returns a list of DevelopmentalEvents for any period status changes
        """
        events = []
        
        # Check each period for status changes
        for period_id, period in self.critical_periods.items():
            min_age, max_age = period.age_range
            old_status = period.status
            
            # Determine new status based on age
            if current_age < min_age:
                new_status = "pending"
            elif min_age <= current_age <= max_age:
                new_status = "active"
            else:
                if period.status == "active":
                    # Period was active and is now complete
                    new_status = "completed"
                elif period.status == "pending":
                    # Period was never activated, so it was missed
                    new_status = "missed"
                else:
                    # Keep existing completed/missed status
                    new_status = period.status
            
            # If status changed, update and record event
            if new_status != old_status:
                period.status = new_status
                
                # Update tracking sets
                if new_status == "active":
                    self.active_periods.add(period_id)
                elif new_status == "completed":
                    self.active_periods.discard(period_id)
                    self.completed_periods.add(period_id)
                elif new_status == "missed":
                    self.missed_periods.add(period_id)
                
                # Create developmental event
                event = DevelopmentalEvent(
                    event_type="critical_period",
                    description=f"Critical period '{period.name}' status changed from {old_status} to {new_status}",
                    age=current_age,
                    affected_modules=period.module_targets,
                    significance=period.importance,
                    details={
                        "period_id": period_id,
                        "capability": period.capability,
                        "old_status": old_status,
                        "new_status": new_status,
                        "permanent_limitation": period.causes_permanent_limitation and new_status == "missed"
                    }
                )
                events.append(event)
                
                # Broadcast event
                if self.event_bus:
                    message = Message(
                        sender="critical_period_manager",
                        message_type="critical_period_update",
                        content={
                            "period_id": period_id,
                            "period_name": period.name,
                            "capability": period.capability,
                            "old_status": old_status,
                            "new_status": new_status,
                            "age": current_age,
                            "event": event.dict()
                        }
                    )
                    self.event_bus.publish(message)
                
                logger.info(f"Critical period '{period.name}' status changed: {old_status} -> {new_status} at age {current_age}")
        
        return events
    
    def get_active_periods(self) -> List[CriticalPeriod]:
        """Get all currently active critical periods"""
        return [self.critical_periods[period_id] for period_id in self.active_periods]
    
    def get_development_multiplier(self, capability: str, module_name: str) -> float:
        """
        Get the development multiplier for a capability and module
        
        This is used to accelerate learning during critical periods.
        Returns 1.0 if no relevant critical period is active.
        """
        multiplier = 1.0
        
        for period_id in self.active_periods:
            period = self.critical_periods[period_id]
            
            # Check if this period affects the requested capability and module
            if (period.capability == capability or capability in period.capability) and \
               (module_name in period.module_targets):
                # Use the maximum multiplier if multiple periods apply
                multiplier = max(multiplier, period.plasticity_multiplier)
        
        return multiplier
    
    def get_capability_limitation_factor(self, capability: str) -> float:
        """
        Get the limitation factor for a capability based on missed critical periods
        
        This is used to determine how much a capability is permanently limited
        if its critical period was missed.
        
        Returns 1.0 if no limitations (full development possible)
        Returns a lower value (e.g., 0.7) if development is limited
        """
        limitation = 1.0
        
        for period_id in self.missed_periods:
            period = self.critical_periods[period_id]
            
            # Check if this period affects the requested capability
            if period.capability == capability or capability in period.capability:
                if period.causes_permanent_limitation:
                    # Use the minimum limitation factor if multiple periods apply
                    limitation = min(limitation, period.limitation_factor)
        
        return limitation
    
    def get_recommended_experiences(self, current_age: float) -> List[Dict[str, Any]]:
        """
        Get recommended experiences for the current age based on active critical periods
        
        Returns a list of dictionaries with experience recommendations
        """
        recommendations = []
        
        for period_id in self.active_periods:
            period = self.critical_periods[period_id]
            
            # Calculate urgency based on how much of the period remains
            _, max_age = period.age_range
            time_remaining = max_age - current_age
            total_duration = max_age - period.age_range[0]
            urgency = 1.0 - (time_remaining / total_duration)
            
            recommendations.append({
                "period_id": period_id,
                "period_name": period.name,
                "capability": period.capability,
                "module_targets": period.module_targets,
                "urgency": urgency,
                "importance": period.importance,
                "recommended_experiences": period.recommended_experiences
            })
        
        # Sort by urgency and importance
        recommendations.sort(
            key=lambda x: (x["urgency"], x["importance"]), 
            reverse=True
        )
        
        return recommendations
    
    def register_custom_critical_period(
        self, 
        name: str,
        capability: str,
        age_range: Tuple[float, float],
        module_targets: List[str],
        plasticity_multiplier: float = 2.0,
        importance: float = 0.5,
        causes_permanent_limitation: bool = False,
        limitation_factor: float = 0.8,
        recommended_experiences: List[str] = None
    ) -> str:
        """
        Register a custom critical period
        
        Returns the ID of the created period
        """
        period = CriticalPeriod(
            name=name,
            capability=capability,
            age_range=age_range,
            plasticity_multiplier=plasticity_multiplier,
            importance=importance,
            module_targets=module_targets,
            causes_permanent_limitation=causes_permanent_limitation,
            limitation_factor=limitation_factor,
            recommended_experiences=recommended_experiences or []
        )
        
        self.critical_periods[period.id] = period
        logger.info(f"Registered custom critical period: {name} for {capability}")
        
        return period.id
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the critical period system"""
        return {
            "critical_periods": {p_id: period.dict() for p_id, period in self.critical_periods.items()},
            "active_periods": list(self.active_periods),
            "completed_periods": list(self.completed_periods),
            "missed_periods": list(self.missed_periods)
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state"""
        if "critical_periods" in state:
            # Clear existing periods
            self.critical_periods.clear()
            
            # Load saved periods
            for period_id, period_data in state["critical_periods"].items():
                self.critical_periods[period_id] = CriticalPeriod(**period_data)
        
        if "active_periods" in state:
            self.active_periods = set(state["active_periods"])
            
        if "completed_periods" in state:
            self.completed_periods = set(state["completed_periods"])
            
        if "missed_periods" in state:
            self.missed_periods = set(state["missed_periods"]) 
