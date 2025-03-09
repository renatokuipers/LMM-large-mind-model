# Empty placeholder files 

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import math
import logging
import random

from lmm_project.core.message import Message
from lmm_project.core.event_bus import EventBus
from lmm_project.core.types import DevelopmentalStage, StateDict
from .models import HomeostaticSystem, HomeostaticNeedType, HomeostaticResponse, NeedState

logger = logging.getLogger(__name__)

class ArousalController:
    """
    Regulates the activation/stimulation level of the cognitive system.
    
    The Arousal Controller:
    - Monitors input stimulation levels
    - Manages attention thresholds
    - Prevents over-stimulation and under-stimulation
    - Modulates emotional responsiveness
    - Influences learning rate and memory formation
    
    Arousal is analogous to the alertness/excitability of the mind.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        initial_arousal: float = 0.5,
        decay_rate: float = 0.03,
        adaptation_rate: float = 0.05
    ):
        self.event_bus = event_bus
        self.homeostatic_system = HomeostaticSystem()
        self.homeostatic_system.initialize_needs()
        
        # Initialize arousal state
        arousal_need = self.homeostatic_system.needs.get(HomeostaticNeedType.AROUSAL)
        if arousal_need:
            arousal_need.current_value = initial_arousal
            arousal_need.last_updated = datetime.now()
        
        # Arousal regulation parameters
        self.decay_rate = decay_rate  # How quickly arousal returns to baseline
        self.adaptation_rate = adaptation_rate  # How quickly system adapts to stimulation
        self.stimulation_history: List[Tuple[datetime, float]] = []
        self.arousal_modifiers: Dict[str, float] = {}  # Sources affecting arousal
        self.last_update_time = datetime.now()
        
        # Learning and memory thresholds
        self.optimal_learning_arousal = 0.6  # Optimal arousal for learning
        self.memory_formation_threshold = 0.4  # Minimum arousal for memory formation
        
        # Developmental parameters
        self.novelty_sensitivity = 1.0
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register handlers for arousal-related events"""
        self.event_bus.subscribe("perception_input", self._handle_perception)
        self.event_bus.subscribe("emotion_update", self._handle_emotion)
        self.event_bus.subscribe("system_cycle", self._handle_system_cycle)
        self.event_bus.subscribe("development_update", self._handle_development_update)
    
    def _handle_perception(self, message: Message):
        """Handle perception input events to update arousal based on stimulation"""
        stimulation_level = message.content.get("stimulation_level", 0.0)
        novelty = message.content.get("novelty", 0.0)
        
        # Novelty increases the stimulation impact
        effective_stimulation = stimulation_level * (1 + (novelty * self.novelty_sensitivity))
        
        # Record stimulation
        self.stimulation_history.append((datetime.now(), effective_stimulation))
        self.stimulation_history = self._prune_history(self.stimulation_history)
        
        # Calculate arousal change based on stimulation
        current_arousal = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].current_value
        
        # Higher current arousal means smaller increases (diminishing returns)
        # Lower current arousal means bigger decreases (floor effect)
        arousal_change = effective_stimulation * (1 - (current_arousal * 0.5)) * 0.1
        
        # Update arousal level
        self.homeostatic_system.update_need(
            HomeostaticNeedType.AROUSAL,
            arousal_change,
            f"Perception input: stim={stimulation_level:.2f}, nov={novelty:.2f}"
        )
        
        # Check if arousal is out of balance
        self._check_arousal_balance()
    
    def _handle_emotion(self, message: Message):
        """Handle emotion updates to modify arousal based on emotional state"""
        emotion_type = message.content.get("emotion_type", "neutral")
        intensity = message.content.get("intensity", 0.0)
        
        # Emotional effects on arousal - different emotions have different effects
        arousal_change = 0.0
        
        # High arousal emotions
        if emotion_type in ["joy", "anger", "fear", "surprise"]:
            arousal_change = intensity * 0.15
        # Low arousal emotions
        elif emotion_type in ["sadness", "contentment"]:
            arousal_change = -intensity * 0.1
        # Neutral emotions
        else:
            arousal_change = 0.0
        
        # Update arousal with emotional effect
        if abs(arousal_change) > 0.01:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.AROUSAL,
                arousal_change,
                f"Emotional response: {emotion_type} ({intensity:.2f})"
            )
            
            # Store as modifier
            self.arousal_modifiers[f"emotion_{emotion_type}"] = arousal_change
        
        # Check arousal balance
        self._check_arousal_balance()
    
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle events to update arousal naturally"""
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update_time).total_seconds()
        
        # Natural arousal decay toward setpoint
        arousal_need = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL]
        deviation = arousal_need.current_value - arousal_need.setpoint
        
        # Decay based on deviation from setpoint (faster when further from setpoint)
        decay_amount = -deviation * self.decay_rate * (time_delta / 60.0)
        
        # Apply decay
        if abs(decay_amount) > 0.001:  # Only if meaningful change
            self.homeostatic_system.update_need(
                HomeostaticNeedType.AROUSAL,
                decay_amount,
                "Natural arousal regulation"
            )
            self.last_update_time = now
            
            # Update arousal modifiers
            for modifier in list(self.arousal_modifiers.keys()):
                self.arousal_modifiers[modifier] *= 0.9  # Decay all modifiers
                if abs(self.arousal_modifiers[modifier]) < 0.01:
                    del self.arousal_modifiers[modifier]
            
            # Publish current arousal state
            self._publish_arousal_state()
    
    def _handle_development_update(self, message: Message):
        """Adapt arousal parameters based on developmental stage"""
        development_level = message.content.get("development_level", 0.0)
        
        # Update homeostatic setpoints based on development
        self.homeostatic_system.adapt_to_development(development_level)
        
        # Adjust arousal parameters based on development
        # Young minds have higher novelty sensitivity but less arousal regulation
        if development_level < 0.3:  # Infant/early child
            self.novelty_sensitivity = 1.5
            self.decay_rate = 0.02  # Slower return to baseline
            self.optimal_learning_arousal = 0.7  # Higher arousal needed for learning
        elif development_level < 0.6:  # Child/adolescent
            self.novelty_sensitivity = 1.2
            self.decay_rate = 0.03
            self.optimal_learning_arousal = 0.6
        else:  # Adult
            self.novelty_sensitivity = 0.8
            self.decay_rate = 0.04  # Faster return to baseline
            self.optimal_learning_arousal = 0.5  # Can learn effectively at lower arousal
            
        logger.info(f"Arousal control adapted to development level {development_level:.2f}")
    
    def _check_arousal_balance(self):
        """Check if arousal is out of balance and trigger appropriate responses"""
        arousal_need = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL]
        
        # Over-arousal (over-stimulated)
        if arousal_need.is_excessive and arousal_need.urgency > 0.5:
            self._trigger_over_arousal_response(arousal_need.urgency)
        
        # Under-arousal (under-stimulated)
        elif arousal_need.is_deficient and arousal_need.urgency > 0.5:
            self._trigger_under_arousal_response(arousal_need.urgency)
    
    def _trigger_over_arousal_response(self, urgency: float):
        """Trigger appropriate responses to over-arousal"""
        # Create response message
        response = HomeostaticResponse(
            need_type=HomeostaticNeedType.AROUSAL,
            response_type="over_arousal_regulation",
            intensity=urgency,
            description="Reducing sensitivity to stimulation due to over-arousal",
            expected_effect={
                HomeostaticNeedType.AROUSAL: -0.2 * urgency,
                HomeostaticNeedType.COGNITIVE_LOAD: -0.1 * urgency
            },
            priority=int(urgency * 10)
        )
        
        # Publish response message
        response_message = Message(
            sender="arousal_controller",
            message_type="homeostatic_response",
            content={
                "response": response.model_dump(),
                "current_arousal": self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].current_value,
                "setpoint": self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].setpoint
            },
            priority=response.priority
        )
        self.event_bus.publish(response_message)
        
        # Publish attention modulation message
        attention_message = Message(
            sender="arousal_controller",
            message_type="attention_modulation",
            content={
                "focus_narrowing": urgency * 0.5,  # Narrow focus when over-aroused
                "distraction_resistance": urgency * 0.3,  # Increase resistance to distraction
                "reason": "Over-arousal regulation"
            }
        )
        self.event_bus.publish(attention_message)
        
        logger.info(f"Over-arousal response triggered: {urgency:.2f} intensity")
    
    def _trigger_under_arousal_response(self, urgency: float):
        """Trigger appropriate responses to under-arousal"""
        # Create response message
        response = HomeostaticResponse(
            need_type=HomeostaticNeedType.AROUSAL,
            response_type="under_arousal_regulation",
            intensity=urgency,
            description="Increasing sensitivity to stimulation due to under-arousal",
            expected_effect={
                HomeostaticNeedType.AROUSAL: 0.15 * urgency,
                HomeostaticNeedType.NOVELTY: 0.2 * urgency
            },
            priority=int(urgency * 8)
        )
        
        # Publish response message
        response_message = Message(
            sender="arousal_controller",
            message_type="homeostatic_response",
            content={
                "response": response.model_dump(),
                "current_arousal": self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].current_value,
                "setpoint": self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].setpoint
            },
            priority=response.priority
        )
        self.event_bus.publish(response_message)
        
        # Publish attention modulation message
        attention_message = Message(
            sender="arousal_controller",
            message_type="attention_modulation",
            content={
                "focus_narrowing": -urgency * 0.3,  # Broaden focus when under-aroused
                "novelty_bias": urgency * 0.4,  # Increase bias towards novel stimuli
                "reason": "Under-arousal regulation"
            }
        )
        self.event_bus.publish(attention_message)
        
        logger.info(f"Under-arousal response triggered: {urgency:.2f} intensity")
    
    def _publish_arousal_state(self):
        """Publish current arousal state to the event bus"""
        arousal_need = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL]
        
        # Calculate learning efficiency based on arousal level
        learning_efficiency = 1.0 - abs(arousal_need.current_value - self.optimal_learning_arousal) / self.optimal_learning_arousal
        learning_efficiency = max(0.1, min(1.0, learning_efficiency))
        
        # Determine if memory formation is enabled
        memory_formation_enabled = arousal_need.current_value >= self.memory_formation_threshold
        
        arousal_message = Message(
            sender="arousal_controller",
            message_type="arousal_state_update",
            content={
                "current_arousal": arousal_need.current_value,
                "setpoint": arousal_need.setpoint,
                "is_excessive": arousal_need.is_excessive,
                "is_deficient": arousal_need.is_deficient,
                "learning_efficiency": learning_efficiency,
                "memory_formation_enabled": memory_formation_enabled,
                "current_modifiers": self.arousal_modifiers
            }
        )
        self.event_bus.publish(arousal_message)
    
    def _prune_history(self, history: List[Tuple[datetime, float]], max_age_minutes: int = 10) -> List[Tuple[datetime, float]]:
        """Remove old entries from the stimulation history"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        return [item for item in history if item[0] > cutoff_time]
    
    def calculate_learning_rate_modifier(self) -> float:
        """Calculate a modifier for learning rate based on current arousal level"""
        arousal_need = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL]
        
        # Distance from optimal (normalized to 0-1 range)
        distance_from_optimal = abs(arousal_need.current_value - self.optimal_learning_arousal)
        normalized_distance = min(1.0, distance_from_optimal / self.optimal_learning_arousal)
        
        # Learning rate modifier (1.0 is normal, <1.0 is slower, >1.0 is faster)
        return 1.0 - (normalized_distance * 0.8)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the arousal controller"""
        arousal_need = self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL]
        return {
            "arousal_level": arousal_need.current_value,
            "arousal_setpoint": arousal_need.setpoint,
            "is_excessive": arousal_need.is_excessive,
            "is_deficient": arousal_need.is_deficient,
            "urgency": arousal_need.urgency,
            "learning_efficiency": self.calculate_learning_rate_modifier(),
            "novelty_sensitivity": self.novelty_sensitivity,
            "memory_formation_enabled": arousal_need.current_value >= self.memory_formation_threshold,
            "arousal_modifiers": self.arousal_modifiers,
            "optimal_learning_arousal": self.optimal_learning_arousal
        }
    
    def load_state(self, state_dict: StateDict) -> None:
        """Load state from the provided state dictionary"""
        if "arousal_level" in state_dict:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.AROUSAL,
                state_dict["arousal_level"] - 
                self.homeostatic_system.needs[HomeostaticNeedType.AROUSAL].current_value,
                "State loaded"
            )
            
        if "arousal_modifiers" in state_dict:
            self.arousal_modifiers = state_dict["arousal_modifiers"]
            
        if "novelty_sensitivity" in state_dict:
            self.novelty_sensitivity = state_dict["novelty_sensitivity"]
            
        if "optimal_learning_arousal" in state_dict:
            self.optimal_learning_arousal = state_dict["optimal_learning_arousal"]
