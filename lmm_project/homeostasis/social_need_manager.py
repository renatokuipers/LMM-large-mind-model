from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import logging
import math
import random

from lmm_project.core.message import Message
from lmm_project.core.event_bus import EventBus
from lmm_project.core.types import DevelopmentalStage, StateDict, RelationshipType
from .models import HomeostaticSystem, HomeostaticNeedType, HomeostaticResponse, NeedState

logger = logging.getLogger(__name__)

class SocialNeedManager:
    """
    Manages the social needs and social interaction balance of the cognitive system.
    
    The Social Need Manager:
    - Tracks need for social interaction
    - Manages attachment formation
    - Regulates social learning opportunities
    - Signals when social deficits exist
    - Balances social needs with other cognitive processes
    
    Social needs are analogous to a developing child's need for caregiving and social learning.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        initial_social_need: float = 0.5,
        satiation_rate: float = 0.15,
        deficit_growth_rate: float = 0.02
    ):
        self.event_bus = event_bus
        self.homeostatic_system = HomeostaticSystem()
        self.homeostatic_system.initialize_needs()
        
        # Initialize social need state
        social_need = self.homeostatic_system.needs.get(HomeostaticNeedType.SOCIAL)
        if social_need:
            social_need.current_value = initial_social_need
            social_need.last_updated = datetime.now()
        
        # Social regulation parameters
        self.satiation_rate = satiation_rate  # How quickly social needs are satisfied
        self.deficit_growth_rate = deficit_growth_rate  # How quickly social needs grow
        self.last_interaction_time = datetime.now()
        self.last_update_time = datetime.now()
        
        # Relationship tracking
        self.relationship_strength: Dict[str, float] = {"mother": 0.7}  # Start with connection to Mother
        self.interaction_history: List[Dict[str, Any]] = []
        self.social_learning_opportunities: List[Dict[str, Any]] = []
        
        # Developmental parameters
        self.attachment_formed = False
        self.attachment_type = "secure"  # Default to secure attachment
        self.social_preference = "caregiver"  # Initial preference for caregivers
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register handlers for social need related events"""
        self.event_bus.subscribe("mother_interaction", self._handle_mother_interaction)
        self.event_bus.subscribe("social_interaction", self._handle_social_interaction)
        self.event_bus.subscribe("system_cycle", self._handle_system_cycle)
        self.event_bus.subscribe("development_update", self._handle_development_update)
        self.event_bus.subscribe("emotion_update", self._handle_emotion_update)
    
    def _handle_mother_interaction(self, message: Message):
        """Handle interactions with the mother/caregiver"""
        interaction_type = message.content.get("interaction_type", "unknown")
        intensity = message.content.get("intensity", 0.5)
        duration = message.content.get("duration", 1.0)  # in minutes
        emotional_tone = message.content.get("emotional_tone", "neutral")
        
        # Calculate social need satisfaction based on interaction
        satisfaction = intensity * min(1.0, duration / 2.0) * self.satiation_rate
        
        # Adjust satisfaction based on emotional tone
        if emotional_tone in ["warm", "loving", "supportive"]:
            satisfaction *= 1.5
        elif emotional_tone in ["cold", "distant", "dismissive"]:
            satisfaction *= 0.5
        
        # Update social need
        self.homeostatic_system.update_need(
            HomeostaticNeedType.SOCIAL,
            -satisfaction,  # Negative because we're satisfying the need
            f"Mother interaction: {interaction_type}"
        )
        
        # Track interaction
        self.last_interaction_time = datetime.now()
        self.interaction_history.append({
            "timestamp": self.last_interaction_time,
            "partner": "mother",
            "type": interaction_type,
            "satisfaction": satisfaction,
            "emotional_tone": emotional_tone
        })
        
        # Update relationship strength with mother
        self._update_relationship_strength("mother", intensity * 0.05)
        
        # Check if this was a learning opportunity
        if message.content.get("is_teaching", False):
            self._record_learning_opportunity("mother", interaction_type, intensity)
        
        # Publish current social state
        self._publish_social_state()
        
        # Check attachment formation early in development
        if not self.attachment_formed and len(self.interaction_history) > 10:
            self._assess_attachment_formation()
    
    def _handle_social_interaction(self, message: Message):
        """Handle general social interactions with various entities"""
        partner = message.content.get("partner", "unknown")
        interaction_type = message.content.get("interaction_type", "unknown")
        intensity = message.content.get("intensity", 0.3)
        emotional_tone = message.content.get("emotional_tone", "neutral")
        
        # Calculate social need satisfaction based on interaction
        # Non-mother interactions typically have less impact on social needs
        partner_type = message.content.get("partner_type", "peer")
        
        # Different impact based on partner type and current social preference
        satisfaction_modifier = 1.0
        if partner_type == self.social_preference:
            satisfaction_modifier = 1.5
        elif partner_type != self.social_preference and partner_type != "caregiver":
            satisfaction_modifier = 0.7
        
        satisfaction = intensity * satisfaction_modifier * self.satiation_rate * 0.7
        
        # Update social need
        self.homeostatic_system.update_need(
            HomeostaticNeedType.SOCIAL,
            -satisfaction,  # Negative because we're satisfying the need
            f"Social interaction with {partner} ({partner_type})"
        )
        
        # Track interaction
        self.last_interaction_time = datetime.now()
        self.interaction_history.append({
            "timestamp": self.last_interaction_time,
            "partner": partner,
            "partner_type": partner_type,
            "type": interaction_type,
            "satisfaction": satisfaction,
            "emotional_tone": emotional_tone
        })
        
        # Update relationship strength
        relationship_change = intensity * 0.03
        self._update_relationship_strength(partner, relationship_change)
        
        # Check if this was a learning opportunity
        if message.content.get("is_learning_opportunity", False):
            self._record_learning_opportunity(partner, interaction_type, intensity)
        
        # Publish current social state
        self._publish_social_state()
    
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle events to update social needs naturally"""
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update_time).total_seconds()
        
        # Social needs naturally increase over time without interaction
        time_since_interaction = (now - self.last_interaction_time).total_seconds()
        
        # Social need increases faster the longer without interaction
        # Using a logarithmic growth to model mounting social need
        if time_since_interaction > 300:  # 5 minutes
            growth_factor = math.log10(max(1, time_since_interaction / 300))
            social_need_increase = self.deficit_growth_rate * growth_factor * (time_delta / 60.0)
            
            # Apply social need increase
            self.homeostatic_system.update_need(
                HomeostaticNeedType.SOCIAL,
                social_need_increase,
                "Natural social need growth"
            )
            self.last_update_time = now
            
            # Check if social need is urgent
            social_need = self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL]
            if social_need.is_deficient and social_need.urgency > 0.7:
                self._signal_social_need()
            
            # Publish current social state
            self._publish_social_state()
    
    def _handle_development_update(self, message: Message):
        """Adapt social parameters based on developmental stage"""
        development_level = message.content.get("development_level", 0.0)
        stage = message.content.get("stage", "prenatal")
        
        # Update homeostatic setpoints based on development
        self.homeostatic_system.adapt_to_development(development_level)
        
        # Adjust social parameters based on development stage
        # Different developmental stages have different social needs
        if stage == "infant" or development_level < 0.3:
            # Infants have high social needs, primarily with caregivers
            self.deficit_growth_rate = 0.04  # Fast social need growth
            self.social_preference = "caregiver"
            
        elif stage == "child" or (development_level >= 0.3 and development_level < 0.6):
            # Children begin to value peer interactions more
            self.deficit_growth_rate = 0.03
            self.social_preference = "peer" if random.random() > 0.4 else "caregiver"
            
        elif stage == "adolescent" or (development_level >= 0.6 and development_level < 0.8):
            # Adolescents strongly prefer peer interactions
            self.deficit_growth_rate = 0.025
            self.social_preference = "peer"
            
        else:  # Adult
            # Adults have more balanced social needs
            self.deficit_growth_rate = 0.015
            # Social preference becomes more varied and situational
            self.social_preference = random.choice(["peer", "authority", "caregiver"])
            
        logger.info(
            f"Social needs adapted to development level {development_level:.2f} (stage: {stage}): "
            f"deficit_rate={self.deficit_growth_rate}, preference={self.social_preference}"
        )
    
    def _handle_emotion_update(self, message: Message):
        """Handle updates to emotional state that affect social needs"""
        emotion_type = message.content.get("emotion_type", "neutral")
        intensity = message.content.get("intensity", 0.5)
        
        # Some emotions increase social needs, others decrease them
        social_need_change = 0.0
        
        # Emotions that typically increase social needs
        if emotion_type in ["sadness", "fear", "loneliness"]:
            social_need_change = intensity * 0.15
            
        # Emotions that typically decrease social needs
        elif emotion_type in ["anger", "disgust"]:
            social_need_change = -intensity * 0.1
            
        # Apply change if significant
        if abs(social_need_change) > 0.01:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.SOCIAL,
                social_need_change,
                f"Emotional impact: {emotion_type}"
            )
            
            # If emotional state is creating urgent social needs, signal
            if social_need_change > 0.1:
                self._signal_emotional_social_need(emotion_type, intensity)
    
    def _signal_social_need(self):
        """Signal that there is an urgent social need"""
        social_need = self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL]
        
        # Create homeostatic response
        response = HomeostaticResponse(
            need_type=HomeostaticNeedType.SOCIAL,
            response_type="seek_social_interaction",
            intensity=social_need.urgency,
            description="Seeking social interaction due to social deficit",
            expected_effect={
                HomeostaticNeedType.SOCIAL: -0.3 * social_need.urgency,
                HomeostaticNeedType.AROUSAL: 0.1 * social_need.urgency
            },
            priority=int(social_need.urgency * 10)
        )
        
        # Publish response message
        response_message = Message(
            sender="social_need_manager",
            message_type="homeostatic_response",
            content={
                "response": response.model_dump(),
                "current_social_need": social_need.current_value,
                "time_since_interaction": (datetime.now() - self.last_interaction_time).total_seconds() / 60.0,
                "preferred_partner_type": self.social_preference
            },
            priority=response.priority
        )
        self.event_bus.publish(response_message)
        
        # Publish social behavior message
        behavior_message = Message(
            sender="social_need_manager",
            message_type="social_behavior_request",
            content={
                "behavior_type": "seek_interaction",
                "urgency": social_need.urgency,
                "preferred_partner": self._get_preferred_partner(),
                "reason": "Social need fulfillment"
            }
        )
        self.event_bus.publish(behavior_message)
        
        logger.info(f"Social need signal sent: urgency={social_need.urgency:.2f}")
    
    def _signal_emotional_social_need(self, emotion_type: str, intensity: float):
        """Signal social need based on emotional state"""
        social_need = self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL]
        
        # Map emotion to appropriate social response
        response_type = "seek_comfort"
        if emotion_type == "fear":
            response_type = "seek_protection"
        elif emotion_type == "sadness":
            response_type = "seek_comfort"
        elif emotion_type == "loneliness":
            response_type = "seek_connection"
        
        # Create emotional response message
        emotion_message = Message(
            sender="social_need_manager",
            message_type="emotional_social_need",
            content={
                "emotion": emotion_type,
                "intensity": intensity,
                "response_type": response_type,
                "current_social_need": social_need.current_value,
                "preferred_partner": "mother" if self.social_preference == "caregiver" else "peer"
            }
        )
        self.event_bus.publish(emotion_message)
    
    def _update_relationship_strength(self, partner: str, change: float):
        """Update relationship strength with a particular partner"""
        current_strength = self.relationship_strength.get(partner, 0.0)
        new_strength = max(0.0, min(1.0, current_strength + change))
        self.relationship_strength[partner] = new_strength
        
        # If this is a significant relationship change, publish an update
        if abs(change) > 0.05:
            relationship_message = Message(
                sender="social_need_manager",
                message_type="relationship_update",
                content={
                    "partner": partner,
                    "strength": new_strength,
                    "change": change,
                    "relationship_rank": self._get_relationship_rank(partner)
                }
            )
            self.event_bus.publish(relationship_message)
    
    def _record_learning_opportunity(self, partner: str, interaction_type: str, quality: float):
        """Record a social learning opportunity"""
        self.social_learning_opportunities.append({
            "timestamp": datetime.now(),
            "partner": partner,
            "type": interaction_type,
            "quality": quality
        })
        
        # Notify learning module about social learning opportunity
        learning_message = Message(
            sender="social_need_manager",
            message_type="social_learning_opportunity",
            content={
                "partner": partner,
                "interaction_type": interaction_type,
                "quality": quality,
                "relationship_strength": self.relationship_strength.get(partner, 0.0)
            }
        )
        self.event_bus.publish(learning_message)
    
    def _assess_attachment_formation(self):
        """Assess attachment formation based on interaction history"""
        # Simple attachment assessment based on early interactions
        # In a more complex implementation, this would analyze interaction patterns
        
        # Count types of interactions
        responsive_count = 0
        non_responsive_count = 0
        
        recent_interactions = self.interaction_history[-10:]
        for interaction in recent_interactions:
            if interaction["partner"] == "mother":
                if interaction["emotional_tone"] in ["warm", "loving", "supportive"]:
                    responsive_count += 1
                elif interaction["emotional_tone"] in ["cold", "distant", "dismissive"]:
                    non_responsive_count += 1
        
        # Determine attachment type
        if responsive_count >= 7:
            self.attachment_type = "secure"
        elif responsive_count >= 4 and non_responsive_count >= 4:
            self.attachment_type = "anxious"
        else:
            self.attachment_type = "avoidant"
        
        self.attachment_formed = True
        
        # Publish attachment formation message
        attachment_message = Message(
            sender="social_need_manager",
            message_type="attachment_formed",
            content={
                "attachment_type": self.attachment_type,
                "responsive_interactions": responsive_count,
                "non_responsive_interactions": non_responsive_count,
                "total_interactions": len(self.interaction_history)
            }
        )
        self.event_bus.publish(attachment_message)
        
        logger.info(f"Attachment formed: {self.attachment_type} (responsive: {responsive_count}, non-responsive: {non_responsive_count})")
    
    def _get_preferred_partner(self) -> str:
        """Get the currently preferred interaction partner"""
        if self.social_preference == "caregiver":
            return "mother"
        
        # If preference is peer or other, check relationships
        strong_relationships = [
            partner for partner, strength in self.relationship_strength.items()
            if strength > 0.5 and partner != "mother"
        ]
        
        if strong_relationships:
            return random.choice(strong_relationships)
        
        # Default to mother if no strong relationships
        return "mother"
    
    def _get_relationship_rank(self, partner: str) -> int:
        """Get the rank of a relationship compared to others (1 is strongest)"""
        sorted_relationships = sorted(
            self.relationship_strength.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (rel_partner, _) in enumerate(sorted_relationships):
            if rel_partner == partner:
                return i + 1
                
        return len(sorted_relationships) + 1
    
    def _publish_social_state(self):
        """Publish current social state to the event bus"""
        social_need = self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL]
        
        social_message = Message(
            sender="social_need_manager",
            message_type="social_state_update",
            content={
                "current_social_need": social_need.current_value,
                "setpoint": social_need.setpoint,
                "is_deficient": social_need.is_deficient,
                "is_excessive": social_need.is_excessive,
                "minutes_since_interaction": (datetime.now() - self.last_interaction_time).total_seconds() / 60.0,
                "social_preference": self.social_preference,
                "top_relationships": self._get_top_relationships(3)
            }
        )
        self.event_bus.publish(social_message)
    
    def _get_top_relationships(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the top N strongest relationships"""
        sorted_relationships = sorted(
            self.relationship_strength.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"partner": partner, "strength": strength}
            for partner, strength in sorted_relationships[:n]
        ]
    
    def request_social_interaction(self, partner: str, intensity: float, reason: str) -> bool:
        """
        Request a social interaction (from modules or external systems)
        
        Arguments:
            partner: Who to interact with
            intensity: How intense the interaction should be (0.0-1.0)
            reason: Reason for requesting interaction
            
        Returns:
            bool: True if request was accepted
        """
        # Create interaction request
        request_message = Message(
            sender="social_need_manager",
            message_type="social_interaction_request",
            content={
                "partner": partner,
                "intensity": intensity,
                "reason": reason,
                "current_social_need": self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL].current_value
            }
        )
        self.event_bus.publish(request_message)
        
        logger.info(f"Requested social interaction with {partner}: {reason}")
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the social need manager"""
        social_need = self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL]
        return {
            "social_need": social_need.current_value,
            "social_setpoint": social_need.setpoint,
            "is_deficient": social_need.is_deficient,
            "is_excessive": social_need.is_excessive,
            "urgency": social_need.urgency,
            "social_preference": self.social_preference,
            "attachment_type": self.attachment_type,
            "attachment_formed": self.attachment_formed,
            "relationships": self.relationship_strength,
            "recent_interactions": self.interaction_history[-5:] if self.interaction_history else []
        }
    
    def load_state(self, state_dict: StateDict) -> None:
        """Load state from the provided state dictionary"""
        if "social_need" in state_dict:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.SOCIAL,
                state_dict["social_need"] - 
                self.homeostatic_system.needs[HomeostaticNeedType.SOCIAL].current_value,
                "State loaded"
            )
            
        if "relationships" in state_dict:
            self.relationship_strength = state_dict["relationships"]
            
        if "social_preference" in state_dict:
            self.social_preference = state_dict["social_preference"]
            
        if "attachment_type" in state_dict:
            self.attachment_type = state_dict["attachment_type"]
            
        if "attachment_formed" in state_dict:
            self.attachment_formed = state_dict["attachment_formed"] 
