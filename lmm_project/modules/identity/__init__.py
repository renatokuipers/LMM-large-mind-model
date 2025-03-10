# Identity module 

# TODO: Implement the identity module factory function to return an integrated IdentitySystem
# This module should be responsible for self-concept, personal narrative,
# preferences, and personality trait development.

# TODO: Create IdentitySystem class that integrates all identity sub-components:
# - self_concept: representation of self-knowledge and beliefs about self
# - personal_narrative: autobiographical story that creates continuity of self
# - preferences: likes, dislikes, and value judgments
# - personality_traits: stable patterns of thinking, feeling, and behaving

# TODO: Implement development tracking for identity
# Identity should develop from minimal self-awareness in early stages
# to complex, integrated self-concept in adulthood

# TODO: Connect identity module to memory, emotion, and social modules
# Identity should be informed by autobiographical memories, emotional
# responses, and social feedback

# TODO: Implement stability vs. change dynamics
# The system should maintain some stability in identity while
# allowing for appropriate change and growth over time

from typing import Optional, Dict, Any, List, Union, Tuple
import uuid
from datetime import datetime

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.identity.models import (
    IdentityState,
    IdentityNeuralState,
    SelfConcept as SelfConceptModel,
    PersonalNarrative as PersonalNarrativeModel,
    PreferenceSystem,
    PersonalityProfile,
    Value,
    PersonalityTrait,
    IdentityChange
)
from lmm_project.modules.identity.self_concept import SelfConcept
from lmm_project.modules.identity.personal_narrative import PersonalNarrative
from lmm_project.modules.identity.preferences import Preferences
from lmm_project.modules.identity.personality_traits import PersonalityTraits


class IdentitySystem(BaseModule):
    """
    Integrated identity system that brings together all identity components
    
    The IdentitySystem coordinates the self-concept, personal narrative,
    preferences, and personality traits to create a coherent sense of self.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic self-recognition",
        0.2: "Simple self-descriptions and preferences",
        0.4: "Coherent self-concept and emerging narrative",
        0.6: "Integrated identity with social comparison",
        0.8: "Stable personality and autobiographical continuity",
        1.0: "Fully differentiated identity with values integration"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the identity system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize component modules
        self.self_concept = SelfConcept(
            module_id=f"{module_id}.self_concept",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.personal_narrative = PersonalNarrative(
            module_id=f"{module_id}.personal_narrative",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.preferences = Preferences(
            module_id=f"{module_id}.preferences",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.personality_traits = PersonalityTraits(
            module_id=f"{module_id}.personality_traits",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize identity state
        self.identity_state = IdentityState(
            self_concept=SelfConceptModel(),
            personal_narrative=PersonalNarrativeModel(),
            preference_system=PreferenceSystem(),
            personality_profile=PersonalityProfile(),
            module_id=module_id,
            developmental_level=development_level
        )
        
        # Identity integration metrics
        self.identity_integration = 0.3  # Starts with low integration
        self.identity_stability = 0.2    # Starts with low stability
        self.identity_clarity = 0.4      # Starts with moderate clarity
        
        # Register for event subscriptions
        if event_bus:
            event_bus.subscribe(
                message_type=f"{module_id}.self_concept", 
                callback=self._handle_self_concept_event
            )
            event_bus.subscribe(
                message_type=f"{module_id}.personal_narrative", 
                callback=self._handle_narrative_event
            )
            event_bus.subscribe(
                message_type=f"{module_id}.preferences", 
                callback=self._handle_preference_event
            )
            event_bus.subscribe(
                message_type=f"{module_id}.personality_traits", 
                callback=self._handle_personality_event
            )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the identity system
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dict with processing results
        """
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "status": "error",
                "message": "Input must be a dictionary"
            }
        
        # Extract process ID if provided
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract target component and operation
        component = input_data.get("component", "identity")
        operation = input_data.get("operation", "")
        
        # Route to appropriate component or handle at system level
        if component == "self_concept":
            return self.self_concept.process_input(input_data)
            
        elif component == "personal_narrative":
            return self.personal_narrative.process_input(input_data)
            
        elif component == "preferences":
            return self.preferences.process_input(input_data)
            
        elif component == "personality_traits":
            return self.personality_traits.process_input(input_data)
            
        elif component == "identity":
            # Handle identity-level operations
            if operation == "get_state":
                return self._get_identity_state(input_data, process_id)
                
            elif operation == "update_integration":
                return self._update_identity_integration(input_data, process_id)
                
            elif operation == "query_identity":
                return self._query_identity(input_data, process_id)
                
            else:
                return {
                    "status": "error",
                    "message": f"Unknown operation for identity component: {operation}",
                    "process_id": process_id
                }
                
        else:
            return {
                "status": "error",
                "message": f"Unknown component: {component}",
                "process_id": process_id
            }
    
    def _get_identity_state(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Get the current identity state
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with identity state information
        """
        # Get current states from components
        self_concept_state = self.self_concept.get_state()
        narrative_state = self.personal_narrative.get_state()
        preference_state = self.preferences.get_state()
        personality_state = self.personality_traits.get_state()
        
        # Combine into unified identity state
        identity_state = {
            "status": "success",
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "identity_integration": self.identity_integration,
            "identity_stability": self.identity_stability,
            "identity_clarity": self.identity_clarity,
            "components": {
                "self_concept": self_concept_state,
                "personal_narrative": narrative_state,
                "preferences": preference_state,
                "personality_traits": personality_state
            },
            "process_id": process_id
        }
        
        return identity_state
    
    def _update_identity_integration(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Update the integration of identity components
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with update results
        """
        # This operation forces an update of identity integration metrics
        self._calculate_identity_integration()
        
        return {
            "status": "success",
            "module_id": self.module_id,
            "identity_integration": self.identity_integration,
            "identity_stability": self.identity_stability,
            "identity_clarity": self.identity_clarity,
            "process_id": process_id
        }
    
    def _query_identity(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query information about identity
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        query_type = input_data.get("query_type", "summary")
        
        if query_type == "summary":
            # Provide a summary of the identity
            return {
                "status": "success",
                "module_id": self.module_id,
                "developmental_level": self.development_level,
                "identity_integration": self.identity_integration,
                "identity_stability": self.identity_stability,
                "identity_clarity": self.identity_clarity,
                "component_levels": {
                    "self_concept": self.self_concept.development_level,
                    "personal_narrative": self.personal_narrative.development_level,
                    "preferences": self.preferences.development_level,
                    "personality_traits": self.personality_traits.development_level
                },
                "process_id": process_id
            }
            
        elif query_type == "milestones":
            # Return developmental milestones
            return {
                "status": "success",
                "module_id": self.module_id,
                "identity_milestones": self.development_milestones,
                "self_concept_milestones": self.self_concept.development_milestones,
                "narrative_milestones": self.personal_narrative.development_milestones,
                "preferences_milestones": self.preferences.development_milestones,
                "personality_milestones": self.personality_traits.development_milestones,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _calculate_identity_integration(self):
        """Calculate the level of integration between identity components"""
        # Identity integration increases with development
        base_integration = 0.3 + (self.development_level * 0.5)
        
        # Identity stability increases with development
        base_stability = 0.2 + (self.development_level * 0.6)
        
        # Identity clarity increases with development
        base_clarity = 0.4 + (self.development_level * 0.5)
        
        # Component development also influences integration
        component_levels = [
            self.self_concept.development_level,
            self.personal_narrative.development_level,
            self.preferences.development_level,
            self.personality_traits.development_level
        ]
        
        # Average component development
        avg_component_level = sum(component_levels) / len(component_levels)
        
        # Variance in component development (less variance = more integration)
        variance = sum((level - avg_component_level) ** 2 for level in component_levels) / len(component_levels)
        variance_factor = max(0.0, 1.0 - (variance * 5.0))  # Low variance gives higher factor
        
        # Calculate final metrics
        self.identity_integration = min(1.0, base_integration * 0.7 + variance_factor * 0.3)
        self.identity_stability = min(1.0, base_stability * 0.8 + variance_factor * 0.2)
        self.identity_clarity = min(1.0, base_clarity * 0.6 + variance_factor * 0.4)
        
        # Update identity state
        self.identity_state.identity_integration = self.identity_integration
        self.identity_state.identity_stability = self.identity_stability
        self.identity_state.identity_clarity = self.identity_clarity
        self.identity_state.developmental_level = self.development_level
        self.identity_state.last_updated = datetime.now()
    
    def _handle_self_concept_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the self-concept component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in self-concept development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update identity integration metrics
            self._calculate_identity_integration()
            
            # Possibly trigger identity-level milestone events
            self._check_identity_milestones()
            
    def _handle_narrative_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the personal narrative component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in narrative development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update identity integration metrics
            self._calculate_identity_integration()
            
            # Possibly trigger identity-level milestone events
            self._check_identity_milestones()
            
    def _handle_preference_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the preferences component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in preference development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update identity integration metrics
            self._calculate_identity_integration()
            
            # Possibly trigger identity-level milestone events
            self._check_identity_milestones()
            
    def _handle_personality_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the personality traits component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in personality development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update identity integration metrics
            self._calculate_identity_integration()
            
            # Possibly trigger identity-level milestone events
            self._check_identity_milestones()
    
    def _check_identity_milestones(self):
        """Check if any identity-level milestones have been reached"""
        # Identity development is influenced by component development
        component_levels = [
            self.self_concept.development_level,
            self.personal_narrative.development_level,
            self.preferences.development_level,
            self.personality_traits.development_level
        ]
        
        # Average component development
        avg_component_level = sum(component_levels) / len(component_levels)
        
        # Get previous development level
        old_level = self.development_level
        
        # Update overall development level (weighted average of components and existing level)
        self.development_level = (self.development_level * 0.3) + (avg_component_level * 0.7)
        self.development_level = max(0.0, min(1.0, self.development_level))
        
        # Check if crossed a milestone
        for level in sorted(self.development_milestones.keys()):
            if old_level < level <= self.development_level:
                milestone = self.development_milestones[level]
                
                # Publish milestone event if we have an event bus
                if self.event_bus:
                    self.event_bus.publish({
                        "sender": self.module_id,
                        "message_type": "development_milestone",
                        "content": {
                            "module": "identity",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                print(f"Identity Development Milestone: {milestone} (level {level})")
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module and its components
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        
        # Update component development levels
        self.self_concept.update_development(amount)
        self.personal_narrative.update_development(amount)
        self.preferences.update_development(amount)
        self.personality_traits.update_development(amount)
        
        # Check identity milestones and update integration
        self._check_identity_milestones()
        self._calculate_identity_integration()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the identity system
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "identity_integration": self.identity_integration,
            "identity_stability": self.identity_stability,
            "identity_clarity": self.identity_clarity,
            "components": {
                "self_concept": self.self_concept.get_state(),
                "personal_narrative": self.personal_narrative.get_state(),
                "preferences": self.preferences.get_state(),
                "personality_traits": self.personality_traits.get_state()
            }
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "identity_integration": self.identity_integration,
            "identity_stability": self.identity_stability,
            "identity_clarity": self.identity_clarity,
            "components": {
                "self_concept": self.self_concept.save_state(),
                "personal_narrative": self.personal_narrative.save_state(),
                "preferences": self.preferences.save_state(),
                "personality_traits": self.personality_traits.save_state()
            },
            "identity_state": self.identity_state.dict()
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state to load
        """
        # Load basic state
        self.module_id = state["module_id"]
        self.development_level = state["developmental_level"]
        self.identity_integration = state["identity_integration"]
        self.identity_stability = state["identity_stability"]
        self.identity_clarity = state["identity_clarity"]
        
        # Load component states
        if "components" in state:
            components = state["components"]
            
            if "self_concept" in components:
                self.self_concept.load_state(components["self_concept"])
                
            if "personal_narrative" in components:
                self.personal_narrative.load_state(components["personal_narrative"])
                
            if "preferences" in components:
                self.preferences.load_state(components["preferences"])
                
            if "personality_traits" in components:
                self.personality_traits.load_state(components["personality_traits"])
        
        # Load identity state
        if "identity_state" in state:
            try:
                self.identity_state = IdentityState(**state["identity_state"])
            except Exception as e:
                print(f"Error loading identity state: {e}")


def get_module(
    module_id: str = "identity",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "IdentitySystem":
    """
    Factory function to create and initialize an identity module
    
    Args:
        module_id: Unique identifier for this module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level (0.0-1.0)
        
    Returns:
        Initialized identity module
    """
    module = IdentitySystem(module_id=module_id, event_bus=event_bus)
    module.set_development_level(development_level)
    return module 