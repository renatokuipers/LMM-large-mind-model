# TODO: Implement the Awareness class to monitor internal and external states
# This component should maintain awareness of:
# - External perceptual inputs
# - Internal emotional states
# - Current goals and motivations
# - Ongoing cognitive processes
# - Current attentional focus

# TODO: Implement developmental progression of awareness:
# - Basic stimulus awareness in early stages
# - Growing peripheral awareness in childhood
# - Self-directed awareness in adolescence
# - Integrated awareness of multiple states in adulthood

# TODO: Create mechanisms for:
# - State monitoring: Track current states across cognitive systems
# - Change detection: Identify significant changes in monitored states
# - Awareness broadcasting: Make aware states available to other systems
# - Attentional modulation: Prioritize awareness based on attention

# TODO: Implement levels of awareness:
# - Subliminal: Below threshold of awareness
# - Peripheral: At the edges of awareness
# - Focal: At the center of awareness
# - Meta-awareness: Awareness of being aware

# TODO: Connect to attention and global workspace systems
# Awareness should be influenced by attention and should feed
# information into the global workspace for conscious processing

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.consciousness.models import AwarenessState
from lmm_project.modules.consciousness.neural_net import ConsciousnessAttention

class Awareness(BaseModule):
    """
    Maintains awareness of internal and external states
    
    This module monitors the state of various cognitive and perceptual
    systems, determining what enters awareness and is available for
    conscious processing.
    
    Developmental progression:
    - Basic stimulus awareness in early stages
    - Growing peripheral awareness in mid stages
    - Self-directed awareness in later stages
    - Integrated awareness of multiple states in advanced stages
    """
    
    # Developmental milestones for awareness
    development_milestones = {
        0.0: "stimulus_awareness",      # Basic awareness of stimuli
        0.25: "peripheral_awareness",   # Awareness of background information
        0.5: "self_awareness",          # Awareness of internal states
        0.75: "integrated_awareness",   # Integrated awareness across domains
        0.9: "meta_awareness"           # Awareness of being aware
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the awareness module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="awareness", event_bus=event_bus)
        
        # Initialize awareness state
        self.state = AwarenessState()
        
        # Neural mechanisms for awareness
        self.input_dim = 128  # Default input dimension
        self.attention = ConsciousnessAttention(self.input_dim)
        
        # State change detection thresholds - adjusted with development
        self.change_thresholds = {
            "external": 0.2,
            "internal": 0.3,
            "social": 0.4,
            "temporal": 0.3
        }
        
        # Initialize monitored states
        self.monitored_states = {
            "perceptual": {},
            "emotional": {},
            "cognitive": {},
            "motivational": {}
        }
        
        # Last update time for tracking temporal changes
        self.last_update = datetime.now()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.event_bus.subscribe("perception_input", self._handle_perception)
            self.event_bus.subscribe("emotion_state", self._handle_emotion)
            self.event_bus.subscribe("cognitive_state", self._handle_cognitive)
            self.event_bus.subscribe("goal_state", self._handle_motivation)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update awareness states
        
        Args:
            input_data: Dictionary containing state information
            
        Returns:
            Dictionary with the results of awareness processing
        """
        # Extract input type and data
        input_type = input_data.get("type", "unknown")
        state_data = input_data.get("state", {})
        source = input_data.get("source", "unknown")
        
        # Update the current time
        current_time = datetime.now()
        time_delta = (current_time - self.last_update).total_seconds()
        self.last_update = current_time
        
        # Update temporal awareness based on time since last update
        self.state.temporal_awareness = min(1.0, self.state.temporal_awareness + 0.05 * time_delta)
        
        # Process based on input type
        if input_type in ["perception", "sensory"]:
            self._update_external_awareness(state_data, source)
        elif input_type in ["emotion", "motivation", "cognitive"]:
            self._update_internal_awareness(state_data, source, input_type)
        elif input_type == "social":
            self._update_social_awareness(state_data, source)
        
        # Store in monitored states
        if input_type in self.monitored_states:
            self.monitored_states[input_type][source] = state_data
        
        # Apply attention based on current development level
        self._apply_attentional_focus()
        
        # Update state with monitored information
        self.state.monitored_states = self._filter_by_awareness_level()
        
        # Create result with current awareness state
        result = {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "state": self.state.model_dump(),
            "developmental_level": self.developmental_level,
            "current_milestone": self._get_current_milestone()
        }
        
        # Publish awareness state if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                msg_type="awareness_state",
                content=result
            )
        
        return result
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.developmental_level
        new_level = super().update_development(amount)
        
        # Update thresholds and parameters based on development
        dev_multiplier = (1.0 - 0.5 * new_level)  # Lower thresholds as development increases
        self.change_thresholds = {
            "external": 0.2 * dev_multiplier,
            "internal": 0.3 * dev_multiplier,
            "social": 0.4 * dev_multiplier,
            "temporal": 0.3 * dev_multiplier
        }
        
        # Expand awareness capabilities at key milestones
        if previous_level < 0.25 and new_level >= 0.25:
            # Enable peripheral awareness
            self.state.external_awareness = max(self.state.external_awareness, 0.3)
        
        if previous_level < 0.5 and new_level >= 0.5:
            # Enable self-awareness
            self.state.internal_awareness = max(self.state.internal_awareness, 0.4)
        
        if previous_level < 0.75 and new_level >= 0.75:
            # Enable social awareness
            self.state.social_awareness = max(self.state.social_awareness, 0.3)
        
        return new_level
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_awareness"
        for level, name in sorted(self.development_milestones.items()):
            if self.developmental_level >= level:
                milestone = name
        return milestone
    
    def _update_external_awareness(self, state_data: Dict[str, Any], source: str) -> None:
        """Update external awareness based on perceptual input"""
        # Calculate state change magnitude
        prev_state = self.monitored_states.get("perceptual", {}).get(source, {})
        change_magnitude = self._calculate_state_change(prev_state, state_data)
        
        # Update external awareness based on change magnitude
        if change_magnitude > self.change_thresholds["external"]:
            # Significant change increases awareness
            self.state.external_awareness = min(1.0, self.state.external_awareness + 0.1 * change_magnitude)
        else:
            # Small changes lead to gradual decrease in awareness
            self.state.external_awareness = max(0.1, self.state.external_awareness - 0.02)
        
        # Store updated state
        if "perceptual" not in self.monitored_states:
            self.monitored_states["perceptual"] = {}
        self.monitored_states["perceptual"][source] = state_data
    
    def _update_internal_awareness(self, state_data: Dict[str, Any], source: str, state_type: str) -> None:
        """Update internal awareness based on emotional, cognitive, or motivational state"""
        # Calculate state change magnitude
        prev_state = self.monitored_states.get(state_type, {}).get(source, {})
        change_magnitude = self._calculate_state_change(prev_state, state_data)
        
        # Update internal awareness based on change magnitude and development
        dev_factor = max(0.1, self.developmental_level)  # Higher development enables better internal awareness
        
        if change_magnitude > self.change_thresholds["internal"]:
            # Significant change increases awareness, scaled by development
            awareness_increase = 0.1 * change_magnitude * dev_factor
            self.state.internal_awareness = min(1.0, self.state.internal_awareness + awareness_increase)
        else:
            # Small changes lead to gradual decrease in awareness
            awareness_decrease = 0.02 * (1.0 - dev_factor)  # Less decrease with higher development
            self.state.internal_awareness = max(0.1, self.state.internal_awareness - awareness_decrease)
        
        # Store updated state
        if state_type not in self.monitored_states:
            self.monitored_states[state_type] = {}
        self.monitored_states[state_type][source] = state_data
    
    def _update_social_awareness(self, state_data: Dict[str, Any], source: str) -> None:
        """Update social awareness based on social perception or interaction"""
        # Social awareness requires higher development
        if self.developmental_level < 0.3:
            return
            
        # Calculate state change magnitude
        prev_state = self.monitored_states.get("social", {}).get(source, {})
        change_magnitude = self._calculate_state_change(prev_state, state_data)
        
        # Update social awareness based on change magnitude
        if change_magnitude > self.change_thresholds["social"]:
            # Significant change increases awareness
            dev_scaling = (self.developmental_level - 0.3) / 0.7  # Scale by development level
            awareness_increase = 0.1 * change_magnitude * dev_scaling
            self.state.social_awareness = min(1.0, self.state.social_awareness + awareness_increase)
        else:
            # Small changes lead to gradual decrease in awareness
            self.state.social_awareness = max(0.0, self.state.social_awareness - 0.02)
        
        # Store updated state
        if "social" not in self.monitored_states:
            self.monitored_states["social"] = {}
        self.monitored_states["social"][source] = state_data
    
    def _calculate_state_change(self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> float:
        """Calculate the magnitude of change between previous and current states"""
        if not prev_state:
            return 0.5  # Moderate change for first input
            
        # Compare keys in both states
        common_keys = set(prev_state.keys()) & set(curr_state.keys())
        if not common_keys:
            return 0.7  # High change for completely different states
        
        # Calculate change for each common key
        total_change = 0.0
        for key in common_keys:
            prev_value = prev_state[key]
            curr_value = curr_state[key]
            
            # Handle different value types
            if isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
                # Normalize numerical change to [0,1]
                max_val = max(abs(prev_value), abs(curr_value), 1.0)
                key_change = abs(prev_value - curr_value) / max_val
                total_change += min(1.0, key_change)
            elif prev_value != curr_value:
                # Binary change for non-numeric values
                total_change += 1.0
        
        # Normalize by number of common keys
        return min(1.0, total_change / len(common_keys))
    
    def _apply_attentional_focus(self) -> None:
        """Apply attentional mechanisms to determine what is in focus"""
        # Simple attentional mechanism weighted by state type
        focus = {}
        
        # Weights adjusted by developmental level
        weights = {
            "perceptual": 0.7 - 0.3 * self.developmental_level,  # Decreases with development
            "emotional": 0.3 + 0.2 * self.developmental_level,   # Increases with development
            "cognitive": 0.2 + 0.4 * self.developmental_level,   # Increases with development
            "motivational": 0.3 + 0.1 * self.developmental_level # Slightly increases with development
        }
        
        # Apply weights to determine attentional focus
        for state_type, states in self.monitored_states.items():
            if state_type in weights:
                weight = weights[state_type]
                for source, data in states.items():
                    # Create a composite key
                    focus_key = f"{state_type}:{source}"
                    # Assign attention weight, modulated by recency
                    recency = 1.0  # Could use timestamp for actual recency calculation
                    focus[focus_key] = weight * recency
        
        # Normalize focus values to sum to 1.0
        total_focus = sum(focus.values())
        if total_focus > 0:
            self.state.attentional_focus = {k: v/total_focus for k, v in focus.items()}
        else:
            self.state.attentional_focus = {}
    
    def _filter_by_awareness_level(self) -> Dict[str, Any]:
        """Filter monitored states based on awareness levels"""
        # Combine awareness levels to determine what is accessible
        awareness_result = {}
        
        # External awareness determines what perceptual information is available
        if self.state.external_awareness > 0.2:
            perceptual_states = self.monitored_states.get("perceptual", {})
            awareness_result["perceptual"] = {
                k: v for k, v in perceptual_states.items() 
                if k in self.state.attentional_focus and 
                self.state.attentional_focus[f"perceptual:{k}"] > 0.3 - 0.2 * self.state.external_awareness
            }
        
        # Internal awareness determines what emotional/cognitive information is available
        if self.state.internal_awareness > 0.3:
            # Add emotional states
            emotional_states = self.monitored_states.get("emotional", {})
            awareness_result["emotional"] = {
                k: v for k, v in emotional_states.items()
                if k in self.state.attentional_focus and
                self.state.attentional_focus.get(f"emotional:{k}", 0) > 0.4 - 0.3 * self.state.internal_awareness
            }
            
            # Add cognitive states
            cognitive_states = self.monitored_states.get("cognitive", {})
            awareness_result["cognitive"] = {
                k: v for k, v in cognitive_states.items()
                if k in self.state.attentional_focus and
                self.state.attentional_focus.get(f"cognitive:{k}", 0) > 0.4 - 0.3 * self.state.internal_awareness
            }
        
        # Social awareness for social information
        if self.state.social_awareness > 0.4:
            social_states = self.monitored_states.get("social", {})
            awareness_result["social"] = {
                k: v for k, v in social_states.items()
                if k in self.state.attentional_focus and
                self.state.attentional_focus.get(f"social:{k}", 0) > 0.5 - 0.4 * self.state.social_awareness
            }
        
        return awareness_result
        
    def _handle_perception(self, message: Message) -> None:
        """Handle perception input messages"""
        self.process_input({
            "type": "perception",
            "state": message.content,
            "source": message.source
        })
    
    def _handle_emotion(self, message: Message) -> None:
        """Handle emotion state messages"""
        self.process_input({
            "type": "emotion",
            "state": message.content,
            "source": message.source
        })
    
    def _handle_cognitive(self, message: Message) -> None:
        """Handle cognitive state messages"""
        self.process_input({
            "type": "cognitive",
            "state": message.content,
            "source": message.source
        })
    
    def _handle_motivation(self, message: Message) -> None:
        """Handle goal and motivation messages"""
        self.process_input({
            "type": "motivation",
            "state": message.content,
            "source": message.source
        }) 
