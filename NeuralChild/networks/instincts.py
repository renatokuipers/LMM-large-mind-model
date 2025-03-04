# instincts.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("InstinctsNetwork")

class InstinctResponse:
    """A hardwired response to a stimulus"""
    def __init__(
        self,
        name: str,
        trigger_pattern: List[str],
        response: str,
        strength: float = 1.0,
        developmental_onset: float = 0.0,
        developmental_decline: Optional[float] = None
    ):
        self.name = name
        self.trigger_pattern = trigger_pattern
        self.response = response
        self.strength = strength  # How strongly this instinct expresses (0-1)
        self.developmental_onset = developmental_onset  # Age in days when it appears
        self.developmental_decline = developmental_decline  # Age when it starts to decline
        self.activation_count = 0
        self.last_activated = None
    
    def matches(self, stimulus: str) -> bool:
        """Check if this instinct is triggered by the stimulus"""
        stimulus_lower = stimulus.lower()
        return any(trigger in stimulus_lower for trigger in self.trigger_pattern)
    
    def activate(self) -> Tuple[str, float]:
        """Activate this instinct and return the response"""
        self.activation_count += 1
        self.last_activated = datetime.now()
        return self.response, self.strength
    
    def developmental_adjustment(self, age_days: float) -> None:
        """Adjust instinct strength based on developmental stage"""
        # Check if instinct is active at this age
        if age_days < self.developmental_onset:
            self.strength = 0.0
            return
        
        # Full strength during peak period
        if self.developmental_decline is None or age_days < self.developmental_decline:
            self.strength = 1.0
            return
        
        # Gradual decline after decline age
        decline_period = 100.0  # Days over which instinct declines
        if self.developmental_decline and age_days >= self.developmental_decline:
            days_past_decline = age_days - self.developmental_decline
            decline_factor = max(0.0, 1.0 - (days_past_decline / decline_period))
            self.strength = max(0.1, decline_factor)  # Minimum residual strength

class InstinctsNetwork(BaseNetwork):
    """
    Hardwired responses network using LSTM-like processing
    
    The instincts network represents innate, hardwired responses to stimuli.
    These are present from birth and may strengthen or weaken during development,
    but they do not require learning.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 0.5,  # Slower learning for instincts
        activation_threshold: float = 0.1,  # Lower threshold for instincts
        name: str = "Instincts"
    ):
        """Initialize the instincts network"""
        super().__init__(
            network_type=NetworkType.INSTINCTS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Initialize basic instincts
        self.instincts = self._initialize_instincts()
        
        # Instinct activation history
        self.instinct_history = deque(maxlen=100)
        
        # Current active instincts
        self.active_instincts = []
        
        # Instinct parameters
        self.instinct_dominance = 0.8  # How much instincts override other processes
        self.developmental_age = 0.0  # Current developmental age in days
        
        logger.info(f"Initialized instincts network with {len(self.instincts)} hardwired responses")
    
    def _initialize_instincts(self) -> List[InstinctResponse]:
        """Initialize the basic instinct responses"""
        instincts = []
        
        # Rooting reflex (turns head toward touch on cheek to find breast)
        instincts.append(InstinctResponse(
            name="rooting_reflex",
            trigger_pattern=["touch", "cheek", "face", "hungry"],
            response="turn_head_toward_stimulus",
            developmental_onset=0.0,
            developmental_decline=90.0  # Declines after about 3 months
        ))
        
        # Sucking reflex
        instincts.append(InstinctResponse(
            name="sucking_reflex",
            trigger_pattern=["mouth", "food", "hungry", "breast", "bottle", "milk"],
            response="sucking_motion",
            developmental_onset=0.0,
            developmental_decline=120.0  # Declines after about 4 months
        ))
        
        # Grasping reflex
        instincts.append(InstinctResponse(
            name="grasping_reflex",
            trigger_pattern=["touch", "palm", "finger", "hand"],
            response="grasp_tightly",
            developmental_onset=0.0,
            developmental_decline=180.0  # Declines after about 6 months
        ))
        
        # Moro (startle) reflex
        instincts.append(InstinctResponse(
            name="startle_reflex",
            trigger_pattern=["loud", "sudden", "fall", "startle", "noise"],
            response="extend_arms_and_legs",
            developmental_onset=0.0,
            developmental_decline=120.0  # Declines after about 4 months
        ))
        
        # Crying response to discomfort
        instincts.append(InstinctResponse(
            name="crying_response",
            trigger_pattern=["pain", "discomfort", "hungry", "tired", "dirty", "alone"],
            response="cry_loudly",
            developmental_onset=0.0,
            developmental_decline=None  # Never fully goes away
        ))
        
        # Orienting to faces
        instincts.append(InstinctResponse(
            name="face_orienting",
            trigger_pattern=["face", "eyes", "smile", "mother", "father", "person"],
            response="focus_gaze_on_face",
            developmental_onset=0.0,
            developmental_decline=None  # Never fully goes away
        ))
        
        # Withdrawal from pain
        instincts.append(InstinctResponse(
            name="pain_withdrawal",
            trigger_pattern=["pain", "hot", "sharp", "hurt"],
            response="withdraw_quickly",
            developmental_onset=0.0,
            developmental_decline=None  # Never goes away
        ))
        
        # Social smile response
        instincts.append(InstinctResponse(
            name="social_smile",
            trigger_pattern=["smile", "happy", "face", "talk", "play"],
            response="smile_back",
            developmental_onset=30.0,  # Emerges around 1 month
            developmental_decline=None  # Never fully goes away
        ))
        
        # Seeking attachment figure when distressed
        instincts.append(InstinctResponse(
            name="attachment_seeking",
            trigger_pattern=["afraid", "pain", "scared", "lonely", "anxious"],
            response="seek_caregiver",
            developmental_onset=60.0,  # Emerges around 2 months
            developmental_decline=None  # Never fully goes away
        ))
        
        # Stranger anxiety
        instincts.append(InstinctResponse(
            name="stranger_anxiety",
            trigger_pattern=["stranger", "unknown", "unfamiliar", "new person"],
            response="show_distress",
            developmental_onset=180.0,  # Emerges around 6 months
            developmental_decline=730.0  # Declines after about 2 years
        ))
        
        return instincts
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to trigger instinctual responses"""
        # Reset active instincts for this cycle
        self.active_instincts = []
        
        if not self.input_buffer:
            return {"network_activation": 0.0, "active_instincts": []}
        
        # Collect all stimuli
        all_stimuli = []
        for input_item in self.input_buffer:
            data = input_item["data"]
            
            # Extract stimuli from various sources
            if "stimuli" in data:
                all_stimuli.extend(data["stimuli"])
            elif "percepts" in data:
                all_stimuli.extend(data["percepts"])
            elif "stimulus" in data:
                all_stimuli.append(data["stimulus"])
            
            # Process any direct instinct activations
            if "activate_instinct" in data:
                instinct_name = data["activate_instinct"]
                for instinct in self.instincts:
                    if instinct.name == instinct_name and instinct.strength > 0.2:
                        response, strength = instinct.activate()
                        self.active_instincts.append({
                            "name": instinct.name,
                            "response": response,
                            "strength": strength,
                            "stimulus": "direct_activation"
                        })
        
        # Process each stimulus against all instincts
        for stimulus in all_stimuli:
            for instinct in self.instincts:
                if instinct.matches(stimulus) and instinct.strength > 0.2:
                    response, strength = instinct.activate()
                    self.active_instincts.append({
                        "name": instinct.name,
                        "response": response,
                        "strength": strength,
                        "stimulus": stimulus
                    })
        
        # Add to history
        for active in self.active_instincts:
            self.instinct_history.append({
                "timestamp": datetime.now(),
                "instinct": active["name"],
                "strength": active["strength"],
                "stimulus": active["stimulus"]
            })
        
        # Calculate overall activation level
        if self.active_instincts:
            # Sum of active instinct strengths, capped at 1.0
            total_strength = sum(i["strength"] for i in self.active_instincts)
            activation = min(1.0, total_strength * self.instinct_dominance)
        else:
            activation = 0.0
        
        # Clear input buffer
        self.input_buffer = []
        
        return {
            "network_activation": activation,
            "active_instincts": self.active_instincts
        }
    
    def update_development(self, age_days: float) -> None:
        """Update instincts based on developmental age"""
        self.developmental_age = age_days
        
        # Update individual instinct strengths
        for instinct in self.instincts:
            instinct.developmental_adjustment(age_days)
        
        # Overall instinct dominance decreases with age as learning increases
        self.instinct_dominance = max(0.3, 0.8 - (age_days / 400))
        
        logger.info(f"Updated instincts for age {age_days:.1f} days")
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Get strongest active instinct
        strongest_instinct = None
        max_strength = 0.0
        
        for active in self.active_instincts:
            if active["strength"] > max_strength:
                max_strength = active["strength"]
                strongest_instinct = active
        
        # Get all instincts with their current strengths
        all_instinct_states = {
            instinct.name: {
                "strength": instinct.strength,
                "activation_count": instinct.activation_count,
                "active": any(active["name"] == instinct.name for active in self.active_instincts)
            } for instinct in self.instincts
        }
        
        # Frequency analysis of recent instinct activations
        recent_activations = {}
        for event in self.instinct_history:
            name = event["instinct"]
            if name not in recent_activations:
                recent_activations[name] = 0
            recent_activations[name] += 1
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "active_instincts": self.active_instincts,
            "strongest_instinct": strongest_instinct,
            "instinct_states": all_instinct_states,
            "instinct_dominance": self.instinct_dominance,
            "developmental_age": self.developmental_age,
            "recent_activation_frequency": recent_activations
        }
    
    def get_instinct(self, name: str) -> Optional[InstinctResponse]:
        """Get a specific instinct by name"""
        for instinct in self.instincts:
            if instinct.name == name:
                return instinct
        return None