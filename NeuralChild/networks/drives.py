# drives.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("DrivesNetwork")

class DrivesNetwork(BaseNetwork):
    """
    Forces propelling behavior network using FFNN-like processing
    
    The drives network represents fundamental motivational forces that
    propel behavior. These basic needs and motivations act as internal
    stimuli that drive the child to interact with the environment.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.2,
        name: str = "Drives"
    ):
        """Initialize the drives network"""
        super().__init__(
            network_type=NetworkType.DRIVES,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Initialize basic drives with their current levels and baselines
        self.drives = {
            "physiological": {
                "level": 0.7,         # Current level (0-1)
                "baseline": 0.6,      # Default level
                "urgency": 0.8,       # How quickly this becomes urgent
                "decay_rate": 0.05,   # How quickly it decreases
                "satisfaction": 0.0,  # Current satisfaction level
                "last_updated": datetime.now()
            },
            "safety": {
                "level": 0.6,
                "baseline": 0.5,
                "urgency": 0.6,
                "decay_rate": 0.02,
                "satisfaction": 0.0,
                "last_updated": datetime.now()
            },
            "attachment": {
                "level": 0.8,
                "baseline": 0.7,
                "urgency": 0.7,
                "decay_rate": 0.03,
                "satisfaction": 0.0,
                "last_updated": datetime.now()
            },
            "exploration": {
                "level": 0.5,
                "baseline": 0.4,
                "urgency": 0.3,
                "decay_rate": 0.01,
                "satisfaction": 0.0,
                "last_updated": datetime.now()
            },
            "rest": {
                "level": 0.4,
                "baseline": 0.5,
                "urgency": 0.4,
                "decay_rate": 0.04,
                "satisfaction": 0.0,
                "last_updated": datetime.now()
            }
        }
        
        # Drive history for tracking patterns
        self.drive_history = {name: deque(maxlen=50) for name in self.drives}
        
        # Developmental adjustments
        self.developmental_stage = "infancy"  # Default stage
        
        logger.info(f"Initialized drives network with {len(self.drives)} basic drives")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to update drive states"""
        # Apply natural decay to all drives since last update
        self._apply_natural_decay()
        
        if not self.input_buffer:
            return self._calculate_activation()
        
        # Process each input item
        for input_item in self.input_buffer:
            data = input_item["data"]
            source = input_item.get("source", "unknown")
            
            # Process mother interactions that satisfy drives
            if "mother_response" in data:
                self._process_mother_response(data["mother_response"])
            
            # Process emotional states that influence drives
            if "emotional_state" in data:
                self._process_emotional_influence(data["emotional_state"])
            
            # Process direct drive satisfaction signals
            if "drive_satisfaction" in data:
                for drive, satisfaction in data["drive_satisfaction"].items():
                    if drive in self.drives:
                        self.satisfy_drive(drive, satisfaction)
            
            # Process environmental inputs
            if "environment" in data:
                self._process_environment(data["environment"])
            
            # Handle explicit developmental stage updates
            if "developmental_stage" in data:
                self.update_developmental_stage(data["developmental_stage"])
        
        # Clear input buffer
        self.input_buffer = []
        
        return self._calculate_activation()
    
    def _calculate_activation(self) -> Dict[str, Any]:
        """Calculate overall network activation based on drive states"""
        # Update drive history
        for name, drive in self.drives.items():
            self.drive_history[name].append((datetime.now(), drive["level"]))
        
        # Calculate the overall activation as weighted average of drive levels
        drive_levels = [drive["level"] * drive["urgency"] for drive in self.drives.values()]
        if drive_levels:
            overall_activation = sum(drive_levels) / sum(drive["urgency"] for drive in self.drives.values())
        else:
            overall_activation = 0.0
        
        # Find the dominant drive (highest level * urgency)
        dominant_drive = max(self.drives.keys(), 
                            key=lambda d: self.drives[d]["level"] * self.drives[d]["urgency"])
        
        return {
            "network_activation": overall_activation,
            "drive_levels": {name: drive["level"] for name, drive in self.drives.items()},
            "dominant_drive": dominant_drive
        }
    
    def _apply_natural_decay(self) -> None:
        """Apply natural decay to all drives based on time elapsed"""
        current_time = datetime.now()
        
        for name, drive in self.drives.items():
            # Calculate time elapsed since last update
            time_diff = (current_time - drive["last_updated"]).total_seconds()
            hours_elapsed = time_diff / 3600  # Convert to hours
            
            # Apply decay based on elapsed time
            decay_amount = drive["decay_rate"] * hours_elapsed
            
            # Drives naturally tend toward their baseline
            if drive["level"] > drive["baseline"]:
                drive["level"] = max(drive["baseline"], drive["level"] - decay_amount)
            else:
                drive["level"] = min(drive["baseline"], drive["level"] + decay_amount)
            
            # Reset satisfaction if too much time has passed
            if hours_elapsed > 1:  # Reset satisfaction after an hour
                drive["satisfaction"] = max(0.0, drive["satisfaction"] - 0.5)
            
            drive["last_updated"] = current_time
    
    def _process_mother_response(self, mother_response: Dict[str, Any]) -> None:
        """Process mother's response to satisfy drives"""
        # Extract relevant aspects of mother's response
        if "verbal" in mother_response and "physical_actions" in mother_response.get("non_verbal", {}):
            verbal = mother_response["verbal"].get("text", "")
            actions = mother_response["non_verbal"].get("physical_actions", [])
            
            # Check for physiological need satisfaction
            if any(word in verbal.lower() for word in ["feed", "food", "milk", "drink", "eat"]):
                self.satisfy_drive("physiological", 0.3)
            
            # Check for safety satisfaction
            if any(word in verbal.lower() for word in ["safe", "protect", "okay", "alright"]):
                self.satisfy_drive("safety", 0.2)
            
            # Check for attachment satisfaction through actions
            attachment_actions = ["hug", "hold", "cuddle", "kiss", "smile"]
            if any(action in " ".join(actions).lower() for action in attachment_actions):
                self.satisfy_drive("attachment", 0.4)
            
            # Check for exploration satisfaction
            if any(word in verbal.lower() for word in ["look", "see", "discover", "try", "explore"]):
                self.satisfy_drive("exploration", 0.3)
    
    def _process_emotional_influence(self, emotional_state: Dict[str, float]) -> None:
        """Process how emotions influence drives"""
        # Fear increases safety drive
        if "fear" in emotional_state and emotional_state["fear"] > 0.5:
            self.drives["safety"]["level"] = min(1.0, self.drives["safety"]["level"] + 0.2)
            self.drives["safety"]["urgency"] = min(1.0, self.drives["safety"]["urgency"] + 0.1)
        
        # Joy increases exploration drive
        if "joy" in emotional_state and emotional_state["joy"] > 0.6:
            self.drives["exploration"]["level"] = min(1.0, self.drives["exploration"]["level"] + 0.15)
        
        # Trust decreases attachment need temporarily
        if "trust" in emotional_state and emotional_state["trust"] > 0.7:
            self.satisfy_drive("attachment", 0.3)
    
    def _process_environment(self, environment: Dict[str, Any]) -> None:
        """Process environmental factors that affect drives"""
        # Examples of environmental factors that could affect drives
        if "time_since_feeding" in environment:
            hours = environment["time_since_feeding"]
            if hours > 2:
                self.drives["physiological"]["level"] = min(1.0, self.drives["physiological"]["level"] + 0.2)
        
        if "noise_level" in environment and environment["noise_level"] > 0.7:
            # Loud environments reduce safety feeling
            self.drives["safety"]["level"] = max(0.0, self.drives["safety"]["level"] - 0.1)
        
        if "social_isolation" in environment and environment["social_isolation"] > 0.5:
            # Social isolation increases attachment need
            self.drives["attachment"]["level"] = min(1.0, self.drives["attachment"]["level"] + 0.2)
    
    def satisfy_drive(self, drive_name: str, satisfaction_amount: float) -> None:
        """Satisfy a specific drive"""
        if drive_name in self.drives:
            drive = self.drives[drive_name]
            
            # Increase satisfaction
            drive["satisfaction"] = min(1.0, drive["satisfaction"] + satisfaction_amount)
            
            # Decrease drive level based on satisfaction
            level_decrease = satisfaction_amount * (1.0 - drive["satisfaction"] * 0.3)
            drive["level"] = max(drive["baseline"] * 0.5, drive["level"] - level_decrease)
            
            # Update timestamp
            drive["last_updated"] = datetime.now()
    
    def update_developmental_stage(self, stage: str) -> None:
        """Update drives based on developmental stage changes"""
        if stage == self.developmental_stage:
            return
        
        self.developmental_stage = stage
        
        # Adjust drive parameters based on developmental stage
        if stage == "infancy":
            # Infants have high physiological and attachment needs
            self.drives["physiological"]["baseline"] = 0.7
            self.drives["physiological"]["urgency"] = 0.9
            self.drives["attachment"]["baseline"] = 0.8
            self.drives["attachment"]["urgency"] = 0.8
            self.drives["exploration"]["baseline"] = 0.3
            
        elif stage == "early_childhood":
            # Young children have increasing exploration drives
            self.drives["physiological"]["baseline"] = 0.6
            self.drives["physiological"]["urgency"] = 0.7
            self.drives["attachment"]["baseline"] = 0.6
            self.drives["attachment"]["urgency"] = 0.6
            self.drives["exploration"]["baseline"] = 0.6
            self.drives["exploration"]["urgency"] = 0.5
            
        elif stage == "middle_childhood":
            # Older children have more balanced drives
            self.drives["physiological"]["baseline"] = 0.5
            self.drives["physiological"]["urgency"] = 0.6
            self.drives["attachment"]["baseline"] = 0.5
            self.drives["attachment"]["urgency"] = 0.5
            self.drives["exploration"]["baseline"] = 0.7
            self.drives["exploration"]["urgency"] = 0.7
            
        logger.info(f"Updated drives for developmental stage: {stage}")
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Find the dominant drive
        dominant_drive = max(self.drives.keys(), 
                           key=lambda d: self.drives[d]["level"] * self.drives[d]["urgency"])
        
        # Calculate urgency scores (level * urgency)
        urgency_scores = {
            name: drive["level"] * drive["urgency"] 
            for name, drive in self.drives.items()
        }
        
        # Check if any drives are in critical state (high level + high urgency)
        critical_drives = [
            name for name, drive in self.drives.items()
            if drive["level"] * drive["urgency"] > 0.8
        ]
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "drive_levels": {name: drive["level"] for name, drive in self.drives.items()},
            "dominant_drive": dominant_drive,
            "urgency_scores": urgency_scores,
            "critical_drives": critical_drives,
            "developmental_stage": self.developmental_stage
        }
        
    def get_drive_status(self) -> Dict[str, Dict[str, float]]:
        """Get the current status of all drives"""
        return {
            name: {
                "level": drive["level"],
                "urgency": drive["urgency"],
                "satisfaction": drive["satisfaction"]
            } for name, drive in self.drives.items()
        }