"""
Drives Module

This module implements basic motivational drives that energize behavior
and direct the system toward need satisfaction. Drives develop from simple
approach/avoidance in early stages to complex, integrated motivations in
later developmental stages.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import os
import json
import numpy as np
import torch
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.motivation.models import Drive, MotivationalState
from lmm_project.modules.motivation.neural_net import MotivationNeuralNetwork
from lmm_project.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class Drives(BaseModule):
    """
    Handles basic motivational drives
    
    This module maintains fundamental motivational drives,
    regulates their intensity, and generates activation
    signals based on current drive states.
    """
    
    # Developmental milestones for the drives module
    development_milestones = {
        0.0: "Basic approach/avoidance drives",
        0.2: "Physiological drives (energy, rest)",
        0.4: "Exploration and curiosity drives",
        0.6: "Social and affiliation drives",
        0.8: "Competence and mastery drives",
        1.0: "Self-actualization drives"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the drives module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id, 
            module_type="drives", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize drive storage
        self.drives: Dict[str, Drive] = {}
        
        # Last update timestamp
        self.last_update = datetime.now()
        
        # Drive activation thresholds
        self.activation_threshold = 0.7  # Drives above this intensity trigger active seeking
        
        # Neural network for drive processing
        self.neural_net = MotivationNeuralNetwork(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            network_type="drive",
            development_level=development_level
        )
        
        # Drive satisfaction data - tracks what satisfies each drive
        self.satisfaction_data: Dict[str, Set[str]] = {}
        
        # Storage path
        self.storage_dir = Path("storage/motivation/drives")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize basic drives based on development level
        self._initialize_drives()
        
        # Subscribe to messages
        if self.event_bus:
            self.subscribe_to_message("perception_result")
            self.subscribe_to_message("action_performed")
            self.subscribe_to_message("reward_received")
            self.subscribe_to_message("need_update")
            self.subscribe_to_message("drive_query")
    
    def _initialize_drives(self):
        """Initialize basic drives based on current development level"""
        # Always present drives
        basic_drives = [
            Drive(
                name="energy",
                description="Drive for maintaining energy levels",
                category="physiological",
                priority=0.9,
                decay_rate=0.02,
                satisfying_actions={"eat", "drink"}
            ),
            Drive(
                name="rest",
                description="Drive for rest and recovery",
                category="physiological",
                priority=0.8,
                decay_rate=0.015,
                satisfying_actions={"sleep", "rest"}
            ),
            Drive(
                name="safety",
                description="Drive for security and stability",
                category="security",
                priority=0.85,
                decay_rate=0.01,
                satisfying_actions={"seek_shelter", "avoid_danger"}
            )
        ]
        
        for drive in basic_drives:
            self.drives[drive.id] = drive
        
        # Development-dependent drives
        if self.development_level >= 0.2:
            self.drives[str(uuid.uuid4())] = Drive(
                name="comfort",
                description="Drive for physical comfort",
                category="physiological",
                priority=0.6,
                decay_rate=0.01,
                satisfying_actions={"adjust_temperature", "find_comfort"}
            )
        
        if self.development_level >= 0.4:
            self.drives[str(uuid.uuid4())] = Drive(
                name="curiosity",
                description="Drive for exploration and information",
                category="cognitive",
                priority=0.7,
                decay_rate=0.008,
                satisfying_actions={"explore", "learn", "investigate"}
            )
            
        if self.development_level >= 0.6:
            self.drives[str(uuid.uuid4())] = Drive(
                name="social",
                description="Drive for social connection",
                category="social",
                priority=0.75,
                decay_rate=0.005,
                satisfying_actions={"interact", "communicate", "bond"}
            )
            
            self.drives[str(uuid.uuid4())] = Drive(
                name="play",
                description="Drive for playful activity",
                category="cognitive",
                priority=0.6,
                decay_rate=0.01,
                satisfying_actions={"play", "experiment"}
            )
            
        if self.development_level >= 0.8:
            self.drives[str(uuid.uuid4())] = Drive(
                name="mastery",
                description="Drive for competence and skill development",
                category="cognitive",
                priority=0.7,
                decay_rate=0.005,
                satisfying_actions={"practice", "develop_skill"}
            )
            
            self.drives[str(uuid.uuid4())] = Drive(
                name="autonomy",
                description="Drive for self-directed action",
                category="cognitive",
                priority=0.65,
                decay_rate=0.005,
                satisfying_actions={"make_choices", "self_direct"}
            )
            
        if self.development_level >= 0.9:
            self.drives[str(uuid.uuid4())] = Drive(
                name="creativity",
                description="Drive for creative expression",
                category="cognitive",
                priority=0.6,
                decay_rate=0.004,
                satisfying_actions={"create", "innovate", "express"}
            )
            
            self.drives[str(uuid.uuid4())] = Drive(
                name="meaning",
                description="Drive for meaning and purpose",
                category="existential",
                priority=0.75,
                decay_rate=0.003,
                satisfying_actions={"reflect", "connect_meaning", "pursue_purpose"}
            )
        
        logger.info(f"Initialized {len(self.drives)} drives at development level {self.development_level}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update drive states
        
        Args:
            input_data: Input data with operation details
                Supported operations:
                - get_drives: Get current drive states
                - update_drive: Update a specific drive
                - get_active_drives: Get drives above activation threshold
                - satisfy_drive: Satisfy a specific drive
            
        Returns:
            Operation result with drive information
        """
        operation = input_data.get("operation", "")
        result = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "module_id": self.module_id
        }
        
        # Update all drives based on elapsed time
        self._update_all_drives()
        
        if operation == "get_drives":
            # Return all drive states
            drives_info = {
                drive_id: {
                    "name": drive.name,
                    "intensity": drive.intensity,
                    "priority": drive.priority,
                    "category": drive.category
                } for drive_id, drive in self.drives.items()
            }
            
            result.update({
                "status": "success",
                "drives": drives_info,
                "count": len(drives_info)
            })
            
        elif operation == "update_drive":
            # Update a specific drive
            drive_id = input_data.get("drive_id")
            change = input_data.get("change", 0.0)
            
            if drive_id and drive_id in self.drives:
                drive = self.drives[drive_id]
                drive.update_intensity(change)
                
                result.update({
                    "status": "success",
                    "drive_id": drive_id,
                    "name": drive.name,
                    "new_intensity": drive.intensity
                })
            else:
                result.update({
                    "status": "error",
                    "error": f"Drive not found: {drive_id}",
                    "available_drives": list(self.drives.keys())
                })
                
        elif operation == "get_active_drives":
            # Get drives above activation threshold
            active_drives = self._get_active_drives()
            
            result.update({
                "status": "success",
                "active_drives": active_drives,
                "count": len(active_drives),
                "activation_threshold": self.activation_threshold
            })
            
        elif operation == "satisfy_drive":
            # Satisfy a specific drive
            drive_id = input_data.get("drive_id")
            amount = input_data.get("amount", 0.3)
            
            if drive_id and drive_id in self.drives:
                drive = self.drives[drive_id]
                old_intensity = drive.intensity
                drive.update_intensity(-amount)  # Negative change = reduction in drive
                
                result.update({
                    "status": "success",
                    "drive_id": drive_id,
                    "name": drive.name,
                    "old_intensity": old_intensity,
                    "new_intensity": drive.intensity,
                    "satisfaction_amount": amount
                })
                
                # Publish drive satisfaction message
                if self.event_bus:
                    self.publish_message("drive_satisfied", {
                        "drive_id": drive_id,
                        "drive_name": drive.name,
                        "satisfaction_amount": amount
                    })
            else:
                result.update({
                    "status": "error",
                    "error": f"Drive not found: {drive_id}",
                    "available_drives": list(self.drives.keys())
                })
                
        else:
            result.update({
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "available_operations": ["get_drives", "update_drive", "get_active_drives", "satisfy_drive"]
            })
            
        return result
    
    def _update_all_drives(self):
        """Update all drives based on time elapsed since last update"""
        now = datetime.now()
        elapsed_seconds = (now - self.last_update).total_seconds()
        
        # Don't update if very little time has passed
        if elapsed_seconds < 0.1:
            return
            
        for drive_id, drive in self.drives.items():
            # Apply natural decay based on drive's decay rate
            drive.decay(elapsed_seconds)
        
        self.last_update = now
    
    def _get_active_drives(self) -> List[Dict[str, Any]]:
        """Get drives that are above the activation threshold"""
        active_drives = []
        
        for drive_id, drive in self.drives.items():
            if drive.intensity >= self.activation_threshold:
                active_drives.append({
                    "id": drive_id,
                    "name": drive.name,
                    "intensity": drive.intensity,
                    "priority": drive.priority,
                    "category": drive.category,
                    "urgency": drive.intensity * drive.priority  # Combined measure of importance
                })
                
        # Sort by urgency (intensity * priority)
        active_drives.sort(key=lambda d: d["urgency"], reverse=True)
        return active_drives
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase the development level
            
        Returns:
            New development level
        """
        old_level = self.development_level
        new_level = super().update_development(amount)
        
        # If we crossed a threshold, update available drives
        if int(old_level * 10) != int(new_level * 10):
            self._initialize_drives()
            
            # Also update neural network development
            self.neural_net.update_development(amount)
            
            logger.info(f"Drives module advanced to development level {new_level:.2f}")
            
        return new_level
    
    def _handle_message(self, message: Message):
        """
        Handle incoming messages
        
        Args:
            message: The message to process
        """
        if message.message_type == "perception_result":
            # Check if perception relates to drive satisfaction
            if message.content and "result" in message.content:
                perception = message.content["result"]
                self._process_perception(perception)
                
        elif message.message_type == "action_performed":
            # Check if action satisfies any drives
            if message.content and "action" in message.content:
                action = message.content["action"]
                self._process_action(action)
                
        elif message.message_type == "reward_received":
            # Reward may indicate drive satisfaction
            if message.content and "reward" in message.content:
                reward = message.content["reward"]
                self._process_reward(reward)
                
        elif message.message_type == "need_update":
            # Need satisfaction may affect drives
            if message.content and "need_id" in message.content:
                need_update = message.content
                self._process_need_update(need_update)
                
        elif message.message_type == "drive_query":
            # Direct query about drive states
            if message.content and "query_type" in message.content:
                query = message.content
                self._process_drive_query(query, message.id)
    
    def _process_perception(self, perception: Dict[str, Any]):
        """Process perception data for potential drive updates"""
        # Check if perception contains information relevant to drives
        if "type" in perception:
            if perception["type"] == "food" and "energy" in [d.name for d in self.drives.values()]:
                # Found food, potentially satisfy energy drive
                for drive_id, drive in self.drives.items():
                    if drive.name == "energy":
                        drive.update_intensity(-0.2)  # Reduce hunger drive
                        break
                        
            elif perception["type"] == "danger" and "safety" in [d.name for d in self.drives.values()]:
                # Detected danger, increase safety drive
                for drive_id, drive in self.drives.items():
                    if drive.name == "safety":
                        drive.update_intensity(0.3)  # Increase safety drive
                        break
                        
            elif perception["type"] == "novelty" and "curiosity" in [d.name for d in self.drives.values()]:
                # Found something novel, potentially satisfy curiosity
                for drive_id, drive in self.drives.items():
                    if drive.name == "curiosity":
                        drive.update_intensity(-0.15)  # Satisfy curiosity somewhat
                        break
    
    def _process_action(self, action: Dict[str, Any]):
        """Process action data for potential drive satisfaction"""
        if "name" in action:
            action_name = action["name"]
            
            # Check all drives to see if this action satisfies any
            for drive_id, drive in self.drives.items():
                if action_name in drive.satisfying_actions:
                    # This action satisfies this drive
                    satisfaction_amount = action.get("effectiveness", 0.3)
                    drive.update_intensity(-satisfaction_amount)
                    
                    # Publish drive satisfaction message
                    if self.event_bus:
                        self.publish_message("drive_satisfied", {
                            "drive_id": drive_id,
                            "drive_name": drive.name,
                            "action": action_name,
                            "satisfaction_amount": satisfaction_amount
                        })
    
    def _process_reward(self, reward: Dict[str, Any]):
        """Process reward data for potential drive adjustments"""
        # Rewards may indicate drive satisfaction
        if "magnitude" in reward and "source" in reward:
            magnitude = reward["magnitude"]
            source = reward["source"]
            
            # Check if this reward is associated with a drive
            for drive_id, drive in self.drives.items():
                if source in drive.satisfying_actions:
                    drive.update_intensity(-magnitude * 0.5)  # Adjust based on reward magnitude
    
    def _process_need_update(self, need_update: Dict[str, Any]):
        """Process need update for potential drive adjustments"""
        # Certain needs may be linked to drives
        need_id = need_update.get("need_id")
        need_name = need_update.get("need_name", "")
        satisfaction = need_update.get("satisfaction", 0.0)
        
        # Map need types to drive categories
        need_to_drive_map = {
            "physiological": ["energy", "rest", "comfort"],
            "safety": ["safety"],
            "social": ["social"],
            "esteem": ["mastery", "autonomy"],
            "cognitive": ["curiosity", "play"],
            "self_actualization": ["creativity", "meaning"]
        }
        
        # Update drives related to this need category
        need_category = need_update.get("category", "")
        if need_category in need_to_drive_map:
            related_drive_names = need_to_drive_map[need_category]
            for drive_id, drive in self.drives.items():
                if drive.name in related_drive_names:
                    # If need satisfaction increased, decrease related drive intensity
                    if "satisfaction_delta" in need_update:
                        delta = need_update["satisfaction_delta"]
                        if delta > 0:
                            drive.update_intensity(-delta * 0.3)
    
    def _process_drive_query(self, query: Dict[str, Any], message_id: str):
        """Process a direct query about drive states"""
        query_type = query.get("query_type")
        
        if query_type == "active_drives":
            # Return active drives
            active_drives = self._get_active_drives()
            
            if self.event_bus:
                self.publish_message("drive_query_response", {
                    "query_id": message_id,
                    "active_drives": active_drives,
                    "count": len(active_drives)
                })
                
        elif query_type == "drive_state":
            # Return specific drive state
            drive_id = query.get("drive_id")
            if drive_id and drive_id in self.drives:
                drive = self.drives[drive_id]
                
                if self.event_bus:
                    self.publish_message("drive_query_response", {
                        "query_id": message_id,
                        "drive_id": drive_id,
                        "name": drive.name,
                        "intensity": drive.intensity,
                        "priority": drive.priority,
                        "category": drive.category
                    })
            else:
                if self.event_bus:
                    self.publish_message("drive_query_response", {
                        "query_id": message_id,
                        "error": f"Drive not found: {drive_id}",
                        "available_drives": list(self.drives.keys())
                    })
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        drive_states = {}
        for drive_id, drive in self.drives.items():
            drive_states[drive_id] = {
                "name": drive.name,
                "intensity": drive.intensity,
                "priority": drive.priority,
                "category": drive.category,
                "last_updated": drive.last_updated.isoformat()
            }
            
        return {
            "development_level": self.development_level,
            "drives": drive_states,
            "activation_threshold": self.activation_threshold,
            "last_update": self.last_update.isoformat(),
            "neural_net": self.neural_net.get_state()
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the current state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, f"{self.module_id}_state.json")
        
        # Convert drives to serializable format
        drive_states = {}
        for drive_id, drive in self.drives.items():
            drive_states[drive_id] = {
                "name": drive.name,
                "description": drive.description,
                "intensity": drive.intensity,
                "decay_rate": drive.decay_rate,
                "priority": drive.priority,
                "category": drive.category,
                "satisfying_actions": list(drive.satisfying_actions),
                "last_updated": drive.last_updated.isoformat(),
                "is_active": drive.is_active
            }
        
        state = {
            "module_id": self.module_id,
            "module_type": "drives",
            "development_level": self.development_level,
            "drives": drive_states,
            "activation_threshold": self.activation_threshold,
            "last_update": self.last_update.isoformat()
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        # Also save neural network state
        net_path = os.path.join(state_dir, f"{self.module_id}_network.pt")
        self.neural_net.save(net_path)
        
        logger.info(f"Saved drives state to {state_path}")
        return state_path
        
    def load_state(self, state_path: str) -> bool:
        """
        Load state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            Success flag
        """
        if not os.path.exists(state_path):
            logger.error(f"State file not found: {state_path}")
            return False
            
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Verify this is the right kind of state
            if state.get("module_type") != "drives":
                logger.error(f"Invalid state file type: {state.get('module_type')}")
                return False
                
            # Load development level
            self.development_level = state.get("development_level", 0.0)
            
            # Load activation threshold
            self.activation_threshold = state.get("activation_threshold", 0.7)
            
            # Load last update time
            last_update_str = state.get("last_update", datetime.now().isoformat())
            self.last_update = datetime.fromisoformat(last_update_str)
            
            # Load drives
            drive_states = state.get("drives", {})
            self.drives = {}
            
            for drive_id, drive_data in drive_states.items():
                drive = Drive(
                    id=drive_id,
                    name=drive_data["name"],
                    description=drive_data.get("description", ""),
                    intensity=drive_data["intensity"],
                    decay_rate=drive_data["decay_rate"],
                    priority=drive_data["priority"],
                    category=drive_data["category"],
                    satisfying_actions=set(drive_data.get("satisfying_actions", [])),
                    last_updated=datetime.fromisoformat(drive_data["last_updated"]),
                    is_active=drive_data.get("is_active", True)
                )
                self.drives[drive_id] = drive
                
            # Load neural network state
            net_path = state_path.replace("_state.json", "_network.pt")
            if os.path.exists(net_path):
                self.neural_net.load(net_path)
                
            logger.info(f"Loaded drives state from {state_path} with {len(self.drives)} drives")
            return True
            
        except Exception as e:
            logger.error(f"Error loading drives state: {e}")
            return False
