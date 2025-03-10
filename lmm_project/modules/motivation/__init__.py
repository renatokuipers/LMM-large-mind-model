"""
Motivation Module

This module handles motivational aspects of the cognitive system, including:
- Basic drives that energize behavior
- Psychological needs based on theories like Maslow's hierarchy
- Goal creation and pursuit
- Reward processing and reinforcement learning

The motivation system develops from simple pleasure/pain and approach/avoidance
motivations in early stages to complex goal hierarchies and intrinsic motivations
in more advanced developmental stages.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

from lmm_project.modules.motivation.drives import Drives
from lmm_project.modules.motivation.needs import Needs
from lmm_project.modules.motivation.rewards import Rewards
from lmm_project.modules.motivation.goal_setting import GoalSetting
from lmm_project.modules.motivation.models import (
    Drive, Need, Goal, RewardEvent, MotivationalState, MotivationNeuralState
)
from lmm_project.modules.motivation.neural_net import MotivationNeuralNetwork

def get_module(
    module_id: str = "motivation",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "MotivationSystem":
    """
    Factory function to create a motivation system
    
    This function creates an integrated motivation system that manages:
    - Basic drives that energize behavior
    - Psychological needs based on theories like Maslow's hierarchy
    - Goal setting and management
    - Reward processing and reinforcement learning
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for inter-module communication
        development_level: Initial development level for the module
        
    Returns:
        MotivationSystem: Integrated motivation system
    """
    return MotivationSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class MotivationSystem(BaseModule):
    """
    Integrated motivation system that brings together all motivation components
    
    The MotivationSystem coordinates drives, needs, rewards, and goal setting
    to create a complete motivational system that energizes behavior and
    guides the system toward valuable outcomes.
    """
    
    # Development milestones for motivation
    development_milestones = {
        0.1: "Basic physiological drives",
        0.2: "Simple approach/avoidance motivation",
        0.3: "Basic needs awareness",
        0.4: "Simple goal formation",
        0.5: "Delayed gratification",
        0.6: "Hierarchical need satisfaction",
        0.7: "Complex goal systems",
        0.8: "Abstract motivation",
        0.9: "Value-based motivation",
        1.0: "Integrated motivational system"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the motivation system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id,
            module_type="motivation",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.motivation.{module_id}")
        
        # Initialize component modules with namespaced IDs
        self.drives = Drives(
            module_id=f"{module_id}.drives",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.needs = Needs(
            module_id=f"{module_id}.needs",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.rewards = Rewards(
            module_id=f"{module_id}.rewards",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.goal_setting = GoalSetting(
            module_id=f"{module_id}.goal_setting",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create unified motivational state
        self.motivational_state = MotivationalState(
            development_level=development_level
        )
        
        # Initialize neural state
        self.neural_state = MotivationNeuralState()
        self.neural_state.development_level = development_level
        
        # Register for event subscriptions
        if event_bus:
            # Subscribe to component module events
            self.subscribe_to_message(f"{module_id}.drives", self._handle_drive_event)
            self.subscribe_to_message(f"{module_id}.needs", self._handle_need_event)
            self.subscribe_to_message(f"{module_id}.rewards", self._handle_reward_event)
            self.subscribe_to_message(f"{module_id}.goal_setting", self._handle_goal_event)
            
            # Subscribe to external events
            self.subscribe_to_message("perception_input")
            self.subscribe_to_message("memory_retrieval")
            self.subscribe_to_message("emotional_state")
            self.subscribe_to_message("social_interaction")
            self.subscribe_to_message("motivation_query")
        
        self.logger.info(f"Motivation system initialized at development level {development_level:.2f}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through the motivation system
        
        Args:
            input_data: Dictionary containing input data
                Required keys:
                - "type": Type of input ("perception", "query", "update", etc.)
                - type-specific keys for each input type
                
        Returns:
            Dictionary with processed results
        """
        input_type = input_data.get("type", "unknown")
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        self.logger.debug(f"Processing {input_type} input (id: {process_id})")
        
        result = {
            "success": False,
            "process_id": process_id,
            "message": f"Unknown input type: {input_type}"
        }
        
        # Process based on input type
        if input_type == "get_state":
            result = self._get_motivational_state(input_data, process_id)
            
        elif input_type == "update_drives":
            # Forward to drives module
            drives_result = self.drives.process_input(input_data)
            self._update_motivational_state()
            result = {
                "success": drives_result.get("success", False),
                "process_id": process_id,
                "drives_result": drives_result,
                "motivational_state": self.motivational_state.dict()
            }
            
        elif input_type == "update_needs":
            # Forward to needs module
            needs_result = self.needs.process_input(input_data)
            self._update_motivational_state()
            result = {
                "success": needs_result.get("success", False),
                "process_id": process_id,
                "needs_result": needs_result,
                "motivational_state": self.motivational_state.dict()
            }
            
        elif input_type == "process_reward":
            # Forward to rewards module
            reward_result = self.rewards.process_input(input_data)
            self._update_motivational_state()
            result = {
                "success": reward_result.get("success", False),
                "process_id": process_id,
                "reward_result": reward_result,
                "motivational_state": self.motivational_state.dict()
            }
            
        elif input_type == "manage_goal":
            # Forward to goal setting module
            goal_result = self.goal_setting.process_input(input_data)
            self._update_motivational_state()
            result = {
                "success": goal_result.get("success", False),
                "process_id": process_id,
                "goal_result": goal_result,
                "motivational_state": self.motivational_state.dict()
            }
            
        elif input_type == "query":
            result = self._process_query(input_data, process_id)
            
        elif input_type == "perception":
            result = self._process_perception(input_data, process_id)
            
        return result
    
    def _get_motivational_state(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Get current motivational state"""
        self._update_motivational_state()
        
        # Determine what components to include
        include_drives = input_data.get("include_drives", True)
        include_needs = input_data.get("include_needs", True)
        include_goals = input_data.get("include_goals", True)
        include_rewards = input_data.get("include_rewards", False)
        
        # Create response
        state_dict = self.motivational_state.dict(
            exclude_unset=True,
            exclude={
                "drives": not include_drives,
                "needs": not include_needs,
                "goals": not include_goals,
                "recent_rewards": not include_rewards
            }
        )
        
        return {
            "success": True,
            "process_id": process_id,
            "motivational_state": state_dict
        }
    
    def _process_query(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Process specific motivational queries"""
        query_type = input_data.get("query_type", "unknown")
        
        if query_type == "dominant_motivation":
            self._update_motivational_state()
            return {
                "success": True,
                "process_id": process_id,
                "query_type": query_type,
                "dominant_motivation": self.motivational_state.dominant_motivation,
                "motivation_level": self.motivational_state.motivation_level
            }
            
        elif query_type == "top_drives":
            # Get top drives by intensity * priority
            limit = input_data.get("limit", 3)
            drive_states = self.drives.process_input({"type": "get_states"})
            
            if not drive_states.get("success", False):
                return {
                    "success": False,
                    "process_id": process_id,
                    "message": "Failed to retrieve drive states"
                }
                
            # Get and sort drives
            drives = drive_states.get("drives", [])
            sorted_drives = sorted(
                drives, 
                key=lambda d: d.get("intensity", 0) * d.get("priority", 0),
                reverse=True
            )[:limit]
            
            return {
                "success": True,
                "process_id": process_id,
                "query_type": query_type,
                "top_drives": sorted_drives
            }
            
        elif query_type == "priority_needs":
            # Get priority needs based on hierarchy and satisfaction
            limit = input_data.get("limit", 3)
            need_states = self.needs.process_input({"type": "get_states"})
            
            if not need_states.get("success", False):
                return {
                    "success": False,
                    "process_id": process_id,
                    "message": "Failed to retrieve need states"
                }
                
            # Get and sort needs
            needs = need_states.get("needs", [])
            sorted_needs = sorted(
                needs, 
                key=lambda n: (1.0 - n.get("satisfaction", 0)) * (6 - n.get("hierarchy_level", 5)),
                reverse=True
            )[:limit]
            
            return {
                "success": True,
                "process_id": process_id,
                "query_type": query_type,
                "priority_needs": sorted_needs
            }
            
        elif query_type == "active_goals":
            # Get currently active goals
            goal_data = self.goal_setting.process_input({
                "type": "query",
                "query_type": "top_priority",
                "limit": input_data.get("limit", 3)
            })
            
            return {
                "success": goal_data.get("success", False),
                "process_id": process_id,
                "query_type": query_type,
                "active_goals": goal_data.get("goals", [])
            }
            
        return {
            "success": False,
            "process_id": process_id,
            "message": f"Unknown query type: {query_type}"
        }
    
    def _process_perception(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Process perception input by routing to appropriate components"""
        perception = input_data.get("perception", {})
        
        # Forward to drives, needs, and goal setting
        drive_result = self.drives.process_input({
            "type": "perception",
            "perception": perception
        })
        
        need_result = self.needs.process_input({
            "type": "perception",
            "perception": perception
        })
        
        goal_result = self.goal_setting.process_input({
            "type": "perception",
            "perception": perception
        })
        
        # Update motivational state
        self._update_motivational_state()
        
        return {
            "success": True,
            "process_id": process_id,
            "drive_result": drive_result,
            "need_result": need_result,
            "goal_result": goal_result,
            "motivational_state": self.motivational_state.dict()
        }
    
    def _update_motivational_state(self):
        """Update the unified motivational state from all components"""
        # Get drive states
        drive_states = self.drives.process_input({"type": "get_states"})
        if drive_states.get("success", False):
            drives_dict = {}
            for drive in drive_states.get("drives", []):
                drive_id = drive.get("id")
                if drive_id:
                    drives_dict[drive_id] = Drive(**drive)
            self.motivational_state.drives = drives_dict
        
        # Get need states
        need_states = self.needs.process_input({"type": "get_states"})
        if need_states.get("success", False):
            needs_dict = {}
            for need in need_states.get("needs", []):
                need_id = need.get("id")
                if need_id:
                    needs_dict[need_id] = Need(**need)
            self.motivational_state.needs = needs_dict
        
        # Get goal states
        goal_data = self.goal_setting.process_input({
            "type": "query",
            "query_type": "all"
        })
        if goal_data.get("success", False):
            goals_dict = {}
            for goal in goal_data.get("goals", []):
                goal_id = goal.get("id")
                if goal_id:
                    goals_dict[goal_id] = Goal(**goal)
            self.motivational_state.goals = goals_dict
        
        # Update overall motivation level
        self.motivational_state.update_motivation_level()
        
        # Update development level
        self.motivational_state.development_level = self.development_level
        
        # Update timestamp
        self.motivational_state.timestamp = datetime.now()
    
    def _handle_message(self, message: Message):
        """Handle incoming messages from other modules"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "perception_input":
            self.process_input({
                "type": "perception",
                "perception": content,
                "process_id": message.id
            })
            
        elif message_type == "memory_retrieval":
            # Process memory retrieval
            memory = content.get("memory", {})
            
            # Check if memory is relevant to motivation
            if "emotional_valence" in memory or "drive_related" in memory or "need_related" in memory:
                self.drives.process_input({
                    "type": "memory",
                    "memory": memory
                })
                
                self.needs.process_input({
                    "type": "memory",
                    "memory": memory
                })
                
        elif message_type == "emotional_state":
            # Process emotional state for its motivational implications
            emotion = content.get("emotion", {})
            valence = emotion.get("valence", 0.0)
            arousal = emotion.get("arousal", 0.0)
            
            # Route to relevant components
            if abs(valence) > 0.3 or arousal > 0.6:
                self.rewards.process_input({
                    "type": "emotion",
                    "emotion": emotion
                })
                
                self.drives.process_input({
                    "type": "emotion",
                    "emotion": emotion
                })
                
        elif message_type == "social_interaction":
            # Process social input for needs and goals
            social_data = content.get("interaction", {})
            
            self.needs.process_input({
                "type": "social_interaction",
                "interaction": social_data
            })
            
            if "feedback" in social_data:
                self.rewards.process_input({
                    "type": "social",
                    "feedback_type": social_data.get("feedback", {}).get("type", "neutral"),
                    "intensity": social_data.get("feedback", {}).get("intensity", 0.5)
                })
                
        elif message_type == "motivation_query":
            # Process query about motivational state
            query_type = content.get("query_type", "dominant_motivation")
            result = self.process_input({
                "type": "query",
                "query_type": query_type,
                "process_id": message.id
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="motivation_query_response",
                    content=result,
                    reply_to=message.id
                ))
    
    def _handle_drive_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the drives component"""
        event_type = event.get("type", "unknown")
        
        if event_type == "drive_update":
            # Update motivational state
            self._update_motivational_state()
            
            # Publish unified motivational state if significant change
            if self.event_bus and event.get("significant_change", False):
                self.publish_message("motivational_state_update", {
                    "motivational_state": self.motivational_state.dict()
                })
    
    def _handle_need_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the needs component"""
        event_type = event.get("type", "unknown")
        
        if event_type == "need_update":
            # Update motivational state
            self._update_motivational_state()
            
            # Publish unified motivational state if significant change
            if self.event_bus and event.get("significant_change", False):
                self.publish_message("motivational_state_update", {
                    "motivational_state": self.motivational_state.dict()
                })
    
    def _handle_reward_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the rewards component"""
        event_type = event.get("type", "unknown")
        
        if event_type == "reward_received":
            # Add to recent rewards
            reward = event.get("reward")
            if reward:
                try:
                    self.motivational_state.recent_rewards.append(RewardEvent(**reward))
                    # Keep only recent rewards (last 10)
                    if len(self.motivational_state.recent_rewards) > 10:
                        self.motivational_state.recent_rewards = self.motivational_state.recent_rewards[-10:]
                except Exception as e:
                    self.logger.error(f"Error processing reward: {e}")
    
    def _handle_goal_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the goal setting component"""
        event_type = event.get("type", "unknown")
        
        if event_type in ["goal_created", "goal_achieved", "goal_abandoned"]:
            # Update motivational state
            self._update_motivational_state()
            
            # Publish unified motivational state
            if self.event_bus:
                self.publish_message("motivational_state_update", {
                    "motivational_state": self.motivational_state.dict()
                })
    
    def _check_motivation_milestones(self):
        """Check if any motivation milestones have been achieved"""
        for level, description in sorted(self.development_milestones.items()):
            if level <= self.development_level and str(level) not in self.achieved_milestones:
                self.achieved_milestones.add(str(level))
                self.logger.info(f"Motivation milestone achieved: {description} (level {level})")
                
                # Publish milestone achievement
                if self.event_bus:
                    self.publish_message("motivation_milestone", {
                        "level": level,
                        "description": description,
                        "module": self.module_id
                    })
    
    def update_development(self, amount: float) -> float:
        """
        Update the module's development level
        
        Args:
            amount: Amount to increase development (0.0 to 1.0)
            
        Returns:
            New development level
        """
        # Update base development level
        old_level = self.development_level
        super().update_development(amount)
        
        # Update sub-components
        self.drives.update_development(amount)
        self.needs.update_development(amount)
        self.rewards.update_development(amount)
        self.goal_setting.update_development(amount)
        
        # Update neural state
        self.neural_state.development_level = self.development_level
        
        # Update motivational state
        self.motivational_state.development_level = self.development_level
        
        # Check for milestones
        self._check_motivation_milestones()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with state information
        """
        # Update motivational state
        self._update_motivational_state()
        
        # Get base state
        base_state = super().get_state()
        
        # Add motivation-specific state
        motivation_state = {
            "motivational_state": self.motivational_state.dict(),
            "neural_state": self.neural_state.dict(),
            "sub_components": {
                "drives": self.drives.get_state(),
                "needs": self.needs.get_state(),
                "rewards": self.rewards.get_state(),
                "goal_setting": self.goal_setting.get_state()
            }
        }
        
        return {**base_state, **motivation_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Save component states
        self.drives.save_state(state_dir)
        self.needs.save_state(state_dir)
        self.rewards.save_state(state_dir)
        self.goal_setting.save_state(state_dir)
        
        # Save system state
        return super().save_state(state_dir)
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Load component states (assuming paths follow convention)
        components = ['drives', 'needs', 'rewards', 'goal_setting']
        for component in components:
            component_path = state_path.replace(
                f"{self.module_type}/{self.module_id}",
                f"{self.module_type}/{self.module_id}.{component}"
            )
            
            if component == 'drives':
                self.drives.load_state(component_path)
            elif component == 'needs':
                self.needs.load_state(component_path)
            elif component == 'rewards':
                self.rewards.load_state(component_path)
            elif component == 'goal_setting':
                self.goal_setting.load_state(component_path)
        
        # Load system state
        return super().load_state(state_path)

# Export public objects
__all__ = [
    'get_module', 
    'MotivationSystem',
    'Drives',
    'Needs',
    'Rewards',
    'GoalSetting',
    'Drive',
    'Need',
    'Goal',
    'RewardEvent',
    'MotivationalState',
    'MotivationNeuralState',
    'MotivationNeuralNetwork'
]
