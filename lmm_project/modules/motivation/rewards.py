"""
Rewards Module

This module handles reward processing - detecting, valuing, and learning from 
both intrinsic and extrinsic rewards. It implements mechanisms for motivation 
through reinforcement and develops increasingly abstract reward systems as 
development progresses.
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
from pydantic import Field

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.motivation.models import RewardEvent, MotivationNeuralState
from lmm_project.modules.motivation.neural_net import MotivationNeuralNetwork, get_device

class Rewards(BaseModule):
    """
    Handles reward processing and reinforcement learning
    
    This module detects and processes reward signals, calculates
    reward prediction errors, and adapts reward significance based
    on context and developmental stage.
    """
    
    # Developmental milestones for rewards module
    development_milestones = {
        0.1: "Simple pleasure/pain processing",
        0.2: "Basic reinforcement learning",
        0.3: "Immediate reward seeking",
        0.4: "Delayed gratification (short-term)",
        0.5: "Social reward recognition",
        0.6: "Intrinsic motivation emergence",
        0.7: "Complex reward prediction",
        0.8: "Long-term reward planning",
        0.9: "Abstract reward valuation",
        1.0: "Self-reinforcing reward systems"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the rewards module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id, 
            module_type="rewards",
            event_bus=event_bus, 
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.motivation.rewards.{module_id}")
        
        # Storage for reward events
        self.reward_history: List[RewardEvent] = []
        self.max_history_size = 100  # Keep the last 100 reward events
        
        # Reward types and their current weights
        self.reward_types = {
            "physiological": 1.0,  # Basic needs satisfaction
            "safety": 0.8,         # Security and stability
            "social": 0.5,         # Social approval and belonging
            "achievement": 0.3,    # Accomplishment and mastery
            "cognitive": 0.2,      # Learning and understanding
            "aesthetic": 0.1,      # Beauty and order
            "self_actualization": 0.1  # Self-fulfillment and growth
        }
        
        # Adjust weights based on development level
        self._adjust_reward_weights()
        
        # Neural network for reward prediction and processing
        input_dim = 20  # State representation dimensions
        hidden_dim = 32
        output_dim = len(self.reward_types)
        
        self.neural_network = MotivationNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            network_type="reward",
            learning_rate=0.01,
            development_level=development_level
        )
        
        # Reward prediction state
        self.last_state = None
        self.last_prediction = None
        
        # Current reward prediction error
        self.prediction_error = 0.0
        
        # Subscribe to relevant messages
        if self.event_bus:
            self.subscribe_to_message("action_completed")
            self.subscribe_to_message("goal_progress")
            self.subscribe_to_message("goal_achieved")
            self.subscribe_to_message("drive_satisfied")
            self.subscribe_to_message("need_satisfied")
            self.subscribe_to_message("social_feedback")
            self.subscribe_to_message("reward_query")
        
        self.logger.info(f"Rewards module initialized at development level {development_level:.2f}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to generate reward signals
        
        Args:
            input_data: Dictionary containing input data
                Required keys depend on the input type:
                - "type": Type of input ("action", "goal", "social", etc.)
                - type-specific keys for each input type
                
        Returns:
            Dictionary with processed results including reward information
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input for rewards")
        
        result = {
            "reward_generated": False,
            "reward_magnitude": 0.0,
            "reward_type": None,
            "prediction_error": 0.0
        }
        
        # Create state representation from input
        state_vector = self._create_state_vector(input_data)
        
        # Get reward prediction for this state
        reward_prediction = self._predict_reward(state_vector)
        
        # Determine actual reward based on input type
        if input_type == "action":
            reward = self._process_action(input_data)
        elif input_type == "goal":
            reward = self._process_goal(input_data)
        elif input_type == "drive":
            reward = self._process_drive(input_data)
        elif input_type == "need":
            reward = self._process_need(input_data)
        elif input_type == "social":
            reward = self._process_social(input_data)
        else:
            reward = self._process_generic(input_data)
        
        # If a reward was generated
        if reward and reward.magnitude != 0:
            # Calculate prediction error
            prediction = reward_prediction[self._get_reward_type_index(reward.reward_type)]
            self.prediction_error = reward.magnitude - prediction.item()
            
            # Update neural network with actual reward
            target = reward_prediction.clone().detach()
            target[self._get_reward_type_index(reward.reward_type)] = reward.magnitude
            
            # Train the network
            training_info = self.neural_network.train(
                inputs=state_vector,
                targets=target,
                learning_rate=self._get_adaptive_learning_rate()
            )
            
            # Save reward to history
            self._add_to_history(reward)
            
            # Update result
            result["reward_generated"] = True
            result["reward_magnitude"] = reward.magnitude
            result["reward_type"] = reward.reward_type
            result["prediction_error"] = self.prediction_error
            result["context"] = reward.context
            
            # Publish reward event if significant
            if abs(reward.magnitude) >= 0.3 and self.event_bus:
                self.publish_message("reward_received", {
                    "reward": reward.dict(),
                    "prediction_error": self.prediction_error
                })
        
        return result
    
    def _process_action(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process action completion rewards"""
        action = input_data.get("action", {})
        success = input_data.get("success", False)
        
        if not success:
            return None
            
        # Base reward on action value and development level
        base_value = input_data.get("value", 0.2)
        
        # Scale reward based on development (younger minds get more reward from simple actions)
        if self.development_level < 0.3:
            magnitude = base_value * 1.5
        else:
            magnitude = base_value * (1.0 - (self.development_level * 0.3))
            
        return RewardEvent(
            reward_type="intrinsic" if self.development_level > 0.5 else "extrinsic",
            magnitude=magnitude,
            source=f"action:{action.get('name', 'unknown')}",
            context=input_data.get("context", ""),
            affected_drives=input_data.get("drives", []),
            affected_needs=input_data.get("needs", [])
        )
    
    def _process_goal(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process goal-related rewards"""
        goal_type = input_data.get("goal_type", "progress")  # progress or achievement
        goal_name = input_data.get("goal_name", "unknown")
        progress = input_data.get("progress", 0.0)
        
        # Different reward for progress vs achievement
        if goal_type == "achievement":
            # Higher reward for goal achievement
            magnitude = 0.5 + (input_data.get("importance", 0.5) * 0.5)
            reward_type = "intrinsic" if self.development_level > 0.4 else "extrinsic"
        else:
            # Smaller reward for progress
            magnitude = progress * 0.3
            reward_type = "intrinsic" if self.development_level > 0.5 else "extrinsic"
            
            # Only generate reward if progress is significant
            if magnitude < 0.05:
                return None
        
        return RewardEvent(
            reward_type=reward_type,
            magnitude=magnitude,
            source=f"goal:{goal_name}",
            context=input_data.get("context", ""),
            affected_drives=input_data.get("drives", []),
            affected_needs=input_data.get("needs", [])
        )
    
    def _process_drive(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process drive satisfaction rewards"""
        drive_name = input_data.get("drive_name", "unknown")
        satisfaction = input_data.get("satisfaction", 0.0)
        
        # Only generate reward if satisfaction is significant
        if satisfaction < 0.1:
            return None
            
        # Physiological rewards are strongest for early development
        dev_factor = 1.0 - (self.development_level * 0.5) if drive_name == "physiological" else 1.0
        magnitude = satisfaction * 0.4 * dev_factor
        
        return RewardEvent(
            reward_type="extrinsic",  # Drive satisfaction is always extrinsic
            magnitude=magnitude,
            source=f"drive:{drive_name}",
            context=input_data.get("context", ""),
            affected_drives=[drive_name],
            affected_needs=input_data.get("needs", [])
        )
    
    def _process_need(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process need satisfaction rewards"""
        need_name = input_data.get("need_name", "unknown")
        satisfaction = input_data.get("satisfaction", 0.0)
        hierarchy_level = input_data.get("hierarchy_level", 1)
        
        # Only generate reward if satisfaction is significant
        if satisfaction < 0.1:
            return None
            
        # Higher-level needs generate more intrinsic reward as development increases
        is_intrinsic = (hierarchy_level > 2 and self.development_level > 0.4)
        
        # Higher-level needs give stronger rewards as development progresses
        level_factor = min(1.0, self.development_level * hierarchy_level / 3)
        magnitude = satisfaction * 0.3 * level_factor
        
        return RewardEvent(
            reward_type="intrinsic" if is_intrinsic else "extrinsic",
            magnitude=magnitude,
            source=f"need:{need_name}",
            context=input_data.get("context", ""),
            affected_drives=input_data.get("drives", []),
            affected_needs=[need_name]
        )
    
    def _process_social(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process social feedback rewards"""
        feedback_type = input_data.get("feedback_type", "neutral")
        intensity = input_data.get("intensity", 0.5)
        
        # Convert feedback type to reward magnitude
        if feedback_type == "positive":
            base_magnitude = intensity
        elif feedback_type == "negative":
            base_magnitude = -intensity
        else:
            return None  # No reward for neutral feedback
            
        # Social rewards become more important as development progresses
        dev_factor = min(1.0, self.development_level * 2)
        magnitude = base_magnitude * 0.5 * dev_factor
        
        return RewardEvent(
            reward_type="extrinsic",  # Social feedback is generally extrinsic
            magnitude=magnitude,
            source="social_feedback",
            context=input_data.get("context", ""),
            affected_drives=input_data.get("drives", []),
            affected_needs=input_data.get("needs", [])
        )
    
    def _process_generic(self, input_data: Dict[str, Any]) -> Optional[RewardEvent]:
        """Process generic reward inputs"""
        magnitude = input_data.get("reward_magnitude", 0.0)
        reward_type = input_data.get("reward_type", "extrinsic")
        
        if abs(magnitude) < 0.01:
            return None
            
        return RewardEvent(
            reward_type=reward_type,
            magnitude=magnitude,
            source=input_data.get("source", "generic"),
            context=input_data.get("context", ""),
            affected_drives=input_data.get("drives", []),
            affected_needs=input_data.get("needs", [])
        )
    
    def _predict_reward(self, state_vector: np.ndarray) -> torch.Tensor:
        """
        Predict reward for a given state
        
        Args:
            state_vector: Vector representation of state
            
        Returns:
            Tensor of predicted rewards for each reward type
        """
        if isinstance(state_vector, np.ndarray):
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        else:
            state_tensor = state_vector
            
        # Forward pass through network
        prediction, _ = self.neural_network.forward(state_tensor)
        
        # Save for later use in calculating prediction error
        self.last_state = state_tensor
        self.last_prediction = prediction
        
        return prediction
    
    def _create_state_vector(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Create a vector representation of the current state
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Numpy array representation of state
        """
        # Initialize state vector with zeros
        state = np.zeros(20)
        
        # Encode input type (one-hot)
        input_types = ["action", "goal", "drive", "need", "social"]
        input_type = input_data.get("type", "unknown")
        if input_type in input_types:
            idx = input_types.index(input_type)
            state[idx] = 1.0
            
        # Encode context (simple hash-based approach)
        context = input_data.get("context", "")
        if context:
            hash_val = hash(context) % 1000 / 1000.0  # Normalize to 0-1
            state[5] = hash_val
            
        # Encode source object
        source = input_data.get("source", "")
        if source:
            hash_val = hash(source) % 1000 / 1000.0
            state[6] = hash_val
            
        # Encode various numerical fields if present
        if "value" in input_data:
            state[7] = float(input_data["value"])
            
        if "progress" in input_data:
            state[8] = float(input_data["progress"])
            
        if "satisfaction" in input_data:
            state[9] = float(input_data["satisfaction"])
            
        if "intensity" in input_data:
            state[10] = float(input_data["intensity"])
            
        if "importance" in input_data:
            state[11] = float(input_data["importance"])
            
        # Include development level
        state[12] = self.development_level
        
        # Add some recency bias for common experience types
        recent_counts = self._get_recent_counts(5)  # Check last 5 rewards
        for i, rtype in enumerate(input_types):
            if rtype in recent_counts:
                state[13 + i] = recent_counts[rtype] / 5.0
                
        return state
    
    def _get_recent_counts(self, n: int) -> Dict[str, int]:
        """Count reward sources in the most recent n rewards"""
        counts = {}
        
        for reward in self.reward_history[-n:]:
            source_type = reward.source.split(":")[0] if ":" in reward.source else reward.source
            counts[source_type] = counts.get(source_type, 0) + 1
            
        return counts
    
    def _add_to_history(self, reward: RewardEvent) -> None:
        """Add reward to history, maintaining max size"""
        self.reward_history.append(reward)
        
        # Trim if needed
        if len(self.reward_history) > self.max_history_size:
            self.reward_history = self.reward_history[-self.max_history_size:]
    
    def _get_reward_type_index(self, reward_type: str) -> int:
        """Get index for reward type in output vector"""
        reward_types = list(self.reward_types.keys())
        if reward_type in reward_types:
            return reward_types.index(reward_type)
        return 0  # Default to first type
    
    def _get_adaptive_learning_rate(self) -> float:
        """
        Get adaptive learning rate based on development and prediction error
        
        Higher prediction errors lead to faster learning
        Development reduces learning rate (stabilization)
        """
        base_rate = 0.01
        
        # Scale by development (younger = faster learning)
        dev_factor = 1.0 - (self.development_level * 0.5)
        
        # Scale by prediction error (larger error = faster learning)
        error_factor = min(2.0, 1.0 + abs(self.prediction_error))
        
        return base_rate * dev_factor * error_factor
    
    def _adjust_reward_weights(self) -> None:
        """
        Adjust reward weights based on development level
        
        Early development: Physiological and safety rewards dominate
        Middle development: Social rewards increase in importance
        Late development: Achievement and cognitive rewards increase
        """
        dev = self.development_level
        
        # Basic physiological rewards decrease slightly with development
        self.reward_types["physiological"] = max(0.5, 1.0 - dev*0.5)
        
        # Safety rewards stay relatively stable
        self.reward_types["safety"] = 0.8
        
        # Social rewards peak in middle development
        if dev < 0.5:
            self.reward_types["social"] = min(1.0, dev*2)
        else:
            self.reward_types["social"] = min(1.0, 2.0 - dev)
            
        # Achievement rewards increase with development
        self.reward_types["achievement"] = min(1.0, dev*1.2)
        
        # Cognitive rewards increase with development
        self.reward_types["cognitive"] = min(1.0, dev*1.3)
        
        # Aesthetic and self-actualization rewards emerge late
        self.reward_types["aesthetic"] = min(1.0, max(0, dev - 0.5)*2)
        self.reward_types["self_actualization"] = min(1.0, max(0, dev - 0.7)*3)
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "action_completed":
            result = self.process_input({
                "type": "action",
                **content
            })
            
        elif message_type == "goal_progress":
            result = self.process_input({
                "type": "goal",
                "goal_type": "progress",
                **content
            })
            
        elif message_type == "goal_achieved":
            result = self.process_input({
                "type": "goal",
                "goal_type": "achievement",
                **content
            })
            
        elif message_type == "drive_satisfied":
            result = self.process_input({
                "type": "drive",
                **content
            })
            
        elif message_type == "need_satisfied":
            result = self.process_input({
                "type": "need",
                **content
            })
            
        elif message_type == "social_feedback":
            result = self.process_input({
                "type": "social",
                **content
            })
            
        elif message_type == "reward_query":
            # Query about reward prediction
            state_vector = self._create_state_vector(content)
            prediction = self._predict_reward(state_vector)
            
            # Return reward prediction
            self.event_bus.publish(Message(
                sender=self.module_id,
                recipient=message.sender,
                message_type="reward_prediction",
                content={
                    "query_id": content.get("query_id", ""),
                    "predictions": {
                        rtype: prediction[i].item() 
                        for i, rtype in enumerate(self.reward_types.keys())
                    },
                    "state_vector": state_vector.tolist(),
                    "development_level": self.development_level
                },
                reply_to=message.id
            ))
    
    def update_development(self, amount: float) -> float:
        """
        Update the module's development level
        
        Args:
            amount: Amount to increase development (0.0 to 1.0)
            
        Returns:
            New development level
        """
        # Update base development
        old_level = self.development_level
        super().update_development(amount)
        
        # Adjust reward weights based on new development level
        self._adjust_reward_weights()
        
        # Update neural network development
        self.neural_network.update_development(amount)
        
        # Log milestone achievements
        self._check_development_milestones(old_level)
        
        return self.development_level
    
    def _check_development_milestones(self, old_level: float) -> None:
        """
        Check if any development milestones have been reached
        
        Parameters:
        old_level: The previous development level
        """
        # Get the current milestone based on the new development level
        current_milestone = None
        for level, milestone in sorted(self.development_milestones.items()):
            if self.development_level >= float(level):
                current_milestone = milestone
        
        # Get the previous milestone based on the old development level
        previous_milestone = None
        for level, milestone in sorted(self.development_milestones.items()):
            if old_level >= float(level):
                previous_milestone = milestone
        
        # If we've reached a new milestone, log it
        if current_milestone != previous_milestone and current_milestone is not None:
            logger = logging.getLogger(f"lmm.motivation.rewards.{self.module_id}")
            logger.info(f"Reached new development milestone: {current_milestone}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with state information
        """
        base_state = super().get_state()
        
        rewards_state = {
            "reward_history": [r.dict() for r in self.reward_history[-10:]],  # Last 10 rewards
            "reward_types": self.reward_types,
            "prediction_error": self.prediction_error,
            "neural_state": self.neural_network.get_state()
        }
        
        return {**base_state, **rewards_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save basic module state
        state_path = os.path.join(module_dir, "module_state.json")
        with open(state_path, 'w') as f:
            # Get state and convert reward history to dicts
            state = self.get_state()
            state["reward_history"] = [r.dict() for r in self.reward_history]
            json.dump(state, f, indent=2, default=str)
            
        # Save neural network state
        nn_path = os.path.join(module_dir, "neural_network.pt")
        self.neural_network.save(nn_path)
        
        self.logger.info(f"Saved rewards module state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            module_dir = os.path.dirname(state_path)
            
            # Load module state
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update development level and milestones
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Load reward history
            from lmm_project.modules.motivation.models import RewardEvent
            self.reward_history = []
            for reward_dict in state.get("reward_history", []):
                try:
                    self.reward_history.append(RewardEvent(**reward_dict))
                except Exception as e:
                    self.logger.warning(f"Could not load reward event: {e}")
            
            # Update reward types
            if "reward_types" in state:
                self.reward_types = state["reward_types"]
            else:
                self._adjust_reward_weights()
                
            # Load neural network state
            nn_path = os.path.join(module_dir, "neural_network.pt")
            if os.path.exists(nn_path):
                self.neural_network.load(nn_path)
                self.logger.info(f"Loaded neural network from {nn_path}")
            else:
                self.logger.warning(f"Neural network state not found at {nn_path}")
                
            self.logger.info(f"Loaded rewards module state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load rewards module state: {e}")
            return False 