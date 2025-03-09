import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import logging
import os
import random
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.learning.models import ReinforcementLearningEvent

logger = logging.getLogger(__name__)

class ReinforcementLearning(BaseModule):
    """
    Learning from rewards and punishments
    
    This module implements reinforcement learning mechanisms that allow the system
    to learn which actions lead to positive outcomes in different contexts.
    """
    
    # Development milestones for reinforcement learning
    development_milestones = {
        0.0: "Basic reward-based learning",
        0.2: "Delayed reward processing",
        0.4: "Context-sensitive reinforcement",
        0.6: "Value-based decision making",
        0.8: "Complex reward integration",
        1.0: "Abstract goal-directed behavior"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the reinforcement learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="reinforcement_learning",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Q-values for action-state pairs
        self.q_values = {}
        
        # Policy mapping states to action probabilities
        self.policy = {}
        
        # Experience replay buffer (only active at higher developmental levels)
        self.experience_buffer = deque(maxlen=100)
        
        # Developmental parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.7  # How much future rewards matter
        self.exploration_rate = 0.3  # Probability of random exploration
        
        # Recent action history
        self.action_history = []
        self.max_history = 20
        
        # Adjust parameters based on development level
        self._adjust_for_development()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("reward_signal", self._handle_reward)
            self.subscribe_to_message("action_performed", self._handle_action)
    
    def _adjust_for_development(self):
        """Adjust learning mechanisms based on developmental level"""
        # Learning rate decreases with development (more stable learning)
        self.learning_rate = max(0.05, 0.3 - (self.development_level * 0.25))
        
        # Discount factor increases with development (more future-oriented)
        self.discount_factor = min(0.95, 0.6 + (self.development_level * 0.35))
        
        # Exploration decreases with development (more exploitation)
        self.exploration_rate = max(0.05, 0.5 - (self.development_level * 0.45))
        
        # Experience buffer size increases with development
        self.experience_buffer = deque(maxlen=max(50, int(100 + (self.development_level * 400))))
        
        # History tracking increases with development
        self.max_history = max(10, int(20 + (self.development_level * 80)))
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reinforcement learning operations
        
        Args:
            input_data: Dictionary containing operation and parameters
            
        Returns:
            Dictionary with operation results
        """
        operation = input_data.get("operation", "learn")
        
        if operation == "learn":
            return self._learn_from_experience(input_data)
        elif operation == "select_action":
            return self._select_action(input_data)
        elif operation == "update_policy":
            return self._update_policy(input_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "module_id": self.module_id
            }
    
    def _learn_from_experience(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a reinforcement experience"""
        # Extract required parameters
        state = input_data.get("state")
        action = input_data.get("action")
        reward = input_data.get("reward", 0.0)
        next_state = input_data.get("next_state")
        
        if not state or not action:
            return {"status": "error", "message": "Missing state or action"}
        
        # Initialize Q-values if needed
        if state not in self.q_values:
            self.q_values[state] = {}
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0.0
        
        # Calculate Q-value update
        current_q = self.q_values[state][action]
        
        if next_state:
            # Use Q-learning formula if we have next state
            # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
            max_next_q = self._get_max_q_value(next_state)
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
        else:
            # Simple update if no next state (terminal state)
            # Q(s,a) = Q(s,a) + α * (r - Q(s,a))
            new_q = current_q + self.learning_rate * (reward - current_q)
        
        # Update Q-value
        self.q_values[state][action] = new_q
        
        # Add to experience buffer (for experience replay)
        if self.development_level >= 0.3 and next_state:
            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "timestamp": datetime.now()
            }
            self.experience_buffer.append(experience)
        
        # Create learning event
        event = ReinforcementLearningEvent(
            source=input_data.get("source", "experience"),
            content=f"Reinforcement learning for action '{action}' in state '{state}'",
            action=action,
            consequence=input_data.get("consequence", "reward" if reward > 0 else "punishment"),
            reward_value=reward,
            delay=input_data.get("delay", 0.0),
            context=state,
            reinforcement_type="positive" if reward > 0 else "negative" if reward == 0 else "punishment",
            developmental_level=self.development_level
        )
        
        # Update policy based on new Q-values
        self._update_state_policy(state)
        
        # Perform experience replay if development level is high enough
        replay_results = None
        if self.development_level >= 0.4 and len(self.experience_buffer) >= 10:
            replay_results = self._perform_experience_replay(input_data.get("replay_batch_size", 5))
        
        return {
            "status": "success",
            "state": state,
            "action": action,
            "previous_q": current_q,
            "updated_q": new_q,
            "reward": reward,
            "learning_event_id": event.id,
            "replay_performed": replay_results is not None,
            "replay_results": replay_results
        }
    
    def _select_action(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select an action based on current state and policy"""
        state = input_data.get("state")
        available_actions = input_data.get("available_actions", [])
        
        if not state:
            return {"status": "error", "message": "Missing state parameter"}
        
        # If we don't have this state in our policy, initialize it
        if state not in self.policy:
            self._initialize_state_policy(state, available_actions)
        
        # If we need to update available actions
        elif available_actions and not all(action in self.policy[state] for action in available_actions):
            self._update_state_policy(state, available_actions)
        
        # Apply exploration-exploitation tradeoff
        if random.random() < self.exploration_rate:
            # Exploration: select random action
            actions = list(self.policy[state].keys())
            selected_action = random.choice(actions)
            selection_type = "exploration"
        else:
            # Exploitation: select best action
            selected_action = self._get_best_action(state)
            selection_type = "exploitation"
        
        # Record this action in history
        self._record_action(state, selected_action)
        
        return {
            "status": "success",
            "state": state,
            "selected_action": selected_action,
            "selection_type": selection_type,
            "action_probability": self.policy[state].get(selected_action, 0.0),
            "exploration_rate": self.exploration_rate
        }
    
    def _update_policy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explicitly update the policy for a state"""
        state = input_data.get("state")
        action_probs = input_data.get("action_probabilities", {})
        
        if not state or not action_probs:
            return {"status": "error", "message": "Missing state or action probabilities"}
        
        # Update policy for this state
        if state not in self.policy:
            self.policy[state] = {}
        
        # Integrate new probabilities
        for action, prob in action_probs.items():
            self.policy[state][action] = prob
        
        # Normalize probabilities
        self._normalize_policy(state)
        
        return {
            "status": "success",
            "state": state,
            "updated_policy": self.policy[state]
        }
    
    def _get_max_q_value(self, state: str) -> float:
        """Get the maximum Q-value for a state"""
        if state not in self.q_values or not self.q_values[state]:
            return 0.0
        
        return max(self.q_values[state].values())
    
    def _get_best_action(self, state: str) -> str:
        """Get the action with highest probability in the policy"""
        if state not in self.policy or not self.policy[state]:
            return ""
        
        return max(self.policy[state].items(), key=lambda x: x[1])[0]
    
    def _initialize_state_policy(self, state: str, available_actions: List[str] = None):
        """Initialize policy for a new state"""
        self.policy[state] = {}
        
        if not available_actions:
            # If no actions provided, check if we have Q-values for this state
            if state in self.q_values:
                available_actions = list(self.q_values[state].keys())
            else:
                return
        
        # Initialize with uniform distribution
        prob = 1.0 / len(available_actions)
        for action in available_actions:
            self.policy[state][action] = prob
    
    def _update_state_policy(self, state: str, available_actions: List[str] = None):
        """Update policy for a state based on Q-values"""
        if state not in self.q_values:
            if available_actions:
                self._initialize_state_policy(state, available_actions)
            return
        
        # Get actions from Q-values if not provided
        if not available_actions:
            available_actions = list(self.q_values[state].keys())
        
        # Make sure all available actions have Q-values
        for action in available_actions:
            if action not in self.q_values[state]:
                self.q_values[state][action] = 0.0
        
        # Using Softmax policy: P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
        # Where τ is temperature (higher = more exploration)
        temperature = max(0.1, 1.0 - self.development_level * 0.8)
        
        # Initialize or clear existing policy
        if state not in self.policy:
            self.policy[state] = {}
        
        # Calculate denominator (sum of exp(Q/τ) for all actions)
        exp_values = [np.exp(self.q_values[state][a] / temperature) for a in available_actions]
        sum_exp = sum(exp_values)
        
        # Calculate probabilities
        if sum_exp > 0:
            for i, action in enumerate(available_actions):
                self.policy[state][action] = exp_values[i] / sum_exp
        else:
            # Fallback to uniform if numerical issues
            prob = 1.0 / len(available_actions)
            for action in available_actions:
                self.policy[state][action] = prob
    
    def _normalize_policy(self, state: str):
        """Ensure policy probabilities sum to 1.0"""
        if state not in self.policy or not self.policy[state]:
            return
        
        total = sum(self.policy[state].values())
        if total <= 0:
            # Reset to uniform if invalid probabilities
            prob = 1.0 / len(self.policy[state])
            for action in self.policy[state]:
                self.policy[state][action] = prob
        elif total != 1.0:
            # Normalize
            for action in self.policy[state]:
                self.policy[state][action] /= total
    
    def _record_action(self, state: str, action: str):
        """Record an action in the history"""
        self.action_history.append({
            "state": state,
            "action": action,
            "timestamp": datetime.now()
        })
        
        # Trim history if needed
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def _perform_experience_replay(self, batch_size: int = 5) -> Dict[str, Any]:
        """Perform experience replay to improve learning"""
        if len(self.experience_buffer) < batch_size:
            return None
        
        # Sample random experiences
        samples = random.sample(list(self.experience_buffer), batch_size)
        
        results = []
        for exp in samples:
            # Apply Q-learning update to each sample
            state = exp["state"]
            action = exp["action"]
            reward = exp["reward"]
            next_state = exp["next_state"]
            
            if state not in self.q_values:
                self.q_values[state] = {}
            if action not in self.q_values[state]:
                self.q_values[state][action] = 0.0
            
            current_q = self.q_values[state][action]
            max_next_q = self._get_max_q_value(next_state)
            
            # Apply discount factor based on time elapsed since experience
            time_factor = 1.0
            if self.development_level >= 0.7:
                # More developed minds can adjust based on recency
                time_delta = (datetime.now() - exp["timestamp"]).total_seconds()
                time_factor = np.exp(-0.001 * time_delta)  # Exponential decay
            
            # Q-learning update
            new_q = current_q + self.learning_rate * time_factor * (
                reward + self.discount_factor * max_next_q - current_q
            )
            
            self.q_values[state][action] = new_q
            
            # Track result
            results.append({
                "state": state,
                "action": action,
                "previous_q": current_q,
                "updated_q": new_q,
                "q_change": new_q - current_q
            })
            
            # Update policy for this state
            self._update_state_policy(state)
        
        return {
            "samples_processed": len(results),
            "updates": results
        }
    
    def _handle_reward(self, message):
        """Handle incoming reward signals"""
        if not message.content:
            return
            
        reward_data = message.content
        
        # Check if we have required fields
        if "state" in reward_data and "action" in reward_data and "reward" in reward_data:
            # Process the reward signal
            self._learn_from_experience(reward_data)
    
    def _handle_action(self, message):
        """Handle action performance messages"""
        if not message.content:
            return
            
        action_data = message.content
        
        # Record this action
        if "state" in action_data and "action" in action_data:
            self._record_action(action_data["state"], action_data["action"])
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # If development changed significantly, adjust parameters
        if abs(new_level - previous_level) >= 0.05:
            self._adjust_for_development()
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        base_state = super().get_state()
        
        # Add reinforcement learning specific state
        module_state = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "states_learned": len(self.q_values),
            "experience_buffer_size": len(self.experience_buffer),
            "action_history_size": len(self.action_history)
        }
        
        base_state.update(module_state)
        return base_state
