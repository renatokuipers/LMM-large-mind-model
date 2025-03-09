"""
Reinforcement Learning Engine Module

This module implements reinforcement learning mechanisms:
- Q-Learning: Learning action values based on rewards
- SARSA: On-policy learning of action values
- Actor-Critic: Learning both a policy and a value function

Reinforcement learning enables the LMM to learn from rewards and feedback,
shaping behavior and associations based on outcomes, and developing
goal-directed behavior through experience.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import logging
import numpy as np
from datetime import datetime
import random

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.learning_engines.models import (
    LearningEngine, ReinforcementParameters, LearningEvent, LearningRuleApplication
)
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class ReinforcementEngine(LearningEngine):
    """
    Reinforcement learning engine for learning from feedback.
    """
    
    def __init__(
        self,
        parameters: Optional[ReinforcementParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the reinforcement learning engine
        
        Args:
            parameters: Parameters for reinforcement learning
            event_bus: Event bus for sending learning events
        """
        super().__init__(
            engine_type="reinforcement",
            learning_rate=parameters.learning_rate if parameters else 0.01
        )
        
        self.parameters = parameters or ReinforcementParameters()
        self.event_bus = event_bus
        
        # Q-values/action values for neurons and connections
        self.neuron_values: Dict[str, float] = {}
        self.synapse_values: Dict[str, float] = {}
        
        # Store state representations
        self.current_state: Dict[str, float] = {}
        self.previous_state: Dict[str, float] = {}
        
        # Store action history
        self.action_history: List[Dict[str, Any]] = []
        
        # Track eligibility traces for credit assignment
        self.eligibility_traces: Dict[str, float] = {}
        
        # Store recent rewards
        self.recent_rewards: List[Tuple[float, datetime]] = []
        
        # Exploration/exploitation balance
        self.exploration_rate = self.parameters.exploration_rate
        
        # Tracking learning applications
        self.learning_applications: List[LearningRuleApplication] = []
        
        logger.info(f"Reinforcement learning engine initialized with method: {self.parameters.update_method}")
    
    def apply_learning(
        self, 
        network: NeuralNetwork, 
        reward: float = 0.0,
        state: Optional[Dict[str, float]] = None
    ) -> List[LearningEvent]:
        """
        Apply reinforcement learning to a neural network
        
        Args:
            network: The neural network to apply learning to
            reward: The reward signal received (positive or negative)
            state: Optional explicit state representation
            
        Returns:
            List of learning events that occurred
        """
        if not self.is_active:
            return []
        
        # Scale the reward
        scaled_reward = self._scale_reward(reward)
        
        # Record reward
        self.recent_rewards.append((scaled_reward, datetime.now()))
        self.recent_rewards = self.recent_rewards[-100:]  # Keep last 100 rewards
        
        # Update state representation
        self._update_state(network, state)
        
        # Select the appropriate learning method
        if self.parameters.update_method == "q_learning":
            events = self._apply_q_learning(network, scaled_reward)
        elif self.parameters.update_method == "sarsa":
            events = self._apply_sarsa(network, scaled_reward)
        elif self.parameters.update_method == "actor_critic":
            events = self._apply_actor_critic(network, scaled_reward)
        else:
            logger.warning(f"Unknown learning method: {self.parameters.update_method}")
            events = []
            
        # Update last applied timestamp
        self.last_applied = datetime.now()
        
        # Broadcast learning events if event bus is available
        if self.event_bus:
            for event in events:
                message = Message(
                    sender="reinforcement_engine",
                    message_type="learning_event",
                    content=event.dict()
                )
                self.event_bus.publish(message)
        
        return events
    
    def _update_state(
        self, 
        network: NeuralNetwork, 
        explicit_state: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update the current state representation
        
        Args:
            network: The neural network
            explicit_state: Optional explicit state representation
        """
        # Store the previous state
        self.previous_state = self.current_state.copy()
        
        if explicit_state:
            # Use provided state
            self.current_state = explicit_state
        else:
            # Derive state from network activity
            new_state = {}
            
            # Use neuron activations as state
            for neuron_id, neuron in network.neurons.items():
                new_state[neuron_id] = neuron.activation
                
                # Update or initialize neuron value
                if neuron_id not in self.neuron_values:
                    self.neuron_values[neuron_id] = 0.0
            
            self.current_state = new_state
    
    def _scale_reward(self, reward: float) -> float:
        """
        Scale the reward according to parameters
        
        Args:
            reward: Raw reward value
            
        Returns:
            Scaled reward
        """
        # Apply scaling factor
        scaled_reward = reward * self.parameters.reward_scaling
        
        # Clip to bounds
        return max(self.parameters.min_reward, min(self.parameters.max_reward, scaled_reward))
    
    def _apply_q_learning(self, network: NeuralNetwork, reward: float) -> List[LearningEvent]:
        """
        Apply Q-learning update rule
        
        Args:
            network: The neural network
            reward: The scaled reward signal
            
        Returns:
            List of learning events
        """
        affected_neurons = []
        affected_synapses = []
        neuron_changes = []
        synapse_changes = []
        total_change = 0.0
        
        # Q-learning is off-policy TD learning: Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        
        # Get the max value of the next state (if no next state, use 0)
        if self.current_state:
            # Get all neuron values in the current state
            state_values = [self.neuron_values.get(n_id, 0.0) for n_id in self.current_state]
            max_next_value = max(state_values) if state_values else 0.0
        else:
            max_next_value = 0.0
        
        # Update eligibility traces
        self._update_eligibility_traces(network)
        
        # Process all neurons (as actions)
        for neuron_id, neuron in network.neurons.items():
            if neuron_id not in self.neuron_values:
                self.neuron_values[neuron_id] = 0.0
                
            # Current Q-value
            current_value = self.neuron_values[neuron_id]
            
            # Calculate the temporal difference error
            td_error = reward + self.parameters.discount_factor * max_next_value - current_value
            
            # Get eligibility trace for this neuron
            eligibility = self.eligibility_traces.get(neuron_id, 0.0)
            
            # Update only if there's a significant change or high eligibility
            if abs(td_error) > 0.01 or eligibility > 0.1:
                # Apply the update rule with eligibility trace
                new_value = current_value + self.parameters.learning_rate * td_error * eligibility
                delta = new_value - current_value
                
                # Update the value
                self.neuron_values[neuron_id] = new_value
                
                # Record the update
                affected_neurons.append(neuron_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        neuron_id=neuron_id,
                        rule_type="q_learning",
                        pre_value=current_value,
                        post_value=new_value,
                        delta=delta
                    )
                )
                
                neuron_changes.append((neuron_id, current_value, new_value, delta))
        
        # Update synapses based on the TD error and eligibility
        for synapse_id, synapse in network.synapses.items():
            if synapse_id not in self.synapse_values:
                self.synapse_values[synapse_id] = synapse.weight
                
            # Current weight as Q-value
            current_value = synapse.weight
            
            # Get eligibility trace for this synapse
            eligibility = self.eligibility_traces.get(synapse_id, 0.0)
            
            # Calculate TD error (using neuron TD errors proportionally)
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id in affected_neurons and target_id in affected_neurons:
                # Find the neuron changes for source and target
                source_delta = next((delta for n_id, _, _, delta in neuron_changes if n_id == source_id), 0.0)
                target_delta = next((delta for n_id, _, _, delta in neuron_changes if n_id == target_id), 0.0)
                
                # Combined TD error
                td_error = (source_delta + target_delta) / 2
                
                # Update only if there's a significant change or high eligibility
                if abs(td_error) > 0.01 or eligibility > 0.1:
                    # Apply the update rule with eligibility trace
                    weight_delta = self.parameters.learning_rate * td_error * eligibility
                    new_weight = current_value + weight_delta
                    
                    # Clip weight to bounds (using parameter bounds or default)
                    min_weight = getattr(self.parameters, "min_weight", -1.0)
                    max_weight = getattr(self.parameters, "max_weight", 1.0)
                    new_weight = max(min_weight, min(max_weight, new_weight))
                    
                    delta = new_weight - current_value
                    
                    # Apply the weight update
                    synapse.update_weight(delta)
                    
                    # Update neuron connection
                    if source_id in network.neurons:
                        network.neurons[source_id].adjust_weight(target_id, delta)
                    
                    # Update the synapse value
                    self.synapse_values[synapse_id] = new_weight
                    
                    # Record the update
                    affected_synapses.append(synapse_id)
                    total_change += abs(delta)
                    
                    # Record the learning application
                    self.learning_applications.append(
                        LearningRuleApplication(
                            synapse_id=synapse_id,
                            rule_type="q_learning",
                            pre_value=current_value,
                            post_value=new_weight,
                            delta=delta
                        )
                    )
                    
                    synapse_changes.append((synapse_id, current_value, new_weight, delta))
        
        # Create learning events for neurons and synapses
        events = []
        
        if affected_neurons:
            neuron_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                neurons_affected=affected_neurons,
                magnitude=total_change / (len(affected_neurons) + 0.001),
                details={
                    "rule": "q_learning",
                    "reward": reward,
                    "neuron_changes": [
                        {
                            "neuron_id": n_id,
                            "old_value": old,
                            "new_value": new,
                            "delta": delta
                        } for n_id, old, new, delta in neuron_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(neuron_event)
            
        if affected_synapses:
            synapse_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / (len(affected_synapses) + 0.001),
                details={
                    "rule": "q_learning",
                    "reward": reward,
                    "synapse_changes": [
                        {
                            "synapse_id": s_id,
                            "old_weight": old,
                            "new_weight": new,
                            "delta": delta
                        } for s_id, old, new, delta in synapse_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(synapse_event)
        
        return events
    
    def _apply_sarsa(self, network: NeuralNetwork, reward: float) -> List[LearningEvent]:
        """
        Apply SARSA (State-Action-Reward-State-Action) update rule
        
        Args:
            network: The neural network
            reward: The scaled reward signal
            
        Returns:
            List of learning events
        """
        affected_neurons = []
        affected_synapses = []
        neuron_changes = []
        synapse_changes = []
        total_change = 0.0
        
        # SARSA is on-policy TD learning: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        
        # For on-policy learning, we need to select an action from the current state
        # using the current policy (typically ε-greedy)
        if self.current_state:
            # Get the value of the current state-action (represented by neuron activations)
            current_state_value = sum(
                self.neuron_values.get(n_id, 0.0) * activation 
                for n_id, activation in self.current_state.items()
            ) / max(len(self.current_state), 1)
        else:
            current_state_value = 0.0
        
        # Update eligibility traces
        self._update_eligibility_traces(network)
        
        # Process all neurons (as state-action values)
        for neuron_id, neuron in network.neurons.items():
            if neuron_id not in self.neuron_values:
                self.neuron_values[neuron_id] = 0.0
                
            # Current Q-value
            current_value = self.neuron_values[neuron_id]
            
            # Calculate the temporal difference error
            td_error = reward + self.parameters.discount_factor * current_state_value - current_value
            
            # Get eligibility trace for this neuron
            eligibility = self.eligibility_traces.get(neuron_id, 0.0)
            
            # Update only if there's a significant change or high eligibility
            if abs(td_error) > 0.01 or eligibility > 0.1:
                # Apply the update rule with eligibility trace
                new_value = current_value + self.parameters.learning_rate * td_error * eligibility
                delta = new_value - current_value
                
                # Update the value
                self.neuron_values[neuron_id] = new_value
                
                # Record the update
                affected_neurons.append(neuron_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        neuron_id=neuron_id,
                        rule_type="sarsa",
                        pre_value=current_value,
                        post_value=new_value,
                        delta=delta
                    )
                )
                
                neuron_changes.append((neuron_id, current_value, new_value, delta))
        
        # Update synapses based on the TD error and eligibility
        for synapse_id, synapse in network.synapses.items():
            if synapse_id not in self.synapse_values:
                self.synapse_values[synapse_id] = synapse.weight
                
            # Current weight as Q-value
            current_value = synapse.weight
            
            # Get eligibility trace for this synapse
            eligibility = self.eligibility_traces.get(synapse_id, 0.0)
            
            # Calculate TD error (using neuron TD errors proportionally)
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id in affected_neurons and target_id in affected_neurons:
                # Find the neuron changes for source and target
                source_delta = next((delta for n_id, _, _, delta in neuron_changes if n_id == source_id), 0.0)
                target_delta = next((delta for n_id, _, _, delta in neuron_changes if n_id == target_id), 0.0)
                
                # Combined TD error
                td_error = (source_delta + target_delta) / 2
                
                # Update only if there's a significant change or high eligibility
                if abs(td_error) > 0.01 or eligibility > 0.1:
                    # Apply the update rule with eligibility trace
                    weight_delta = self.parameters.learning_rate * td_error * eligibility
                    new_weight = current_value + weight_delta
                    
                    # Clip weight to bounds (using parameter bounds or default)
                    min_weight = getattr(self.parameters, "min_weight", -1.0)
                    max_weight = getattr(self.parameters, "max_weight", 1.0)
                    new_weight = max(min_weight, min(max_weight, new_weight))
                    
                    delta = new_weight - current_value
                    
                    # Apply the weight update
                    synapse.update_weight(delta)
                    
                    # Update neuron connection
                    if source_id in network.neurons:
                        network.neurons[source_id].adjust_weight(target_id, delta)
                    
                    # Update the synapse value
                    self.synapse_values[synapse_id] = new_weight
                    
                    # Record the update
                    affected_synapses.append(synapse_id)
                    total_change += abs(delta)
                    
                    # Record the learning application
                    self.learning_applications.append(
                        LearningRuleApplication(
                            synapse_id=synapse_id,
                            rule_type="sarsa",
                            pre_value=current_value,
                            post_value=new_weight,
                            delta=delta
                        )
                    )
                    
                    synapse_changes.append((synapse_id, current_value, new_weight, delta))
        
        # Create learning events for neurons and synapses
        events = []
        
        if affected_neurons:
            neuron_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                neurons_affected=affected_neurons,
                magnitude=total_change / (len(affected_neurons) + 0.001),
                details={
                    "rule": "sarsa",
                    "reward": reward,
                    "neuron_changes": [
                        {
                            "neuron_id": n_id,
                            "old_value": old,
                            "new_value": new,
                            "delta": delta
                        } for n_id, old, new, delta in neuron_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(neuron_event)
            
        if affected_synapses:
            synapse_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / (len(affected_synapses) + 0.001),
                details={
                    "rule": "sarsa",
                    "reward": reward,
                    "synapse_changes": [
                        {
                            "synapse_id": s_id,
                            "old_weight": old,
                            "new_weight": new,
                            "delta": delta
                        } for s_id, old, new, delta in synapse_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(synapse_event)
        
        return events
    
    def _apply_actor_critic(self, network: NeuralNetwork, reward: float) -> List[LearningEvent]:
        """
        Apply Actor-Critic learning update
        
        Args:
            network: The neural network
            reward: The scaled reward signal
            
        Returns:
            List of learning events
        """
        affected_neurons = []
        affected_synapses = []
        neuron_changes = []
        synapse_changes = []
        total_change = 0.0
        
        # Actor-Critic separates:
        # - Critic: learns state values V(s)
        # - Actor: updates policy based on critic's evaluation
        
        # For simplicity, we'll use output neurons as actors and hidden neurons as critics
        
        # Calculate current state value (critic)
        if self.current_state and self.previous_state:
            # Get average value of previous and current states
            prev_state_value = sum(
                self.neuron_values.get(n_id, 0.0) * activation 
                for n_id, activation in self.previous_state.items()
            ) / max(len(self.previous_state), 1)
            
            current_state_value = sum(
                self.neuron_values.get(n_id, 0.0) * activation 
                for n_id, activation in self.current_state.items()
            ) / max(len(self.current_state), 1)
            
            # Calculate temporal difference error
            td_error = reward + self.parameters.discount_factor * current_state_value - prev_state_value
        else:
            # No previous state, use reward as TD error
            td_error = reward
            prev_state_value = 0.0
            current_state_value = 0.0
        
        # Update eligibility traces
        self._update_eligibility_traces(network)
        
        # Update critic (state values in hidden/non-output neurons)
        for neuron_id, neuron in network.neurons.items():
            # Skip output neurons (they're actors)
            if neuron_id in network.output_neurons:
                continue
                
            if neuron_id not in self.neuron_values:
                self.neuron_values[neuron_id] = 0.0
                
            # Current value
            current_value = self.neuron_values[neuron_id]
            
            # Get eligibility trace for this neuron
            eligibility = self.eligibility_traces.get(neuron_id, 0.0)
            
            # Update only if there's a significant error or high eligibility
            if abs(td_error) > 0.01 or eligibility > 0.1:
                # Apply the critic update rule with eligibility trace
                new_value = current_value + self.parameters.learning_rate * td_error * eligibility
                delta = new_value - current_value
                
                # Update the value
                self.neuron_values[neuron_id] = new_value
                
                # Record the update
                affected_neurons.append(neuron_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        neuron_id=neuron_id,
                        rule_type="actor_critic",
                        pre_value=current_value,
                        post_value=new_value,
                        delta=delta
                    )
                )
                
                neuron_changes.append((neuron_id, current_value, new_value, delta))
        
        # Update actor (policy parameters in output neurons and their connections)
        for neuron_id in network.output_neurons:
            if neuron_id in network.neurons:
                neuron = network.neurons[neuron_id]
                
                if neuron_id not in self.neuron_values:
                    self.neuron_values[neuron_id] = 0.0
                    
                # Current value
                current_value = self.neuron_values[neuron_id]
                activation = neuron.activation
                
                # Determine policy gradient direction
                # In a simple case: if TD error is positive, reinforce current action, otherwise discourage it
                policy_gradient = td_error * activation
                
                # Get eligibility trace for this neuron
                eligibility = self.eligibility_traces.get(neuron_id, 0.0)
                
                # Update only if there's a significant gradient or high eligibility
                if abs(policy_gradient) > 0.01 or eligibility > 0.1:
                    # Apply the actor update rule with eligibility trace
                    new_value = current_value + self.parameters.learning_rate * policy_gradient * eligibility
                    delta = new_value - current_value
                    
                    # Update the value
                    self.neuron_values[neuron_id] = new_value
                    
                    # Record the update
                    affected_neurons.append(neuron_id)
                    total_change += abs(delta)
                    
                    # Record the learning application
                    self.learning_applications.append(
                        LearningRuleApplication(
                            neuron_id=neuron_id,
                            rule_type="actor_critic",
                            pre_value=current_value,
                            post_value=new_value,
                            delta=delta
                        )
                    )
                    
                    neuron_changes.append((neuron_id, current_value, new_value, delta))
        
        # Update policy implementation (synapses)
        for synapse_id, synapse in network.synapses.items():
            if synapse_id not in self.synapse_values:
                self.synapse_values[synapse_id] = synapse.weight
                
            # Current weight
            current_weight = synapse.weight
            
            # Get eligibility trace for this synapse
            eligibility = self.eligibility_traces.get(synapse_id, 0.0)
            
            # Target neuron is an output neuron (actor)
            target_id = synapse.target_id
            if target_id in network.output_neurons:
                # Apply policy gradient
                weight_delta = self.parameters.learning_rate * td_error * eligibility
                new_weight = current_weight + weight_delta
                
                # Clip weight to bounds (using parameter bounds or default)
                min_weight = getattr(self.parameters, "min_weight", -1.0)
                max_weight = getattr(self.parameters, "max_weight", 1.0)
                new_weight = max(min_weight, min(max_weight, new_weight))
                
                delta = new_weight - current_weight
                
                if abs(delta) > 0.001:  # Only update if significant change
                    # Apply the weight update
                    synapse.update_weight(delta)
                    
                    # Update neuron connection
                    source_id = synapse.source_id
                    if source_id in network.neurons:
                        network.neurons[source_id].adjust_weight(target_id, delta)
                    
                    # Update the synapse value
                    self.synapse_values[synapse_id] = new_weight
                    
                    # Record the update
                    affected_synapses.append(synapse_id)
                    total_change += abs(delta)
                    
                    # Record the learning application
                    self.learning_applications.append(
                        LearningRuleApplication(
                            synapse_id=synapse_id,
                            rule_type="actor_critic",
                            pre_value=current_weight,
                            post_value=new_weight,
                            delta=delta
                        )
                    )
                    
                    synapse_changes.append((synapse_id, current_weight, new_weight, delta))
        
        # Create learning events for neurons and synapses
        events = []
        
        if affected_neurons:
            neuron_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                neurons_affected=affected_neurons,
                magnitude=total_change / (len(affected_neurons) + 0.001),
                details={
                    "rule": "actor_critic",
                    "reward": reward,
                    "td_error": td_error,
                    "prev_state_value": prev_state_value,
                    "current_state_value": current_state_value,
                    "neuron_changes": [
                        {
                            "neuron_id": n_id,
                            "old_value": old,
                            "new_value": new,
                            "delta": delta
                        } for n_id, old, new, delta in neuron_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(neuron_event)
            
        if affected_synapses:
            synapse_event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / (len(affected_synapses) + 0.001),
                details={
                    "rule": "actor_critic",
                    "reward": reward,
                    "td_error": td_error,
                    "synapse_changes": [
                        {
                            "synapse_id": s_id,
                            "old_weight": old,
                            "new_weight": new,
                            "delta": delta
                        } for s_id, old, new, delta in synapse_changes[:10]  # Include first 10 changes
                    ]
                }
            )
            events.append(synapse_event)
        
        return events
    
    def _update_eligibility_traces(self, network: NeuralNetwork) -> None:
        """
        Update eligibility traces for all neurons and synapses
        
        Args:
            network: The neural network
        """
        # Decay all existing traces
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= self.parameters.eligibility_trace_decay
            
            # Remove trace if it's too small
            if self.eligibility_traces[key] < 0.01:
                del self.eligibility_traces[key]
        
        # Update traces for active neurons
        for neuron_id, neuron in network.neurons.items():
            if neuron.activation > 0:
                # Increase eligibility proportionally to activation
                if neuron_id not in self.eligibility_traces:
                    self.eligibility_traces[neuron_id] = 0.0
                    
                self.eligibility_traces[neuron_id] += neuron.activation
                self.eligibility_traces[neuron_id] = min(1.0, self.eligibility_traces[neuron_id])
        
        # Update traces for active synapses (based on pre-synaptic activity)
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            
            if source_id in network.neurons and network.neurons[source_id].activation > 0:
                if synapse_id not in self.eligibility_traces:
                    self.eligibility_traces[synapse_id] = 0.0
                    
                # Increase eligibility proportionally to source activation and weight
                source_activation = network.neurons[source_id].activation
                self.eligibility_traces[synapse_id] += source_activation * abs(synapse.weight)
                self.eligibility_traces[synapse_id] = min(1.0, self.eligibility_traces[synapse_id])
    
    def provide_reward(self, reward: float) -> None:
        """
        Provide an external reward signal
        
        Args:
            reward: Reward value (positive or negative)
        """
        # Scale the reward
        scaled_reward = self._scale_reward(reward)
        
        # Record the reward
        self.recent_rewards.append((scaled_reward, datetime.now()))
        self.recent_rewards = self.recent_rewards[-100:]  # Keep last 100 rewards
        
        logger.info(f"Reward provided: {reward} (scaled: {scaled_reward})")
    
    def get_action_selection(self, 
                           options: Dict[str, float], 
                           use_exploration: bool = True) -> Tuple[str, float]:
        """
        Select an action using the current policy (ε-greedy)
        
        Args:
            options: Dictionary mapping action IDs to values
            use_exploration: Whether to use exploration (ε-greedy) or pure greedy
            
        Returns:
            Tuple of (selected_action_id, value)
        """
        if not options:
            return None, 0.0
            
        # Determine whether to explore or exploit
        if use_exploration and random.random() < self.exploration_rate:
            # Explore: random action
            action_id = random.choice(list(options.keys()))
            value = options[action_id]
        else:
            # Exploit: best action
            action_id = max(options.items(), key=lambda x: x[1])[0]
            value = options[action_id]
            
        return action_id, value
    
    def set_exploration_rate(self, rate: float) -> None:
        """
        Set the exploration rate (epsilon)
        
        Args:
            rate: New exploration rate (0.0 to 1.0)
        """
        self.exploration_rate = max(0.0, min(1.0, rate))
        self.parameters.exploration_rate = self.exploration_rate
        logger.info(f"Changed exploration rate to: {rate}")
    
    def set_learning_method(self, method: str) -> None:
        """
        Set the reinforcement learning method
        
        Args:
            method: Learning method ("q_learning", "sarsa", "actor_critic")
        """
        if method in ["q_learning", "sarsa", "actor_critic"]:
            self.parameters.update_method = method
            logger.info(f"Changed reinforcement learning method to: {method}")
        else:
            logger.warning(f"Unknown learning method: {method}")
    
    def decay_exploration(self, decay_factor: float = 0.99) -> None:
        """
        Decay the exploration rate
        
        Args:
            decay_factor: Factor to multiply current exploration rate by
        """
        self.exploration_rate *= decay_factor
        self.parameters.exploration_rate = self.exploration_rate
        logger.info(f"Decayed exploration rate to: {self.exploration_rate}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the reinforcement learning engine
        
        Returns:
            Dictionary with engine state
        """
        return {
            "engine_id": self.engine_id,
            "engine_type": self.engine_type,
            "learning_rate": self.learning_rate,
            "is_active": self.is_active,
            "parameters": self.parameters.dict(),
            "exploration_rate": self.exploration_rate,
            "neuron_values_count": len(self.neuron_values),
            "synapse_values_count": len(self.synapse_values),
            "eligibility_traces_count": len(self.eligibility_traces),
            "recent_rewards_count": len(self.recent_rewards),
            "learning_applications_count": len(self.learning_applications),
            "last_applied": self.last_applied
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: Dictionary with engine state
        """
        if "parameters" in state:
            self.parameters = ReinforcementParameters(**state["parameters"])
            
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
            
        if "is_active" in state:
            self.is_active = state["is_active"]
            
        if "exploration_rate" in state:
            self.exploration_rate = state["exploration_rate"]
            
        if "last_applied" in state:
            self.last_applied = state["last_applied"]
