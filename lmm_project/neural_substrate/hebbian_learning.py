"""
Hebbian learning implementation for the neural substrate.

This module provides mechanisms for Hebbian learning, implementing the principle
that "neurons that fire together, wire together." It offers various formulations
of Hebbian learning for synaptic weight adjustment.
"""
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lmm_project.neural_substrate.synapse import PlasticityType, Synapse
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("neural_substrate.hebbian_learning")


class HebbianRule(Enum):
    """Different formulations of Hebbian learning rules."""
    BASIC = auto()           # Basic Hebbian: weight += lr * pre * post
    COVARIANCE = auto()      # Covariance rule: weight += lr * (pre - mean_pre) * (post - mean_post)
    BCM = auto()             # Bienenstock-Cooper-Munro: weight += lr * pre * post * (post - theta)
    OJA = auto()             # Oja's rule: weight += lr * post * (pre - post * weight)
    GENERALIZED = auto()     # Generalized Hebbian Algorithm (GHA): PCA-based learning


class HebbianLearner:
    """
    Implementation of Hebbian learning for neural networks.
    
    Applies Hebbian plasticity rules to modify synaptic weights based on
    the activities of connected neurons.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        rule: HebbianRule = HebbianRule.BASIC,
        requires_history: bool = False,
        history_window: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a Hebbian learning mechanism.
        
        Parameters:
        learning_rate: Base learning rate
        rule: Hebbian learning rule to apply
        requires_history: Whether to track activity history for certain rules
        history_window: Size of activity history window if required
        device: Torch device for tensor operations
        """
        self.learning_rate = learning_rate
        self.rule = rule
        self.requires_history = requires_history or rule in (HebbianRule.COVARIANCE, HebbianRule.BCM)
        self.history_window = history_window
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Activity history for rules that need it
        self.pre_activity_history: List[float] = []
        self.post_activity_history: List[float] = []
        
        # BCM threshold variables
        self.theta = 0.5  # Sliding threshold for BCM rule
        self.theta_update_rate = 0.01  # Rate for threshold adaptation
        
        # Select learning rule implementation
        self.learning_functions = {
            HebbianRule.BASIC: self._basic_hebbian,
            HebbianRule.COVARIANCE: self._covariance_rule,
            HebbianRule.BCM: self._bcm_rule,
            HebbianRule.OJA: self._oja_rule,
            HebbianRule.GENERALIZED: self._generalized_hebbian
        }
        
        logger.info(f"Initialized HebbianLearner with rule: {rule.name}")
    
    def update_synapse(
        self, 
        synapse: Synapse, 
        pre_activation: float, 
        post_activation: float
    ) -> None:
        """
        Update a synapse using the selected Hebbian rule.
        
        Parameters:
        synapse: Synapse to update
        pre_activation: Activation of presynaptic neuron
        post_activation: Activation of postsynaptic neuron
        """
        # Skip update if plasticity is disabled for this synapse
        if not synapse.config.plasticity_enabled:
            return
            
        # Only update if we have real activations
        if pre_activation == 0.0 and post_activation == 0.0:
            return
        
        # Track activity history if needed
        if self.requires_history:
            self.pre_activity_history.append(pre_activation)
            self.post_activity_history.append(post_activation)
            
            # Trim history to window size
            if len(self.pre_activity_history) > self.history_window:
                self.pre_activity_history = self.pre_activity_history[-self.history_window:]
                self.post_activity_history = self.post_activity_history[-self.history_window:]
        
        # Get the appropriate learning rule function
        learning_rule_fn = self.learning_functions.get(self.rule)
        
        # Calculate weight change using the selected rule
        weight_change = learning_rule_fn(
            synapse.weight.item(), 
            pre_activation, 
            post_activation
        )
        
        # Apply weight change to synapse
        with torch.no_grad():
            synapse.weight += weight_change
            
        # Apply weight constraints
        synapse._constrain_weight()
    
    def update_synapses(
        self, 
        synapses: List[Synapse], 
        pre_activations: List[float], 
        post_activations: List[float]
    ) -> None:
        """
        Update multiple synapses using the selected Hebbian rule.
        
        Parameters:
        synapses: List of synapses to update
        pre_activations: List of presynaptic neuron activations
        post_activations: List of postsynaptic neuron activations
        """
        for synapse, pre_act, post_act in zip(synapses, pre_activations, post_activations):
            self.update_synapse(synapse, pre_act, post_act)
    
    def _basic_hebbian(
        self, 
        weight: float, 
        pre_activation: float, 
        post_activation: float
    ) -> float:
        """
        Basic Hebbian rule: weight += lr * pre * post
        
        Parameters:
        weight: Current weight
        pre_activation: Presynaptic neuron activation
        post_activation: Postsynaptic neuron activation
        
        Returns:
        Weight change amount
        """
        return self.learning_rate * pre_activation * post_activation
    
    def _covariance_rule(
        self, 
        weight: float, 
        pre_activation: float, 
        post_activation: float
    ) -> float:
        """
        Covariance Hebbian rule: weight += lr * (pre - mean_pre) * (post - mean_post)
        
        Parameters:
        weight: Current weight
        pre_activation: Presynaptic neuron activation
        post_activation: Postsynaptic neuron activation
        
        Returns:
        Weight change amount
        """
        # Calculate means from history
        if not self.pre_activity_history:
            return 0.0
            
        mean_pre = sum(self.pre_activity_history) / len(self.pre_activity_history)
        mean_post = sum(self.post_activity_history) / len(self.post_activity_history)
        
        # Apply covariance rule
        return self.learning_rate * (pre_activation - mean_pre) * (post_activation - mean_post)
    
    def _bcm_rule(
        self, 
        weight: float, 
        pre_activation: float, 
        post_activation: float
    ) -> float:
        """
        Bienenstock-Cooper-Munro rule: weight += lr * pre * post * (post - theta)
        
        Parameters:
        weight: Current weight
        pre_activation: Presynaptic neuron activation
        post_activation: Postsynaptic neuron activation
        
        Returns:
        Weight change amount
        """
        # Update sliding threshold based on recent postsynaptic activity
        if self.post_activity_history:
            mean_post_squared = np.mean([p**2 for p in self.post_activity_history])
            self.theta += self.theta_update_rate * (mean_post_squared - self.theta)
            
        # Apply BCM rule
        return self.learning_rate * pre_activation * post_activation * (post_activation - self.theta)
    
    def _oja_rule(
        self, 
        weight: float, 
        pre_activation: float, 
        post_activation: float
    ) -> float:
        """
        Oja's rule: weight += lr * post * (pre - post * weight)
        
        Parameters:
        weight: Current weight
        pre_activation: Presynaptic neuron activation
        post_activation: Postsynaptic neuron activation
        
        Returns:
        Weight change amount
        """
        return self.learning_rate * post_activation * (pre_activation - post_activation * weight)
    
    def _generalized_hebbian(
        self, 
        weight: float, 
        pre_activation: float, 
        post_activation: float
    ) -> float:
        """
        Generalized Hebbian Algorithm (GHA): PCA-based learning
        
        Parameters:
        weight: Current weight
        pre_activation: Presynaptic neuron activation
        post_activation: Postsynaptic neuron activation
        
        Returns:
        Weight change amount
        """
        # This is a simplified version for individual synapse updates
        # Full GHA requires knowledge of all weights to same postsynaptic neuron
        return self.learning_rate * (post_activation * pre_activation - 
                                   post_activation**2 * weight)
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate.
        
        Parameters:
        learning_rate: New learning rate
        """
        self.learning_rate = learning_rate
    
    def set_rule(self, rule: HebbianRule) -> None:
        """
        Set the Hebbian learning rule to use.
        
        Parameters:
        rule: New Hebbian rule
        """
        self.rule = rule
        self.requires_history = rule in (HebbianRule.COVARIANCE, HebbianRule.BCM)
        
        # Clear history if no longer needed
        if not self.requires_history:
            self.pre_activity_history = []
            self.post_activity_history = []
            
        logger.info(f"Changed Hebbian learning rule to: {rule.name}")
    
    def reset(self) -> None:
        """Reset learner state."""
        self.pre_activity_history = []
        self.post_activity_history = []
        self.theta = 0.5
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the Hebbian learner.
        
        Returns:
        Dictionary containing learner state
        """
        return {
            "learning_rate": self.learning_rate,
            "rule": self.rule.name,
            "requires_history": self.requires_history,
            "history_window": self.history_window,
            "theta": self.theta,
            "theta_update_rate": self.theta_update_rate
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load learner state from a dictionary.
        
        Parameters:
        state: Dictionary containing learner state
        """
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
        if "rule" in state:
            try:
                self.rule = HebbianRule[state["rule"]]
                self.requires_history = self.rule in (HebbianRule.COVARIANCE, HebbianRule.BCM)
            except KeyError:
                logger.warning(f"Unknown Hebbian rule in state: {state['rule']}")
        if "theta" in state:
            self.theta = state["theta"]
        if "theta_update_rate" in state:
            self.theta_update_rate = state["theta_update_rate"]
