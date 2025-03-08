"""
Hebbian Learning Engine Module

This module implements Hebbian learning and its variants:
- Classic Hebbian Learning: "Neurons that fire together, wire together"
- Oja's Rule: A normalized version of Hebbian learning
- BCM Rule: Bienenstock-Cooper-Munro rule for LTP/LTD
- STDP: Spike-Timing-Dependent Plasticity

Hebbian learning is a core associative learning mechanism in the LMM, enabling
the formation of connections between neurons that are activated in close temporal
proximity. This creates emergent associations and pattern recognition.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import numpy as np
from datetime import datetime, timedelta

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.neural_substrate.hebbian_learning import HebbianLearning
from lmm_project.learning_engines.models import (
    LearningEngine, HebbianParameters, LearningEvent, LearningRuleApplication,
    NeuronUsageStats, SynapseUsageStats
)
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class HebbianEngine(LearningEngine):
    """
    Implementation of Hebbian learning and variants
    
    This engine applies Hebbian learning rules to modify connections
    between neurons based on their co-activation patterns, strengthening
    connections between neurons that activate together.
    """
    
    def __init__(
        self,
        parameters: Optional[HebbianParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the Hebbian learning engine
        
        Args:
            parameters: Parameters for Hebbian learning
            event_bus: Event bus for sending learning events
        """
        super().__init__(
            engine_type="hebbian",
            learning_rate=parameters.learning_rate if parameters else 0.01
        )
        
        self.parameters = parameters or HebbianParameters()
        self.event_bus = event_bus
        
        # Track neuron and synapse activity for learning
        self.neuron_activations: Dict[str, float] = {}
        self.previous_activations: Dict[str, float] = {}
        self.synapse_activations: Dict[str, float] = {}
        
        # Track spike times for STDP
        self.neuron_spike_times: Dict[str, List[datetime]] = {}
        
        # Usage statistics for neurons and synapses
        self.neuron_stats: Dict[str, NeuronUsageStats] = {}
        self.synapse_stats: Dict[str, SynapseUsageStats] = {}
        
        # Track learning applications
        self.learning_applications: List[LearningRuleApplication] = []
        
        # Hebbian learning implementation
        self.hebbian_learning = HebbianLearning(
            learning_rate=self.parameters.learning_rate,
            decay_rate=self.parameters.decay_rate,
            min_weight=self.parameters.min_weight,
            max_weight=self.parameters.max_weight
        )
        
        logger.info(f"Hebbian learning engine initialized with rule: {self.parameters.learning_rule}")
    
    def apply_learning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply Hebbian learning to a neural network
        
        Args:
            network: The neural network to apply learning to
            
        Returns:
            List of learning events that occurred
        """
        if not self.is_active:
            return []
            
        # Update activation records
        self._update_activations(network)
        
        # Apply the appropriate learning rule
        if self.parameters.learning_rule == "hebbian":
            events = self._apply_hebbian_rule(network)
        elif self.parameters.learning_rule == "oja":
            events = self._apply_oja_rule(network)
        elif self.parameters.learning_rule == "bcm":
            events = self._apply_bcm_rule(network)
        elif self.parameters.learning_rule == "stdp":
            events = self._apply_stdp_rule(network)
        else:
            logger.warning(f"Unknown learning rule: {self.parameters.learning_rule}")
            events = []
            
        # Update last applied timestamp
        self.last_applied = datetime.now()
        
        # Broadcast learning events if event bus is available
        if self.event_bus:
            for event in events:
                message = Message(
                    sender="hebbian_engine",
                    message_type="learning_event",
                    content=event.dict()
                )
                self.event_bus.publish(message)
        
        return events
    
    def _update_activations(self, network: NeuralNetwork) -> None:
        """
        Update activation records for all neurons and synapses
        
        Args:
            network: The neural network
        """
        # Store previous activations
        self.previous_activations = self.neuron_activations.copy()
        
        # Update neuron activations
        for neuron_id, neuron in network.neurons.items():
            # Record activation
            self.neuron_activations[neuron_id] = neuron.activation
            
            # Update spike times for STDP if neuron is firing
            if neuron.activation >= neuron.activation_threshold:
                if neuron_id not in self.neuron_spike_times:
                    self.neuron_spike_times[neuron_id] = []
                self.neuron_spike_times[neuron_id].append(datetime.now())
                
                # Keep only recent spike times (within time window)
                time_window = timedelta(milliseconds=self.parameters.time_window)
                current_time = datetime.now()
                self.neuron_spike_times[neuron_id] = [
                    t for t in self.neuron_spike_times[neuron_id] 
                    if current_time - t <= time_window
                ]
            
            # Update usage statistics
            if neuron_id not in self.neuron_stats:
                self.neuron_stats[neuron_id] = NeuronUsageStats(neuron_id=neuron_id)
                
            stats = self.neuron_stats[neuron_id]
            if neuron.activation > 0:
                stats.activation_count += 1
                stats.total_activation += neuron.activation
                stats.average_activation = stats.total_activation / stats.activation_count
                stats.last_activated = datetime.now()
        
        # Update synapse activations
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            if source_id in self.neuron_activations:
                activation = self.neuron_activations[source_id] * abs(synapse.weight)
                self.synapse_activations[synapse_id] = activation
                
                # Update usage statistics
                if synapse_id not in self.synapse_stats:
                    self.synapse_stats[synapse_id] = SynapseUsageStats(synapse_id=synapse_id)
                    
                stats = self.synapse_stats[synapse_id]
                if activation > 0:
                    stats.transmission_count += 1
                    stats.total_transmission += activation
                    stats.average_transmission = stats.total_transmission / stats.transmission_count
                    stats.last_used = datetime.now()
    
    def _apply_hebbian_rule(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply classic Hebbian learning rule
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events
        """
        affected_synapses = []
        synapse_changes = []
        total_change = 0.0
        
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id not in self.neuron_activations or target_id not in self.neuron_activations:
                continue
                
            pre_activation = self.neuron_activations[source_id]
            post_activation = self.neuron_activations[target_id]
            
            # Apply modulated Hebbian rule
            old_weight = synapse.weight
            new_weight = self.hebbian_learning.update_weight(
                pre_activation, 
                post_activation * self.parameters.modulation_factor, 
                old_weight
            )
            
            # Update the synapse weight if significant change
            if abs(new_weight - old_weight) > self.parameters.stability_threshold:
                delta = new_weight - old_weight
                synapse.update_weight(delta)
                
                # Update neuron connection
                if source_id in network.neurons:
                    network.neurons[source_id].adjust_weight(target_id, delta)
                
                affected_synapses.append(synapse_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        synapse_id=synapse_id,
                        rule_type="hebbian",
                        pre_value=old_weight,
                        post_value=new_weight,
                        delta=delta
                    )
                )
                
                synapse_changes.append((synapse_id, old_weight, new_weight, delta))
        
        # Create learning event if any synapses were affected
        if affected_synapses:
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / len(affected_synapses) if affected_synapses else 0.0,
                details={
                    "rule": "hebbian",
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
            return [event]
        
        return []
    
    def _apply_oja_rule(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply Oja's rule for normalized Hebbian learning
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events
        """
        affected_synapses = []
        synapse_changes = []
        total_change = 0.0
        
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id not in self.neuron_activations or target_id not in self.neuron_activations:
                continue
                
            pre_activation = self.neuron_activations[source_id]
            post_activation = self.neuron_activations[target_id]
            
            # Apply Oja's rule
            old_weight = synapse.weight
            new_weight = self.hebbian_learning.apply_oja_rule(
                pre_activation, 
                post_activation * self.parameters.modulation_factor, 
                old_weight
            )
            
            # Update the synapse weight if significant change
            if abs(new_weight - old_weight) > self.parameters.stability_threshold:
                delta = new_weight - old_weight
                synapse.update_weight(delta)
                
                # Update neuron connection
                if source_id in network.neurons:
                    network.neurons[source_id].adjust_weight(target_id, delta)
                
                affected_synapses.append(synapse_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        synapse_id=synapse_id,
                        rule_type="oja",
                        pre_value=old_weight,
                        post_value=new_weight,
                        delta=delta
                    )
                )
                
                synapse_changes.append((synapse_id, old_weight, new_weight, delta))
        
        # Create learning event if any synapses were affected
        if affected_synapses:
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / len(affected_synapses) if affected_synapses else 0.0,
                details={
                    "rule": "oja",
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
            return [event]
        
        return []
    
    def _apply_bcm_rule(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply BCM (Bienenstock-Cooper-Munro) rule
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events
        """
        affected_synapses = []
        synapse_changes = []
        total_change = 0.0
        
        # Calculate average post-synaptic activity for threshold
        avg_post_activity = 0.0
        active_neurons = 0
        
        for neuron_id, activation in self.neuron_activations.items():
            if activation > 0:
                avg_post_activity += activation
                active_neurons += 1
                
        if active_neurons > 0:
            avg_post_activity /= active_neurons
        else:
            avg_post_activity = 0.1  # Default threshold if no activity
        
        threshold = avg_post_activity  # BCM threshold is typically average activity
        
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id not in self.neuron_activations or target_id not in self.neuron_activations:
                continue
                
            pre_activation = self.neuron_activations[source_id]
            post_activation = self.neuron_activations[target_id]
            
            # Apply BCM rule
            old_weight = synapse.weight
            new_weight = self.hebbian_learning.apply_bcm_rule(
                pre_activation, 
                post_activation * self.parameters.modulation_factor, 
                old_weight,
                threshold
            )
            
            # Update the synapse weight if significant change
            if abs(new_weight - old_weight) > self.parameters.stability_threshold:
                delta = new_weight - old_weight
                synapse.update_weight(delta)
                
                # Update neuron connection
                if source_id in network.neurons:
                    network.neurons[source_id].adjust_weight(target_id, delta)
                
                affected_synapses.append(synapse_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        synapse_id=synapse_id,
                        rule_type="bcm",
                        pre_value=old_weight,
                        post_value=new_weight,
                        delta=delta
                    )
                )
                
                synapse_changes.append((synapse_id, old_weight, new_weight, delta))
        
        # Create learning event if any synapses were affected
        if affected_synapses:
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / len(affected_synapses) if affected_synapses else 0.0,
                details={
                    "rule": "bcm",
                    "threshold": threshold,
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
            return [event]
        
        return []
    
    def _apply_stdp_rule(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply Spike-Timing-Dependent Plasticity rule
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events
        """
        affected_synapses = []
        synapse_changes = []
        total_change = 0.0
        
        current_time = datetime.now()
        
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            # Skip if either neuron hasn't spiked
            if (source_id not in self.neuron_spike_times or 
                target_id not in self.neuron_spike_times or
                not self.neuron_spike_times[source_id] or 
                not self.neuron_spike_times[target_id]):
                continue
                
            # Get the most recent spike times
            pre_spikes = self.neuron_spike_times[source_id]
            post_spikes = self.neuron_spike_times[target_id]
            
            # Find closest spike pair
            min_time_diff = float('inf')
            pre_post_order = True  # True if pre before post, False if post before pre
            
            for pre_time in pre_spikes:
                for post_time in post_spikes:
                    time_diff_ms = (post_time - pre_time).total_seconds() * 1000  # Convert to ms
                    
                    if abs(time_diff_ms) < abs(min_time_diff):
                        min_time_diff = time_diff_ms
                        pre_post_order = time_diff_ms > 0  # pre before post
            
            # Skip if no valid spike pairs found
            if min_time_diff == float('inf'):
                continue
                
            # Apply STDP rule
            tau = self.parameters.time_window / 4  # Time constant for STDP curve
            
            if pre_post_order:  # Pre before post - LTP
                # A*exp(-Δt/τ) for LTP
                weight_change = self.parameters.learning_rate * np.exp(-abs(min_time_diff) / tau)
            else:  # Post before pre - LTD
                # -A*exp(-Δt/τ) for LTD
                weight_change = -self.parameters.learning_rate * np.exp(-abs(min_time_diff) / tau)
            
            # Apply modulation
            weight_change *= self.parameters.modulation_factor
            
            # Update synapse weight
            old_weight = synapse.weight
            new_weight = max(
                self.parameters.min_weight,
                min(self.parameters.max_weight, old_weight + weight_change)
            )
            
            # Update the synapse weight if significant change
            if abs(new_weight - old_weight) > self.parameters.stability_threshold:
                delta = new_weight - old_weight
                synapse.update_weight(delta)
                
                # Update neuron connection
                if source_id in network.neurons:
                    network.neurons[source_id].adjust_weight(target_id, delta)
                
                affected_synapses.append(synapse_id)
                total_change += abs(delta)
                
                # Record the learning application
                self.learning_applications.append(
                    LearningRuleApplication(
                        synapse_id=synapse_id,
                        rule_type="stdp",
                        pre_value=old_weight,
                        post_value=new_weight,
                        delta=delta
                    )
                )
                
                synapse_changes.append((synapse_id, old_weight, new_weight, delta))
        
        # Create learning event if any synapses were affected
        if affected_synapses:
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=affected_synapses,
                magnitude=total_change / len(affected_synapses) if affected_synapses else 0.0,
                details={
                    "rule": "stdp",
                    "time_window": self.parameters.time_window,
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
            return [event]
        
        return []
    
    def get_neuron_importance(self, neuron_id: str) -> float:
        """
        Calculate the importance score for a neuron
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        if neuron_id not in self.neuron_stats:
            return 0.5  # Default importance
            
        stats = self.neuron_stats[neuron_id]
        
        # Simple importance calculation based on activity level
        return min(1.0, stats.average_activation * 2.0)
    
    def get_synapse_importance(self, synapse_id: str) -> float:
        """
        Calculate the importance score for a synapse
        
        Args:
            synapse_id: ID of the synapse
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        if synapse_id not in self.synapse_stats:
            return 0.5  # Default importance
            
        stats = self.synapse_stats[synapse_id]
        
        # Simple importance calculation based on usage and strength
        return min(1.0, stats.average_transmission * 2.0)
    
    def set_learning_rule(self, rule: str) -> None:
        """
        Set the active learning rule
        
        Args:
            rule: Learning rule name ("hebbian", "oja", "bcm", "stdp")
        """
        if rule in ["hebbian", "oja", "bcm", "stdp"]:
            self.parameters.learning_rule = rule
            logger.info(f"Changed Hebbian learning rule to: {rule}")
        else:
            logger.warning(f"Unknown learning rule: {rule}")
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate
        
        Args:
            learning_rate: New learning rate (0.0 to 1.0)
        """
        self.parameters.learning_rate = max(0.0, min(1.0, learning_rate))
        self.learning_rate = self.parameters.learning_rate
        self.hebbian_learning.learning_rate = self.parameters.learning_rate
        logger.info(f"Changed Hebbian learning rate to: {learning_rate}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the learning engine
        
        Returns:
            Dictionary with engine state
        """
        return {
            "engine_id": self.engine_id,
            "engine_type": self.engine_type,
            "learning_rate": self.learning_rate,
            "is_active": self.is_active,
            "parameters": self.parameters.dict(),
            "neuron_stats_count": len(self.neuron_stats),
            "synapse_stats_count": len(self.synapse_stats),
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
            self.parameters = HebbianParameters(**state["parameters"])
            
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
            
        if "is_active" in state:
            self.is_active = state["is_active"]
            
        if "last_applied" in state:
            self.last_applied = state["last_applied"]
