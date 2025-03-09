"""
Neural Pruning Engine Module

This module implements neural pruning mechanisms that selectively remove 
weak or unused connections to optimize neural networks. Key features include:

- Weight-based pruning: Removes connections with weights below a threshold
- Usage-based pruning: Removes connections that are rarely used
- Importance-based pruning: Preserves connections deemed important for function
- Balanced pruning: Maintains network structure while removing redundancy

Neural pruning is essential for efficient cognitive function, mimicking 
the brain's process of eliminating unused connections while strengthening
important pathways.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import numpy as np
from datetime import datetime
import random

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.learning_engines.models import (
    LearningEngine, PruningParameters, LearningEvent, LearningRuleApplication,
    NeuronUsageStats, SynapseUsageStats
)
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class PruningEngine(LearningEngine):
    """
    Pruning engine for optimizing neural connections.
    """
    
    def __init__(
        self,
        parameters: Optional[PruningParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the pruning engine
        
        Args:
            parameters: Parameters for neural pruning
            event_bus: Event bus for sending learning events
        """
        super().__init__(
            engine_type="pruning",
            learning_rate=0.01  # Not directly used in pruning
        )
        
        self.parameters = parameters or PruningParameters()
        self.event_bus = event_bus
        
        # Track neuron and synapse usage statistics
        self.neuron_stats: Dict[str, NeuronUsageStats] = {}
        self.synapse_stats: Dict[str, SynapseUsageStats] = {}
        
        # Track pruned connections for possible recovery
        self.pruned_synapses: Dict[str, Dict[str, Any]] = {}
        
        # Count operations for scheduling
        self.operation_count = 0
        
        # Keep track of pruning history
        self.pruning_history: List[Dict[str, Any]] = []
        
        logger.info(f"Pruning engine initialized with strategy: {self.parameters.pruning_strategy}")
    
    def apply_learning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply neural pruning to a neural network
        
        Args:
            network: The neural network to apply pruning to
            
        Returns:
            List of learning events that occurred
        """
        if not self.is_active:
            return []
            
        # Update usage statistics
        self._update_usage_stats(network)
        
        # Increment operation count
        self.operation_count += 1
        
        # Check if it's time to perform pruning
        if self.operation_count % self.parameters.pruning_frequency != 0:
            return []
            
        # Select the appropriate pruning strategy
        if self.parameters.pruning_strategy == "weight":
            events = self._apply_weight_pruning(network)
        elif self.parameters.pruning_strategy == "usage":
            events = self._apply_usage_pruning(network)
        elif self.parameters.pruning_strategy == "importance":
            events = self._apply_importance_pruning(network)
        elif self.parameters.pruning_strategy == "combined":
            events = self._apply_combined_pruning(network)
        else:
            logger.warning(f"Unknown pruning strategy: {self.parameters.pruning_strategy}")
            events = []
            
        # Try to recover some pruned synapses
        recovery_events = self._attempt_synapse_recovery(network)
        events.extend(recovery_events)
            
        # Update last applied timestamp
        self.last_applied = datetime.now()
        
        # Broadcast learning events if event bus is available
        if self.event_bus:
            for event in events:
                message = Message(
                    sender="pruning_engine",
                    message_type="learning_event",
                    content=event.dict()
                )
                self.event_bus.publish(message)
        
        return events
    
    def _update_usage_stats(self, network: NeuralNetwork) -> None:
        """
        Update usage statistics for all neurons and synapses
        
        Args:
            network: The neural network
        """
        # Update neuron usage statistics
        for neuron_id, neuron in network.neurons.items():
            if neuron_id not in self.neuron_stats:
                self.neuron_stats[neuron_id] = NeuronUsageStats(neuron_id=neuron_id)
                
            stats = self.neuron_stats[neuron_id]
            
            # Update only if neuron is active
            if neuron.activation > 0:
                stats.activation_count += 1
                stats.total_activation += neuron.activation
                stats.average_activation = stats.total_activation / stats.activation_count
                stats.last_activated = datetime.now()
                
                # Calculate importance based on activity and connections
                incoming_connections = sum(1 for synapse in network.synapses.values() 
                                         if synapse.target_id == neuron_id)
                outgoing_connections = sum(1 for synapse in network.synapses.values() 
                                         if synapse.source_id == neuron_id)
                
                # Higher importance for neurons with more connections and higher activation
                connectivity_factor = min(1.0, (incoming_connections + outgoing_connections) / 20)
                activity_factor = min(1.0, stats.average_activation * 2)
                
                # Calculate importance (weighted average)
                stats.importance_score = 0.4 * activity_factor + 0.6 * connectivity_factor
        
        # Update synapse usage statistics
        for synapse_id, synapse in network.synapses.items():
            if synapse_id not in self.synapse_stats:
                self.synapse_stats[synapse_id] = SynapseUsageStats(synapse_id=synapse_id)
                
            stats = self.synapse_stats[synapse_id]
            
            # Get source neuron activation
            source_id = synapse.source_id
            source_activation = network.neurons[source_id].activation if source_id in network.neurons else 0.0
            
            # Update if there's transmission
            if source_activation > 0:
                transmission = source_activation * abs(synapse.weight)
                stats.transmission_count += 1
                stats.total_transmission += transmission
                stats.average_transmission = stats.total_transmission / stats.transmission_count
                stats.last_used = datetime.now()
                
                # Calculate importance based on usage and weight
                usage_factor = min(1.0, stats.transmission_count / 100)
                weight_factor = min(1.0, abs(synapse.weight) * 2)
                
                # Calculate importance (weighted average)
                stats.importance_score = 0.5 * usage_factor + 0.5 * weight_factor
    
    def _apply_weight_pruning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply weight-based pruning to remove weak connections
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        pruned_synapses = []
        total_synapses = len(network.synapses)
        max_to_prune = int(total_synapses * self.parameters.max_prune_percent)
        
        # Sort synapses by absolute weight
        sorted_synapses = sorted(
            network.synapses.items(),
            key=lambda x: abs(x[1].weight)
        )
        
        # Collect synapses to prune based on weight threshold
        candidates_to_prune = []
        
        for synapse_id, synapse in sorted_synapses:
            # Skip if this is an I/O connection and preserve_io_neurons is True
            if self.parameters.preserve_io_neurons:
                source_id = synapse.source_id
                target_id = synapse.target_id
                
                if (source_id in network.input_neurons or 
                    target_id in network.input_neurons or
                    source_id in network.output_neurons or
                    target_id in network.output_neurons):
                    continue
            
            # Check if weight is below threshold
            if abs(synapse.weight) < self.parameters.weight_threshold:
                candidates_to_prune.append((synapse_id, synapse))
        
        # Limit the number of synapses to prune
        synapses_to_prune = candidates_to_prune[:max_to_prune]
        
        # Prune the selected synapses
        for synapse_id, synapse in synapses_to_prune:
            # Store synapse info for possible recovery
            self._store_pruned_synapse(network, synapse_id, synapse)
            
            # Remove from network
            del network.synapses[synapse_id]
            
            # Remove connection from source neuron
            source_id = synapse.source_id
            target_id = synapse.target_id
            if source_id in network.neurons and target_id in network.neurons.get(source_id, {}).connections:
                network.neurons[source_id].connections.pop(target_id, None)
            
            pruned_synapses.append(synapse_id)
            
            logger.debug(f"Pruned synapse {synapse_id} with weight {synapse.weight}")
            
        # Create a learning event if any synapses were pruned
        if pruned_synapses:
            pruning_event = {
                "strategy": "weight",
                "threshold": self.parameters.weight_threshold,
                "total_synapses": total_synapses,
                "pruned_count": len(pruned_synapses),
                "pruning_percent": len(pruned_synapses) / total_synapses * 100 if total_synapses > 0 else 0
            }
            
            self.pruning_history.append({
                "timestamp": datetime.now(),
                **pruning_event
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=pruned_synapses,
                magnitude=len(pruned_synapses) / total_synapses if total_synapses > 0 else 0,
                details=pruning_event
            )
            
            return [event]
        
        return []
    
    def _apply_usage_pruning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply usage-based pruning to remove rarely used connections
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        pruned_synapses = []
        total_synapses = len(network.synapses)
        max_to_prune = int(total_synapses * self.parameters.max_prune_percent)
        
        # Count total operations to normalize usage
        operation_norm = max(1, self.operation_count)
        
        # Calculate usage ratio for each synapse
        synapse_usage = {}
        for synapse_id, synapse in network.synapses.items():
            if synapse_id in self.synapse_stats:
                usage_stat = self.synapse_stats[synapse_id]
                # Usage ratio = transmission count / total operations
                usage_ratio = usage_stat.transmission_count / operation_norm
                synapse_usage[synapse_id] = usage_ratio
            else:
                # No usage data, assume zero usage
                synapse_usage[synapse_id] = 0.0
        
        # Sort synapses by usage
        sorted_synapses = sorted(
            network.synapses.items(),
            key=lambda x: synapse_usage.get(x[0], 0.0)
        )
        
        # Collect synapses to prune based on usage threshold
        candidates_to_prune = []
        
        for synapse_id, synapse in sorted_synapses:
            # Skip if this is an I/O connection and preserve_io_neurons is True
            if self.parameters.preserve_io_neurons:
                source_id = synapse.source_id
                target_id = synapse.target_id
                
                if (source_id in network.input_neurons or 
                    target_id in network.input_neurons or
                    source_id in network.output_neurons or
                    target_id in network.output_neurons):
                    continue
            
            # Check if usage is below threshold
            if synapse_usage.get(synapse_id, 0.0) < self.parameters.usage_threshold:
                candidates_to_prune.append((synapse_id, synapse))
        
        # Limit the number of synapses to prune
        synapses_to_prune = candidates_to_prune[:max_to_prune]
        
        # Prune the selected synapses
        for synapse_id, synapse in synapses_to_prune:
            # Store synapse info for possible recovery
            self._store_pruned_synapse(network, synapse_id, synapse)
            
            # Remove from network
            del network.synapses[synapse_id]
            
            # Remove connection from source neuron
            source_id = synapse.source_id
            target_id = synapse.target_id
            if source_id in network.neurons and target_id in network.neurons.get(source_id, {}).connections:
                network.neurons[source_id].connections.pop(target_id, None)
            
            pruned_synapses.append(synapse_id)
            
            logger.debug(f"Pruned synapse {synapse_id} with usage {synapse_usage.get(synapse_id, 0.0)}")
            
        # Create a learning event if any synapses were pruned
        if pruned_synapses:
            pruning_event = {
                "strategy": "usage",
                "threshold": self.parameters.usage_threshold,
                "total_synapses": total_synapses,
                "pruned_count": len(pruned_synapses),
                "pruning_percent": len(pruned_synapses) / total_synapses * 100 if total_synapses > 0 else 0
            }
            
            self.pruning_history.append({
                "timestamp": datetime.now(),
                **pruning_event
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=pruned_synapses,
                magnitude=len(pruned_synapses) / total_synapses if total_synapses > 0 else 0,
                details=pruning_event
            )
            
            return [event]
        
        return []
    
    def _apply_importance_pruning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply importance-based pruning to preserve functionally important connections
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        pruned_synapses = []
        total_synapses = len(network.synapses)
        max_to_prune = int(total_synapses * self.parameters.max_prune_percent)
        
        # Calculate importance for each synapse
        synapse_importance = {}
        for synapse_id, synapse in network.synapses.items():
            if synapse_id in self.synapse_stats:
                importance = self.synapse_stats[synapse_id].importance_score
                synapse_importance[synapse_id] = importance
            else:
                # No importance data, use weight as proxy
                importance = abs(synapse.weight)
                synapse_importance[synapse_id] = importance
        
        # Sort synapses by importance
        sorted_synapses = sorted(
            network.synapses.items(),
            key=lambda x: synapse_importance.get(x[0], 0.0)
        )
        
        # Collect synapses to prune based on importance threshold
        candidates_to_prune = []
        
        for synapse_id, synapse in sorted_synapses:
            # Skip if this is an I/O connection and preserve_io_neurons is True
            if self.parameters.preserve_io_neurons:
                source_id = synapse.source_id
                target_id = synapse.target_id
                
                if (source_id in network.input_neurons or 
                    target_id in network.input_neurons or
                    source_id in network.output_neurons or
                    target_id in network.output_neurons):
                    continue
            
            # Check if importance is below threshold
            if synapse_importance.get(synapse_id, 0.0) < self.parameters.importance_threshold:
                candidates_to_prune.append((synapse_id, synapse))
        
        # Limit the number of synapses to prune
        synapses_to_prune = candidates_to_prune[:max_to_prune]
        
        # Prune the selected synapses
        for synapse_id, synapse in synapses_to_prune:
            # Store synapse info for possible recovery
            self._store_pruned_synapse(network, synapse_id, synapse)
            
            # Remove from network
            del network.synapses[synapse_id]
            
            # Remove connection from source neuron
            source_id = synapse.source_id
            target_id = synapse.target_id
            if source_id in network.neurons and target_id in network.neurons.get(source_id, {}).connections:
                network.neurons[source_id].connections.pop(target_id, None)
            
            pruned_synapses.append(synapse_id)
            
            logger.debug(f"Pruned synapse {synapse_id} with importance {synapse_importance.get(synapse_id, 0.0)}")
            
        # Create a learning event if any synapses were pruned
        if pruned_synapses:
            pruning_event = {
                "strategy": "importance",
                "threshold": self.parameters.importance_threshold,
                "total_synapses": total_synapses,
                "pruned_count": len(pruned_synapses),
                "pruning_percent": len(pruned_synapses) / total_synapses * 100 if total_synapses > 0 else 0
            }
            
            self.pruning_history.append({
                "timestamp": datetime.now(),
                **pruning_event
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=pruned_synapses,
                magnitude=len(pruned_synapses) / total_synapses if total_synapses > 0 else 0,
                details=pruning_event
            )
            
            return [event]
        
        return []
    
    def _apply_combined_pruning(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply combined pruning strategy using multiple criteria
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        pruned_synapses = []
        total_synapses = len(network.synapses)
        max_to_prune = int(total_synapses * self.parameters.max_prune_percent)
        
        # Calculate combined score for each synapse
        synapse_scores = {}
        for synapse_id, synapse in network.synapses.items():
            # Weight score (lower weight = higher score)
            weight_score = 1.0 - min(1.0, abs(synapse.weight) / self.parameters.weight_threshold)
            
            # Usage score
            if synapse_id in self.synapse_stats:
                usage_stat = self.synapse_stats[synapse_id]
                usage_score = 1.0 - min(1.0, usage_stat.transmission_count / max(1, self.operation_count) / self.parameters.usage_threshold)
            else:
                usage_score = 1.0  # No usage data = high score (should prune)
            
            # Importance score
            if synapse_id in self.synapse_stats:
                importance = self.synapse_stats[synapse_id].importance_score
                importance_score = 1.0 - importance  # Lower importance = higher score
            else:
                importance_score = 0.5  # Neutral score
            
            # Combined score (higher = more likely to prune)
            combined_score = 0.4 * weight_score + 0.3 * usage_score + 0.3 * importance_score
            synapse_scores[synapse_id] = combined_score
        
        # Sort synapses by combined score (descending)
        sorted_synapses = sorted(
            network.synapses.items(),
            key=lambda x: synapse_scores.get(x[0], 0.0),
            reverse=True
        )
        
        # Collect synapses to prune based on combined threshold
        candidates_to_prune = []
        
        for synapse_id, synapse in sorted_synapses:
            # Skip if this is an I/O connection and preserve_io_neurons is True
            if self.parameters.preserve_io_neurons:
                source_id = synapse.source_id
                target_id = synapse.target_id
                
                if (source_id in network.input_neurons or 
                    target_id in network.input_neurons or
                    source_id in network.output_neurons or
                    target_id in network.output_neurons):
                    continue
            
            # Check if combined score is above threshold (higher = more likely to prune)
            if synapse_scores.get(synapse_id, 0.0) > 0.7:  # Threshold for combined score
                candidates_to_prune.append((synapse_id, synapse))
        
        # Limit the number of synapses to prune
        synapses_to_prune = candidates_to_prune[:max_to_prune]
        
        # Prune the selected synapses
        for synapse_id, synapse in synapses_to_prune:
            # Store synapse info for possible recovery
            self._store_pruned_synapse(network, synapse_id, synapse)
            
            # Remove from network
            del network.synapses[synapse_id]
            
            # Remove connection from source neuron
            source_id = synapse.source_id
            target_id = synapse.target_id
            if source_id in network.neurons and target_id in network.neurons.get(source_id, {}).connections:
                network.neurons[source_id].connections.pop(target_id, None)
            
            pruned_synapses.append(synapse_id)
            
            logger.debug(f"Pruned synapse {synapse_id} with combined score {synapse_scores.get(synapse_id, 0.0)}")
            
        # Create a learning event if any synapses were pruned
        if pruned_synapses:
            pruning_event = {
                "strategy": "combined",
                "total_synapses": total_synapses,
                "pruned_count": len(pruned_synapses),
                "pruning_percent": len(pruned_synapses) / total_synapses * 100 if total_synapses > 0 else 0,
                "weight_threshold": self.parameters.weight_threshold,
                "usage_threshold": self.parameters.usage_threshold,
                "importance_threshold": self.parameters.importance_threshold
            }
            
            self.pruning_history.append({
                "timestamp": datetime.now(),
                **pruning_event
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=pruned_synapses,
                magnitude=len(pruned_synapses) / total_synapses if total_synapses > 0 else 0,
                details=pruning_event
            )
            
            return [event]
        
        return []
    
    def _store_pruned_synapse(self, network: NeuralNetwork, synapse_id: str, synapse: Synapse) -> None:
        """
        Store information about a pruned synapse for possible recovery
        
        Args:
            network: The neural network
            synapse_id: ID of the synapse being pruned
            synapse: The synapse being pruned
        """
        self.pruned_synapses[synapse_id] = {
            "source_id": synapse.source_id,
            "target_id": synapse.target_id,
            "weight": synapse.weight,
            "pruned_at": datetime.now(),
            "operation_count": self.operation_count
        }
        
        # Limit the number of stored pruned synapses to prevent memory issues
        if len(self.pruned_synapses) > 1000:
            # Remove oldest pruned synapses
            sorted_pruned = sorted(
                self.pruned_synapses.items(),
                key=lambda x: x[1]["pruned_at"]
            )
            
            # Keep only the most recent 1000
            self.pruned_synapses = dict(sorted_pruned[-1000:])
    
    def _attempt_synapse_recovery(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Attempt to recover some previously pruned synapses
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        recovered_synapses = []
        
        # Only attempt recovery if there are pruned synapses
        if not self.pruned_synapses:
            return []
            
        # Calculate how many synapses to try to recover
        total_pruned = len(self.pruned_synapses)
        recovery_attempts = min(10, total_pruned)  # Limit to 10 attempts per cycle
        
        # Select random pruned synapses to consider for recovery
        recovery_candidates = random.sample(list(self.pruned_synapses.items()), recovery_attempts)
        
        for synapse_id, synapse_data in recovery_candidates:
            # Only recover with probability determined by recovery_probability
            if random.random() > self.parameters.recovery_probability:
                continue
                
            source_id = synapse_data["source_id"]
            target_id = synapse_data["target_id"]
            weight = synapse_data["weight"]
            
            # Check if the neurons still exist
            if source_id not in network.neurons or target_id not in network.neurons:
                # Neurons no longer exist, can't recover
                del self.pruned_synapses[synapse_id]
                continue
                
            # Check if the connection already exists
            connection_exists = False
            for existing_synapse in network.synapses.values():
                if existing_synapse.source_id == source_id and existing_synapse.target_id == target_id:
                    connection_exists = True
                    break
                    
            if connection_exists:
                # Connection already exists, no need to recover
                del self.pruned_synapses[synapse_id]
                continue
                
            # Create a new synapse with the old connection data
            try:
                new_synapse = network.create_synapse(source_id, target_id, weight)
                
                recovered_synapses.append(new_synapse.synapse_id)
                
                # Remove from pruned synapses
                del self.pruned_synapses[synapse_id]
                
                logger.debug(f"Recovered synapse from {source_id} to {target_id} with weight {weight}")
            except Exception as e:
                logger.error(f"Failed to recover synapse: {e}")
                continue
        
        # Create a learning event if any synapses were recovered
        if recovered_synapses:
            recovery_event = {
                "action": "synapse_recovery",
                "recovered_count": len(recovered_synapses),
                "recovery_probability": self.parameters.recovery_probability
            }
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=recovered_synapses,
                magnitude=len(recovered_synapses) / max(1, len(network.synapses)),
                details=recovery_event
            )
            
            return [event]
        
        return []
    
    def get_neuron_importance(self, neuron_id: str) -> float:
        """
        Get the importance score for a neuron
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        if neuron_id not in self.neuron_stats:
            return 0.5  # Default importance
            
        return self.neuron_stats[neuron_id].importance_score
    
    def get_synapse_importance(self, synapse_id: str) -> float:
        """
        Get the importance score for a synapse
        
        Args:
            synapse_id: ID of the synapse
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        if synapse_id not in self.synapse_stats:
            return 0.5  # Default importance
            
        return self.synapse_stats[synapse_id].importance_score
    
    def set_pruning_strategy(self, strategy: str) -> None:
        """
        Set the pruning strategy
        
        Args:
            strategy: Pruning strategy name
        """
        if strategy in ["weight", "usage", "importance", "combined"]:
            self.parameters.pruning_strategy = strategy
            logger.info(f"Changed pruning strategy to: {strategy}")
        else:
            logger.warning(f"Unknown pruning strategy: {strategy}")
    
    def set_weight_threshold(self, threshold: float) -> None:
        """
        Set the weight threshold for pruning
        
        Args:
            threshold: New weight threshold
        """
        self.parameters.weight_threshold = max(0.0, threshold)
        logger.info(f"Changed weight threshold to: {threshold}")
    
    def set_usage_threshold(self, threshold: float) -> None:
        """
        Set the usage threshold for pruning
        
        Args:
            threshold: New usage threshold
        """
        self.parameters.usage_threshold = max(0.0, threshold)
        logger.info(f"Changed usage threshold to: {threshold}")
    
    def set_importance_threshold(self, threshold: float) -> None:
        """
        Set the importance threshold for pruning
        
        Args:
            threshold: New importance threshold
        """
        self.parameters.importance_threshold = max(0.0, threshold)
        logger.info(f"Changed importance threshold to: {threshold}")
    
    def set_max_prune_percent(self, percent: float) -> None:
        """
        Set the maximum percentage of connections to prune in one operation
        
        Args:
            percent: Maximum percentage (0.0 to 1.0)
        """
        self.parameters.max_prune_percent = max(0.0, min(1.0, percent))
        logger.info(f"Changed max prune percent to: {percent}")
    
    def set_pruning_frequency(self, frequency: int) -> None:
        """
        Set how often to perform pruning operations
        
        Args:
            frequency: Number of operations between pruning
        """
        self.parameters.pruning_frequency = max(1, frequency)
        logger.info(f"Changed pruning frequency to: {frequency}")
    
    def set_recovery_probability(self, probability: float) -> None:
        """
        Set the probability of recovering pruned connections
        
        Args:
            probability: Recovery probability (0.0 to 1.0)
        """
        self.parameters.recovery_probability = max(0.0, min(1.0, probability))
        logger.info(f"Changed recovery probability to: {probability}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the pruning engine
        
        Returns:
            Dictionary with engine state
        """
        return {
            "engine_id": self.engine_id,
            "engine_type": self.engine_type,
            "is_active": self.is_active,
            "parameters": self.parameters.dict(),
            "operation_count": self.operation_count,
            "pruning_history_count": len(self.pruning_history),
            "neuron_stats_count": len(self.neuron_stats),
            "synapse_stats_count": len(self.synapse_stats),
            "pruned_synapses_count": len(self.pruned_synapses),
            "last_applied": self.last_applied
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: Dictionary with engine state
        """
        if "parameters" in state:
            self.parameters = PruningParameters(**state["parameters"])
            
        if "is_active" in state:
            self.is_active = state["is_active"]
            
        if "operation_count" in state:
            self.operation_count = state["operation_count"]
            
        if "last_applied" in state:
            self.last_applied = state["last_applied"]
