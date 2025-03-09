"""
Memory Consolidation Engine Module

This module implements memory consolidation mechanisms, which are critical for
transforming temporary neural patterns into stable, long-term memories. Key features:

- Pattern stability: Identifies and stabilizes important neural activation patterns
- Synaptic tagging: Marks synapses for later consolidation based on activity
- Sleep simulation: Enhanced consolidation during "sleep" phases
- Reactivation: Selective strengthening of important neural patterns
- Rehearsal: Reinforces memories based on recency, frequency, and importance

Memory consolidation is essential for forming persistent memories and is inspired
by biological processes like long-term potentiation and systems consolidation.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import numpy as np
from datetime import datetime, timedelta
import random

from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.neuron import Neuron
from lmm_project.neural_substrate.synapse import Synapse
from lmm_project.neural_substrate.neural_cluster import NeuralCluster
from lmm_project.learning_engines.models import (
    LearningEngine, ConsolidationParameters, LearningEvent, LearningRuleApplication,
    SynapticTaggingParameters, SynapseUsageStats
)
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class ConsolidationEngine(LearningEngine):
    """
    Consolidation engine for stabilizing memories.
    """
    
    def __init__(
        self,
        parameters: Optional[ConsolidationParameters] = None,
        tagging_parameters: Optional[SynapticTaggingParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the consolidation engine
        
        Args:
            parameters: Parameters for memory consolidation
            tagging_parameters: Parameters for synaptic tagging
            event_bus: Event bus for sending consolidation events
        """
        super().__init__(
            engine_type="consolidation",
            learning_rate=parameters.stabilization_rate if parameters else 0.1
        )
        
        self.parameters = parameters or ConsolidationParameters()
        self.tagging_parameters = tagging_parameters or SynapticTaggingParameters()
        self.event_bus = event_bus
        
        # Track activation patterns
        self.activation_patterns: Dict[str, Dict[str, float]] = {}
        self.pattern_recency: Dict[str, datetime] = {}
        self.pattern_frequency: Dict[str, int] = {}
        self.pattern_importance: Dict[str, float] = {}
        
        # Track synapse tags for consolidation
        self.tagged_synapses: Dict[str, Dict[str, Any]] = {}
        
        # Track pattern reactivations
        self.reactivation_history: List[Dict[str, Any]] = []
        
        # Track consolidation processes
        self.consolidation_history: List[Dict[str, Any]] = []
        
        # Sleep mode for enhanced consolidation
        self.sleep_mode = False
        
        # Count operations for scheduling
        self.operation_count = 0
        
        logger.info("Consolidation engine initialized")
    
    def apply_learning(
        self, 
        network: NeuralNetwork,
        sleep_mode: bool = False
    ) -> List[LearningEvent]:
        """
        Apply memory consolidation to a neural network
        
        Args:
            network: The neural network to apply consolidation to
            sleep_mode: Whether the system is in "sleep" mode for enhanced consolidation
            
        Returns:
            List of learning events that occurred
        """
        if not self.is_active:
            return []
            
        self.sleep_mode = sleep_mode
        
        # Increment operation count
        self.operation_count += 1
        
        # Capture current activation pattern
        self._capture_activation_pattern(network)
        
        # Update synaptic tags
        self._update_synaptic_tags(network)
        
        # Check if it's time to perform consolidation
        if self.operation_count % self.parameters.consolidation_frequency != 0 and not sleep_mode:
            return []
            
        # Apply consolidation process
        consolidation_events = self._apply_consolidation(network)
        
        # If in sleep mode, also perform pattern reactivation
        if sleep_mode:
            reactivation_events = self._apply_pattern_reactivation(network)
            consolidation_events.extend(reactivation_events)
            
        # Update last applied timestamp
        self.last_applied = datetime.now()
        
        # Broadcast learning events if event bus is available
        if self.event_bus:
            for event in consolidation_events:
                message = Message(
                    sender="consolidation_engine",
                    message_type="learning_event",
                    content=event.dict()
                )
                self.event_bus.publish(message)
        
        return consolidation_events
    
    def _capture_activation_pattern(self, network: NeuralNetwork) -> None:
        """
        Capture the current activation pattern of the network
        
        Args:
            network: The neural network
        """
        # Generate a pattern ID based on time
        pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Capture activations of all neurons
        activations = {}
        for neuron_id, neuron in network.neurons.items():
            if neuron.activation > 0:
                activations[neuron_id] = neuron.activation
        
        # Only store patterns with significant activation
        if len(activations) >= 3:  # At least 3 active neurons
            # Store the pattern
            self.activation_patterns[pattern_id] = activations
            self.pattern_recency[pattern_id] = datetime.now()
            self.pattern_frequency[pattern_id] = 1
            
            # Calculate pattern importance based on activation strength
            avg_activation = sum(activations.values()) / len(activations)
            self.pattern_importance[pattern_id] = avg_activation
            
            # Limit the number of stored patterns to prevent memory issues
            if len(self.activation_patterns) > 100:
                # Remove oldest patterns
                oldest_pattern = min(self.pattern_recency.items(), key=lambda x: x[1])[0]
                del self.activation_patterns[oldest_pattern]
                del self.pattern_recency[oldest_pattern]
                del self.pattern_frequency[oldest_pattern]
                del self.pattern_importance[oldest_pattern]
        
        # Check for pattern similarity with existing patterns
        for existing_id, existing_pattern in list(self.activation_patterns.items()):
            if existing_id != pattern_id and existing_id in self.activation_patterns:
                similarity = self._calculate_pattern_similarity(activations, existing_pattern)
                
                # If similar, update existing pattern instead of creating new one
                if similarity > 0.7:  # High similarity threshold
                    # Update recency
                    self.pattern_recency[existing_id] = datetime.now()
                    
                    # Increment frequency
                    self.pattern_frequency[existing_id] += 1
                    
                    # Update importance based on frequency and recency
                    frequency_factor = min(1.0, self.pattern_frequency[existing_id] / 10)
                    self.pattern_importance[existing_id] = 0.7 * self.pattern_importance[existing_id] + 0.3 * frequency_factor
                    
                    # Remove the new pattern since it's similar to an existing one
                    if pattern_id in self.activation_patterns:
                        del self.activation_patterns[pattern_id]
                        del self.pattern_recency[pattern_id]
                        del self.pattern_frequency[pattern_id]
                        del self.pattern_importance[pattern_id]
                    
                    break
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, float], pattern2: Dict[str, float]) -> float:
        """
        Calculate similarity between two activation patterns
        
        Args:
            pattern1: First activation pattern
            pattern2: Second activation pattern
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Find common neurons
        common_neurons = set(pattern1.keys()) & set(pattern2.keys())
        all_neurons = set(pattern1.keys()) | set(pattern2.keys())
        
        if not all_neurons:
            return 0.0
            
        # Calculate Jaccard similarity for neuron overlap
        jaccard_similarity = len(common_neurons) / len(all_neurons)
        
        # Calculate activation similarity for common neurons
        activation_similarity = 0.0
        if common_neurons:
            activation_diffs = []
            for neuron_id in common_neurons:
                diff = 1.0 - abs(pattern1[neuron_id] - pattern2[neuron_id])
                activation_diffs.append(diff)
            
            activation_similarity = sum(activation_diffs) / len(activation_diffs)
        
        # Combine similarities (weighted average)
        return 0.7 * jaccard_similarity + 0.3 * activation_similarity
    
    def _update_synaptic_tags(self, network: NeuralNetwork) -> None:
        """
        Update synaptic tags based on neuronal activity
        
        Args:
            network: The neural network
        """
        current_time = datetime.now()
        
        # Decay and remove expired tags
        for synapse_id in list(self.tagged_synapses.keys()):
            tag_data = self.tagged_synapses[synapse_id]
            
            # Check if tag has expired
            if current_time - tag_data["tagged_at"] > timedelta(seconds=self.tagging_parameters.tag_duration):
                del self.tagged_synapses[synapse_id]
                continue
                
            # Decay tag strength
            decay_factor = 0.99  # Slow decay
            tag_data["tag_strength"] *= decay_factor
            
            # Remove if tag strength is too low
            if tag_data["tag_strength"] < 0.1:
                del self.tagged_synapses[synapse_id]
        
        # Tag new synapses based on pre/post-synaptic activity
        for synapse_id, synapse in network.synapses.items():
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            if source_id not in network.neurons or target_id not in network.neurons:
                continue
                
            source_neuron = network.neurons[source_id]
            target_neuron = network.neurons[target_id]
            
            # Calculate co-activation
            coactivation = source_neuron.activation * target_neuron.activation
            
            # Tag synapse if co-activation is above threshold
            if coactivation > self.tagging_parameters.tag_threshold:
                # If already tagged, update tag strength
                if synapse_id in self.tagged_synapses:
                    # Increase tag strength (with ceiling)
                    new_strength = min(
                        1.0, 
                        self.tagged_synapses[synapse_id]["tag_strength"] + 
                        coactivation * self.tagging_parameters.tag_strength_factor
                    )
                    self.tagged_synapses[synapse_id]["tag_strength"] = new_strength
                    self.tagged_synapses[synapse_id]["tagged_at"] = current_time
                else:
                    # Create new tag
                    self.tagged_synapses[synapse_id] = {
                        "synapse_id": synapse_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "initial_weight": synapse.weight,
                        "tag_strength": coactivation * self.tagging_parameters.tag_strength_factor,
                        "tagged_at": current_time,
                        "consolidation_count": 0
                    }
    
    def _apply_consolidation(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply consolidation to tagged synapses
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        consolidated_synapses = []
        synapse_changes = []
        total_change = 0.0
        
        # Calculate consolidation multiplier for sleep mode
        consolidation_multiplier = (
            self.parameters.sleep_consolidation_boost if self.sleep_mode else 1.0
        )
        
        # Identify important patterns for consolidation
        important_patterns = self._select_patterns_for_consolidation()
        
        # Apply consolidation to tagged synapses
        for synapse_id, tag_data in list(self.tagged_synapses.items()):
            # Skip if synapse no longer exists
            if synapse_id not in network.synapses:
                del self.tagged_synapses[synapse_id]
                continue
                
            synapse = network.synapses[synapse_id]
            source_id = synapse.source_id
            target_id = synapse.target_id
            
            # Determine if this synapse is part of important patterns
            pattern_importance = 0.0
            for pattern_id, activations in important_patterns:
                if source_id in activations and target_id in activations:
                    # Synapse connects neurons in important pattern
                    pattern_importance = max(pattern_importance, self.pattern_importance[pattern_id])
            
            # Calculate consolidation strength based on tag strength and pattern importance
            tag_strength = tag_data["tag_strength"]
            consolidation_strength = (
                self.parameters.stabilization_rate * 
                tag_strength * 
                (0.5 + 0.5 * pattern_importance) *
                consolidation_multiplier
            )
            
            # Apply consolidation if strength is significant
            if consolidation_strength > 0.01:
                # Get current weight
                old_weight = synapse.weight
                
                # Calculate weight change (strengthen in direction of existing weight)
                if abs(old_weight) < 0.01:
                    # For very weak connections, choose a direction based on tag
                    weight_direction = 1.0 if tag_strength > 0.5 else -1.0
                    weight_delta = consolidation_strength * weight_direction
                else:
                    # Strengthen in current direction
                    weight_direction = 1.0 if old_weight > 0 else -1.0
                    weight_delta = consolidation_strength * weight_direction * abs(old_weight)
                
                # Apply weight change
                new_weight = old_weight + weight_delta
                
                # Apply the change to the synapse
                synapse.update_weight(weight_delta)
                
                # Update neuron connection
                if source_id in network.neurons:
                    network.neurons[source_id].adjust_weight(target_id, weight_delta)
                
                # Update tag data
                tag_data["consolidation_count"] += 1
                
                # Record the change
                consolidated_synapses.append(synapse_id)
                total_change += abs(weight_delta)
                
                # Record synapse change details
                synapse_changes.append((synapse_id, old_weight, new_weight, weight_delta))
                
                logger.debug(f"Consolidated synapse {synapse_id}: {old_weight:.4f} -> {new_weight:.4f}")
        
        # Create a learning event if any synapses were consolidated
        if consolidated_synapses:
            consolidation_event = {
                "action": "consolidation",
                "consolidated_count": len(consolidated_synapses),
                "sleep_mode": self.sleep_mode,
                "consolidation_multiplier": consolidation_multiplier,
                "important_patterns": len(important_patterns),
                "synapse_changes": [
                    {
                        "synapse_id": s_id,
                        "old_weight": old,
                        "new_weight": new,
                        "delta": delta
                    } for s_id, old, new, delta in synapse_changes[:10]  # Include first 10 changes
                ]
            }
            
            self.consolidation_history.append({
                "timestamp": datetime.now(),
                "consolidated_count": len(consolidated_synapses),
                "sleep_mode": self.sleep_mode,
                "total_change": total_change
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                synapses_affected=consolidated_synapses,
                magnitude=total_change / len(consolidated_synapses) if consolidated_synapses else 0.0,
                details=consolidation_event
            )
            
            return [event]
        
        return []
    
    def _select_patterns_for_consolidation(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Select activation patterns for consolidation based on importance
        
        Returns:
            List of (pattern_id, activations) tuples
        """
        # Calculate consolidated importance score for each pattern
        pattern_scores = {}
        current_time = datetime.now()
        
        for pattern_id in self.activation_patterns:
            # Get base importance
            importance = self.pattern_importance.get(pattern_id, 0.5)
            
            # Factor in recency
            recency = self.pattern_recency.get(pattern_id, current_time)
            time_since_activation = (current_time - recency).total_seconds()
            
            # Time decay function (exponential decay)
            recency_factor = np.exp(-time_since_activation / (3600 * 24))  # 24 hour half-life
            
            # Factor in frequency
            frequency = self.pattern_frequency.get(pattern_id, 1)
            frequency_factor = min(1.0, frequency / 10)  # Cap at 10 occurrences
            
            # Calculate consolidated score
            score = (
                importance * self.parameters.importance_factor +
                recency_factor * self.parameters.recency_factor +
                frequency_factor * 0.1  # Small contribution from frequency
            )
            
            pattern_scores[pattern_id] = score
        
        # Sort patterns by score (descending)
        sorted_patterns = sorted(
            [(p_id, self.activation_patterns[p_id]) for p_id in pattern_scores],
            key=lambda x: pattern_scores.get(x[0], 0.0),
            reverse=True
        )
        
        # Select patterns above consolidation threshold
        selected_patterns = []
        
        for pattern_id, activations in sorted_patterns:
            score = pattern_scores.get(pattern_id, 0.0)
            if score >= self.parameters.consolidation_threshold:
                selected_patterns.append((pattern_id, activations))
                
        return selected_patterns[:10]  # Limit to top 10 patterns
    
    def _apply_pattern_reactivation(self, network: NeuralNetwork) -> List[LearningEvent]:
        """
        Apply pattern reactivation during sleep mode
        
        Args:
            network: The neural network
            
        Returns:
            List of learning events that occurred
        """
        if not self.sleep_mode:
            return []
            
        reactivated_neurons = []
        neuron_activations = []
        total_activation = 0.0
        
        # Select patterns for reactivation
        patterns_to_reactivate = self._select_patterns_for_consolidation()
        
        if not patterns_to_reactivate:
            return []
            
        # Randomly select one pattern to reactivate
        pattern_id, activations = random.choice(patterns_to_reactivate)
        
        # Apply reactivation
        for neuron_id, activation_level in activations.items():
            if neuron_id not in network.neurons:
                continue
                
            # Scale activation by reactivation strength
            reactivation_level = activation_level * self.parameters.reactivation_strength
            
            # Apply activation
            network.neurons[neuron_id].activation = reactivation_level
            
            # Record activation
            reactivated_neurons.append(neuron_id)
            total_activation += reactivation_level
            
            neuron_activations.append((neuron_id, reactivation_level))
            
        # Create a learning event for the reactivation
        if reactivated_neurons:
            reactivation_event = {
                "action": "pattern_reactivation",
                "pattern_id": pattern_id,
                "reactivated_count": len(reactivated_neurons),
                "reactivation_strength": self.parameters.reactivation_strength,
                "neuron_activations": [
                    {
                        "neuron_id": n_id,
                        "activation": activation
                    } for n_id, activation in neuron_activations[:10]  # Include first 10 activations
                ]
            }
            
            self.reactivation_history.append({
                "timestamp": datetime.now(),
                "pattern_id": pattern_id,
                "reactivated_count": len(reactivated_neurons)
            })
            
            event = LearningEvent(
                engine_id=self.engine_id,
                engine_type=self.engine_type,
                target_module="neural_substrate",
                target_network_id=network.network_id,
                neurons_affected=reactivated_neurons,
                magnitude=total_activation / len(reactivated_neurons) if reactivated_neurons else 0.0,
                details=reactivation_event
            )
            
            return [event]
        
        return []
    
    def enter_sleep_mode(self) -> None:
        """Enter sleep mode for enhanced consolidation"""
        self.sleep_mode = True
        logger.info("Entered sleep mode for enhanced consolidation")
    
    def exit_sleep_mode(self) -> None:
        """Exit sleep mode"""
        self.sleep_mode = False
        logger.info("Exited sleep mode")
    
    def is_in_sleep_mode(self) -> bool:
        """Check if the system is in sleep mode"""
        return self.sleep_mode
    
    def set_consolidation_threshold(self, threshold: float) -> None:
        """
        Set the threshold for pattern consolidation
        
        Args:
            threshold: New consolidation threshold (0.0 to 1.0)
        """
        self.parameters.consolidation_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Changed consolidation threshold to: {threshold}")
    
    def set_reactivation_strength(self, strength: float) -> None:
        """
        Set the strength of pattern reactivation
        
        Args:
            strength: New reactivation strength (0.0 to 1.0)
        """
        self.parameters.reactivation_strength = max(0.0, min(1.0, strength))
        logger.info(f"Changed reactivation strength to: {strength}")
    
    def set_stabilization_rate(self, rate: float) -> None:
        """
        Set the rate of synaptic stabilization
        
        Args:
            rate: New stabilization rate (0.0 to 1.0)
        """
        self.parameters.stabilization_rate = max(0.0, min(1.0, rate))
        self.learning_rate = self.parameters.stabilization_rate
        logger.info(f"Changed stabilization rate to: {rate}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the consolidation engine
        
        Returns:
            Dictionary with engine state
        """
        return {
            "engine_id": self.engine_id,
            "engine_type": self.engine_type,
            "is_active": self.is_active,
            "sleep_mode": self.sleep_mode,
            "parameters": self.parameters.dict(),
            "tagging_parameters": self.tagging_parameters.dict(),
            "operation_count": self.operation_count,
            "activation_patterns_count": len(self.activation_patterns),
            "tagged_synapses_count": len(self.tagged_synapses),
            "reactivation_history_count": len(self.reactivation_history),
            "consolidation_history_count": len(self.consolidation_history),
            "last_applied": self.last_applied
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: Dictionary with engine state
        """
        if "parameters" in state:
            self.parameters = ConsolidationParameters(**state["parameters"])
            
        if "tagging_parameters" in state:
            self.tagging_parameters = SynapticTaggingParameters(**state["tagging_parameters"])
            
        if "is_active" in state:
            self.is_active = state["is_active"]
            
        if "sleep_mode" in state:
            self.sleep_mode = state["sleep_mode"]
            
        if "operation_count" in state:
            self.operation_count = state["operation_count"]
            
        if "last_applied" in state:
            self.last_applied = state["last_applied"]
