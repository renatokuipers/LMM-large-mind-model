"""
Learning Engines Package

This package contains engines that implement different learning mechanisms for the LMM:

- Hebbian Learning: Associates neurons that fire together through various Hebbian rules
- Reinforcement Learning: Learns from rewards and feedback
- Neural Pruning: Removes weak or unused connections to optimize neural networks
- Memory Consolidation: Stabilizes important neural patterns into long-term memories

These learning engines work together to create a comprehensive learning system
that enables the LMM to develop and adapt through experience.
"""

from typing import Dict, List, Any, Optional, Set, Union, Tuple
import logging
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.learning_engines.models import (
    LearningEngine, 
    HebbianParameters, 
    ReinforcementParameters,
    PruningParameters,
    ConsolidationParameters,
    SynapticTaggingParameters,
    LearningEvent
)
from lmm_project.learning_engines.hebbian_engine import HebbianEngine
from lmm_project.learning_engines.reinforcement_engine import ReinforcementEngine
from lmm_project.learning_engines.pruning_engine import PruningEngine
from lmm_project.learning_engines.consolidation_engine import ConsolidationEngine

logger = logging.getLogger(__name__)

def get_learning_system(event_bus: Optional[EventBus] = None) -> "LearningSystem":
    """
    Factory function to create a complete learning system
    
    Args:
        event_bus: Optional event bus for broadcasting learning events
    
    Returns:
        A fully configured LearningSystem instance
    """
    return LearningSystem(event_bus=event_bus)

class LearningSystem:
    """
    Integrated learning system for the LMM
    
    This class integrates all learning engines:
    - Hebbian Learning
    - Reinforcement Learning
    - Neural Pruning
    - Memory Consolidation
    
    It provides a unified interface for applying different learning mechanisms
    to neural networks in the LMM.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the learning system with all engines
        
        Args:
            event_bus: Optional event bus for broadcasting learning events
        """
        self.event_bus = event_bus
        
        # Initialize learning engines
        self.hebbian_engine = HebbianEngine(
            parameters=HebbianParameters(),
            event_bus=event_bus
        )
        
        self.reinforcement_engine = ReinforcementEngine(
            parameters=ReinforcementParameters(),
            event_bus=event_bus
        )
        
        self.pruning_engine = PruningEngine(
            parameters=PruningParameters(),
            event_bus=event_bus
        )
        
        self.consolidation_engine = ConsolidationEngine(
            parameters=ConsolidationParameters(),
            tagging_parameters=SynapticTaggingParameters(),
            event_bus=event_bus
        )
        
        # Dictionary of all engines for easy access
        self.engines: Dict[str, LearningEngine] = {
            "hebbian": self.hebbian_engine,
            "reinforcement": self.reinforcement_engine,
            "pruning": self.pruning_engine,
            "consolidation": self.consolidation_engine
        }
        
        # Track learning events
        self.learning_events: List[Dict[str, Any]] = []
        
        # Sleep mode for consolidation
        self.sleep_mode = False
        
        logger.info("Learning system initialized with all engines")
    
    def apply_learning(
        self, 
        network: NeuralNetwork,
        reward: float = 0.0,
        hebbian_enabled: bool = True,
        reinforcement_enabled: bool = True,
        pruning_enabled: bool = True,
        consolidation_enabled: bool = True
    ) -> Dict[str, List[LearningEvent]]:
        """
        Apply all learning mechanisms to a neural network
        
        Args:
            network: The neural network to apply learning to
            reward: Reward signal for reinforcement learning
            hebbian_enabled: Whether to apply Hebbian learning
            reinforcement_enabled: Whether to apply reinforcement learning
            pruning_enabled: Whether to apply neural pruning
            consolidation_enabled: Whether to apply memory consolidation
            
        Returns:
            Dictionary mapping engine types to lists of learning events
        """
        learning_results: Dict[str, List[LearningEvent]] = {}
        
        # Apply Hebbian learning
        if hebbian_enabled and self.hebbian_engine.is_active:
            hebbian_events = self.hebbian_engine.apply_learning(network)
            learning_results["hebbian"] = hebbian_events
            self._record_events(hebbian_events, "hebbian")
        
        # Apply reinforcement learning
        if reinforcement_enabled and self.reinforcement_engine.is_active:
            reinforcement_events = self.reinforcement_engine.apply_learning(network, reward)
            learning_results["reinforcement"] = reinforcement_events
            self._record_events(reinforcement_events, "reinforcement")
        
        # Apply memory consolidation
        if consolidation_enabled and self.consolidation_engine.is_active:
            consolidation_events = self.consolidation_engine.apply_learning(
                network, 
                sleep_mode=self.sleep_mode
            )
            learning_results["consolidation"] = consolidation_events
            self._record_events(consolidation_events, "consolidation")
        
        # Apply neural pruning (typically less frequent)
        if pruning_enabled and self.pruning_engine.is_active:
            pruning_events = self.pruning_engine.apply_learning(network)
            learning_results["pruning"] = pruning_events
            self._record_events(pruning_events, "pruning")
        
        return learning_results
    
    def _record_events(self, events: List[LearningEvent], engine_type: str) -> None:
        """
        Record learning events for tracking
        
        Args:
            events: List of learning events
            engine_type: Type of engine that generated the events
        """
        for event in events:
            self.learning_events.append({
                "timestamp": datetime.now(),
                "engine_type": engine_type,
                "event": event.dict()
            })
            
        # Limit the number of stored events
        if len(self.learning_events) > 1000:
            self.learning_events = self.learning_events[-1000:]
    
    def provide_reward(self, reward: float) -> None:
        """
        Provide an external reward signal to the reinforcement engine
        
        Args:
            reward: Reward value (positive or negative)
        """
        self.reinforcement_engine.provide_reward(reward)
    
    def enter_sleep_mode(self) -> None:
        """Enter sleep mode for enhanced memory consolidation"""
        self.sleep_mode = True
        self.consolidation_engine.enter_sleep_mode()
        logger.info("Learning system entered sleep mode")
    
    def exit_sleep_mode(self) -> None:
        """Exit sleep mode"""
        self.sleep_mode = False
        self.consolidation_engine.exit_sleep_mode()
        logger.info("Learning system exited sleep mode")
    
    def is_in_sleep_mode(self) -> bool:
        """Check if the system is in sleep mode"""
        return self.sleep_mode
    
    def enable_engine(self, engine_type: str) -> None:
        """
        Enable a specific learning engine
        
        Args:
            engine_type: Type of engine to enable
        """
        if engine_type in self.engines:
            self.engines[engine_type].is_active = True
            logger.info(f"Enabled {engine_type} engine")
        else:
            logger.warning(f"Unknown engine type: {engine_type}")
    
    def disable_engine(self, engine_type: str) -> None:
        """
        Disable a specific learning engine
        
        Args:
            engine_type: Type of engine to disable
        """
        if engine_type in self.engines:
            self.engines[engine_type].is_active = False
            logger.info(f"Disabled {engine_type} engine")
        else:
            logger.warning(f"Unknown engine type: {engine_type}")
    
    def get_engine(self, engine_type: str) -> Optional[LearningEngine]:
        """
        Get a specific learning engine
        
        Args:
            engine_type: Type of engine to get
            
        Returns:
            The requested learning engine, or None if not found
        """
        return self.engines.get(engine_type)
    
    def get_active_engines(self) -> Dict[str, LearningEngine]:
        """
        Get all active learning engines
        
        Returns:
            Dictionary of active engines
        """
        return {
            engine_type: engine 
            for engine_type, engine in self.engines.items()
            if engine.is_active
        }
    
    def set_learning_rate(self, engine_type: str, learning_rate: float) -> None:
        """
        Set the learning rate for a specific engine
        
        Args:
            engine_type: Type of engine to update
            learning_rate: New learning rate
        """
        if engine_type in self.engines:
            engine = self.engines[engine_type]
            
            if hasattr(engine, "set_learning_rate"):
                engine.set_learning_rate(learning_rate)
            else:
                engine.learning_rate = learning_rate
                
            logger.info(f"Set learning rate for {engine_type} engine to {learning_rate}")
        else:
            logger.warning(f"Unknown engine type: {engine_type}")
    
    # ---------- Hebbian Engine Methods ----------
    
    def set_hebbian_learning_rule(self, rule: str) -> None:
        """
        Set the learning rule for the Hebbian engine
        
        Args:
            rule: Learning rule to use (hebbian, oja, bcm, or stdp)
        """
        self.hebbian_engine.set_learning_rule(rule)
        logger.info(f"Set Hebbian learning rule to {rule}")
    
    # ---------- Reinforcement Engine Methods ----------
    
    def set_exploration_rate(self, rate: float) -> None:
        """
        Set the exploration rate for the reinforcement engine
        
        Args:
            rate: New exploration rate (0.0 to 1.0)
        """
        self.reinforcement_engine.set_exploration_rate(rate)
        logger.info(f"Set reinforcement exploration rate to {rate}")
    
    def set_learning_method(self, method: str) -> None:
        """
        Set the learning method for the reinforcement engine
        
        Args:
            method: Learning method to use (q_learning, sarsa, or actor_critic)
        """
        self.reinforcement_engine.set_learning_method(method)
        logger.info(f"Set reinforcement learning method to {method}")
    
    def decay_exploration(self, decay_factor: float = 0.99) -> None:
        """
        Decay the exploration rate for the reinforcement engine
        
        Args:
            decay_factor: Factor to multiply the current exploration rate by
        """
        self.reinforcement_engine.decay_exploration(decay_factor)
        logger.info(f"Decayed reinforcement exploration rate by factor {decay_factor}")
    
    def get_action_selection(
        self, 
        options: Dict[str, float], 
        use_exploration: bool = True
    ) -> Tuple[str, float]:
        """
        Get an action selection from the reinforcement engine
        
        Args:
            options: Dictionary mapping action IDs to their values
            use_exploration: Whether to use exploration in selection
            
        Returns:
            Tuple of (selected action ID, action value)
        """
        return self.reinforcement_engine.get_action_selection(options, use_exploration)
    
    # ---------- Pruning Engine Methods ----------
    
    def set_pruning_strategy(self, strategy: str) -> None:
        """
        Set the pruning strategy for the pruning engine
        
        Args:
            strategy: Pruning strategy to use (weight, usage, importance, or combined)
        """
        self.pruning_engine.set_pruning_strategy(strategy)
        logger.info(f"Set pruning strategy to {strategy}")
    
    def set_weight_threshold(self, threshold: float) -> None:
        """
        Set the weight threshold for the pruning engine
        
        Args:
            threshold: New weight threshold
        """
        self.pruning_engine.set_weight_threshold(threshold)
        logger.info(f"Set pruning weight threshold to {threshold}")
    
    def set_usage_threshold(self, threshold: float) -> None:
        """
        Set the usage threshold for the pruning engine
        
        Args:
            threshold: New usage threshold
        """
        self.pruning_engine.set_usage_threshold(threshold)
        logger.info(f"Set pruning usage threshold to {threshold}")
    
    def set_importance_threshold(self, threshold: float) -> None:
        """
        Set the importance threshold for the pruning engine
        
        Args:
            threshold: New importance threshold
        """
        self.pruning_engine.set_importance_threshold(threshold)
        logger.info(f"Set pruning importance threshold to {threshold}")
    
    def set_max_prune_percent(self, percent: float) -> None:
        """
        Set the maximum pruning percentage for the pruning engine
        
        Args:
            percent: New maximum percentage (0.0 to 1.0)
        """
        self.pruning_engine.set_max_prune_percent(percent)
        logger.info(f"Set maximum pruning percentage to {percent}")
    
    def set_pruning_frequency(self, frequency: int) -> None:
        """
        Set how often pruning is applied
        
        Args:
            frequency: New pruning frequency
        """
        self.pruning_engine.set_pruning_frequency(frequency)
        logger.info(f"Set pruning frequency to {frequency}")
    
    def set_recovery_probability(self, probability: float) -> None:
        """
        Set the probability of recovering pruned synapses
        
        Args:
            probability: New recovery probability (0.0 to 1.0)
        """
        self.pruning_engine.set_recovery_probability(probability)
        logger.info(f"Set pruning recovery probability to {probability}")
    
    # ---------- Consolidation Engine Methods ----------
    
    def set_consolidation_threshold(self, threshold: float) -> None:
        """
        Set the consolidation threshold for the consolidation engine
        
        Args:
            threshold: New consolidation threshold (0.0 to 1.0)
        """
        self.consolidation_engine.set_consolidation_threshold(threshold)
        logger.info(f"Set consolidation threshold to {threshold}")
    
    def set_reactivation_strength(self, strength: float) -> None:
        """
        Set the pattern reactivation strength for the consolidation engine
        
        Args:
            strength: New reactivation strength (0.0 to 1.0)
        """
        self.consolidation_engine.set_reactivation_strength(strength)
        logger.info(f"Set consolidation reactivation strength to {strength}")
    
    def set_stabilization_rate(self, rate: float) -> None:
        """
        Set the memory stabilization rate for the consolidation engine
        
        Args:
            rate: New stabilization rate (0.0 to 1.0)
        """
        self.consolidation_engine.set_stabilization_rate(rate)
        logger.info(f"Set consolidation stabilization rate to {rate}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the learning system
        
        Returns:
            Dictionary with system state
        """
        return {
            "sleep_mode": self.sleep_mode,
            "engine_states": {
                engine_type: engine.get_state()
                for engine_type, engine in self.engines.items()
            },
            "learning_events_count": len(self.learning_events)
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: Dictionary with system state
        """
        if "sleep_mode" in state:
            self.sleep_mode = state["sleep_mode"]
            if self.sleep_mode:
                self.consolidation_engine.enter_sleep_mode()
            else:
                self.consolidation_engine.exit_sleep_mode()
                
        if "engine_states" in state:
            for engine_type, engine_state in state["engine_states"].items():
                if engine_type in self.engines:
                    self.engines[engine_type].load_state(engine_state)
                    
        logger.info("Learning system state loaded") 