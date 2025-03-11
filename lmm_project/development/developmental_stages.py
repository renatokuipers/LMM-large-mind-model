"""
Developmental stages management for the LMM project.

This module implements the developmental stage progression system,
managing transitions between cognitive developmental stages and
providing appropriate parameters for each stage.
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np

from lmm_project.core.event_bus import EventBus, get_event_bus
from lmm_project.core.message import Message, TextContent
from lmm_project.core.types import StateDict, ModuleType, MessageType
from lmm_project.development.models import (
    DevelopmentalStage, 
    StageDefinition,
    DevelopmentConfig,
    StageRange
)
from lmm_project.storage import get_storage_manager
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

class DevelopmentalStages:
    """
    Manages the progression through developmental stages.
    
    This class tracks the mind's age and developmental stage,
    handling transitions between stages based on configured age ranges
    and developmental criteria.
    """
    
    def __init__(self, config: Optional[DevelopmentConfig] = None, event_bus: Optional[EventBus] = None):
        """
        Initialize the developmental stages manager.
        
        Parameters:
        -----------
        config : Optional[DevelopmentConfig]
            Configuration for developmental stages. If None, default settings will be loaded.
        event_bus : Optional[EventBus]
            Event bus for publishing events. If None, a new one will be created.
        """
        self.event_system = event_bus or get_event_bus()
        self._config = config or self._load_default_config()
        
        # Stage definitions indexed by stage enum for faster access
        self._stage_definitions: Dict[DevelopmentalStage, StageDefinition] = {
            stage_def.stage: stage_def for stage_def in self._config.stage_definitions
        }
        
        # Sort stage definitions by min_age for stage determination
        self._sorted_stages = sorted(
            self._config.stage_definitions, 
            key=lambda s: s.range.min_age
        )
        
        # Initialize state
        self._start_time = time.time()
        self._last_update_time = self._start_time
        self._age = self._config.initial_age
        self._current_stage = self._determine_stage(self._age)
        self._previous_stage = None
        self._stage_history: List[Tuple[datetime, DevelopmentalStage, float]] = [
            (datetime.now(), self._current_stage, self._age)
        ]
        
        logger.info(f"Developmental stages initialized with age {self._age}, "
                   f"stage {self._current_stage}")
    
    def _load_default_config(self) -> DevelopmentConfig:
        """
        Load default configuration for developmental stages.
        
        Returns:
        --------
        DevelopmentConfig
            Default configuration
        """
        # Create default stage definitions based on established developmental theory
        stages = [
            StageDefinition(
                stage=DevelopmentalStage.PRENATAL,
                range=StageRange(min_age=0.0, max_age=0.1),
                description="Neural foundation formation stage",
                learning_rate_multiplier=0.8,
                key_capabilities=["basic pattern recognition", "sensory processing"],
                neural_characteristics={
                    "synapse_formation_rate": 2.0,
                    "network_sparsity": 0.4,
                    "plasticity": 0.9
                }
            ),
            StageDefinition(
                stage=DevelopmentalStage.INFANT,
                range=StageRange(min_age=0.1, max_age=1.0),
                description="Rapid learning and basic skill acquisition",
                learning_rate_multiplier=1.5,
                key_capabilities=["object permanence", "basic language", "emotional bonding"],
                neural_characteristics={
                    "synapse_formation_rate": 1.8,
                    "network_sparsity": 0.5,
                    "plasticity": 0.8
                }
            ),
            StageDefinition(
                stage=DevelopmentalStage.CHILD,
                range=StageRange(min_age=1.0, max_age=3.0),
                description="Language mastery and social development",
                learning_rate_multiplier=1.2,
                key_capabilities=["complex language", "social understanding", "causal reasoning"],
                neural_characteristics={
                    "synapse_formation_rate": 1.5,
                    "network_sparsity": 0.6,
                    "plasticity": 0.7
                }
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADOLESCENT,
                range=StageRange(min_age=3.0, max_age=6.0),
                description="Abstract reasoning and identity formation",
                learning_rate_multiplier=1.0,
                key_capabilities=["abstract thinking", "complex problem solving", "identity"],
                neural_characteristics={
                    "synapse_formation_rate": 1.2,
                    "network_sparsity": 0.7,
                    "plasticity": 0.6
                }
            ),
            StageDefinition(
                stage=DevelopmentalStage.ADULT,
                range=StageRange(min_age=6.0, max_age=None),
                description="Integrated cognition and specialized expertise",
                learning_rate_multiplier=0.8,
                key_capabilities=["wisdom", "expertise", "self-actualization"],
                neural_characteristics={
                    "synapse_formation_rate": 0.9,
                    "network_sparsity": 0.8,
                    "plasticity": 0.5
                }
            )
        ]
        
        # Create minimal default config
        return DevelopmentConfig(
            initial_age=0.0,
            time_acceleration=1000.0,
            stage_definitions=stages,
            milestone_definitions=[],
            critical_period_definitions=[]
        )
    
    def _determine_stage(self, age: float) -> DevelopmentalStage:
        """
        Determine the developmental stage based on age.
        
        Parameters:
        -----------
        age : float
            Current developmental age in age units
            
        Returns:
        --------
        DevelopmentalStage
            The determined developmental stage
        """
        for stage_def in self._sorted_stages:
            min_age = stage_def.range.min_age
            max_age = stage_def.range.max_age
            
            if max_age is None:  # Final stage with no upper bound
                if age >= min_age:
                    return stage_def.stage
            elif min_age <= age < max_age:
                return stage_def.stage
                
        # Fallback - should never reach here if config is valid
        logger.error(f"Could not determine stage for age {age}, defaulting to ADULT")
        return DevelopmentalStage.ADULT
    
    def update(self) -> None:
        """
        Update the developmental age and stage based on elapsed time.
        
        This method should be called regularly to update the developmental state.
        """
        current_time = time.time()
        elapsed_real_time = current_time - self._last_update_time
        
        # Convert real time to developmental time using acceleration factor
        elapsed_dev_time = elapsed_real_time * self._config.time_acceleration
        
        # Update age
        new_age = self._age + elapsed_dev_time
        
        # Determine new stage
        new_stage = self._determine_stage(new_age)
        
        # If stage changed, record it and emit event
        if new_stage != self._current_stage:
            self._previous_stage = self._current_stage
            self._current_stage = new_stage
            self._stage_history.append((datetime.now(), new_stage, new_age))
            
            # Emit stage transition event if event bus is running
            try:
                self.event_system.publish(Message(
                    sender="developmental_stages",
                    sender_type=ModuleType.LEARNING,
                    message_type=MessageType.DEVELOPMENT_MILESTONE,
                    content=TextContent(
                        data=f"Developmental stage changed from {self._previous_stage} to {self._current_stage} at age {new_age:.3f}"
                    ),
                    metadata={
                        "previous_stage": self._previous_stage,
                        "new_stage": self._current_stage,
                        "age": new_age
                    }
                ))
            except Exception as e:
                logger.warning(f"Could not publish stage transition event: {e}")
            
            logger.info(f"Developmental stage transitioned from {self._previous_stage} "
                       f"to {self._current_stage} at age {new_age:.3f}")
        
        # Update age and time
        self._age = new_age
        self._last_update_time = current_time
    
    def get_current_stage(self) -> DevelopmentalStage:
        """
        Get the current developmental stage.
        
        Returns:
        --------
        DevelopmentalStage
            Current developmental stage
        """
        return self._current_stage
    
    def get_stage_definition(self, stage: Optional[DevelopmentalStage] = None) -> StageDefinition:
        """
        Get the definition for a specific stage or the current stage.
        
        Parameters:
        -----------
        stage : Optional[DevelopmentalStage]
            The stage to get definition for. If None, current stage is used.
            
        Returns:
        --------
        StageDefinition
            Definition of the requested stage
        """
        stage = stage or self._current_stage
        return self._stage_definitions[stage]
    
    def get_age(self) -> float:
        """
        Get the current developmental age.
        
        Returns:
        --------
        float
            Current age in age units
        """
        return self._age
    
    def set_age(self, age: float) -> None:
        """
        Set the developmental age manually and update the stage.
        
        Parameters:
        -----------
        age : float
            New developmental age in age units
        """
        if age < 0:
            raise ValueError("Age cannot be negative")
            
        self._age = age
        new_stage = self._determine_stage(age)
        
        if new_stage != self._current_stage:
            self._previous_stage = self._current_stage
            self._current_stage = new_stage
            self._stage_history.append((datetime.now(), new_stage, age))
            
            # Emit stage transition event
            self.event_system.publish(Message(
                sender="developmental_stages",
                sender_type=ModuleType.LEARNING,
                message_type=MessageType.DEVELOPMENT_MILESTONE,
                content=TextContent(
                    data=f"Developmental stage changed from {self._previous_stage} to {self._current_stage} at age {age:.3f}"
                ),
                metadata={
                    "previous_stage": self._previous_stage,
                    "new_stage": self._current_stage,
                    "age": age
                }
            ))
            
            logger.info(f"Developmental stage manually set from {self._previous_stage} "
                       f"to {self._current_stage} at age {age:.3f}")
    
    def get_learning_rate_multiplier(self) -> float:
        """
        Get the learning rate multiplier for the current stage.
        
        Returns:
        --------
        float
            Learning rate multiplier for the current developmental stage
        """
        return self._stage_definitions[self._current_stage].learning_rate_multiplier
    
    def get_stage_history(self) -> List[Tuple[datetime, DevelopmentalStage, float]]:
        """
        Get the history of stage transitions.
        
        Returns:
        --------
        List[Tuple[datetime, DevelopmentalStage, float]]
            List of (timestamp, stage, age) records
        """
        return self._stage_history.copy()
    
    def get_time_in_current_stage(self) -> float:
        """
        Get the time spent in the current stage.
        
        Returns:
        --------
        float
            Time spent in current stage in age units
        """
        # Get stage range
        stage_def = self._stage_definitions[self._current_stage]
        min_age = stage_def.range.min_age
        
        # Time in stage is current age minus stage start age
        return self._age - min_age
    
    def get_state(self) -> StateDict:
        """
        Get the current state as a dictionary for saving.
        
        Returns:
        --------
        StateDict
            Current state dictionary
        """
        return {
            "age": self._age,
            "current_stage": self._current_stage,
            "previous_stage": self._previous_stage,
            "stage_history": self._stage_history,
            "start_time": self._start_time,
            "last_update_time": self._last_update_time
        }
    
    def load_state(self, state: StateDict) -> None:
        """
        Load state from a state dictionary.
        
        Parameters:
        -----------
        state : StateDict
            State dictionary to load from
        """
        self._age = state["age"]
        self._current_stage = state["current_stage"]
        self._previous_stage = state["previous_stage"]
        self._stage_history = state["stage_history"]
        self._start_time = state["start_time"]
        self._last_update_time = state["last_update_time"]
        
        logger.info(f"Developmental stages state loaded: age {self._age}, "
                   f"stage {self._current_stage}")
    
    def estimate_time_to_next_stage(self) -> Optional[float]:
        """
        Estimate time until the next developmental stage.
        
        Returns:
        --------
        Optional[float]
            Estimated time in seconds until next stage transition, or None if in final stage
        """
        current_def = self._stage_definitions[self._current_stage]
        
        # If in final stage, return None
        if current_def.range.max_age is None:
            return None
            
        # Calculate age difference to next stage
        age_difference = current_def.range.max_age - self._age
        
        # Convert to real time using acceleration factor
        return age_difference / self._config.time_acceleration
