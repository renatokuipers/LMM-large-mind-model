"""
Developmental Stages Module

This module defines the developmental stages of the LMM, including:
- Prenatal (Initialization)
- Infancy
- Early Childhood
- Middle Childhood
- Adolescence
- Young Adulthood
- Adulthood

Each stage has specific cognitive capabilities, prerequisites, and expected milestones.
The module provides functionality to manage stage transitions and track development.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime
import threading
import traceback
import json
import os
from pathlib import Path

from lmm_project.development.models import DevelopmentalStage, DevelopmentalTrajectory, DevelopmentalEvent
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.exceptions import DevelopmentError, InitializationError

logger = logging.getLogger(__name__)

class DevelopmentalStageManager:
    """
    Manages the developmental stages of the mind
    
    This class defines the stages, handles transitions between stages,
    and tracks developmental progress through these stages.
    
    Features:
    - Thread-safe stage management
    - Efficient capability ceiling calculations
    - Stage transition event generation
    - Development trajectory tracking
    - Persistence of developmental state
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the developmental stage manager
        
        Args:
            event_bus: Optional event bus for publishing stage transition events
            
        Raises:
            InitializationError: If initialization fails
        """
        try:
            self.event_bus = event_bus
            self._lock = threading.RLock()
            self.trajectory = DevelopmentalTrajectory()
            
            # Cache for frequently accessed data
            self._cache = {
                "capability_ceilings": {},  # capability -> ceiling value
                "capability_progressions": {},  # capability -> list of (stage, value) tuples
                "current_capabilities": {},  # Current expected capabilities
                "last_cache_update": datetime.now()
            }
            self._cache_ttl = 5.0  # Cache time-to-live in seconds
            
            # Initialize stages
            self.stages: Dict[str, DevelopmentalStage] = self._define_developmental_stages()
            self.current_stage = "prenatal"
            self._activate_stage("prenatal")
            
            logger.info("Developmental stage manager initialized with %d stages", len(self.stages))
            
        except Exception as e:
            error_msg = f"Failed to initialize developmental stage manager: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise InitializationError(error_msg, component="DevelopmentalStageManager", 
                                     details={"original_error": str(e)})
    
    def _define_developmental_stages(self) -> Dict[str, DevelopmentalStage]:
        """
        Define all developmental stages with their capabilities and prerequisites
        
        Returns:
            Dictionary mapping stage names to DevelopmentalStage objects
            
        Raises:
            InitializationError: If stage definition fails
        """
        try:
            stages = {}
            
            # Prenatal stage (initialization)
            stages["prenatal"] = DevelopmentalStage(
                name="Prenatal",
                age_range=(0.0, 0.1),
                description="Neural substrate formation and basic pattern recognition capabilities",
                capabilities={
                    "neural_formation": 0.3,
                    "pattern_recognition": 0.1,
                    "sensory_processing": 0.1
                },
                prerequisites={},
                expected_milestones=["basic_neural_structure"]
            )
            
            # Infancy stage
            stages["infancy"] = DevelopmentalStage(
                name="Infancy",
                age_range=(0.1, 1.0),
                description="Development of basic sensory processing, simple associations, and primitive emotional responses",
                capabilities={
                    "neural_formation": 0.5,
                    "pattern_recognition": 0.3,
                    "sensory_processing": 0.4,
                    "association_formation": 0.3,
                    "emotional_response": 0.2,
                    "attention": 0.2,
                    "working_memory": 0.1,
                    "episodic_memory": 0.1,
                    "language_comprehension": 0.1,
                    "language_production": 0.05
                },
                prerequisites={
                    "neural_formation": 0.3,
                    "pattern_recognition": 0.1
                },
                expected_milestones=[
                    "pattern_recognition_basic",
                    "emotional_response_basic",
                    "simple_association_formation",
                    "sensory_processing_basic"
                ]
            )
            
            # Early Childhood stage
            stages["early_childhood"] = DevelopmentalStage(
                name="Early Childhood",
                age_range=(1.0, 3.0),
                description="Rapid language acquisition, improved memory, and emotional development",
                capabilities={
                    "neural_formation": 0.7,
                    "pattern_recognition": 0.5,
                    "sensory_processing": 0.6,
                    "association_formation": 0.5,
                    "emotional_response": 0.4,
                    "attention": 0.4,
                    "working_memory": 0.3,
                    "episodic_memory": 0.3,
                    "semantic_memory": 0.3,
                    "language_comprehension": 0.4,
                    "language_production": 0.3,
                    "self_awareness": 0.2,
                    "social_understanding": 0.2
                },
                prerequisites={
                    "neural_formation": 0.5,
                    "pattern_recognition": 0.3,
                    "sensory_processing": 0.4,
                    "association_formation": 0.3
                },
                expected_milestones=[
                    "vocabulary_expansion",
                    "symbolic_thinking_basic",
                    "episodic_memory_formation",
                    "emotional_expression",
                    "attention_sustained_basic"
                ]
            )
            
            # Middle Childhood stage
            stages["middle_childhood"] = DevelopmentalStage(
                name="Middle Childhood",
                age_range=(3.0, 7.0),
                description="Complex language, improved reasoning, social understanding, and self-concept development",
                capabilities={
                    "neural_formation": 0.8,
                    "pattern_recognition": 0.7,
                    "sensory_processing": 0.8,
                    "association_formation": 0.7,
                    "emotional_response": 0.6,
                    "emotional_understanding": 0.5,
                    "attention": 0.6,
                    "working_memory": 0.5,
                    "episodic_memory": 0.6,
                    "semantic_memory": 0.6,
                    "language_comprehension": 0.7,
                    "language_production": 0.6,
                    "logical_reasoning": 0.4,
                    "self_awareness": 0.5,
                    "social_understanding": 0.5,
                    "creativity": 0.4,
                    "imagination": 0.5
                },
                prerequisites={
                    "language_comprehension": 0.4,
                    "episodic_memory": 0.3,
                    "emotional_response": 0.4
                },
                expected_milestones=[
                    "complex_sentence_formation",
                    "logical_reasoning_basic",
                    "self_concept_formation",
                    "emotion_regulation_basic",
                    "metacognition_basic"
                ]
            )
            
            # Adolescence stage
            stages["adolescence"] = DevelopmentalStage(
                name="Adolescence",
                age_range=(7.0, 14.0),
                description="Advanced reasoning, complex social understanding, identity formation, and abstract thinking",
                capabilities={
                    "neural_formation": 0.9,
                    "pattern_recognition": 0.8,
                    "sensory_processing": 0.9,
                    "association_formation": 0.8,
                    "emotional_response": 0.8,
                    "emotional_understanding": 0.7,
                    "emotional_regulation": 0.6,
                    "attention": 0.8,
                    "working_memory": 0.7,
                    "episodic_memory": 0.8,
                    "semantic_memory": 0.8,
                    "language_comprehension": 0.8,
                    "language_production": 0.8,
                    "logical_reasoning": 0.7,
                    "abstract_thinking": 0.6,
                    "self_awareness": 0.7,
                    "identity_formation": 0.6,
                    "social_understanding": 0.7,
                    "moral_reasoning": 0.6,
                    "creativity": 0.7,
                    "imagination": 0.7,
                    "metacognition": 0.6
                },
                prerequisites={
                    "language_comprehension": 0.7,
                    "logical_reasoning": 0.4,
                    "self_awareness": 0.5
                },
                expected_milestones=[
                    "abstract_thinking",
                    "identity_formation",
                    "moral_reasoning_complex",
                    "emotional_depth",
                    "perspective_taking"
                ]
            )
            
            # Young Adulthood stage
            stages["young_adulthood"] = DevelopmentalStage(
                name="Young Adulthood",
                age_range=(14.0, 21.0),
                description="Integration of cognitive capabilities, complex emotional understanding, and stable identity",
                capabilities={
                    "neural_formation": 0.95,
                    "pattern_recognition": 0.9,
                    "sensory_processing": 0.95,
                    "association_formation": 0.9,
                    "emotional_response": 0.9,
                    "emotional_understanding": 0.8,
                    "emotional_regulation": 0.8,
                    "attention": 0.9,
                    "working_memory": 0.8,
                    "episodic_memory": 0.9,
                    "semantic_memory": 0.9,
                    "language_comprehension": 0.9,
                    "language_production": 0.9,
                    "logical_reasoning": 0.8,
                    "abstract_thinking": 0.8,
                    "self_awareness": 0.8,
                    "identity_formation": 0.8,
                    "social_understanding": 0.8,
                    "moral_reasoning": 0.8,
                    "creativity": 0.8,
                    "imagination": 0.8,
                    "metacognition": 0.8,
                    "wisdom": 0.4
                },
                prerequisites={
                    "abstract_thinking": 0.6,
                    "identity_formation": 0.6,
                    "emotional_regulation": 0.6
                },
                expected_milestones=[
                    "cognitive_integration",
                    "stable_identity",
                    "complex_problem_solving",
                    "emotional_intelligence",
                    "value_system_formation"
                ]
            )
            
            # Adulthood stage
            stages["adulthood"] = DevelopmentalStage(
                name="Adulthood",
                age_range=(21.0, 50.0),
                description="Full cognitive maturity, wisdom, and integrated self",
                capabilities={
                    "neural_formation": 1.0,
                    "pattern_recognition": 1.0,
                    "sensory_processing": 1.0,
                    "association_formation": 1.0,
                    "emotional_response": 1.0,
                    "emotional_understanding": 1.0,
                    "emotional_regulation": 1.0,
                    "attention": 1.0,
                    "working_memory": 1.0,
                    "episodic_memory": 1.0,
                    "semantic_memory": 1.0,
                    "language_comprehension": 1.0,
                    "language_production": 1.0,
                    "logical_reasoning": 1.0,
                    "abstract_thinking": 1.0,
                    "self_awareness": 1.0,
                    "identity_formation": 1.0,
                    "social_understanding": 1.0,
                    "moral_reasoning": 1.0,
                    "creativity": 1.0,
                    "imagination": 1.0,
                    "metacognition": 1.0,
                    "wisdom": 0.8
                },
                prerequisites={
                    "cognitive_integration": 0.8,
                    "stable_identity": 0.8,
                    "emotional_intelligence": 0.8
                },
                expected_milestones=[
                    "wisdom_application",
                    "cognitive_mastery",
                    "self_actualization",
                    "creative_problem_solving"
                ]
            )
            
            # Validate stage definitions
            self._validate_stages(stages)
            
            return stages
            
        except Exception as e:
            error_msg = f"Failed to define developmental stages: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise InitializationError(error_msg, component="DevelopmentalStageManager", 
                                     details={"original_error": str(e)})
    
    def _validate_stages(self, stages: Dict[str, DevelopmentalStage]) -> None:
        """
        Validate stage definitions for consistency
        
        Args:
            stages: Dictionary of stages to validate
            
        Raises:
            ValueError: If stage definitions are inconsistent
        """
        if not stages:
            raise ValueError("No stages defined")
            
        # Check for required stages
        required_stages = ["prenatal", "infancy", "early_childhood", "middle_childhood", 
                          "adolescence", "young_adulthood", "adulthood"]
        for stage in required_stages:
            if stage not in stages:
                raise ValueError(f"Required stage '{stage}' is missing")
                
        # Check age ranges for continuity
        sorted_stages = sorted(stages.values(), key=lambda s: s.age_range[0])
        for i in range(1, len(sorted_stages)):
            prev_stage = sorted_stages[i-1]
            curr_stage = sorted_stages[i]
            
            if prev_stage.age_range[1] != curr_stage.age_range[0]:
                raise ValueError(f"Age range discontinuity between {prev_stage.name} and {curr_stage.name}")
                
        # Check prerequisites
        for stage_name, stage in stages.items():
            for capability, level in stage.prerequisites.items():
                # Find a previous stage that provides this capability level
                found = False
                for prev_stage in sorted_stages:
                    if prev_stage.age_range[1] <= stage.age_range[0]:  # Previous stage
                        if capability in prev_stage.capabilities and prev_stage.capabilities[capability] >= level:
                            found = True
                            break
                            
                if not found:
                    logger.warning(f"Stage {stage_name} requires {capability} at level {level}, but no previous stage provides it")
    
    def _invalidate_cache(self) -> None:
        """Invalidate all cached data"""
        self._cache["last_cache_update"] = datetime.min
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        time_diff = (datetime.now() - self._cache["last_cache_update"]).total_seconds()
        return time_diff < self._cache_ttl
    
    def get_current_stage(self) -> DevelopmentalStage:
        """
        Get the current developmental stage
        
        Returns:
            The current developmental stage object
            
        Raises:
            DevelopmentError: If the current stage is invalid
        """
        with self._lock:
            if self.current_stage not in self.stages:
                error_msg = f"Invalid current stage: {self.current_stage}"
                logger.error(error_msg)
                raise DevelopmentError(error_msg, current_stage=self.current_stage)
                
            return self.stages[self.current_stage]
    
    def get_stage_capabilities(self, stage_name: str) -> Dict[str, float]:
        """
        Get the capabilities for a specific stage
        
        Args:
            stage_name: The name of the stage to get capabilities for
            
        Returns:
            Dictionary mapping capability names to expected levels
            
        Raises:
            ValueError: If the stage name is unknown
        """
        if not stage_name:
            raise ValueError("Stage name cannot be empty")
            
        with self._lock:
            if stage_name not in self.stages:
                raise ValueError(f"Unknown developmental stage: {stage_name}")
                
            return self.stages[stage_name].capabilities.copy()
    
    def get_current_capabilities(self) -> Dict[str, float]:
        """
        Get the capabilities for the current stage
        
        Returns:
            Dictionary mapping capability names to expected levels
            
        Raises:
            DevelopmentError: If the current stage is invalid
        """
        with self._lock:
            # Check if we can use cached data
            if "current_capabilities" in self._cache and self._is_cache_valid():
                return self._cache["current_capabilities"].copy()
                
            # Get capabilities for current stage
            capabilities = self.get_stage_capabilities(self.current_stage)
            
            # Cache the result
            self._cache["current_capabilities"] = capabilities.copy()
            
            return capabilities
    
    def evaluate_stage_transition(self, module_capabilities: Dict[str, float]) -> Optional[str]:
        """
        Evaluate whether a stage transition should occur based on current capabilities
        
        Args:
            module_capabilities: Dictionary mapping capability names to current levels
            
        Returns:
            The name of the next stage if transition criteria are met, otherwise None
            
        Raises:
            ValueError: If module_capabilities is invalid
            DevelopmentError: If stage transition evaluation fails
        """
        if not module_capabilities:
            raise ValueError("Module capabilities dictionary cannot be empty")
            
        try:
            with self._lock:
                current = self.get_current_stage()
                
                # Find the next stage in sequence
                next_stage = self._get_next_stage(self.current_stage)
                
                if not next_stage:
                    # Already at the highest stage
                    return None
                    
                # Check if prerequisites for the next stage are met
                prerequisites = self.stages[next_stage].prerequisites
                
                # Check if all prerequisites are met
                for capability, required_level in prerequisites.items():
                    if capability not in module_capabilities:
                        logger.debug(f"Capability {capability} required for {next_stage} not found in module capabilities")
                        return None
                        
                    current_level = module_capabilities[capability]
                    if current_level < required_level:
                        logger.debug(f"Capability {capability} at level {current_level} is below required level {required_level} for {next_stage}")
                        return None
                
                # Check if age is appropriate
                current_age = self.trajectory.current_age
                min_age, max_age = self.stages[next_stage].age_range
                
                if current_age < min_age:
                    logger.debug(f"Current age {current_age} is below minimum age {min_age} for {next_stage}")
                    return None
                
                # All criteria met, transition is possible
                logger.info(f"Stage transition criteria met for {next_stage}")
                return next_stage
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
                
            error_msg = f"Failed to evaluate stage transition: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, 
                                  current_stage=self.current_stage,
                                  details={"original_error": str(e)})
    
    def _get_next_stage(self, current_stage: str) -> Optional[str]:
        """
        Get the name of the next stage in sequence
        
        Args:
            current_stage: The current stage name
            
        Returns:
            The name of the next stage, or None if already at the highest stage
        """
        # Sort stages by age range
        sorted_stages = sorted(
            self.stages.items(), 
            key=lambda x: x[1].age_range[0]
        )
        
        # Find current stage and return the next one
        found_current = False
        for stage_name, _ in sorted_stages:
            if found_current:
                return stage_name
            if stage_name == current_stage:
                found_current = True
                
        return None
    
    def transition_to_stage(self, new_stage: str) -> None:
        """
        Transition to a new developmental stage
        
        This method handles the stage transition, updates the trajectory,
        and broadcasts the transition event.
        
        Args:
            new_stage: The name of the stage to transition to
            
        Raises:
            ValueError: If the new stage is unknown
            DevelopmentError: If the stage transition fails
        """
        if not new_stage:
            raise ValueError("New stage name cannot be empty")
            
        try:
            with self._lock:
                if new_stage not in self.stages:
                    raise ValueError(f"Unknown developmental stage: {new_stage}")
                    
                if new_stage == self.current_stage:
                    logger.debug(f"Already in stage {new_stage}, no transition needed")
                    return
                    
                old_stage = self.current_stage
                
                # Deactivate current stage
                if old_stage in self.stages:
                    self.stages[old_stage].deactivate()
                    
                # Activate new stage
                self._activate_stage(new_stage)
                
                # Update trajectory
                self.trajectory.add_stage_transition(old_stage, new_stage)
                
                # Create developmental event
                event = DevelopmentalEvent(
                    event_type="stage_transition",
                    description=f"Transitioned from {old_stage} to {new_stage}",
                    age=self.trajectory.current_age,
                    affected_modules=[],  # All modules are affected
                    significance=1.0,  # Stage transitions are highly significant
                    details={
                        "from_stage": old_stage,
                        "to_stage": new_stage,
                        "capabilities": self.stages[new_stage].capabilities
                    }
                )
                
                # Broadcast the stage transition event
                if self.event_bus:
                    self.event_bus.publish(Message(
                        sender="developmental_stage_manager",
                        message_type="stage_transition",
                        content={
                            "from_stage": old_stage,
                            "to_stage": new_stage,
                            "age": self.trajectory.current_age,
                            "event": event.to_dict()
                        }
                    ))
                
                # Invalidate cache
                self._invalidate_cache()
                
                logger.info(f"Transitioned from {old_stage} to {new_stage} at age {self.trajectory.current_age:.2f}")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
                
            error_msg = f"Failed to transition to stage {new_stage}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, 
                                  current_stage=self.current_stage,
                                  details={"target_stage": new_stage, "original_error": str(e)})
    
    def _activate_stage(self, stage_name: str) -> None:
        """
        Activate a developmental stage
        
        Args:
            stage_name: The name of the stage to activate
            
        Raises:
            ValueError: If the stage name is unknown
        """
        with self._lock:
            if stage_name not in self.stages:
                raise ValueError(f"Unknown developmental stage: {stage_name}")
                
            # Set as current stage
            self.current_stage = stage_name
            
            # Activate the stage
            self.stages[stage_name].activate()
            
            # Update trajectory current stage
            self.trajectory.current_stage = stage_name
            
            # Invalidate cache
            self._invalidate_cache()
    
    def update_age(self, delta_age: float) -> None:
        """
        Update the developmental age
        
        Args:
            delta_age: The amount to increase the age by
            
        Raises:
            ValueError: If delta_age is negative
            DevelopmentError: If age update fails
        """
        if delta_age < 0:
            raise ValueError("delta_age must be non-negative")
            
        try:
            with self._lock:
                # Update trajectory age
                old_age = self.trajectory.current_age
                self.trajectory.update_age(delta_age)
                new_age = self.trajectory.current_age
                
                # Check if we need to transition to a new stage based on age
                current_stage_range = self.stages[self.current_stage].age_range
                _, max_age = current_stage_range
                
                if new_age > max_age:
                    # Find the appropriate stage for this age
                    appropriate_stage = self.get_stage_by_age(new_age)
                    
                    if appropriate_stage != self.current_stage:
                        # Transition to the appropriate stage
                        logger.info(f"Age-based stage transition triggered: {self.current_stage} -> {appropriate_stage}")
                        self.transition_to_stage(appropriate_stage)
                
                # Invalidate cache
                self._invalidate_cache()
                
                logger.debug(f"Updated age from {old_age:.2f} to {new_age:.2f}")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
                
            error_msg = f"Failed to update age: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, 
                                  current_level=self.trajectory.current_age,
                                  details={"delta_age": delta_age, "original_error": str(e)})
    
    def get_stage_by_age(self, age: float) -> str:
        """
        Get the appropriate stage name for a given age
        
        Args:
            age: The developmental age
            
        Returns:
            The name of the appropriate stage
            
        Raises:
            ValueError: If age is negative
            DevelopmentError: If no appropriate stage is found
        """
        if age < 0:
            raise ValueError("Age cannot be negative")
            
        with self._lock:
            for stage_name, stage in self.stages.items():
                min_age, max_age = stage.age_range
                if min_age <= age <= max_age:
                    return stage_name
                    
            # If we get here, use the highest stage
            sorted_stages = sorted(
                self.stages.items(), 
                key=lambda x: x[1].age_range[1],
                reverse=True
            )
            
            if sorted_stages:
                return sorted_stages[0][0]
                
            # This should never happen if stages are properly defined
            error_msg = f"No appropriate stage found for age {age}"
            logger.error(error_msg)
            raise DevelopmentError(error_msg, details={"age": age})
    
    def get_capability_ceiling(self, capability: str) -> float:
        """
        Get the maximum possible level for a capability at the current stage
        
        Args:
            capability: The capability to get the ceiling for
            
        Returns:
            The maximum possible level (0.0-1.0)
        """
        with self._lock:
            # Check if we can use cached data
            if "capability_ceilings" in self._cache and self._is_cache_valid():
                if capability in self._cache["capability_ceilings"]:
                    return self._cache["capability_ceilings"][capability]
            
            # Get current stage capabilities
            current_capabilities = self.get_current_capabilities()
            
            # If capability is defined in current stage, use that as ceiling
            if capability in current_capabilities:
                ceiling = current_capabilities[capability]
            else:
                # Otherwise, find the highest value for this capability across all stages
                ceiling = 0.0
                for stage in self.stages.values():
                    if capability in stage.capabilities:
                        ceiling = max(ceiling, stage.capabilities[capability])
            
            # Cache the result
            if "capability_ceilings" not in self._cache:
                self._cache["capability_ceilings"] = {}
            self._cache["capability_ceilings"][capability] = ceiling
            
            return ceiling
        
    def get_capability_progression(self, capability: str) -> List[Tuple[str, float]]:
        """
        Get the expected progression of a capability across all developmental stages
        
        Returns a list of (stage_name, level) tuples showing how the capability
        should develop over time.
        """
        progression = []
        for stage_name, stage in sorted(
            self.stages.items(), 
            key=lambda x: x[1].age_range[0]
        ):
            cap_level = stage.capabilities.get(capability, 0.0)
            progression.append((stage_name, cap_level))
        
        return progression
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the developmental stage system"""
        return {
            "current_stage": self.current_stage,
            "age": self.trajectory.current_age,
            "trajectory": self.trajectory.dict(),
            "stages": {name: stage.dict() for name, stage in self.stages.items()}
        }
        
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state"""
        if "current_stage" in state:
            self.current_stage = state["current_stage"]
        
        if "age" in state:
            self.trajectory.current_age = state["age"]
            
        if "trajectory" in state:
            self.trajectory = DevelopmentalTrajectory(**state["trajectory"])
            
        if "stages" in state:
            for name, stage_data in state["stages"].items():
                if name in self.stages:
                    # Update existing stage with saved state
                    self.stages[name] = DevelopmentalStage(**stage_data) 
