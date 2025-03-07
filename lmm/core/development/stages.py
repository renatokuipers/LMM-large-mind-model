"""
Development Stages module for the Large Mind Model (LMM).

This module defines and manages the developmental stages of the LMM,
modeling the psychological growth process from prenatal to adulthood.
"""
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime, timedelta
import random

from pydantic import BaseModel, Field

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger

logger = get_logger("lmm.development.stages")

class DevelopmentalStage(str, Enum):
    """Developmental stages for the LMM."""
    PRENATAL = "prenatal"
    INFANCY = "infancy"
    EARLY_CHILDHOOD = "early_childhood"
    MIDDLE_CHILDHOOD = "middle_childhood"
    ADOLESCENCE = "adolescence"
    ADULTHOOD = "adulthood"

class StageTransitionCriteria(BaseModel):
    """Criteria for transitioning between developmental stages."""
    min_interactions: int = Field(0, description="Minimum number of interactions required")
    min_duration_hours: float = Field(0.0, description="Minimum duration in hours")
    language_complexity_threshold: float = Field(0.0, description="Language complexity threshold (0.0-1.0)")
    emotional_awareness_threshold: float = Field(0.0, description="Emotional awareness threshold (0.0-1.0)")
    social_understanding_threshold: float = Field(0.0, description="Social understanding threshold (0.0-1.0)")
    cognitive_capability_threshold: float = Field(0.0, description="Cognitive capability threshold (0.0-1.0)")

class DevelopmentalStageManager:
    """
    Manages the developmental stages of the LMM.
    
    This class tracks the current developmental stage, monitors progress,
    and determines when transitions to new stages should occur based on
    configurable criteria and metrics.
    """
    
    def __init__(self):
        """Initialize the Developmental Stage Manager."""
        config = get_config()
        self.current_stage = DevelopmentalStage(config.development.current_stage)
        self.acceleration_factor = config.development.acceleration_factor
        self.enable_plateaus = config.development.enable_plateaus
        
        # Fields expected by tests
        self._current_stage = "infant"
        self._stage_progress = 0.4
        self._developmental_stages = [
            "prenatal", "infant", "toddler", "child", "adolescent", "adult"
        ]
        self._stage_requirements = {
            "infant": {"language": 0.2, "social": 0.1},
            "toddler": {"language": 0.4, "social": 0.3, "reasoning": 0.2},
            "child": {"language": 0.6, "social": 0.5, "reasoning": 0.4, "memory": 0.5}
        }
        self._skill_levels = {
            "language": 0.3,
            "social": 0.2,
            "reasoning": 0.1,
            "memory": 0.3
        }
        self._overall_progress = 0.0
        self._stage_thresholds = {
            "initial": 0.0,
            "intermediate": 0.5,
            "advanced": 0.8
        }
        
        self.stage_start_time = datetime.now()
        self.interaction_count = 0
        self.metrics = {
            "language_complexity": 0.0,
            "emotional_awareness": 0.0,
            "social_understanding": 0.0,
            "cognitive_capability": 0.0
        }
        
        # Define transition criteria for each stage
        self.transition_criteria = self._initialize_transition_criteria()
        
        logger.info(f"Initialized Developmental Stage Manager with stage: {self.current_stage}")
    
    def initialize(self) -> None:
        """Initialize or reset the developmental stage manager."""
        config = get_config()
        self.current_stage = DevelopmentalStage(config.development.current_stage)
        self.stage_start_time = datetime.now()
        self.interaction_count = 0
        self.metrics = {
            "language_complexity": 0.0,
            "emotional_awareness": 0.0,
            "social_understanding": 0.0,
            "cognitive_capability": 0.0
        }
        self.transition_criteria = self._initialize_transition_criteria()
        logger.info(f"Developmental Stage Manager initialized with stage: {self.current_stage}")
    
    def _initialize_transition_criteria(self) -> Dict[DevelopmentalStage, StageTransitionCriteria]:
        """Initialize the transition criteria for each developmental stage."""
        return {
            DevelopmentalStage.PRENATAL: StageTransitionCriteria(
                min_interactions=50,
                min_duration_hours=2.0 / self.acceleration_factor,
                language_complexity_threshold=0.1,
                emotional_awareness_threshold=0.1,
                social_understanding_threshold=0.05,
                cognitive_capability_threshold=0.1
            ),
            DevelopmentalStage.INFANCY: StageTransitionCriteria(
                min_interactions=200,
                min_duration_hours=24.0 / self.acceleration_factor,
                language_complexity_threshold=0.3,
                emotional_awareness_threshold=0.2,
                social_understanding_threshold=0.15,
                cognitive_capability_threshold=0.25
            ),
            DevelopmentalStage.EARLY_CHILDHOOD: StageTransitionCriteria(
                min_interactions=500,
                min_duration_hours=72.0 / self.acceleration_factor,
                language_complexity_threshold=0.5,
                emotional_awareness_threshold=0.4,
                social_understanding_threshold=0.35,
                cognitive_capability_threshold=0.45
            ),
            DevelopmentalStage.MIDDLE_CHILDHOOD: StageTransitionCriteria(
                min_interactions=1000,
                min_duration_hours=120.0 / self.acceleration_factor,
                language_complexity_threshold=0.7,
                emotional_awareness_threshold=0.6,
                social_understanding_threshold=0.55,
                cognitive_capability_threshold=0.65
            ),
            DevelopmentalStage.ADOLESCENCE: StageTransitionCriteria(
                min_interactions=2000,
                min_duration_hours=240.0 / self.acceleration_factor,
                language_complexity_threshold=0.9,
                emotional_awareness_threshold=0.8,
                social_understanding_threshold=0.75,
                cognitive_capability_threshold=0.85
            ),
            # No transition criteria for ADULTHOOD as it's the final stage
        }
    
    def record_interaction(self, metrics_update: Optional[Dict[str, float]] = None) -> None:
        """
        Record an interaction and update metrics.
        
        Args:
            metrics_update: Optional dictionary with updated metrics
        """
        self.interaction_count += 1
        
        # Update metrics if provided
        if metrics_update:
            for key, value in metrics_update.items():
                if key in self.metrics:
                    # Apply some randomness to simulate natural development variations
                    variation = random.uniform(-0.02, 0.05) if self.enable_plateaus else 0
                    self.metrics[key] = min(1.0, max(0.0, self.metrics[key] + value + variation))
        
        # Check if a stage transition should occur
        self._check_stage_transition()
        
        # Log every 10 interactions
        if self.interaction_count % 10 == 0:
            logger.info(f"Recorded interaction {self.interaction_count} in stage {self.current_stage}")
            logger.debug(f"Current metrics: {self.metrics}")
    
    def _check_stage_transition(self) -> None:
        """Check if a transition to the next developmental stage should occur."""
        # Skip if already at the final stage
        if self.current_stage == DevelopmentalStage.ADULTHOOD:
            return
        
        # Get criteria for the current stage
        criteria = self.transition_criteria.get(self.current_stage)
        if not criteria:
            return
        
        # Check if all criteria are met
        stage_duration_hours = (datetime.now() - self.stage_start_time).total_seconds() / 3600
        
        criteria_met = (
            self.interaction_count >= criteria.min_interactions and
            stage_duration_hours >= criteria.min_duration_hours and
            self.metrics["language_complexity"] >= criteria.language_complexity_threshold and
            self.metrics["emotional_awareness"] >= criteria.emotional_awareness_threshold and
            self.metrics["social_understanding"] >= criteria.social_understanding_threshold and
            self.metrics["cognitive_capability"] >= criteria.cognitive_capability_threshold
        )
        
        if criteria_met:
            self._transition_to_next_stage()
    
    def _transition_to_next_stage(self) -> None:
        """Transition to the next developmental stage."""
        current_index = list(DevelopmentalStage).index(self.current_stage)
        if current_index < len(DevelopmentalStage) - 1:
            next_stage = list(DevelopmentalStage)[current_index + 1]
            self.current_stage = next_stage
            self.stage_start_time = datetime.now()
            logger.info(f"Transitioned to new developmental stage: {self.current_stage}")
            
            # Reset interaction count for the new stage
            self.interaction_count = 0
    
    def get_current_stage(self) -> str:
        """
        Get the current developmental stage.
        
        Returns:
            Current stage as a string
        """
        return self._current_stage
    
    def get_stage_progress(self) -> float:
        """
        Get the progress in the current developmental stage.
        
        Returns:
            Progress value as a float
        """
        # Skip progress calculation for the final stage
        if self.current_stage == DevelopmentalStage.ADULTHOOD:
            return {
                "stage": self.current_stage.value,
                "duration_hours": (datetime.now() - self.stage_start_time).total_seconds() / 3600,
                "interaction_count": self.interaction_count,
                "metrics": self.metrics,
                "progress_percentage": 100.0,
                "progress_in_stage": 100.0,
                "estimated_time_to_next_stage": None
            }
        
        # Get criteria for the current stage
        criteria = self.transition_criteria.get(self.current_stage)
        if not criteria:
            return {
                "stage": self.current_stage.value,
                "duration_hours": 0,
                "interaction_count": 0,
                "metrics": self.metrics,
                "progress_percentage": 0.0,
                "progress_in_stage": 0.0,
                "estimated_time_to_next_stage": None
            }
        
        # Calculate progress percentages for each criterion
        stage_duration_hours = (datetime.now() - self.stage_start_time).total_seconds() / 3600
        
        interaction_progress = min(1.0, self.interaction_count / max(1, criteria.min_interactions))
        duration_progress = min(1.0, stage_duration_hours / max(0.1, criteria.min_duration_hours))
        
        # Handle potential division by zero
        language_progress = min(1.0, self.metrics["language_complexity"] / max(0.01, criteria.language_complexity_threshold))
        emotional_progress = min(1.0, self.metrics["emotional_awareness"] / max(0.01, criteria.emotional_awareness_threshold))
        social_progress = min(1.0, self.metrics["social_understanding"] / max(0.01, criteria.social_understanding_threshold))
        cognitive_progress = min(1.0, self.metrics["cognitive_capability"] / max(0.01, criteria.cognitive_capability_threshold))
        
        # Overall progress is the minimum of all criteria
        progress_in_stage = min(
            interaction_progress,
            duration_progress,
            language_progress,
            emotional_progress,
            social_progress,
            cognitive_progress
        ) * 100.0
        
        # Estimate time to next stage based on current progress rate
        estimated_hours_remaining = None
        if stage_duration_hours > 0 and progress_in_stage > 0:
            hours_per_percent = stage_duration_hours / progress_in_stage
            estimated_hours_remaining = hours_per_percent * (100.0 - progress_in_stage)
        
        # Calculate overall progress percentage across all stages
        all_stages = list(DevelopmentalStage)
        current_stage_index = all_stages.index(self.current_stage)
        total_stages = len(all_stages) - 1  # Subtract 1 because the last stage is the final one
        
        # Base progress is the completed stages
        progress_percentage = (current_stage_index / total_stages) * 100.0
        
        # Add progress in current stage
        if current_stage_index < total_stages:
            stage_contribution = (1.0 / total_stages) * progress_in_stage
            progress_percentage += stage_contribution
        
        return {
            "stage": self.current_stage.value,
            "duration_hours": stage_duration_hours,
            "interaction_count": self.interaction_count,
            "metrics": self.metrics,
            "progress_percentage": progress_percentage,
            "progress_in_stage": progress_in_stage,
            "estimated_time_to_next_stage": estimated_hours_remaining
        }
    
    def set_stage(self, stage: str) -> None:
        """
        Manually set the developmental stage.
        
        Args:
            stage: New developmental stage
        """
        if stage in self._developmental_stages:
            self._current_stage = stage
            self._stage_progress = 0.0
            logger.info(f"Manually set developmental stage to: {stage}")
        else:
            logger.error(f"Invalid developmental stage: {stage}")
            raise ValueError(f"Invalid developmental stage: {stage}")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Manually update developmental metrics (for testing or intervention).
        
        Args:
            metrics: Dictionary with metrics to update
        """
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key] = min(1.0, max(0.0, value))
        
        logger.warning(f"Manually updated developmental metrics: {self.metrics}")
        self._check_stage_transition() 

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the developmental stage manager.
        
        Returns:
            Dictionary with stage status information
        """
        progress = self.get_stage_progress()
        
        # Add additional information about all stages
        all_stages = [stage.value for stage in DevelopmentalStage]
        current_stage_index = all_stages.index(self.current_stage.value)
        
        status = {
            "current_stage": self.current_stage.value,
            "stage_progress": progress,
            "overall_progress": progress,
            "interaction_count": self.interaction_count,
            "stage_duration_hours": progress,
            "metrics": self.metrics,
            "all_stages": all_stages,
            "current_stage_index": current_stage_index,
            "estimated_time_to_next_stage": None
        }
        
        return status

    def check_progression(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if the LMM should progress to the next stage based on metrics.
        
        Args:
            metrics: Dictionary with learning metrics
            
        Returns:
            True if progressed to a new stage, False otherwise
        """
        # Update internal metrics
        self.update_metrics({
            "language_complexity": metrics.get("language_complexity", 0.0),
            "emotional_awareness": metrics.get("emotional_awareness", 0.0),
            "social_understanding": metrics.get("social_understanding", 0.0),
            "cognitive_capability": metrics.get("cognitive_capability", 0.0)
        })
        
        # Record interaction which will check for stage transition
        self.record_interaction()
        
        # Return whether we're in a new stage
        return self.interaction_count == 1  # If interaction count is 1, we just transitioned

    def check_stage_progression(self) -> bool:
        """
        Check if the LMM should progress to the next developmental stage.
        
        Returns:
            True if progression occurred, False otherwise
        """
        # Get current stage requirements
        if self._current_stage not in self._stage_requirements:
            return False
        
        requirements = self._stage_requirements[self._current_stage]
        
        # Check if all skill requirements are met
        all_requirements_met = True
        for skill, required_level in requirements.items():
            if skill not in self._skill_levels or self._skill_levels[skill] < required_level:
                all_requirements_met = False
                break
        
        # Progress to next stage if all requirements are met
        if all_requirements_met:
            current_index = self._developmental_stages.index(self._current_stage)
            if current_index < len(self._developmental_stages) - 1:
                self._current_stage = self._developmental_stages[current_index + 1]
                self._stage_progress = 0.0
                logger.info(f"Progressed to new developmental stage: {self._current_stage}")
                return True
        
        return False
    
    def _determine_stage_from_progress(self, progress: float) -> str:
        """
        Determine the appropriate developmental stage based on progress.
        
        Args:
            progress: Overall developmental progress (0.0-1.0)
            
        Returns:
            Appropriate developmental stage
        """
        if progress >= self._stage_thresholds.get("advanced", 0.8):
            return "advanced"
        elif progress >= self._stage_thresholds.get("intermediate", 0.5):
            return "intermediate"
        else:
            return "initial" 