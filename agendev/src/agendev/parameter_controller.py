# parameter_controller.py
"""Dynamic adjustment of LLM parameters based on task type."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from enum import Enum
from datetime import datetime
import math
import json
import random
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator

from .models.task_models import Task, TaskType, TaskRisk, TaskPriority
from .llm_integration import LLMConfig
from .utils.fs_utils import resolve_path, load_json, save_json

class ParameterProfile(BaseModel):
    """Configuration profile for a specific task type."""
    task_type: TaskType
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(2000, ge=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=0.0, le=2.0)
    
    # Success tracking
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate for this profile."""
        if self.total_uses == 0:
            return 0.5  # Default to 50% if unused
        return self.success_count / self.total_uses
    
    def to_llm_config(self) -> LLMConfig:
        """Convert to LLMConfig object."""
        return LLMConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def record_outcome(self, success: bool) -> None:
        """Record the outcome of using this profile."""
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

class AdjustmentStrategy(str, Enum):
    """Strategies for parameter adjustment."""
    CONSERVATIVE = "conservative"  # Small, gradual changes
    MODERATE = "moderate"          # Medium changes
    AGGRESSIVE = "aggressive"      # Larger changes, faster adaptation
    EXPLORATION = "exploration"    # Occasionally try very different values

class ParameterController:
    """Controls and adapts LLM parameters based on task characteristics and past performance."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the parameter controller.
        
        Args:
            storage_path: Optional path to store parameter data
        """
        self.storage_path = resolve_path(storage_path or "artifacts/models/parameters.json", create_parents=True)
        
        # Default profiles for each task type
        self.profiles: Dict[TaskType, ParameterProfile] = {}
        
        # Task-specific overrides
        self.task_overrides: Dict[UUID, ParameterProfile] = {}
        
        # Adjustment settings
        self.adjustment_strategy = AdjustmentStrategy.MODERATE
        self.exploration_rate = 0.1  # 10% chance of exploration
        
        # Load existing data
        self._initialize_profiles()
        self._load_profiles()
    
    def _initialize_profiles(self) -> None:
        """Initialize default profiles for all task types."""
        self.profiles = {
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.7,
                max_tokens=2000
            ),
            TaskType.REFACTOR: ParameterProfile(
                task_type=TaskType.REFACTOR,
                temperature=0.5,
                max_tokens=2000
            ),
            TaskType.BUGFIX: ParameterProfile(
                task_type=TaskType.BUGFIX,
                temperature=0.2,
                max_tokens=1500
            ),
            TaskType.TEST: ParameterProfile(
                task_type=TaskType.TEST,
                temperature=0.4,
                max_tokens=2000
            ),
            TaskType.DOCUMENTATION: ParameterProfile(
                task_type=TaskType.DOCUMENTATION,
                temperature=0.8,
                max_tokens=1500
            ),
            TaskType.PLANNING: ParameterProfile(
                task_type=TaskType.PLANNING,
                temperature=0.9,
                max_tokens=3000
            )
        }
    
    def _load_profiles(self) -> None:
        """Load profiles from storage."""
        if not self.storage_path.exists():
            return
            
        try:
            data = load_json(self.storage_path)
            
            # Load profiles
            if "profiles" in data:
                for task_type_str, profile_data in data["profiles"].items():
                    try:
                        task_type = TaskType(task_type_str)
                        self.profiles[task_type] = ParameterProfile.model_validate(profile_data)
                    except Exception as e:
                        print(f"Error loading profile for {task_type_str}: {e}")
            
            # Load task overrides
            if "task_overrides" in data:
                for task_id_str, profile_data in data["task_overrides"].items():
                    try:
                        task_id = UUID(task_id_str)
                        self.task_overrides[task_id] = ParameterProfile.model_validate(profile_data)
                    except Exception as e:
                        print(f"Error loading task override for {task_id_str}: {e}")
            
            # Load adjustment settings
            if "adjustment_strategy" in data:
                try:
                    self.adjustment_strategy = AdjustmentStrategy(data["adjustment_strategy"])
                except:
                    pass
                    
            if "exploration_rate" in data:
                self.exploration_rate = float(data["exploration_rate"])
                
        except Exception as e:
            print(f"Error loading parameter profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles to storage."""
        data = {
            "profiles": {
                task_type.value: profile.model_dump()
                for task_type, profile in self.profiles.items()
            },
            "task_overrides": {
                str(task_id): profile.model_dump()
                for task_id, profile in self.task_overrides.items()
            },
            "adjustment_strategy": self.adjustment_strategy.value,
            "exploration_rate": self.exploration_rate,
            "last_updated": datetime.now().isoformat()
        }
        
        save_json(data, self.storage_path)
    
    def get_profile_for_task(self, task: Task) -> ParameterProfile:
        """
        Get the parameter profile for a specific task.
        
        Args:
            task: The task to get parameters for
            
        Returns:
            Parameter profile for the task
        """
        # Check for task-specific override
        if task.id in self.task_overrides:
            return self.task_overrides[task.id]
            
        # Use task type profile
        if task.task_type in self.profiles:
            profile = self.profiles[task.task_type]
            
            # Apply task-specific adjustments
            adjusted_profile = self._adjust_for_task_properties(profile, task)
            
            # Potentially explore new parameters
            if random.random() < self.exploration_rate and self.adjustment_strategy == AdjustmentStrategy.EXPLORATION:
                adjusted_profile = self._explore_parameters(adjusted_profile)
                
            return adjusted_profile
            
        # Fallback to default
        return ParameterProfile(task_type=task.task_type)
    
    def _adjust_for_task_properties(self, profile: ParameterProfile, task: Task) -> ParameterProfile:
        """
        Adjust parameters based on task properties.
        
        Args:
            profile: Base parameter profile
            task: Task to adjust for
            
        Returns:
            Adjusted parameter profile
        """
        # Create a copy to modify
        adjusted = ParameterProfile.model_validate(profile.model_dump())
        
        # Adjust temperature based on task priority and risk
        if task.priority == TaskPriority.CRITICAL:
            # Lower temperature for critical tasks (more deterministic)
            adjusted.temperature *= 0.8
        elif task.priority == TaskPriority.LOW:
            # Higher temperature for low priority tasks (more creative)
            adjusted.temperature = min(1.0, adjusted.temperature * 1.2)
            
        if task.risk == TaskRisk.HIGH or task.risk == TaskRisk.CRITICAL:
            # Lower temperature for high-risk tasks
            adjusted.temperature *= 0.7
            
        # Adjust based on task complexity
        if task.estimated_complexity > 1.5:
            # Increase max_tokens for complex tasks
            adjusted.max_tokens = int(adjusted.max_tokens * 1.3)
            
        # Ensure parameters are within valid ranges
        adjusted.temperature = max(0.1, min(1.0, adjusted.temperature))
        adjusted.max_tokens = max(500, min(4000, adjusted.max_tokens))
        
        return adjusted
    
    def _explore_parameters(self, profile: ParameterProfile) -> ParameterProfile:
        """
        Explore new parameter values for learning.
        
        Args:
            profile: Base parameter profile
            
        Returns:
            Modified parameter profile with exploration
        """
        # Create a copy
        explored = ParameterProfile.model_validate(profile.model_dump())
        
        # Randomly adjust temperature within a reasonable range
        temp_change = random.uniform(-0.3, 0.3)
        explored.temperature = max(0.1, min(1.0, explored.temperature + temp_change))
        
        # Randomly adjust max_tokens
        token_factor = random.uniform(0.8, 1.5)
        explored.max_tokens = max(500, min(4000, int(explored.max_tokens * token_factor)))
        
        return explored
    
    def record_task_outcome(self, task: Task, success: bool, add_override: bool = False) -> None:
        """
        Record the outcome of a task to improve future parameter selection.
        
        Args:
            task: The task that was completed
            success: Whether the task was successful
            add_override: Whether to add a task-specific override
        """
        # Update the base profile
        if task.task_type in self.profiles:
            self.profiles[task.task_type].record_outcome(success)
            
        # Update task override if it exists
        if task.id in self.task_overrides:
            self.task_overrides[task.id].record_outcome(success)
            
        # Add a new task override if requested
        if add_override and task.id not in self.task_overrides:
            # Create a new profile based on what was likely used
            profile = self.get_profile_for_task(task)
            profile.record_outcome(success)
            self.task_overrides[task.id] = profile
            
        # Save changes
        self._save_profiles()
        
        # Adjust parameters based on outcomes
        self._adapt_parameters()
    
    def _adapt_parameters(self) -> None:
        """Adapt parameters based on outcomes."""
        for task_type, profile in self.profiles.items():
            # Only adapt if we have enough data
            if profile.total_uses < 5:
                continue
                
            # Determine adjustment magnitude based on strategy
            if self.adjustment_strategy == AdjustmentStrategy.CONSERVATIVE:
                magnitude = 0.05
            elif self.adjustment_strategy == AdjustmentStrategy.MODERATE:
                magnitude = 0.1
            elif self.adjustment_strategy == AdjustmentStrategy.AGGRESSIVE:
                magnitude = 0.2
            else:
                magnitude = 0.1
                
            # Calculate success rate
            success_rate = profile.success_rate
            
            # If success rate is poor, adjust parameters
            if success_rate < 0.5:
                # Make more conservative (lower temperature)
                new_temp = profile.temperature - magnitude
                profile.temperature = max(0.1, new_temp)
                
                # Increase max_tokens slightly
                profile.max_tokens = int(profile.max_tokens * 1.1)
            elif success_rate > 0.8 and profile.total_uses > 10:
                # If very successful, we can be slightly more adventurous
                new_temp = profile.temperature + (magnitude / 2)
                profile.temperature = min(1.0, new_temp)
    
    def create_task_specific_profile(self, task: Task, temperature: float, max_tokens: int) -> None:
        """
        Create a task-specific parameter profile.
        
        Args:
            task: The task to create a profile for
            temperature: Temperature value
            max_tokens: Max tokens value
        """
        profile = ParameterProfile(
            task_type=task.task_type,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.task_overrides[task.id] = profile
        self._save_profiles()
    
    def get_llm_config(self, task: Task) -> LLMConfig:
        """
        Get LLMConfig for a task.
        
        Args:
            task: The task to get config for
            
        Returns:
            LLMConfig for the task
        """
        profile = self.get_profile_for_task(task)
        return profile.to_llm_config()
    
    def set_adjustment_strategy(self, strategy: AdjustmentStrategy) -> None:
        """
        Set the adjustment strategy.
        
        Args:
            strategy: Strategy to use
        """
        self.adjustment_strategy = strategy
        self._save_profiles()
    
    def set_exploration_rate(self, rate: float) -> None:
        """
        Set the exploration rate.
        
        Args:
            rate: Exploration rate (0-1)
        """
        self.exploration_rate = max(0.0, min(1.0, rate))
        self._save_profiles()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for different parameters.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            "task_types": {},
            "overall_success_rate": 0.0,
            "total_tasks": 0
        }
        
        total_success = 0
        total_tasks = 0
        
        for task_type, profile in self.profiles.items():
            if profile.total_uses > 0:
                stats["task_types"][task_type.value] = {
                    "success_rate": profile.success_rate,
                    "total_uses": profile.total_uses,
                    "temperature": profile.temperature,
                    "max_tokens": profile.max_tokens
                }
                
                total_success += profile.success_count
                total_tasks += profile.total_uses
        
        if total_tasks > 0:
            stats["overall_success_rate"] = total_success / total_tasks
            
        stats["total_tasks"] = total_tasks
        stats["adjustment_strategy"] = self.adjustment_strategy.value
        stats["exploration_rate"] = self.exploration_rate
        
        return stats