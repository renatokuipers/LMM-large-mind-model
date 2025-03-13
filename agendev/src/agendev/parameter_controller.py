# parameter_controller.py
"""Dynamic adjustment of LLM parameters based on task type and execution outcomes."""

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
    
    # Performance tracking
    avg_execution_time: float = 0.0
    avg_tokens_used: float = 0.0
    
    # Profile metadata
    profile_type: str = "default"  # e.g., "web_app", "cli_utility", "implementation", "documentation"
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
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
    
    def record_outcome(self, success: bool, execution_time: float = 0.0, tokens_used: float = 0.0) -> None:
        """
        Record the outcome of using this profile.
        
        Args:
            success: Whether the task was successful
            execution_time: Execution time in seconds
            tokens_used: Number of tokens used
        """
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update performance metrics using exponential moving average
        if execution_time > 0:
            if self.avg_execution_time == 0:
                self.avg_execution_time = execution_time
            else:
                # Use a weight of 0.2 for the new value, 0.8 for the history
                self.avg_execution_time = (0.8 * self.avg_execution_time) + (0.2 * execution_time)
        
        if tokens_used > 0:
            if self.avg_tokens_used == 0:
                self.avg_tokens_used = tokens_used
            else:
                # Use a weight of 0.2 for the new value, 0.8 for the history
                self.avg_tokens_used = (0.8 * self.avg_tokens_used) + (0.2 * tokens_used)
        
        # Update timestamp
        self.updated_at = datetime.now()

class AdjustmentStrategy(str, Enum):
    """Strategies for parameter adjustment."""
    CONSERVATIVE = "conservative"  # Small, gradual changes
    MODERATE = "moderate"          # Medium changes
    AGGRESSIVE = "aggressive"      # Larger changes, faster adaptation
    EXPLORATION = "exploration"    # Occasionally try very different values

class ProjectType(str, Enum):
    """Types of projects for specialized parameter profiles."""
    WEB_APP = "web_app"           # Web application projects
    CLI_UTILITY = "cli_utility"    # Command line interface utilities
    API_SERVICE = "api_service"    # API service projects
    DATA_PIPELINE = "data_pipeline"  # Data processing pipeline projects
    GENERAL = "general"            # General or unspecified project type

class ContentType(str, Enum):
    """Types of content generation for specialized parameter profiles."""
    IMPLEMENTATION = "implementation"  # Code implementation
    DOCUMENTATION = "documentation"    # Documentation generation
    TESTING = "testing"                # Test generation
    DEBUGGING = "debugging"            # Debugging and error fixing
    PLANNING = "planning"              # Architecture and planning

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
        
        # Project-specific profiles with specialized settings
        self.project_profiles: Dict[ProjectType, Dict[TaskType, ParameterProfile]] = {}
        
        # Content-specific profiles
        self.content_profiles: Dict[ContentType, Dict[TaskType, ParameterProfile]] = {}
        
        # Task-specific overrides
        self.task_overrides: Dict[UUID, ParameterProfile] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Adjustment settings
        self.adjustment_strategy = AdjustmentStrategy.MODERATE
        self.exploration_rate = 0.1  # 10% chance of exploration
        
        # Current project type
        self.current_project_type = ProjectType.GENERAL
        
        # Load existing data
        self._initialize_profiles()
        self._initialize_project_profiles()
        self._initialize_content_profiles()
        self._load_profiles()
    
    def _initialize_profiles(self) -> None:
        """Initialize default profiles for all task types."""
        self.profiles = {
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.7,
                max_tokens=2000,
                description="Default implementation profile for general code generation"
            ),
            TaskType.REFACTOR: ParameterProfile(
                task_type=TaskType.REFACTOR,
                temperature=0.5,
                max_tokens=2000,
                description="Default refactoring profile for code restructuring"
            ),
            TaskType.BUGFIX: ParameterProfile(
                task_type=TaskType.BUGFIX,
                temperature=0.2,
                max_tokens=1500,
                description="Default bugfix profile for error correction"
            ),
            TaskType.TEST: ParameterProfile(
                task_type=TaskType.TEST,
                temperature=0.4,
                max_tokens=2000,
                description="Default test profile for generating test cases"
            ),
            TaskType.DOCUMENTATION: ParameterProfile(
                task_type=TaskType.DOCUMENTATION,
                temperature=0.8,
                max_tokens=1500,
                description="Default documentation profile for generating comments and docs"
            ),
            TaskType.PLANNING: ParameterProfile(
                task_type=TaskType.PLANNING,
                temperature=0.9,
                max_tokens=3000,
                description="Default planning profile for architecture and design"
            )
        }
    
    def _initialize_project_profiles(self) -> None:
        """Initialize project-specific profiles."""
        # Web Application profiles
        web_app_profiles = {
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.6,
                max_tokens=2500,
                profile_type="web_app",
                description="Web application code implementation with focus on security and standards compliance"
            ),
            TaskType.REFACTOR: ParameterProfile(
                task_type=TaskType.REFACTOR,
                temperature=0.4,
                max_tokens=2000,
                profile_type="web_app",
                description="Web application code refactoring with focus on performance optimization"
            ),
            TaskType.BUGFIX: ParameterProfile(
                task_type=TaskType.BUGFIX,
                temperature=0.2,
                max_tokens=1500,
                profile_type="web_app",
                description="Web application bug fixing with additional security checks"
            ),
            TaskType.TEST: ParameterProfile(
                task_type=TaskType.TEST,
                temperature=0.3,
                max_tokens=2500,
                profile_type="web_app",
                description="Web application test generation with focus on edge cases and security tests"
            ),
            TaskType.DOCUMENTATION: ParameterProfile(
                task_type=TaskType.DOCUMENTATION,
                temperature=0.7,
                max_tokens=2000,
                profile_type="web_app",
                description="Web application documentation with API references and usage examples"
            ),
            TaskType.PLANNING: ParameterProfile(
                task_type=TaskType.PLANNING,
                temperature=0.8,
                max_tokens=3000,
                profile_type="web_app",
                description="Web application planning with focus on architecture and scalability"
            )
        }
        
        # CLI Utility profiles
        cli_utility_profiles = {
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.5,
                max_tokens=2000,
                profile_type="cli_utility",
                description="CLI utility implementation with focus on robustness and error handling"
            ),
            TaskType.REFACTOR: ParameterProfile(
                task_type=TaskType.REFACTOR,
                temperature=0.4,
                max_tokens=1800,
                profile_type="cli_utility",
                description="CLI utility refactoring with focus on performance and usability"
            ),
            TaskType.BUGFIX: ParameterProfile(
                task_type=TaskType.BUGFIX,
                temperature=0.2,
                max_tokens=1500,
                profile_type="cli_utility",
                description="CLI utility bug fixing with focus on input validation"
            ),
            TaskType.TEST: ParameterProfile(
                task_type=TaskType.TEST,
                temperature=0.3,
                max_tokens=2000,
                profile_type="cli_utility",
                description="CLI utility test generation with focus on command line args and exit codes"
            ),
            TaskType.DOCUMENTATION: ParameterProfile(
                task_type=TaskType.DOCUMENTATION,
                temperature=0.7,
                max_tokens=1500,
                profile_type="cli_utility",
                description="CLI utility documentation with usage examples and command references"
            ),
            TaskType.PLANNING: ParameterProfile(
                task_type=TaskType.PLANNING,
                temperature=0.7,
                max_tokens=2500,
                profile_type="cli_utility",
                description="CLI utility planning with focus on user interface and workflow"
            )
        }
        
        # Store the project profiles
        self.project_profiles = {
            ProjectType.WEB_APP: web_app_profiles,
            ProjectType.CLI_UTILITY: cli_utility_profiles
        }
    
    def _initialize_content_profiles(self) -> None:
        """Initialize content-specific profiles."""
        # Implementation-focused profiles
        implementation_profiles = {
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.5,
                max_tokens=2500,
                profile_type="implementation",
                description="Pure implementation profile optimized for clean code generation"
            ),
            TaskType.REFACTOR: ParameterProfile(
                task_type=TaskType.REFACTOR,
                temperature=0.4,
                max_tokens=2200,
                profile_type="implementation",
                description="Refactoring profile with focus on code quality improvements"
            ),
            TaskType.BUGFIX: ParameterProfile(
                task_type=TaskType.BUGFIX,
                temperature=0.2,
                max_tokens=1800,
                profile_type="implementation",
                description="Bug fixing profile with focus on technical correctness"
            ),
            TaskType.TEST: ParameterProfile(
                task_type=TaskType.TEST,
                temperature=0.3,
                max_tokens=2200,
                profile_type="implementation",
                description="Technical test generation profile with thorough test coverage"
            ),
        }
        
        # Documentation-focused profiles
        documentation_profiles = {
            TaskType.DOCUMENTATION: ParameterProfile(
                task_type=TaskType.DOCUMENTATION,
                temperature=0.8,
                max_tokens=2500,
                profile_type="documentation",
                description="Comprehensive documentation optimized for clarity and completeness"
            ),
            TaskType.IMPLEMENTATION: ParameterProfile(
                task_type=TaskType.IMPLEMENTATION,
                temperature=0.6,
                max_tokens=2000,
                profile_type="documentation",
                description="Implementation with focus on self-documenting code and comments"
            ),
            TaskType.PLANNING: ParameterProfile(
                task_type=TaskType.PLANNING,
                temperature=0.85,
                max_tokens=3500,
                profile_type="documentation",
                description="Planning documentation with focus on clarity and audience understanding"
            ),
        }
        
        # Store the content profiles
        self.content_profiles = {
            ContentType.IMPLEMENTATION: implementation_profiles,
            ContentType.DOCUMENTATION: documentation_profiles
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
            
            # Load project profiles
            if "project_profiles" in data:
                for project_type_str, project_data in data["project_profiles"].items():
                    try:
                        project_type = ProjectType(project_type_str)
                        if project_type not in self.project_profiles:
                            self.project_profiles[project_type] = {}
                        
                        for task_type_str, profile_data in project_data.items():
                            task_type = TaskType(task_type_str)
                            self.project_profiles[project_type][task_type] = ParameterProfile.model_validate(profile_data)
                    except Exception as e:
                        print(f"Error loading project profile for {project_type_str}: {e}")
            
            # Load content profiles
            if "content_profiles" in data:
                for content_type_str, content_data in data["content_profiles"].items():
                    try:
                        content_type = ContentType(content_type_str)
                        if content_type not in self.content_profiles:
                            self.content_profiles[content_type] = {}
                        
                        for task_type_str, profile_data in content_data.items():
                            task_type = TaskType(task_type_str)
                            self.content_profiles[content_type][task_type] = ParameterProfile.model_validate(profile_data)
                    except Exception as e:
                        print(f"Error loading content profile for {content_type_str}: {e}")
            
            # Load task overrides
            if "task_overrides" in data:
                for task_id_str, profile_data in data["task_overrides"].items():
                    try:
                        task_id = UUID(task_id_str)
                        self.task_overrides[task_id] = ParameterProfile.model_validate(profile_data)
                    except Exception as e:
                        print(f"Error loading task override for {task_id_str}: {e}")
            
            # Load performance history
            if "performance_history" in data:
                self.performance_history = data["performance_history"]
            
            # Load adjustment settings
            if "adjustment_strategy" in data:
                try:
                    self.adjustment_strategy = AdjustmentStrategy(data["adjustment_strategy"])
                except:
                    pass
                    
            if "exploration_rate" in data:
                self.exploration_rate = float(data["exploration_rate"])
                
            # Load current project type
            if "current_project_type" in data:
                try:
                    self.current_project_type = ProjectType(data["current_project_type"])
                except:
                    pass
                
        except Exception as e:
            print(f"Error loading parameter profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles to storage."""
        data = {
            "profiles": {
                task_type.value: profile.model_dump()
                for task_type, profile in self.profiles.items()
            },
            "project_profiles": {
                project_type.value: {
                    task_type.value: profile.model_dump()
                    for task_type, profile in profiles.items()
                }
                for project_type, profiles in self.project_profiles.items()
            },
            "content_profiles": {
                content_type.value: {
                    task_type.value: profile.model_dump()
                    for task_type, profile in profiles.items()
                }
                for content_type, profiles in self.content_profiles.items()
            },
            "task_overrides": {
                str(task_id): profile.model_dump()
                for task_id, profile in self.task_overrides.items()
            },
            "performance_history": self.performance_history[-100:],  # Keep only the last 100 entries
            "adjustment_strategy": self.adjustment_strategy.value,
            "exploration_rate": self.exploration_rate,
            "current_project_type": self.current_project_type.value,
            "last_updated": datetime.now().isoformat()
        }
        
        save_json(data, self.storage_path)
    
    def get_profile_for_task(self, task: Task, content_type: Optional[ContentType] = None) -> ParameterProfile:
        """
        Get the parameter profile for a specific task.
        
        Args:
            task: The task to get parameters for
            content_type: Optional content type to prioritize specific parameters
            
        Returns:
            Parameter profile for the task
        """
        # Check for task-specific override
        if task.id in self.task_overrides:
            return self.task_overrides[task.id]
        
        # Start with the base profile for this task type
        profile = None
        
        # Check for project-specific profiles first
        if self.current_project_type in self.project_profiles and task.task_type in self.project_profiles[self.current_project_type]:
            profile = self.project_profiles[self.current_project_type][task.task_type]
        
        # Check for content-specific profile if specified
        elif content_type is not None and content_type in self.content_profiles and task.task_type in self.content_profiles[content_type]:
            profile = self.content_profiles[content_type][task.task_type]
        
        # Fall back to default profile
        elif task.task_type in self.profiles:
            profile = self.profiles[task.task_type]
        else:
            # Final fallback is a new default profile
            return ParameterProfile(task_type=task.task_type)
            
        # Make a copy to avoid modifying the original
        adjusted_profile = ParameterProfile.model_validate(profile.model_dump())
        
        # Apply task-specific adjustments
        adjusted_profile = self._adjust_for_task_properties(adjusted_profile, task)
        
        # Potentially explore new parameters
        if random.random() < self.exploration_rate and self.adjustment_strategy == AdjustmentStrategy.EXPLORATION:
            adjusted_profile = self._explore_parameters(adjusted_profile)
            
        return adjusted_profile
    
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
        
        # Indicate that this is an exploration profile
        explored.profile_type = f"{profile.profile_type}_exploration"
        explored.description = f"Exploration variant of {profile.profile_type} profile"
        
        return explored
    
    def record_task_outcome(self, 
                           task: Task, 
                           success: bool, 
                           add_override: bool = False, 
                           execution_time: float = 0.0, 
                           tokens_used: float = 0.0) -> None:
        """
        Record the outcome of a task to improve future parameter selection.
        
        Args:
            task: The task that was completed
            success: Whether the task was successful
            add_override: Whether to add a task-specific override
            execution_time: Execution time in seconds
            tokens_used: Number of tokens used
        """
        # Update the base profile
        if task.task_type in self.profiles:
            self.profiles[task.task_type].record_outcome(success, execution_time, tokens_used)
            
        # Update project profile if applicable
        if (self.current_project_type in self.project_profiles and 
            task.task_type in self.project_profiles[self.current_project_type]):
            self.project_profiles[self.current_project_type][task.task_type].record_outcome(
                success, execution_time, tokens_used
            )
            
        # Update task override if it exists
        if task.id in self.task_overrides:
            self.task_overrides[task.id].record_outcome(success, execution_time, tokens_used)
            
        # Add a new task override if requested
        if add_override and task.id not in self.task_overrides:
            # Create a new profile based on what was likely used
            profile = self.get_profile_for_task(task)
            profile.record_outcome(success, execution_time, tokens_used)
            self.task_overrides[task.id] = profile
            
        # Add to performance history
        self.performance_history.append({
            "task_id": str(task.id),
            "task_type": task.task_type.value,
            "project_type": self.current_project_type.value,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "temperature": task.temperature,
            "max_tokens": task.max_tokens,
            "execution_time": execution_time,
            "tokens_used": tokens_used
        })
            
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
                
            # Update timestamp
            profile.updated_at = datetime.now()
                
        # Also adapt project-specific profiles if we have enough data
        for project_type, profiles in self.project_profiles.items():
            for task_type, profile in profiles.items():
                if profile.total_uses < 5:
                    continue
                    
                # Same adjustment logic as above
                if self.adjustment_strategy == AdjustmentStrategy.CONSERVATIVE:
                    magnitude = 0.05
                elif self.adjustment_strategy == AdjustmentStrategy.MODERATE:
                    magnitude = 0.1
                elif self.adjustment_strategy == AdjustmentStrategy.AGGRESSIVE:
                    magnitude = 0.2
                else:
                    magnitude = 0.1
                    
                success_rate = profile.success_rate
                
                if success_rate < 0.5:
                    new_temp = profile.temperature - magnitude
                    profile.temperature = max(0.1, new_temp)
                    profile.max_tokens = int(profile.max_tokens * 1.1)
                elif success_rate > 0.8 and profile.total_uses > 10:
                    new_temp = profile.temperature + (magnitude / 2)
                    profile.temperature = min(1.0, new_temp)
                    
                # Update timestamp
                profile.updated_at = datetime.now()
    
    def set_project_type(self, project_type: ProjectType) -> None:
        """
        Set the current project type to use specialized profiles.
        
        Args:
            project_type: Project type to use
        """
        self.current_project_type = project_type
        self._save_profiles()
    
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
            max_tokens=max_tokens,
            profile_type="task_specific",
            description=f"Custom profile for task: {task.title}"
        )
        
        self.task_overrides[task.id] = profile
        self._save_profiles()
    
    def get_llm_config(self, task: Task, content_type: Optional[ContentType] = None) -> LLMConfig:
        """
        Get LLMConfig for a task.
        
        Args:
            task: The task to get config for
            content_type: Optional content type to prioritize specific parameters
            
        Returns:
            LLMConfig for the task
        """
        profile = self.get_profile_for_task(task, content_type)
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
            "project_types": {},
            "content_types": {},
            "overall_success_rate": 0.0,
            "total_tasks": 0,
            "performance_trends": self._calculate_performance_trends()
        }
        
        total_success = 0
        total_tasks = 0
        
        # Basic task type stats
        for task_type, profile in self.profiles.items():
            if profile.total_uses > 0:
                stats["task_types"][task_type.value] = {
                    "success_rate": profile.success_rate,
                    "total_uses": profile.total_uses,
                    "temperature": profile.temperature,
                    "max_tokens": profile.max_tokens,
                    "avg_execution_time": profile.avg_execution_time,
                    "avg_tokens_used": profile.avg_tokens_used
                }
                
                total_success += profile.success_count
                total_tasks += profile.total_uses
        
        # Project type stats
        for project_type, profiles in self.project_profiles.items():
            project_stats = {
                "total_uses": 0,
                "success_count": 0,
                "task_types": {}
            }
            
            for task_type, profile in profiles.items():
                if profile.total_uses > 0:
                    project_stats["task_types"][task_type.value] = {
                        "success_rate": profile.success_rate,
                        "total_uses": profile.total_uses,
                        "temperature": profile.temperature,
                        "max_tokens": profile.max_tokens,
                        "avg_execution_time": profile.avg_execution_time,
                        "avg_tokens_used": profile.avg_tokens_used
                    }
                    project_stats["total_uses"] += profile.total_uses
                    project_stats["success_count"] += profile.success_count
            
            if project_stats["total_uses"] > 0:
                project_stats["success_rate"] = project_stats["success_count"] / project_stats["total_uses"]
                stats["project_types"][project_type.value] = project_stats
                
        # Content type stats
        for content_type, profiles in self.content_profiles.items():
            content_stats = {
                "total_uses": 0,
                "success_count": 0,
                "task_types": {}
            }
            
            for task_type, profile in profiles.items():
                if profile.total_uses > 0:
                    content_stats["task_types"][task_type.value] = {
                        "success_rate": profile.success_rate,
                        "total_uses": profile.total_uses,
                        "temperature": profile.temperature,
                        "max_tokens": profile.max_tokens,
                        "avg_execution_time": profile.avg_execution_time,
                        "avg_tokens_used": profile.avg_tokens_used
                    }
                    content_stats["total_uses"] += profile.total_uses
                    content_stats["success_count"] += profile.success_count
            
            if content_stats["total_uses"] > 0:
                content_stats["success_rate"] = content_stats["success_count"] / content_stats["total_uses"]
                stats["content_types"][content_type.value] = content_stats
        
        if total_tasks > 0:
            stats["overall_success_rate"] = total_success / total_tasks
            
        stats["total_tasks"] = total_tasks
        stats["adjustment_strategy"] = self.adjustment_strategy.value
        stats["exploration_rate"] = self.exploration_rate
        stats["current_project_type"] = self.current_project_type.value
        
        return stats
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """
        Calculate performance trends over time.
        
        Returns:
            Dictionary with performance trend data
        """
        # Ensure we have enough data
        if len(self.performance_history) < 5:
            return {
                "success_rate_trend": "stable",
                "execution_time_trend": "stable",
                "tokens_used_trend": "stable"
            }
        
        # Divide history into windows
        window_size = min(10, len(self.performance_history) // 2)
        recent_window = self.performance_history[-window_size:]
        older_window = self.performance_history[-2*window_size:-window_size]
        
        # Calculate success rate trend
        recent_success_rate = sum(1 for entry in recent_window if entry.get("success", False)) / window_size
        older_success_rate = sum(1 for entry in older_window if entry.get("success", False)) / window_size
        
        success_diff = recent_success_rate - older_success_rate
        success_trend = "improving" if success_diff > 0.1 else "declining" if success_diff < -0.1 else "stable"
        
        # Calculate execution time trend
        recent_exec_times = [entry.get("execution_time", 0) for entry in recent_window]
        older_exec_times = [entry.get("execution_time", 0) for entry in older_window]
        
        recent_avg_exec_time = sum(recent_exec_times) / window_size if recent_exec_times else 0
        older_avg_exec_time = sum(older_exec_times) / window_size if older_exec_times else 0
        
        # Lower time is better, so comparison is inverted
        exec_time_diff = older_avg_exec_time - recent_avg_exec_time
        exec_time_trend = "improving" if exec_time_diff > 0.5 else "declining" if exec_time_diff < -0.5 else "stable"
        
        # Calculate token usage trend
        recent_token_usage = [entry.get("tokens_used", 0) for entry in recent_window]
        older_token_usage = [entry.get("tokens_used", 0) for entry in older_window]
        
        recent_avg_tokens = sum(recent_token_usage) / window_size if recent_token_usage else 0
        older_avg_tokens = sum(older_token_usage) / window_size if older_token_usage else 0
        
        # Lower token usage is better, so comparison is inverted
        token_diff = older_avg_tokens - recent_avg_tokens
        token_trend = "improving" if token_diff > 50 else "declining" if token_diff < -50 else "stable"
        
        return {
            "success_rate_trend": success_trend,
            "execution_time_trend": exec_time_trend,
            "tokens_used_trend": token_trend,
            "recent_success_rate": recent_success_rate,
            "older_success_rate": older_success_rate,
            "recent_avg_exec_time": recent_avg_exec_time,
            "older_avg_exec_time": older_avg_exec_time,
            "recent_avg_tokens": recent_avg_tokens,
            "older_avg_tokens": older_avg_tokens
        }