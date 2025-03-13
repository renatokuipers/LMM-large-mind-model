# task_models.py
"""Pydantic models for tasks, epics, and dependencies."""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Literal, Any
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class TaskStatus(str, Enum):
    """Status states for tasks and epics."""
    PLANNED = "planned"             # Initial state for a task that's been created but not started
    IN_PROGRESS = "in_progress"     # Task is currently being worked on
    BLOCKED = "blocked"             # Task is blocked by dependencies
    WAITING_REVIEW = "waiting_review"  # Task implementation is complete but awaiting review
    BEING_REVISED = "being_revised"  # Task is being revised after review feedback
    TESTING = "testing"             # Task implementation is complete and under testing
    TEST_FAILED = "test_failed"     # Task failed its tests
    COMPLETED = "completed"         # Task has been successfully completed
    FAILED = "failed"               # Task failed to be completed
    CANCELLED = "cancelled"         # Task was cancelled
    PAUSED = "paused"               # Task work has been temporarily suspended

class TaskPriority(str, Enum):
    """Priority levels for tasks and epics."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskRisk(str, Enum):
    """Risk levels for tasks and epics."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    """Types of tasks to support different parameter strategies."""
    IMPLEMENTATION = "implementation"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    TEST = "test"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"

class ErrorCategory(str, Enum):
    """Categories of errors that can occur during task execution."""
    LLM_SERVICE = "llm_service"     # Issues with the LLM service (network, timeout)
    LLM_RESPONSE = "llm_response"   # Issues with the LLM response (malformed, incomplete)
    CODE_QUALITY = "code_quality"   # Issues with the generated code (syntax errors, bugs)
    FILESYSTEM = "filesystem"       # Issues with file operations
    DEPENDENCY = "dependency"       # Issues with task dependencies
    RESOURCE = "resource"           # Resource constraints (memory, CPU)
    UNKNOWN = "unknown"             # Unclassified errors

class Dependency(BaseModel):
    """Represents a dependency between tasks."""
    source_id: UUID
    target_id: UUID
    dependency_type: Literal["blocks", "influences", "implements"] = "blocks"
    strength: float = Field(1.0, ge=0.0, le=1.0, description="How strong the dependency is (0.0-1.0)")
    
    @field_validator("source_id", "target_id")
    @classmethod
    def validate_not_same(cls, v, info):
        """Ensure source and target are not the same."""
        if info.data.get("source_id") == info.data.get("target_id") == v:
            raise ValueError("Source and target tasks cannot be the same")
        return v

class ExecutionAttempt(BaseModel):
    """Records a single execution attempt of a task."""
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    duration_seconds: float = 0.0
    llm_parameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_paths: List[str] = Field(default_factory=list)
    snapshot_id: Optional[str] = None
    context_used: bool = False
    context_elements: int = 0
    retry_count: int = 0

class TaskStatistics(BaseModel):
    """Statistics tracking for task execution."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_duration_seconds: float = 0.0
    last_attempt_timestamp: Optional[datetime] = None
    error_counts: Dict[str, int] = Field(default_factory=dict)
    execution_attempts: List[ExecutionAttempt] = Field(default_factory=list)
    
    def record_attempt(self, attempt: ExecutionAttempt) -> None:
        """Record a new execution attempt."""
        self.total_attempts += 1
        if attempt.success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if attempt.error_category:
                category = attempt.error_category.value
                self.error_counts[category] = self.error_counts.get(category, 0) + 1
                
        self.total_duration_seconds += attempt.duration_seconds
        self.last_attempt_timestamp = attempt.timestamp
        self.execution_attempts.append(attempt)
    
    def get_success_rate(self) -> float:
        """Get the success rate of attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    def get_average_duration(self) -> float:
        """Get the average duration of successful attempts."""
        successful_attempts = [a for a in self.execution_attempts if a.success]
        if not successful_attempts:
            return 0.0
        return sum(a.duration_seconds for a in successful_attempts) / len(successful_attempts)
    
    def get_most_common_error(self) -> Optional[str]:
        """Get the most common error category."""
        if not self.error_counts:
            return None
        return max(self.error_counts.items(), key=lambda x: x[1])[0]

class BaseTask(BaseModel):
    """Base model with common fields for tasks and epics."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    status: TaskStatus = TaskStatus.PLANNED
    priority: TaskPriority = TaskPriority.MEDIUM
    risk: TaskRisk = TaskRisk.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Metadata for tracking
    completion_probability: float = Field(0.5, ge=0.0, le=1.0)
    estimated_complexity: float = Field(1.0, ge=0.1)
    estimated_duration_hours: float = Field(1.0, ge=0.0)
    actual_duration_hours: Optional[float] = None
    
    # Task statistics
    statistics: TaskStatistics = Field(default_factory=TaskStatistics)
    
    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_updated(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def record_execution_attempt(self, attempt: ExecutionAttempt) -> None:
        """Record a task execution attempt."""
        self.statistics.record_attempt(attempt)
        self.mark_updated()
    
    def transition_status(self, new_status: TaskStatus) -> None:
        """
        Transition task to a new status with validation.
        
        Args:
            new_status: The new status to transition to
        """
        # Record the transition in metadata
        if "status_history" not in self.metadata:
            self.metadata["status_history"] = []
        
        self.metadata["status_history"].append({
            "from": self.status.value,
            "to": new_status.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update the status
        self.status = new_status
        self.mark_updated()

class Task(BaseTask):
    """Represents a specific unit of work in the system."""
    epic_id: Optional[UUID] = None
    task_type: TaskType = TaskType.IMPLEMENTATION
    parent_task_id: Optional[UUID] = None
    subtasks: List[UUID] = Field(default_factory=list)
    dependencies: List[UUID] = Field(default_factory=list, description="Tasks that this task depends on")
    dependents: List[UUID] = Field(default_factory=list, description="Tasks that depend on this task")
    artifact_paths: List[str] = Field(default_factory=list, description="Paths to artifacts produced by this task")
    
    # LLM parameters to use for this task
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(2000, ge=0)
    
    # Context and progress tracking
    context_ids: List[str] = Field(default_factory=list, description="IDs of context elements relevant to this task")
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    # Retry configuration
    max_retry_attempts: int = Field(3, ge=0, description="Maximum number of retry attempts for this task")
    current_retry_count: int = Field(0, ge=0, description="Current number of retry attempts for this task")
    
    @model_validator(mode='after')
    def validate_task_structure(self) -> 'Task':
        """Validate the task structure relationships."""
        if self.parent_task_id is not None and self.epic_id is not None:
            raise ValueError("A task cannot have both a parent task and an epic")
        return self
    
    def calculate_completion_percentage(self) -> float:
        """Calculate completion percentage based on status."""
        status_percentages = {
            TaskStatus.PLANNED: 0.0,
            TaskStatus.IN_PROGRESS: 25.0,
            TaskStatus.BLOCKED: 0.0,
            TaskStatus.WAITING_REVIEW: 75.0,
            TaskStatus.BEING_REVISED: 50.0,
            TaskStatus.TESTING: 80.0,
            TaskStatus.TEST_FAILED: 60.0,
            TaskStatus.COMPLETED: 100.0,
            TaskStatus.FAILED: 0.0,
            TaskStatus.CANCELLED: 0.0,
            TaskStatus.PAUSED: 25.0,
        }
        
        base_percentage = status_percentages.get(self.status, 0.0)
        
        # If the task has subtasks, factor in their completion
        if self.subtasks:
            return base_percentage
        
        return base_percentage
    
    def can_transition_to(self, new_status: TaskStatus) -> bool:
        """
        Check if the task can transition to the given status.
        
        Args:
            new_status: The status to check transition to
            
        Returns:
            Whether the transition is valid
        """
        # Define valid transitions for each status
        valid_transitions = {
            TaskStatus.PLANNED: {
                TaskStatus.IN_PROGRESS, 
                TaskStatus.BLOCKED, 
                TaskStatus.CANCELLED
            },
            TaskStatus.IN_PROGRESS: {
                TaskStatus.WAITING_REVIEW, 
                TaskStatus.TESTING, 
                TaskStatus.FAILED, 
                TaskStatus.PAUSED, 
                TaskStatus.BLOCKED
            },
            TaskStatus.BLOCKED: {
                TaskStatus.IN_PROGRESS, 
                TaskStatus.CANCELLED
            },
            TaskStatus.WAITING_REVIEW: {
                TaskStatus.COMPLETED, 
                TaskStatus.BEING_REVISED
            },
            TaskStatus.BEING_REVISED: {
                TaskStatus.WAITING_REVIEW, 
                TaskStatus.IN_PROGRESS, 
                TaskStatus.FAILED
            },
            TaskStatus.TESTING: {
                TaskStatus.COMPLETED, 
                TaskStatus.TEST_FAILED
            },
            TaskStatus.TEST_FAILED: {
                TaskStatus.IN_PROGRESS, 
                TaskStatus.FAILED
            },
            TaskStatus.COMPLETED: {
                TaskStatus.IN_PROGRESS  # Reopen for fixes
            },
            TaskStatus.FAILED: {
                TaskStatus.PLANNED,  # Retry from beginning
                TaskStatus.IN_PROGRESS  # Continue from where it failed
            },
            TaskStatus.CANCELLED: {
                TaskStatus.PLANNED  # Reinstate the task
            },
            TaskStatus.PAUSED: {
                TaskStatus.IN_PROGRESS,
                TaskStatus.CANCELLED
            }
        }
        
        return new_status in valid_transitions.get(self.status, set())
    
    def update_progress(self) -> None:
        """Update the task's completion percentage based on current status."""
        self.completion_percentage = self.calculate_completion_percentage()
        self.mark_updated()

class Epic(BaseTask):
    """Represents a high-level goal containing multiple tasks."""
    tasks: List[UUID] = Field(default_factory=list)
    parent_epic_id: Optional[UUID] = None
    sub_epics: List[UUID] = Field(default_factory=list)
    
    # Milestone tracking
    target_completion_date: Optional[datetime] = None
    milestone_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    def calculate_progress(self, task_map: Dict[UUID, Task]) -> float:
        """Calculate the overall progress of this epic."""
        if not self.tasks:
            return 0.0
            
        total_weight = sum(task_map[task_id].estimated_complexity for task_id in self.tasks 
                           if task_id in task_map)
        
        if total_weight == 0:
            return 0.0
            
        weighted_completion = sum(
            task_map[task_id].completion_percentage * task_map[task_id].estimated_complexity
            for task_id in self.tasks
            if task_id in task_map
        )
        
        return weighted_completion / total_weight if total_weight > 0 else 0.0
    
    def calculate_risk_level(self, task_map: Dict[UUID, Task]) -> TaskRisk:
        """Calculate the overall risk level of this epic based on its tasks."""
        if not self.tasks:
            return self.risk
        
        # Count tasks by risk level
        risk_counts = {
            TaskRisk.LOW: 0,
            TaskRisk.MEDIUM: 0,
            TaskRisk.HIGH: 0,
            TaskRisk.CRITICAL: 0
        }
        
        # Count tasks by risk level, weighted by estimated complexity
        for task_id in self.tasks:
            if task_id in task_map:
                task = task_map[task_id]
                risk_counts[task.risk] += task.estimated_complexity
        
        # Determine the highest significant risk level
        # (significant means it represents at least 20% of the tasks by complexity)
        total_complexity = sum(risk_counts.values())
        if total_complexity == 0:
            return self.risk
            
        for risk in [TaskRisk.CRITICAL, TaskRisk.HIGH, TaskRisk.MEDIUM]:
            if risk_counts[risk] / total_complexity >= 0.2:
                return risk
                
        return TaskRisk.LOW

class TaskGraph(BaseModel):
    """Represents the complete graph of tasks, epics, and their dependencies."""
    epics: Dict[UUID, Epic] = Field(default_factory=dict)
    tasks: Dict[UUID, Task] = Field(default_factory=dict)
    dependencies: List[Dependency] = Field(default_factory=list)
    
    # Statistics tracking
    creation_timestamp: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    task_execution_attempts: int = Field(0)
    successful_executions: int = Field(0)
    failed_executions: int = Field(0)
    
    def add_task(self, task: Task) -> UUID:
        """Add a task to the graph and return its ID."""
        self.tasks[task.id] = task
        if task.epic_id and task.epic_id in self.epics:
            epic = self.epics[task.epic_id]
            if task.id not in epic.tasks:
                epic.tasks.append(task.id)
                epic.mark_updated()
        self.last_updated = datetime.now()
        return task.id
    
    def add_epic(self, epic: Epic) -> UUID:
        """Add an epic to the graph and return its ID."""
        self.epics[epic.id] = epic
        if epic.parent_epic_id and epic.parent_epic_id in self.epics:
            parent_epic = self.epics[epic.parent_epic_id]
            if epic.id not in parent_epic.sub_epics:
                parent_epic.sub_epics.append(epic.id)
                parent_epic.mark_updated()
        self.last_updated = datetime.now()
        return epic.id
    
    def add_dependency(self, dependency: Dependency) -> None:
        """Add a dependency between tasks."""
        self.dependencies.append(dependency)
        
        # Update the task dependency references
        if dependency.source_id in self.tasks and dependency.target_id in self.tasks:
            source_task = self.tasks[dependency.source_id]
            target_task = self.tasks[dependency.target_id]
            
            if dependency.target_id not in source_task.dependencies:
                source_task.dependencies.append(dependency.target_id)
                source_task.mark_updated()
                
            if dependency.source_id not in target_task.dependents:
                target_task.dependents.append(dependency.source_id)
                target_task.mark_updated()
        
        self.last_updated = datetime.now()
    
    def get_critical_path(self) -> List[UUID]:
        """Identify the critical path through the task graph."""
        # Simple implementation that finds the longest path in terms of estimated duration
        # A full implementation would use A* pathfinding
        
        # Start with tasks with no dependencies
        start_tasks = [task_id for task_id, task in self.tasks.items() if not task.dependencies]
        if not start_tasks:
            return []
            
        # Find paths from each starting task
        paths = []
        for start_task_id in start_tasks:
            paths.extend(self._find_paths_from(start_task_id))
            
        if not paths:
            return []
            
        # Return the path with the longest total duration
        return max(paths, key=lambda path: sum(
            self.tasks[task_id].estimated_duration_hours for task_id in path
        ))
    
    def _find_paths_from(self, task_id: UUID, current_path: List[UUID] = None) -> List[List[UUID]]:
        """Find all paths from the given task."""
        if current_path is None:
            current_path = []
            
        # Avoid cycles
        if task_id in current_path:
            return [current_path]
            
        current_path = current_path + [task_id]
        
        if task_id not in self.tasks:
            return [current_path]
            
        task = self.tasks[task_id]
        if not task.dependents:
            return [current_path]
            
        paths = []
        for dependent_id in task.dependents:
            new_paths = self._find_paths_from(dependent_id, current_path)
            paths.extend(new_paths)
            
        return paths if paths else [current_path]
    
    def update_task_statuses(self) -> None:
        """Update the status of tasks based on their dependencies."""
        for task_id, task in self.tasks.items():
            # Skip tasks that are completed, failed, or cancelled
            if task.status in [
                TaskStatus.COMPLETED, 
                TaskStatus.FAILED, 
                TaskStatus.CANCELLED
            ]:
                continue
                
            # Check if task is blocked
            if task.status in [TaskStatus.PLANNED, TaskStatus.BLOCKED]:
                is_blocked = False
                for dep_id in task.dependencies:
                    if dep_id in self.tasks and self.tasks[dep_id].status != TaskStatus.COMPLETED:
                        is_blocked = True
                        break
                
                new_status = TaskStatus.BLOCKED if is_blocked else TaskStatus.PLANNED
                if task.status != new_status:
                    task.transition_status(new_status)
                    task.mark_updated()
        
        self.last_updated = datetime.now()
    
    def calculate_epic_progress(self) -> None:
        """Update progress for all epics based on task completion."""
        for epic_id, epic in self.epics.items():
            epic.milestone_percentage = epic.calculate_progress(self.tasks)
            epic.mark_updated()
        
        self.last_updated = datetime.now()
    
    def record_task_execution(self, task_id: UUID, attempt: ExecutionAttempt) -> None:
        """
        Record a task execution attempt in both the task and graph statistics.
        
        Args:
            task_id: ID of the task
            attempt: Execution attempt details
        """
        if task_id in self.tasks:
            # Record in task statistics
            self.tasks[task_id].record_execution_attempt(attempt)
            
            # Update graph statistics
            self.task_execution_attempts += 1
            if attempt.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
                
            self.last_updated = datetime.now()
    
    def get_execution_success_rate(self) -> float:
        """Get the overall success rate for task executions."""
        if self.task_execution_attempts == 0:
            return 0.0
        return self.successful_executions / self.task_execution_attempts
    
    def get_blocking_tasks(self) -> List[Task]:
        """Get tasks that are blocking other tasks from being started."""
        blocking_tasks = []
        for task_id, task in self.tasks.items():
            # Skip completed tasks
            if task.status == TaskStatus.COMPLETED:
                continue
                
            # Check if this task is blocking others
            is_blocking = False
            for dep_task_id, dep_task in self.tasks.items():
                if task_id in dep_task.dependencies and dep_task.status in [TaskStatus.PLANNED, TaskStatus.BLOCKED]:
                    is_blocking = True
                    break
            
            if is_blocking:
                blocking_tasks.append(task)
                
        return blocking_tasks
    
    def get_task_by_title(self, title: str) -> Optional[Task]:
        """Find a task by its title (case-insensitive)."""
        title_lower = title.lower()
        for task in self.tasks.values():
            if task.title.lower() == title_lower:
                return task
        return None
    
    def get_epic_by_title(self, title: str) -> Optional[Epic]:
        """Find an epic by its title (case-insensitive)."""
        title_lower = title.lower()
        for epic in self.epics.values():
            if epic.title.lower() == title_lower:
                return epic
        return None