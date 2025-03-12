# task_models.py
"""Pydantic models for tasks, epics, and dependencies."""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class TaskStatus(str, Enum):
    """Status states for tasks and epics."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"

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
    
    def mark_updated(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

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
    
    @model_validator(mode='after')
    def validate_task_structure(self) -> 'Task':
        """Validate the task structure relationships."""
        if self.parent_task_id is not None and self.epic_id is not None:
            raise ValueError("A task cannot have both a parent task and an epic")
        return self

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

class TaskGraph(BaseModel):
    """Represents the complete graph of tasks, epics, and their dependencies."""
    epics: Dict[UUID, Epic] = Field(default_factory=dict)
    tasks: Dict[UUID, Task] = Field(default_factory=dict)
    dependencies: List[Dependency] = Field(default_factory=list)
    
    def add_task(self, task: Task) -> UUID:
        """Add a task to the graph and return its ID."""
        self.tasks[task.id] = task
        if task.epic_id and task.epic_id in self.epics:
            epic = self.epics[task.epic_id]
            if task.id not in epic.tasks:
                epic.tasks.append(task.id)
                epic.mark_updated()
        return task.id
    
    def add_epic(self, epic: Epic) -> UUID:
        """Add an epic to the graph and return its ID."""
        self.epics[epic.id] = epic
        if epic.parent_epic_id and epic.parent_epic_id in self.epics:
            parent_epic = self.epics[epic.parent_epic_id]
            if epic.id not in parent_epic.sub_epics:
                parent_epic.sub_epics.append(epic.id)
                parent_epic.mark_updated()
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
            # Check if task is blocked
            if task.status in [TaskStatus.PLANNED, TaskStatus.BLOCKED]:
                is_blocked = False
                for dep_id in task.dependencies:
                    if dep_id in self.tasks and self.tasks[dep_id].status != TaskStatus.COMPLETED:
                        is_blocked = True
                        break
                
                task.status = TaskStatus.BLOCKED if is_blocked else TaskStatus.PLANNED
                task.mark_updated()
    
    def calculate_epic_progress(self) -> None:
        """Update progress for all epics based on task completion."""
        for epic_id, epic in self.epics.items():
            epic.milestone_percentage = epic.calculate_progress(self.tasks)
            epic.mark_updated()