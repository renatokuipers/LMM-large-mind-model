"""Pydantic models for tasks, epics, and dependencies."""
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    """Status of a task in the development lifecycle."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Types of tasks in the development system."""
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"


class RiskLevel(str, Enum):
    """Risk levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TaskDependency(BaseModel):
    """Represents a dependency between tasks."""
    task_id: UUID
    dependency_type: str = "requires"  # requires, enhances, conflicts, etc.
    description: Optional[str] = None


class Task(BaseModel):
    """Represents a single development task."""
    id: UUID = Field(default_factory=uuid4)
    epic_id: Optional[UUID] = None
    title: str
    description: str
    status: TaskStatus = TaskStatus.PLANNED
    priority: TaskPriority = TaskPriority.MEDIUM
    task_type: TaskType
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Time estimates
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    
    # Dependencies
    dependencies: List[TaskDependency] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    # Implementation details
    implementation_path: Optional[str] = None
    test_path: Optional[str] = None
    
    # Probability data
    success_probability: float = 0.5  # 0.0 to 1.0
    
    @field_validator('title')
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate that the title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v
    
    @field_validator('success_probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate that the probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v


class Epic(BaseModel):
    """Represents a collection of related tasks (epic)."""
    id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    status: TaskStatus = TaskStatus.PLANNED
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Tasks in this epic
    task_ids: List[UUID] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('title')
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate that the title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v
    
    def add_task(self, task_id: UUID) -> None:
        """Add a task to this epic."""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.updated_at = datetime.now()
    
    def remove_task(self, task_id: UUID) -> None:
        """Remove a task from this epic."""
        if task_id in self.task_ids:
            self.task_ids.remove(task_id)
            self.updated_at = datetime.now()


class TaskGraph(BaseModel):
    """Represents a graph of tasks and their dependencies."""
    tasks: Dict[UUID, Task] = Field(default_factory=dict)
    epics: Dict[UUID, Epic] = Field(default_factory=dict)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        self.tasks[task.id] = task
        
    def add_epic(self, epic: Epic) -> None:
        """Add an epic to the graph."""
        self.epics[epic.id] = epic
    
    def get_dependencies(self, task_id: UUID) -> List[Task]:
        """Get all tasks that this task depends on."""
        if task_id not in self.tasks:
            return []
        
        dependencies = []
        for dep in self.tasks[task_id].dependencies:
            if dep.task_id in self.tasks:
                dependencies.append(self.tasks[dep.task_id])
        
        return dependencies
    
    def get_dependents(self, task_id: UUID) -> List[Task]:
        """Get all tasks that depend on this task."""
        dependents = []
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep.task_id == task_id:
                    dependents.append(task)
                    break
        
        return dependents
    
    def is_blocked(self, task_id: UUID) -> bool:
        """Check if a task is blocked by dependencies."""
        if task_id not in self.tasks:
            return False
        
        for dep in self.tasks[task_id].dependencies:
            if dep.task_id in self.tasks and self.tasks[dep.task_id].status != TaskStatus.COMPLETED:
                return True
        
        return False
    
    def get_next_tasks(self) -> List[Task]:
        """Get all tasks that are ready to be worked on."""
        next_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PLANNED and not self.is_blocked(task.id):
                next_tasks.append(task)
        
        # Sort by priority
        next_tasks.sort(key=lambda t: TaskPriority[t.priority.name].value, reverse=True)
        return next_tasks