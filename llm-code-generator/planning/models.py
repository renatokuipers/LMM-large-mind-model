from typing import List, Dict, Optional, Set, Literal, Union, Any
from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, root_validator

class TaskStatus(str, Enum):
    """Status of a planning task."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

class ComponentType(str, Enum):
    """Type of component to be generated."""
    DATA_MODEL = "data_model"
    DATABASE_MODEL = "database_model"
    REPOSITORY = "repository"
    SERVICE = "service"
    USE_CASE = "use_case"
    API_ENDPOINT = "api_endpoint"
    API_ROUTER = "api_router"
    SCHEMA = "schema"
    CONFIG = "config"
    UTIL = "util"
    TEST = "test"

class DependencyType(str, Enum):
    """Type of dependency between tasks."""
    REQUIRED = "required"       # Hard dependency - must be completed first
    PREFERRED = "preferred"     # Soft dependency - should be completed first
    REFERENCED = "referenced"   # Referenced but not blocking

class Subtask(BaseModel):
    """Represents a granular unit of work within a task."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique subtask ID")
    title: str = Field(..., description="Subtask title")
    description: str = Field(..., description="Detailed description of the subtask")
    status: TaskStatus = Field(default=TaskStatus.PLANNED, description="Current status")
    estimated_complexity: int = Field(1, ge=1, le=5, description="Complexity estimate (1-5)")
    
    @validator('description')
    def description_must_be_detailed(cls, v):
        if len(v.split()) < 5:
            raise ValueError("Description must be detailed (at least 5 words)")
        return v

class Task(BaseModel):
    """Represents a specific code generation task."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task ID")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed description of what needs to be generated")
    component_type: ComponentType = Field(..., description="Type of component to generate")
    module_path: str = Field(..., description="Module path where this component will be defined")
    status: TaskStatus = Field(default=TaskStatus.PLANNED, description="Current status")
    
    # Component details
    class_or_function_name: Optional[str] = Field(None, description="Name of the class or function to generate")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements for this component")
    
    # Dependencies and relationships
    dependencies: Dict[str, DependencyType] = Field(
        default_factory=dict, 
        description="Task IDs this task depends on, with dependency type"
    )
    dependents: List[str] = Field(
        default_factory=list, 
        description="Task IDs that depend on this task"
    )
    
    # Task metadata
    estimated_complexity: int = Field(1, ge=1, le=5, description="Complexity estimate (1-5)")
    priority: int = Field(3, ge=1, le=5, description="Priority (1-5, 5 being highest)")
    subtasks: List[Subtask] = Field(default_factory=list, description="Subtasks for this task")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    
    # Execution details
    assigned_to: Optional[str] = Field(None, description="Generator assigned to this task")
    started_at: Optional[datetime] = Field(None, description="When task execution started")
    completed_at: Optional[datetime] = Field(None, description="When task was completed")
    
    @property
    def is_blocked(self) -> bool:
        """Check if this task is blocked by dependencies."""
        return self.status == TaskStatus.BLOCKED
    
    @property
    def can_start(self) -> bool:
        """Check if this task can be started (all required dependencies complete)."""
        # Implementation would check if all REQUIRED dependencies are completed
        return True
    
    @validator('module_path')
    def validate_module_path(cls, v):
        if not all(part.isidentifier() for part in v.split('.')):
            raise ValueError("Module path must be a valid Python import path")
        return v

class Epic(BaseModel):
    """Represents a group of related tasks focused on a specific aspect of the system."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique epic ID")
    title: str = Field(..., description="Epic title")
    description: str = Field(..., description="Detailed description of this epic")
    tasks: List[Task] = Field(default_factory=list, description="Tasks in this epic")
    status: TaskStatus = Field(default=TaskStatus.PLANNED, description="Current status")
    
    # Metadata
    order: int = Field(0, description="Execution order (lower numbers first)")
    estimated_complexity: int = Field(1, ge=1, le=5, description="Overall complexity estimate")
    priority: int = Field(3, ge=1, le=5, description="Priority (1-5, 5 being highest)")
    
    @property
    def completion_percentage(self) -> float:
        """Calculate the percentage of completed tasks."""
        if not self.tasks:
            return 0.0
        
        completed = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100

    @property
    def is_completed(self) -> bool:
        """Check if this epic is completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)
    
    @validator('tasks')
    def tasks_must_have_same_epic(cls, v):
        # In a full implementation, we'd ensure task-epic relationships are consistent
        return v

class ProjectPlan(BaseModel):
    """Complete plan for a code generation project with validation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique plan ID")
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Detailed project description")
    epics: List[Epic] = Field(default_factory=list, description="Epics in this project")
    
    # Project metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When plan was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When plan was last updated")
    version: int = Field(1, description="Plan version number")
    
    @property
    def all_tasks(self) -> List[Task]:
        """Get all tasks across all epics."""
        return [task for epic in self.epics for task in epic.tasks]
    
    @property
    def completion_percentage(self) -> float:
        """Calculate the percentage of completed tasks."""
        tasks = self.all_tasks
        if not tasks:
            return 0.0
        
        completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        return (completed / len(tasks)) * 100
    
    @property
    def task_by_id(self) -> Dict[str, Task]:
        """Get a mapping of task ID to task."""
        return {task.id: task for task in self.all_tasks}
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all dependencies exist and there are no circular dependencies.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        task_map = self.task_by_id
        
        # Check that all dependencies exist
        for task in self.all_tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_map:
                    errors.append(f"Task {task.id} ({task.title}) depends on non-existent task {dep_id}")
        
        # Check for circular dependencies (simplistic check)
        for task in self.all_tasks:
            visited = set()
            to_visit = list(task.dependencies.keys())
            
            while to_visit:
                current = to_visit.pop()
                if current == task.id:
                    errors.append(f"Circular dependency detected for task {task.id} ({task.title})")
                    break
                
                if current in visited:
                    continue
                    
                visited.add(current)
                if current in task_map:
                    to_visit.extend(task_map[current].dependencies.keys())
        
        return errors
    
    @root_validator
    def validate_plan(cls, values):
        """Validate the entire project plan."""
        # In a full implementation, we'd do more thorough validation here
        return values

class PlanReview(BaseModel):
    """Review of a project plan with suggestions for improvement."""
    
    plan_id: str = Field(..., description="ID of the reviewed plan")
    review_date: datetime = Field(default_factory=datetime.now, description="When review was conducted")
    
    # Review items
    issues: List[str] = Field(default_factory=list, description="Issues identified in the plan")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    missing_components: List[str] = Field(default_factory=list, description="Components that might be missing")
    
    # Specific review areas
    dependency_issues: List[str] = Field(default_factory=list, description="Issues with task dependencies")
    complexity_issues: List[str] = Field(default_factory=list, description="Issues with complexity estimates")
    coverage_issues: List[str] = Field(default_factory=list, description="Areas not covered by the plan")
    
    # Overall assessment
    is_approved: bool = Field(False, description="Whether the plan is approved")
    overall_assessment: str = Field(..., description="Overall assessment of the plan")
    
    @property
    def has_issues(self) -> bool:
        """Check if review found any issues."""
        return bool(self.issues or self.dependency_issues or 
                    self.complexity_issues or self.coverage_issues)