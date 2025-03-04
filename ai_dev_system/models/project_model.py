from pydantic import BaseModel, Field, validator, constr
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CodeItem(BaseModel):
    """Representation of a code item (function, class, method)"""
    type: str = Field(..., description="Type of code item (Function, Class, Method)")
    name: str = Field(..., description="Name of the code item")
    parameters: List[str] = Field(default_factory=list, description="Parameters for the code item")
    result: str = Field("None", description="Return value or result of the code item")
    implemented: bool = Field(False, description="Whether this item has been implemented")
    file_path: Optional[str] = Field(None, description="Path to the file containing this code item")
    parent: Optional[str] = Field(None, description="Parent item name (for methods)")
    description: str = Field("", description="Description of what this code item does")


class Task(BaseModel):
    """A specific development task within an epic"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    status: TaskStatus = Field(default=TaskStatus.PLANNED, description="Current task status")
    code_items: List[CodeItem] = Field(default_factory=list, description="Code items for this task")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks this depends on")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Task last update timestamp")
    
    @validator('updated_at', always=True)
    def update_timestamp(cls, v, values):
        """Always update the timestamp when the model is updated"""
        return datetime.now()


class Epic(BaseModel):
    """A group of related tasks representing a major feature"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique epic identifier")
    title: str = Field(..., description="Epic title")
    description: str = Field(..., description="Epic description")
    tasks: List[Task] = Field(default_factory=list, description="Tasks within this epic")
    created_at: datetime = Field(default_factory=datetime.now, description="Epic creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Epic last update timestamp")
    
    @property
    def status(self) -> TaskStatus:
        """Calculate epic status based on task statuses"""
        if not self.tasks:
            return TaskStatus.PLANNED
        
        all_completed = all(task.status == TaskStatus.COMPLETED for task in self.tasks)
        if all_completed:
            return TaskStatus.COMPLETED
        
        any_in_progress = any(task.status == TaskStatus.IN_PROGRESS for task in self.tasks)
        if any_in_progress:
            return TaskStatus.IN_PROGRESS
        
        any_failed = any(task.status == TaskStatus.FAILED for task in self.tasks)
        if any_failed:
            return TaskStatus.FAILED
            
        return TaskStatus.PLANNED


class Project(BaseModel):
    """Complete project definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique project identifier")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    epics: List[Epic] = Field(default_factory=list, description="Epics within this project")
    created_at: datetime = Field(default_factory=datetime.now, description="Project creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Project last update timestamp")
    tech_stack: Dict[str, List[str]] = Field(
        default_factory=lambda: {"frontend": [], "backend": [], "database": []},
        description="Technologies used in the project"
    )
    output_directory: str = Field(..., description="Directory where project files will be generated")
    
    @property
    def status(self) -> TaskStatus:
        """Calculate project status based on epic statuses"""
        if not self.epics:
            return TaskStatus.PLANNED
        
        all_completed = all(epic.status == TaskStatus.COMPLETED for epic in self.epics)
        if all_completed:
            return TaskStatus.COMPLETED
        
        any_in_progress = any(epic.status == TaskStatus.IN_PROGRESS for epic in self.epics)
        if any_in_progress:
            return TaskStatus.IN_PROGRESS
        
        any_failed = any(epic.status == TaskStatus.FAILED for epic in self.epics)
        if any_failed:
            return TaskStatus.FAILED
            
        return TaskStatus.PLANNED
    
    @property
    def completion_percentage(self) -> float:
        """Calculate project completion percentage"""
        if not self.epics:
            return 0.0
            
        total_tasks = 0
        completed_tasks = 0
        
        for epic in self.epics:
            for task in epic.tasks:
                total_tasks += 1
                if task.status == TaskStatus.COMPLETED:
                    completed_tasks += 1
        
        if total_tasks == 0:
            return 0.0
            
        return (completed_tasks / total_tasks) * 100


class ProjectStore(BaseModel):
    """Store for multiple projects"""
    projects: Dict[str, Project] = Field(default_factory=dict, description="Dictionary of projects")
    
    def add_project(self, project: Project) -> None:
        """Add a project to the store"""
        self.projects[project.id] = project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID"""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[Project]:
        """List all projects"""
        return list(self.projects.values())
    
    def save_to_file(self, filepath: str) -> None:
        """Save projects to a file"""
        with open(filepath, 'w') as f:
            f.write(self.json(indent=2))
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectStore':
        """Load projects from a file"""
        try:
            with open(filepath, 'r') as f:
                data = f.read()
                return cls.parse_raw(data)
        except FileNotFoundError:
            return cls()