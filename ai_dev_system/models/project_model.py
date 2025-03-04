from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Optional, Dict, Any, ClassVar
from enum import Enum
from datetime import datetime
import uuid
import json
import os
from pathlib import Path


class TaskStatus(str, Enum):
    """Status of a task or epic in the project."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CodeItem(BaseModel):
    """Representation of a code item (function, class, method)"""
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    
    type: str = Field(
        ..., 
        description="Type of code item (Function, Class, Method)"
    )
    name: str = Field(
        ..., 
        description="Name of the code item"
    )
    parameters: List[str] = Field(
        default_factory=list, 
        description="Parameters for the code item"
    )
    result: str = Field(
        "None", 
        description="Return value or result of the code item"
    )
    implemented: bool = Field(
        False, 
        description="Whether this item has been implemented"
    )
    file_path: Optional[str] = Field(
        None, 
        description="Path to the file containing this code item"
    )
    parent: Optional[str] = Field(
        None, 
        description="Parent item name (for methods)"
    )
    description: str = Field(
        "", 
        description="Description of what this code item does"
    )
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure the type is one of the accepted values."""
        normalized = v.lower().capitalize()
        if normalized in {"Function", "Class", "Method"}:
            return normalized
        raise ValueError(f"Type must be Function, Class, or Method (got {v})")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure the name is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError(f"Name '{v}' is not a valid Python identifier")
        return v


class Task(BaseModel):
    """A specific development task within an epic"""
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique task identifier"
    )
    title: str = Field(
        ..., 
        description="Task title"
    )
    description: str = Field(
        ..., 
        description="Detailed task description"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PLANNED, 
        description="Current task status"
    )
    code_items: List[CodeItem] = Field(
        default_factory=list, 
        description="Code items for this task"
    )
    dependencies: List[str] = Field(
        default_factory=list, 
        description="IDs of tasks this depends on"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, 
        description="Task creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, 
        description="Task last update timestamp"
    )
    
    @model_validator(mode='before')
    @classmethod
    def update_timestamps(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Always update the updated_at timestamp when the model is changed"""
        if isinstance(data, dict):
            data['updated_at'] = datetime.now()
        return data
    
    @property
    def is_blocked(self) -> bool:
        """Return True if task is blocked by dependencies"""
        return bool(self.dependencies)
    
    @property
    def completion_percentage(self) -> float:
        """Calculate task completion percentage based on code items"""
        if not self.code_items:
            return 100.0 if self.status == TaskStatus.COMPLETED else 0.0
            
        implemented_count = sum(1 for item in self.code_items if item.implemented)
        return (implemented_count / len(self.code_items)) * 100


class Epic(BaseModel):
    """A group of related tasks representing a major feature"""
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique epic identifier"
    )
    title: str = Field(
        ..., 
        description="Epic title"
    )
    description: str = Field(
        ..., 
        description="Epic description"
    )
    tasks: List[Task] = Field(
        default_factory=list, 
        description="Tasks within this epic"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, 
        description="Epic creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Epic last update timestamp"
    )
    
    @model_validator(mode='before')
    @classmethod
    def update_timestamps(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Always update the updated_at timestamp when the model is changed"""
        if isinstance(data, dict):
            data['updated_at'] = datetime.now()
        return data
    
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
    
    @property
    def completion_percentage(self) -> float:
        """Calculate epic completion percentage"""
        if not self.tasks:
            return 0.0
            
        total_percentage = sum(task.completion_percentage for task in self.tasks)
        return total_percentage / len(self.tasks)


class Project(BaseModel):
    """Complete project definition"""
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique project identifier"
    )
    name: str = Field(
        ..., 
        description="Project name"
    )
    description: str = Field(
        ..., 
        description="Project description"
    )
    epics: List[Epic] = Field(
        default_factory=list, 
        description="Epics within this project"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, 
        description="Project creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, 
        description="Project last update timestamp"
    )
    tech_stack: Dict[str, List[str]] = Field(
        default_factory=lambda: {"frontend": [], "backend": [], "database": []},
        description="Technologies used in the project"
    )
    output_directory: str = Field(
        ..., 
        description="Directory where project files will be generated"
    )
    
    @model_validator(mode='before')
    @classmethod
    def update_timestamps(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Always update the updated_at timestamp when the model is changed"""
        if isinstance(data, dict):
            data['updated_at'] = datetime.now()
        return data
    
    @field_validator('output_directory')
    @classmethod
    def validate_output_directory(cls, v: str) -> str:
        """Validate that the output directory exists or can be created"""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except Exception as e:
            raise ValueError(f"Invalid output directory: {e}")
    
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
    """Store for multiple projects with efficient loading and saving"""
    model_config = ConfigDict(
        frozen=False,
    )
    
    projects: Dict[str, Project] = Field(
        default_factory=dict, 
        description="Dictionary of projects"
    )
    
    # Class variable for singleton pattern
    _instance: ClassVar[Optional['ProjectStore']] = None
    
    def add_project(self, project: Project) -> None:
        """Add a project to the store"""
        self.projects[project.id] = project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID"""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[Project]:
        """List all projects"""
        return list(self.projects.values())
    
    def remove_project(self, project_id: str) -> bool:
        """Remove a project from the store"""
        if project_id in self.projects:
            del self.projects[project_id]
            return True
        return False
    
    def save_to_file(self, filepath: str) -> None:
        """Save projects to a file, splitting large projects into individual files"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save each project to its own file to avoid large single file
        project_ids = []
        data_dir = Path(filepath).parent / "project_data"
        data_dir.mkdir(exist_ok=True)
        
        for project_id, project in self.projects.items():
            project_file = data_dir / f"{project_id}.json"
            with open(project_file, 'w') as f:
                f.write(project.model_dump_json(indent=2))
            project_ids.append(project_id)
        
        # Save index file with just project IDs
        with open(filepath, 'w') as f:
            json.dump({"project_ids": project_ids}, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectStore':
        """Load projects from a file"""
        store = cls()
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                return store
                
            # Load the index file
            with open(filepath, 'r') as f:
                index_data = json.load(f)
                
            # Load each project from its own file
            data_dir = Path(filepath).parent / "project_data"
            for project_id in index_data.get("project_ids", []):
                project_file = data_dir / f"{project_id}.json"
                
                if os.path.exists(project_file):
                    with open(project_file, 'r') as f:
                        project_data = f.read()
                        project = Project.model_validate_json(project_data)
                        store.projects[project_id] = project
                
        except Exception as e:
            print(f"Error loading projects: {str(e)}")
            
        return store
    
    @classmethod
    def get_instance(cls, filepath: str = "projects.json") -> 'ProjectStore':
        """Get or create a singleton instance of ProjectStore"""
        if cls._instance is None:
            cls._instance = cls.load_from_file(filepath)
        return cls._instance