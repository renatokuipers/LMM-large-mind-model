# core/project_context.py
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Set
from datetime import datetime
import os
from .schemas import Epic, FileStructure, ProjectConfig

class ProjectContext(BaseModel):
    """Core model maintaining the entire project state"""
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    output_dir: str
    config: ProjectConfig
    epics: List[Epic] = Field(default_factory=list)
    files: Dict[str, FileStructure] = Field(default_factory=dict)
    current_epic_id: Optional[str] = None
    current_task_id: Optional[str] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
        return v
    
    def update_modified_time(self):
        self.modified_at = datetime.now()
    
    def add_file_structure(self, file_path: str, file_structure: FileStructure):
        self.files[file_path] = file_structure
        self.update_modified_time()
    
    def get_task_by_id(self, task_id: str):
        for epic in self.epics:
            for task in epic.tasks:
                if task.id == task_id:
                    return task
        return None
    
    def get_epic_by_id(self, epic_id: str):
        for epic in self.epics:
            if epic.id == epic_id:
                return epic
        return None
    
    def update_task_status(self, task_id: str, status: str):
        task = self.get_task_by_id(task_id)
        if task:
            task.status = status
            self.update_modified_time()
            return True
        return False
    
    def get_all_file_paths(self) -> Set[str]:
        return set(self.files.keys())
    
    def get_pending_tasks(self):
        """Get all pending tasks that have no incomplete dependencies"""
        result = []
        for epic in self.epics:
            for task in epic.tasks:
                if task.status == "pending":
                    # Check if all dependencies are completed
                    deps_completed = True
                    for dep_id in task.dependencies:
                        dep_task = self.get_task_by_id(dep_id)
                        if dep_task and dep_task.status != "completed":
                            deps_completed = False
                            break
                    
                    if deps_completed:
                        result.append(task)
        return result