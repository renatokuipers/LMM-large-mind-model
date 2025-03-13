"""
Core integration module for the AgenDev UI.

This module handles the integration between the UI and the core AgenDev functionality.
"""
from typing import Dict, List, Any, Tuple, Optional
import json
import time
from datetime import datetime
from pathlib import Path
import os

# Core module imports for AgenDev integration
try:
    from agendev.core import AgenDev, AgenDevConfig
    from agendev.models.task_models import (
        Task, TaskStatus, TaskPriority, TaskRisk, TaskType, 
        Epic, TaskGraph, Dependency
    )
    from agendev.models.planning_models import (
        SimulationConfig, PlanSnapshot, PlanningHistory, 
        SearchNodeType, PlanningPhase, SimulationResult
    )
    from agendev.llm_integration import LLMIntegration, LLMConfig
    from agendev.search_algorithms import MCTSPlanner, AStarPathfinder
    from agendev.probability_modeling import TaskProbabilityModel, ProjectRiskModel
    from agendev.parameter_controller import ParameterController
    from agendev.tts_notification import NotificationManager
    AGENDEV_AVAILABLE = True
except ImportError:
    # If core modules are not available, use mock implementations
    AGENDEV_AVAILABLE = False
    print("Warning: AgenDev core modules not available. Using mock implementations.")

class MockTask:
    """Mock implementation of a Task for testing without core modules."""
    def __init__(self, title, description, status="planned"):
        self.id = f"task_{int(time.time())}"
        self.title = title
        self.description = description
        self.status = status
        self.dependencies = []
        self.artifact_paths = []

class MockAgenDev:
    """Mock implementation of AgenDev for testing without core modules."""
    def __init__(self, project_name):
        self.project_name = project_name
        self.tasks = []
    
    def create_task(self, title, description):
        task = MockTask(title, description)
        self.tasks.append(task)
        return task.id
    
    def implement_task(self, task_id):
        # Find task
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            return {"error": "Task not found"}
        
        # Simulate implementation
        task.status = "completed"
        task.artifact_paths.append(f"{task.title.lower().replace(' ', '_')}.py")
        
        return {
            "success": True,
            "task_id": task_id,
            "implementation": f"# Implementation for {task.title}\n\ndef main():\n    print('Hello from {task.title}')\n\nif __name__ == '__main__':\n    main()",
            "file_path": task.artifact_paths[0]
        }
    
    def get_project_status(self):
        return {
            "project_name": self.project_name,
            "state": "implementing",
            "tasks": {
                "total": len(self.tasks),
                "by_status": {
                    "planned": sum(1 for t in self.tasks if t.status == "planned"),
                    "completed": sum(1 for t in self.tasks if t.status == "completed")
                }
            }
        }

class CoreIntegration:
    """
    Handles integration between the UI and AgenDev core functionality.
    
    This class provides methods for initializing projects, executing tasks,
    and getting status updates from the core system.
    """
    
    def __init__(self, llm_base_url: str = "http://192.168.2.12:1234", 
                tts_base_url: str = "http://127.0.0.1:7860"):
        """
        Initialize the core integration.
        
        Args:
            llm_base_url: URL for the LLM API
            tts_base_url: URL for the TTS API
        """
        self.llm_base_url = llm_base_url
        self.tts_base_url = tts_base_url
        self.agendev_instance = None
        
    def initialize_project(self, project_name: str, project_description: str) -> Dict[str, Any]:
        """
        Initialize a new project in AgenDev.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            
        Returns:
            Dictionary with project initialization details
        """
        try:
            if AGENDEV_AVAILABLE:
                # Initialize actual AgenDev instance
                config = AgenDevConfig(
                    project_name=project_name,
                    llm_base_url=self.llm_base_url,
                    tts_base_url=self.tts_base_url
                )
                self.agendev_instance = AgenDev(config)
                
                # Generate implementation plan
                plan = self.agendev_instance.generate_implementation_plan()
                
                # Get project status
                status = self.agendev_instance.get_project_status()
                
                return {
                    "success": True,
                    "project_name": project_name,
                    "plan": plan.model_dump() if hasattr(plan, "model_dump") else vars(plan),
                    "status": status
                }
            else:
                # Use mock implementation for testing
                self.agendev_instance = MockAgenDev(project_name)
                
                # Create some sample tasks
                tasks = [
                    {"title": "Initialize project repository", "description": "Set up Git repository and initial file structure"},
                    {"title": "Create basic project structure", "description": "Set up directories and configuration files"},
                    {"title": "Implement core functionality", "description": f"Write code for main features of {project_name}"},
                    {"title": "Add unit tests", "description": "Write tests for the implemented functionality"},
                    {"title": "Create documentation", "description": "Write documentation for the project"}
                ]
                
                task_ids = []
                for task in tasks:
                    task_id = self.agendev_instance.create_task(task["title"], task["description"])
                    task_ids.append(task_id)
                
                return {
                    "success": True,
                    "project_name": project_name,
                    "mock": True,
                    "task_ids": task_ids,
                    "status": self.agendev_instance.get_project_status()
                }
        except Exception as e:
            print(f"Error initializing project: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks for the current project.
        
        Returns:
            List of task details
        """
        if not self.agendev_instance:
            return []
            
        try:
            if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
                # Get tasks from actual AgenDev instance
                tasks = []
                for task_id, task in self.agendev_instance.task_graph.tasks.items():
                    tasks.append({
                        "id": str(task_id),
                        "title": task.title,
                        "description": task.description,
                        "status": task.status.value,
                        "dependencies": [str(dep_id) for dep_id in task.dependencies],
                        "artifact_paths": task.artifact_paths
                    })
                return tasks
            else:
                # Get tasks from mock instance
                return [
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "status": task.status,
                        "dependencies": [str(dep_id) for dep_id in task.dependencies],
                        "artifact_paths": task.artifact_paths
                    }
                    for task in self.agendev_instance.tasks
                ]
        except Exception as e:
            print(f"Error getting tasks: {e}")
            return []
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task using AgenDev.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Dictionary with execution results
        """
        if not self.agendev_instance:
            return {"error": "No active project"}
            
        try:
            # Convert string ID to UUID if needed
            if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "implement_task"):
                from uuid import UUID
                if not isinstance(task_id, UUID):
                    task_id = UUID(task_id)
                    
                # Execute task with actual AgenDev instance
                return self.agendev_instance.implement_task(task_id)
            else:
                # Execute task with mock instance
                return self.agendev_instance.implement_task(task_id)
        except Exception as e:
            print(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project.
        
        Returns:
            Dictionary with project status
        """
        if not self.agendev_instance:
            return {"error": "No active project"}
            
        try:
            return self.agendev_instance.get_project_status()
        except Exception as e:
            print(f"Error getting project status: {e}")
            return {
                "error": str(e)
            }
    
    def generate_todo_markdown(self, project_name: str, tasks: List[Dict[str, Any]]) -> str:
        """
        Generate todo.md content based on tasks.
        
        Args:
            project_name: Name of the project
            tasks: List of task details
            
        Returns:
            Markdown content for todo.md
        """
        # Group tasks by status
        planned_tasks = [t for t in tasks if t["status"] == "planned"]
        in_progress_tasks = [t for t in tasks if t["status"] == "in_progress"]
        completed_tasks = [t for t in tasks if t["status"] == "completed"]
        
        # Build Markdown content
        markdown = f"# {project_name}\n\n"
        
        if in_progress_tasks:
            markdown += "## In Progress\n"
            for task in in_progress_tasks:
                markdown += f"- [ ] {task['title']}\n"
            markdown += "\n"
        
        if planned_tasks:
            markdown += "## Planned\n"
            for task in planned_tasks:
                markdown += f"- [ ] {task['title']}\n"
            markdown += "\n"
        
        if completed_tasks:
            markdown += "## Completed\n"
            for task in completed_tasks:
                markdown += f"- [x] {task['title']}\n"
            markdown += "\n"
        
        return markdown
    
    def generate_playback_steps(self, task_execution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate playback steps for a task execution.
        
        Args:
            task_execution_result: Result of task execution
            
        Returns:
            List of playback steps
        """
        if not task_execution_result.get("success", False):
            return []
            
        steps = []
        task_id = task_execution_result.get("task_id", "")
        file_path = task_execution_result.get("file_path", "")
        implementation = task_execution_result.get("implementation", "")
        
        # Find task details
        task_title = ""
        if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
            from uuid import UUID
            if not isinstance(task_id, UUID):
                task_id = UUID(task_id)
                
            if task_id in self.agendev_instance.task_graph.tasks:
                task_title = self.agendev_instance.task_graph.tasks[task_id].title
        else:
            # Use mock
            task = next((t for t in self.agendev_instance.tasks if t.id == task_id), None)
            if task:
                task_title = task.title
        
        # Step 1: Planning
        steps.append({
            "type": "terminal",
            "content": f"$ echo 'Planning implementation for {task_title}'\nPlanning implementation for {task_title}\n$ mkdir -p $(dirname {file_path})\n",
            "operation_type": "Planning",
            "file_path": task_title
        })
        
        # Step 2: Implementation
        steps.append({
            "type": "editor",
            "filename": file_path,
            "content": implementation,
            "operation_type": "Implementing",
            "file_path": file_path
        })
        
        # Step 3: Saving
        steps.append({
            "type": "terminal",
            "content": f"$ echo 'Saving implementation to {file_path}'\nSaving implementation to {file_path}\n$ git add {file_path}\n$ git commit -m 'Implement {task_title}'\n[main] Commit message: Implement {task_title}\n 1 file changed, {len(implementation.split('n'))} insertions(+)",
            "operation_type": "Saving",
            "file_path": file_path
        })
        
        return steps