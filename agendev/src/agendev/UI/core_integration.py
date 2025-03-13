"""
Core integration module for the AgenDev UI.

This module handles the integration between the UI and the core AgenDev functionality.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import time
from datetime import datetime
from pathlib import Path
import os
from uuid import UUID, uuid4
import random
import logging
import traceback
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core module imports for AgenDev integration
try:
    from ..core import AgenDev, AgenDevConfig
    from ..models.task_models import (
        Task, TaskStatus, TaskPriority, TaskRisk, TaskType, 
        Epic, TaskGraph, Dependency
    )
    from ..models.planning_models import (
        SimulationConfig, PlanSnapshot, PlanningHistory, 
        SearchNodeType, PlanningPhase, SimulationResult
    )
    from agendev.llm_integration import LLMIntegration, LLMConfig
    from agendev.search_algorithms import MCTSPlanner, AStarPathfinder
    from agendev.probability_modeling import TaskProbabilityModel, ProjectRiskModel
    from agendev.parameter_controller import ParameterController
    from agendev.tts_notification import NotificationManager
    AGENDEV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import AgenDev core components: {e}")
    AGENDEV_AVAILABLE = False

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
            llm_base_url: Base URL for LLM API
            tts_base_url: Base URL for TTS API
        """
        # Store configuration values
        self.llm_base_url = llm_base_url
        self.tts_base_url = tts_base_url
        
        # Initialize AgenDev instance to None initially
        self.agendev_instance = None
        
        # Import AgenDev conditionally to handle potential import errors
        try:
            from ..core import AgenDev, AgenDevConfig
            self.AgenDev = AgenDev
            self.AgenDevConfig = AgenDevConfig
            self.AGENDEV_AVAILABLE = True
            logger.info("AgenDev core components successfully imported")
        except ImportError as e:
            logger.error(f"Failed to import AgenDev core components: {e}")
            self.AGENDEV_AVAILABLE = False
    
    def _handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """
        Handle errors gracefully with proper logging.
        
        Args:
            error: The exception that occurred
            context: Context description for the error
            
        Returns:
            Error response dictionary
        """
        error_msg = str(error)
        logger.error(f"Error in {context}: {error_msg}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": error_msg,
            "context": context
        }
    
    def initialize_project(self, project_name: str, project_description: str) -> Dict[str, Any]:
        """
        Initialize a new project.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            
        Returns:
            Dictionary with initialization results
        """
        try:
            if not self.AGENDEV_AVAILABLE:
                raise ImportError("AgenDev core module is not available")
                
            # Import the required components
            from ..core import AgenDevConfig, TaskType, TaskPriority, TaskRisk
            
            # Configure workspace directory based on project name
            workspace_dir = os.path.join("workspace", project_name.lower().replace(' ', '_'))
            
            # Create AgenDev configuration
            config = self.AgenDevConfig(
                project_name=project_name,
                workspace_dir=workspace_dir,
                llm_base_url=self.llm_base_url,
                tts_base_url=self.tts_base_url
            )
            
            # Initialize AgenDev instance
            logger.info(f"Initializing AgenDev instance for project: {project_name}")
            self.agendev_instance = self.AgenDev(config)
            
            # Create initial tasks based on project description
            logger.info(f"Creating initial tasks for project: {project_name}")
            
            # Define core tasks for the project
            tasks = [
                {
                    "title": "Initialize project directory",
                    "description": "Set up project directory and initial file structure",
                    "type": TaskType.PLANNING,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.LOW,
                    "duration": 0.5
                },
                {
                    "title": "Create basic project structure",
                    "description": "Set up directories and configuration files",
                    "type": TaskType.PLANNING,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.LOW,
                    "duration": 0.5
                },
                {
                    "title": f"Implement core functionality",
                    "description": f"Write code for main features of {project_name}: {project_description}",
                    "type": TaskType.IMPLEMENTATION,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.MEDIUM,
                    "duration": 2.0
                },
                {
                    "title": "Add unit tests",
                    "description": "Write tests for the implemented functionality",
                    "type": TaskType.TEST,
                    "priority": TaskPriority.MEDIUM,
                    "risk": TaskRisk.LOW,
                    "duration": 1.0
                },
                {
                    "title": "Create documentation",
                    "description": "Write documentation for the project",
                    "type": TaskType.DOCUMENTATION,
                    "priority": TaskPriority.MEDIUM,
                    "risk": TaskRisk.LOW,
                    "duration": 1.0
                }
            ]
            
            # Add tasks to AgenDev instance
            task_ids = []
            previous_task_id = None
            
            for task in tasks:
                # Create task with optional dependency on previous task
                dependencies = [previous_task_id] if previous_task_id is not None else None
                
                task_id = self.agendev_instance.create_task(
                    title=task["title"],
                    description=task["description"],
                    task_type=task["type"],
                    priority=task["priority"],
                    risk=task["risk"],
                    estimated_duration_hours=task["duration"],
                    dependencies=dependencies
                )
                
                task_ids.append(task_id)
                previous_task_id = task_id
            
            # Generate implementation plan
            logger.info("Generating implementation plan")
            plan = self.agendev_instance.generate_implementation_plan()
            
            return {
                "success": True,
                "project_name": project_name,
                "task_ids": [str(task_id) for task_id in task_ids],
                "plan_id": str(plan.id) if plan else None,
                "status": self.agendev_instance.get_project_status()
            }
                
        except Exception as e:
            return self._handle_error(e, f"initialize_project({project_name})")
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks for the current project.
        
        Returns:
            List of task details
        """
        if not self.agendev_instance:
            return []
            
        try:
            if not self.AGENDEV_AVAILABLE:
                logger.warning("AgenDev not available for get_tasks")
                return []
                
            # Get tasks from AgenDev instance
            tasks = []
            for task_id, task in self.agendev_instance.task_graph.tasks.items():
                tasks.append({
                    "id": str(task_id),
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "dependencies": [str(dep_id) for dep_id in task.dependencies],
                    "artifact_paths": task.artifact_paths,
                    "priority": task.priority.value,
                    "risk": task.risk.value,
                    "completion_percentage": task.completion_percentage
                })
            return tasks
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            traceback.print_exc()
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
            return {"success": False, "error": "No active project"}
            
        try:
            logger.info(f"Executing task with ID: {task_id}")
            
            if not self.AGENDEV_AVAILABLE:
                raise ImportError("AgenDev core module is required for task execution")
                
            # Convert string ID to UUID if needed
            if not isinstance(task_id, UUID):
                task_id = UUID(task_id)
            
            # Execute task with AgenDev instance
            logger.info(f"Calling implement_task on AgenDev instance for task {task_id}")
            result = self.agendev_instance.implement_task(task_id)
            logger.info(f"Task implementation complete with status: {result.get('success', False)}")
            
            if not result.get("success", False):
                logger.error(f"Task implementation failed: {result.get('error', 'Unknown error')}")
                return result
            
            # Get updated task information
            task = None
            if task_id in self.agendev_instance.task_graph.tasks:
                task = self.agendev_instance.task_graph.tasks[task_id]
            
            # Return execution results with additional metadata
            return {
                "success": True,
                "task_id": str(task_id),
                "implementation": result.get("implementation", ""),
                "file_path": result.get("file_path", ""),
                "completed": task.status.value == "completed" if task else True,
                "snapshot_id": result.get("snapshot_id"),
                "metadata": {
                    "title": task.title if task else "Unknown Task",
                    "description": task.description if task else "",
                    "status": task.status.value if task else "unknown",
                    "artifact_paths": task.artifact_paths if task else []
                }
            }
                
        except Exception as e:
            return self._handle_error(e, f"execute_task({task_id})")
    
    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project.
        
        Returns:
            Dictionary with status information
        """
        if not self.agendev_instance:
            return {"success": False, "error": "No active project"}
            
        try:
            if not self.AGENDEV_AVAILABLE:
                logger.warning("AgenDev not available for get_project_status")
                return {
                    "success": False,
                    "error": "AgenDev not available",
                    "status": "unknown"
                }
                
            # Get status from AgenDev
            status = self.agendev_instance.get_project_status()
            return {"success": True, **status}
        except Exception as e:
            return self._handle_error(e, "get_project_status")
    
    def generate_todo_markdown(self, project_name: str, tasks: List[Dict[str, Any]]) -> str:
        """
        Generate markdown content for todo.md.
        
        Args:
            project_name: Name of the project
            tasks: List of tasks
            
        Returns:
            Markdown content for todo.md
        """
        # Group tasks by status
        task_groups = {}
        for task in tasks:
            status = task.get("status", "planned")
            if status not in task_groups:
                task_groups[status] = []
            task_groups[status].append(task)
        
        # Create markdown content
        markdown = f"# {project_name}\n\n"
        
        # Add sections for each status
        status_order = ["planned", "in_progress", "blocked", "completed", "failed"]
        section_titles = {
            "planned": "Planned Tasks",
            "in_progress": "In Progress",
            "blocked": "Blocked Tasks",
            "completed": "Completed Tasks",
            "failed": "Failed Tasks"
        }
        
        for status in status_order:
            if status in task_groups and task_groups[status]:
                markdown += f"## {section_titles.get(status, status.title())}\n"
                
                for task in task_groups[status]:
                    # Add checkbox for completed tasks
                    if status == "completed":
                        markdown += f"- [x] {task['title']}\n"
                    else:
                        markdown += f"- [ ] {task['title']}\n"
                        
                    # Add description as indented text if available
                    if task.get("description"):
                        markdown += f"  - {task['description']}\n"
                        
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
            # Create error step for failed execution
            return [{
                "type": "terminal",
                "content": f"$ echo 'Task execution failed'\nError: {task_execution_result.get('error', 'Unknown error')}",
                "operation_type": "Error",
                "file_path": "Task execution"
            }]
            
        steps = []
        task_id = task_execution_result.get("task_id", "")
        file_path = task_execution_result.get("file_path", "")
        implementation = task_execution_result.get("implementation", "")
        metadata = task_execution_result.get("metadata", {})
        task_title = metadata.get("title", "")
        
        if not task_title and self.agendev_instance:
            # Try to get task title from instance
            if self.AGENDEV_AVAILABLE:
                if not isinstance(task_id, UUID):
                    task_id = UUID(task_id)
                    
                if task_id in self.agendev_instance.task_graph.tasks:
                    task_title = self.agendev_instance.task_graph.tasks[task_id].title
        
        # Step 1: Planning (no version control operation)
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
        
        # Step 3: Saving (file system operations only)
        steps.append({
            "type": "terminal",
            "content": f"$ echo 'Saving implementation to {file_path}'\nSaving implementation to {file_path}\n[SUCCESS] Implementation saved to {file_path}",
            "operation_type": "Saving",
            "file_path": file_path
        })
        
        return steps