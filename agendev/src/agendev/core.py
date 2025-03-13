# core.py
"""Core orchestration module for AgenDev system."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple, Literal
from enum import Enum
from pathlib import Path
import os
import time
import logging
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

# Import models
from .models.task_models import (
    Task, TaskStatus, TaskPriority, TaskRisk, TaskType, 
    Epic, TaskGraph, Dependency
)
from .models.planning_models import (
    SimulationConfig, PlanSnapshot, PlanningHistory, 
    SearchNodeType, PlanningPhase, SimulationResult
)

# Import components
from .utils.fs_utils import resolve_path, ensure_workspace_structure, save_json, load_json
from .context_management import ContextManager, ContextElement
from .llm_integration import LLMIntegration, LLMConfig
from .tts_notification import NotificationManager, NotificationType, NotificationPriority
from .search_algorithms import MCTSPlanner, AStarPathfinder
from .probability_modeling import TaskProbabilityModel, ProjectRiskModel
from .snapshot_engine import SnapshotEngine
from .parameter_controller import ParameterController
from .test_generation import TestGenerator, TestType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectState(str, Enum):
    """States of the project lifecycle."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    COMPLETED = "completed"

class AgenDevConfig(BaseModel):
    """Configuration for the AgenDev system."""
    project_name: str
    workspace_dir: Optional[str] = None
    llm_base_url: str = "http://192.168.2.12:1234"
    tts_base_url: str = "http://127.0.0.1:7860"
    default_model: str = "qwen2.5-7b-instruct"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    notifications_enabled: bool = True
    auto_save: bool = True
    auto_save_interval_minutes: float = 5.0
    
    @model_validator(mode='after')
    def validate_config(self) -> 'AgenDevConfig':
        """Ensure configuration values are valid."""
        if self.auto_save_interval_minutes <= 0:
            self.auto_save_interval_minutes = 5.0
        return self

class AgenDev:
    """Main orchestration class for the AgenDev system."""
    
    def __init__(self, config: Optional[AgenDevConfig] = None):
        """
        Initialize the AgenDev system.
        
        Args:
            config: Configuration for the system
        """
        self.config = config or AgenDevConfig(project_name="agendev_project")
        self.project_state = ProjectState.INITIALIZED
        self.last_save_time = time.time()
        
        # Set up workspace
        self.workspace_dir = resolve_path(self.config.workspace_dir or "")
        ensure_workspace_structure()
        
        # Initialize components
        self._init_components()
        
        # Set up default models and data structures
        self.task_graph = TaskGraph()
        self.planning_history = PlanningHistory()
        
        # Load project if it exists
        self._load_project_state()
        
        # Announce system initialization
        if self.notification_manager:
            self.notification_manager.info(
                f"AgenDev system initialized for project '{self.config.project_name}'."
            )
    
    def _init_components(self) -> None:
        """Initialize all system components."""
        # Initialize LLM integration
        self.llm = LLMIntegration(
            base_url=self.config.llm_base_url,
            config=LLMConfig(model=self.config.default_model)
        )
        
        # Initialize context management
        self.context_manager = ContextManager(
            embedding_model=self.config.embedding_model,
            llm_base_url=self.config.llm_base_url
        )
        
        # Initialize TTS notification
        if self.config.notifications_enabled:
            self.notification_manager = NotificationManager(
                tts_base_url=self.config.tts_base_url
            )
        else:
            self.notification_manager = None
        
        # Initialize parameter controller
        self.parameter_controller = ParameterController()
        
        # Initialize snapshot engine
        self.snapshot_engine = SnapshotEngine(workspace_dir=self.workspace_dir)
        
        # Initialize test generator
        self.test_generator = TestGenerator(
            llm_integration=self.llm,
            context_manager=self.context_manager
        )
        
        # The following components will be initialized as needed:
        # - MCTSPlanner (needs task graph)
        # - AStarPathfinder (needs task graph)
        # - TaskProbabilityModel (needs task graph)
        # - ProjectRiskModel (needs probability model and task graph)
    
    def _load_project_state(self) -> None:
        """Load project state from disk."""
        tasks_path = resolve_path("planning/tasks.json")
        epics_path = resolve_path("planning/epics.json")
        
        if tasks_path.exists() and epics_path.exists():
            try:
                # Load tasks
                tasks_data = load_json(tasks_path)
                for task_dict in tasks_data.get("tasks", []):
                    task = Task.model_validate(task_dict)
                    self.task_graph.tasks[task.id] = task
                
                # Load epics
                epics_data = load_json(epics_path)
                for epic_dict in epics_data.get("epics", []):
                    epic = Epic.model_validate(epic_dict)
                    self.task_graph.epics[epic.id] = epic
                
                # Load planning history if it exists
                history_path = resolve_path("planning/planning_history.json")
                if history_path.exists():
                    history_data = load_json(history_path)
                    self.planning_history = PlanningHistory.model_validate(history_data)
                
                # Notify about project load
                if self.notification_manager:
                    self.notification_manager.info(
                        f"Project {self.config.project_name} loaded with "
                        f"{len(self.task_graph.tasks)} tasks and {len(self.task_graph.epics)} epics."
                    )
            except Exception as e:
                logger.error(f"Error loading project state: {e}")
    
    def _save_project_state(self) -> bool:
        """
        Save project state to disk.
        
        Returns:
            Whether the save was successful
        """
        try:
            # Save tasks
            tasks_data = {
                "tasks": [task.model_dump() for task in self.task_graph.tasks.values()]
            }
            save_json(tasks_data, resolve_path("planning/tasks.json"))
            
            # Save epics
            epics_data = {
                "epics": [epic.model_dump() for epic in self.task_graph.epics.values()]
            }
            save_json(epics_data, resolve_path("planning/epics.json"))
            
            # Save planning history
            history_data = self.planning_history.model_dump()
            save_json(history_data, resolve_path("planning/planning_history.json"))
            
            # Update last save time
            self.last_save_time = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Error saving project state: {e}")
            if self.notification_manager:
                self.notification_manager.error(f"Failed to save project state: {e}")
            return False
    
    def _check_auto_save(self) -> None:
        """Check if it's time to auto-save the project state."""
        if self.config.auto_save:
            current_time = time.time()
            minutes_since_save = (current_time - self.last_save_time) / 60
            if minutes_since_save >= self.config.auto_save_interval_minutes:
                if self._save_project_state():
                    logger.info("Auto-saved project state")
    
    def create_task(
        self,
        title: str,
        description: str,
        task_type: TaskType = TaskType.IMPLEMENTATION,
        priority: TaskPriority = TaskPriority.MEDIUM,
        risk: TaskRisk = TaskRisk.MEDIUM,
        estimated_duration_hours: float = 1.0,
        epic_id: Optional[UUID] = None,
        dependencies: Optional[List[UUID]] = None
    ) -> UUID:
        """
        Create a new task.
        
        Args:
            title: Task title
            description: Task description
            task_type: Type of task
            priority: Task priority
            risk: Risk level
            estimated_duration_hours: Estimated duration in hours
            epic_id: Optional epic to associate with
            dependencies: Optional task dependencies
            
        Returns:
            ID of the created task
        """
        # Create task object
        task = Task(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            risk=risk,
            estimated_duration_hours=estimated_duration_hours,
            epic_id=epic_id,
            dependencies=dependencies or []
        )
        
        # Add to task graph
        task_id = self.task_graph.add_task(task)
        
        # Add dependencies if provided
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.task_graph.tasks:
                    dependency = Dependency(
                        source_id=dep_id,
                        target_id=task_id,
                        dependency_type="blocks"
                    )
                    self.task_graph.add_dependency(dependency)
        
        # Update task statuses
        self.task_graph.update_task_statuses()
        
        # Notify about task creation
        if self.notification_manager:
            self.notification_manager.info(f"Task '{title}' created.")
        
        # Auto-save if needed
        self._check_auto_save()
        
        return task_id
    
    def create_epic(
        self,
        title: str,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        risk: TaskRisk = TaskRisk.MEDIUM
    ) -> UUID:
        """
        Create a new epic.
        
        Args:
            title: Epic title
            description: Epic description
            priority: Epic priority
            risk: Risk level
            
        Returns:
            ID of the created epic
        """
        # Create epic object
        epic = Epic(
            title=title,
            description=description,
            priority=priority,
            risk=risk
        )
        
        # Add to task graph
        epic_id = self.task_graph.add_epic(epic)
        
        # Notify about epic creation
        if self.notification_manager:
            self.notification_manager.info(f"Epic '{title}' created.")
        
        # Auto-save if needed
        self._check_auto_save()
        
        return epic_id
    
    def generate_implementation_plan(self, max_iterations: int = 1000) -> PlanSnapshot:
        """
        Generate an implementation plan using MCTS.
        
        Args:
            max_iterations: Maximum MCTS iterations
            
        Returns:
            Plan snapshot with the generated plan
        """
        # Update project state
        self.project_state = ProjectState.PLANNING
        
        # Initialize probability model
        probability_model = TaskProbabilityModel(
            task_graph=self.task_graph, 
            llm_integration=self.llm
        )
        
        # Initialize MCTS planner
        planner = MCTSPlanner(
            task_graph=self.task_graph,
            llm_integration=self.llm,
            config=SimulationConfig(max_iterations=max_iterations)
        )
        
        # Notify about planning start
        if self.notification_manager:
            self.notification_manager.info(f"Generating implementation plan...")
        
        # Run simulation
        simulation_results = planner.run_simulation(iterations=max_iterations)
        
        # Get best sequence
        task_sequence = simulation_results["best_sequence"]
        
        # Ensure task_sequence is not empty - add at least one default task if needed
        if not task_sequence and self.task_graph.tasks:
            # Add first task as fallback
            task_sequence = [next(iter(self.task_graph.tasks.keys()))]
            
        # If still empty, create a dummy task
        if not task_sequence:
            dummy_task = Task(
                title="Project Setup",
                description="Initialize the project and prepare the environment",
                task_type=TaskType.PLANNING,
                priority=TaskPriority.HIGH,
                risk=TaskRisk.LOW,
                estimated_duration_hours=1.0
            )
            task_id = self.task_graph.add_task(dummy_task)
            task_sequence = [task_id]
        
        # Calculate expected duration
        total_duration = sum(
            self.task_graph.tasks[task_id].estimated_duration_hours
            for task_id in task_sequence
            if task_id in self.task_graph.tasks
        )
        
        # Create risk assessment
        risk_model = ProjectRiskModel(task_probability_model=probability_model, task_graph=self.task_graph)
        risk_hotspots = risk_model.identify_risk_hotspots()
        risk_assessment = {
            str(item["task_id"]): item["success_probability"]
            for item in risk_hotspots
        }
        
        # If risk_assessment is empty, add a default entry
        if not risk_assessment and task_sequence:
            # Add a default risk assessment for the first task
            risk_assessment = {str(task_sequence[0]): 0.8}  # Default 80% success probability
        
        # Create plan snapshot
        plan = PlanSnapshot(
            plan_version=len(self.planning_history.snapshots) + 1,
            task_sequence=task_sequence,
            expected_duration_hours=total_duration or 1.0,  # Default to 1 hour if no tasks
            confidence_score=simulation_results.get("success_rate", 0.7),  # Default confidence if missing
            risk_assessment=risk_assessment,
            simulation_id=UUID(simulation_results["session_id"]) if "session_id" in simulation_results else None,
            generated_by="mcts",
            description=f"Plan generated with {max_iterations} iterations, {simulation_results.get('success_rate', 0.0):.2f} success rate"
        )
        
        # Add to planning history
        plan_id = self.planning_history.add_snapshot(plan)
        
        # Notify about plan generation
        if self.notification_manager:
            self.notification_manager.success(
                f"Implementation plan generated with {len(task_sequence)} tasks, "
                f"estimated duration: {total_duration:.1f} hours, "
                f"confidence: {simulation_results.get('success_rate', 0.0):.0%}"
            )
        
        # Save project state
        self._save_project_state()
        
        # Update project state
        self.project_state = ProjectState.IMPLEMENTING
        
        return plan
    
    def optimize_task_sequence(self) -> List[UUID]:
        """
        Optimize the task sequence using A* pathfinding.
        
        Returns:
            Optimized task sequence
        """
        # Initialize A* pathfinder
        pathfinder = AStarPathfinder(task_graph=self.task_graph)
        
        # Find optimal path
        path, metadata = pathfinder.find_path()
        
        # Notify about optimization
        if self.notification_manager and path:
            self.notification_manager.info(
                f"Task sequence optimized with {len(path)} tasks, "
                f"estimated duration: {metadata.get('estimated_duration', 0):.1f} hours."
            )
        
        return path
    
    def implement_task(self, task_id: UUID) -> Dict[str, Any]:
        """
        Implement a task using LLM.
        
        Args:
            task_id: ID of the task to implement
            
        Returns:
            Dictionary with implementation details
        """
        if task_id not in self.task_graph.tasks:
            error_msg = f"Task not found: {task_id}"
            if self.notification_manager:
                self.notification_manager.error(error_msg)
            return {"error": error_msg}
        
        task = self.task_graph.tasks[task_id]
        
        # Notify about implementation start
        if self.notification_manager:
            self.notification_manager.info(f"Implementing task: {task.title}")
        
        # Wrap everything in try/except for detailed error logging
        try:
            print(f"Starting implementation of task: {task.title} ({task_id})")
            
            # Get optimal parameters for this task
            llm_config = self.parameter_controller.get_llm_config(task)
            print(f"Using LLM config: temperature={llm_config.temperature}, max_tokens={llm_config.max_tokens}")
            
            # Generate implementation context
            print("Generating context for task...")
            context_str = ""
            if self.context_manager:
                try:
                    context = self.context_manager.generate_context_for_task(task.description, top_k=5)
                    context_str = "\n\n".join(
                        f"File: {elem['source_file']}\n```\n{elem['content']}\n```"
                        for elem in context.get("elements", [])
                    )
                    print(f"Generated context with {len(context.get('elements', []))} elements")
                except Exception as context_error:
                    print(f"Error generating context: {context_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue without context
            
            # Generate implementation
            prompt = f"""
            Task: {task.title}
            Description: {task.description}
            
            {context_str}
            
            Implement this task according to the description. 
            Generate clean, well-structured, production-ready code.
            Include detailed comments explaining the code.
            """
            
            print("Sending prompt to LLM for implementation...")
            
            # Use LLM to generate implementation with explicit error handling
            try:
                # Make sure not to clear context in case that's causing issues
                implementation = self.llm.query(
                    prompt=prompt, 
                    config=llm_config,
                    clear_context=False,
                    save_to_context=True
                )
                print(f"Received implementation response of length {len(implementation)}")
            except Exception as llm_error:
                print(f"LLM error during implementation: {llm_error}")
                import traceback
                traceback.print_exc()
                raise Exception(f"Failed to generate implementation: {llm_error}")
            
            # Extract code from implementation (removing any markdown formatting)
            code = implementation
            if "```" in implementation:
                import re
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", implementation, re.DOTALL)
                if code_blocks:
                    code = "\n\n".join(code_blocks)
                    print(f"Extracted {len(code_blocks)} code blocks")
            
            # Create safe file name from task title
            import re
            safe_filename = re.sub(r'[^\w\-_\.]', '_', task.title.lower().replace(' ', '_'))
            file_path = f"src/{safe_filename}.py"
            print(f"Will save implementation to {file_path}")
            
            # Make sure the src directory exists
            src_dir = resolve_path("src", create_parents=True)
            
            # Create a snapshot with timeout
            import threading
            import time
            
            snapshot_completed = False
            snapshot_exception = None
            snapshot_metadata = None
            
            def create_snapshot_with_timeout():
                nonlocal snapshot_completed, snapshot_exception, snapshot_metadata
                try:
                    snapshot_metadata = self.snapshot_engine.create_snapshot(
                        file_path=file_path,
                        content=code,
                        message=f"Implementation of {task.title}",
                        tags=[task.task_type.value]
                    )
                    snapshot_completed = True
                except Exception as e:
                    snapshot_exception = e
                    
            # Start snapshot creation in a separate thread
            snapshot_thread = threading.Thread(target=create_snapshot_with_timeout)
            snapshot_thread.daemon = True
            snapshot_thread.start()
            
            # Wait for completion with timeout (10 seconds)
            timeout = 10  # seconds
            start_time = time.time()
            while not snapshot_completed and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not snapshot_completed:
                print(f"WARNING: Snapshot creation timed out after {timeout} seconds, continuing without snapshot")
                if snapshot_exception:
                    print(f"Snapshot exception: {snapshot_exception}")
            
            # Actually save the implementation file with timeout
            file_saved = False
            file_exception = None
            
            def save_file_with_timeout():
                nonlocal file_saved, file_exception
                try:
                    full_path = resolve_path(file_path, create_parents=True)
                    with open(full_path, 'w') as f:
                        f.write(code)
                    file_saved = True
                except Exception as e:
                    file_exception = e
            
            # Start file saving in a separate thread
            file_thread = threading.Thread(target=save_file_with_timeout)
            file_thread.daemon = True
            file_thread.start()
            
            # Wait for completion with timeout (5 seconds)
            timeout = 5  # seconds
            start_time = time.time()
            while not file_saved and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not file_saved:
                print(f"WARNING: File saving timed out after {timeout} seconds")
                if file_exception:
                    print(f"File exception: {file_exception}")
                raise TimeoutError(f"File saving operation timed out after {timeout} seconds")
                
            # Update task status
            old_status = task.status
            task.status = TaskStatus.COMPLETED
            task.completion_percentage = 100.0
            task.actual_duration_hours = task.estimated_duration_hours  # In real system, we'd track actual time
            task.artifact_paths.append(file_path)
            
            # Notify about task completion
            if self.notification_manager:
                self.notification_manager.task_status_update(task, old_status)
            
            # Update dependencies with timeout
            deps_completed = False
            
            def update_deps_with_timeout():
                nonlocal deps_completed
                self.task_graph.update_task_statuses()
                deps_completed = True
            
            # Start dependency update in a separate thread
            deps_thread = threading.Thread(target=update_deps_with_timeout)
            deps_thread.daemon = True
            deps_thread.start()
            
            # Wait for completion with timeout (5 seconds)
            timeout = 5  # seconds
            start_time = time.time()
            while not deps_completed and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not deps_completed:
                print(f"WARNING: Dependencies update timed out after {timeout} seconds, continuing anyway")
            
            # Auto-save with timeout
            save_completed = False
            
            def save_project_with_timeout():
                nonlocal save_completed
                self._save_project_state()
                save_completed = True
            
            # Start project saving in a separate thread
            save_thread = threading.Thread(target=save_project_with_timeout)
            save_thread.daemon = True
            save_thread.start()
            
            # Wait for completion with timeout (5 seconds)
            timeout = 5  # seconds
            start_time = time.time()
            while not save_completed and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not save_completed:
                print(f"WARNING: Project state saving timed out after {timeout} seconds, continuing anyway")
            
            return {
                "success": True,
                "task_id": str(task_id),
                "implementation": implementation,
                "file_path": file_path,
                "snapshot_id": snapshot_metadata.snapshot_id if snapshot_metadata else None
            }
            
        except Exception as e:
            error_msg = f"Failed to implement task '{task.title}': {str(e)}"
            print(f"Implementation error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Update task status to failed
            old_status = task.status
            task.status = TaskStatus.FAILED
            
            # Notify about failure
            if self.notification_manager:
                self.notification_manager.error(error_msg)
            
            # Save state even on failure
            self._save_project_state()
            
            return {
                "success": False,
                "task_id": str(task_id),
                "error": error_msg
            }
    
    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project.
        
        Returns:
            Dictionary with project status details
        """
        # Count tasks by status
        status_counts = {}
        for task in self.task_graph.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        # Calculate progress
        total_tasks = len(self.task_graph.tasks)
        completed_tasks = status_counts.get(TaskStatus.COMPLETED.value, 0)
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Get current plan
        current_plan = self.planning_history.get_current_plan()
        
        # Get risk assessment
        risk_assessment = {}
        if current_plan:
            risk_assessment = current_plan.risk_assessment
        
        # Compile status
        status = {
            "project_name": self.config.project_name,
            "state": self.project_state.value,
            "tasks": {
                "total": total_tasks,
                "by_status": status_counts
            },
            "epics": {
                "total": len(self.task_graph.epics),
                "completed": sum(1 for epic in self.task_graph.epics.values() 
                                if epic.milestone_percentage >= 100)
            },
            "progress": {
                "percentage": progress_percentage,
                "estimated_remaining_hours": sum(task.estimated_duration_hours 
                                               for task in self.task_graph.tasks.values()
                                               if task.status != TaskStatus.COMPLETED)
            },
            "current_plan": {
                "id": str(current_plan.id) if current_plan else None,
                "confidence": current_plan.confidence_score if current_plan else None,
                "remaining_tasks": len(current_plan.task_sequence) if current_plan else 0
            },
            "risk_assessment": risk_assessment
        }
        
        return status
    
    def summarize_progress(self, voice_summary: bool = True) -> str:
        """
        Generate a summary of project progress.
        
        Args:
            voice_summary: Whether to generate a voice summary
            
        Returns:
            Text summary of progress
        """
        # Get project status
        status = self.get_project_status()
        
        # Create summary text
        summary = f"""
        Project: {status['project_name']}
        State: {status['state']}
        
        Progress: {status['progress']['percentage']:.1f}% complete
        Tasks: {status['tasks']['total']} total, {status['tasks'].get('by_status', {}).get(TaskStatus.COMPLETED.value, 0)} completed
        Epics: {status['epics']['total']} total, {status['epics']['completed']} completed
        
        Estimated remaining work: {status['progress']['estimated_remaining_hours']:.1f} hours
        """
        
        # Generate voice summary if requested
        if voice_summary and self.notification_manager:
            self.notification_manager.progress(
                f"Project {status['project_name']} is {status['progress']['percentage']:.0f}% complete with "
                f"{status['tasks'].get('by_status', {}).get(TaskStatus.COMPLETED.value, 0)} of {status['tasks']['total']} "
                f"tasks completed. Estimated remaining work is {status['progress']['estimated_remaining_hours']:.1f} hours."
            )
        
        return summary.strip()
    
    def shutdown(self) -> None:
        """Save state and shut down the system."""
        # Save project state
        self._save_project_state()
        
        # Notify about shutdown
        if self.notification_manager:
            self.notification_manager.info(f"AgenDev system shutting down, project state saved.")
            
        logger.info(f"AgenDev system shutdown complete for project '{self.config.project_name}'")