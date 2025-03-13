# core.py
"""Core orchestration module for AgenDev system."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple, Literal
from enum import Enum
from pathlib import Path
import os
import time
import logging
import re
import traceback
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

class ErrorCategory(str, Enum):
    """Categories of errors that can occur during task execution."""
    LLM_SERVICE = "llm_service"  # Issues with the LLM service (network, timeout)
    LLM_RESPONSE = "llm_response"  # Issues with the LLM response (malformed, incomplete)
    CODE_QUALITY = "code_quality"  # Issues with the generated code (syntax errors, bugs)
    FILESYSTEM = "filesystem"  # Issues with file operations
    DEPENDENCY = "dependency"  # Issues with task dependencies
    RESOURCE = "resource"  # Resource constraints (memory, CPU)
    UNKNOWN = "unknown"  # Unclassified errors

class RetryStrategy(str, Enum):
    """Strategies for retrying failed tasks."""
    NONE = "none"  # No retry
    SIMPLE = "simple"  # Simple retry with same parameters
    ADJUST_PARAMETERS = "adjust_parameters"  # Retry with adjusted parameters
    SIMPLIFY_CONTEXT = "simplify_context"  # Retry with simplified context
    DECOMPOSE_TASK = "decompose_task"  # Break task into smaller subtasks
    CHANGE_APPROACH = "change_approach"  # Use a completely different approach

class ExecutionResult(BaseModel):
    """Result of a task execution attempt."""
    success: bool = False
    task_id: UUID
    implementation: str = ""
    file_path: str = ""
    error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    snapshot_id: Optional[str] = None
    retry_strategy: RetryStrategy = RetryStrategy.NONE
    retry_count: int = 0
    parameter_adjustments: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    
    # Task execution configuration
    max_retry_attempts: int = 3
    execution_timeout: float = 60.0  # Seconds
    preserve_work_on_failure: bool = True
    
    @model_validator(mode='after')
    def validate_config(self) -> 'AgenDevConfig':
        """Ensure configuration values are valid."""
        if self.auto_save_interval_minutes <= 0:
            self.auto_save_interval_minutes = 5.0
        if self.max_retry_attempts < 0:
            self.max_retry_attempts = 3
        if self.execution_timeout <= 0:
            self.execution_timeout = 60.0
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
    
    def _classify_error(self, error: Exception, context: str) -> Tuple[ErrorCategory, str, RetryStrategy]:
        """
        Classify an error to determine its category and appropriate retry strategy.
        
        Args:
            error: The exception that occurred
            context: The context in which the error occurred
            
        Returns:
            Tuple of (error_category, error_message, retry_strategy)
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Determine error category
        if error_type == "TimeoutError" or "timeout" in error_msg.lower():
            category = ErrorCategory.LLM_SERVICE
            retry_strategy = RetryStrategy.SIMPLE
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            category = ErrorCategory.LLM_SERVICE
            retry_strategy = RetryStrategy.SIMPLE
        elif "response" in error_msg.lower() or "parse" in error_msg.lower() or "json" in error_msg.lower():
            category = ErrorCategory.LLM_RESPONSE
            retry_strategy = RetryStrategy.ADJUST_PARAMETERS
        elif "syntax error" in error_msg.lower() or "invalid syntax" in error_msg.lower():
            category = ErrorCategory.CODE_QUALITY
            retry_strategy = RetryStrategy.ADJUST_PARAMETERS
        elif "file" in error_msg.lower() or "permission" in error_msg.lower() or "directory" in error_msg.lower():
            category = ErrorCategory.FILESYSTEM
            retry_strategy = RetryStrategy.SIMPLE
        elif "memory" in error_msg.lower() or "resource" in error_msg.lower():
            category = ErrorCategory.RESOURCE
            retry_strategy = RetryStrategy.SIMPLIFY_CONTEXT
        elif "dependency" in error_msg.lower() or "import" in error_msg.lower():
            category = ErrorCategory.DEPENDENCY
            retry_strategy = RetryStrategy.ADJUST_PARAMETERS
        else:
            category = ErrorCategory.UNKNOWN
            retry_strategy = RetryStrategy.ADJUST_PARAMETERS
        
        # Format detailed error message
        detail_msg = f"{error_type} in {context}: {error_msg}"
        
        return category, detail_msg, retry_strategy
    
    def _adjust_parameters_for_retry(
        self, 
        task: Task, 
        error_category: ErrorCategory, 
        attempt: int
    ) -> LLMConfig:
        """
        Adjust LLM parameters based on error type and retry attempt.
        
        Args:
            task: The task being executed
            error_category: Category of the error
            attempt: The current retry attempt number
            
        Returns:
            Adjusted LLM configuration
        """
        # Get base configuration
        base_config = self.parameter_controller.get_llm_config(task)
        
        # Make adjustments based on error category
        if error_category == ErrorCategory.CODE_QUALITY:
            # Reduce temperature for more precise code generation
            temperature = max(0.1, base_config.temperature * (0.8 ** attempt))
            # Increase max_tokens for more complete responses
            max_tokens = min(4000, int(base_config.max_tokens * 1.2))
        elif error_category == ErrorCategory.LLM_RESPONSE:
            # Reduce temperature for more deterministic responses
            temperature = max(0.1, base_config.temperature * (0.7 ** attempt))
            # Slightly increase max_tokens
            max_tokens = min(4000, int(base_config.max_tokens * 1.1))
        elif error_category == ErrorCategory.RESOURCE:
            # Reduce max_tokens to conserve resources
            max_tokens = max(500, int(base_config.max_tokens * (0.8 ** attempt)))
            # Keep temperature the same
            temperature = base_config.temperature
        else:
            # Default adjustment: slight reduction in temperature
            temperature = max(0.1, base_config.temperature * (0.9 ** attempt))
            max_tokens = base_config.max_tokens
        
        # Create new config with adjusted parameters
        return LLMConfig(
            model=base_config.model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def implement_task(self, task_id: UUID) -> Dict[str, Any]:
        """
        Implement a task using LLM with enhanced error handling and retries.
        
        Args:
            task_id: ID of the task to implement
            
        Returns:
            Dictionary with implementation details
        """
        if task_id not in self.task_graph.tasks:
            error_msg = f"Task not found: {task_id}"
            if self.notification_manager:
                self.notification_manager.error(error_msg)
            return {"success": False, "error": error_msg}
        
        task = self.task_graph.tasks[task_id]
        
        # Initialize execution result
        result = ExecutionResult(
            success=False,
            task_id=task_id
        )
        
        # Notify about implementation start
        if self.notification_manager:
            self.notification_manager.info(f"Implementing task: {task.title}")
        
        # Record execution start time
        start_time = time.time()
        
        # Initialize retry counter
        retry_count = 0
        max_retries = self.config.max_retry_attempts
        
        # Keep track of the latest snapshot ID for recovery
        latest_snapshot_id = None
        
        # Main implementation loop with retries
        while retry_count <= max_retries:
            try:
                # If this is a retry, log and notify
                if retry_count > 0:
                    retry_msg = f"Retry attempt {retry_count}/{max_retries} for task: {task.title}"
                    logger.info(retry_msg)
                    if self.notification_manager:
                        self.notification_manager.info(retry_msg)
                
                # Get optimal parameters for this task, adjusted for retries if needed
                if retry_count > 0 and result.error_category:
                    llm_config = self._adjust_parameters_for_retry(
                        task, 
                        result.error_category, 
                        retry_count
                    )
                    result.parameter_adjustments = {
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens
                    }
                else:
                    llm_config = self.parameter_controller.get_llm_config(task)
                
                logger.info(f"Using LLM config: temperature={llm_config.temperature}, max_tokens={llm_config.max_tokens}")
                
                # Generate implementation context
                logger.info("Generating context for task...")
                context_str = ""
                if self.context_manager:
                    try:
                        context = self.context_manager.generate_context_for_task(task.description, top_k=5)
                        context_str = "\n\n".join(
                            f"File: {elem['source_file']}\n```\n{elem['content']}\n```"
                            for elem in context.get("elements", [])
                        )
                        logger.info(f"Generated context with {len(context.get('elements', []))} elements")
                    except Exception as context_error:
                        logger.warning(f"Error generating context: {context_error}")
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
                
                logger.info("Sending prompt to LLM for implementation...")
                
                # Take snapshot of current state before LLM query (for work preservation)
                if self.config.preserve_work_on_failure and retry_count > 0 and result.implementation:
                    try:
                        # Only take a snapshot if we have an implementation from a previous attempt
                        recovery_snapshot_id = self.snapshot_engine.create_snapshot(
                            file_path=f"recovery/{task_id}.py",
                            content=result.implementation,
                            message=f"Recovery snapshot before retry {retry_count} for {task.title}",
                            tags=["recovery", task.task_type.value]
                        ).snapshot_id
                        logger.info(f"Created recovery snapshot {recovery_snapshot_id} before retry")
                    except Exception as snapshot_error:
                        logger.warning(f"Error creating recovery snapshot: {snapshot_error}")
                
                # Use LLM to generate implementation with explicit error handling
                try:
                    implementation = self.llm.query(
                        prompt=prompt, 
                        config=llm_config,
                        clear_context=False,
                        save_to_context=True
                    )
                    logger.info(f"Received implementation response of length {len(implementation)}")
                except Exception as llm_error:
                    logger.error(f"LLM error during implementation: {llm_error}")
                    error_category, error_detail, retry_strategy = self._classify_error(
                        llm_error, "LLM generation"
                    )
                    
                    # Update result with error information
                    result.error = error_detail
                    result.error_category = error_category
                    result.retry_strategy = retry_strategy
                    result.retry_count = retry_count
                    
                    # If we should retry, continue to next attempt
                    if retry_count < max_retries and retry_strategy != RetryStrategy.NONE:
                        retry_count += 1
                        continue
                    else:
                        # If we're out of retries or shouldn't retry, return error
                        raise Exception(f"Failed to generate implementation: {error_detail}")
                
                # Extract code from implementation (removing any markdown formatting)
                code = implementation
                if "```" in implementation:
                    import re
                    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", implementation, re.DOTALL)
                    if code_blocks:
                        code = "\n\n".join(code_blocks)
                        logger.info(f"Extracted {len(code_blocks)} code blocks")
                
                # Create safe file name from task title
                import re
                safe_filename = re.sub(r'[^\w\-_\.]', '_', task.title.lower().replace(' ', '_'))
                file_path = f"src/{safe_filename}.py"
                logger.info(f"Will save implementation to {file_path}")
                
                # Make sure the src directory exists
                src_dir = resolve_path("src", create_parents=True)
                
                # Create a snapshot
                try:
                    snapshot_metadata = self.snapshot_engine.create_snapshot(
                        file_path=file_path,
                        content=code,
                        message=f"Implementation of {task.title}",
                        tags=[task.task_type.value]
                    )
                    latest_snapshot_id = snapshot_metadata.snapshot_id
                    logger.info(f"Created snapshot {latest_snapshot_id}")
                except Exception as snapshot_error:
                    logger.warning(f"Error creating snapshot: {snapshot_error}")
                    error_category, error_detail, retry_strategy = self._classify_error(
                        snapshot_error, "snapshot creation"
                    )
                    
                    # Update result with error information but continue (non-critical error)
                    result.error = error_detail
                    result.error_category = error_category
                
                # Save the implementation file
                try:
                    full_path = resolve_path(file_path, create_parents=True)
                    with open(full_path, 'w') as f:
                        f.write(code)
                    logger.info(f"Saved implementation to {file_path}")
                except Exception as file_error:
                    logger.error(f"Error saving implementation file: {file_error}")
                    error_category, error_detail, retry_strategy = self._classify_error(
                        file_error, "file saving"
                    )
                    
                    # Update result with error information
                    result.error = error_detail
                    result.error_category = error_category
                    result.retry_strategy = retry_strategy
                    result.retry_count = retry_count
                    
                    # If we should retry, continue to next attempt
                    if retry_count < max_retries and retry_strategy != RetryStrategy.NONE:
                        retry_count += 1
                        continue
                    else:
                        # If we're out of retries or shouldn't retry, return error
                        raise Exception(f"Failed to save implementation file: {error_detail}")
                
                # Verify the implementation quality
                # This could include syntax checking or even running tests if available
                try:
                    self._verify_implementation_quality(code, file_path)
                except Exception as quality_error:
                    logger.error(f"Quality verification failed: {quality_error}")
                    error_category, error_detail, retry_strategy = self._classify_error(
                        quality_error, "quality verification"
                    )
                    
                    # Update result with error information
                    result.error = error_detail
                    result.error_category = error_category
                    result.retry_strategy = retry_strategy
                    result.retry_count = retry_count
                    
                    # If we should retry, continue to next attempt
                    if retry_count < max_retries and retry_strategy != RetryStrategy.NONE:
                        retry_count += 1
                        continue
                    else:
                        # If we're out of retries or shouldn't retry, return error
                        raise Exception(f"Implementation failed quality check: {error_detail}")
                
                # Update task status
                old_status = task.status
                task.status = TaskStatus.COMPLETED
                task.completion_percentage = 100.0
                task.actual_duration_hours = (time.time() - start_time) / 3600  # Convert seconds to hours
                task.artifact_paths.append(file_path)
                
                # Update result with success information
                result.success = True
                result.implementation = implementation
                result.file_path = file_path
                result.snapshot_id = latest_snapshot_id
                result.retry_count = retry_count
                result.execution_time = time.time() - start_time
                result.metadata = {
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "artifact_paths": task.artifact_paths
                }
                
                # Notify about task completion
                if self.notification_manager:
                    self.notification_manager.task_status_update(task, old_status)
                
                # Update dependencies and save state
                self.task_graph.update_task_statuses()
                self._save_project_state()
                
                # Task completed successfully, break the retry loop
                break
                
            except Exception as e:
                # Handle unexpected errors
                error_msg = f"Failed to implement task '{task.title}': {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # Classify the error
                error_category, error_detail, retry_strategy = self._classify_error(e, "task implementation")
                
                # Update result with error information
                result.error = error_detail
                result.error_category = error_category
                result.retry_strategy = retry_strategy
                result.retry_count = retry_count
                result.execution_time = time.time() - start_time
                
                # If we're out of retries or shouldn't retry, update task status and return error
                if retry_count >= max_retries or retry_strategy == RetryStrategy.NONE:
                    # Update task status to failed
                    old_status = task.status
                    task.status = TaskStatus.FAILED
                    
                    # Add error information to task metadata
                    if not hasattr(task, "metadata"):
                        task.metadata = {}
                    task.metadata["error"] = error_detail
                    task.metadata["error_category"] = error_category.value
                    
                    # Notify about failure
                    if self.notification_manager:
                        self.notification_manager.error(error_msg)
                    
                    # Save state even on failure
                    self._save_project_state()
                    
                    # Break the retry loop
                    break
                
                # Increment retry counter and continue to next attempt
                retry_count += 1
        
        # Create return dictionary
        return result.model_dump()
    
    def _verify_implementation_quality(self, code: str, file_path: str) -> None:
        """
        Verify the quality of the implementation.
        
        Args:
            code: The generated code to verify
            file_path: Path to the implementation file
            
        Raises:
            Exception: If the implementation has quality issues
        """
        # Basic syntax check for Python files
        if file_path.endswith(".py"):
            try:
                # Try to parse the code to check for syntax errors
                ast_module = __import__('ast')
                ast_module.parse(code)
            except SyntaxError as e:
                # Extract line and column information
                line = e.lineno
                col = e.offset
                context = e.text
                
                raise Exception(f"Syntax error in generated code at line {line}, column {col}: {str(e)}. Context: {context}")
        
        # TODO: Add more specialized quality checks based on file type
        # This could include running linters, validators, etc.
    
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