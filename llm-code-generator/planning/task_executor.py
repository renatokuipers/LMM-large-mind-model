from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging
import asyncio
from datetime import datetime
from pathlib import Path

from planning.models import (
    ProjectPlan,
    Epic,
    Task,
    TaskStatus,
    ComponentType
)
from generators.base import GeneratorContext
from generators.data_layer import DataLayerGenerator
from generators.business_logic import BusinessLogicGenerator
from generators.api_layer import APILayerGenerator
from core.code_memory import CodeMemory
from core.llm_manager import LLMManager
from core.project_manager import ProjectManager
from core.exceptions import PlanningError, GenerationError, ValidationError

logger = logging.getLogger(__name__)


class TaskExecutionManager:
    """Manages the execution of tasks in a project plan.
    
    This class is responsible for executing the tasks in a project plan,
    coordinating between different generators based on component types,
    and maintaining the state of the plan during execution.
    """
    
    def __init__(self, 
                 project_manager: ProjectManager,
                 llm_manager: LLMManager,
                 output_dir: Optional[Union[str, Path]] = None):
        """Initialize the task execution manager.
        
        Args:
            project_manager: Project manager for code generation
            llm_manager: LLM manager for generation
            output_dir: Optional output directory
        """
        self.project_manager = project_manager
        self.llm_manager = llm_manager
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Initialize generators
        self.code_memory = self.project_manager.code_memory or CodeMemory(project_name="unknown")
        self.data_layer_generator = DataLayerGenerator(llm_manager=llm_manager, code_memory=self.code_memory)
        self.business_logic_generator = BusinessLogicGenerator(llm_manager=llm_manager, code_memory=self.code_memory)
        self.api_layer_generator = APILayerGenerator(llm_manager=llm_manager, code_memory=self.code_memory)
        
        # Generator mapping
        self.generators = {
            ComponentType.DATA_MODEL: self.data_layer_generator,
            ComponentType.DATABASE_MODEL: self.data_layer_generator,
            ComponentType.REPOSITORY: self.data_layer_generator,
            ComponentType.SERVICE: self.business_logic_generator,
            ComponentType.USE_CASE: self.business_logic_generator,
            ComponentType.API_ENDPOINT: self.api_layer_generator,
            ComponentType.API_ROUTER: self.api_layer_generator,
            ComponentType.SCHEMA: self.data_layer_generator,
            ComponentType.CONFIG: self.data_layer_generator,
            ComponentType.UTIL: self.business_logic_generator,
            ComponentType.TEST: self.business_logic_generator
        }
        
        # State
        self.current_plan: Optional[ProjectPlan] = None
        self.component_files: Dict[str, Path] = {}
    
    async def execute_plan(self, plan: ProjectPlan) -> Dict[str, Path]:
        """Execute a project plan.
        
        Args:
            plan: Project plan to execute
            
        Returns:
            Dictionary mapping component names to file paths
            
        Raises:
            PlanningError: If plan execution fails
        """
        logger.info(f"Executing project plan: {plan.project_name}")
        
        # Initialize or update project manager state
        if not self.project_manager.state:
            output_dir = self.output_dir or Path("./generated") / self._sanitize_name(plan.project_name)
            await self.project_manager.initialize_project(
                project_name=plan.project_name,
                project_description=plan.project_description
            )
        
        # Store the current plan
        self.current_plan = plan
        
        # Reset component files
        self.component_files = {}
        
        try:
            # Execute tasks in dependency order
            completed_tasks = set()
            remaining_tasks = set(task.id for task in plan.all_tasks)
            
            # Keep track of failures to avoid infinite loops
            failed_tasks = set()
            
            # Execute until all tasks are completed or failed
            while remaining_tasks:
                # Find tasks that can be executed (all dependencies satisfied)
                executable_tasks = []
                
                for task_id in remaining_tasks:
                    task = plan.task_by_id.get(task_id)
                    if not task:
                        continue
                    
                    # Check if all required dependencies are completed
                    dependencies_satisfied = True
                    for dep_id, dep_type in task.dependencies.items():
                        if dep_type == DependencyType.REQUIRED and dep_id not in completed_tasks:
                            dependencies_satisfied = False
                            break
                    
                    if dependencies_satisfied:
                        executable_tasks.append(task)
                
                if not executable_tasks:
                    # No executable tasks but still have remaining tasks - deadlock
                    blocked_tasks = [task.title for task_id in remaining_tasks if task_id not in failed_tasks 
                                   for task in [plan.task_by_id.get(task_id)] if task]
                    logger.error(f"Deadlock detected. Blocked tasks: {blocked_tasks}")
                    raise PlanningError(f"Deadlock detected in plan execution. Blocked tasks: {blocked_tasks}")
                
                # Execute tasks in parallel (up to 3 at a time)
                results = await asyncio.gather(
                    *[self._execute_task(task) for task in executable_tasks[:3]],
                    return_exceptions=True
                )
                
                # Process results
                for i, result in enumerate(results):
                    task = executable_tasks[i]
                    
                    if isinstance(result, Exception):
                        # Task failed
                        logger.error(f"Task execution failed: {task.title} - {str(result)}")
                        task.status = TaskStatus.FAILED
                        failed_tasks.add(task.id)
                    else:
                        # Task succeeded
                        task.status = TaskStatus.COMPLETED
                        completed_tasks.add(task.id)
                        
                        if result:  # If we got a file path
                            self.component_files[f"{task.module_path}.{task.class_or_function_name or task.title}"] = result
                    
                    # Remove from remaining tasks
                    remaining_tasks.discard(task.id)
                
                # Update epic status
                for epic in plan.epics:
                    if all(task.status == TaskStatus.COMPLETED for task in epic.tasks):
                        epic.status = TaskStatus.COMPLETED
                    elif any(task.status == TaskStatus.IN_PROGRESS for task in epic.tasks):
                        epic.status = TaskStatus.IN_PROGRESS
                    elif any(task.status == TaskStatus.FAILED for task in epic.tasks):
                        if all(task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for task in epic.tasks):
                            epic.status = TaskStatus.FAILED
            
            return self.component_files
            
        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}")
            raise PlanningError(f"Failed to execute plan: {str(e)}")
    
    async def _execute_task(self, task: Task) -> Optional[Path]:
        """Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Path to the generated file if successful, None otherwise
            
        Raises:
            GenerationError: If task execution fails
        """
        logger.info(f"Executing task: {task.title} ({task.component_type})")
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Create generator context
            context = self._create_generator_context(task)
            
            # Get the appropriate generator
            generator = self.generators.get(task.component_type)
            if not generator:
                raise GenerationError(f"No generator available for component type: {task.component_type}")
            
            # Generate the component
            component = await generator.generate(context)
            
            # Write to file
            file_path = self._write_component_to_file(component)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error executing task {task.title}: {str(e)}")
            task.status = TaskStatus.FAILED
            raise GenerationError(f"Failed to execute task {task.title}: {str(e)}")
    
    def _create_generator_context(self, task: Task) -> GeneratorContext:
        """Create a generator context from a task.
        
        Args:
            task: Task to create context for
            
        Returns:
            Generator context
        """
        # Get project description
        project_description = self.current_plan.project_description if self.current_plan else ""
        
        # Get dependencies
        dependencies = []
        if self.current_plan:
            task_map = self.current_plan.task_by_id
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    dep_task = task_map[dep_id]
                    if dep_task.module_path and dep_task.class_or_function_name:
                        dependencies.append(f"{dep_task.module_path}.{dep_task.class_or_function_name}")
        
        # Create context
        return GeneratorContext(
            component_type=task.component_type.value,
            name=task.class_or_function_name or task.title.replace(" ", ""),
            module_path=task.module_path,
            description=task.description,
            requirements=task.requirements,
            dependencies=dependencies,
            project_description=project_description
        )
    
    def _write_component_to_file(self, component) -> Path:
        """Write a component to a file.
        
        Args:
            component: Component to write
            
        Returns:
            Path to the created file
            
        Raises:
            GenerationError: If file writing fails
        """
        try:
            return self.project_manager._write_component_to_file(component)
        except Exception as e:
            raise GenerationError(f"Failed to write component to file: {str(e)}")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for filesystem use.
        
        Args:
            name: Name to sanitize
            
        Returns:
            Sanitized name
        """
        return self.project_manager._sanitize_name(name)