# core/orchestrator.py
from typing import Optional, List
import os
import logging
from datetime import datetime
from core.project_context import ProjectContext
from core.schemas import ProjectConfig, Task
from modules.planning.epic_generator import EpicGenerator
from modules.planning.task_validator import TaskValidator
from modules.generation.code_generator import CodeGenerator
from modules.validation.code_validator import CodeValidator
from modules.memory.codebase_context import CodebaseContext
from utils.llm_client import LLMClient
from utils.file_manager import FileManager

logger = logging.getLogger(__name__)

class Orchestrator:
    """Linear workflow orchestrator for the LLM fullstack generator"""
    
    def __init__(self, output_dir: str = "./output"):
        """Initialize the orchestrator"""
        self.output_dir = output_dir
        self.llm_client = LLMClient()
        self.epic_generator = EpicGenerator(self.llm_client)
        self.task_validator = TaskValidator(self.llm_client)
        self.code_generator = CodeGenerator(self.llm_client)
        self.code_validator = CodeValidator(self.llm_client)
        self.codebase_context = CodebaseContext()
        self.file_manager = FileManager(output_dir)
        self.project_context = None
    
    def initialize_project(self, name: str, description: str, config: ProjectConfig) -> ProjectContext:
        """Initialize a new project"""
        logger.info(f"Initializing project: {name}")
        
        # Create project directory
        project_dir = os.path.join(self.output_dir, name.lower().replace(" ", "_"))
        os.makedirs(project_dir, exist_ok=True)
        
        # Create project context
        self.project_context = ProjectContext(
            name=name,
            description=description,
            output_dir=project_dir,
            config=config
        )
        
        # Generate EPICs
        epics = self.epic_generator.generate_epics(name, description, config)
        
        # Validate and refine EPICs
        refined_epics = self.task_validator.validate_and_refine_epics(epics, name, description)
        
        # Update project context
        self.project_context.epics = refined_epics
        
        return self.project_context
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute"""
        if not self.project_context:
            logger.error("Project not initialized")
            return None
        
        pending_tasks = self.project_context.get_pending_tasks()
        if not pending_tasks:
            logger.info("No pending tasks available")
            return None
        
        # Return the first pending task
        # In a more sophisticated implementation, we could prioritize tasks
        return pending_tasks[0]
    
    def execute_task(self, task: Task) -> bool:
        """Execute a single task"""
        if not self.project_context:
            logger.error("Project not initialized")
            return False
        
        logger.info(f"Executing task: {task.id} - {task.title}")
        
        # Update task status
        self.project_context.update_task_status(task.id, "in_progress")
        
        try:
            # Generate code for the task
            for file_path in task.output_files:
                
                # Get context from codebase
                context = self.codebase_context.get_context_for_file(file_path)
                
                # Generate code
                code = self.code_generator.generate_component(
                    task=task,
                    project_context=self.project_context,
                    file_path=file_path,
                    context=context
                )
                
                # Validate code
                validated_code = self.code_validator.validate_code(
                    code=code,
                    task=task,
                    file_path=file_path,
                    project_context=self.project_context
                )
                
                # Save code to file
                self.file_manager.write_file(file_path, validated_code)
                
                # Update codebase context
                self.codebase_context.update_from_code(file_path, validated_code)
            
            # Mark task as completed
            self.project_context.update_task_status(task.id, "completed")
            return True
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {str(e)}")
            self.project_context.update_task_status(task.id, "failed")
            return False
    
    def run_full_generation(self) -> bool:
        """Run the complete generation process"""
        if not self.project_context:
            logger.error("Project not initialized")
            return False
        
        logger.info(f"Starting full generation for project: {self.project_context.name}")
        
        while True:
            next_task = self.get_next_task()
            if not next_task:
                break
                
            success = self.execute_task(next_task)
            if not success:
                logger.error(f"Failed to execute task: {next_task.id}")
                # Continue with next task instead of failing completely
        
        # Check if all tasks are completed
        all_completed = True
        for epic in self.project_context.epics:
            for task in epic.tasks:
                if task.status != "completed":
                    all_completed = False
                    break
        
        logger.info(f"Project generation {'completed successfully' if all_completed else 'finished with some failures'}")
        return all_completed