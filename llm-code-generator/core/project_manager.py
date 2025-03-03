from typing import Dict, List, Optional, Set, Tuple, Union, Any
import asyncio
import logging
import json
import os
from pathlib import Path
import shutil
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator

from schemas.code_entities import CodeComponent, FunctionSignature, ClassSignature
from schemas.llm_io import (
    ArchitectureGenerationRequest,
    ArchitectureGenerationResponse,
    CodeGenerationRequest,
    CodeGenerationResponse,
    ComponentDefinition
)
from core.code_memory import CodeMemory
from core.llm_manager import LLMManager
from core.validators import CodeValidator, ValidationResult
from core.exceptions import (
    ProjectError,
    ValidationError,
    DependencyError,
    LLMError,
    ParseError
)

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a generation task."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    INVALID = "invalid"


class GenerationTask(BaseModel):
    """A code generation task."""
    
    component_type: str = Field(..., description="Type of component to generate")
    name: str = Field(..., description="Name of the component")
    module_path: str = Field(..., description="Module path where this component will be defined")
    description: str = Field(..., description="Description of the component")
    requirements: str = Field(..., description="Requirements for the component")
    additional_context: Optional[str] = Field(None, description="Additional context")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies for this task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    attempts: int = Field(default=0, description="Number of generation attempts")
    max_attempts: int = Field(default=3, description="Maximum number of attempts")
    result: Optional[str] = Field(None, description="Generated code")
    
    def to_request(self) -> CodeGenerationRequest:
        """Convert to a code generation request."""
        return CodeGenerationRequest(
            component_type=self.component_type,
            name=self.name,
            module_path=self.module_path,
            description=self.description,
            requirements=self.requirements,
            additional_context=self.additional_context
        )


class ProjectState(BaseModel):
    """State of a code generation project."""
    
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Description of the project")
    output_dir: str = Field(..., description="Output directory for generated code")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Architecture components
    components: List[ComponentDefinition] = Field(default_factory=list, description="Components in the architecture")
    
    # Tasks
    pending_tasks: List[GenerationTask] = Field(default_factory=list, description="Pending tasks")
    completed_tasks: List[GenerationTask] = Field(default_factory=list, description="Completed tasks")
    failed_tasks: List[GenerationTask] = Field(default_factory=list, description="Failed tasks")
    
    # Status tracking
    is_architecture_complete: bool = Field(default=False, description="Whether architecture is complete")
    is_generation_complete: bool = Field(default=False, description="Whether generation is complete")
    
    @property
    def total_tasks(self) -> int:
        """Get the total number of tasks."""
        return len(self.pending_tasks) + len(self.completed_tasks) + len(self.failed_tasks)
    
    @property
    def progress_percentage(self) -> float:
        """Get the progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        return len(self.completed_tasks) / self.total_tasks * 100
    
    class Config:
        arbitrary_types_allowed = True


class ProjectManager:
    """Manages the code generation project.
    
    This class orchestrates the entire code generation process, from
    architecture design to code generation, validation, and file creation.
    """
    
    def __init__(self, 
                 llm_manager: LLMManager,
                 output_base_dir: Union[str, Path] = "./generated"):
        """Initialize the project manager.
        
        Args:
            llm_manager: LLM manager instance
            output_base_dir: Base directory for generated projects
        """
        self.llm_manager = llm_manager
        self.output_base_dir = Path(output_base_dir)
        self.state: Optional[ProjectState] = None
        self.code_memory: Optional[CodeMemory] = None
        self.validator: Optional[CodeValidator] = None
        
        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_project(self, 
                                project_name: str, 
                                project_description: str) -> ProjectState:
        """Initialize a new code generation project.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            
        Returns:
            Project state
            
        Raises:
            ProjectError: If the project cannot be initialized
        """
        # Sanitize project name for filesystem
        safe_name = self._sanitize_name(project_name)
        output_dir = self.output_base_dir / safe_name
        
        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ProjectError(f"Failed to create output directory: {str(e)}")
        
        # Initialize project state
        self.state = ProjectState(
            project_name=project_name,
            project_description=project_description,
            output_dir=str(output_dir)
        )
        
        # Initialize code memory
        self.code_memory = CodeMemory(project_name=project_name, project_root=output_dir)
        
        # Initialize validator
        self.validator = CodeValidator(code_memory=self.code_memory)
        
        # Update LLM manager with code memory
        self.llm_manager.code_memory = self.code_memory
        
        return self.state
    
    async def generate_architecture(self) -> ArchitectureGenerationResponse:
        """Generate the project architecture using the LLM.
        
        Returns:
            Architecture generation response
            
        Raises:
            ProjectError: If the project is not initialized
            LLMError: If there is an error generating the architecture
        """
        if not self.state:
            raise ProjectError("Project not initialized")
        
        # Create architecture request
        request = ArchitectureGenerationRequest(
            project_name=self.state.project_name,
            project_description=self.state.project_description,
            requirements=[
                "Create a well-structured application following best practices",
                "Use Pydantic for data validation",
                "Implement comprehensive error handling",
                "Follow type safety principles",
                "Design clean, maintainable interfaces between components"
            ]
        )
        
        # Define the expected response schema for structured output
        architecture_schema = {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "description": {"type": "string"},
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["class", "function", "module"]},
                            "module_path": {"type": "string"},
                            "description": {"type": "string"},
                            "responsibilities": {"type": "array", "items": {"type": "string"}},
                            "dependencies": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "type", "module_path", "description", "responsibilities"]
                    }
                },
                "data_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "attributes": {"type": "array", "items": {"type": "object"}}
                        }
                    }
                }
            },
            "required": ["project_name", "description", "components"]
        }
        
        # Build the prompt
        prompt = f"""
        # Architecture Design Task
        
        Design the architecture for the following project:
        
        Project Name: {self.state.project_name}
        
        ## Project Description
        {self.state.project_description}
        
        ## Requirements
        - Create a well-structured application following best practices
        - Use Pydantic for data validation
        - Implement comprehensive error handling
        - Follow type safety principles
        - Design clean, maintainable interfaces between components
        
        ## Expected Output Format
        Return a JSON object with these fields:
        - project_name: Project name
        - description: Project description
        - components: Array of components in the architecture with these fields:
          - name: Component name
          - type: Component type (class, function, module)
          - module_path: Module path where the component will be defined
          - description: Description of the component
          - responsibilities: Array of responsibilities for this component
          - dependencies: Array of component names this depends on
        - data_models: Array of data models with these fields:
          - name: Model name
          - attributes: Array of attributes for this model
        
        Design a complete architecture that covers all aspects of the project description.
        Each component should have a clear responsibility and well-defined interfaces.
        Ensure proper separation of concerns and dependency management.
        
        Do not include explanatory text outside the JSON.
        """
        
        try:
            # Generate architecture with structured schema
            response = await self.llm_manager.generate(
                prompt=prompt,
                task_type="architecture",
                schema=architecture_schema
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                raise ParseError(f"Failed to parse LLM response as JSON: {response.content}")
            
            # Validate and extract fields
            if not isinstance(response_data, dict) or "components" not in response_data:
                raise ParseError(f"Invalid response structure: {response_data}")
            
            # Create response object
            architecture = ArchitectureGenerationResponse(
                project_name=response_data.get("project_name", self.state.project_name),
                description=response_data.get("description", self.state.project_description),
                components=[ComponentDefinition(**comp) for comp in response_data.get("components", [])],
                data_models=response_data.get("data_models", []),
                api_endpoints=response_data.get("api_endpoints")
            )
            
            # Update project state with components
            self.state.components = architecture.components
            self.state.is_architecture_complete = True
            self.state.updated_at = datetime.now()
            
            # Create tasks from components
            self._create_tasks_from_architecture(architecture)
            
            return architecture
            
        except Exception as e:
            logger.error(f"Error generating architecture: {str(e)}")
            raise LLMError(f"Failed to generate architecture: {str(e)}")
    
    def _create_tasks_from_architecture(self, architecture: ArchitectureGenerationResponse) -> None:
        """Create generation tasks from the architecture.
        
        Args:
            architecture: Architecture generation response
        """
        if not self.state:
            raise ProjectError("Project not initialized")
        
        # Create a task for each component
        for component in architecture.components:
            # Extract dependencies
            dependencies = component.dependencies
            
            # Create task
            task = GenerationTask(
                component_type=component.type,
                name=component.name,
                module_path=component.module_path,
                description=component.description,
                requirements="\n".join(component.responsibilities),
                dependencies=dependencies
            )
            
            # Add task to pending tasks
            self.state.pending_tasks.append(task)
    
    async def _generate_component(self, task: GenerationTask) -> Optional[CodeComponent]:
        """Generate a code component.
        
        Args:
            task: Generation task
            
        Returns:
            Generated component if successful, None otherwise
            
        Raises:
            LLMError: If there is an error generating the component
        """
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1
        task.updated_at = datetime.now()
        
        try:
            # Generate code
            response = await self.llm_manager.generate_code(task.to_request())
            
            # Create component
            component = CodeComponent(
                component_type=task.component_type,
                name=task.name,
                module_path=task.module_path,
                implementation=response.code,
                dependencies=response.dependencies,
                signature=(
                    ClassSignature(name=task.name, module_path=task.module_path)
                    if task.component_type == "class"
                    else FunctionSignature(name=task.name, module_path=task.module_path)
                )
            )
            
            # Store result
            task.result = response.code
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating component {task.name}: {str(e)}")
            task.errors.append(str(e))
            task.status = TaskStatus.FAILED
            return None
    
    async def _validate_component(self, 
                                component: CodeComponent, 
                                task: GenerationTask) -> ValidationResult:
        """Validate a generated component.
        
        Args:
            component: Generated component
            task: Generation task
            
        Returns:
            Validation result
            
        Raises:
            ValidationError: If validation fails
        """
        if not self.validator:
            raise ProjectError("Validator not initialized")
        
        # Perform validation
        result = self.validator.validate_component(component)
        
        # Update task status based on validation result
        if result.is_valid:
            task.status = TaskStatus.VALIDATED
        else:
            task.status = TaskStatus.INVALID
            task.errors.extend(result.errors)
        
        task.updated_at = datetime.now()
        
        return result
    
    def _write_component_to_file(self, component: CodeComponent) -> Path:
        """Write a component to a file.
        
        Args:
            component: Component to write
            
        Returns:
            Path to the created file
            
        Raises:
            ProjectError: If the file cannot be created
        """
        if not self.state:
            raise ProjectError("Project not initialized")
        
        # Get output directory
        output_dir = Path(self.state.output_dir)
        
        # Convert module path to file path
        module_parts = component.module_path.split(".")
        file_path = output_dir.joinpath(*module_parts).with_suffix(".py")
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the code to the file
        try:
            with open(file_path, "w") as f:
                f.write(component.implementation or "")
        except Exception as e:
            raise ProjectError(f"Failed to write component to file: {str(e)}")
        
        return file_path
    
    async def generate_next_component(self) -> Optional[Tuple[CodeComponent, Path]]:
        """Generate the next component based on dependencies.
        
        Returns:
            Tuple of generated component and file path if successful, None otherwise
        """
        if not self.state or not self.code_memory:
            raise ProjectError("Project not initialized")
        
        # Find a task that is ready (all dependencies satisfied)
        ready_task = None
        for task in self.state.pending_tasks:
            # Skip failed tasks that have reached max attempts
            if task.status == TaskStatus.FAILED and task.attempts >= task.max_attempts:
                continue
                
            # Check dependencies
            dependencies_satisfied = True
            for dep in task.dependencies:
                if not self.code_memory.is_implemented(dep):
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                ready_task = task
                break
        
        if not ready_task:
            # No ready tasks found
            return None
        
        # Generate component
        component = await self._generate_component(ready_task)
        if not component:
            # Move to failed tasks if max attempts reached
            if ready_task.attempts >= ready_task.max_attempts:
                self.state.pending_tasks.remove(ready_task)
                self.state.failed_tasks.append(ready_task)
            return None
        
        # Validate component
        validation_result = await self._validate_component(component, ready_task)
        
        if not validation_result.is_valid:
            # Regenerate if validation failed and attempts remain
            if ready_task.attempts < ready_task.max_attempts:
                # Add validation errors to context for next attempt
                validation_context = "\n".join([
                    "Previous generation attempt had these issues:",
                    *[f"- {error}" for error in validation_result.errors]
                ])
                ready_task.additional_context = (ready_task.additional_context or "") + "\n" + validation_context
                
                # Reset status
                ready_task.status = TaskStatus.PENDING
                
                return None
            else:
                # Move to failed tasks
                self.state.pending_tasks.remove(ready_task)
                self.state.failed_tasks.append(ready_task)
                return None
        
        # Write component to file
        file_path = self._write_component_to_file(component)
        
        # Update code memory
        self.code_memory.add_component(component)
        
        # Update task status
        ready_task.status = TaskStatus.COMPLETED
        
        # Move task to completed tasks
        self.state.pending_tasks.remove(ready_task)
        self.state.completed_tasks.append(ready_task)
        
        # Check if all tasks are complete
        if not self.state.pending_tasks:
            self.state.is_generation_complete = True
        
        self.state.updated_at = datetime.now()
        
        return component, file_path
    
    async def generate_all_components(self) -> Dict[str, Path]:
        """Generate all components based on the architecture.
        
        Returns:
            Dictionary mapping component names to file paths
            
        Raises:
            ProjectError: If the project is not initialized
        """
        if not self.state:
            raise ProjectError("Project not initialized")
        
        if not self.state.is_architecture_complete:
            raise ProjectError("Architecture not generated yet")
        
        # Generate components until all tasks are complete
        generated_components = {}
        
        while not self.state.is_generation_complete:
            result = await self.generate_next_component()
            
            if result:
                component, file_path = result
                component_name = f"{component.module_path}.{component.name}"
                generated_components[component_name] = file_path
        
        return generated_components
    
    def get_state(self) -> Optional[ProjectState]:
        """Get the current project state.
        
        Returns:
            Project state if initialized, None otherwise
        """
        return self.state
    
    def get_task_by_name(self, name: str, module_path: str) -> Optional[GenerationTask]:
        """Get a task by component name and module path.
        
        Args:
            name: Component name
            module_path: Module path
            
        Returns:
            Task if found, None otherwise
        """
        if not self.state:
            return None
        
        # Check pending tasks
        for task in self.state.pending_tasks:
            if task.name == name and task.module_path == module_path:
                return task
        
        # Check completed tasks
        for task in self.state.completed_tasks:
            if task.name == name and task.module_path == module_path:
                return task
        
        # Check failed tasks
        for task in self.state.failed_tasks:
            if task.name == name and task.module_path == module_path:
                return task
        
        return None
    
    def save_state(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """Save the project state to a file.
        
        Args:
            file_path: Optional file path, defaults to project directory
            
        Returns:
            Path to the state file
            
        Raises:
            ProjectError: If the project is not initialized
        """
        if not self.state:
            raise ProjectError("Project not initialized")
        
        if not file_path:
            # Use default path in project directory
            file_path = Path(self.state.output_dir) / "project_state.json"
        else:
            file_path = Path(file_path)
        
        # Save state to file
        try:
            with open(file_path, "w") as f:
                json.dump(self.state.dict(), f, indent=2, default=str)
        except Exception as e:
            raise ProjectError(f"Failed to save state: {str(e)}")
        
        return file_path
    
    def load_state(self, file_path: Union[str, Path]) -> ProjectState:
        """Load project state from a file.
        
        Args:
            file_path: Path to the state file
            
        Returns:
            Loaded project state
            
        Raises:
            ProjectError: If the state file cannot be loaded
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, "r") as f:
                state_dict = json.load(f)
        except Exception as e:
            raise ProjectError(f"Failed to load state: {str(e)}")
        
        # Create project state from dictionary
        try:
            self.state = ProjectState.parse_obj(state_dict)
        except Exception as e:
            raise ProjectError(f"Failed to parse state: {str(e)}")
        
        # Initialize code memory
        self.code_memory = CodeMemory(
            project_name=self.state.project_name,
            project_root=Path(self.state.output_dir)
        )
        
        # Initialize validator
        self.validator = CodeValidator(code_memory=self.code_memory)
        
        # Update LLM manager with code memory
        self.llm_manager.code_memory = self.code_memory
        
        return self.state
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for filesystem use.
        
        Args:
            name: Name to sanitize
            
        Returns:
            Sanitized name
        """
        # Replace spaces with underscores and remove special characters
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.replace(" ", "_"))