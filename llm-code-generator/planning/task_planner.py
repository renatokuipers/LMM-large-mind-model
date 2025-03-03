from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field, validator

from planning.models import (
    ProjectPlan, 
    Epic, 
    Task, 
    Subtask, 
    TaskStatus, 
    ComponentType,
    DependencyType
)
from core.llm_manager import LLMManager
from core.exceptions import PlanningError, ParseError

logger = logging.getLogger(__name__)


class EpicGenerationRequest(BaseModel):
    """Request for generating project epics."""
    
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Description of the project")
    specific_requirements: List[str] = Field(default_factory=list, description="Specific project requirements")
    constraints: List[str] = Field(default_factory=list, description="Project constraints")


class TaskPlanner:
    """Plans project execution by generating epics and tasks using the LLM.
    
    This class is responsible for translating high-level project descriptions
    into structured epics and tasks with dependencies, which can then be
    executed by the project manager.
    """
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the task planner.
        
        Args:
            llm_manager: LLM manager for generating plans
        """
        self.llm_manager = llm_manager
    
    async def generate_project_plan(self, request: EpicGenerationRequest) -> ProjectPlan:
        """Generate a complete project plan with epics and tasks.
        
        Args:
            request: Epic generation request
            
        Returns:
            Complete project plan
            
        Raises:
            PlanningError: If plan generation fails
            ParseError: If LLM output cannot be parsed
        """
        logger.info(f"Generating project plan for: {request.project_name}")
        
        try:
            # Generate epics first
            epics = await self._generate_epics(request)
            
            # Create initial project plan
            plan = ProjectPlan(
                project_name=request.project_name,
                project_description=request.project_description,
                epics=epics
            )
            
            # Generate tasks for each epic
            for i, epic in enumerate(plan.epics):
                epic.tasks = await self._generate_tasks_for_epic(epic, plan, i)
            
            # Analyze and establish dependencies between tasks
            self._establish_task_dependencies(plan)
            
            # Validate the plan
            errors = plan.validate_dependencies()
            if errors:
                logger.warning(f"Plan validation found errors: {errors}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating project plan: {str(e)}")
            raise PlanningError(f"Failed to generate project plan: {str(e)}")
    
    async def _generate_epics(self, request: EpicGenerationRequest) -> List[Epic]:
        """Generate epics for the project.
        
        Args:
            request: Epic generation request
            
        Returns:
            List of generated epics
            
        Raises:
            ParseError: If LLM output cannot be parsed
        """
        # Define the expected response schema for structured output
        epic_schema = {
            "type": "object",
            "properties": {
                "epics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "order": {"type": "integer"},
                            "estimated_complexity": {"type": "integer"},
                            "priority": {"type": "integer"}
                        },
                        "required": ["title", "description", "order"]
                    }
                }
            },
            "required": ["epics"]
        }
        
        # Build the prompt for epic generation
        prompt = f"""
        # Epic Planning Task
        
        I need to plan the development of a software project by breaking it down into logical epics.
        
        ## Project Details
        
        Project Name: {request.project_name}
        
        ## Project Description
        {request.project_description}
        
        ## Specific Requirements
        {self._format_list(request.specific_requirements)}
        
        ## Constraints
        {self._format_list(request.constraints)}
        
        ## Expected Output Format
        Return a JSON object with an "epics" array containing objects with these fields:
        - title: Epic title (concise but descriptive)
        - description: Detailed description of what this epic covers
        - order: Execution order (lower numbers first)
        - estimated_complexity: Complexity estimate (1-5, 5 being most complex)
        - priority: Priority (1-5, 5 being highest)
        
        Follow these guidelines:
        1. Create a complete set of epics covering all aspects of the project
        2. Each epic should focus on a specific aspect or layer of the system
        3. Epics should be logically grouped and sequenced appropriately
        4. Use a systematic naming convention for epics
        5. Consider dependencies between epics when assigning order
        6. Include detailed descriptions that explain the scope of each epic
        7. Use appropriate complexity and priority ratings
        8. Include 5-10 epics depending on project size (neither too granular nor too broad)
        
        Example of a good epic:
        {
          "title": "Data Model Implementation",
          "description": "Design and implement all data models using Pydantic, including domain entities, validation rules, and relationships between models. This forms the foundation of the application's type system.",
          "order": 1,
          "estimated_complexity": 3,
          "priority": 5
        }
        
        Create a comprehensive set of epics for this specific project.
        """
        
        try:
            # Generate epics with structured schema
            response = await self.llm_manager.generate(
                prompt=prompt,
                task_type="planning",
                schema=epic_schema
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                raise ParseError(f"Failed to parse LLM response as JSON: {response.content}")
            
            # Validate and extract epics
            if not isinstance(response_data, dict) or "epics" not in response_data:
                raise ParseError(f"Invalid response structure: {response_data}")
            
            # Create Epic objects
            epics = []
            for i, epic_data in enumerate(response_data["epics"]):
                epic = Epic(
                    title=epic_data["title"],
                    description=epic_data["description"],
                    order=epic_data.get("order", i),
                    estimated_complexity=epic_data.get("estimated_complexity", 3),
                    priority=epic_data.get("priority", 3)
                )
                epics.append(epic)
            
            # Sort epics by order
            epics.sort(key=lambda x: x.order)
            
            return epics
            
        except Exception as e:
            logger.error(f"Error generating epics: {str(e)}")
            raise
    
    async def _generate_tasks_for_epic(self, epic: Epic, plan: ProjectPlan, epic_index: int) -> List[Task]:
        """Generate tasks for an epic.
        
        Args:
            epic: Epic to generate tasks for
            plan: Current project plan
            epic_index: Index of the epic in the plan
            
        Returns:
            List of generated tasks
            
        Raises:
            ParseError: If LLM output cannot be parsed
        """
        # Define the expected response schema for structured output
        task_schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "component_type": {"type": "string"},
                            "module_path": {"type": "string"},
                            "class_or_function_name": {"type": "string"},
                            "requirements": {"type": "array", "items": {"type": "string"}},
                            "estimated_complexity": {"type": "integer"},
                            "priority": {"type": "integer"},
                            "subtasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"}
                                    }
                                }
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "epic_title": {"type": "string"},
                                        "task_title": {"type": "string"},
                                        "dependency_type": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["title", "description", "component_type", "module_path"]
                    }
                }
            },
            "required": ["tasks"]
        }
        
        # Prepare context of previously defined epics
        previous_epics_context = ""
        if epic_index > 0:
            previous_epics_context = "## Previously Defined Epics\n\n"
            for i, prev_epic in enumerate(plan.epics[:epic_index]):
                previous_epics_context += f"### Epic {i+1}: {prev_epic.title}\n"
                if prev_epic.tasks:
                    previous_epics_context += "Tasks:\n"
                    for task in prev_epic.tasks:
                        previous_epics_context += f"- {task.title} ({task.component_type}: {task.module_path})\n"
                previous_epics_context += "\n"
        
        # Build the prompt for task generation
        prompt = f"""
        # Task Planning for Epic
        
        I need to break down an epic into specific, actionable tasks.
        
        ## Project Details
        
        Project Name: {plan.project_name}
        
        ## Project Description
        {plan.project_description}
        
        ## Current Epic
        
        Epic: {epic.title}
        Description: {epic.description}
        
        {previous_epics_context}
        
        ## Expected Output Format
        Return a JSON object with a "tasks" array containing objects with these fields:
        - title: Task title (concise but descriptive)
        - description: Detailed description of what this task involves
        - component_type: Type of component to generate (one of: data_model, database_model, repository, service, use_case, api_endpoint, api_router, schema, config, util, test)
        - module_path: Module path where the component will be defined (e.g., "app.models.user")
        - class_or_function_name: Name of the class or function to generate
        - requirements: Array of specific requirements for this component
        - estimated_complexity: Complexity estimate (1-5, 5 being most complex)
        - priority: Priority (1-5, 5 being highest)
        - subtasks: Array of subtasks, each with "title" and "description"
        - dependencies: Array of dependencies, each with "epic_title", "task_title", and "dependency_type" (required, preferred, or referenced)
        
        Follow these guidelines:
        1. Create detailed, specific tasks that directly implement the epic
        2. Each task should result in a single concrete component (class or function)
        3. Tasks should be logically sequenced considering dependencies
        4. Tasks should be sized appropriately (not too large or small)
        5. Each component should have a clear module path and name
        6. Consider both internal dependencies (within this epic) and external dependencies (from other epics)
        7. Include 3-10 tasks per epic, depending on complexity
        8. Be specific about implementation details and requirements
        
        Example of a good task:
        {
          "title": "Create User Data Model",
          "description": "Implement the core User Pydantic model with all fields, validators, and relationships",
          "component_type": "data_model",
          "module_path": "app.models.user",
          "class_or_function_name": "User",
          "requirements": [
            "Include email, username, hashed_password, and is_active fields",
            "Add email validation with regex",
            "Implement password hashing logic"
          ],
          "estimated_complexity": 3,
          "priority": 5,
          "subtasks": [
            {
              "title": "Implement base fields",
              "description": "Add core fields with type annotations and validations"
            },
            {
              "title": "Add relationship fields",
              "description": "Implement fields for relationships to other models"
            }
          ],
          "dependencies": []
        }
        
        Create a comprehensive set of tasks for this specific epic.
        """
        
        try:
            # Generate tasks with structured schema
            response = await self.llm_manager.generate(
                prompt=prompt,
                task_type="planning",
                schema=task_schema
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                raise ParseError(f"Failed to parse LLM response as JSON: {response.content}")
            
            # Validate and extract tasks
            if not isinstance(response_data, dict) or "tasks" not in response_data:
                raise ParseError(f"Invalid response structure: {response_data}")
            
            # Create Task objects
            tasks = []
            for task_data in response_data["tasks"]:
                # Create subtasks
                subtasks = []
                for subtask_data in task_data.get("subtasks", []):
                    subtask = Subtask(
                        title=subtask_data["title"],
                        description=subtask_data["description"]
                    )
                    subtasks.append(subtask)
                
                # Validate component type
                component_type_str = task_data["component_type"]
                try:
                    component_type = ComponentType(component_type_str)
                except ValueError:
                    logger.warning(f"Invalid component type: {component_type_str}, defaulting to util")
                    component_type = ComponentType.UTIL
                
                # Create task
                task = Task(
                    title=task_data["title"],
                    description=task_data["description"],
                    component_type=component_type,
                    module_path=task_data["module_path"],
                    class_or_function_name=task_data.get("class_or_function_name"),
                    requirements=task_data.get("requirements", []),
                    estimated_complexity=task_data.get("estimated_complexity", 3),
                    priority=task_data.get("priority", 3),
                    subtasks=subtasks
                )
                tasks.append(task)
                
                # Track dependencies (these will be resolved later)
                dependency_data = task_data.get("dependencies", [])
                if dependency_data:
                    # Store raw dependency data for later resolution
                    task._raw_dependencies = dependency_data
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error generating tasks for epic {epic.title}: {str(e)}")
            raise
    
    def _establish_task_dependencies(self, plan: ProjectPlan) -> None:
        """Establish dependencies between tasks based on raw dependency data.
        
        Args:
            plan: Project plan to process
        """
        # Create lookup maps
        epic_map = {epic.title: epic for epic in plan.epics}
        
        # Map to find tasks by title (assumes unique titles within epics)
        task_by_epic_and_title = {}
        for epic in plan.epics:
            task_by_epic_and_title[epic.title] = {task.title: task for task in epic.tasks}
        
        # Process dependencies
        for epic in plan.epics:
            for task in epic.tasks:
                # Skip if no raw dependencies
                if not hasattr(task, '_raw_dependencies'):
                    continue
                
                for dep_data in task._raw_dependencies:
                    epic_title = dep_data.get("epic_title")
                    task_title = dep_data.get("task_title")
                    dependency_type_str = dep_data.get("dependency_type", "required")
                    
                    # Validate dependency type
                    try:
                        dependency_type = DependencyType(dependency_type_str)
                    except ValueError:
                        logger.warning(f"Invalid dependency type: {dependency_type_str}, defaulting to required")
                        dependency_type = DependencyType.REQUIRED
                    
                    # Find the target epic
                    if epic_title in epic_map:
                        target_epic = epic_map[epic_title]
                        
                        # Find the target task
                        if task_title in task_by_epic_and_title.get(epic_title, {}):
                            target_task = task_by_epic_and_title[epic_title][task_title]
                            
                            # Add dependency
                            task.dependencies[target_task.id] = dependency_type
                            
                            # Add to dependents list
                            if task.id not in target_task.dependents:
                                target_task.dependents.append(task.id)
                        else:
                            logger.warning(f"Dependency task not found: {task_title} in epic {epic_title}")
                    else:
                        logger.warning(f"Dependency epic not found: {epic_title}")
                
                # Remove raw dependencies
                delattr(task, '_raw_dependencies')
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a bulleted list.
        
        Args:
            items: List of items
            
        Returns:
            Formatted string
        """
        if not items:
            return "None specified"
        
        return "\n".join(f"- {item}" for item in items)