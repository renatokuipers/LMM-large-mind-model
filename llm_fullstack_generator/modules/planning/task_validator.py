# modules/planning/task_validator.py
from typing import List
from core.schemas import Epic, Task
from utils.llm_client import LLMClient, Message
import logging

logger = logging.getLogger(__name__)

class TaskValidator:
    """Validates and refines the generated tasks"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def validate_and_refine_epics(self, epics: List[Epic], project_name: str, project_description: str) -> List[Epic]:
        """Validate and refine the generated EPICs and tasks"""
        logger.info(f"Validating EPICs for project: {project_name}")
        
        epic_schema = {
            "type": "object",
            "properties": {
                "epics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "dependencies": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "output_files": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["id", "title", "description"]
                                }
                            }
                        },
                        "required": ["id", "title", "description", "tasks"]
                    }
                }
            },
            "required": ["epics"]
        }
        
        # Convert epics to a JSON representation for the LLM
        epics_data = {
            "epics": [
                {
                    "id": epic.id,
                    "title": epic.title,
                    "description": epic.description,
                    "tasks": [
                        {
                            "id": task.id,
                            "title": task.title,
                            "description": task.description,
                            "dependencies": task.dependencies,
                            "output_files": task.output_files
                        }
                        for task in epic.tasks
                    ]
                }
                for epic in epics
            ]
        }
        
        messages = [
            Message(role="system", content="""You are an expert software architect and project planner.
                   Your task is to validate and refine a project plan consisting of EPICs and tasks.
                   Ensure that:
                   1. All EPICs are necessary and well-defined
                   2. All tasks are granular enough (one class/function per task)
                   3. Dependencies between tasks are correct
                   4. No essential components are missing
                   5. Output files for each task are correctly specified
                   
                   Return an improved version of the plan with any necessary changes."""),
            Message(role="user", content=f"""I have a project plan for: {project_name}
                   Description: {project_description}
                   
                   Here is the current plan:
                   {epics_data}
                   
                   Please validate and refine this plan according to the criteria mentioned.
                   Make sure each task is focused on implementing a single component.
                   Ensure the task dependencies form a valid directed acyclic graph.
                   Add any missing tasks or EPICs that would be required for a complete implementation.""")
        ]
        
        try:
            result = self.llm_client.structured_completion(messages, epic_schema)
            
            # Convert the result back to Epic objects
            refined_epics = []
            for epic_data in result["epics"]:
                tasks = []
                for task_data in epic_data.get("tasks", []):
                    task = Task(
                        id=task_data["id"],
                        title=task_data["title"],
                        description=task_data["description"],
                        dependencies=task_data.get("dependencies", []),
                        output_files=task_data.get("output_files", [])
                    )
                    tasks.append(task)
                    
                epic = Epic(
                    id=epic_data["id"],
                    title=epic_data["title"],
                    description=epic_data["description"],
                    tasks=tasks
                )
                refined_epics.append(epic)
            
            logger.info(f"Refined plan: {len(refined_epics)} EPICs with {sum(len(epic.tasks) for epic in refined_epics)} tasks")
            return refined_epics
            
        except Exception as e:
            logger.error(f"Error validating EPICs: {str(e)}")
            # Return original epics if validation fails
            return epics