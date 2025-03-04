from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from models.project_model import Project, Epic, Task, CodeItem, TaskStatus
from llm_module import LLMClient, Message


class PlannerService:
    """Service for generating project plans using LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def create_project_plan(self, project: Project) -> Project:
        """Create a complete project plan with epics and tasks"""
        # Generate the initial epics framework
        project = self._generate_epic_framework(project)
        
        # Generate detailed tasks for each epic
        for i, epic in enumerate(project.epics):
            project.epics[i] = self._generate_tasks_for_epic(project, epic)
            
        return project
    
    def _generate_epic_framework(self, project: Project) -> Project:
        """Generate the initial set of epics for the project"""
        epic_schema = {
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "technical_areas": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["title", "description", "technical_areas"]
            }
        }
        
        # First, get a high-level overview of what epics we need
        epic_overview_prompt = f"""
        You are an expert software architect planning a new project.
        
        Project Name: {project.name}
        Project Description: {project.description}
        
        Think step by step about what major components (epics) would be needed for this project.
        Consider the frontend, backend, database, and any other technical areas.
        
        Return ONE epic at a time. We'll build the project plan incrementally.
        
        For the FIRST epic, focus on the most fundamental component needed.
        """
        
        messages = [
            Message(role="system", content="You are an expert software architect specialized in full-stack development."),
            Message(role="user", content=epic_overview_prompt)
        ]
        
        # We'll generate up to 8 epics (can be adjusted)
        max_epics = 8
        generated_epics = []
        
        for i in range(max_epics):
            try:
                # Create context from previous epics
                epic_context = ""
                if generated_epics:
                    epic_context = "Previously defined epics:\n"
                    for j, prev_epic in enumerate(generated_epics):
                        epic_context += f"{j+1}. {prev_epic.title}: {prev_epic.description}\n"
                
                epic_prompt = f"""
                {epic_context}
                
                Based on the project description and previously defined epics, define the next logical epic for this project.
                If you believe the project planning is complete, respond with {{"title": "PLANNING_COMPLETE", "description": "Project planning is complete", "technical_areas": []}}
                
                Return just ONE epic in JSON format.
                """
                
                if i == 0:
                    # For the first epic, use the initial prompt
                    response = self.llm_client.structured_completion(
                        messages, 
                        epic_schema,
                        temperature=0.3,
                        max_tokens=500
                    )
                else:
                    # For subsequent epics, add the context
                    context_messages = messages.copy()
                    context_messages.append(Message(role="user", content=epic_prompt))
                    response = self.llm_client.structured_completion(
                        context_messages,
                        epic_schema,
                        temperature=0.3,
                        max_tokens=500
                    )
                
                # Check if planning is complete
                if response["title"] == "PLANNING_COMPLETE":
                    break
                    
                # Create a new epic
                epic = Epic(
                    title=response["title"],
                    description=response["description"],
                    tasks=[]
                )
                
                # Add technical areas to project tech stack
                for area in response["technical_areas"]:
                    # Simplified categorization
                    if any(keyword in area.lower() for keyword in ["frontend", "ui", "interface", "react", "angular", "vue"]):
                        if area not in project.tech_stack["frontend"]:
                            project.tech_stack["frontend"].append(area)
                    elif any(keyword in area.lower() for keyword in ["backend", "server", "api", "flask", "django", "express"]):
                        if area not in project.tech_stack["backend"]:
                            project.tech_stack["backend"].append(area)
                    elif any(keyword in area.lower() for keyword in ["database", "db", "sql", "mongo", "postgres"]):
                        if area not in project.tech_stack["database"]:
                            project.tech_stack["database"].append(area)
                
                generated_epics.append(epic)
                
                # Update messages to provide context for next epic
                feedback_msg = f"Epic '{epic.title}' has been added to the plan."
                messages.append(Message(role="assistant", content=json.dumps(response)))
                messages.append(Message(role="user", content=feedback_msg))
                
            except Exception as e:
                print(f"Error generating epic: {str(e)}")
                break
        
        # Update the project with the generated epics
        project.epics = generated_epics
        project.updated_at = datetime.now()
        
        return project
    
    def _generate_tasks_for_epic(self, project: Project, epic: Epic) -> Epic:
        """Generate detailed tasks for a specific epic"""
        task_schema = {
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "code_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "name": {"type": "string"},
                                "parameters": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "result": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["type", "name", "description"]
                        }
                    }
                },
                "required": ["title", "description", "code_items"]
            }
        }
        
        # Generate context from project and epic
        context = f"""
        Project: {project.name}
        Project Description: {project.description}
        
        Epic: {epic.title}
        Epic Description: {epic.description}
        
        Tech Stack:
        - Frontend: {', '.join(project.tech_stack['frontend']) if project.tech_stack['frontend'] else 'Not specified'}
        - Backend: {', '.join(project.tech_stack['backend']) if project.tech_stack['backend'] else 'Not specified'}
        - Database: {', '.join(project.tech_stack['database']) if project.tech_stack['database'] else 'Not specified'}
        """
        
        task_prompt = f"""
        {context}
        
        Break down this epic into specific, actionable tasks.
        Return ONE task at a time. Each task should have:
        1. A clear title
        2. A detailed description of what needs to be implemented
        3. A list of code items that need to be created (functions, classes, methods)
        
        For code items, specify:
        - Type (Function, Class, Method)
        - Name
        - Parameters (if applicable)
        - Return value (if applicable)
        - Description of what the code item does
        
        Return just ONE task in JSON format.
        """
        
        messages = [
            Message(role="system", content="You are an expert software developer specialized in breaking down complex features into manageable tasks."),
            Message(role="user", content=task_prompt)
        ]
        
        # We'll generate up to 10 tasks per epic (can be adjusted)
        max_tasks = 10
        generated_tasks = []
        
        for i in range(max_tasks):
            try:
                # Create context from previous tasks
                task_context = ""
                if generated_tasks:
                    task_context = "Previously defined tasks for this epic:\n"
                    for j, prev_task in enumerate(generated_tasks):
                        task_context += f"{j+1}. {prev_task.title}\n"
                
                next_task_prompt = f"""
                {context}
                
                {task_context}
                
                Based on the epic description and previously defined tasks, define the next logical task for this epic.
                If you believe the epic planning is complete, respond with {{"title": "TASKS_COMPLETE", "description": "Epic tasks are complete", "code_items": []}}
                
                Return just ONE task in JSON format.
                """
                
                if i == 0:
                    # For the first task, use the initial prompt
                    response = self.llm_client.structured_completion(
                        messages, 
                        task_schema,
                        temperature=0.3,
                        max_tokens=800
                    )
                else:
                    # For subsequent tasks, add the context
                    context_messages = messages.copy()
                    context_messages.append(Message(role="user", content=next_task_prompt))
                    response = self.llm_client.structured_completion(
                        context_messages,
                        task_schema,
                        temperature=0.3,
                        max_tokens=800
                    )
                
                # Check if tasks are complete
                if response["title"] == "TASKS_COMPLETE":
                    break
                    
                # Create code items
                code_items = []
                for item_data in response["code_items"]:
                    code_item = CodeItem(
                        type=item_data["type"],
                        name=item_data["name"],
                        parameters=item_data.get("parameters", []),
                        result=item_data.get("result", "None"),
                        description=item_data["description"]
                    )
                    code_items.append(code_item)
                
                # Create a new task
                task = Task(
                    title=response["title"],
                    description=response["description"],
                    code_items=code_items
                )
                
                generated_tasks.append(task)
                
                # Update messages to provide context for next task
                feedback_msg = f"Task '{task.title}' has been added to the epic."
                messages.append(Message(role="assistant", content=json.dumps(response)))
                messages.append(Message(role="user", content=feedback_msg))
                
            except Exception as e:
                print(f"Error generating task: {str(e)}")
                break
        
        # Update the epic with the generated tasks
        epic.tasks = generated_tasks
        epic.updated_at = datetime.now()
        
        return epic
    
    def evaluate_project_plan(self, project: Project) -> Tuple[bool, str]:
        """Evaluate the project plan for completeness and coherence"""
        plan_overview = f"""
        Project: {project.name}
        Description: {project.description}
        
        Epics ({len(project.epics)}):
        """
        
        for i, epic in enumerate(project.epics):
            plan_overview += f"\n{i+1}. {epic.title}: {epic.description}"
            plan_overview += f"\n   Tasks ({len(epic.tasks)}):"
            
            for j, task in enumerate(epic.tasks):
                plan_overview += f"\n   {i+1}.{j+1}. {task.title}"
        
        evaluation_prompt = f"""
        {plan_overview}
        
        Evaluate this project plan for:
        1. Completeness - Does it cover all necessary aspects of the project?
        2. Coherence - Do the epics and tasks flow logically?
        3. Feasibility - Is the plan realistic and achievable?
        4. Technical soundness - Are the right technologies being used?
        
        First, identify any issues or gaps in the plan.
        Then, provide an overall assessment: is this plan READY for implementation, or does it need REVISION?
        
        Respond with a JSON object that has two fields:
        - "status": either "READY" or "NEEDS_REVISION"
        - "feedback": detailed explanation of your assessment
        """
        
        messages = [
            Message(role="system", content="You are an expert software project manager with extensive experience in evaluating project plans."),
            Message(role="user", content=evaluation_prompt)
        ]
        
        evaluation_schema = {
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["READY", "NEEDS_REVISION"]},
                    "feedback": {"type": "string"}
                },
                "required": ["status", "feedback"]
            }
        }
        
        try:
            response = self.llm_client.structured_completion(
                messages,
                evaluation_schema,
                temperature=0.3,
                max_tokens=1000
            )
            
            return (response["status"] == "READY", response["feedback"])
            
        except Exception as e:
            print(f"Error evaluating project plan: {str(e)}")
            return (False, f"Error during evaluation: {str(e)}")