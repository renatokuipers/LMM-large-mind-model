# utils/llm_client.py
from typing import List, Dict, Any, Optional
import requests
import json
from dataclasses import dataclass
from core.schemas import ProjectConfig, Task, Epic

@dataclass
class Message:
    role: str
    content: str

class LLMClient:
    """Wrapper for interacting with our local LLM"""
    def __init__(self, base_url: str = "http://192.168.2.12:1234"):
        """Initialize the LLM client with the base URL of the API."""
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}

    def chat_completion(
        self,
        messages: List[Message],
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Send a chat completion request to the API."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        schema_name: str = "structured_response",
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict:
        """Send a structured completion request to the API."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        # Ensure the schema is properly wrapped
        if "name" in json_schema and "strict" in json_schema and "schema" in json_schema:
            wrapped_schema = json_schema
        else:
            wrapped_schema = {
                "name": schema_name,
                "strict": "true",  # Note: API expects a string "true", not a boolean
                "schema": json_schema
            }
        
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": wrapped_schema
            },
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    
    def generate_epics(self, project_name: str, project_description: str, config: ProjectConfig) -> List[Epic]:
        """Generate EPICs for a project"""
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
        
        messages = [
            Message(role="system", content=f"""You are an expert software architect with deep expertise in {config.language} development.
                   You will break down a project into logical EPICs and tasks for implementation.
                   Each EPIC should represent a major component or feature of the project.
                   Each task should be granular enough to implement one class, function, or component at a time.
                   Ensure dependencies between tasks are properly tracked.
                   For each task, list the output files that will be created or modified."""),
            Message(role="user", content=f"""Create a detailed plan for a {config.language} project with the following details:
                   Project Name: {project_name}
                   Description: {project_description}
                   
                   Create EPICs and tasks that would allow an AI to implement this step by step.
                   Use logical IDs, e.g., 'setup-core', 'implement-database', etc.
                   Be specific and detailed in your descriptions.
                   Ensure task dependencies are correctly specified (task IDs as strings).
                   For each task, specify the files that will be created or modified.""")
        ]
        
        result = self.structured_completion(messages, epic_schema)
        epics = []
        
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
            epics.append(epic)
            
        return epics