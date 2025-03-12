"""Enhanced LLM client for diverse query types and context management."""
from typing import List, Dict, Union, Optional, Any, Tuple
from uuid import UUID, uuid4
import json
import time

from .llm_module import LLMClient, Message
from .context_management import ContextManager
from .models.task_models import Task, Epic, TaskType, TaskStatus


class LLMIntegration:
    """Enhanced LLM client with context management and specialized queries."""
    
    def __init__(
        self, 
        llm_client: Optional[LLMClient] = None,
        context_manager: Optional[ContextManager] = None,
        default_model: str = "qwen2.5-7b-instruct",
        default_temperature: float = 0.7
    ):
        """
        Initialize the LLM integration with clients and settings.
        
        Args:
            llm_client: LLM client for API calls
            context_manager: Context manager for retrieval
            default_model: Default model to use for completions
            default_temperature: Default temperature setting
        """
        self.llm_client = llm_client or LLMClient()
        self.context_manager = context_manager
        self.default_model = default_model
        self.default_temperature = default_temperature
        
        # Initialize the context manager if it's not provided
        if self.context_manager is None and self.llm_client is not None:
            self.context_manager = ContextManager(llm_client=self.llm_client)
    
    def generate_with_context(
        self,
        prompt: str,
        task_description: Optional[str] = None,
        additional_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = -1,
        stream: bool = False
    ) -> str:
        """
        Generate content with relevant context.
        
        Args:
            prompt: The prompt to complete
            task_description: Description of the task (for context retrieval)
            additional_context: Any additional context to include
            model: Model to use (defaults to self.default_model)
            temperature: Temperature setting (defaults to self.default_temperature)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated content
        """
        # Build context
        context = ""
        
        if task_description and self.context_manager:
            context += self.context_manager.build_context_for_task(task_description)
        
        if additional_context:
            context += f"\nAdditional context:\n{additional_context}\n\n"
        
        # Create system message with context
        system_message = Message(
            role="system",
            content=f"You are a helpful assistant for software development tasks. "
                    f"Respond based on the following context:\n\n{context}"
        )
        
        # Create user message
        user_message = Message(role="user", content=prompt)
        
        # Generate completion
        return self.llm_client.chat_completion(
            messages=[system_message, user_message],
            model=model or self.default_model,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens,
            stream=stream
        )
    
    def generate_implementation(
        self,
        task: Task,
        additional_context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate an implementation for a specific task.
        
        Args:
            task: The task to implement
            additional_context: Any additional context to include
            temperature: Temperature setting
            
        Returns:
            Generated implementation code
        """
        # Adjust temperature based on task type
        if temperature is None:
            if task.task_type == TaskType.IMPLEMENTATION:
                # Lower temperature for core implementation
                temperature = max(0.3, self.default_temperature - 0.3)
            elif task.task_type == TaskType.DOCUMENTATION:
                # Higher temperature for documentation
                temperature = min(0.9, self.default_temperature + 0.2)
            else:
                temperature = self.default_temperature
        
        # Build the prompt
        prompt = f"""
Task: {task.title}

Description: {task.description}

Implement a high-quality solution for this task. Your implementation should be:
1. Well-structured and organized
2. Properly documented with docstrings
3. Following best practices for Python development
4. Compatible with the project's existing code

Return only the implementation code without any explanations or comments outside of the code.
"""
        
        return self.generate_with_context(
            prompt=prompt,
            task_description=task.description,
            additional_context=additional_context,
            temperature=temperature
        )
    
    def generate_task_breakdown(
        self,
        epic: Epic,
        num_tasks: int = 5,
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a breakdown of tasks for an epic.
        
        Args:
            epic: The epic to break down
            num_tasks: Target number of tasks to generate
            temperature: Temperature setting
            
        Returns:
            List of task data dictionaries
        """
        # Use a structured completion for the task breakdown
        json_schema = {
            "name": "task_breakdown",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "task_type": {"type": "string", "enum": [t.value for t in TaskType]},
                                "estimated_hours": {"type": "number"},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "risk_level": {"type": "string", "enum": ["low", "medium", "high", "very_high"]}
                            },
                            "required": ["title", "description", "task_type", "estimated_hours"]
                        }
                    }
                },
                "required": ["tasks"]
            }
        }
        
        # Create system message
        system_message = Message(
            role="system",
            content="You are a project planning assistant that breaks down epics into implementable tasks."
        )
        
        # Create user message
        user_message = Message(
            role="user",
            content=f"""
Break down the following epic into approximately {num_tasks} implementable tasks:

Epic: {epic.title}
Description: {epic.description}

For each task, provide:
1. A clear, specific title
2. A detailed description of what needs to be done
3. The task type (implementation, refactoring, testing, documentation, research)
4. Estimated hours to complete
5. Dependencies on other tasks (if any, as task titles)
6. Risk level (low, medium, high, very_high)

Tasks should be granular enough to be completed in 1-8 hours each.
"""
        )
        
        # Generate task breakdown
        response = self.llm_client.structured_completion(
            messages=[system_message, user_message],
            json_schema=json_schema,
            model=self.default_model,
            temperature=temperature or self.default_temperature
        )
        
        # Parse the response
        if isinstance(response, str):
            try:
                return json.loads(response)["tasks"]
            except (json.JSONDecodeError, KeyError):
                return []
        
        return response.get("tasks", [])
    
    def estimate_task_success(
        self,
        task: Task,
        context: Optional[str] = None
    ) -> float:
        """
        Estimate the probability of successfully implementing a task.
        
        Args:
            task: The task to estimate
            context: Additional context
            
        Returns:
            Estimated success probability (0.0 to 1.0)
        """
        json_schema = {
            "name": "task_estimate",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"}
                },
                "required": ["success_probability", "reasoning"]
            }
        }
        
        # Create system message
        system_message = Message(
            role="system",
            content="You are a project risk assessment assistant that estimates the probability of successfully implementing tasks."
        )
        
        # Create user message
        user_message = Message(
            role="user",
            content=f"""
Estimate the probability of successfully implementing the following task:

Task: {task.title}
Description: {task.description}
Type: {task.task_type.value}
Estimated hours: {task.estimated_hours}
Risk level: {task.risk_level.value}

{context or ''}

Provide:
1. A probability between 0.0 (certain failure) and 1.0 (certain success)
2. Brief reasoning for your estimate

Base your estimate on the complexity, clarity of requirements, technical challenges, and dependencies.
"""
        )
        
        # Generate estimate
        response = self.llm_client.structured_completion(
            messages=[system_message, user_message],
            json_schema=json_schema,
            model=self.default_model,
            temperature=0.3  # Lower temperature for more consistent estimates
        )
        
        # Parse the response
        if isinstance(response, str):
            try:
                success_probability = json.loads(response)["success_probability"]
                return float(success_probability)
            except (json.JSONDecodeError, KeyError, ValueError):
                return 0.5  # Default probability
        
        return float(response.get("success_probability", 0.5))
    
    def generate_code_review(
        self,
        code: str,
        task: Optional[Task] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a code review for the provided code.
        
        Args:
            code: The code to review
            task: The related task (optional)
            temperature: Temperature setting
            
        Returns:
            Review data
        """
        json_schema = {
            "name": "code_review",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "overall_quality": {"type": "number", "minimum": 1, "maximum": 10},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["bug", "style", "performance", "security", "design"]},
                                "description": {"type": "string"},
                                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                                "suggestion": {"type": "string"}
                            },
                            "required": ["type", "description", "severity", "suggestion"]
                        }
                    },
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "summary": {"type": "string"}
                },
                "required": ["overall_quality", "issues", "strengths", "summary"]
            }
        }
        
        # Create system message
        system_message = Message(
            role="system",
            content="You are a code review assistant that provides constructive feedback on code quality."
        )
        
        # Create user message
        user_content = "Review the following code and provide feedback on its quality:\n\n```python\n" + code + "\n```\n"
        
        if task:
            user_content += f"\nThis code implements the following task:\n\nTask: {task.title}\nDescription: {task.description}\n"
        
        user_message = Message(role="user", content=user_content)
        
        # Generate review
        response = self.llm_client.structured_completion(
            messages=[system_message, user_message],
            json_schema=json_schema,
            model=self.default_model,
            temperature=temperature or 0.4  # Lower temperature for consistent reviews
        )
        
        # Parse the response
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "overall_quality": 5,
                    "issues": [],
                    "strengths": [],
                    "summary": "Could not analyze the code."
                }
        
        return response