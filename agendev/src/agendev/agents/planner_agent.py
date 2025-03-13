# planner_agent.py
"""
Planner agent for requirement analysis and planning.

This agent is responsible for analyzing requirements, breaking down tasks,
and creating implementation plans.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set
import asyncio
import json
import os
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

from .agent_base import Agent, AgentStatus
from ..models.task_models import Task, TaskType, TaskPriority, TaskRisk, TaskGraph, TaskStatus
from ..models.planning_models import PlanSnapshot, SimulationConfig, MCTSNode
from ..search_algorithms import MCTSPlanner, AStarPathfinder
from ..probability_modeling import TaskProbabilityModel


class RequirementAnalysis(BaseModel):
    """Analysis of project requirements."""
    project_type: str
    frontend_framework: Optional[str] = None
    backend_framework: Optional[str] = None
    database_type: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    deployment_platforms: List[str] = Field(default_factory=list)
    third_party_services: List[str] = Field(default_factory=list)
    technical_stack: Dict[str, List[str]] = Field(default_factory=dict)
    complexity_estimate: int = Field(1, ge=1, le=10)
    estimated_duration_days: float = 1.0


class PlannerAgent(Agent):
    """
    Planner agent for requirement analysis and planning.
    
    This agent is responsible for:
    1. Analyzing user requirements and project prompts
    2. Breaking down projects into implementable tasks
    3. Creating implementation plans
    """
    
    def __init__(
        self,
        name: str = "Planner",
        description: str = "Analyzes requirements and plans implementation",
        agent_type: str = "planner"
    ):
        """Initialize the PlannerAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Store the most recent analysis and plan 
        # Initialize these as instance variables, not as Pydantic model fields
        self._current_analysis = None
        self._current_plan = None
        self._current_task_breakdown = None
    
    # Define properties to access these attributes safely
    @property
    def current_analysis(self) -> Optional[RequirementAnalysis]:
        return self._current_analysis
    
    @current_analysis.setter
    def current_analysis(self, value: Optional[RequirementAnalysis]):
        self._current_analysis = value
    
    @property
    def current_plan(self) -> Optional[List[Task]]:
        return self._current_plan
    
    @current_plan.setter
    def current_plan(self, value: Optional[List[Task]]):
        self._current_plan = value
    
    @property
    def current_task_breakdown(self) -> Optional[Dict[str, Any]]:
        return self._current_task_breakdown
    
    @current_task_breakdown.setter
    def current_task_breakdown(self, value: Optional[Dict[str, Any]]):
        self._current_task_breakdown = value
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process planning requests.
        
        Args:
            input_data: Input data for the planning request
            
        Returns:
            Dictionary with the planning results
        """
        # Check if LLM integration is available
        if not self.llm:
            raise ValueError("LLM integration not available")
        
        # Extract the action
        action = input_data.get("action", "")
        
        if action == "analyze_requirements":
            prompt = input_data.get("prompt", "")
            return await self._analyze_prompt(prompt)
        elif action == "create_plan":
            requirements = input_data.get("requirements", [])
            return await self._generate_plan()
        elif action == "analyze_feature":
            feature_description = input_data.get("feature_description", "")
            existing_requirements = input_data.get("existing_requirements", [])
            return await self._analyze_feature(feature_description, existing_requirements)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a project prompt to extract requirements.
        
        Args:
            prompt: The project prompt to analyze
            
        Returns:
            Dictionary with the analysis results
        """
        if not prompt:
            return {
                "success": False,
                "error": "Empty prompt provided"
            }
        
        # Create an action
        action = self.create_action(
            action_type="analyze_requirements",
            description="Analyzing project requirements",
            parameters={"prompt": prompt}
        )
        
        try:
            # Prepare a specific context for the LLM
            system_message = """You are an expert software architect specialized in analyzing project requirements.
            Your task is to analyze a project description and extract key information about the technical requirements.
            Focus on identifying the type of project, technology stack, key features, complexity, and requirements.
            Pay special attention to both frontend and backend requirements, as well as database needs.
            Be thorough in your analysis and consider both explicit and implicit requirements.
            
            For Python projects, be specific about frameworks (DASH, Flask, Django, FastAPI), database technologies (SQLite, PostgreSQL, MongoDB, FireBase),
            and whether it's a CLI tool, web app, data processing, machine learning application, etc.
            
            For web applications, identify whether it's a Single Page Application (SPA), static site, or server-rendered application.
            
            Provide your analysis in a structured JSON format that thoroughly represents all aspects of the project requirements."""
            
            # Define the analysis schema
            analysis_schema = {
                "name": "project_requirements",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "project_type": {
                            "type": "string", 
                            "description": "General project type (e.g., 'web-application', 'cli-tool', 'api', 'data-processing', 'machine-learning')"
                        },
                        "frontend_framework": {
                            "type": ["string", "null"],
                            "description": "The frontend framework if applicable (e.g., 'React', 'Vue', 'Angular', 'None')"
                        },
                        "backend_framework": {
                            "type": ["string", "null"],
                            "description": "The backend framework if applicable (e.g., 'Django', 'Flask', 'FastAPI', 'Express', 'None')"
                        },
                        "database_type": {
                            "type": ["string", "null"],
                            "description": "The database technology if applicable (e.g., 'PostgreSQL', 'MongoDB', 'SQLite', 'None')"
                        },
                        "language": {
                            "type": "string",
                            "description": "Primary programming language (e.g., 'Python', 'JavaScript', 'TypeScript')"
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key features to implement"
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of functional and non-functional requirements"
                        },
                        "deployment_platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Potential deployment platforms (e.g., 'Heroku', 'AWS', 'Vercel', 'Local')"
                        },
                        "third_party_services": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Third-party services or APIs to integrate with"
                        },
                        "technical_stack": {
                            "type": "object",
                            "properties": {
                                "frontend": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Frontend technologies and libraries"
                                },
                                "backend": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Backend technologies and libraries"
                                },
                                "database": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Database technologies and tools"
                                },
                                "devops": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "DevOps tools and technologies"
                                }
                            }
                        },
                        "complexity_estimate": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Estimated complexity on a scale of 1 to 10"
                        },
                        "estimated_duration_days": {
                            "type": "number",
                            "description": "Estimated development time in days"
                        },
                        "technologies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of all technologies involved"
                        }
                    },
                    "required": [
                        "project_type",
                        "language",
                        "features",
                        "requirements",
                        "technical_stack",
                        "complexity_estimate",
                        "estimated_duration_days",
                        "technologies"
                    ]
                }
            }
            
            # Query the LLM for structured analysis
            query_result = self.llm.structured_query(
                prompt=f"Analyze the following project requirements:\n\n{prompt}",
                json_schema=analysis_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                error_message = query_result.get("error", "Unknown error")
                raw_content = query_result.get("raw_content", "No content")
                
                # Log detailed error for debugging
                print(f"Analysis query failed: {error_message}")
                print(f"Raw content: {raw_content}")
                
                raise ValueError(f"Failed to analyze requirements: {error_message}")
            
            # Extract the result from the response
            analysis = query_result.get("result", {})
            
            # Store the analysis for future reference
            self.current_analysis = analysis
            
            # Complete the action
            self.complete_action(
                action=action,
                result={"analysis": analysis},
                status=AgentStatus.SUCCESS
            )
            
            return {
                "success": True,
                "requirements": analysis.get("requirements", []),
                "features": analysis.get("features", []),
                "project_type": analysis.get("project_type", ""),
                "language": analysis.get("language", ""),
                "technical_stack": analysis.get("technical_stack", {}),
                "frontend_framework": analysis.get("frontend_framework"),
                "backend_framework": analysis.get("backend_framework"),
                "database_type": analysis.get("database_type"),
                "technologies": analysis.get("technologies", []),
                "complexity_estimate": analysis.get("complexity_estimate", 5),
                "estimated_duration_days": analysis.get("estimated_duration_days", 1.0)
            }
            
        except Exception as e:
            # Handle any errors
            self.complete_action(
                action=action,
                result={"error": str(e)},
                status=AgentStatus.ERROR
            )
            
            return {
                "success": False,
                "error": f"Requirements analysis failed: {str(e)}"
            }
    
    async def _analyze_feature(self, feature_description: str, existing_requirements: List[str]) -> Dict[str, Any]:
        """
        Analyze a feature request to integrate with existing requirements.
        
        Args:
            feature_description: Description of the feature to analyze
            existing_requirements: List of existing project requirements
            
        Returns:
            Dictionary with the feature analysis results
        """
        if not feature_description:
            return {
                "success": False,
                "error": "Empty feature description provided"
            }
        
        # Create an action
        action = self.create_action(
            action_type="analyze_feature",
            description="Analyzing feature request",
            parameters={
                "feature_description": feature_description,
                "existing_requirements": existing_requirements
            }
        )
        
        try:
            # Prepare a specific context for the LLM
            system_message = """You are an expert software architect specialized in analyzing feature requests.
            Your task is to analyze a feature description and extract key information about the requirements.
            Focus on identifying how this feature integrates with the existing project, its technical requirements,
            and any new dependencies or changes needed to the current architecture.
            Consider both explicit and implicit requirements.
            
            Provide your analysis in a structured JSON format that thoroughly represents all aspects of the feature requirements."""
            
            # Define the feature analysis schema
            feature_schema = {
                "name": "feature_analysis",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "feature_name": {
                            "type": "string",
                            "description": "A concise name for the feature"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the feature"
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of requirements for this feature"
                        },
                        "components_affected": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Components that will need to be modified or created"
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New dependencies required for this feature"
                        },
                        "integration_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Points where this feature integrates with existing functionality"
                        },
                        "complexity_estimate": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Estimated complexity on a scale of 1 to 10"
                        },
                        "estimated_hours": {
                            "type": "number",
                            "description": "Estimated development time in hours"
                        }
                    },
                    "required": [
                        "feature_name",
                        "description",
                        "requirements",
                        "components_affected",
                        "integration_points",
                        "complexity_estimate",
                        "estimated_hours"
                    ]
                }
            }
            
            # Prepare information about existing requirements
            existing_req_info = "\n".join([f"- {req}" for req in existing_requirements])
            
            # Query the LLM for structured analysis
            query_result = self.llm.structured_query(
                prompt=f"""Analyze the following feature request:

Feature Request:
{feature_description}

Existing Project Requirements:
{existing_req_info}

Provide a detailed analysis of this feature request, including how it relates to the existing requirements.""",
                json_schema=feature_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                error_message = query_result.get("error", "Unknown error")
                raw_content = query_result.get("raw_content", "No content")
                
                # Log detailed error for debugging
                print(f"Feature analysis query failed: {error_message}")
                print(f"Raw content: {raw_content}")
                
                raise ValueError(f"Failed to analyze feature: {error_message}")
            
            # Extract the result from the response
            feature_analysis = query_result.get("result", {})
            
            # Complete the action
            self.complete_action(
                action=action,
                result={"feature_analysis": feature_analysis},
                status=AgentStatus.SUCCESS
            )
            
            return {
                "success": True,
                "feature": {
                    "name": feature_analysis.get("feature_name", ""),
                    "description": feature_analysis.get("description", ""),
                    "requirements": feature_analysis.get("requirements", []),
                    "components_affected": feature_analysis.get("components_affected", []),
                    "dependencies": feature_analysis.get("dependencies", []),
                    "integration_points": feature_analysis.get("integration_points", []),
                    "complexity_estimate": feature_analysis.get("complexity_estimate", 5),
                    "estimated_hours": feature_analysis.get("estimated_hours", 8)
                },
                "requirements": feature_analysis.get("requirements", [])
            }
            
        except Exception as e:
            # Handle any errors
            self.complete_action(
                action=action,
                result={"error": str(e)},
                status=AgentStatus.ERROR
            )
            
            return {
                "success": False,
                "error": f"Feature analysis failed: {str(e)}"
            }
    
    async def _break_down_tasks(self) -> Dict[str, Any]:
        """
        Break down a project or feature into implementable tasks.
        
        Returns:
            Dictionary with the task breakdown results
        """
        if not self.current_analysis:
            return {
                "success": False,
                "error": "No current analysis available for task breakdown"
            }
        
        # Create an action
        action = self.create_action(
            action_type="break_down_tasks",
            description="Breaking down project into tasks",
            parameters={"analysis": self.current_analysis}
        )
        
        try:
            # Prepare a specific context for the LLM
            system_message = """You are an expert software architect specialized in breaking down projects into implementable tasks.
            Your task is to take a project analysis and break it down into concrete, actionable implementation tasks.
            Each task should be clear, specific, and focused on a single aspect of the implementation.
            Tasks should be organized in a logical sequence and grouped by component or feature.
            Consider dependencies between tasks and prioritize them accordingly.
            
            Provide your task breakdown in a structured JSON format that thoroughly represents all aspects of the implementation plan."""
            
            # Define the task breakdown schema
            task_schema = {
                "name": "task_breakdown",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project"
                        },
                        "component_groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the component group (e.g., 'Frontend', 'Backend', 'Database')"
                                    },
                                    "components": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "Name of the component"
                                                },
                                                "description": {
                                                    "type": "string",
                                                    "description": "Description of the component"
                                                },
                                                "tasks": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {
                                                                "type": "string",
                                                                "description": "Unique identifier for the task"
                                                            },
                                                            "name": {
                                                                "type": "string",
                                                                "description": "Name of the task"
                                                            },
                                                            "description": {
                                                                "type": "string",
                                                                "description": "Detailed description of the task"
                                                            },
                                                            "priority": {
                                                                "type": "string",
                                                                "enum": ["high", "medium", "low"],
                                                                "description": "Priority of the task"
                                                            },
                                                            "dependencies": {
                                                                "type": "array",
                                                                "items": {"type": "string"},
                                                                "description": "List of task IDs that this task depends on"
                                                            },
                                                            "estimated_hours": {
                                                                "type": "number",
                                                                "description": "Estimated time to complete in hours"
                                                            }
                                                        },
                                                        "required": [
                                                            "id",
                                                            "name",
                                                            "description",
                                                            "priority",
                                                            "estimated_hours"
                                                        ]
                                                    }
                                                }
                                            },
                                            "required": ["name", "description", "tasks"]
                                        }
                                    }
                                },
                                "required": ["name", "components"]
                            }
                        }
                    },
                    "required": ["project_name", "component_groups"]
                }
            }
            
            # Query the LLM for structured task breakdown
            query_result = self.llm.structured_query(
                prompt=f"""Break down the following project into implementable tasks:

Project Analysis:
{json.dumps(self.current_analysis, indent=2)}

Provide a detailed task breakdown for this project, organizing tasks by component groups and components within each group.""",
                json_schema=task_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                error_message = query_result.get("error", "Unknown error")
                raw_content = query_result.get("raw_content", "No content")
                
                # Log detailed error for debugging
                print(f"Task breakdown query failed: {error_message}")
                print(f"Raw content: {raw_content}")
                
                raise ValueError(f"Failed to break down tasks: {error_message}")
            
            # Extract the result from the response
            task_breakdown = query_result.get("result", {})
            
            # Store for later use
            self.current_task_breakdown = task_breakdown
            
            # Complete the action
            self.complete_action(
                action=action,
                result={"task_breakdown": task_breakdown},
                status=AgentStatus.SUCCESS
            )
            
            return {
                "success": True,
                "task_breakdown": task_breakdown
            }
            
        except Exception as e:
            # Handle any errors
            self.complete_action(
                action=action,
                result={"error": str(e)},
                status=AgentStatus.ERROR
            )
            
            return {
                "success": False,
                "error": f"Task breakdown failed: {str(e)}"
            }
    
    async def _generate_plan(self) -> Dict[str, Any]:
        """
        Generate an implementation plan based on the current analysis.
        
        Returns:
            Dictionary with the implementation plan
        """
        if not self.current_analysis:
            return {
                "success": False,
                "error": "No current analysis available for planning"
            }
        
        # Create an action
        action = self.create_action(
            action_type="generate_plan",
            description="Generating implementation plan",
            parameters={"analysis": self.current_analysis}
        )
        
        try:
            # First, break down tasks if not already done
            if not self.current_task_breakdown:
                task_breakdown_result = await self._break_down_tasks()
                if not task_breakdown_result.get("success", False):
                    raise ValueError(f"Failed to break down tasks: {task_breakdown_result.get('error', 'Unknown error')}")
            
            # Create a Task Graph from the task breakdown
            task_graph, tasks = self._create_task_graph_from_breakdown()
            
            # Use MCTS to find optimal plan
            optimal_sequence = self._generate_optimal_plan_with_mcts(task_graph)
            
            # Prepare a specific context for the LLM
            system_message = """You are an expert software architect specialized in creating implementation plans.
            Your task is to take a project analysis, task breakdown, and an optimal task sequence and create a detailed implementation plan.
            The plan should include project setup, development phases, testing strategies, and deployment steps.
            Consider dependencies between tasks and follow the provided optimal sequence for implementation.
            Be specific about frameworks, libraries, and tools to be used.
            
            Pay special attention to project structure and organization, especially for Python projects.
            For Python projects, include details about virtual environments, package management, module structure, 
            and testing frameworks.
            
            Provide your implementation plan in a structured JSON format that thoroughly represents all aspects of the plan."""
            
            # Define the implementation plan schema
            plan_schema = {
                "name": "implementation_plan",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project"
                        },
                        "project_structure": {
                            "type": "object",
                            "properties": {
                                "directories": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": {
                                                "type": "string",
                                                "description": "Path to the directory"
                                            },
                                            "purpose": {
                                                "type": "string",
                                                "description": "Purpose of the directory"
                                            }
                                        },
                                        "required": ["path", "purpose"]
                                    }
                                },
                                "key_files": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": {
                                                "type": "string",
                                                "description": "Path to the file"
                                            },
                                            "purpose": {
                                                "type": "string",
                                                "description": "Purpose of the file"
                                            },
                                            "content_outline": {
                                                "type": "string",
                                                "description": "Outline of the file's contents"
                                            }
                                        },
                                        "required": ["path", "purpose"]
                                    }
                                }
                            },
                            "required": ["directories", "key_files"]
                        },
                        "phases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the phase (e.g., 'Setup', 'Core Implementation', 'Testing')"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the phase"
                                    },
                                    "steps": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "Name of the step"
                                                },
                                                "description": {
                                                    "type": "string",
                                                    "description": "Description of the step"
                                                },
                                                "tasks": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "List of task IDs associated with this step"
                                                },
                                                "commands": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "Example commands to execute for this step"
                                                }
                                            },
                                            "required": ["name", "description"]
                                        }
                                    }
                                },
                                "required": ["name", "description", "steps"]
                            }
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the dependency"
                                    },
                                    "version": {
                                        "type": "string",
                                        "description": "Version of the dependency"
                                    },
                                    "purpose": {
                                        "type": "string",
                                        "description": "Purpose of the dependency"
                                    }
                                },
                                "required": ["name", "purpose"]
                            }
                        },
                        "setup_instructions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Instructions for setting up the project"
                        }
                    },
                    "required": [
                        "project_name",
                        "project_structure",
                        "phases",
                        "dependencies",
                        "setup_instructions"
                    ]
                }
            }
            
            # Include the optimal sequence in the prompt
            sequence_info = "\n\nOptimal Task Sequence (determined by Monte Carlo Tree Search algorithm):\n"
            for i, task_id in enumerate(optimal_sequence):
                task = tasks.get(task_id)
                if task:
                    sequence_info += f"{i+1}. {task.title} - {task.description[:50]}...\n"
            
            # Query the LLM for structured implementation plan
            query_result = self.llm.structured_query(
                prompt=f"""Generate an implementation plan based on the following project analysis, task breakdown, and optimal task sequence:

Project Analysis:
{json.dumps(self.current_analysis, indent=2)}

Task Breakdown:
{json.dumps(self.current_task_breakdown, indent=2)}

{sequence_info}

Provide a detailed implementation plan for this project, following the optimal task sequence provided, including project structure, implementation phases, and setup instructions.""",
                json_schema=plan_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                error_message = query_result.get("error", "Unknown error")
                raw_content = query_result.get("raw_content", "No content")
                
                # Log detailed error for debugging
                print(f"Implementation plan query failed: {error_message}")
                print(f"Raw content: {raw_content}")
                
                raise ValueError(f"Failed to generate plan: {error_message}")
            
            # Extract the result from the response
            implementation_plan = query_result.get("result", {})
            
            # Store the plan for future reference
            self.current_plan = implementation_plan
            
            # Complete the action
            self.complete_action(
                action=action,
                result={"plan": implementation_plan},
                status=AgentStatus.SUCCESS
            )
            
            return {
                "success": True,
                "plan": implementation_plan
            }
            
        except Exception as e:
            # Handle any errors
            self.complete_action(
                action=action,
                result={"error": str(e)},
                status=AgentStatus.ERROR
            )
            
            return {
                "success": False,
                "error": f"Plan generation failed: {str(e)}"
            }
    
    def _create_task_graph_from_breakdown(self) -> Tuple[TaskGraph, Dict[UUID, Task]]:
        """
        Create a TaskGraph from the current task breakdown.
        
        Returns:
            Tuple of (TaskGraph, Dict[UUID, Task])
        """
        task_graph = TaskGraph()
        tasks = {}
        
        # Extract tasks from breakdown
        task_data = self.current_task_breakdown.get("tasks", [])
        
        # Create Task objects
        for task_info in task_data:
            task_id = uuid4()
            task = Task(
                id=task_id,
                title=task_info.get("name", "Unnamed Task"),
                description=task_info.get("description", ""),
                task_type=TaskType(task_info.get("type", "implementation")),
                priority=TaskPriority(task_info.get("priority", "medium")),
                risk=TaskRisk(task_info.get("risk", "medium")),
                estimated_duration_hours=task_info.get("estimated_hours", 1.0)
            )
            tasks[task_id] = task
            task_graph.add_task(task)
        
        # Add dependencies
        for task_info in task_data:
            if "dependencies" in task_info and task_info["dependencies"]:
                # Find the task with this name
                task_name = task_info.get("name")
                source_id = None
                for task_id, task in tasks.items():
                    if task.title == task_name:
                        source_id = task_id
                        break
                
                if source_id:
                    for dep_name in task_info["dependencies"]:
                        # Find the dependency task id
                        target_id = None
                        for task_id, task in tasks.items():
                            if task.title == dep_name:
                                target_id = task_id
                                break
                        
                        if target_id:
                            # Add the dependency
                            task_graph.add_dependency(
                                Dependency(
                                    source_id=source_id,
                                    target_id=target_id,
                                    dependency_type="blocks",
                                    strength=1.0
                                )
                            )
        
        return task_graph, tasks
    
    def _generate_optimal_plan_with_mcts(self, task_graph: TaskGraph) -> List[UUID]:
        """
        Use Monte Carlo Tree Search to generate an optimal task sequence.
        
        Args:
            task_graph: The TaskGraph containing tasks and dependencies
            
        Returns:
            List of task IDs in optimal sequence
        """
        # Create a simulation config
        config = SimulationConfig(
            max_iterations=1000,  # Adjust based on complexity
            exploration_weight=1.414,  # UCT constant
            max_depth=30,
            time_limit_seconds=20.0  # Maximum time to spend on optimization
        )
        
        # Create the MCTS planner
        mcts_planner = MCTSPlanner(
            task_graph=task_graph,
            llm_integration=self.llm,
            config=config
        )
        
        try:
            # Run the simulation
            simulation_result = mcts_planner.run_simulation()
            
            # Get the best sequence
            best_sequence = simulation_result.get("best_sequence", [])
            
            if not best_sequence:
                print("MCTS optimization returned empty sequence, falling back to A* pathfinding")
                # Fallback to A* if MCTS fails
                best_sequence = self._generate_optimal_plan_with_astar(task_graph)
            
            return best_sequence
        except Exception as e:
            print(f"Error in MCTS planning: {e}")
            # Fallback to A* if MCTS fails
            return self._generate_optimal_plan_with_astar(task_graph)
    
    def _generate_optimal_plan_with_astar(self, task_graph: TaskGraph) -> List[UUID]:
        """
        Use A* pathfinding to generate an optimal task sequence.
        
        Args:
            task_graph: The TaskGraph containing tasks and dependencies
            
        Returns:
            List of task IDs in optimal sequence
        """
        try:
            from ..search_algorithms import AStarPathfinder
            
            # Define a custom heuristic function for A* that considers task priority and risk
            def custom_heuristic(node_id: UUID, remaining_tasks: Set[UUID]) -> float:
                # The fewer remaining tasks, the better
                base_score = len(remaining_tasks) * 10
                
                # Add risk and priority factors
                if node_id in task_graph.tasks:
                    task = task_graph.tasks[node_id]
                    
                    # Prioritize high-priority tasks
                    if task.priority == TaskPriority.HIGH:
                        base_score -= 5
                    elif task.priority == TaskPriority.CRITICAL:
                        base_score -= 10
                    
                    # Add penalty for high-risk tasks
                    if task.risk == TaskRisk.HIGH:
                        base_score += 5
                    elif task.risk == TaskRisk.CRITICAL:
                        base_score += 10
                    
                    # Consider task duration
                    base_score += task.estimated_duration_hours * 2
                
                return base_score
            
            # Create the A* pathfinder
            pathfinder = AStarPathfinder(
                task_graph=task_graph,
                heuristic_function=custom_heuristic
            )
            
            # Find tasks with no dependencies (start tasks)
            start_tasks = []
            for task_id, task in task_graph.tasks.items():
                if not task.dependencies:
                    start_tasks.append(task_id)
            
            if not start_tasks:
                # If no start tasks, just use the first task
                if task_graph.tasks:
                    start_tasks = [next(iter(task_graph.tasks.keys()))]
                else:
                    return []
            
            # Define goal condition: all tasks completed
            def all_tasks_completed(completed_tasks: Set[UUID]) -> bool:
                return len(completed_tasks) == len(task_graph.tasks)
            
            # Find the path
            path, path_info = pathfinder.find_path(
                start_tasks=start_tasks,
                goal_condition=all_tasks_completed
            )
            
            print(f"A* pathfinding found a path of length {len(path)} with cost {path_info.get('total_cost', 0)}")
            
            return path
        except Exception as e:
            print(f"Error in A* pathfinding: {e}")
            # Final fallback to critical path if A* fails
            try:
                return task_graph.get_critical_path()
            except Exception:
                # If everything fails, just return tasks in order
                return list(task_graph.tasks.keys()) 