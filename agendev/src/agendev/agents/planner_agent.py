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
from ..search_algorithms import MCTSPlanner
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
        self.current_analysis: Optional[RequirementAnalysis] = None
        self.current_plan: Optional[List[Task]] = None
    
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
            query_result = await self.llm.structured_query(
                prompt=f"Analyze the following project requirements:\n\n{prompt}",
                json_schema=analysis_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                raise ValueError(f"Failed to analyze requirements: {query_result.get('error', 'Unknown error')}")
            
            analysis = query_result.get("result", {})
            
            # Store for later use
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
            query_result = await self.llm.structured_query(
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
                raise ValueError(f"Failed to analyze feature: {query_result.get('error', 'Unknown error')}")
            
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
            query_result = await self.llm.structured_query(
                prompt=f"""Break down the following project into implementable tasks:

Project Analysis:
{json.dumps(self.current_analysis, indent=2)}

Provide a detailed task breakdown for this project, organizing tasks by component groups and components within each group.""",
                json_schema=task_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                raise ValueError(f"Failed to break down tasks: {query_result.get('error', 'Unknown error')}")
            
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
            if not hasattr(self, "current_task_breakdown"):
                task_breakdown_result = await self._break_down_tasks()
                if not task_breakdown_result.get("success", False):
                    raise ValueError(f"Failed to break down tasks: {task_breakdown_result.get('error', 'Unknown error')}")
            
            # Prepare a specific context for the LLM
            system_message = """You are an expert software architect specialized in creating implementation plans.
            Your task is to take a project analysis and task breakdown and create a detailed implementation plan.
            The plan should include project setup, development phases, testing strategies, and deployment steps.
            Consider dependencies between tasks and provide a logical sequence for implementation.
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
            
            # Query the LLM for structured implementation plan
            query_result = await self.llm.structured_query(
                prompt=f"""Generate an implementation plan based on the following project analysis and task breakdown:

Project Analysis:
{json.dumps(self.current_analysis, indent=2)}

Task Breakdown:
{json.dumps(self.current_task_breakdown, indent=2)}

Provide a detailed implementation plan for this project, including project structure, implementation phases, and setup instructions.""",
                json_schema=plan_schema,
                clear_context=True
            )
            
            # Check for successful result
            if not query_result.get("success", False):
                raise ValueError(f"Failed to generate plan: {query_result.get('error', 'Unknown error')}")
            
            implementation_plan = query_result.get("result", {})
            
            # Store for later use
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