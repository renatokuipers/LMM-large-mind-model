"""
AgenflowManager to orchestrate the multi-agent workflow.

This module provides the main manager class that coordinates the flow between 
different specialized agents in the AgenDev system.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import asyncio
import json
import logging
import os
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

from .agents.agent_base import Agent, AgentStatus
from .agents.planner_agent import PlannerAgent
from .agents.code_agent import CodeAgent
from .agents.deployment_agent import DeploymentAgent
from .agents.integration_agent import IntegrationAgent
from .agents.knowledge_agent import KnowledgeAgent
from .agents.web_automation_agent import WebAutomationAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgenflowManager")


class ProjectContext(BaseModel):
    """Project context information."""
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    project_name: str
    workspace_path: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    requirements: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of a task execution."""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    

class AgenflowManager:
    """
    Manages the flow between different specialized agents.
    
    This class is responsible for orchestrating the entire workflow,
    from handling user requests to delegating tasks to appropriate agents
    and managing the overall state of the system.
    """
    
    def __init__(self, workspace_path: Optional[str] = None):
        """
        Initialize the AgenflowManager.
        
        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = workspace_path or os.getcwd()
        
        # Initialize agents
        self.agents: Dict[str, Agent] = {}
        self._initialize_agents()
        
        # Track project contexts
        self.projects: Dict[str, ProjectContext] = {}
        
        # Track task results
        self.task_results: List[TaskResult] = []
        
        logger.info(f"AgenflowManager initialized with workspace: {self.workspace_path}")
    
    def _initialize_agents(self):
        """Initialize all agent instances."""
        # Create agent instances
        self.agents["planner"] = PlannerAgent()
        self.agents["code"] = CodeAgent()
        self.agents["deployment"] = DeploymentAgent()
        self.agents["integration"] = IntegrationAgent()
        self.agents["knowledge"] = KnowledgeAgent()
        self.agents["web_automation"] = WebAutomationAgent()
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user request by routing it to the appropriate flow.
        
        Args:
            request: Dictionary containing the user request
            
        Returns:
            Dictionary with the processing results
        """
        request_type = request.get("type", "")
        
        if request_type == "create_project":
            return await self._create_project(request)
        elif request_type == "implement_feature":
            return await self._implement_feature(request)
        elif request_type == "deploy_project":
            return await self._deploy_project(request)
        elif request_type == "answer_query":
            return await self._answer_query(request)
        elif request_type == "setup_repository":
            return await self._setup_repository(request)
        else:
            return {
                "success": False,
                "error": f"Unknown request type: {request_type}"
            }
    
    async def _create_project(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new project based on user requirements.
        
        Args:
            request: Dictionary containing project creation details
            
        Returns:
            Dictionary with project creation results
        """
        project_name = request.get("project_name", "")
        description = request.get("description", "")
        prompt = request.get("prompt", "")
        
        if not project_name:
            return {
                "success": False,
                "error": "Project name is required"
            }
        
        if not prompt:
            return {
                "success": False,
                "error": "Project prompt is required"
            }
        
        # Create project directory
        project_path = os.path.join(self.workspace_path, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create project context
        project_context = ProjectContext(
            project_name=project_name,
            workspace_path=project_path,
            description=description
        )
        
        self.projects[project_context.project_id] = project_context
        
        # Step 1: Use the planner agent to analyze requirements
        planning_input = {
            "action": "analyze_requirements",
            "prompt": prompt,
            "project_name": project_name,
            "workspace_path": project_path
        }
        
        planning_result = await self.agents["planner"].process(planning_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="planner",
            action="analyze_requirements",
            input_data=planning_input,
            output_data=planning_result,
            success=planning_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        if not planning_result.get("success", False):
            return {
                "success": False,
                "error": f"Requirements analysis failed: {planning_result.get('error', 'Unknown error')}",
                "planning_result": planning_result
            }
        
        # Step 2: Use the planner to create an implementation plan
        planning_input = {
            "action": "create_plan",
            "requirements": planning_result.get("requirements", []),
            "project_name": project_name,
            "workspace_path": project_path
        }
        
        plan_result = await self.agents["planner"].process(planning_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="planner",
            action="create_plan",
            input_data=planning_input,
            output_data=plan_result,
            success=plan_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        if not plan_result.get("success", False):
            return {
                "success": False,
                "error": f"Implementation planning failed: {plan_result.get('error', 'Unknown error')}",
                "planning_result": planning_result,
                "plan_result": plan_result
            }
        
        # Step 3: Use the code agent to implement the project
        implementation_input = {
            "action": "implement_project",
            "plan": plan_result.get("plan", {}),
            "project_name": project_name,
            "workspace_path": project_path,
            "requirements": planning_result.get("requirements", [])
        }
        
        implementation_result = await self.agents["code"].process(implementation_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="code",
            action="implement_project",
            input_data=implementation_input,
            output_data=implementation_result,
            success=implementation_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        # Step 4: Use the integration agent to check and fix integration issues
        integration_input = {
            "action": "integrate_components",
            "project_name": project_name,
            "workspace_path": project_path,
            "modified_files": implementation_result.get("modified_files", [])
        }
        
        integration_result = await self.agents["integration"].process(integration_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="integration",
            action="integrate_components",
            input_data=integration_input,
            output_data=integration_result,
            success=integration_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        # Update project context with requirements and technologies
        self.projects[project_context.project_id].requirements = planning_result.get("requirements", [])
        self.projects[project_context.project_id].technologies = planning_result.get("technologies", [])
        self.projects[project_context.project_id].updated_at = datetime.now()
        
        # Return the combined results
        return {
            "success": implementation_result.get("success", False) and integration_result.get("success", False),
            "project_id": project_context.project_id,
            "project_name": project_name,
            "project_path": project_path,
            "planning_result": planning_result,
            "plan_result": plan_result,
            "implementation_result": implementation_result,
            "integration_result": integration_result,
            "summary": f"Project {project_name} created successfully"
        }
    
    async def _implement_feature(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement a new feature in an existing project.
        
        Args:
            request: Dictionary containing feature implementation details
            
        Returns:
            Dictionary with feature implementation results
        """
        project_id = request.get("project_id", "")
        feature_description = request.get("feature_description", "")
        
        if not project_id or project_id not in self.projects:
            return {
                "success": False,
                "error": "Invalid project ID"
            }
        
        if not feature_description:
            return {
                "success": False,
                "error": "Feature description is required"
            }
        
        project_context = self.projects[project_id]
        
        # Step 1: Use the planner agent to analyze the feature request
        planning_input = {
            "action": "analyze_feature",
            "feature_description": feature_description,
            "project_name": project_context.project_name,
            "workspace_path": project_context.workspace_path,
            "existing_requirements": project_context.requirements
        }
        
        planning_result = await self.agents["planner"].process(planning_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="planner",
            action="analyze_feature",
            input_data=planning_input,
            output_data=planning_result,
            success=planning_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        if not planning_result.get("success", False):
            return {
                "success": False,
                "error": f"Feature analysis failed: {planning_result.get('error', 'Unknown error')}",
                "planning_result": planning_result
            }
        
        # Step 2: Use the code agent to implement the feature
        implementation_input = {
            "action": "implement_feature",
            "feature": planning_result.get("feature", {}),
            "project_name": project_context.project_name,
            "workspace_path": project_context.workspace_path
        }
        
        implementation_result = await self.agents["code"].process(implementation_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="code",
            action="implement_feature",
            input_data=implementation_input,
            output_data=implementation_result,
            success=implementation_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        # Step 3: Use the integration agent to check and fix integration issues
        integration_input = {
            "action": "integrate_feature",
            "project_name": project_context.project_name,
            "workspace_path": project_context.workspace_path,
            "modified_files": implementation_result.get("modified_files", [])
        }
        
        integration_result = await self.agents["integration"].process(integration_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="integration",
            action="integrate_feature",
            input_data=integration_input,
            output_data=integration_result,
            success=integration_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        # Update project context
        self.projects[project_id].updated_at = datetime.now()
        
        # Add new requirements if any
        new_requirements = planning_result.get("requirements", [])
        for req in new_requirements:
            if req not in self.projects[project_id].requirements:
                self.projects[project_id].requirements.append(req)
        
        # Return the combined results
        return {
            "success": implementation_result.get("success", False) and integration_result.get("success", False),
            "project_id": project_id,
            "planning_result": planning_result,
            "implementation_result": implementation_result,
            "integration_result": integration_result,
            "summary": f"Feature '{feature_description}' implemented successfully"
        }
    
    async def _deploy_project(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a project to a specified platform.
        
        Args:
            request: Dictionary containing deployment details
            
        Returns:
            Dictionary with deployment results
        """
        project_id = request.get("project_id", "")
        platform = request.get("platform", "")
        
        if not project_id or project_id not in self.projects:
            return {
                "success": False,
                "error": "Invalid project ID"
            }
        
        if not platform:
            return {
                "success": False,
                "error": "Deployment platform is required"
            }
        
        project_context = self.projects[project_id]
        
        # Use the deployment agent to deploy the project
        deployment_input = {
            "action": "deploy_project",
            "project_name": project_context.project_name,
            "workspace_path": project_context.workspace_path,
            "platform": platform,
            "config": request.get("config", {})
        }
        
        deployment_result = await self.agents["deployment"].process(deployment_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="deployment",
            action="deploy_project",
            input_data=deployment_input,
            output_data=deployment_result,
            success=deployment_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        # Update project context
        self.projects[project_id].updated_at = datetime.now()
        
        # Add deployment information to project metadata
        if "deployments" not in self.projects[project_id].metadata:
            self.projects[project_id].metadata["deployments"] = []
        
        if deployment_result.get("success", False):
            self.projects[project_id].metadata["deployments"].append({
                "platform": platform,
                "url": deployment_result.get("url", ""),
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": deployment_result.get("success", False),
            "project_id": project_id,
            "deployment_result": deployment_result,
            "summary": f"Project deployment to {platform} {'succeeded' if deployment_result.get('success', False) else 'failed'}"
        }
    
    async def _answer_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a knowledge query.
        
        Args:
            request: Dictionary containing the query
            
        Returns:
            Dictionary with the answer
        """
        query = request.get("query", "")
        project_id = request.get("project_id", "")
        
        if not query:
            return {
                "success": False,
                "error": "Query is required"
            }
        
        # Prepare context for the knowledge agent
        context = {}
        
        if project_id and project_id in self.projects:
            project_context = self.projects[project_id]
            context = {
                "project_name": project_context.project_name,
                "workspace_path": project_context.workspace_path,
                "requirements": project_context.requirements,
                "technologies": project_context.technologies
            }
        
        # Use the knowledge agent to answer the query
        knowledge_input = {
            "action": "answer_query",
            "query": query,
            "context": context
        }
        
        knowledge_result = await self.agents["knowledge"].process(knowledge_input)
        
        # Record task result
        task_result = TaskResult(
            agent_type="knowledge",
            action="answer_query",
            input_data=knowledge_input,
            output_data=knowledge_result,
            success=knowledge_result.get("success", False)
        )
        
        self.task_results.append(task_result)
        
        return {
            "success": knowledge_result.get("success", False),
            "query": query,
            "response": knowledge_result.get("response", ""),
            "sources": knowledge_result.get("sources", []),
            "summary": f"Query answered {'successfully' if knowledge_result.get('success', False) else 'with errors'}"
        }
    
    async def _setup_repository(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up a code repository for a project.
        
        Args:
            request: Dictionary containing repository setup details
            
        Returns:
            Dictionary with repository setup results
        """
        project_id = request.get("project_id", "")
        repository_name = request.get("repository_name", "")
        provider = request.get("provider", "github")
        
        if not project_id or project_id not in self.projects:
            return {
                "success": False,
                "error": "Invalid project ID"
            }
        
        project_context = self.projects[project_id]
        
        # Use default repository name if not provided
        if not repository_name:
            repository_name = project_context.project_name
        
        # Use the web automation agent to set up the repository
        if provider == "github":
            setup_input = {
                "action": "github_setup",
                "repo_name": repository_name,
                "description": project_context.description,
                "private": request.get("private", False)
            }
            
            setup_result = await self.agents["web_automation"].process(setup_input)
            
            # Record task result
            task_result = TaskResult(
                agent_type="web_automation",
                action="github_setup",
                input_data=setup_input,
                output_data=setup_result,
                success=setup_result.get("success", False)
            )
            
            self.task_results.append(task_result)
            
            # Update project context
            self.projects[project_id].updated_at = datetime.now()
            
            # Add repository information to project metadata
            if setup_result.get("success", False):
                self.projects[project_id].metadata["repository"] = {
                    "provider": provider,
                    "name": repository_name,
                    "url": setup_result.get("results", {}).get("repo_url", ""),
                    "created_at": datetime.now().isoformat()
                }
            
            return {
                "success": setup_result.get("success", False),
                "project_id": project_id,
                "repository_result": setup_result,
                "summary": f"Repository setup for {repository_name} {'succeeded' if setup_result.get('success', False) else 'failed'}"
            }
        else:
            return {
                "success": False,
                "error": f"Repository provider {provider} not supported"
            } 