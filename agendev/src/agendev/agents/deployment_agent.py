# deployment_agent.py
"""Deployment agent for deploying the project to various platforms."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import asyncio
import os
import json
import subprocess
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

from .agent_base import Agent, AgentStatus
from ..utils.fs_utils import resolve_path


class DeploymentConfig(BaseModel):
    """Configuration for a deployment."""
    platform: str
    config: Dict[str, Any] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    scripts: List[str] = Field(default_factory=list)


class DeploymentResult(BaseModel):
    """Result of a deployment operation."""
    success: bool
    platform: str
    url: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class DeploymentAgent(Agent):
    """
    Deployment agent responsible for deploying the project to various platforms.
    
    This agent handles the deployment of the project to platforms like GitHub Pages,
    Vercel, Netlify, etc.
    """
    
    def __init__(
        self,
        name: str = "Deployment",
        description: str = "Deploys projects to various platforms",
        agent_type: str = "deployment",
        workspace_dir: Optional[str] = None
    ):
        """Initialize the DeploymentAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Workspace for deployment operations - using private variables
        self._workspace_dir = workspace_dir or "workspace/src"
        
        # Track deployment results
        self._deployment_results: List[DeploymentResult] = []
        
        # Store available platforms
        self._platforms: List[str] = [
            "local",
            "github-pages",
            "vercel",
            "netlify",
            "heroku",
            "aws-amplify"
        ]
    
    # Define properties for safer access
    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir
    
    @workspace_dir.setter
    def workspace_dir(self, value: str):
        self._workspace_dir = value
    
    @property
    def deployment_results(self) -> List[DeploymentResult]:
        return self._deployment_results
    
    @deployment_results.setter
    def deployment_results(self, value: List[DeploymentResult]):
        self._deployment_results = value
    
    @property
    def platforms(self) -> List[str]:
        return self._platforms
    
    @platforms.setter
    def platforms(self, value: List[str]):
        self._platforms = value
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process deployment-related requests.
        
        Args:
            input_data: Input data for the deployment request
            
        Returns:
            Dictionary with the deployment results
        """
        # Check if LLM integration is available
        if not self.llm:
            raise ValueError("LLM integration not available")
        
        # Extract the action
        action = input_data.get("action", "")
        
        if action == "deploy_project":
            platforms = input_data.get("platforms", ["github_pages"])
            return await self._deploy_project(platforms)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _deploy_project(self, platforms: List[str]) -> Dict[str, Any]:
        """
        Deploy the project to specified platforms.
        
        Args:
            platforms: List of platforms to deploy to
            
        Returns:
            Dictionary with deployment results
        """
        try:
            # Create workspace path if it doesn't exist
            workspace_path = resolve_path(self.workspace_dir, create_parents=True)
            
            # Create deployment configurations
            deployment_configs = await self._create_deployment_configs(platforms)
            
            # Execute deployments
            results = []
            for config in deployment_configs:
                result = await self._execute_deployment(config)
                results.append(result)
                self.deployment_results.append(result)
            
            # Create deployment summary
            successful_deployments = [r for r in results if r.success]
            failed_deployments = [r for r in results if not r.success]
            
            summary = (
                f"Deployed to {len(successful_deployments)}/{len(results)} platforms successfully.\n\n"
            )
            
            # Add URLs for successful deployments
            if successful_deployments:
                summary += "Deployment URLs:\n"
                for result in successful_deployments:
                    if result.url:
                        summary += f"- {result.platform}: {result.url}\n"
            
            # Add errors for failed deployments
            if failed_deployments:
                summary += "\nFailed deployments:\n"
                for result in failed_deployments:
                    summary += f"- {result.platform}: {', '.join(result.errors[:1])}\n"
            
            return {
                "success": len(successful_deployments) > 0,
                "results": [r.dict() for r in results],
                "summary": summary,
                "url": successful_deployments[0].url if successful_deployments and successful_deployments[0].url else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Deployment failed: {str(e)}"
            }
    
    async def _create_deployment_configs(self, platforms: List[str]) -> List[DeploymentConfig]:
        """
        Create deployment configurations for specified platforms.
        
        Args:
            platforms: List of platforms to create configurations for
            
        Returns:
            List of deployment configurations
        """
        # Define the schema for the LLM response
        schema = {
            "name": "deployment_configs",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "configs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "platform": {
                                    "type": "string",
                                    "description": "Deployment platform name"
                                },
                                "config": {
                                    "type": "object",
                                    "description": "Platform-specific configuration"
                                },
                                "environment_variables": {
                                    "type": "object",
                                    "description": "Environment variables for the deployment"
                                },
                                "scripts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Scripts to execute for deployment"
                                }
                            },
                            "required": ["platform", "scripts"]
                        }
                    }
                },
                "required": ["configs"]
            }
        }
        
        # Create workspace path if it doesn't exist
        workspace_path = resolve_path(self.workspace_dir, create_parents=True)
        
        # Scan workspace to determine project type
        project_info = self._analyze_project_structure(workspace_path)
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are a DevOps expert. Create deployment configurations for the following project.
        
        Project structure:
        {json.dumps(project_info, indent=2)}
        
        Target platforms: {', '.join(platforms)}
        
        For each platform, provide:
        1. Platform-specific configuration
        2. Environment variables needed for deployment
        3. Exact deployment scripts to execute
        
        Focus on creating working deployment scripts that follow best practices for each platform.
        For GitHub Pages, generate the correct commands for pushing to gh-pages branch.
        For Vercel or Netlify, generate the appropriate configuration files and deployment commands.
        
        If the project is a React or Next.js app, include build steps before deployment.
        If it's a Node.js backend, include steps to prepare the application for production.
        
        Use platform-specific deployment tools and CLIs where appropriate.
        """
        
        try:
            # Call the LLM with the prompt
            response = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            # Extract configs
            configs_data = response.get("configs", [])
            
            # Create DeploymentConfig objects
            return [
                DeploymentConfig(
                    platform=config_data.get("platform", "unknown"),
                    config=config_data.get("config", {}),
                    environment_variables=config_data.get("environment_variables", {}),
                    scripts=config_data.get("scripts", [])
                )
                for config_data in configs_data
            ]
            
        except Exception as e:
            # Default configurations if LLM fails
            return [self._create_default_config(platform) for platform in platforms]
    
    def _analyze_project_structure(self, workspace_path: Path) -> Dict[str, Any]:
        """
        Analyze the project structure to determine project type.
        
        Args:
            workspace_path: Path to the workspace directory
            
        Returns:
            Dictionary with project analysis
        """
        result = {
            "files": [],
            "has_package_json": False,
            "has_requirements_txt": False,
            "framework": "unknown",
            "language": "unknown"
        }
        
        # List root files
        root_files = []
        try:
            root_files = [f for f in os.listdir(workspace_path) if os.path.isfile(os.path.join(workspace_path, f))]
            result["files"] = root_files[:20]  # Limit to first 20 files
        except Exception:
            pass
        
        # Check for package.json
        if "package.json" in root_files:
            result["has_package_json"] = True
            try:
                with open(os.path.join(workspace_path, "package.json"), 'r') as f:
                    package_data = json.load(f)
                    dependencies = package_data.get("dependencies", {})
                    
                    # Detect framework
                    if "react" in dependencies:
                        result["framework"] = "react"
                        result["language"] = "javascript"
                    elif "next" in dependencies:
                        result["framework"] = "nextjs"
                        result["language"] = "javascript"
                    elif "express" in dependencies:
                        result["framework"] = "express"
                        result["language"] = "javascript"
                    elif "vue" in dependencies:
                        result["framework"] = "vue"
                        result["language"] = "javascript"
                    elif "angular" in dependencies:
                        result["framework"] = "angular"
                        result["language"] = "javascript"
                    else:
                        result["framework"] = "node"
                        result["language"] = "javascript"
            except Exception:
                pass
        
        # Check for requirements.txt
        if "requirements.txt" in root_files:
            result["has_requirements_txt"] = True
            result["language"] = "python"
            
            try:
                with open(os.path.join(workspace_path, "requirements.txt"), 'r') as f:
                    requirements = f.read()
                    
                    # Detect framework
                    if "flask" in requirements.lower():
                        result["framework"] = "flask"
                    elif "django" in requirements.lower():
                        result["framework"] = "django"
                    elif "fastapi" in requirements.lower():
                        result["framework"] = "fastapi"
            except Exception:
                pass
        
        return result
    
    def _create_default_config(self, platform: str) -> DeploymentConfig:
        """
        Create a default deployment configuration for a platform.
        
        Args:
            platform: Platform to create configuration for
            
        Returns:
            Default deployment configuration
        """
        if platform == "github_pages":
            return DeploymentConfig(
                platform="github_pages",
                config={},
                scripts=[
                    "git init",
                    "git add .",
                    "git commit -m \"Initial commit\"",
                    "git branch -M main",
                    "git checkout -b gh-pages",
                    "git push -u origin gh-pages"
                ]
            )
        elif platform == "vercel":
            return DeploymentConfig(
                platform="vercel",
                config={},
                scripts=[
                    "npm install -g vercel",
                    "vercel --prod"
                ]
            )
        elif platform == "netlify":
            return DeploymentConfig(
                platform="netlify",
                config={},
                scripts=[
                    "npm install -g netlify-cli",
                    "netlify deploy --prod"
                ]
            )
        else:
            return DeploymentConfig(
                platform=platform,
                scripts=[]
            )
    
    async def _execute_deployment(self, config: DeploymentConfig) -> DeploymentResult:
        """
        Execute a deployment using the provided configuration.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Deployment result
        """
        logs = []
        errors = []
        success = True
        url = None
        
        # Create workspace path
        workspace_path = resolve_path(self.workspace_dir, create_parents=True)
        
        # Set environment variables
        original_env = os.environ.copy()
        try:
            # Apply environment variables
            for key, value in config.environment_variables.items():
                os.environ[key] = value
            
            # Execute deployment scripts
            for script in config.scripts:
                logs.append(f"Executing: {script}")
                
                try:
                    process = subprocess.Popen(
                        script,
                        shell=True,
                        cwd=workspace_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Wait for the process to complete with a timeout
                    stdout, stderr = process.communicate(timeout=300)
                    
                    logs.append(stdout)
                    
                    if process.returncode != 0:
                        errors.append(stderr)
                        success = False
                        break
                    
                    # Extract URL from output if available
                    if config.platform == "github_pages" and "github.io" in stdout:
                        url_lines = [line for line in stdout.split("\n") if "github.io" in line]
                        if url_lines:
                            url = url_lines[0].strip()
                    elif config.platform == "vercel" and "https://" in stdout:
                        url_lines = [line for line in stdout.split("\n") if "https://" in line and "vercel" in line]
                        if url_lines:
                            url = url_lines[0].strip()
                    elif config.platform == "netlify" and "https://" in stdout:
                        url_lines = [line for line in stdout.split("\n") if "https://" in line and "netlify" in line]
                        if url_lines:
                            url = url_lines[0].strip()
                    
                except subprocess.TimeoutExpired:
                    # Kill the process if it times out
                    process.kill()
                    errors.append(f"Command timed out after 300 seconds: {script}")
                    success = False
                    break
                except Exception as e:
                    errors.append(f"Error executing command: {str(e)}")
                    success = False
                    break
            
            # If we don't have a URL but the deployment was successful, generate a placeholder
            if success and not url and config.platform == "github_pages":
                url = "https://your-username.github.io/repository-name"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
        
        # Create and return result
        return DeploymentResult(
            success=success,
            platform=config.platform,
            url=url,
            logs=logs,
            errors=errors
        ) 