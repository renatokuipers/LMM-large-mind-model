# executor_agent.py
"""Executor agent responsible for orchestrating the entire development workflow."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from .agent_base import Agent, AgentStatus, AgentAction
from ..models.task_models import Task, TaskGraph, TaskStatus
from ..tts_notification import NotificationManager


class WorkflowStage(str, Enum):
    """Stages in the development workflow."""
    INITIAL_ANALYSIS = "initial_analysis"
    TASK_BREAKDOWN = "task_breakdown"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"


class WorkflowStatus(BaseModel):
    """Status of the current workflow execution."""
    stage: WorkflowStage = WorkflowStage.INITIAL_ANALYSIS
    progress: float = 0.0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_task_id: Optional[UUID] = None
    errors: List[str] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class ExecutorAgent(Agent):
    """
    Executor agent that orchestrates the entire development workflow.
    
    This agent is responsible for coordinating the activities of specialized agents
    and ensuring the development flow progresses correctly.
    """
    
    def __init__(
        self,
        name: str = "Executor",
        description: str = "Orchestrates the entire development workflow",
        agent_type: str = "executor"
    ):
        """Initialize the ExecutorAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Workflow status
        self.workflow_status = WorkflowStatus()
        
        # Agent registry
        self.agents: Dict[str, Agent] = {}
        
        # Task graph for the project
        self.task_graph = TaskGraph()
        
        # Notification manager
        self.notification_manager: Optional[NotificationManager] = None
    
    def register_agent(self, agent_type: str, agent: Agent) -> None:
        """
        Register a specialized agent.
        
        Args:
            agent_type: Type identifier for the agent
            agent: Agent instance
        """
        self.agents[agent_type] = agent
        
        # Initialize the agent with shared resources
        if self.llm and not agent.llm:
            agent.initialize(self.llm, self.context_manager)
    
    def register_notification_manager(self, notification_manager: NotificationManager) -> None:
        """
        Register the notification manager.
        
        Args:
            notification_manager: Notification manager for voice and visual notifications
        """
        self.notification_manager = notification_manager
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the workflow status.
        
        Args:
            role: Role of the message sender (e.g., "system", "agent", "user")
            content: Message content
        """
        self.workflow_status.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Also send a notification if appropriate
        if self.notification_manager and role == "system":
            self.notification_manager.info(content)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a development request from start to finish.
        
        Args:
            input_data: Input data containing the development prompt
            
        Returns:
            Dictionary with the development results
        """
        # Reset workflow status
        self.workflow_status = WorkflowStatus()
        
        # Extract the development prompt
        prompt = input_data.get("prompt", "")
        if not prompt:
            error_msg = "No development prompt provided"
            self.workflow_status.errors.append(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "workflow_status": self.workflow_status.dict()
            }
        
        # Add the initial system message
        self.add_message("system", f"Starting new development project based on: '{prompt}'")
        
        try:
            # Step 1: Initial Analysis
            await self._execute_initial_analysis(prompt)
            
            # Step 2: Task Breakdown
            await self._execute_task_breakdown()
            
            # Step 3: Planning
            await self._execute_planning()
            
            # Step 4: Implementation
            await self._execute_implementation()
            
            # Step 5: Testing
            await self._execute_testing()
            
            # Step 6: Deployment
            await self._execute_deployment()
            
            # Complete workflow
            self.workflow_status.stage = WorkflowStage.COMPLETED
            self.workflow_status.progress = 1.0
            self.workflow_status.end_time = datetime.now()
            
            self.add_message("system", "Development project completed successfully")
            
            return {
                "success": True,
                "workflow_status": self.workflow_status.dict(),
                "task_graph": self.task_graph.dict()
            }
            
        except Exception as e:
            error_msg = f"Error during {self.workflow_status.stage}: {str(e)}"
            self.workflow_status.errors.append(error_msg)
            
            self.add_message("system", f"Error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "workflow_status": self.workflow_status.dict()
            }
    
    async def _execute_initial_analysis(self, prompt: str) -> None:
        """
        Execute the initial analysis stage.
        
        Args:
            prompt: Development prompt to analyze
        """
        self.workflow_status.stage = WorkflowStage.INITIAL_ANALYSIS
        self.add_message("system", "Analyzing development requirements...")
        
        # Check if we have a planner agent
        if "planner" not in self.agents:
            raise ValueError("Planner agent not registered")
        
        # Create an action for the analysis
        action = self.create_action(
            action_type="initial_analysis",
            description="Analyze development prompt",
            parameters={"prompt": prompt}
        )
        
        # Execute the analysis using the planner agent
        planner_agent = self.agents["planner"]
        analysis_result = await planner_agent.process({
            "action": "analyze_prompt",
            "prompt": prompt
        })
        
        # Complete the action
        self.complete_action(action, analysis_result)
        
        # Update progress
        self.workflow_status.progress = 0.1
        
        # Add the analysis result as a message
        analysis_summary = analysis_result.get("summary", "No analysis summary available")
        self.add_message("agent", f"Analysis: {analysis_summary}")
        
        # Move to next stage
        self.add_message("system", "Initial analysis completed")
    
    async def _execute_task_breakdown(self) -> None:
        """Execute the task breakdown stage."""
        self.workflow_status.stage = WorkflowStage.TASK_BREAKDOWN
        self.add_message("system", "Breaking down development into tasks...")
        
        # Check if we have a planner agent
        if "planner" not in self.agents:
            raise ValueError("Planner agent not registered")
        
        # Create an action for the task breakdown
        action = self.create_action(
            action_type="task_breakdown",
            description="Break down project into tasks",
            parameters={}
        )
        
        # Execute the task breakdown using the planner agent
        planner_agent = self.agents["planner"]
        breakdown_result = await planner_agent.process({
            "action": "break_down_tasks"
        })
        
        # Complete the action
        self.complete_action(action, breakdown_result)
        
        # Update the task graph
        if "task_graph" in breakdown_result:
            self.task_graph = breakdown_result["task_graph"]
        
        # Update progress
        self.workflow_status.progress = 0.2
        
        # Add task summary as a message
        tasks = breakdown_result.get("tasks", [])
        task_summary = "\n".join([f"- {task.get('title')}: {task.get('description')}" for task in tasks[:5]])
        if len(tasks) > 5:
            task_summary += f"\n- ... and {len(tasks) - 5} more tasks"
        
        self.add_message("agent", f"Task Breakdown:\n{task_summary}")
        
        # Move to next stage
        self.add_message("system", "Task breakdown completed")
    
    async def _execute_planning(self) -> None:
        """Execute the planning stage."""
        self.workflow_status.stage = WorkflowStage.PLANNING
        self.add_message("system", "Planning implementation sequence...")
        
        # Check if we have a planner agent
        if "planner" not in self.agents:
            raise ValueError("Planner agent not registered")
        
        # Create an action for planning
        action = self.create_action(
            action_type="planning",
            description="Generate implementation plan",
            parameters={}
        )
        
        # Execute planning using the planner agent
        planner_agent = self.agents["planner"]
        planning_result = await planner_agent.process({
            "action": "generate_plan",
            "task_graph": self.task_graph
        })
        
        # Complete the action
        self.complete_action(action, planning_result)
        
        # Update progress
        self.workflow_status.progress = 0.3
        
        # Add plan summary as a message
        plan_summary = planning_result.get("summary", "No plan summary available")
        self.add_message("agent", f"Implementation Plan: {plan_summary}")
        
        # Move to next stage
        self.add_message("system", "Planning completed")
    
    async def _execute_implementation(self) -> None:
        """Execute the implementation stage."""
        self.workflow_status.stage = WorkflowStage.IMPLEMENTATION
        self.add_message("system", "Starting implementation phase...")
        
        # Check if we have a code agent
        if "code" not in self.agents:
            raise ValueError("Code agent not registered")
        
        # Get the implementation sequence
        task_sequence = self._get_implementation_sequence()
        
        # Execute each task in sequence
        for i, task_id in enumerate(task_sequence):
            # Get the task
            task = self.task_graph.tasks.get(task_id)
            if not task:
                continue
                
            # Update current task
            self.workflow_status.current_task_id = task_id
            
            # Create an action for implementing this task
            action = self.create_action(
                action_type="implement_task",
                description=f"Implement task: {task.title}",
                parameters={"task_id": str(task_id)}
            )
            
            # Calculate and update progress
            progress_increment = 0.5 / len(task_sequence)
            self.workflow_status.progress = 0.3 + (i * progress_increment)
            
            # Notify about current task
            self.add_message("system", f"Implementing task {i+1}/{len(task_sequence)}: {task.title}")
            
            # Execute the task using the code agent
            code_agent = self.agents["code"]
            implementation_result = await code_agent.process({
                "action": "implement_task",
                "task": task
            })
            
            # Complete the action
            self.complete_action(action, implementation_result)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completion_percentage = 100.0
            
            # Add implementation result as a message
            result_summary = implementation_result.get("summary", "Task implemented")
            self.add_message("agent", f"Implemented '{task.title}': {result_summary}")
            
            # Connect with integration agent if needed
            if "integration" in self.agents and implementation_result.get("needs_integration", False):
                await self._perform_integration(task_id, implementation_result)
        
        # Update progress
        self.workflow_status.progress = 0.8
        
        # Move to next stage
        self.add_message("system", "Implementation phase completed")
    
    async def _perform_integration(self, task_id: UUID, implementation_result: Dict[str, Any]) -> None:
        """
        Perform integration for the implemented task.
        
        Args:
            task_id: ID of the task that was implemented
            implementation_result: Result of the implementation
        """
        # Check if we have an integration agent
        if "integration" not in self.agents:
            return
            
        # Get the task
        task = self.task_graph.tasks.get(task_id)
        if not task:
            return
            
        # Create an action for integration
        action = self.create_action(
            action_type="integrate_task",
            description=f"Integrate task: {task.title}",
            parameters={"task_id": str(task_id)}
        )
        
        # Notify about integration
        self.add_message("system", f"Integrating: {task.title}")
        
        # Execute integration using the integration agent
        integration_agent = self.agents["integration"]
        integration_result = await integration_agent.process({
            "action": "integrate_task",
            "task": task,
            "implementation_result": implementation_result
        })
        
        # Complete the action
        self.complete_action(action, integration_result)
        
        # Add integration result as a message
        result_summary = integration_result.get("summary", "Task integrated")
        self.add_message("agent", f"Integrated '{task.title}': {result_summary}")
    
    async def _execute_testing(self) -> None:
        """Execute the testing stage."""
        self.workflow_status.stage = WorkflowStage.TESTING
        self.add_message("system", "Starting testing phase...")
        
        # Check if we have a code agent
        if "code" not in self.agents:
            raise ValueError("Code agent not registered")
        
        # Create an action for testing
        action = self.create_action(
            action_type="run_tests",
            description="Execute tests for the project",
            parameters={}
        )
        
        # Execute testing using the code agent
        code_agent = self.agents["code"]
        testing_result = await code_agent.process({
            "action": "run_tests"
        })
        
        # Complete the action
        self.complete_action(action, testing_result)
        
        # Update progress
        self.workflow_status.progress = 0.9
        
        # Add testing result as a message
        test_summary = testing_result.get("summary", "No test summary available")
        self.add_message("agent", f"Testing Results: {test_summary}")
        
        # Move to next stage
        self.add_message("system", "Testing phase completed")
    
    async def _execute_deployment(self) -> None:
        """Execute the deployment stage."""
        self.workflow_status.stage = WorkflowStage.DEPLOYMENT
        self.add_message("system", "Starting deployment phase...")
        
        # Check if we have a deployment agent
        if "deployment" not in self.agents:
            raise ValueError("Deployment agent not registered")
        
        # Create an action for deployment
        action = self.create_action(
            action_type="deploy",
            description="Deploy the project",
            parameters={}
        )
        
        # Execute deployment using the deployment agent
        deployment_agent = self.agents["deployment"]
        deployment_result = await deployment_agent.process({
            "action": "deploy_project"
        })
        
        # Complete the action
        self.complete_action(action, deployment_result)
        
        # Update progress
        self.workflow_status.progress = 1.0
        
        # Add deployment result as a message
        deployment_summary = deployment_result.get("summary", "No deployment summary available")
        deployment_url = deployment_result.get("url", "")
        
        message = f"Deployment Results: {deployment_summary}"
        if deployment_url:
            message += f"\nDeployed URL: {deployment_url}"
            
        self.add_message("agent", message)
        
        # Move to next stage
        self.add_message("system", "Deployment phase completed")
    
    def _get_implementation_sequence(self) -> List[UUID]:
        """
        Get the sequence of tasks to implement.
        
        Returns:
            List of task IDs in implementation order
        """
        # If we have a task sequence from planning, use that
        # Otherwise, create a topological sort of the task graph
        
        # Get all tasks that don't have dependencies
        tasks_with_no_deps = [
            task_id for task_id, task in self.task_graph.tasks.items()
            if not task.dependencies
        ]
        
        # Start with tasks that have no dependencies
        result = list(tasks_with_no_deps)
        
        # Track visited tasks
        visited = set(result)
        
        # Process remaining tasks based on dependencies
        remaining_tasks = set(self.task_graph.tasks.keys()) - visited
        
        while remaining_tasks:
            # Find tasks whose dependencies are all visited
            next_tasks = [
                task_id for task_id in remaining_tasks
                if all(dep_id in visited for dep_id in self.task_graph.tasks[task_id].dependencies)
            ]
            
            if not next_tasks:
                # If there are no tasks to add, we may have a cycle or incorrect task graph
                # Add the remaining tasks in arbitrary order
                result.extend(remaining_tasks)
                break
                
            # Add the next tasks to the result and mark as visited
            result.extend(next_tasks)
            visited.update(next_tasks)
            
            # Update remaining tasks
            remaining_tasks -= set(next_tasks)
        
        return result 