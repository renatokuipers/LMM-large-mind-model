# TODO: Implement the Planning class to develop and execute plans for goal achievement
# This component should be able to:
# - Create sequences of actions to achieve goals
# - Anticipate obstacles and develop contingency plans
# - Monitor plan execution and adjust as needed
# - Coordinate with other cognitive modules during plan execution

# TODO: Implement developmental progression in planning abilities:
# - Simple one-step plans in early stages
# - Short sequential plans in childhood
# - Complex hierarchical planning in adolescence
# - Strategic, flexible planning in adulthood

# TODO: Create mechanisms for:
# - Goal representation: Maintain clear goal states
# - Action sequencing: Order actions appropriately
# - Temporal projection: Anticipate future states
# - Error detection: Identify deviations from the plan

# TODO: Implement different planning approaches:
# - Forward planning: Plan from current state to goal
# - Backward planning: Plan from goal to current state
# - Hierarchical planning: Break complex goals into subgoals
# - Opportunistic planning: Flexibly adapt plans to changing conditions

# TODO: Connect to working memory and attention systems
# Planning requires working memory resources to maintain plans
# and attention to monitor execution

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.executive.models import Plan, PlanStep, ExecutiveNeuralState
from lmm_project.modules.executive.neural_net import PlanningNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class Planning(BaseModule):
    """
    Develops and executes plans to achieve goals
    
    This module creates sequences of actions to reach goal states,
    monitors plan execution, and adapts plans as needed.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Single step plans",
        0.2: "Multi-step sequential plans",
        0.4: "Conditional branching in plans",
        0.6: "Hierarchical planning",
        0.8: "Parallel action planning",
        1.0: "Strategic planning with contingencies"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the planning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="planning", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device
        self.device = get_device()
        
        # Initialize neural network
        self.planning_network = PlanningNetwork(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        ).to(self.device)
        
        # Set development level for network
        self.planning_network.set_development_level(development_level)
        
        # Create neural state for tracking
        self.neural_state = ExecutiveNeuralState()
        self.neural_state.planning_development = development_level
        
        # Active plans
        self.active_plans = {}
        
        # Plan history
        self.plan_history = deque(maxlen=20)
        
        # Planning parameters
        self.params = {
            "max_steps": 3,  # Maximum number of steps in a plan - increases with development
            "plan_horizon": 1,  # How many steps ahead to plan - increases with development
            "success_threshold": 0.6,  # Threshold for considering a plan successful
            "revision_threshold": 0.4,  # Threshold for when to revise a plan
            "hierarchical_planning": False,  # Whether hierarchical planning is enabled
            "parallel_actions": False  # Whether parallel actions are allowed
        }
        
        # Update parameters based on development
        self._adjust_parameters_for_development()
        
        logger.info(f"Planning module initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust planning parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very simple planning at early stages
            self.params.update({
                "max_steps": max(1, int(2 * self.development_level) + 1),
                "plan_horizon": 1,
                "success_threshold": 0.7,  # Higher threshold (more conservative)
                "revision_threshold": 0.5,  # More likely to revise plans
                "hierarchical_planning": False,
                "parallel_actions": False
            })
        elif self.development_level < 0.4:
            # More steps but still simple planning
            self.params.update({
                "max_steps": 3,
                "plan_horizon": 2,
                "success_threshold": 0.65, 
                "revision_threshold": 0.45,
                "hierarchical_planning": False,
                "parallel_actions": False
            })
        elif self.development_level < 0.6:
            # Introduction of conditionals in plans
            self.params.update({
                "max_steps": 5,
                "plan_horizon": 3,
                "success_threshold": 0.6,
                "revision_threshold": 0.4,
                "hierarchical_planning": False,
                "parallel_actions": False
            })
        elif self.development_level < 0.8:
            # Hierarchical planning enabled
            self.params.update({
                "max_steps": 7,
                "plan_horizon": 4,
                "success_threshold": 0.55,
                "revision_threshold": 0.35,
                "hierarchical_planning": True,
                "parallel_actions": False
            })
        else:
            # Full planning capabilities
            self.params.update({
                "max_steps": 10,
                "plan_horizon": 5,
                "success_threshold": 0.5,
                "revision_threshold": 0.3,
                "hierarchical_planning": True,
                "parallel_actions": True
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to create or update plans
        
        Args:
            input_data: Dictionary containing goal and state information
                Required keys depend on operation:
                - For 'create': 'goal', 'initial_state'
                - For 'execute': 'plan_id'
                - For 'update': 'plan_id', 'step_id', 'status', 'progress'
                - For 'revise': 'plan_id', 'current_state'
                - For 'query': 'plan_id' or 'all' flag
            
        Returns:
            Dictionary with the results of planning
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        operation = input_data.get("operation", "create")
        
        # Different operations based on the request
        if operation == "create":
            return self._create_plan(input_data, process_id)
        elif operation == "execute":
            return self._execute_plan(input_data, process_id)
        elif operation == "update":
            return self._update_plan(input_data, process_id)
        elif operation == "revise":
            return self._revise_plan(input_data, process_id)
        elif operation == "query":
            return self._query_plans(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _create_plan(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Create a new plan"""
        # Extract required data
        if "goal" not in input_data:
            return {"status": "error", "message": "No goal provided", "process_id": process_id}
        
        goal = input_data.get("goal", "")
        description = input_data.get("description", f"Plan for: {goal}")
        initial_state = input_data.get("initial_state", {})
        constraints = input_data.get("constraints", [])
        
        # Convert goal and state to tensors for neural processing
        goal_features = self._extract_features(goal)
        state_features = self._extract_features(initial_state)
        
        # Process through neural network
        with torch.no_grad():
            planning_result = self.planning_network(
                goal=goal_features.to(self.device),
                state=state_features.to(self.device)
            )
        
        # Record activation in neural state
        self.neural_state.add_activation('planning', {
            'goal': goal,
            'plan_features': planning_result["plan_features"].cpu().numpy().tolist(),
            'success_probability': planning_result["success_probability"].cpu().item()
        })
        
        # Generate plan steps based on development level
        num_steps = min(
            self.params["max_steps"],
            max(1, int(3 * self.development_level) + 1)
        )
        
        plan_steps = []
        for i in range(num_steps):
            # Create more detailed steps at higher development levels
            if self.development_level < 0.3:
                # Very simple steps at early stages
                action = f"Perform action {i+1} to achieve goal"
                outcome = "Goal progress"
            else:
                # More specific steps at higher levels
                action = self._generate_action_for_step(goal, initial_state, i, num_steps)
                outcome = self._generate_outcome_for_step(goal, action, i, num_steps)
            
            # Create the step
            step = PlanStep(
                action=action,
                description=f"Step {i+1}: {action}",
                expected_outcome=outcome,
                prerequisites=[str(j) for j in range(i) if j < i],  # Prior steps are prerequisites
                estimated_difficulty=0.5  # Default difficulty
            )
            plan_steps.append(step)
        
        # Create the plan
        plan = Plan(
            goal=goal,
            description=description,
            steps=plan_steps,
            success_likelihood=planning_result["success_probability"].cpu().item(),
            completion_percentage=0.0
        )
        
        # Store the plan
        self.active_plans[plan.plan_id] = plan
        self.plan_history.append(plan.plan_id)
        
        # Return the created plan
        return {
            "status": "success",
            "plan_id": plan.plan_id,
            "plan": plan.dict(),
            "process_id": process_id
        }
    
    def _execute_plan(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Execute a plan or advance to next step"""
        # Extract plan ID
        plan_id = input_data.get("plan_id")
        if not plan_id or plan_id not in self.active_plans:
            return {"status": "error", "message": "Invalid plan ID", "process_id": process_id}
        
        plan = self.active_plans[plan_id]
        
        # Start execution or advance to next step
        if plan.current_step_index is None:
            # Starting execution
            plan.current_step_index = 0
            plan.status = "in_progress"
            current_step = plan.steps[0]
            current_step.status = "in_progress"
            
            plan.updated_at = datetime.now()
        else:
            # Advance to next step if current is complete
            current_idx = plan.current_step_index
            
            if current_idx >= len(plan.steps):
                return {
                    "status": "complete",
                    "message": "Plan already completed",
                    "plan": plan.dict(),
                    "process_id": process_id
                }
            
            current_step = plan.steps[current_idx]
            
            if current_step.status == "completed":
                # Move to next step
                if current_idx + 1 < len(plan.steps):
                    plan.current_step_index = current_idx + 1
                    next_step = plan.steps[plan.current_step_index]
                    next_step.status = "in_progress"
                    
                    plan.updated_at = datetime.now()
                else:
                    # Plan complete
                    plan.status = "completed"
                    plan.completion_percentage = 1.0
                    
                    plan.updated_at = datetime.now()
                    
                    return {
                        "status": "complete",
                        "message": "Plan execution completed",
                        "plan": plan.dict(),
                        "process_id": process_id
                    }
            
            # Get current step again after possible advancement
            current_step = plan.steps[plan.current_step_index]
        
        # Calculate plan progress
        completed_steps = sum(1 for step in plan.steps if step.status == "completed")
        plan.completion_percentage = completed_steps / len(plan.steps)
        
        # Return current step information
        return {
            "status": "in_progress",
            "current_step_index": plan.current_step_index,
            "current_step": plan.steps[plan.current_step_index].dict(),
            "completion_percentage": plan.completion_percentage,
            "plan": plan.dict(),
            "process_id": process_id
        }
    
    def _update_plan(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Update a plan or step status"""
        # Extract plan and step information
        plan_id = input_data.get("plan_id")
        if not plan_id or plan_id not in self.active_plans:
            return {"status": "error", "message": "Invalid plan ID", "process_id": process_id}
        
        plan = self.active_plans[plan_id]
        
        # Check if updating plan status or step status
        if "step_id" in input_data:
            # Updating a specific step
            step_id = input_data["step_id"]
            step = next((s for s in plan.steps if s.step_id == step_id), None)
            
            if not step:
                return {"status": "error", "message": "Invalid step ID", "process_id": process_id}
            
            # Update step status if provided
            if "status" in input_data:
                step.status = input_data["status"]
            
            # Update completion percentage if provided
            if "completion_percentage" in input_data:
                step.completion_percentage = input_data["completion_percentage"]
            
            step.updated_at = datetime.now()
            
            # Update plan completion percentage
            completed_steps = sum(1 for s in plan.steps if s.status == "completed")
            in_progress_steps = sum(s.completion_percentage for s in plan.steps if s.status == "in_progress")
            
            plan.completion_percentage = (completed_steps + in_progress_steps / len(plan.steps)) / len(plan.steps)
            
        else:
            # Updating the plan as a whole
            if "status" in input_data:
                plan.status = input_data["status"]
            
            if "completion_percentage" in input_data:
                plan.completion_percentage = input_data["completion_percentage"]
        
        plan.updated_at = datetime.now()
        
        # Return updated plan
        return {
            "status": "success",
            "plan": plan.dict(),
            "process_id": process_id
        }
    
    def _revise_plan(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Revise a plan based on new information"""
        # Extract plan information
        plan_id = input_data.get("plan_id")
        if not plan_id or plan_id not in self.active_plans:
            return {"status": "error", "message": "Invalid plan ID", "process_id": process_id}
        
        plan = self.active_plans[plan_id]
        current_state = input_data.get("current_state", {})
        
        # Get the current step
        current_step_idx = plan.current_step_index
        if current_step_idx is None or current_step_idx >= len(plan.steps):
            return {"status": "error", "message": "No current step to revise", "process_id": process_id}
        
        # Convert goal and state to tensors for neural processing
        goal_features = self._extract_features(plan.goal)
        state_features = self._extract_features(current_state)
        
        # Process through neural network to evaluate plan
        with torch.no_grad():
            planning_result = self.planning_network(
                goal=goal_features.to(self.device),
                state=state_features.to(self.device)
            )
        
        # Record activation
        self.neural_state.add_activation('planning', {
            'operation': 'revise',
            'goal': plan.goal,
            'revision_needed': planning_result["revision_needed"].cpu().item()
        })
        
        # Determine if revision is needed
        revision_needed = planning_result["revision_needed"].cpu().item() > self.params["revision_threshold"]
        
        if not revision_needed:
            return {
                "status": "success",
                "message": "No revision needed",
                "plan": plan.dict(),
                "process_id": process_id
            }
        
        # Create revised steps
        # Keep completed steps and revise remaining ones
        completed_steps = [step for step in plan.steps[:current_step_idx] if step.status == "completed"]
        
        # Generate new steps based on current state
        num_new_steps = min(
            self.params["max_steps"] - len(completed_steps),
            max(1, int(3 * self.development_level) + 1)
        )
        
        new_steps = []
        for i in range(num_new_steps):
            # Create more detailed steps at higher development levels
            action = self._generate_action_for_step(plan.goal, current_state, i, num_new_steps)
            outcome = self._generate_outcome_for_step(plan.goal, action, i, num_new_steps)
            
            # Create the step
            step = PlanStep(
                action=action,
                description=f"Revised Step {len(completed_steps) + i + 1}: {action}",
                expected_outcome=outcome,
                prerequisites=[s.step_id for s in completed_steps]  # Completed steps are prerequisites
            )
            new_steps.append(step)
        
        # Update the plan
        plan.steps = completed_steps + new_steps
        plan.current_step_index = len(completed_steps)
        plan.updated_at = datetime.now()
        
        # If any steps remain, next one should be in progress
        if plan.current_step_index < len(plan.steps):
            plan.steps[plan.current_step_index].status = "in_progress"
        
        # Update completion percentage
        completed_count = len(completed_steps)
        plan.completion_percentage = completed_count / len(plan.steps) if plan.steps else 0.0
        
        return {
            "status": "revised",
            "message": "Plan revised",
            "plan": plan.dict(),
            "process_id": process_id
        }
    
    def _query_plans(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Query plan information"""
        # Check if querying specific plan or all plans
        if "plan_id" in input_data:
            plan_id = input_data["plan_id"]
            if plan_id not in self.active_plans:
                return {"status": "error", "message": "Invalid plan ID", "process_id": process_id}
            
            return {
                "status": "success",
                "plan": self.active_plans[plan_id].dict(),
                "process_id": process_id
            }
        else:
            # Return all active plans
            return {
                "status": "success",
                "active_plans": {pid: plan.dict() for pid, plan in self.active_plans.items()},
                "plan_count": len(self.active_plans),
                "process_id": process_id
            }
    
    def _extract_features(self, data) -> torch.Tensor:
        """
        Extract features from input data for neural processing
        
        Args:
            data: Text, dict, or other data to extract features from
            
        Returns:
            Tensor of features [1, feature_dim]
        """
        # For demonstration, create simple random features
        # In a real implementation, this would use proper feature extraction
        feature_dim = 64
        
        if isinstance(data, str):
            # Seed random generator with hash of string to ensure consistent features
            seed = hash(data) % 10000
            np.random.seed(seed)
            
            # Generate "features" based on the text
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        elif isinstance(data, dict):
            # For dictionary data, use keys and values to generate features
            seed = hash(str(sorted(data.items()))) % 10000
            np.random.seed(seed)
            
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        else:
            # Default random features
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _generate_action_for_step(self, goal: str, state: Dict[str, Any], step_idx: int, total_steps: int) -> str:
        """Generate appropriate action text for a plan step"""
        # Simple action generation based on development level
        if self.development_level < 0.3:
            return f"Perform step {step_idx + 1} toward {goal}"
            
        elif self.development_level < 0.6:
            progress = (step_idx + 1) / total_steps
            if progress < 0.4:
                return f"Begin {goal} by taking initial actions"
            elif progress < 0.8:
                return f"Continue making progress toward {goal}"
            else:
                return f"Finalize actions to complete {goal}"
                
        else:
            # More specific actions at higher development levels
            progress = (step_idx + 1) / total_steps
            
            if progress < 0.3:
                return f"Initialize approach for {goal} by establishing prerequisites"
            elif progress < 0.6:
                return f"Implement core methods to achieve {goal} using appropriate techniques"
            elif progress < 0.9:
                return f"Integrate components and verify progress toward {goal}"
            else:
                return f"Finalize and validate completion of {goal}"
    
    def _generate_outcome_for_step(self, goal: str, action: str, step_idx: int, total_steps: int) -> str:
        """Generate expected outcome text for a plan step"""
        # Simple outcome generation based on development level
        progress = (step_idx + 1) / total_steps
        
        if self.development_level < 0.3:
            return f"Progress toward {goal}"
            
        elif self.development_level < 0.6:
            if progress < 0.4:
                return f"Initial progress established"
            elif progress < 0.8:
                return f"Substantial progress made toward {goal}"
            else:
                return f"Goal nearly achieved"
                
        else:
            # More specific outcomes at higher development levels
            if progress < 0.3:
                return f"Foundations established for achieving {goal}"
            elif progress < 0.6:
                return f"Core components implemented, {int(progress*100)}% progress toward {goal}"
            elif progress < 0.9:
                return f"Major milestones achieved, final steps for {goal} identified"
            else:
                return f"All requirements for {goal} completion in place"
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update network development level
        self.planning_network.set_development_level(new_level)
        
        # Update neural state
        self.neural_state.planning_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # Adjust parameters based on new development level
        self._adjust_parameters_for_development()
        
        logger.info(f"Planning module development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Get base state from parent
        base_state = super().get_state()
        
        # Add planning-specific state
        planning_state = {
            "params": self.params,
            "active_plans": {pid: plan.dict() for pid, plan in self.active_plans.items()},
            "plan_count": len(self.active_plans),
            "recent_plan_ids": list(self.plan_history)
        }
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.planning_development,
            "accuracy": self.neural_state.planning_accuracy,
            "recent_activations_count": len(self.neural_state.recent_planning_activations)
        }
        
        # Combine states
        combined_state = {**base_state, **planning_state, **neural_state}
        
        return combined_state
