"""
Goal Setting Module

This module handles the creation, prioritization, and management of goals.
It enables the system to formulate objectives based on drives, needs, and values,
tracking progress toward achieving these goals and adjusting them as needed.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import os
import json
import numpy as np
import torch
from pathlib import Path
import heapq

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.motivation.models import Goal, MotivationNeuralState
from lmm_project.modules.motivation.neural_net import MotivationNeuralNetwork, get_device

class GoalSetting(BaseModule):
    """
    Handles the creation and management of goals
    
    This module creates goals based on drives, needs, and values,
    maintains goal hierarchies, tracks progress, and adjusts goals
    based on changing priorities.
    """
    
    # Developmental milestones for goal setting
    development_milestones = {
        0.1: "Basic approach/avoidance responses",
        0.2: "Simple immediate goals",
        0.3: "Goal persistence over short periods",
        0.4: "Sequential goal formation",
        0.5: "Simple goal hierarchies",
        0.6: "Medium-term planning",
        0.7: "Complex goal hierarchies",
        0.8: "Long-term goal formation",
        0.9: "Abstract goal concepts",
        1.0: "Self-directed goal systems"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the goal setting module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id, 
            module_type="goal_setting",
            event_bus=event_bus, 
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.motivation.goal_setting.{module_id}")
        
        # Active and archived goals
        self.active_goals: Dict[str, Goal] = {}
        self.archived_goals: Dict[str, Goal] = {}
        self.max_archived_goals = 100
        
        # Maximum number of concurrent active goals (increases with development)
        self.max_active_goals = self._calculate_max_active_goals()
        
        # Maximum goal planning horizon (in simulated time units)
        self.planning_horizon = self._calculate_planning_horizon()
        
        # Minimum development level for various goal types
        self.goal_type_thresholds = {
            "approach_immediate": 0.0,    # Simple approach goals
            "avoidance_immediate": 0.1,   # Simple avoidance goals
            "sequential": 0.3,            # Goals requiring sequences of actions
            "hierarchical": 0.5,          # Goals with subgoals
            "abstract": 0.7,              # Goals with abstract concepts
            "long_term": 0.8              # Goals far in the future
        }
        
        # Neural network for goal valuation and generation
        input_dim = 30  # State + need + drive representation dimensions
        hidden_dim = 48
        output_dim = 10  # Goal properties (importance, urgency, etc.)
        
        self.neural_network = MotivationNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            network_type="goal",
            learning_rate=0.01,
            development_level=development_level
        )
        
        # Goal generation cooldown (to prevent goal spam)
        self.last_goal_generation = datetime.now()
        self.goal_generation_cooldown = timedelta(seconds=10)
        
        # Subscribe to relevant messages
        if self.event_bus:
            self.subscribe_to_message("drive_update")
            self.subscribe_to_message("need_update")
            self.subscribe_to_message("action_completed")
            self.subscribe_to_message("goal_query")
            self.subscribe_to_message("create_goal")
            self.subscribe_to_message("update_goal")
            self.subscribe_to_message("abandon_goal")
            self.subscribe_to_message("perception_update")
        
        self.logger.info(f"Goal setting module initialized at development level {development_level:.2f}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to manage goals
        
        Args:
            input_data: Dictionary containing input data
                Required keys depend on the input type:
                - "type": Type of input ("create", "update", "query", etc.)
                - type-specific keys for each input type
                
        Returns:
            Dictionary with processed results
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input for goal setting")
        
        result = {
            "success": False,
            "message": ""
        }
        
        if input_type == "create":
            goal = self._create_goal(input_data)
            if goal:
                self.active_goals[goal.id] = goal
                result["success"] = True
                result["goal_id"] = goal.id
                result["message"] = f"Created goal: {goal.name}"
                
                # Publish goal creation event
                if self.event_bus:
                    self.publish_message("goal_created", {
                        "goal": goal.dict()
                    })
            else:
                result["message"] = "Failed to create goal"
                
        elif input_type == "update":
            goal_id = input_data.get("goal_id")
            if goal_id in self.active_goals:
                self._update_goal(goal_id, input_data)
                result["success"] = True
                result["message"] = f"Updated goal: {self.active_goals[goal_id].name}"
                
                # Publish goal update event if significant progress
                if input_data.get("progress_delta", 0) > 0.1 and self.event_bus:
                    self.publish_message("goal_progress", {
                        "goal": self.active_goals[goal_id].dict()
                    })
                    
                # Check if goal is now complete
                if self.active_goals[goal_id].progress >= 1.0:
                    self._complete_goal(goal_id)
            else:
                result["message"] = f"Goal {goal_id} not found"
                
        elif input_type == "query":
            query_type = input_data.get("query_type", "all")
            goals = self._query_goals(query_type, input_data)
            result["success"] = True
            result["goals"] = [g.dict() for g in goals]
            
        elif input_type == "abandon":
            goal_id = input_data.get("goal_id")
            if goal_id in self.active_goals:
                self._abandon_goal(goal_id, input_data.get("reason", "manual"))
                result["success"] = True
                result["message"] = f"Abandoned goal: {goal_id}"
            else:
                result["message"] = f"Goal {goal_id} not found"
                
        elif input_type == "generate":
            # Generate goals based on current motivational state
            new_goals = self._generate_goals(input_data)
            result["success"] = True
            result["goals_generated"] = len(new_goals)
            result["goal_ids"] = [g.id for g in new_goals]
            
        return result
    
    def _create_goal(self, input_data: Dict[str, Any]) -> Optional[Goal]:
        """Create a new goal from input data"""
        # Check if we're at max capacity and development level allows goal type
        if len(self.active_goals) >= self.max_active_goals:
            # If at capacity, only create if more important than least important current goal
            if not self._is_more_important_than_least(input_data.get("importance", 0.5)):
                self.logger.info("Goal creation rejected: at capacity and not important enough")
                return None
                
        # Check if developmental level supports this goal type
        goal_type = input_data.get("goal_type", "approach_immediate")
        if self.development_level < self.goal_type_thresholds.get(goal_type, 0):
            self.logger.info(f"Goal creation rejected: development level {self.development_level} too low for {goal_type}")
            return None
            
        # Create the goal
        try:
            goal = Goal(
                name=input_data.get("name", "Unnamed Goal"),
                description=input_data.get("description", ""),
                importance=input_data.get("importance", 0.5),
                urgency=input_data.get("urgency", 0.5),
                satisfies_needs=input_data.get("satisfies_needs", []),
                satisfies_drives=input_data.get("satisfies_drives", []),
                parent_goal_id=input_data.get("parent_goal_id"),
                expected_reward=input_data.get("expected_reward", 0.5),
                goal_type=input_data.get("approach_or_avoidance", "approach"),
                deadline=input_data.get("deadline")
            )
            
            # If this is a subgoal, update the parent
            if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
                parent_goal = self.active_goals[goal.parent_goal_id]
                if goal.id not in parent_goal.subgoals:
                    parent_goal.subgoals.append(goal.id)
                    
            return goal
            
        except Exception as e:
            self.logger.error(f"Error creating goal: {e}")
            return None
    
    def _update_goal(self, goal_id: str, update_data: Dict[str, Any]) -> None:
        """Update an existing goal with new information"""
        if goal_id not in self.active_goals:
            return
            
        goal = self.active_goals[goal_id]
        
        # Update progress if specified
        if "progress_delta" in update_data:
            delta = update_data["progress_delta"]
            goal.update_progress(delta)
            
        elif "progress" in update_data:
            # Direct setting of progress value
            old_progress = goal.progress
            goal.progress = min(1.0, max(0.0, update_data["progress"]))
            
            # If significant progress was made, check for goal completion
            if goal.progress >= 1.0 and old_progress < 1.0:
                self._complete_goal(goal_id)
                
        # Update other attributes if specified
        for attr in ["importance", "urgency", "expected_reward"]:
            if attr in update_data:
                setattr(goal, attr, update_data[attr])
                
        # Update deadline if specified
        if "deadline" in update_data:
            goal.deadline = update_data["deadline"]
            
        # Update last_updated timestamp
        goal.last_updated = datetime.now()
    
    def _complete_goal(self, goal_id: str) -> None:
        """Handle goal completion"""
        if goal_id not in self.active_goals:
            return
            
        goal = self.active_goals[goal_id]
        goal.is_achieved = True
        goal.progress = 1.0
        
        # Move to archived goals
        self.archived_goals[goal_id] = goal
        del self.active_goals[goal_id]
        
        # Limit archived goals size
        if len(self.archived_goals) > self.max_archived_goals:
            oldest_id = min(self.archived_goals.keys(), 
                         key=lambda k: self.archived_goals[k].last_updated)
            del self.archived_goals[oldest_id]
        
        # Update parent goal progress if applicable
        if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
            parent = self.active_goals[goal.parent_goal_id]
            # Calculate progress based on subgoal completion
            if parent.subgoals:
                completed_subgoals = sum(1 for sg_id in parent.subgoals 
                                      if sg_id in self.archived_goals and self.archived_goals[sg_id].is_achieved)
                new_progress = completed_subgoals / len(parent.subgoals)
                parent.progress = new_progress
        
        # Publish goal achieved event
        if self.event_bus:
            self.publish_message("goal_achieved", {
                "goal": goal.dict(),
                "satisfaction_level": goal.expected_reward
            })
            
        self.logger.info(f"Goal completed: {goal.name} ({goal_id})")
    
    def _abandon_goal(self, goal_id: str, reason: str) -> None:
        """Abandon a goal that is no longer relevant or feasible"""
        if goal_id not in self.active_goals:
            return
            
        goal = self.active_goals[goal_id]
        goal.abandon()
        
        # Move to archived goals
        self.archived_goals[goal_id] = goal
        del self.active_goals[goal_id]
        
        # Also abandon subgoals
        for subgoal_id in goal.subgoals:
            if subgoal_id in self.active_goals:
                self._abandon_goal(subgoal_id, f"parent_abandoned:{goal_id}")
                
        # Update parent goal if applicable
        if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
            parent = self.active_goals[goal.parent_goal_id]
            # Remove this goal from parent's subgoals
            if goal_id in parent.subgoals:
                parent.subgoals.remove(goal_id)
        
        # Publish goal abandoned event
        if self.event_bus:
            self.publish_message("goal_abandoned", {
                "goal": goal.dict(),
                "reason": reason
            })
            
        self.logger.info(f"Goal abandoned: {goal.name} ({goal_id}), reason: {reason}")
    
    def _query_goals(self, query_type: str, query_data: Dict[str, Any]) -> List[Goal]:
        """Query goals based on various criteria"""
        results = []
        
        if query_type == "all":
            # Return all active goals
            results = list(self.active_goals.values())
            
        elif query_type == "by_id":
            # Query specific goal by ID
            goal_id = query_data.get("goal_id")
            if goal_id in self.active_goals:
                results = [self.active_goals[goal_id]]
            elif goal_id in self.archived_goals:
                results = [self.archived_goals[goal_id]]
                
        elif query_type == "by_need":
            # Query goals that satisfy a specific need
            need_id = query_data.get("need_id")
            results = [g for g in self.active_goals.values() if need_id in g.satisfies_needs]
            
        elif query_type == "by_drive":
            # Query goals that satisfy a specific drive
            drive_id = query_data.get("drive_id")
            results = [g for g in self.active_goals.values() if drive_id in g.satisfies_drives]
            
        elif query_type == "by_parent":
            # Query subgoals of a specific parent
            parent_id = query_data.get("parent_id")
            results = [g for g in self.active_goals.values() if g.parent_goal_id == parent_id]
            
        elif query_type == "top_priority":
            # Get the highest priority goals
            limit = query_data.get("limit", 3)
            results = self._get_top_priority_goals(limit)
            
        elif query_type == "recent":
            # Get recently modified goals
            limit = query_data.get("limit", 5)
            results = sorted(self.active_goals.values(), 
                           key=lambda g: g.last_updated, reverse=True)[:limit]
        
        return results
    
    def _generate_goals(self, input_data: Dict[str, Any]) -> List[Goal]:
        """Generate goals based on current motivational state"""
        # Check cooldown to prevent goal spam
        if datetime.now() - self.last_goal_generation < self.goal_generation_cooldown:
            return []
            
        self.last_goal_generation = datetime.now()
        
        # Extract motivational data
        drives = input_data.get("drives", [])
        needs = input_data.get("needs", [])
        
        # Track generated goals
        generated_goals = []
        
        # Generate goals for unsatisfied drives (if development allows)
        if self.development_level >= 0.1:  # Basic goal generation
            for drive in drives:
                # Only generate for high-intensity drives
                if drive.get("intensity", 0) < 0.6:
                    continue
                    
                drive_id = drive.get("id")
                drive_name = drive.get("name", "unknown")
                
                # Check if we already have a goal for this drive
                existing = [g for g in self.active_goals.values() 
                          if drive_id in g.satisfies_drives]
                
                if not existing:
                    # Create a drive satisfaction goal
                    goal_data = {
                        "name": f"Satisfy {drive_name}",
                        "description": f"Goal to satisfy the {drive_name} drive",
                        "importance": min(1.0, drive.get("intensity", 0.5) + 0.2),
                        "urgency": drive.get("intensity", 0.5),
                        "satisfies_drives": [drive_id],
                        "expected_reward": 0.6,
                        "goal_type": "approach_immediate"
                    }
                    
                    new_goal = self._create_goal(goal_data)
                    if new_goal:
                        generated_goals.append(new_goal)
        
        # Generate goals for needs with development >= 0.3
        if self.development_level >= 0.3:
            for need in needs:
                # Only generate for unsatisfied important needs
                satisfaction = need.get("satisfaction", 1.0)
                if satisfaction > 0.7:
                    continue
                    
                need_id = need.get("id")
                need_name = need.get("name", "unknown")
                hierarchy_level = need.get("hierarchy_level", 1)
                
                # Higher-level needs require higher development
                if hierarchy_level > 2 and self.development_level < 0.5:
                    continue
                    
                # Check for existing goals for this need
                existing = [g for g in self.active_goals.values() 
                          if need_id in g.satisfies_needs]
                
                if not existing:
                    # Importance increases with hierarchy level and development
                    importance_factor = min(1.0, (hierarchy_level / 5) + (self.development_level / 2))
                    
                    # Create a need satisfaction goal
                    goal_data = {
                        "name": f"Fulfill {need_name}",
                        "description": f"Goal to satisfy the {need_name} need",
                        "importance": min(1.0, (1.0 - satisfaction) * importance_factor),
                        "urgency": max(0.3, 1.0 - satisfaction),
                        "satisfies_needs": [need_id],
                        "expected_reward": 0.5 + (hierarchy_level * 0.1),
                        "goal_type": "hierarchical" if self.development_level >= 0.5 else "sequential"
                    }
                    
                    new_goal = self._create_goal(goal_data)
                    if new_goal and self.development_level >= 0.5:
                        # Create subgoals if development allows
                        self._create_subgoals(new_goal)
                        generated_goals.append(new_goal)
        
        # Return list of generated goals
        return generated_goals
    
    def _create_subgoals(self, parent_goal: Goal) -> None:
        """Create appropriate subgoals for a parent goal (if development allows)"""
        # Only allowed if development is high enough
        if self.development_level < 0.5:
            return
            
        # Different subgoal generation based on goal type
        if "need" in parent_goal.name.lower():
            # For need-based goals, create steps to satisfy the need
            need_id = parent_goal.satisfies_needs[0] if parent_goal.satisfies_needs else None
            if not need_id:
                return
                
            # Query for more information about the need
            if self.event_bus:
                response = None
                query_id = str(uuid.uuid4())
                
                # Send query
                self.publish_message("need_query", {
                    "need_id": need_id,
                    "query_id": query_id
                })
                
                # Create 2-3 subgoals based on need type
                # In real implementation, would wait for async response
                # For now, create generic subgoals
                
                subgoal_count = 2 if self.development_level < 0.7 else 3
                for i in range(subgoal_count):
                    sg_data = {
                        "name": f"Step {i+1} for {parent_goal.name}",
                        "description": f"Subgoal to help achieve {parent_goal.name}",
                        "importance": parent_goal.importance * 0.9,
                        "urgency": parent_goal.urgency,
                        "satisfies_needs": parent_goal.satisfies_needs,
                        "parent_goal_id": parent_goal.id,
                        "expected_reward": parent_goal.expected_reward * 0.7,
                        "goal_type": "sequential"
                    }
                    
                    self._create_goal(sg_data)
                    
        elif "drive" in parent_goal.name.lower():
            # For drive-based goals, create steps to satisfy the drive
            drive_id = parent_goal.satisfies_drives[0] if parent_goal.satisfies_drives else None
            if not drive_id:
                return
                
            # Create 1-2 subgoals for drive satisfaction
            subgoal_count = 1 if self.development_level < 0.7 else 2
            for i in range(subgoal_count):
                sg_data = {
                    "name": f"Step {i+1} for {parent_goal.name}",
                    "description": f"Subgoal to help achieve {parent_goal.name}",
                    "importance": parent_goal.importance * 0.9,
                    "urgency": parent_goal.urgency,
                    "satisfies_drives": parent_goal.satisfies_drives,
                    "parent_goal_id": parent_goal.id,
                    "expected_reward": parent_goal.expected_reward * 0.7,
                    "goal_type": "approach_immediate"
                }
                
                self._create_goal(sg_data)
    
    def _get_top_priority_goals(self, limit: int = 3) -> List[Goal]:
        """Get the top priority goals based on importance and urgency"""
        if not self.active_goals:
            return []
            
        # Calculate priority score for each goal
        priority_goals = []
        for goal in self.active_goals.values():
            # Score calculation: combination of importance and urgency
            score = (goal.importance * 0.7) + (goal.urgency * 0.3)
            
            # Add deadline factor if present
            if goal.deadline:
                time_remaining = (goal.deadline - datetime.now()).total_seconds()
                if time_remaining > 0:
                    # Urgency increases as deadline approaches
                    urgency_factor = min(1.0, 86400 / max(time_remaining, 3600))  # 1.0 within one day
                    score += urgency_factor * 0.5
            
            priority_goals.append((score, goal))
            
        # Return top N by score
        return [g for _, g in heapq.nlargest(limit, priority_goals)]
    
    def _is_more_important_than_least(self, importance: float) -> bool:
        """Check if a new goal is more important than the least important current goal"""
        if not self.active_goals:
            return True
            
        least_important = min(self.active_goals.values(), key=lambda g: g.importance)
        return importance > least_important.importance
    
    def _calculate_max_active_goals(self) -> int:
        """Calculate maximum number of active goals based on development level"""
        # Development increases cognitive capacity for goals
        base_capacity = 3
        dev_bonus = int(self.development_level * 10)
        return base_capacity + dev_bonus
    
    def _calculate_planning_horizon(self) -> float:
        """Calculate planning horizon (in simulated time units) based on development"""
        # Higher development enables longer-term planning
        if self.development_level < 0.3:
            return 0.1  # Very short-term only
        elif self.development_level < 0.5:
            return 1.0  # Short-term
        elif self.development_level < 0.7:
            return 10.0  # Medium-term
        elif self.development_level < 0.9:
            return 100.0  # Long-term
        else:
            return 1000.0  # Very long-term
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "drive_update" or message_type == "need_update":
            # Check if we should generate new goals
            motivational_state = content.get("motivational_state", {})
            
            # Only generate goals periodically and when sufficiently developed
            if (self.development_level >= 0.1 and 
                datetime.now() - self.last_goal_generation > self.goal_generation_cooldown):
                
                self._generate_goals(motivational_state)
            
        elif message_type == "action_completed":
            # Check if this action contributes to any goal
            action_name = content.get("action", {}).get("name", "")
            affected_goals = content.get("affected_goals", [])
            success = content.get("success", False)
            
            if success and affected_goals:
                # Update progress for each affected goal
                for goal_id in affected_goals:
                    if goal_id in self.active_goals:
                        # Update progress
                        progress_increment = content.get("progress_increment", 0.1)
                        self._update_goal(goal_id, {"progress_delta": progress_increment})
                        
        elif message_type == "goal_query":
            # Handle goal query
            query_type = content.get("query_type", "all")
            query_id = content.get("query_id", "")
            
            # Get matching goals
            goals = self._query_goals(query_type, content)
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="goal_query_response",
                    content={
                        "query_id": query_id,
                        "goals": [g.dict() for g in goals]
                    },
                    reply_to=message.id
                ))
                
        elif message_type == "create_goal":
            result = self.process_input({
                "type": "create",
                **content
            })
            
            # Send response if requested
            if "query_id" in content and self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="create_goal_response",
                    content={
                        "query_id": content["query_id"],
                        **result
                    },
                    reply_to=message.id
                ))
                
        elif message_type == "update_goal":
            goal_id = content.get("goal_id")
            if goal_id:
                self.process_input({
                    "type": "update",
                    **content
                })
                
        elif message_type == "abandon_goal":
            goal_id = content.get("goal_id")
            if goal_id:
                self.process_input({
                    "type": "abandon",
                    **content
                })
                
        elif message_type == "perception_update":
            # Check if any perception is relevant to goals
            perception = content.get("perception", {})
            
            # TODO: More sophisticated goal relevance detection
            # For now, a simple keyword check
            if "perception_text" in perception:
                text = perception["perception_text"].lower()
                
                # Check for relevance to each goal
                for goal in list(self.active_goals.values()):
                    # Simple keyword match for relevance
                    if (goal.name.lower() in text or
                        any(word in text for word in goal.name.lower().split())):
                        
                        # Update goal with small progress increment
                        self._update_goal(goal.id, {"progress_delta": 0.05})
    
    def update_development(self, amount: float) -> float:
        """
        Update the module's development level
        
        Args:
            amount: Amount to increase development (0.0 to 1.0)
            
        Returns:
            New development level
        """
        # Update base development
        old_level = self.development_level
        super().update_development(amount)
        
        # Update derived values
        self.max_active_goals = self._calculate_max_active_goals()
        self.planning_horizon = self._calculate_planning_horizon()
        
        # Update neural network development
        self.neural_network.update_development(amount)
        
        # Log milestone achievements
        self._check_development_milestones(old_level)
        
        return self.development_level
    
    def _check_development_milestones(self, old_level: float) -> None:
        """
        Check if any development milestones have been reached
        
        Parameters:
        old_level: The previous development level
        """
        # Get the current milestone based on the new development level
        current_milestone = None
        for level, milestone in sorted(self.development_milestones.items()):
            if self.development_level >= float(level):
                current_milestone = milestone
        
        # Get the previous milestone based on the old development level
        previous_milestone = None
        for level, milestone in sorted(self.development_milestones.items()):
            if old_level >= float(level):
                previous_milestone = milestone
        
        # If we've reached a new milestone, log it
        if current_milestone != previous_milestone and current_milestone is not None:
            logger = logging.getLogger(f"lmm.motivation.goal_setting.{self.module_id}")
            logger.info(f"Reached new development milestone: {current_milestone}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with state information
        """
        base_state = super().get_state()
        
        # Convert goals to dictionaries
        active_goals_dict = {gid: goal.dict() for gid, goal in self.active_goals.items()}
        
        # Only include recent archived goals in state
        recent_archived = dict(sorted(self.archived_goals.items(), 
                                   key=lambda x: x[1].last_updated, reverse=True)[:10])
        archived_goals_dict = {gid: goal.dict() for gid, goal in recent_archived.items()}
        
        goal_state = {
            "active_goals": active_goals_dict,
            "archived_goals": archived_goals_dict,
            "max_active_goals": self.max_active_goals,
            "planning_horizon": self.planning_horizon,
            "neural_state": self.neural_network.get_state()
        }
        
        return {**base_state, **goal_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save basic module state
        state_path = os.path.join(module_dir, "module_state.json")
        with open(state_path, 'w') as f:
            # Get state and convert goals to dicts
            state = self.get_state()
            json.dump(state, f, indent=2, default=str)
            
        # Save neural network state
        nn_path = os.path.join(module_dir, "neural_network.pt")
        self.neural_network.save(nn_path)
        
        # Save complete goal archives separately (if many)
        if len(self.archived_goals) > 10:
            archives_path = os.path.join(module_dir, "archived_goals.json")
            with open(archives_path, 'w') as f:
                archived_dict = {gid: goal.dict() for gid, goal in self.archived_goals.items()}
                json.dump(archived_dict, f, indent=2, default=str)
        
        self.logger.info(f"Saved goal setting module state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            module_dir = os.path.dirname(state_path)
            
            # Load module state
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update development level and milestones
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Update derived values
            self.max_active_goals = self._calculate_max_active_goals()
            self.planning_horizon = self._calculate_planning_horizon()
            
            # Load goals
            from lmm_project.modules.motivation.models import Goal
            self.active_goals = {}
            for goal_id, goal_dict in state.get("active_goals", {}).items():
                try:
                    self.active_goals[goal_id] = Goal(**goal_dict)
                except Exception as e:
                    self.logger.warning(f"Could not load active goal {goal_id}: {e}")
                    
            # Load recent archived goals
            self.archived_goals = {}
            for goal_id, goal_dict in state.get("archived_goals", {}).items():
                try:
                    self.archived_goals[goal_id] = Goal(**goal_dict)
                except Exception as e:
                    self.logger.warning(f"Could not load archived goal {goal_id}: {e}")
            
            # Check for full archives file
            archives_path = os.path.join(module_dir, "archived_goals.json")
            if os.path.exists(archives_path):
                with open(archives_path, 'r') as f:
                    archives = json.load(f)
                    
                # Only add archives not already loaded
                for goal_id, goal_dict in archives.items():
                    if goal_id not in self.archived_goals:
                        try:
                            self.archived_goals[goal_id] = Goal(**goal_dict)
                        except Exception as e:
                            self.logger.warning(f"Could not load full archived goal {goal_id}: {e}")
                
            # Load neural network state
            nn_path = os.path.join(module_dir, "neural_network.pt")
            if os.path.exists(nn_path):
                self.neural_network.load(nn_path)
                self.logger.info(f"Loaded neural network from {nn_path}")
            else:
                self.logger.warning(f"Neural network state not found at {nn_path}")
                
            self.logger.info(f"Loaded goal setting module state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load goal setting module state: {e}")
            return False 