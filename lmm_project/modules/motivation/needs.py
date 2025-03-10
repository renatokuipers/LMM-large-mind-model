"""
Needs Module

This module implements psychological needs based on theories like Maslow's hierarchy.
It manages need-based motivation and tracks satisfaction levels across different
need categories, from basic physiological needs to higher-level self-actualization.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import os
import json
import numpy as np
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.motivation.models import Need, MotivationalState
from lmm_project.modules.motivation.neural_net import MotivationNeuralNetwork

logger = logging.getLogger(__name__)

class Needs(BaseModule):
    """
    Handles psychological needs based on Maslow's hierarchy
    
    This module tracks need satisfaction levels, generates motivation
    based on unsatisfied needs, and implements hierarchical satisfaction
    principles (higher needs become active when lower needs are satisfied).
    """
    
    # Developmental milestones for needs module
    development_milestones = {
        0.0: "Basic physiological needs",
        0.2: "Safety and security needs",
        0.4: "Belonging and social needs",
        0.6: "Esteem and recognition needs",
        0.8: "Cognitive needs", 
        1.0: "Self-actualization needs"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the needs module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="needs",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize need storage
        self.needs: Dict[str, Need] = {}
        
        # Last update timestamp
        self.last_update = datetime.now()
        
        # Neural network for need evaluation
        self.neural_net = MotivationNeuralNetwork(
            input_dim=48,
            hidden_dim=64,
            output_dim=16,
            network_type="need",
            development_level=development_level
        )
        
        # Need priority thresholds
        self.high_priority_threshold = 0.3  # Needs with satisfaction below this are high priority
        
        # Storage path
        self.storage_dir = Path("storage/motivation/needs")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize needs based on development level
        self._initialize_needs()
        
        # Subscribe to messages
        if self.event_bus:
            self.subscribe_to_message("drive_satisfied")
            self.subscribe_to_message("need_query")
            self.subscribe_to_message("goal_achieved")
            self.subscribe_to_message("perception_result")
            self.subscribe_to_message("social_interaction")
    
    def _initialize_needs(self):
        """Initialize needs based on current development level"""
        # Level 1: Physiological needs (always present)
        physiological_needs = [
            Need(
                name="nourishment",
                description="Need for adequate food and water",
                hierarchy_level=1,
                satisfaction=0.7,
                decay_rate=0.01,
                related_drives=["energy"],
                min_development_level=0.0
            ),
            Need(
                name="rest",
                description="Need for sleep and recovery",
                hierarchy_level=1,
                satisfaction=0.6,
                decay_rate=0.008,
                related_drives=["rest"],
                min_development_level=0.0
            ),
            Need(
                name="physical_comfort",
                description="Need for comfortable physical conditions",
                hierarchy_level=1,
                satisfaction=0.8,
                decay_rate=0.005,
                related_drives=["comfort"],
                min_development_level=0.0
            )
        ]
        
        for need in physiological_needs:
            self.needs[need.id] = need
        
        # Level 2: Safety needs
        if self.development_level >= 0.2:
            safety_needs = [
                Need(
                    name="security",
                    description="Need for safety and security",
                    hierarchy_level=2,
                    satisfaction=0.6,
                    decay_rate=0.004,
                    related_drives=["safety"],
                    prerequisites=["nourishment", "rest"],
                    min_development_level=0.2
                ),
                Need(
                    name="stability",
                    description="Need for routine and predictability",
                    hierarchy_level=2,
                    satisfaction=0.5,
                    decay_rate=0.003,
                    related_drives=["safety"],
                    min_development_level=0.2
                )
            ]
            
            for need in safety_needs:
                self.needs[need.id] = need
                
        # Level 3: Social needs
        if self.development_level >= 0.4:
            social_needs = [
                Need(
                    name="belonging",
                    description="Need to belong to a group",
                    hierarchy_level=3,
                    satisfaction=0.4,
                    decay_rate=0.003,
                    related_drives=["social"],
                    prerequisites=["security"],
                    min_development_level=0.4
                ),
                Need(
                    name="affection",
                    description="Need for affection and positive regard",
                    hierarchy_level=3,
                    satisfaction=0.4,
                    decay_rate=0.002,
                    related_drives=["social"],
                    min_development_level=0.4
                )
            ]
            
            for need in social_needs:
                self.needs[need.id] = need
                
        # Level 4: Esteem needs
        if self.development_level >= 0.6:
            esteem_needs = [
                Need(
                    name="competence",
                    description="Need to feel capable and effective",
                    hierarchy_level=4,
                    satisfaction=0.3,
                    decay_rate=0.002,
                    related_drives=["mastery"],
                    prerequisites=["belonging"],
                    min_development_level=0.6
                ),
                Need(
                    name="achievement",
                    description="Need for achievement and recognition",
                    hierarchy_level=4,
                    satisfaction=0.3,
                    decay_rate=0.002,
                    related_drives=["mastery"],
                    min_development_level=0.6
                ),
                Need(
                    name="independence",
                    description="Need for autonomy and self-direction",
                    hierarchy_level=4,
                    satisfaction=0.3,
                    decay_rate=0.002,
                    related_drives=["autonomy"],
                    min_development_level=0.6
                )
            ]
            
            for need in esteem_needs:
                self.needs[need.id] = need
                
        # Level 5: Cognitive and aesthetic needs  
        if self.development_level >= 0.8:
            cognitive_needs = [
                Need(
                    name="understanding",
                    description="Need to understand and make sense of experience",
                    hierarchy_level=5,
                    satisfaction=0.2,
                    decay_rate=0.001,
                    related_drives=["curiosity"],
                    prerequisites=["competence"],
                    min_development_level=0.8
                ),
                Need(
                    name="exploration",
                    description="Need to explore and discover",
                    hierarchy_level=5,
                    satisfaction=0.2,
                    decay_rate=0.001,
                    related_drives=["curiosity"],
                    min_development_level=0.8
                ),
                Need(
                    name="aesthetics",
                    description="Need for beauty, order, and symmetry",
                    hierarchy_level=5,
                    satisfaction=0.3,
                    decay_rate=0.001,
                    related_drives=["creativity"],
                    min_development_level=0.8
                )
            ]
            
            for need in cognitive_needs:
                self.needs[need.id] = need
                
        # Level 6: Self-actualization needs
        if self.development_level >= 0.95:  # Only at very high development
            self_actualization_needs = [
                Need(
                    name="self_development",
                    description="Need to grow and develop full potential",
                    hierarchy_level=6,
                    satisfaction=0.1,
                    decay_rate=0.0005,
                    related_drives=["meaning"],
                    prerequisites=["understanding", "independence"],
                    min_development_level=0.95
                ),
                Need(
                    name="purpose",
                    description="Need for meaning and purpose",
                    hierarchy_level=6,
                    satisfaction=0.1,
                    decay_rate=0.0005,
                    related_drives=["meaning"],
                    min_development_level=0.95
                )
            ]
            
            for need in self_actualization_needs:
                self.needs[need.id] = need
                
        # Set initial active state based on prerequisites
        self._update_need_active_states()
        
        logger.info(f"Initialized {len(self.needs)} needs at development level {self.development_level}")
    
    def _update_need_active_states(self):
        """Update which needs are active based on their prerequisites"""
        # Needs at level 1 are always active if they meet the development level
        for need_id, need in self.needs.items():
            # First check development level requirement
            if need.min_development_level > self.development_level:
                need.is_active = False
                continue
                
            # Level 1 needs are always active if they meet development requirement
            if need.hierarchy_level == 1:
                need.is_active = True
                continue
                
            # For higher-level needs, check prerequisites
            if not need.prerequisites:
                # No prerequisites, just check development level (already done)
                need.is_active = True
                continue
                
            # Check if all prerequisites are active and sufficiently satisfied
            prerequisites_met = True
            for prereq_name in need.prerequisites:
                # Find the prerequisite need
                prereq_need = None
                for n in self.needs.values():
                    if n.name == prereq_name:
                        prereq_need = n
                        break
                        
                if not prereq_need or not prereq_need.is_active or prereq_need.satisfaction < 0.5:
                    prerequisites_met = False
                    break
                    
            need.is_active = prerequisites_met
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update need states
        
        Args:
            input_data: Input data with operation details
                Supported operations:
                - get_needs: Get all need states
                - update_need: Update a specific need
                - get_priority_needs: Get high-priority (low-satisfaction) needs
                - get_needs_by_level: Get needs at a specific hierarchy level
                - satisfy_need: Satisfy a specific need
            
        Returns:
            Operation result with need information
        """
        operation = input_data.get("operation", "")
        result = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "module_id": self.module_id
        }
        
        # Update all needs based on elapsed time
        self._update_all_needs()
        
        if operation == "get_needs":
            # Return all need states
            needs_info = {
                need_id: {
                    "name": need.name,
                    "satisfaction": need.satisfaction,
                    "hierarchy_level": need.hierarchy_level,
                    "is_active": need.is_active
                } for need_id, need in self.needs.items()
            }
            
            result.update({
                "status": "success",
                "needs": needs_info,
                "count": len(needs_info)
            })
            
        elif operation == "update_need":
            # Update a specific need
            need_id = input_data.get("need_id")
            change = input_data.get("change", 0.0)
            
            if need_id and need_id in self.needs:
                need = self.needs[need_id]
                old_satisfaction = need.satisfaction
                need.update_satisfaction(change)
                
                # Recalculate active states since satisfaction changed
                self._update_need_active_states()
                
                result.update({
                    "status": "success",
                    "need_id": need_id,
                    "name": need.name,
                    "old_satisfaction": old_satisfaction,
                    "new_satisfaction": need.satisfaction,
                    "is_active": need.is_active
                })
                
                # Publish need update message
                if self.event_bus:
                    self.publish_message("need_update", {
                        "need_id": need_id,
                        "need_name": need.name,
                        "satisfaction": need.satisfaction,
                        "hierarchy_level": need.hierarchy_level,
                        "satisfaction_delta": change,
                        "category": self._get_need_category(need.hierarchy_level)
                    })
            else:
                result.update({
                    "status": "error",
                    "error": f"Need not found: {need_id}",
                    "available_needs": list(self.needs.keys())
                })
                
        elif operation == "get_priority_needs":
            # Get needs with low satisfaction (high priority)
            priority_needs = self._get_priority_needs()
            
            result.update({
                "status": "success",
                "priority_needs": priority_needs,
                "count": len(priority_needs),
                "threshold": self.high_priority_threshold
            })
            
        elif operation == "get_needs_by_level":
            # Get needs at a specific hierarchy level
            level = input_data.get("level", 1)
            
            needs_at_level = [
                {
                    "id": need_id,
                    "name": need.name,
                    "satisfaction": need.satisfaction,
                    "is_active": need.is_active
                }
                for need_id, need in self.needs.items()
                if need.hierarchy_level == level
            ]
            
            result.update({
                "status": "success",
                "level": level,
                "needs": needs_at_level,
                "count": len(needs_at_level)
            })
            
        elif operation == "satisfy_need":
            # Satisfy a specific need
            need_id = input_data.get("need_id")
            amount = input_data.get("amount", 0.2)
            
            if need_id and need_id in self.needs:
                need = self.needs[need_id]
                old_satisfaction = need.satisfaction
                need.update_satisfaction(amount)  # Positive change = increase in satisfaction
                
                # Recalculate active states
                self._update_need_active_states()
                
                result.update({
                    "status": "success",
                    "need_id": need_id,
                    "name": need.name,
                    "old_satisfaction": old_satisfaction,
                    "new_satisfaction": need.satisfaction,
                    "satisfaction_amount": amount,
                    "is_active": need.is_active
                })
                
                # Publish need update message
                if self.event_bus:
                    self.publish_message("need_satisfied", {
                        "need_id": need_id,
                        "need_name": need.name,
                        "satisfaction": need.satisfaction,
                        "satisfaction_amount": amount,
                        "hierarchy_level": need.hierarchy_level,
                        "category": self._get_need_category(need.hierarchy_level)
                    })
            else:
                result.update({
                    "status": "error",
                    "error": f"Need not found: {need_id}",
                    "available_needs": list(self.needs.keys())
                })
                
        else:
            result.update({
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "get_needs", "update_need", "get_priority_needs", 
                    "get_needs_by_level", "satisfy_need"
                ]
            })
            
        return result
    
    def _update_all_needs(self):
        """Update all needs based on time elapsed since last update"""
        now = datetime.now()
        elapsed_seconds = (now - self.last_update).total_seconds()
        
        # Don't update if very little time has passed
        if elapsed_seconds < 0.1:
            return
            
        for need_id, need in self.needs.items():
            # Only active needs decay
            if need.is_active:
                # Apply natural decay based on need's decay rate
                need.decay(elapsed_seconds)
        
        # After changing satisfaction, update active states
        self._update_need_active_states()
        
        self.last_update = now
    
    def _get_priority_needs(self) -> List[Dict[str, Any]]:
        """Get needs with low satisfaction (high priority)"""
        priority_needs = []
        
        for need_id, need in self.needs.items():
            # Only active needs can be priority needs
            if need.is_active and need.satisfaction <= self.high_priority_threshold:
                # Calculate a priority score that considers hierarchy level and satisfaction
                # Lower levels and lower satisfaction create higher priority
                priority_score = (1.0 - need.satisfaction) * (7 - need.hierarchy_level) / 6
                
                priority_needs.append({
                    "id": need_id,
                    "name": need.name,
                    "satisfaction": need.satisfaction,
                    "hierarchy_level": need.hierarchy_level,
                    "priority_score": priority_score
                })
                
        # Sort by priority score (highest first)
        priority_needs.sort(key=lambda n: n["priority_score"], reverse=True)
        return priority_needs
    
    def _get_need_category(self, hierarchy_level: int) -> str:
        """Convert hierarchy level to a category name"""
        categories = {
            1: "physiological",
            2: "safety",
            3: "social",
            4: "esteem",
            5: "cognitive",
            6: "self_actualization"
        }
        return categories.get(hierarchy_level, "unknown")
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase the development level
            
        Returns:
            New development level
        """
        old_level = self.development_level
        new_level = super().update_development(amount)
        
        # If we crossed a threshold, update available needs
        if int(old_level * 10) != int(new_level * 10):
            self._initialize_needs()
            
            # Also update neural network development
            self.neural_net.update_development(amount)
            
            logger.info(f"Needs module advanced to development level {new_level:.2f}")
            
        return new_level
    
    def _handle_message(self, message: Message):
        """
        Handle incoming messages
        
        Args:
            message: The message to process
        """
        if message.message_type == "drive_satisfied":
            # Drive satisfaction may affect related needs
            if message.content:
                self._process_drive_satisfaction(message.content)
                
        elif message.message_type == "need_query":
            # Direct query about need states
            if message.content and "query_type" in message.content:
                self._process_need_query(message.content, message.id)
                
        elif message.message_type == "goal_achieved":
            # Goal achievement may satisfy related needs
            if message.content and "goal_id" in message.content:
                self._process_goal_achievement(message.content)
                
        elif message.message_type == "perception_result":
            # Perception may relate to need satisfaction
            if message.content and "result" in message.content:
                self._process_perception(message.content["result"])
                
        elif message.message_type == "social_interaction":
            # Social interactions affect social needs
            if message.content:
                self._process_social_interaction(message.content)
    
    def _process_drive_satisfaction(self, drive_data: Dict[str, Any]):
        """Process drive satisfaction for related needs"""
        drive_name = drive_data.get("drive_name", "")
        satisfaction_amount = drive_data.get("satisfaction_amount", 0.0)
        
        # Find needs related to this drive
        for need_id, need in self.needs.items():
            if drive_name in need.related_drives and need.is_active:
                # Drive satisfaction translates to need satisfaction, but typically at a lower rate
                # Lower level needs are more directly affected by drives
                satisfaction_factor = 0.5 if need.hierarchy_level <= 2 else 0.2
                need.update_satisfaction(satisfaction_amount * satisfaction_factor)
        
        # Recalculate active states
        self._update_need_active_states()
    
    def _process_need_query(self, query: Dict[str, Any], message_id: str):
        """Process a direct query about need states"""
        query_type = query.get("query_type")
        
        if query_type == "priority_needs":
            # Return priority needs
            priority_needs = self._get_priority_needs()
            
            if self.event_bus:
                self.publish_message("need_query_response", {
                    "query_id": message_id,
                    "priority_needs": priority_needs,
                    "count": len(priority_needs)
                })
                
        elif query_type == "need_state":
            # Return specific need state
            need_id = query.get("need_id")
            if need_id and need_id in self.needs:
                need = self.needs[need_id]
                
                if self.event_bus:
                    self.publish_message("need_query_response", {
                        "query_id": message_id,
                        "need_id": need_id,
                        "name": need.name,
                        "satisfaction": need.satisfaction,
                        "hierarchy_level": need.hierarchy_level,
                        "is_active": need.is_active,
                        "category": self._get_need_category(need.hierarchy_level)
                    })
            elif query.get("need_name"):
                # Try finding by name
                need_name = query.get("need_name")
                for need_id, need in self.needs.items():
                    if need.name == need_name:
                        if self.event_bus:
                            self.publish_message("need_query_response", {
                                "query_id": message_id,
                                "need_id": need_id,
                                "name": need.name,
                                "satisfaction": need.satisfaction,
                                "hierarchy_level": need.hierarchy_level,
                                "is_active": need.is_active,
                                "category": self._get_need_category(need.hierarchy_level)
                            })
                        break
                else:
                    # Need not found
                    if self.event_bus:
                        self.publish_message("need_query_response", {
                            "query_id": message_id,
                            "error": f"Need not found: {need_name}",
                            "available_needs": [n.name for n in self.needs.values()]
                        })
            else:
                # Need not found
                if self.event_bus:
                    self.publish_message("need_query_response", {
                        "query_id": message_id,
                        "error": f"Need not found: {need_id}",
                        "available_needs": list(self.needs.keys())
                    })
    
    def _process_goal_achievement(self, goal_data: Dict[str, Any]):
        """Process goal achievement for need satisfaction"""
        # Goals often satisfy needs
        need_ids = goal_data.get("satisfies_needs", [])
        satisfaction_amount = goal_data.get("satisfaction_amount", 0.3)
        
        for need_id in need_ids:
            if need_id in self.needs:
                self.needs[need_id].update_satisfaction(satisfaction_amount)
        
        # Also check if the goal has a specific need category
        need_category = goal_data.get("need_category")
        if need_category:
            # Find needs in this category
            for need in self.needs.values():
                if self._get_need_category(need.hierarchy_level) == need_category and need.is_active:
                    need.update_satisfaction(satisfaction_amount * 0.5)  # Partial satisfaction
        
        # Recalculate active states
        self._update_need_active_states()
    
    def _process_perception(self, perception: Dict[str, Any]):
        """Process perception data for need updates"""
        # Some perceptions may relate to need satisfaction
        perception_type = perception.get("type", "")
        
        if perception_type == "threat":
            # Threat reduces safety needs satisfaction
            for need in self.needs.values():
                if need.hierarchy_level == 2 and need.is_active:  # Safety needs
                    need.update_satisfaction(-0.2)
                    
        elif perception_type == "social_opportunity":
            # Social opportunities can satisfy social needs
            for need in self.needs.values():
                if need.hierarchy_level == 3 and need.is_active:  # Social needs
                    need.update_satisfaction(0.1)
                    
        elif perception_type == "learning_opportunity":
            # Learning opportunities can satisfy cognitive needs
            for need in self.needs.values():
                if need.hierarchy_level == 5 and need.is_active:  # Cognitive needs
                    need.update_satisfaction(0.1)
        
        # Recalculate active states
        self._update_need_active_states()
    
    def _process_social_interaction(self, interaction_data: Dict[str, Any]):
        """Process social interaction data for social needs"""
        # Social interactions primarily affect social needs
        quality = interaction_data.get("quality", 0.0)  # -1.0 to 1.0
        
        # Find social needs
        for need in self.needs.values():
            if need.hierarchy_level == 3 and need.is_active:  # Social needs
                # Quality affects satisfaction (positive quality increases satisfaction)
                need.update_satisfaction(quality * 0.2)
        
        # Certain interactions also affect esteem needs
        if "recognition" in interaction_data or "feedback" in interaction_data:
            for need in self.needs.values():
                if need.hierarchy_level == 4 and need.is_active:  # Esteem needs
                    feedback_quality = interaction_data.get("feedback_quality", quality)
                    need.update_satisfaction(feedback_quality * 0.15)
        
        # Recalculate active states
        self._update_need_active_states()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        need_states = {}
        for need_id, need in self.needs.items():
            need_states[need_id] = {
                "name": need.name,
                "hierarchy_level": need.hierarchy_level,
                "satisfaction": need.satisfaction,
                "is_active": need.is_active,
                "last_updated": need.last_updated.isoformat()
            }
            
        return {
            "development_level": self.development_level,
            "needs": need_states,
            "high_priority_threshold": self.high_priority_threshold,
            "last_update": self.last_update.isoformat(),
            "neural_net": self.neural_net.get_state()
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the current state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, f"{self.module_id}_state.json")
        
        # Convert needs to serializable format
        need_states = {}
        for need_id, need in self.needs.items():
            need_states[need_id] = {
                "name": need.name,
                "description": need.description,
                "satisfaction": need.satisfaction,
                "decay_rate": need.decay_rate,
                "hierarchy_level": need.hierarchy_level,
                "min_development_level": need.min_development_level,
                "prerequisites": need.prerequisites,
                "related_drives": need.related_drives,
                "satisfying_goals": list(need.satisfying_goals),
                "last_updated": need.last_updated.isoformat(),
                "is_active": need.is_active
            }
        
        state = {
            "module_id": self.module_id,
            "module_type": "needs",
            "development_level": self.development_level,
            "needs": need_states,
            "high_priority_threshold": self.high_priority_threshold,
            "last_update": self.last_update.isoformat()
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        # Also save neural network state
        net_path = os.path.join(state_dir, f"{self.module_id}_network.pt")
        self.neural_net.save(net_path)
        
        logger.info(f"Saved needs state to {state_path}")
        return state_path
        
    def load_state(self, state_path: str) -> bool:
        """
        Load state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            Success flag
        """
        if not os.path.exists(state_path):
            logger.error(f"State file not found: {state_path}")
            return False
            
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Verify this is the right kind of state
            if state.get("module_type") != "needs":
                logger.error(f"Invalid state file type: {state.get('module_type')}")
                return False
                
            # Load development level
            self.development_level = state.get("development_level", 0.0)
            
            # Load high priority threshold
            self.high_priority_threshold = state.get("high_priority_threshold", 0.3)
            
            # Load last update time
            last_update_str = state.get("last_update", datetime.now().isoformat())
            self.last_update = datetime.fromisoformat(last_update_str)
            
            # Load needs
            need_states = state.get("needs", {})
            self.needs = {}
            
            for need_id, need_data in need_states.items():
                need = Need(
                    id=need_id,
                    name=need_data["name"],
                    description=need_data.get("description", ""),
                    satisfaction=need_data["satisfaction"],
                    decay_rate=need_data["decay_rate"],
                    hierarchy_level=need_data["hierarchy_level"],
                    min_development_level=need_data.get("min_development_level", 0.0),
                    prerequisites=need_data.get("prerequisites", []),
                    related_drives=need_data.get("related_drives", []),
                    satisfying_goals=set(need_data.get("satisfying_goals", [])),
                    last_updated=datetime.fromisoformat(need_data["last_updated"]),
                    is_active=need_data.get("is_active", True)
                )
                self.needs[need_id] = need
                
            # Load neural network state
            net_path = state_path.replace("_state.json", "_network.pt")
            if os.path.exists(net_path):
                self.neural_net.load(net_path)
                
            logger.info(f"Loaded needs state from {state_path} with {len(self.needs)} needs")
            return True
            
        except Exception as e:
            logger.error(f"Error loading needs state: {e}")
            return False
