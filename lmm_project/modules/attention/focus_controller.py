from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.attention.models import (
    AttentionFocus, AttentionTarget, AttentionParameters, SalienceScore
)

class FocusController(BaseModule):
    """
    Controls the focus of attention
    
    This module manages where attention is directed, maintaining a limited
    capacity focus buffer and handling the shifting of attention between
    different targets based on salience and task demands.
    """
    # Current focus of attention
    current_focus: AttentionFocus = Field(default_factory=AttentionFocus)
    # Parameters controlling attention behavior
    parameters: AttentionParameters = Field(default_factory=AttentionParameters)
    # History of focus shifts (for learning patterns)
    focus_shift_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Maximum history size
    max_history_size: int = Field(default=100)
    # Last time attention was updated
    last_update: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize focus controller module"""
        super().__init__(
            module_id=module_id,
            module_type="focus_controller",
            event_bus=event_bus,
            **data
        )
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("salience_detected", self._handle_salience_detected)
            self.subscribe_to_message("executive_command", self._handle_executive_command)
            self.subscribe_to_message("perception_input", self._handle_perception_input)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform
            - Additional parameters specific to the operation
            
        Returns:
        Operation result
        """
        # Update state to handle time-based decay
        self._update_state()
        
        operation = input_data.get("operation", "")
        
        if operation == "get_focus":
            return {
                "status": "success",
                "focus": self.current_focus.model_dump()
            }
            
        elif operation == "update_focus":
            # Update based on salience scores
            salience_scores = input_data.get("salience_scores", {})
            return self.update_focus_from_salience(salience_scores)
            
        elif operation == "add_target":
            # Add a specific target to attention
            target_data = input_data.get("target", {})
            target_id = target_data.get("target_id", "")
            target_type = target_data.get("target_type", "unknown")
            activation = target_data.get("activation", 1.0)
            description = target_data.get("description", "")
            
            return self.add_attention_target(
                target_id=target_id,
                target_type=target_type,
                activation=activation,
                description=description
            )
            
        elif operation == "remove_target":
            # Remove a target from attention
            target_id = input_data.get("target_id", "")
            return self.remove_attention_target(target_id)
            
        elif operation == "shift_focus":
            # Explicitly shift focus to a specified target
            target_id = input_data.get("target_id", "")
            priority = input_data.get("priority", 0.5)
            
            return self.shift_focus_to(target_id, priority)
            
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update module's developmental level
        
        As the focus controller develops:
        - Attention capacity increases
        - Focus becomes more stable (lower decay rate)
        - Attention shifting becomes more controlled
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development level change
        delta = self.development_level - prev_level
        
        # Decrease decay rate (more stable focus)
        decay_decrease = delta * 0.02
        self.parameters.decay_rate = max(0.01, self.parameters.decay_rate - decay_decrease)
        
        # Increase capacity (can attend to more things simultaneously)
        capacity_increase = delta * 0.5  # Gradually increase capacity
        self.current_focus.capacity = min(7.0, self.current_focus.capacity + capacity_increase)
        
        # Reduce shift threshold (more controlled attention shifting)
        threshold_decrease = delta * 0.05
        self.parameters.shift_threshold = max(0.1, self.parameters.shift_threshold - threshold_decrease)
        
        return self.development_level
    
    def update_focus_from_salience(self, salience_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update attention focus based on salience scores
        
        Parameters:
        salience_scores: Dictionary mapping item IDs to salience information
        
        Returns:
        Operation result
        """
        # Track what changes we make
        added_targets = []
        removed_targets = []
        updated_targets = []
        
        # Ensure salience scores are normalized
        if not salience_scores:
            return {
                "status": "success", 
                "message": "No salience scores provided",
                "added": added_targets,
                "removed": removed_targets,
                "updated": updated_targets
            }
        
        # Create a priority queue of items by salience
        items_by_salience = []
        for item_id, item_data in salience_scores.items():
            # Handle both object and direct score
            if isinstance(item_data, dict):
                score = item_data.get("score", 0.0)
                item_type = item_data.get("target_type", "unknown")
                description = item_data.get("description", "")
            else:
                score = float(item_data)
                item_type = "unknown"
                description = ""
                
            items_by_salience.append({
                "item_id": item_id,
                "score": score,
                "type": item_type,
                "description": description
            })
        
        # Sort by salience score
        items_by_salience.sort(key=lambda x: x["score"], reverse=True)
        
        # Update existing targets first
        for item in items_by_salience:
            item_id = item["item_id"]
            salience = item["score"]
            
            if item_id in self.current_focus.targets:
                # Update existing target's activation based on salience
                prev_activation = self.current_focus.targets[item_id]
                new_activation = min(1.0, prev_activation + salience * self.parameters.salience_sensitivity)
                
                # Apply the update
                target = self.current_focus.target_details[item_id]
                target.update_activation(new_activation - prev_activation)
                self.current_focus.targets[item_id] = new_activation
                
                updated_targets.append({
                    "target_id": item_id,
                    "prev_activation": prev_activation,
                    "new_activation": new_activation
                })
        
        # Now consider adding new targets
        remaining_capacity = max(0, self.current_focus.capacity - len(self.current_focus.targets))
        
        for item in items_by_salience:
            item_id = item["item_id"]
            salience = item["score"]
            
            # Skip if already in focus
            if item_id in self.current_focus.targets:
                continue
                
            # Check if salience exceeds shift threshold
            if salience >= self.parameters.shift_threshold:
                if remaining_capacity > 0:
                    # We have capacity, so add this target
                    target = AttentionTarget(
                        target_id=item_id,
                        target_type=item["type"],
                        description=item["description"],
                        activation=salience
                    )
                    
                    success = self.current_focus.add_target(target)
                    if success:
                        remaining_capacity -= 1
                        added_targets.append({
                            "target_id": item_id,
                            "activation": salience
                        })
                else:
                    # At capacity, so consider replacing least activated target
                    least_active_id = min(self.current_focus.targets.items(), key=lambda x: x[1])[0]
                    least_activation = self.current_focus.targets[least_active_id]
                    
                    # Only replace if new item is significantly more salient
                    if salience > least_activation * 1.5:
                        # Remove least active
                        self.current_focus.remove_target(least_active_id)
                        removed_targets.append({
                            "target_id": least_active_id,
                            "activation": least_activation
                        })
                        
                        # Add new target
                        target = AttentionTarget(
                            target_id=item_id,
                            target_type=item["type"],
                            description=item["description"],
                            activation=salience
                        )
                        
                        success = self.current_focus.add_target(target)
                        if success:
                            added_targets.append({
                                "target_id": item_id,
                                "activation": salience
                            })
        
        # Record focus shift in history
        self._record_focus_shift(added_targets, removed_targets, updated_targets)
        
        # If any changes occurred, publish an event
        if added_targets or removed_targets or updated_targets:
            self.publish_message("attention_focus_updated", {
                "added": added_targets,
                "removed": removed_targets,
                "updated": updated_targets,
                "current_focus": self.current_focus.model_dump()
            })
        
        return {
            "status": "success",
            "added": added_targets,
            "removed": removed_targets,
            "updated": updated_targets,
            "current_focus": self.current_focus.model_dump()
        }
    
    def add_attention_target(
        self, 
        target_id: str, 
        target_type: str, 
        activation: float = 1.0,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Explicitly add a target to attention focus
        
        Parameters:
        target_id: ID of the target
        target_type: Type of the target
        activation: Initial activation level
        description: Description of the target
        
        Returns:
        Operation result
        """
        # If already in focus, just update activation
        if target_id in self.current_focus.targets:
            prev_activation = self.current_focus.targets[target_id]
            target = self.current_focus.target_details[target_id]
            target.update_activation(activation - prev_activation)
            self.current_focus.targets[target_id] = target.activation
            
            self.publish_message("attention_target_updated", {
                "target_id": target_id,
                "prev_activation": prev_activation,
                "new_activation": target.activation
            })
            
            return {
                "status": "success",
                "operation": "updated",
                "target_id": target_id,
                "prev_activation": prev_activation,
                "new_activation": target.activation
            }
        
        # Create new target
        target = AttentionTarget(
            target_id=target_id,
            target_type=target_type,
            activation=activation,
            description=description
        )
        
        # Check if we're at capacity
        if self.current_focus.is_at_capacity:
            # Try to remove least active target
            least_active_id = min(self.current_focus.targets.items(), key=lambda x: x[1])[0]
            self.current_focus.remove_target(least_active_id)
            
            self.publish_message("attention_target_removed", {
                "target_id": least_active_id,
                "reason": "capacity_limit"
            })
        
        # Add the new target
        success = self.current_focus.add_target(target)
        
        if success:
            self.publish_message("attention_target_added", {
                "target_id": target_id,
                "target_type": target_type,
                "activation": activation
            })
            
            return {
                "status": "success",
                "operation": "added",
                "target_id": target_id,
                "activation": activation
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add target to attention focus"
            }
    
    def remove_attention_target(self, target_id: str) -> Dict[str, Any]:
        """
        Remove a target from attention focus
        
        Parameters:
        target_id: ID of the target to remove
        
        Returns:
        Operation result
        """
        if target_id in self.current_focus.targets:
            prev_activation = self.current_focus.targets[target_id]
            success = self.current_focus.remove_target(target_id)
            
            if success:
                self.publish_message("attention_target_removed", {
                    "target_id": target_id,
                    "prev_activation": prev_activation,
                    "reason": "explicit_request"
                })
                
                return {
                    "status": "success",
                    "target_id": target_id,
                    "prev_activation": prev_activation
                }
        
        return {
            "status": "error",
            "message": f"Target not in attention focus: {target_id}"
        }
    
    def shift_focus_to(self, target_id: str, priority: float = 0.5) -> Dict[str, Any]:
        """
        Shift focus to a specific target with given priority
        
        Parameters:
        target_id: ID of the target to focus on
        priority: How important this focus shift is (affects willingness to clear other targets)
        
        Returns:
        Operation result
        """
        # If high priority, clear other low-activation targets
        if priority > 0.7:
            # Remove all targets with activation below 0.5
            for tid, activation in list(self.current_focus.targets.items()):
                if activation < 0.5 and tid != target_id:
                    self.current_focus.remove_target(tid)
        
        # If target is already in focus, increase its activation
        if target_id in self.current_focus.targets:
            target = self.current_focus.target_details[target_id]
            prev_activation = target.activation
            
            # Set activation based on priority
            new_activation = max(target.activation, priority)
            target.update_activation(new_activation - prev_activation)
            self.current_focus.targets[target_id] = new_activation
            
            self.publish_message("attention_focus_shifted", {
                "target_id": target_id,
                "prev_activation": prev_activation,
                "new_activation": new_activation,
                "priority": priority
            })
            
            return {
                "status": "success",
                "operation": "enhanced",
                "target_id": target_id,
                "prev_activation": prev_activation,
                "new_activation": new_activation
            }
        else:
            # Target not in focus, so add it
            # If at capacity, make room by removing lowest activation target
            if self.current_focus.is_at_capacity:
                lowest_id = min(self.current_focus.targets.items(), key=lambda x: x[1])[0]
                self.current_focus.remove_target(lowest_id)
            
            # Create target with unknown type (will be updated when we get more info)
            target = AttentionTarget(
                target_id=target_id,
                target_type="unknown",
                activation=priority
            )
            
            success = self.current_focus.add_target(target)
            
            if success:
                self.publish_message("attention_focus_shifted", {
                    "target_id": target_id,
                    "activation": priority,
                    "priority": priority
                })
                
                return {
                    "status": "success",
                    "operation": "added",
                    "target_id": target_id,
                    "activation": priority
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to add target to attention"
                }
    
    def _update_state(self) -> None:
        """
        Update attention state based on time
        
        This handles time-based decay of attention.
        """
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds()
        self.last_update = now
        
        if time_delta <= 0:
            return
        
        # Apply decay to all targets
        removed_ids = self.current_focus.decay_all(self.parameters.decay_rate * time_delta)
        
        # Notify about removed targets
        for target_id in removed_ids:
            self.publish_message("attention_target_removed", {
                "target_id": target_id,
                "reason": "decay"
            })
    
    def _record_focus_shift(
        self, 
        added_targets: List[Dict[str, Any]], 
        removed_targets: List[Dict[str, Any]],
        updated_targets: List[Dict[str, Any]]
    ) -> None:
        """Record focus shift in history for learning attention patterns"""
        if not (added_targets or removed_targets or updated_targets):
            return
            
        # Create a focus shift record
        record = {
            "timestamp": datetime.now().isoformat(),
            "added": added_targets,
            "removed": removed_targets,
            "updated": updated_targets,
            "development_level": self.development_level
        }
        
        # Add to history
        self.focus_shift_history.append(record)
        
        # Trim history if needed
        if len(self.focus_shift_history) > self.max_history_size:
            self.focus_shift_history = self.focus_shift_history[-self.max_history_size:]
    
    # Event handlers
    
    def _handle_salience_detected(self, message: Message) -> None:
        """Handle salience detection events from the salience detector"""
        content = message.content
        salience_scores = content.get("salience_scores", {})
        
        if salience_scores:
            self.update_focus_from_salience(salience_scores)
    
    def _handle_executive_command(self, message: Message) -> None:
        """Handle commands from the executive module"""
        content = message.content
        command = content.get("command", "")
        
        if command == "focus_on":
            target_id = content.get("target_id", "")
            priority = content.get("priority", 0.8)  # Executive commands get high priority
            
            if target_id:
                self.shift_focus_to(target_id, priority)
                
        elif command == "clear_focus":
            # Clear all or specific targets
            target_ids = content.get("target_ids", [])
            
            if target_ids:
                # Clear specific targets
                for target_id in target_ids:
                    self.remove_attention_target(target_id)
            else:
                # Clear all targets
                for target_id in list(self.current_focus.targets.keys()):
                    self.remove_attention_target(target_id)
    
    def _handle_perception_input(self, message: Message) -> None:
        """Handle inputs from the perception module"""
        content = message.content
        perception_data = content.get("perception_data", {})
        
        # Check if this contains salience information
        salience_info = content.get("salience", {})
        
        if salience_info:
            self.update_focus_from_salience(salience_info)