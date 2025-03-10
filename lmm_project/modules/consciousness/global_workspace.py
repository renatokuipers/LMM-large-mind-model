# TODO: Implement the GlobalWorkspace class based on Global Workspace Theory
# This component should serve as an integration point where:
# - Specialized cognitive modules compete for access
# - Information becomes broadly available to multiple systems
# - Serial conscious processing emerges from parallel unconscious processing
# - Broadcasting of information creates a unified conscious experience

# TODO: Implement development progression in the global workspace:
# - Simple integration of basic inputs in early stages
# - Expanded capacity and sophistication in later stages
# - Increasing selectivity in information broadcasting
# - Metacognitive access to workspace contents in advanced stages

# TODO: Create mechanisms for:
# - Competition for access: Determine which information enters consciousness
# - Information broadcasting: Share conscious information with multiple modules
# - Maintenance of conscious content: Keep information active over time
# - Attentional modulation: Prioritize information based on attention signals

# TODO: Implement variable conscious access levels:
# - Primary consciousness: Awareness of perceptions and emotions
# - Higher-order consciousness: Awareness of being aware (metacognition)

# TODO: Create workspace capacity limitations that are:
# - Developmentally appropriate (expanding with age)
# - Reflective of human cognitive limitations
# - Subject to attentional control

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.consciousness.models import WorkspaceState, GlobalWorkspaceItem
from lmm_project.modules.consciousness.neural_net import GlobalWorkspaceNetwork

class GlobalWorkspace(BaseModule):
    """
    Implements the Global Workspace Theory of consciousness
    
    This module serves as an integration and distribution center
    for information from multiple cognitive modules, determining
    what information becomes conscious and available to all modules.
    
    Developmental progression:
    - Basic information gathering in early stages
    - Simple information integration in childhood
    - Complex integration and distribution in adolescence
    - Sophisticated parallel processing in adulthood
    """
    
    # Developmental milestones for the global workspace
    development_milestones = {
        0.0: "information_gathering",    # Basic collection of information
        0.25: "simple_integration",      # Basic integration of related information
        0.5: "complex_integration",      # Multi-source information integration
        0.75: "parallel_processing",     # Multiple information streams
        0.9: "meta_workspace"            # Workspace can reflect on its own contents
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the global workspace
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="global_workspace", event_bus=event_bus)
        
        # Initialize workspace state
        self.state = WorkspaceState()
        
        # Configure capacity based on development level
        self._update_capacity()
        
        # Neural network for workspace processing
        self.input_dim = 128  # Default dimension
        self.network = GlobalWorkspaceNetwork(
            input_dim=self.input_dim,
            hidden_dim=256,
            output_dim=self.input_dim
        )
        
        # Track last update time for decay calculations
        self.last_update = datetime.now()
        
        # Subscribe to all relevant events if event bus is available
        if self.event_bus:
            # Listen for inputs from all cognitive modules
            self.event_bus.subscribe("awareness_state", self._handle_awareness)
            self.event_bus.subscribe("perception_processed", self._handle_perception)
            self.event_bus.subscribe("memory_retrieved", self._handle_memory)
            self.event_bus.subscribe("language_processed", self._handle_language)
            self.event_bus.subscribe("reasoning_result", self._handle_reasoning)
            self.event_bus.subscribe("emotion_state", self._handle_emotion)
            self.event_bus.subscribe("attention_focus", self._handle_attention)
            self.event_bus.subscribe("motor_state", self._handle_motor)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update the global workspace
        
        Args:
            input_data: Dictionary containing information to be processed
            
        Returns:
            Dictionary with the results of global workspace processing
        """
        # Extract inputs
        content = input_data.get("content", {})
        source_module = input_data.get("source", "unknown")
        activation = input_data.get("activation", 0.5)  # Default activation level
        decay_rate = input_data.get("decay_rate", 0.05)  # Default decay rate
        
        # Update time and decay existing workspace items
        self._decay_workspace_items()
        
        # Only process if the input has content
        if content:
            # Create a workspace item
            new_item = GlobalWorkspaceItem(
                content=content,
                source_module=source_module,
                activation_level=activation,
                decay_rate=decay_rate
            )
            
            # Add to workspace if it meets threshold or if workspace not at capacity
            if (activation >= self.state.competition_threshold or 
                    len(self.state.active_items) < self.state.capacity):
                self._add_to_workspace(new_item)
                
                # If we're over capacity, remove lowest activated item
                if len(self.state.active_items) > self.state.capacity:
                    self._remove_lowest_activated()
        
        # Create result with current workspace state
        result = {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "state": self.state.model_dump(),
            "developmental_level": self.developmental_level,
            "current_milestone": self._get_current_milestone()
        }
        
        # Broadcast workspace contents if event bus is available
        if self.event_bus:
            # Create a simplified version of the workspace for broadcasting
            broadcast = {
                "workspace_contents": {item_id: {
                    "content": item.content,
                    "source": item.source_module,
                    "activation": item.activation_level
                } for item_id, item in self.state.active_items.items()},
                "workspace_capacity": self.state.capacity,
                "developmental_level": self.developmental_level
            }
            
            self.event_bus.publish(
                msg_type="global_workspace_broadcast",
                content=broadcast
            )
        
        return result
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.developmental_level
        new_level = super().update_development(amount)
        
        # Update competition threshold based on development
        self.state.competition_threshold = 0.3 - 0.1 * new_level  # Lower threshold as development increases
        
        # Update capacity at key developmental milestones
        self._update_capacity()
        
        return new_level
    
    def _update_capacity(self) -> None:
        """Update the workspace capacity based on developmental level"""
        # Calculate new capacity (ranges from 3 to 9)
        base_capacity = 3
        dev_bonus = int(self.developmental_level * 6)  # 0 to 6 bonus capacity
        self.state.capacity = base_capacity + dev_bonus
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_workspace"
        for level, name in sorted(self.development_milestones.items()):
            if self.developmental_level >= level:
                milestone = name
        return milestone
    
    def _decay_workspace_items(self) -> None:
        """Decay the activation of workspace items over time"""
        current_time = datetime.now()
        time_delta = (current_time - self.last_update).total_seconds()
        self.last_update = current_time
        
        items_to_remove = []
        
        # Apply decay to each item
        for item_id, item in self.state.active_items.items():
            # Calculate decay based on time and item's decay rate
            decay_amount = item.decay_rate * time_delta
            
            # Update activation level
            item.activation_level = max(0.0, item.activation_level - decay_amount)
            
            # Mark for removal if activation is too low
            if item.activation_level < 0.1:  # Threshold for removal
                items_to_remove.append(item_id)
        
        # Remove items with low activation
        for item_id in items_to_remove:
            self.state.active_items.pop(item_id, None)
    
    def _add_to_workspace(self, item: GlobalWorkspaceItem) -> None:
        """Add an item to the workspace, merging with similar items if needed"""
        # Check for similar items first
        similar_item_id = self._find_similar_item(item)
        
        if similar_item_id:
            # Update existing item instead of adding a new one
            existing_item = self.state.active_items[similar_item_id]
            
            # Increase activation (capped at 1.0)
            existing_item.activation_level = min(1.0, existing_item.activation_level + 0.2)
            
            # Update with new content (simplified merge)
            # In a more sophisticated version, this would do a deep merge of content
            existing_item.content.update(item.content)
            
            # Update timestamp
            existing_item.timestamp = datetime.now()
            
        else:
            # Add as a new item
            self.state.active_items[item.item_id] = item
    
    def _find_similar_item(self, item: GlobalWorkspaceItem) -> Optional[str]:
        """Find an existing workspace item similar to the given item"""
        # This is a simplified implementation
        # A more sophisticated version would use vector representations and similarity
        
        # Simple matching based on source and content keys
        for existing_id, existing_item in self.state.active_items.items():
            if existing_item.source_module == item.source_module:
                # Check content overlap
                if set(existing_item.content.keys()) & set(item.content.keys()):
                    return existing_id
        
        return None
    
    def _remove_lowest_activated(self) -> None:
        """Remove the item with the lowest activation level"""
        if not self.state.active_items:
            return
            
        # Find the item with the lowest activation
        lowest_id = min(
            self.state.active_items.keys(),
            key=lambda k: self.state.active_items[k].activation_level
        )
        
        # Remove it
        self.state.active_items.pop(lowest_id, None)
    
    def _integrate_workspace_contents(self) -> Dict[str, Any]:
        """Integrate the contents of the workspace into a unified representation"""
        # This would use neural networks in a sophisticated implementation
        # For now, we'll use a simplified approach
        
        integrated = {}
        sources = set()
        
        # Combine contents weighted by activation level
        for item in self.state.active_items.values():
            # Add source to the list of contributing sources
            sources.add(item.source_module)
            
            # Extract content with activation scaling
            for key, value in item.content.items():
                # Skip if the value is None
                if value is None:
                    continue
                    
                if key in integrated:
                    # Average with existing value, weighted by activation
                    if isinstance(value, (int, float)) and isinstance(integrated[key], (int, float)):
                        integrated[key] = (integrated[key] + value * item.activation_level) / (1 + item.activation_level)
                    else:
                        # For non-numeric types, use the one with higher activation
                        pass  # Keep the existing one for now
                else:
                    # Initial value
                    integrated[key] = value
        
        # Add metadata about the integration
        result = {
            "integrated_content": integrated,
            "contributing_sources": list(sources),
            "integration_level": min(1.0, 0.3 + 0.7 * self.developmental_level)
        }
        
        return result
    
    def _handle_awareness(self, message: Message) -> None:
        """Handle awareness state messages"""
        if isinstance(message.content, dict) and "state" in message.content:
            self.process_input({
                "content": message.content["state"],
                "source": "awareness",
                "activation": 0.7  # Awareness typically has high activation
            })
    
    def _handle_perception(self, message: Message) -> None:
        """Handle processed perception messages"""
        self.process_input({
            "content": message.content,
            "source": "perception",
            "activation": 0.6  # Perception typically has high activation
        })
    
    def _handle_memory(self, message: Message) -> None:
        """Handle memory retrieval messages"""
        self.process_input({
            "content": message.content,
            "source": "memory",
            "activation": 0.5  # Moderate activation for memory
        })
    
    def _handle_language(self, message: Message) -> None:
        """Handle language processing messages"""
        self.process_input({
            "content": message.content,
            "source": "language",
            "activation": 0.6  # High activation for language
        })
    
    def _handle_reasoning(self, message: Message) -> None:
        """Handle reasoning result messages"""
        self.process_input({
            "content": message.content,
            "source": "reasoning",
            "activation": 0.7  # High activation for reasoning
        })
    
    def _handle_emotion(self, message: Message) -> None:
        """Handle emotion state messages"""
        # Emotions have variable activation based on intensity
        intensity = 0.5  # Default intensity
        if isinstance(message.content, dict) and "intensity" in message.content:
            intensity = message.content["intensity"]
        
        self.process_input({
            "content": message.content,
            "source": "emotion",
            "activation": intensity
        })
    
    def _handle_attention(self, message: Message) -> None:
        """Handle attention focus messages"""
        self.process_input({
            "content": message.content,
            "source": "attention",
            "activation": 0.8  # Very high activation for attention
        })
    
    def _handle_motor(self, message: Message) -> None:
        """Handle motor state messages"""
        self.process_input({
            "content": message.content,
            "source": "motor",
            "activation": 0.4  # Lower activation for motor information
        }) 
