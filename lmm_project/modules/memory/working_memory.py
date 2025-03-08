# Empty placeholder files 

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time
from datetime import datetime, timedelta
import numpy as np
import uuid

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.memory.models import WorkingMemoryItem

class WorkingMemory(BaseModule):
    """
    Working memory system with a limited capacity buffer
    
    Working memory provides a temporary storage mechanism for information
    that is currently being processed or attended to. It has limited capacity
    and information decays over time unless actively maintained.
    """
    # Maximum items that can be held in working memory
    max_capacity: int = Field(default=7)
    # Items in working memory
    items: Dict[str, WorkingMemoryItem] = Field(default_factory=dict)
    # Forgetting rate for non-rehearsed items (items/second)
    forgetting_rate: float = Field(default=0.05)
    # Last update timestamp
    last_update: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize working memory module"""
        super().__init__(
            module_id=module_id,
            module_type="working_memory",
            event_bus=event_bus,
            **data
        )
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("attention_focus", self._handle_attention_focus)
            self.subscribe_to_message("memory_retrieval", self._handle_memory_retrieval)
            self.subscribe_to_message("perception_input", self._handle_perception_input)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data by adding it to working memory
        
        Parameters:
        input_data: Dictionary containing input data
            - content: Content to add to working memory
            - importance: Optional importance value (0.0-1.0)
            - source_id: Optional source memory ID if retrieved from long-term memory
            
        Returns:
        Dictionary containing processed results
        """
        # Update working memory
        self._update_state()
        
        content = input_data.get("content", "")
        if not content:
            return {"status": "error", "message": "No content provided"}
            
        importance = input_data.get("importance", 0.5)
        source_id = input_data.get("source_id")
        
        # Create new working memory item
        item = WorkingMemoryItem(
            content=content,
            importance=importance,
            source_memory_id=source_id,
            buffer_position=len(self.items),
            activation_level=1.0,  # Start fully activated
            time_remaining=30.0    # Default 30 seconds
        )
        
        # Add to working memory
        self._add_item(item)
        
        # Publish event
        self.publish_message("working_memory_update", {
            "action": "add",
            "item_id": item.id,
            "content": item.content,
            "current_capacity": len(self.items),
            "max_capacity": self.max_capacity
        })
        
        return {
            "status": "success",
            "item_id": item.id,
            "current_capacity": len(self.items),
            "max_capacity": self.max_capacity
        }
    
    def update_development(self, amount: float) -> float:
        """
        Update working memory's developmental level
        
        As working memory develops:
        - Capacity increases
        - Forgetting rate decreases
        - Ability to maintain items improves
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development
        delta = self.development_level - prev_level
        
        # Increase capacity (from ~3 to ~7)
        capacity_increase = delta * 4
        self.max_capacity = min(7, self.max_capacity + capacity_increase)
        
        # Decrease forgetting rate
        forgetting_decrease = delta * 0.01
        self.forgetting_rate = max(0.01, self.forgetting_rate - forgetting_decrease)
        
        return self.development_level
    
    def get_items(self) -> List[WorkingMemoryItem]:
        """Get all items in working memory"""
        self._update_state()
        return list(self.items.values())
    
    def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Get a specific item from working memory"""
        self._update_state()
        return self.items.get(item_id)
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from working memory"""
        if item_id in self.items:
            del self.items[item_id]
            
            # Reindex buffer positions
            self._reindex_buffer_positions()
            
            # Publish event
            self.publish_message("working_memory_update", {
                "action": "remove",
                "item_id": item_id,
                "current_capacity": len(self.items),
                "max_capacity": self.max_capacity
            })
            
            return True
        return False
    
    def rehearse_item(self, item_id: str) -> bool:
        """
        Actively rehearse an item to keep it in working memory
        
        Parameters:
        item_id: ID of the item to rehearse
        
        Returns:
        Success status
        """
        self._update_state()
        
        if item_id in self.items:
            item = self.items[item_id]
            item.is_rehearsed = True
            item.time_remaining = 30.0  # Reset decay timer
            item.update_activation(0.2)  # Boost activation
            
            # Move to front of buffer
            self._move_to_front(item_id)
            
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from working memory"""
        self.items = {}
        
        # Publish event
        self.publish_message("working_memory_update", {
            "action": "clear",
            "current_capacity": 0,
            "max_capacity": self.max_capacity
        })
    
    def _add_item(self, item: WorkingMemoryItem) -> None:
        """
        Add an item to working memory, handling capacity constraints
        
        If working memory is full, the least active item is removed
        """
        # Check if we need to make room
        if len(self.items) >= self.max_capacity:
            self._remove_least_active_item()
        
        # Add the new item
        self.items[item.id] = item
        
        # Reindex buffer positions
        self._reindex_buffer_positions()
    
    def _remove_least_active_item(self) -> None:
        """Remove the least active item from working memory"""
        if not self.items:
            return
            
        # Find item with lowest activation
        least_active_id = min(
            self.items, 
            key=lambda i: (self.items[i].is_rehearsed, self.items[i].activation_level)
        )
        
        # Remove it
        if least_active_id:
            self.remove_item(least_active_id)
    
    def _update_state(self) -> None:
        """Update the state of working memory, handling time decay"""
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds()
        self.last_update = now
        
        if time_delta <= 0:
            return
            
        # List of items to remove (can't modify during iteration)
        to_remove = []
        
        for item_id, item in self.items.items():
            # Update time remaining
            if not item.is_rehearsed:
                item.time_remaining -= time_delta
                
                # Decay activation
                item.decay_activation(time_delta)
                
                # Mark for removal if time expired
                if item.time_remaining <= 0:
                    to_remove.append(item_id)
        
        # Remove expired items
        for item_id in to_remove:
            self.remove_item(item_id)
    
    def _reindex_buffer_positions(self) -> None:
        """Reindex buffer positions after items are added or removed"""
        sorted_items = sorted(
            self.items.values(),
            key=lambda item: (-item.activation_level, item.buffer_position)
        )
        
        for i, item in enumerate(sorted_items):
            item.buffer_position = i
    
    def _move_to_front(self, item_id: str) -> None:
        """Move an item to the front of the buffer (position 0)"""
        if item_id not in self.items:
            return
            
        # Set buffer position to -1 to ensure it will be at position 0
        # after reindexing
        self.items[item_id].buffer_position = -1
        
        # Reindex
        self._reindex_buffer_positions()
    
    # Handler methods for event bus
    
    def _handle_attention_focus(self, message: Message) -> None:
        """Handle attention focus events"""
        content = message.content
        target_id = content.get("target_id")
        
        if target_id and target_id in self.items:
            # Boost activation and move to front
            self.rehearse_item(target_id)
    
    def _handle_memory_retrieval(self, message: Message) -> None:
        """Handle memory retrieval events"""
        content = message.content
        memory_id = content.get("memory_id")
        memory_content = content.get("content")
        source_id = content.get("source_id")
        
        if memory_content:
            # Add retrieved memory to working memory
            self.process_input({
                "content": memory_content,
                "importance": content.get("importance", 0.7),
                "source_id": source_id
            })
    
    def _handle_perception_input(self, message: Message) -> None:
        """Handle perception input events"""
        content = message.content
        
        if "perception_data" in content:
            # Add perceived input to working memory
            self.process_input({
                "content": str(content["perception_data"]),
                "importance": content.get("salience", 0.5)
            }) 
