# working_memory.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypeVar
import logging
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
import heapq
from enum import Enum, auto

# Import from memory manager
from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority

# Set up logging
logger = logging.getLogger("WorkingMemory")

class AttentionState(str, Enum):
    """States of attention for working memory items"""
    FOCUSED = "focused"       # Currently being actively processed
    ACTIVE = "active"         # In active memory but not the focus
    PERIPHERAL = "peripheral" # On the edge of consciousness
    FADING = "fading"         # About to be forgotten

class WorkingMemoryItem(BaseModel):
    """An item in working memory with attention metadata"""
    memory_id: str = Field(..., description="Reference to the item in memory manager")
    attention_state: AttentionState = Field(AttentionState.ACTIVE, description="Current attention state")
    activation_level: float = Field(1.0, ge=0.0, le=1.0, description="Current activation level")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Item priority for retention")
    entry_time: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(0, ge=0)
    
    def update_access(self) -> None:
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Boost activation level when accessed
        self.activation_level = min(1.0, self.activation_level + 0.2)
    
    def decay(self, rate: float = 0.1) -> None:
        """Apply decay based on time since last access"""
        # Calculate time-based decay
        time_since_access = (datetime.now() - self.last_accessed).total_seconds()
        decay_factor = min(0.95, rate * (time_since_access / 60.0))  # Scaled by minutes
        
        # Apply decay to activation
        self.activation_level *= (1.0 - decay_factor)
        
        # Update attention state based on activation level
        if self.activation_level < 0.2:
            self.attention_state = AttentionState.FADING
        elif self.activation_level < 0.5:
            self.attention_state = AttentionState.PERIPHERAL
        elif self.activation_level < 0.8:
            self.attention_state = AttentionState.ACTIVE
        else:
            self.attention_state = AttentionState.FOCUSED

class WorkingMemory:
    """Short-term active processing system"""
    
    def __init__(self, capacity: int = 7, decay_rate: float = 0.1):
        """Initialize working memory
        
        Args:
            capacity: Maximum items that can be held in working memory (Miller's 7±2)
            decay_rate: Rate at which items decay if not accessed
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.focus_of_attention: Optional[str] = None
        self.last_update = datetime.now()
        self.memory_manager: Optional[MemoryManager] = None
        
        # External network inputs
        self.attention_boost: Dict[str, float] = {}  # memory_id -> boost value
        
        logger.info(f"Working memory initialized with capacity {capacity}")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def add_item(self, memory_item: MemoryItem) -> bool:
        """Add an item to working memory
        
        If at capacity, least active item will be removed to make space.
        """
        if memory_item.id in self.items:
            # Already in working memory, just update
            self.items[memory_item.id].update_access()
            logger.debug(f"Updated existing item in working memory: {memory_item.id}")
            return True
        
        # Make room if needed
        if len(self.items) >= self.capacity:
            self._make_room()
        
        # Add the new item
        priority = memory_item.attributes.salience
        
        # Adjust priority based on emotional intensity
        priority += memory_item.attributes.emotional_intensity * 0.3
        
        working_item = WorkingMemoryItem(
            memory_id=memory_item.id,
            attention_state=AttentionState.ACTIVE,
            activation_level=1.0,  # New items start highly activated
            priority=min(1.0, priority)
        )
        
        self.items[memory_item.id] = working_item
        
        # Set as focus of attention if it's the most salient item
        if not self.focus_of_attention or priority > self.items.get(self.focus_of_attention, WorkingMemoryItem(memory_id="")).priority:
            self.focus_of_attention = memory_item.id
        
        logger.info(f"Added item to working memory: {memory_item.id}")
        return True
    
    def _make_room(self) -> None:
        """Remove the least active/important item to make room"""
        if not self.items:
            return
            
        # Calculate a combined score for each item (activation + priority)
        item_scores = {}
        for memory_id, item in self.items.items():
            # Score considers activation, priority, and recency
            recency_factor = 1.0 - min(1.0, (datetime.now() - item.last_accessed).total_seconds() / 300.0)  # 5 minute scale
            score = (item.activation_level * 0.5) + (item.priority * 0.3) + (recency_factor * 0.2)
            item_scores[memory_id] = score
        
        # Find the item with the lowest score
        if item_scores:
            to_remove = min(item_scores.keys(), key=lambda k: item_scores[k])
            
            # If this is the focus of attention, clear that reference
            if to_remove == self.focus_of_attention:
                self.focus_of_attention = None
            
            # Remove the item
            dropped_item = self.items.pop(to_remove)
            logger.info(f"Removed item from working memory to make room: {to_remove}")
            
            # Queue for potential consolidation if it was important enough
            if dropped_item.priority > 0.3 and self.memory_manager:
                logger.info(f"Marking removed item for potential consolidation: {to_remove}")
    
    def access_item(self, memory_id: str) -> bool:
        """Access an item in working memory"""
        if memory_id not in self.items:
            return False
        
        # Update the item's activation
        self.items[memory_id].update_access()
        
        # Set as new focus of attention
        self.focus_of_attention = memory_id
        
        logger.debug(f"Accessed item in working memory: {memory_id}")
        return True
    
    def update_item(self, memory_id: str, memory_item: Optional[MemoryItem] = None) -> bool:
        """Update an item in working memory"""
        if memory_id not in self.items:
            return False
        
        # Update access metadata
        self.items[memory_id].update_access()
        
        # Update priority if memory_item is provided
        if memory_item:
            priority = memory_item.attributes.salience
            priority += memory_item.attributes.emotional_intensity * 0.3
            self.items[memory_id].priority = min(1.0, priority)
        
        logger.debug(f"Updated item in working memory: {memory_id}")
        return True
    
    def remove_item(self, memory_id: str) -> bool:
        """Remove an item from working memory"""
        if memory_id not in self.items:
            return False
        
        # If this is the focus of attention, clear that reference
        if memory_id == self.focus_of_attention:
            self.focus_of_attention = None
        
        # Remove the item
        del self.items[memory_id]
        
        logger.info(f"Removed item from working memory: {memory_id}")
        return True
    
    def get_focus_of_attention(self) -> Optional[str]:
        """Get the current focus of attention"""
        return self.focus_of_attention
    
    def set_focus_of_attention(self, memory_id: str) -> bool:
        """Explicitly set the focus of attention"""
        if memory_id not in self.items:
            return False
        
        self.focus_of_attention = memory_id
        self.items[memory_id].update_access()
        self.items[memory_id].attention_state = AttentionState.FOCUSED
        
        logger.info(f"Set focus of attention to: {memory_id}")
        return True
    
    def receive_attention_input(self, memory_id: str, boost_value: float) -> None:
        """Receive attention boost from attention network"""
        self.attention_boost[memory_id] = boost_value
    
    def update(self) -> None:
        """Update working memory state
        
        This should be called periodically to apply decay, process
        attention inputs, and update focus.
        """
        current_time = datetime.now()
        elapsed_seconds = (current_time - self.last_update).total_seconds()
        
        # Don't update too frequently
        if elapsed_seconds < 1.0:
            return
        
        # Apply attention boosts
        for memory_id, boost in self.attention_boost.items():
            if memory_id in self.items:
                self.items[memory_id].activation_level = min(1.0, self.items[memory_id].activation_level + boost)
                
                # Update attention state based on new activation
                if self.items[memory_id].activation_level >= 0.8:
                    self.items[memory_id].attention_state = AttentionState.FOCUSED
                elif self.items[memory_id].activation_level >= 0.5:
                    self.items[memory_id].attention_state = AttentionState.ACTIVE
        
        # Clear attention boosts after applying
        self.attention_boost.clear()
        
        # Apply decay to all items
        for memory_id, item in list(self.items.items()):
            item.decay(self.decay_rate * (elapsed_seconds / 60.0))  # Scale by elapsed minutes
            
            # Remove items that have decayed too much
            if item.activation_level < 0.1:
                logger.info(f"Removing decayed item from working memory: {memory_id}")
                self.remove_item(memory_id)
        
        # Update focus of attention if needed
        if not self.focus_of_attention or self.focus_of_attention not in self.items:
            # Select new focus based on activation and priority
            highest_score = -1.0
            highest_id = None
            
            for memory_id, item in self.items.items():
                score = (item.activation_level * 0.7) + (item.priority * 0.3)
                if score > highest_score:
                    highest_score = score
                    highest_id = memory_id
            
            if highest_id:
                self.focus_of_attention = highest_id
                self.items[highest_id].attention_state = AttentionState.FOCUSED
        
        self.last_update = current_time
    
    def get_active_items(self, min_activation: float = 0.3) -> List[str]:
        """Get list of active items in working memory"""
        return [memory_id for memory_id, item in self.items.items() 
                if item.activation_level >= min_activation]
    
    def get_items_by_state(self, state: AttentionState) -> List[str]:
        """Get items in a specific attention state"""
        return [memory_id for memory_id, item in self.items.items()
                if item.attention_state == state]
    
    def get_consolidation_candidates(self) -> List[str]:
        """Get items that should be considered for consolidation to long-term memory"""
        candidates = []
        
        for memory_id, item in self.items.items():
            # Consider for consolidation if:
            # 1. Has been accessed multiple times
            # 2. Has been in working memory for a while
            # 3. Has significant priority
            time_in_memory = (datetime.now() - item.entry_time).total_seconds() / 60.0  # minutes
            
            if item.access_count >= 3 and time_in_memory > 5.0 and item.priority > 0.4:
                candidates.append(memory_id)
        
        return candidates
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of working memory"""
        return {
            "capacity": self.capacity,
            "used_capacity": len(self.items),
            "focus_of_attention": self.focus_of_attention,
            "items": {
                memory_id: {
                    "attention_state": item.attention_state,
                    "activation_level": item.activation_level,
                    "priority": item.priority,
                    "time_in_memory": (datetime.now() - item.entry_time).total_seconds() / 60.0,  # minutes
                    "access_count": item.access_count
                } for memory_id, item in self.items.items()
            }
        }