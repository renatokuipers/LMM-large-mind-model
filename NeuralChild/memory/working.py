"""
Working memory implementation for the NeuralChild system.
Handles short-term active information processing with limited capacity.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Generic, TypeVar, Tuple, Set
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from pydantic import ValidationError

from ..models.memory_models import (
    MemoryItem, WorkingMemory, WorkingMemoryConfig, MemoryType, 
    MemoryAttributes, MemoryAccessibility, MemoryStage
)
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic memory content
T = TypeVar('T')

class WorkingMemorySystem(Generic[T]):
    """
    Working memory system that holds and manages currently active information.
    Implements limited capacity, decay over time, and prioritization.
    """
    
    def __init__(
        self,
        model: Optional[WorkingMemory] = None,
        config_override: Optional[WorkingMemoryConfig] = None
    ):
        """
        Initialize the working memory system.
        
        Args:
            model: Optional existing WorkingMemory model
            config_override: Optional configuration override
        """
        if model is None:
            config_params = config.MEMORY["working_memory"]
            memory_config = config_override or WorkingMemoryConfig(
                capacity=config_params["capacity"],
                decay_rate=config_params["decay_rate"],
                max_duration=config_params["max_duration_seconds"],
                attention_boost=config_params["attention_boost"]
            )
            self.model = WorkingMemory(config=memory_config)
        else:
            self.model = model
            
        self._last_update = time.time()
        self._focused_items: Set[UUID] = set()
        logger.debug(f"Initialized working memory with capacity {self.model.config.capacity}")
    
    @property
    def capacity(self) -> int:
        """Get the working memory capacity"""
        return self.model.config.capacity
    
    @property
    def current_load(self) -> int:
        """Get the current number of items in working memory"""
        return self.model.current_load
    
    @property
    def items(self) -> Dict[UUID, MemoryItem]:
        """Get all items currently in working memory"""
        return self.model.items
    
    def add_item(self, content: T, memory_type: MemoryType = MemoryType.WORKING,
                attributes: Optional[MemoryAttributes] = None,
                tags: List[str] = None) -> UUID:
        """
        Add an item to working memory.
        
        Args:
            content: The content to store in working memory
            memory_type: Type of memory
            attributes: Optional memory attributes
            tags: Optional tags for categorization
            
        Returns:
            ID of the newly created memory item
        """
        # Create memory item
        item = MemoryItem(
            memory_type=memory_type,
            attributes=attributes or MemoryAttributes(
                accessibility=MemoryAccessibility.EASY,
                stage=MemoryStage.ENCODING
            ),
            content=content,
            tags=tags or []
        )
        
        # Add to working memory
        success = self.model.add_item(item)
        if success:
            logger.debug(f"Added item to working memory: {item.id}")
            return item.id
        else:
            logger.warning(f"Failed to add item to working memory")
            return None
    
    def get_item(self, item_id: UUID) -> Optional[MemoryItem[T]]:
        """
        Retrieve an item from working memory.
        Refreshes the item's access time.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        return self.model.get_item(item_id)
    
    def remove_item(self, item_id: UUID) -> bool:
        """
        Remove an item from working memory.
        
        Args:
            item_id: ID of the item to remove
            
        Returns:
            True if the item was removed, False otherwise
        """
        if item_id in self._focused_items:
            self._focused_items.remove(item_id)
        return self.model.remove_item(item_id)
    
    def set_focus(self, item_id: UUID, is_focused: bool = True) -> None:
        """
        Focus or unfocus attention on a specific item.
        Focused items decay more slowly.
        
        Args:
            item_id: ID of the item to focus on
            is_focused: Whether to focus or unfocus
        """
        if item_id in self.model.items:
            if is_focused and item_id not in self._focused_items:
                self._focused_items.add(item_id)
                logger.debug(f"Focus set on item: {item_id}")
            elif not is_focused and item_id in self._focused_items:
                self._focused_items.remove(item_id)
                logger.debug(f"Focus removed from item: {item_id}")
    
    def get_items_by_tags(self, tags: List[str], require_all: bool = False) -> List[MemoryItem[T]]:
        """
        Get items that match the given tags.
        
        Args:
            tags: List of tags to match
            require_all: If True, items must have all tags; if False, any tag matches
            
        Returns:
            List of matching memory items
        """
        result = []
        
        for item in self.model.items.values():
            if require_all:
                if all(tag in item.tags for tag in tags):
                    result.append(item)
            else:
                if any(tag in item.tags for tag in tags):
                    result.append(item)
        
        return result
    
    def update(self, elapsed_seconds: Optional[float] = None) -> List[UUID]:
        """
        Update working memory, handling decay and cleanup.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
        
        Returns:
            List of IDs of items that were removed due to decay
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds = current_time - self._last_update
            self._last_update = current_time
        
        # Process decay
        removed_items = self.model.update_decay(elapsed_seconds)
        
        # Apply attention boost to focused items
        for item_id in self._focused_items:
            if item_id in self.model.items:
                # Reset the access time to now
                self.model.items[item_id].attributes.last_accessed = datetime.now()
        
        return removed_items
    
    def clear(self) -> None:
        """Clear all items from working memory"""
        self.model.items.clear()
        self._focused_items.clear()
        self.model.update_metrics()
        logger.debug("Working memory cleared")
    
    def get_most_recent_items(self, count: int = 3) -> List[MemoryItem[T]]:
        """
        Get the most recently accessed items.
        
        Args:
            count: Maximum number of items to return
            
        Returns:
            List of the most recently accessed items
        """
        sorted_items = sorted(
            self.model.items.values(),
            key=lambda x: x.attributes.last_accessed,
            reverse=True
        )
        return sorted_items[:min(count, len(sorted_items))]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the working memory to a dictionary for serialization"""
        return self.model.model_dump()