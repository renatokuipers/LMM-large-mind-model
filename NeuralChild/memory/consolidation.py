"""
Memory consolidation processes for the NeuralChild system.
Handles moving memories between different memory systems over time.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from pydantic import ValidationError

from ..models.memory_models import (
    MemoryItem, ConsolidationSystem, ConsolidationConfig, ConsolidationTask,
    ConsolidationStatus, MemoryType, MemoryAttributes, MemoryStage
)
from . import working, episodic, long_term
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

class MemoryConsolidation:
    """
    Memory consolidation system that manages the movement of memories
    between working memory, episodic memory, and long-term memory.
    """
    
    def __init__(
        self,
        working_memory: working.WorkingMemorySystem,
        episodic_memory: episodic.EpisodicMemorySystem,
        long_term_memory: long_term.LongTermMemorySystem,
        model: Optional[ConsolidationSystem] = None,
        config_override: Optional[ConsolidationConfig] = None
    ):
        """
        Initialize the memory consolidation system.
        
        Args:
            working_memory: Working memory system
            episodic_memory: Episodic memory system
            long_term_memory: Long-term memory system
            model: Optional existing ConsolidationSystem model
            config_override: Optional configuration override
        """
        # Connect to memory systems
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.long_term_memory = long_term_memory
        
        # Initialize consolidation system
        if model is None:
            config_params = config.MEMORY["consolidation"]
            consolidation_config = config_override or ConsolidationConfig(
                working_to_episodic_threshold=config_params["working_to_episodic_threshold"],
                episodic_to_longterm_threshold=config_params["episodic_to_longterm_threshold"],
                consolidation_rate=config_params["consolidation_rate"],
                sleep_consolidation_boost=config_params["sleep_consolidation_boost"],
                emotional_consolidation_factor=config_params["emotional_consolidation_factor"],
                importance_weight=config_params["importance_weight"]
            )
            self.model = ConsolidationSystem(config=consolidation_config)
        else:
            self.model = model
            
        self._last_update = time.time()
        logger.debug("Initialized memory consolidation system")
    
    def start_sleep_consolidation(self) -> None:
        """Start sleep-based memory consolidation mode"""
        self.model.start_sleep_consolidation()
        logger.debug("Started sleep-based memory consolidation")
    
    def end_sleep_consolidation(self) -> None:
        """End sleep-based memory consolidation mode"""
        self.model.end_sleep_consolidation()
        logger.debug("Ended sleep-based memory consolidation")
    
    def create_working_to_episodic_task(self, item_id: UUID, priority: float = 0.5) -> UUID:
        """
        Create a task to consolidate a working memory item to episodic memory.
        
        Args:
            item_id: ID of the working memory item
            priority: Task priority (0.0 to 1.0)
            
        Returns:
            ID of the created task
        """
        return self.model.create_task(
            source_id=item_id,
            source_type="working",
            target_type="episodic",
            priority=priority
        )
    
    def create_episodic_to_longterm_task(self, episode_id: UUID, priority: float = 0.5) -> UUID:
        """
        Create a task to consolidate an episodic memory to long-term memory.
        
        Args:
            episode_id: ID of the episodic memory
            priority: Task priority (0.0 to 1.0)
            
        Returns:
            ID of the created task
        """
        return self.model.create_task(
            source_id=episode_id,
            source_type="episodic",
            target_type="longterm",
            priority=priority
        )
    
    def get_task_status(self, task_id: UUID) -> Optional[ConsolidationStatus]:
        """
        Get the status of a consolidation task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            The task status if found, None otherwise
        """
        return self.model.get_task_status(task_id)
    
    def process_working_to_episodic(self, task: ConsolidationTask) -> bool:
        """
        Process a working memory to episodic memory consolidation task.
        
        Args:
            task: The consolidation task to process
            
        Returns:
            True if the task was completed, False otherwise
        """
        # Retrieve the item from working memory
        memory_item = self.working_memory.get_item(task.source_id)
        if memory_item is None:
            logger.warning(f"Item {task.source_id} not found in working memory")
            return False
        
        # Update memory attributes for consolidation
        memory_item.attributes.stage = MemoryStage.CONSOLIDATION
        
        # If no current episode, create one
        if self.episodic_memory.current_episode_id is None:
            # Create a simple episode title based on the memory content
            title = f"Memory from {memory_item.attributes.created_at.strftime('%Y-%m-%d %H:%M')}"
            self.episodic_memory.create_episode(title)
        
        # Add the memory item to the current episode
        success = self.episodic_memory.add_memory_to_episode(memory_item)
        
        if success:
            # Update emotional content of the episode if the memory has emotional valence
            if memory_item.attributes.emotional_intensity > 0.2:
                self.episodic_memory.update_episode_emotion(
                    emotion=memory_item.attributes.emotional_valence.value,
                    intensity=memory_item.attributes.emotional_intensity
                )
            
            # Remove from working memory if consolidation is complete
            if task.progress >= 1.0:
                self.working_memory.remove_item(task.source_id)
                logger.debug(f"Consolidated item {task.source_id} from working to episodic memory")
                return True
        
        return False
    
    def process_episodic_to_longterm(self, task: ConsolidationTask) -> bool:
        """
        Process an episodic memory to long-term memory consolidation task.
        
        Args:
            task: The consolidation task to process
            
        Returns:
            True if the task was completed, False otherwise
        """
        # Retrieve the episode from episodic memory
        episode = self.episodic_memory.get_episode(task.source_id)
        if episode is None:
            logger.warning(f"Episode {task.source_id} not found in episodic memory")
            return False
        
        # Check if the episode has ended
        if episode.end_time is None:
            logger.warning(f"Cannot consolidate ongoing episode {task.source_id}")
            return False
        
        # Get memory items from the episode
        memory_items = self.episodic_memory.get_episode_memories(task.source_id)
        
        # Determine appropriate domains for each memory item
        domain_assignments = self._assign_domains_to_memories(memory_items)
        
        # Add memory items to long-term memory in appropriate domains
        for item_id, domain in domain_assignments.items():
            # Find the item in the list
            item = next((item for item in memory_items if item.id == item_id), None)
            if item:
                self.long_term_memory.add_item(item, domain)
        
        # Add the episode itself to long-term memory
        success = self.long_term_memory.add_episode(episode)
        
        if success and task.progress >= 1.0:
            # Episode stays in episodic memory but is marked as consolidated
            episode.is_consolidated = True
            logger.debug(f"Consolidated episode {task.source_id} to long-term memory")
            return True
        
        return False
    
    def _assign_domains_to_memories(self, memory_items: List[MemoryItem]) -> Dict[UUID, Any]:
        """
        Assign appropriate long-term memory domains to memory items.
        
        Args:
            memory_items: List of memory items to assign domains to
            
        Returns:
            Dictionary mapping memory item IDs to domain assignments
        """
        from ..models.memory_models import LongTermMemoryDomain
        
        result = {}
        
        for item in memory_items:
            # Simple rule-based domain assignment
            # In a real system, this would be more sophisticated
            if "personal" in item.tags or "self" in item.tags:
                domain = LongTermMemoryDomain.PERSONAL
            elif "social" in item.tags or any(tag.startswith("person:") for tag in item.tags):
                domain = LongTermMemoryDomain.SOCIAL
            elif "skill" in item.tags or "action" in item.tags:
                domain = LongTermMemoryDomain.PROCEDURAL
            elif "fact" in item.tags or "knowledge" in item.tags:
                domain = LongTermMemoryDomain.DECLARATIVE
            elif item.attributes.emotional_intensity > 0.5:
                domain = LongTermMemoryDomain.EMOTIONAL
            elif "word" in item.tags or "language" in item.tags:
                domain = LongTermMemoryDomain.LINGUISTIC
            elif "value" in item.tags or "belief" in item.tags or "moral" in item.tags:
                domain = LongTermMemoryDomain.VALUES
            else:
                # Default to declarative memory
                domain = LongTermMemoryDomain.DECLARATIVE
            
            result[item.id] = domain
        
        return result
    
    def process_tasks(self, processing_time: float) -> List[UUID]:
        """
        Process consolidation tasks for a given amount of time.
        
        Args:
            processing_time: Time to spend processing (in seconds)
            
        Returns:
            List of IDs of completed tasks
        """
        # Process tasks in the model
        completed_task_ids = self.model.process_tasks(processing_time)
        
        # Execute actual consolidation for in-progress tasks
        for task_id, task in self.model.tasks.items():
            if task.status == ConsolidationStatus.IN_PROGRESS:
                # Execute the appropriate consolidation process
                if task.source_type == "working" and task.target_type == "episodic":
                    self.process_working_to_episodic(task)
                elif task.source_type == "episodic" and task.target_type == "longterm":
                    self.process_episodic_to_longterm(task)
        
        return completed_task_ids
    
    def check_for_consolidation_candidates(self) -> None:
        """
        Check all memory systems for items that should be consolidated
        and create appropriate consolidation tasks.
        """
        self._check_working_memory_for_consolidation()
        self._check_episodic_memory_for_consolidation()
    
    def _check_working_memory_for_consolidation(self) -> None:
        """Check working memory for items that should be consolidated to episodic memory"""
        threshold = self.model.config.working_to_episodic_threshold
        
        for item_id, item in self.working_memory.items.items():
            # Skip items that already have consolidation tasks
            if any(task.source_id == item_id and task.source_type == "working" 
                  for task in self.model.tasks.values()):
                continue
            
            # Check if the item meets the consolidation criteria
            importance = item.attributes.importance
            emotional_factor = item.attributes.emotional_intensity * self.model.config.emotional_consolidation_factor
            combined_score = importance * self.model.config.importance_weight + emotional_factor
            
            if combined_score >= threshold:
                # Calculate priority based on importance and emotional weight
                priority = min(1.0, combined_score)
                
                # Create consolidation task
                self.create_working_to_episodic_task(item_id, priority)
                logger.debug(f"Created consolidation task for working memory item {item_id} with priority {priority:.2f}")
    
    def _check_episodic_memory_for_consolidation(self) -> None:
        """Check episodic memory for episodes that should be consolidated to long-term memory"""
        threshold = self.model.config.episodic_to_longterm_threshold
        
        for episode_id, episode in self.episodic_memory.episodes.items():
            # Skip ongoing episodes
            if episode.end_time is None:
                continue
                
            # Skip episodes that already have consolidation tasks
            if any(task.source_id == episode_id and task.source_type == "episodic" 
                  for task in self.model.tasks.values()):
                continue
            
            # Skip already consolidated episodes
            if episode.is_consolidated:
                continue
            
            # Calculate episode importance
            importance = episode.importance
            
            # Calculate emotional weight
            emotional_factor = 0.0
            if episode.emotional_summary:
                emotional_intensities = list(episode.emotional_summary.values())
                avg_emotional_intensity = sum(emotional_intensities) / len(emotional_intensities)
                emotional_factor = avg_emotional_intensity * self.model.config.emotional_consolidation_factor
            
            # Combine factors
            combined_score = importance * self.model.config.importance_weight + emotional_factor
            
            if combined_score >= threshold:
                # Calculate priority based on importance and emotional weight
                priority = min(1.0, combined_score)
                
                # Create consolidation task
                self.create_episodic_to_longterm_task(episode_id, priority)
                logger.debug(f"Created consolidation task for episode {episode_id} with priority {priority:.2f}")
    
    def update(self, elapsed_seconds: Optional[float] = None) -> None:
        """
        Update memory consolidation, processing tasks and checking for new candidates.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds = current_time - self._last_update
            self._last_update = current_time
        
        # Check for new consolidation candidates
        self.check_for_consolidation_candidates()
        
        # Process existing tasks
        self.process_tasks(elapsed_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the consolidation system to a dictionary for serialization"""
        return self.model.model_dump()