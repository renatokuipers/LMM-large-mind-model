"""
Episodic memory implementation for the NeuralChild system.
Handles event-based memories structured as episodes.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Generic, TypeVar, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from pydantic import ValidationError

from ..models.memory_models import (
    MemoryItem, EpisodicMemory, EpisodicMemoryConfig, Episode,
    MemoryType, MemoryAttributes, MemoryAccessibility, MemoryStage
)
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic memory content
T = TypeVar('T')

class EpisodicMemorySystem:
    """
    Episodic memory system that organizes memories into coherent episodes.
    Supports event sequences, contextual binding, and emotional weighting.
    """
    
    def __init__(
        self,
        model: Optional[EpisodicMemory] = None,
        config_override: Optional[EpisodicMemoryConfig] = None
    ):
        """
        Initialize the episodic memory system.
        
        Args:
            model: Optional existing EpisodicMemory model
            config_override: Optional configuration override
        """
        if model is None:
            config_params = config.MEMORY["episodic_memory"]
            memory_config = config_override or EpisodicMemoryConfig(
                consolidation_threshold=config_params["consolidation_threshold"],
                emotional_weight=config_params["emotional_weight"],
                decay_rate=config_params["decay_rate"],
                coherence_factor=config_params["coherence_factor"]
            )
            self.model = EpisodicMemory(config=memory_config)
        else:
            self.model = model
            
        self._last_update = time.time()
        self._memory_items: Dict[UUID, MemoryItem] = {}  # Local cache of memory items
        self._current_episode_summary = ""
        logger.debug("Initialized episodic memory system")
    
    @property
    def current_episode_id(self) -> Optional[UUID]:
        """Get the ID of the currently active episode"""
        return self.model.current_episode
    
    @property
    def episodes(self) -> Dict[UUID, Episode]:
        """Get all episodes in episodic memory"""
        return self.model.episodes
    
    def create_episode(self, title: str, location: Optional[str] = None,
                      people_involved: List[str] = None) -> UUID:
        """
        Create a new episode.
        
        Args:
            title: Brief description of the episode
            location: Optional location
            people_involved: Optional list of people involved
            
        Returns:
            ID of the newly created episode
        """
        episode_id = self.model.create_episode(
            title=title,
            location=location,
            people_involved=people_involved or []
        )
        logger.debug(f"Created new episode: {title} (ID: {episode_id})")
        return episode_id
    
    def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """
        Retrieve an episode, updating its access time.
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            The episode if found, None otherwise
        """
        return self.model.get_episode(episode_id)
    
    def end_episode(self, episode_id: Optional[UUID] = None) -> bool:
        """
        Mark an episode as ended.
        
        Args:
            episode_id: ID of the episode to end, or None for current episode
            
        Returns:
            True if the episode was ended, False otherwise
        """
        episode_id = episode_id or self.model.current_episode
        
        if episode_id is None:
            logger.warning("No current episode to end")
            return False
        
        # Generate summary for the episode
        if episode_id in self.model.episodes:
            episode = self.model.episodes[episode_id]
            
            # Update the end time
            result = self.model.end_episode(episode_id)
            
            # Clear the current episode summary
            self._current_episode_summary = ""
            
            logger.debug(f"Ended episode: {episode.title} (ID: {episode_id})")
            return result
        
        return False
    
    def add_memory_to_episode(self, memory_item: MemoryItem, 
                             episode_id: Optional[UUID] = None) -> bool:
        """
        Add a memory item to an episode.
        
        Args:
            memory_item: The memory item to add
            episode_id: Optional episode ID, defaults to current episode
            
        Returns:
            True if the memory was added, False otherwise
        """
        episode_id = episode_id or self.model.current_episode
        
        if episode_id is None:
            logger.warning("No current episode to add memory to")
            return False
        
        # Store the memory item in our local cache
        self._memory_items[memory_item.id] = memory_item
        
        # Add to the episode
        result = self.model.add_to_episode(episode_id, memory_item.id)
        
        # Update episode summary
        if result and episode_id == self.model.current_episode:
            self._update_episode_summary(memory_item)
        
        return result
    
    def update_episode_emotion(self, emotion: str, intensity: float,
                              episode_id: Optional[UUID] = None) -> bool:
        """
        Update the emotional content of an episode.
        
        Args:
            emotion: Emotion name
            intensity: Intensity of the emotion (0.0 to 1.0)
            episode_id: Optional episode ID, defaults to current episode
            
        Returns:
            True if the emotion was updated, False otherwise
        """
        episode_id = episode_id or self.model.current_episode
        
        if episode_id is None:
            logger.warning("No current episode to update emotions for")
            return False
        
        return self.model.update_episode_emotion(episode_id, emotion, intensity)
    
    def get_episode_memories(self, episode_id: UUID) -> List[MemoryItem]:
        """
        Get all memory items in an episode.
        
        Args:
            episode_id: ID of the episode
            
        Returns:
            List of memory items in the episode
        """
        if episode_id not in self.model.episodes:
            logger.warning(f"Episode not found: {episode_id}")
            return []
        
        episode = self.model.episodes[episode_id]
        result = []
        
        for memory_id in episode.memory_items:
            if memory_id in self._memory_items:
                result.append(self._memory_items[memory_id])
        
        return result
    
    def _update_episode_summary(self, memory_item: MemoryItem) -> None:
        """
        Update the internal summary of the current episode.
        
        Args:
            memory_item: New memory item to incorporate in the summary
        """
        # Simple implementation for now - in a real system this would be more sophisticated
        if hasattr(memory_item.content, "__str__"):
            content_str = str(memory_item.content)
            short_content = content_str[:50] + "..." if len(content_str) > 50 else content_str
            
            if self._current_episode_summary:
                self._current_episode_summary += f" → {short_content}"
            else:
                self._current_episode_summary = short_content
    
    def get_episodes_by_timeframe(self, start_time: datetime, 
                                 end_time: Optional[datetime] = None) -> List[Episode]:
        """
        Get episodes that occurred within a specified timeframe.
        
        Args:
            start_time: Start of the timeframe
            end_time: Optional end of the timeframe, defaults to now
            
        Returns:
            List of episodes within the timeframe
        """
        end_time = end_time or datetime.now()
        
        return [episode for episode in self.model.episodes.values()
                if episode.start_time >= start_time and 
                (episode.end_time is None or episode.end_time <= end_time)]
    
    def get_episodes_by_location(self, location: str) -> List[Episode]:
        """
        Get episodes that occurred at a specific location.
        
        Args:
            location: Location to search for
            
        Returns:
            List of episodes at the location
        """
        return [episode for episode in self.model.episodes.values()
                if episode.location and episode.location.lower() == location.lower()]
    
    def get_episodes_with_person(self, person: str) -> List[Episode]:
        """
        Get episodes involving a specific person.
        
        Args:
            person: Person to search for
            
        Returns:
            List of episodes involving the person
        """
        return [episode for episode in self.model.episodes.values()
                if person.lower() in [p.lower() for p in episode.people_involved]]
    
    def update_decay(self, elapsed_seconds: float) -> None:
        """
        Update episodic memory decay based on elapsed time.
        
        Args:
            elapsed_seconds: Time elapsed since last update
        """
        decay_rate = self.model.config.decay_rate * elapsed_seconds
        
        # Apply decay to episode accessibility
        for episode in self.model.episodes.values():
            # Emotional memories decay more slowly
            emotional_factor = 1.0
            if episode.emotional_summary:
                # Calculate a factor based on emotional intensity
                total_emotional_intensity = sum(episode.emotional_summary.values())
                if total_emotional_intensity > 0:
                    emotional_factor = max(0.5, 1.0 - (total_emotional_intensity / len(episode.emotional_summary)) * self.model.config.emotional_weight)
            
            # Apply decay with emotional modulation
            new_accessibility = episode.accessibility * (1.0 - decay_rate * emotional_factor)
            episode.accessibility = max(0.1, new_accessibility)  # Never go below 0.1
    
    def update(self, elapsed_seconds: Optional[float] = None) -> None:
        """
        Update episodic memory, handling decay and other time-based processes.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds = current_time - self._last_update
            self._last_update = current_time
        
        # Update decay
        self.update_decay(elapsed_seconds)
    
    def get_current_episode_summary(self) -> str:
        """
        Get a summary of the current episode.
        
        Returns:
            String summary of the current episode
        """
        if self.model.current_episode is None:
            return "No active episode"
        
        episode = self.model.episodes[self.model.current_episode]
        
        summary = f"Episode: {episode.title}"
        if episode.location:
            summary += f" at {episode.location}"
        if episode.people_involved:
            summary += f" with {', '.join(episode.people_involved)}"
        
        # Add emotional summary if available
        if episode.emotional_summary:
            top_emotions = sorted(
                episode.emotional_summary.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            if top_emotions:
                summary += f" | Emotions: {', '.join(f'{emotion} ({intensity:.1f})' for emotion, intensity in top_emotions)}"
        
        # Add memory content summary
        if self._current_episode_summary:
            summary += f" | Content: {self._current_episode_summary}"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the episodic memory to a dictionary for serialization"""
        return self.model.model_dump()