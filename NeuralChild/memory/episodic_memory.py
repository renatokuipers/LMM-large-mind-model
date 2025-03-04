# episodic_memory.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from pathlib import Path
import json
import os
import random
from collections import deque

from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority

logger = logging.getLogger("EpisodicMemory")

class Episode:
    """A single episodic memory event"""
    def __init__(
        self,
        content: Any,
        timestamp: Optional[datetime] = None,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.0,
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.emotional_valence = emotional_valence
        self.emotional_intensity = emotional_intensity
        self.importance = importance
        self.context = context or {}
        self.recall_count = 0
        self.last_recalled = self.timestamp
        self.memory_strength = 0.8  # Initial strength
    
    def recall(self) -> None:
        """Record a recall of this memory, strengthening it"""
        self.recall_count += 1
        self.last_recalled = datetime.now()
        
        # Strengthen memory with diminishing returns
        self.memory_strength = min(1.0, self.memory_strength + (0.05 / (1 + 0.1 * self.recall_count)))
    
    def decay(self, rate: float = 0.01) -> None:
        """Apply decay based on time and emotional factors"""
        # Emotional memories decay more slowly
        if self.emotional_intensity > 0.7:
            rate *= 0.5
        elif self.emotional_intensity > 0.4:
            rate *= 0.8
        
        # Important memories decay more slowly
        if self.importance > 0.7:
            rate *= 0.6
        
        # Apply decay
        self.memory_strength *= (1.0 - rate)

class EpisodicMemory:
    """Manages episodic memory events"""
    
    def __init__(self, data_dir: Optional[Path] = None, max_episodes: int = 1000):
        """Initialize episodic memory"""
        self.data_dir = data_dir or Path("./data/memory/episodic")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes: Dict[str, Episode] = {}
        self.max_episodes = max_episodes
        self.temporal_index: Dict[str, List[str]] = {}  # date -> episode_ids
        self.emotional_index: Dict[str, Set[str]] = {  # emotion -> episode_ids
            "positive": set(),
            "negative": set(),
            "neutral": set()
        }
        self.importance_index: Dict[str, Set[str]] = {  # importance -> episode_ids
            "high": set(),
            "medium": set(),
            "low": set()
        }
        self.memory_manager: Optional[MemoryManager] = None
        
        # Recent episodic buffer
        self.recent_episodes = deque(maxlen=20)
        
        logger.info(f"Episodic memory initialized with capacity {max_episodes}")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def add_memory(self, memory_item: MemoryItem) -> str:
        """Add a new episodic memory"""
        episode_id = memory_item.id
        
        # Create episode
        episode = Episode(
            content=memory_item.content,
            timestamp=memory_item.attributes.created_at,
            emotional_valence=memory_item.attributes.emotional_valence,
            emotional_intensity=memory_item.attributes.emotional_intensity,
            importance=memory_item.attributes.salience
        )
        
        # Store the episode
        self.episodes[episode_id] = episode
        
        # Update recent buffer
        self.recent_episodes.append(episode_id)
        
        # Update indices
        self._update_indices(episode_id, episode)
        
        # Check capacity
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
        
        logger.info(f"Added episodic memory: {episode_id}")
        return episode_id
    
    def _update_indices(self, episode_id: str, episode: Episode) -> None:
        """Update memory indices"""
        # Update temporal index
        date_str = episode.timestamp.strftime("%Y-%m-%d")
        if date_str not in self.temporal_index:
            self.temporal_index[date_str] = []
        self.temporal_index[date_str].append(episode_id)
        
        # Update emotional index
        if episode.emotional_valence > 0.3:
            self.emotional_index["positive"].add(episode_id)
        elif episode.emotional_valence < -0.3:
            self.emotional_index["negative"].add(episode_id)
        else:
            self.emotional_index["neutral"].add(episode_id)
        
        # Update importance index
        if episode.importance > 0.7:
            self.importance_index["high"].add(episode_id)
        elif episode.importance > 0.3:
            self.importance_index["medium"].add(episode_id)
        else:
            self.importance_index["low"].add(episode_id)
    
    def _prune_episodes(self) -> None:
        """Remove least important episodes to stay within capacity"""
        # Calculate pruning score (lower is more likely to be pruned)
        pruning_scores = {}
        for episode_id, episode in self.episodes.items():
            # Score based on recency, emotional intensity, importance, and recall count
            recency_factor = min(1.0, (datetime.now() - episode.timestamp).days / 30)
            score = (episode.memory_strength * 0.3 + 
                    episode.importance * 0.3 + 
                    episode.emotional_intensity * 0.2 + 
                    (episode.recall_count / 10) * 0.2) * (1 - recency_factor * 0.5)
            
            pruning_scores[episode_id] = score
        
        # Sort by score (ascending)
        to_prune = sorted(pruning_scores.keys(), key=lambda k: pruning_scores[k])
        
        # Remove lowest scoring episodes until we're under capacity
        num_to_prune = len(self.episodes) - self.max_episodes
        for i in range(min(num_to_prune, len(to_prune))):
            episode_id = to_prune[i]
            self._remove_episode(episode_id)
            logger.info(f"Pruned episodic memory due to capacity: {episode_id}")
    
    def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode and update indices"""
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # Remove from temporal index
        date_str = episode.timestamp.strftime("%Y-%m-%d")
        if date_str in self.temporal_index and episode_id in self.temporal_index[date_str]:
            self.temporal_index[date_str].remove(episode_id)
        
        # Remove from emotional index
        if episode.emotional_valence > 0.3:
            self.emotional_index["positive"].discard(episode_id)
        elif episode.emotional_valence < -0.3:
            self.emotional_index["negative"].discard(episode_id)
        else:
            self.emotional_index["neutral"].discard(episode_id)
        
        # Remove from importance index
        if episode.importance > 0.7:
            self.importance_index["high"].discard(episode_id)
        elif episode.importance > 0.3:
            self.importance_index["medium"].discard(episode_id)
        else:
            self.importance_index["low"].discard(episode_id)
        
        # Remove from episodes
        del self.episodes[episode_id]
    
    def recall_memory(self, episode_id: str) -> Optional[Episode]:
        """Recall a specific memory by id"""
        if episode_id not in self.episodes:
            return None
        
        episode = self.episodes[episode_id]
        episode.recall()
        
        logger.debug(f"Recalled episode: {episode_id}")
        return episode
    
    def recall_by_date(self, date: datetime) -> List[str]:
        """Recall memories from a specific date"""
        date_str = date.strftime("%Y-%m-%d")
        return self.temporal_index.get(date_str, [])
    
    def recall_recent(self, count: int = 5) -> List[str]:
        """Recall the most recent episodes"""
        return list(self.recent_episodes)[-count:]
    
    def recall_emotional(self, valence: str = "positive", count: int = 5) -> List[str]:
        """Recall emotional memories of a given valence"""
        if valence not in self.emotional_index:
            return []
        
        # Sort by emotional intensity
        sorted_by_intensity = sorted(
            [(episode_id, self.episodes[episode_id].emotional_intensity) 
             for episode_id in self.emotional_index[valence]
             if episode_id in self.episodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [episode_id for episode_id, _ in sorted_by_intensity[:count]]
    
    def recall_important(self, count: int = 5) -> List[str]:
        """Recall the most important episodes"""
        # Get high importance episodes
        high_importance = sorted(
            [(episode_id, self.episodes[episode_id].importance)
             for episode_id in self.importance_index["high"]
             if episode_id in self.episodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        result = [episode_id for episode_id, _ in high_importance[:count]]
        
        # If we need more, get medium importance episodes
        if len(result) < count and self.importance_index["medium"]:
            medium_importance = sorted(
                [(episode_id, self.episodes[episode_id].importance)
                for episode_id in self.importance_index["medium"]
                if episode_id in self.episodes],
                key=lambda x: x[1],
                reverse=True
            )
            
            remaining = count - len(result)
            result.extend([episode_id for episode_id, _ in medium_importance[:remaining]])
        
        return result
    
    def consolidate_recent_episodes(self) -> Optional[str]:
        """Consolidate recent episodes into a summary episode"""
        # Need at least 3 recent episodes to consolidate
        if len(self.recent_episodes) < 3:
            return None
        
        # Get the episodes
        recent_ids = list(self.recent_episodes)
        recent_episodes = [self.episodes[episode_id] for episode_id in recent_ids
                          if episode_id in self.episodes]
        
        if len(recent_episodes) < 3:
            return None
        
        # Create a summary content
        earliest = min(recent_episodes, key=lambda e: e.timestamp)
        latest = max(recent_episodes, key=lambda e: e.timestamp)
        
        summary_content = {
            "summary": f"Consolidated memory of events from {earliest.timestamp.isoformat()} to {latest.timestamp.isoformat()}",
            "episode_count": len(recent_episodes),
            "duration_minutes": (latest.timestamp - earliest.timestamp).total_seconds() / 60,
            "component_ids": recent_ids
        }
        
        # Calculate aggregate emotional values
        avg_valence = sum(e.emotional_valence for e in recent_episodes) / len(recent_episodes)
        max_intensity = max(e.emotional_intensity for e in recent_episodes)
        max_importance = max(e.importance for e in recent_episodes)
        
        # Create consolidated episode
        if self.memory_manager:
            consolidated_id = self.memory_manager.store(
                content=summary_content,
                memory_type=MemoryType.EPISODIC,
                tags=["consolidated", "summary"],
                emotional_valence=avg_valence,
                emotional_intensity=max_intensity,
                salience=max_importance
            )
            
            # Link to component episodes
            for episode_id in recent_ids:
                self.memory_manager.associate(consolidated_id, episode_id, 0.9)
            
            logger.info(f"Consolidated {len(recent_episodes)} episodes into {consolidated_id}")
            return consolidated_id
        
        return None
    
    def update_memory(self, episode_id: str, memory_item: Optional[MemoryItem] = None) -> bool:
        """Update an episode"""
        if episode_id not in self.episodes:
            return False
        
        episode = self.episodes[episode_id]
        
        # Update from memory_item if provided
        if memory_item:
            updated = False
            
            if hasattr(memory_item, "content"):
                episode.content = memory_item.content
                updated = True
            
            if hasattr(memory_item.attributes, "emotional_valence"):
                episode.emotional_valence = memory_item.attributes.emotional_valence
                updated = True
            
            if hasattr(memory_item.attributes, "emotional_intensity"):
                episode.emotional_intensity = memory_item.attributes.emotional_intensity
                updated = True
            
            if hasattr(memory_item.attributes, "salience"):
                episode.importance = memory_item.attributes.salience
                updated = True
            
            if updated:
                # Reindex the episode
                self._update_indices(episode_id, episode)
                logger.info(f"Updated episode: {episode_id}")
                
            return updated
        
        return False
    
    def delete_memory(self, episode_id: str) -> bool:
        """Delete an episode"""
        if episode_id not in self.episodes:
            self._remove_episode(episode_id)
            logger.info(f"Deleted episode: {episode_id}")
            return True
        return False
    
    def apply_decay(self, rate: float = 0.01) -> None:
        """Apply decay to all episodes"""
        for episode in self.episodes.values():
            episode.decay(rate)
        
        logger.debug(f"Applied decay with rate {rate} to all episodes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory"""
        # Count by category
        emotional_counts = {
            "positive": len(self.emotional_index["positive"]),
            "negative": len(self.emotional_index["negative"]),
            "neutral": len(self.emotional_index["neutral"])
        }
        
        importance_counts = {
            "high": len(self.importance_index["high"]),
            "medium": len(self.importance_index["medium"]),
            "low": len(self.importance_index["low"])
        }
        
        # Calculate date range
        dates = list(self.temporal_index.keys())
        date_range = None
        if dates:
            first_date = min(dates)
            last_date = max(dates)
            date_range = (first_date, last_date)
        
        # Calculate average stats
        episodes = list(self.episodes.values())
        avg_recall_count = sum(e.recall_count for e in episodes) / max(1, len(episodes))
        avg_strength = sum(e.memory_strength for e in episodes) / max(1, len(episodes))
        
        return {
            "total_episodes": len(self.episodes),
            "capacity": self.max_episodes,
            "capacity_used": len(self.episodes) / self.max_episodes,
            "date_range": date_range,
            "emotional_counts": emotional_counts,
            "importance_counts": importance_counts,
            "avg_recall_count": avg_recall_count,
            "avg_memory_strength": avg_strength,
            "recent_count": len(self.recent_episodes)
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the state of episodic memory"""
        if filepath is None:
            filepath = self.data_dir / f"episodic_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        episodes_data = {}
        for episode_id, episode in self.episodes.items():
            # Handle content serialization
            content = episode.content
            if hasattr(content, "model_dump"):
                content = content.model_dump()
            elif not isinstance(content, (str, int, float, bool, list, dict, type(None))):
                content = str(content)
            
            episodes_data[episode_id] = {
                "content": content,
                "timestamp": episode.timestamp.isoformat(),
                "emotional_valence": episode.emotional_valence,
                "emotional_intensity": episode.emotional_intensity,
                "importance": episode.importance,
                "context": episode.context,
                "recall_count": episode.recall_count,
                "last_recalled": episode.last_recalled.isoformat(),
                "memory_strength": episode.memory_strength
            }
        
        state = {
            "episodes": episodes_data,
            "recent_episodes": list(self.recent_episodes),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "episode_count": len(self.episodes),
                "max_episodes": self.max_episodes
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Episodic memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of episodic memory"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.episodes.clear()
            self.recent_episodes.clear()
            self.temporal_index.clear()
            self.emotional_index = {"positive": set(), "negative": set(), "neutral": set()}
            self.importance_index = {"high": set(), "medium": set(), "low": set()}
            
            # Load episodes
            for episode_id, episode_data in state.get("episodes", {}).items():
                episode = Episode(
                    content=episode_data["content"],
                    timestamp=datetime.fromisoformat(episode_data["timestamp"]),
                    emotional_valence=episode_data["emotional_valence"],
                    emotional_intensity=episode_data["emotional_intensity"],
                    importance=episode_data["importance"],
                    context=episode_data.get("context", {})
                )
                episode.recall_count = episode_data["recall_count"]
                episode.last_recalled = datetime.fromisoformat(episode_data["last_recalled"])
                episode.memory_strength = episode_data["memory_strength"]
                
                # Store the episode
                self.episodes[episode_id] = episode
                
                # Update indices
                self._update_indices(episode_id, episode)
            
            # Load recent episodes
            for episode_id in state.get("recent_episodes", []):
                if episode_id in self.episodes:
                    self.recent_episodes.append(episode_id)
            
            # Load metadata
            if "metadata" in state and "max_episodes" in state["metadata"]:
                self.max_episodes = state["metadata"]["max_episodes"]
            
            logger.info(f"Loaded episodic memory state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading episodic memory state: {str(e)}")
            return False