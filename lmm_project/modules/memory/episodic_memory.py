"""
Episodic Memory Module

This module implements episodic memory capabilities, storing temporally organized
experiences and events with their contextual details. Episodic memories include
information about what happened, where it happened, and when it happened.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import time
import uuid
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import pickle
import json
from collections import deque
import torch

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus 
from lmm_project.core.message import Message
from lmm_project.modules.memory.models import EpisodicMemory, MemoryConsolidationEvent
from lmm_project.modules.memory.neural_net import MemoryNeuralNetwork

logger = logging.getLogger(__name__)

class EpisodicMemorySystem:
    """
    Episodic memory system for storing and retrieving event memories
    
    Episodic memory stores experiences with their temporal, spatial, and emotional
    context. It develops over time, starting with basic event recording and
    eventually supporting detailed autobiographical memory.
    
    Development stages:
    - Stage 0 (0.0-0.2): Basic event storage with minimal context
    - Stage 1 (0.2-0.4): Time-based memory organization and simple retrieval
    - Stage 2 (0.4-0.6): Context-based episodic recall and emotional tagging
    - Stage 3 (0.6-0.8): Narrative connection between related episodes
    - Stage 4 (0.8-1.0): Autobiographical memory with self-reference
    """
    
    def __init__(
        self, 
        max_episodes: int = 1000,
        development_level: float = 0.0,
        embedding_dim: int = 128,
        base_directory: Optional[str] = None
    ):
        """
        Initialize episodic memory system
        
        Args:
            max_episodes: Maximum number of episodes to store
            development_level: Initial development level (0.0-1.0)
            embedding_dim: Dimension for memory embeddings
            base_directory: Directory for persistent storage
        """
        self.episodes: Dict[str, EpisodicMemory] = {}
        self.temporal_index: List[str] = []  # Episode IDs in temporal order
        self.context_index: Dict[str, List[str]] = {}  # Context -> episode IDs
        self.entity_index: Dict[str, List[str]] = {}  # Entity -> episode IDs
        self.narrative_index: Dict[str, List[str]] = {}  # Narrative ID -> episode IDs
        self.emotion_index: Dict[str, List[str]] = {}  # Emotion category -> episode IDs
        
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.development_level = development_level
        self.embedding_dim = embedding_dim
        self.last_consolidation = datetime.now()
        self.base_directory = base_directory
        
        # Recent episodes cache (for faster access to recent memories)
        self.recent_episodes = deque(maxlen=50)
        
        # Memory for important life events (never pruned)
        self.important_events: Dict[str, EpisodicMemory] = {}
        
        # Neural network for encoding and retrieval
        self.neural_network = MemoryNeuralNetwork(
            input_dim=embedding_dim,
            hidden_dim=256,
            output_dim=embedding_dim,
            memory_type="episodic",
            learning_rate=0.01,
            device="auto"  # Use GPU if available
        )
        
        # Update neural network development level
        self.neural_network.update_development(development_level)
        
        # Load stored episodes if base directory is provided
        if base_directory:
            self._load_episodes()
    
    def store_episode(self, episode_data: Dict[str, Any]) -> str:
        """
        Store a new episode in memory
        
        Args:
            episode_data: Data representing the episode
                Must include 'content' and 'context' keys
                
        Returns:
            ID of the stored episode
        """
        # Basic validation
        if 'content' not in episode_data or 'context' not in episode_data:
            logger.error("Episode data must include 'content' and 'context'")
            return ""
            
        # Generate episode ID
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        
        # Add timestamp if not provided
        if "timestamp" not in episode_data:
            episode_data["timestamp"] = datetime.now()
            
        # Add event_time if not provided
        if "event_time" not in episode_data:
            episode_data["event_time"] = episode_data["timestamp"]
            
        # Generate embedding based on development level
        embedding = self._generate_embedding(episode_data)
        if embedding is not None:
            episode_data["embedding"] = embedding.tolist()
            
        # Create EpisodicMemory object
        try:
            episode = EpisodicMemory(**episode_data)
        except Exception as e:
            logger.error(f"Error creating episodic memory: {e}")
            # Create with minimal data
            episode = EpisodicMemory(
                content=episode_data.get('content', ''),
                context=episode_data.get('context', ''),
                importance=episode_data.get('importance', 0.5)
            )
            
        # Store the episode
        self.episodes[episode_id] = episode
        
        # Add to indices based on development level
        self._add_to_indices(episode_id, episode)
        
        # Add to recent episodes
        self.recent_episodes.appendleft(episode_id)
        
        # If it's an important event, add to important events
        if episode.importance >= 0.8:
            self.important_events[episode_id] = episode
            
        # Increment episode count
        self.episode_count += 1
        
        # Check if we need to consolidate/prune
        current_time = datetime.now()
        if (current_time - self.last_consolidation).total_seconds() > 3600:  # Every hour
            self._consolidate_memories()
            self.last_consolidation = current_time
            
        # If we're over capacity, prune
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
            
        # Save episode if base directory is set
        if self.base_directory and self.development_level >= 0.4:
            self._save_episode(episode_id, episode)
            
        return episode_id
        
    def retrieve_episode(self, episode_id: str) -> Optional[EpisodicMemory]:
        """
        Retrieve a specific episode by ID
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            EpisodicMemory object or None if not found
        """
        # Check if episode exists
        if episode_id in self.episodes:
            episode = self.episodes[episode_id]
            
            # Update access count and timestamp
            episode.access_count += 1
            episode.last_accessed = datetime.now()
            
            # Increase activation level
            episode.update_activation(0.2)
            
            return episode
            
        return None
        
    def search_by_content(self, query: str, limit: int = 10) -> List[EpisodicMemory]:
        """
        Search for episodes by content
        
        Args:
            query: Text to search for in episode content
            limit: Maximum number of results to return
            
        Returns:
            List of matching EpisodicMemory objects
        """
        # Simple text search by default
        results = []
        
        # For more advanced developmental stages, use embeddings
        if self.development_level >= 0.4 and hasattr(self, 'neural_network'):
            # Generate query embedding
            query_embedding = self._text_to_embedding(query)
            
            if query_embedding is not None:
                return self._search_by_embedding(query_embedding, limit)
        
        # Fallback to simple text search
        for episode_id, episode in self.episodes.items():
            if query.lower() in episode.content.lower():
                results.append(episode)
                if len(results) >= limit:
                    break
                    
        return results
        
    def search_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        limit: int = 10
    ) -> List[EpisodicMemory]:
        """
        Search for episodes within a time range
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of results to return
            
        Returns:
            List of matching EpisodicMemory objects
        """
        results = []
        
        # Only enabled at development level 0.2+
        if self.development_level < 0.2:
            logger.warning("Time-based search requires development level 0.2+")
            return results
            
        # Search through temporal index
        for episode_id in self.temporal_index:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                
                # Check if event_time is within range
                if start_time <= episode.event_time <= end_time:
                    results.append(episode)
                    if len(results) >= limit:
                        break
                        
        return results
        
    def search_by_context(self, context: str, limit: int = 10) -> List[EpisodicMemory]:
        """
        Search for episodes by context
        
        Args:
            context: Context to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching EpisodicMemory objects
        """
        results = []
        
        # Only enabled at development level 0.4+
        if self.development_level < 0.4:
            logger.warning("Context-based search requires development level 0.4+")
            return results
            
        # Direct lookup in context index
        if context in self.context_index:
            episode_ids = self.context_index[context][:limit]
            for episode_id in episode_ids:
                if episode_id in self.episodes:
                    results.append(self.episodes[episode_id])
        
        # If we didn't find enough direct matches, try partial matching
        if len(results) < limit:
            for ctx, episode_ids in self.context_index.items():
                if context.lower() in ctx.lower():
                    for episode_id in episode_ids:
                        if episode_id in self.episodes and self.episodes[episode_id] not in results:
                            results.append(self.episodes[episode_id])
                            if len(results) >= limit:
                                break
                                
        return results
        
    def get_recent_episodes(self, limit: int = 10) -> List[EpisodicMemory]:
        """
        Get most recent episodes
        
        Args:
            limit: Maximum number of episodes to return
            
        Returns:
            List of recent EpisodicMemory objects
        """
        results = []
        
        # Use recent episodes cache for faster access
        for episode_id in list(self.recent_episodes)[:limit]:
            if episode_id in self.episodes:
                results.append(self.episodes[episode_id])
                
        return results
        
    def get_emotional_episodes(
        self, 
        valence: Optional[float] = None, 
        arousal: Optional[float] = None,
        limit: int = 10
    ) -> List[EpisodicMemory]:
        """
        Get episodes with specific emotional characteristics
        
        Args:
            valence: Target emotional valence (-1.0 to 1.0), or None for any
            arousal: Target emotional arousal (0.0 to 1.0), or None for any
            limit: Maximum number of episodes to return
            
        Returns:
            List of matching EpisodicMemory objects
        """
        results = []
        
        # Only enabled at development level 0.6+
        if self.development_level < 0.6:
            logger.warning("Emotional search requires development level 0.6+")
            return results
            
        # Search all episodes and filter by emotional characteristics
        for episode_id, episode in self.episodes.items():
            match = True
            
            if valence is not None:
                # Match within a range of the target valence
                if abs(episode.emotional_valence - valence) > 0.3:
                    match = False
                    
            if arousal is not None and match:
                # Match within a range of the target arousal
                if abs(episode.emotional_arousal - arousal) > 0.3:
                    match = False
                    
            if match:
                results.append(episode)
                if len(results) >= limit:
                    break
                    
        return results
        
    def get_narrative_episodes(self, narrative_id: str) -> List[EpisodicMemory]:
        """
        Get all episodes in a narrative sequence
        
        Args:
            narrative_id: ID of the narrative to retrieve
            
        Returns:
            List of episodes in the narrative, ordered by sequence position
        """
        results = []
        
        # Only enabled at development level 0.8+
        if self.development_level < 0.8:
            logger.warning("Narrative retrieval requires development level 0.8+")
            return results
            
        # Check if narrative exists
        if narrative_id in self.narrative_index:
            episode_ids = self.narrative_index[narrative_id]
            
            # Get all episodes in the narrative
            episodes = []
            for episode_id in episode_ids:
                if episode_id in self.episodes:
                    episodes.append(self.episodes[episode_id])
                    
            # Sort by sequence position
            episodes.sort(key=lambda x: x.sequence_position if x.sequence_position is not None else 9999)
            results = episodes
            
        return results
        
    def create_narrative(self, episode_ids: List[str], narrative_name: str = "") -> str:
        """
        Create a narrative by linking episodes in sequence
        
        Args:
            episode_ids: List of episode IDs to include in the narrative
            narrative_name: Optional name for the narrative
            
        Returns:
            ID of the created narrative
        """
        # Only enabled at development level 0.8+
        if self.development_level < 0.8:
            logger.warning("Narrative creation requires development level 0.8+")
            return ""
            
        # Generate narrative ID
        narrative_id = f"narr_{uuid.uuid4().hex[:8]}"
        if not narrative_name:
            narrative_name = f"Narrative {narrative_id}"
            
        # Create narrative
        self.narrative_index[narrative_id] = []
        
        # Add episodes to narrative
        for i, episode_id in enumerate(episode_ids):
            if episode_id in self.episodes:
                # Update episode with narrative information
                episode = self.episodes[episode_id]
                episode.narrative_id = narrative_id
                episode.sequence_position = i
                
                # Add to narrative index
                self.narrative_index[narrative_id].append(episode_id)
                
        return narrative_id
        
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of episodic memory
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        self.development_level = min(1.0, max(0.0, self.development_level + amount))
        
        # If significant change, adjust neural network
        if abs(self.development_level - old_level) >= 0.1:
            self.neural_network.update_development(amount)
            
            # Log development change
            logger.info(f"Episodic memory development updated to {self.development_level:.2f}")
            
            # If development crosses certain thresholds, enable new features
            if old_level < 0.4 <= self.development_level:
                logger.info("Episodic memory now supports context-based recall")
                
            if old_level < 0.6 <= self.development_level:
                logger.info("Episodic memory now supports emotional tagging")
                
            if old_level < 0.8 <= self.development_level:
                logger.info("Episodic memory now supports narrative sequencing")
                
        return self.development_level
        
    def _generate_embedding(self, episode_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate embedding for an episode based on content and context"""
        # Simple embedding generation at basic development levels
        if self.development_level < 0.3:
            # Random embedding as placeholder
            return np.random.randn(self.embedding_dim).astype(np.float32)
            
        # More sophisticated embedding at higher development levels
        if hasattr(self, 'neural_network'):
            # Combine content and context for input
            input_text = f"{episode_data['content']} {episode_data.get('context', '')}"
            return self._text_to_embedding(input_text)
            
        return None
        
    def _text_to_embedding(self, text: str) -> Optional[np.ndarray]:
        """Convert text to an embedding vector"""
        try:
            # Simple character-based embedding for demonstration
            # In a real system, this would use a more sophisticated embedding model
            chars = list(text.lower())
            char_codes = [ord(c) % 256 for c in chars]
            
            # Pad or truncate to embedding_dim
            if len(char_codes) >= self.embedding_dim:
                char_codes = char_codes[:self.embedding_dim]
            else:
                char_codes = char_codes + [0] * (self.embedding_dim - len(char_codes))
                
            # Convert to numpy array and normalize
            embedding = np.array(char_codes, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
            
    def _search_by_embedding(self, query_embedding: np.ndarray, limit: int = 10) -> List[EpisodicMemory]:
        """Search for episodes by embedding similarity"""
        results = []
        similarities = []
        
        # Convert query embedding to tensor
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        query_tensor = query_tensor.to(self.neural_network.device)
        
        # Compare with all episode embeddings
        for episode_id, episode in self.episodes.items():
            if episode.embedding is not None:
                # Convert episode embedding to tensor
                episode_tensor = torch.tensor(episode.embedding, dtype=torch.float32)
                episode_tensor = episode_tensor.to(self.neural_network.device)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    query_tensor.unsqueeze(0),
                    episode_tensor.unsqueeze(0)
                ).item()
                
                similarities.append((episode, similarity))
                
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        results = [item[0] for item in similarities[:limit]]
        return results
        
    def _add_to_indices(self, episode_id: str, episode: EpisodicMemory) -> None:
        """Add episode to appropriate indices based on development level"""
        # Temporal index (available at all levels)
        # Insert at appropriate position based on event_time
        inserted = False
        for i, existing_id in enumerate(self.temporal_index):
            if existing_id in self.episodes:
                existing_episode = self.episodes[existing_id]
                if episode.event_time < existing_episode.event_time:
                    self.temporal_index.insert(i, episode_id)
                    inserted = True
                    break
        if not inserted:
            self.temporal_index.append(episode_id)
            
        # Context index (level 0.4+)
        if self.development_level >= 0.4:
            if episode.context not in self.context_index:
                self.context_index[episode.context] = []
            if episode_id not in self.context_index[episode.context]:
                self.context_index[episode.context].append(episode_id)
                
        # Entity index (level 0.5+)
        if self.development_level >= 0.5:
            for entity in episode.involved_entities:
                if entity not in self.entity_index:
                    self.entity_index[entity] = []
                if episode_id not in self.entity_index[entity]:
                    self.entity_index[entity].append(episode_id)
                    
        # Narrative index (level 0.8+)
        if self.development_level >= 0.8 and episode.narrative_id:
            if episode.narrative_id not in self.narrative_index:
                self.narrative_index[episode.narrative_id] = []
            if episode_id not in self.narrative_index[episode.narrative_id]:
                self.narrative_index[episode.narrative_id].append(episode_id)
                
        # Emotion index (level 0.6+)
        if self.development_level >= 0.6:
            # Categorize emotions into buckets
            valence = episode.emotional_valence
            arousal = episode.emotional_arousal
            
            # Simple emotion categorization
            if valence > 0.3 and arousal > 0.5:
                emotion = "excitement"
            elif valence > 0.3 and arousal <= 0.5:
                emotion = "contentment"
            elif valence <= 0.3 and valence >= -0.3:
                emotion = "neutral"
            elif valence < -0.3 and arousal > 0.5:
                emotion = "fear/anger"
            else:
                emotion = "sadness"
                
            if emotion not in self.emotion_index:
                self.emotion_index[emotion] = []
            if episode_id not in self.emotion_index[emotion]:
                self.emotion_index[emotion].append(episode_id)
                
    def _consolidate_memories(self) -> None:
        """
        Consolidate memories to strengthen important ones and link related episodes
        
        This process simulates memory consolidation during rest/sleep phases.
        """
        # Only run consolidation at development level 0.3+
        if self.development_level < 0.3:
            return
            
        # Track changes for consolidation event
        strength_changes = {}
        
        # Strengthen frequently accessed memories
        for episode_id, episode in self.episodes.items():
            if episode.access_count > 0:
                # Calculate memory strength increase based on access count and importance
                strength_increase = min(0.1, 0.01 * episode.access_count * (0.5 + episode.importance))
                episode.decay_rate = max(0.001, episode.decay_rate - 0.0005 * strength_increase)
                
                strength_changes[episode_id] = strength_increase
                
                # Reset access count after consolidation
                episode.access_count = 0
                
        # Link related episodes at higher development levels
        if self.development_level >= 0.7:
            self._link_related_episodes()
            
        # Create consolidation event record
        if strength_changes:
            consolidation_event = MemoryConsolidationEvent(
                memory_ids=list(strength_changes.keys()),
                strength_changes=strength_changes,
                reason="sleep"
            )
            
            logger.info(f"Memory consolidation complete: {len(strength_changes)} episodes affected")
            
    def _link_related_episodes(self) -> None:
        """Link episodes that are related by context, time, or content"""
        # Only available at development level 0.7+
        if self.development_level < 0.7:
            return
            
        # Get recent episodes to analyze
        recent_ids = list(self.recent_episodes)[:100]  # Limit to 100 most recent
        
        # Group by context
        context_groups = {}
        for episode_id in recent_ids:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                if episode.context not in context_groups:
                    context_groups[episode.context] = []
                context_groups[episode.context].append(episode_id)
                
        # Create narratives for sequential episodes in same context
        for context, episode_ids in context_groups.items():
            if len(episode_ids) >= 3:  # Need at least 3 episodes to form a meaningful narrative
                # Sort by time
                episodes_with_time = []
                for ep_id in episode_ids:
                    if ep_id in self.episodes:
                        episodes_with_time.append((ep_id, self.episodes[ep_id].event_time))
                
                # Sort chronologically
                episodes_with_time.sort(key=lambda x: x[1])
                sorted_ids = [x[0] for x in episodes_with_time]
                
                # Check if these episodes span a reasonable timeframe (e.g., hours not years)
                if len(episodes_with_time) >= 2:
                    earliest = episodes_with_time[0][1]
                    latest = episodes_with_time[-1][1]
                    timespan = (latest - earliest).total_seconds()
                    
                    # If timespan is reasonable (< 24 hours), create a narrative
                    if timespan < 86400:  # 24 hours in seconds
                        narrative_name = f"Events at {context} on {earliest.strftime('%Y-%m-%d')}"
                        self.create_narrative(sorted_ids, narrative_name)
                        
    def _prune_episodes(self) -> None:
        """Remove least important episodes when over capacity"""
        # Don't prune if under capacity
        if len(self.episodes) <= self.max_episodes:
            return
            
        # Number to remove
        to_remove = len(self.episodes) - self.max_episodes
        
        # Sort episodes by importance (least important first)
        candidates = []
        for episode_id, episode in self.episodes.items():
            # Skip important life events
            if episode_id in self.important_events:
                continue
                
            # Calculate pruning score (lower = more likely to be pruned)
            # Consider: importance, recency, access count, emotional impact
            recency_factor = 1.0
            
            if episode.last_accessed:
                # Calculate days since last access
                days_since_access = (datetime.now() - episode.last_accessed).days
                recency_factor = max(0.1, 1.0 - (days_since_access / 100.0))
                
            pruning_score = (
                episode.importance * 0.4 + 
                recency_factor * 0.3 + 
                min(1.0, episode.access_count / 10.0) * 0.2 +
                abs(episode.emotional_valence) * 0.1
            )
            
            candidates.append((episode_id, pruning_score))
            
        # Sort by pruning score (ascending)
        candidates.sort(key=lambda x: x[1])
        
        # Remove the lowest-scoring episodes
        for episode_id, _ in candidates[:to_remove]:
            # Remove from all indices
            self._remove_from_indices(episode_id)
            
            # Remove the episode
            if episode_id in self.episodes:
                del self.episodes[episode_id]
                
            # Also remove from recent episodes if present
            if episode_id in self.recent_episodes:
                self.recent_episodes.remove(episode_id)
                
        logger.info(f"Pruned {to_remove} episodes from episodic memory")
        
    def _remove_from_indices(self, episode_id: str) -> None:
        """Remove episode from all indices"""
        # Temporal index
        if episode_id in self.temporal_index:
            self.temporal_index.remove(episode_id)
            
        # Context index
        for context, ids in self.context_index.items():
            if episode_id in ids:
                ids.remove(episode_id)
                
        # Entity index
        for entity, ids in self.entity_index.items():
            if episode_id in ids:
                ids.remove(episode_id)
                
        # Narrative index
        for narrative, ids in self.narrative_index.items():
            if episode_id in ids:
                ids.remove(episode_id)
                
        # Emotion index
        for emotion, ids in self.emotion_index.items():
            if episode_id in ids:
                ids.remove(episode_id)
                
    def _save_episode(self, episode_id: str, episode: EpisodicMemory) -> None:
        """Save episode to disk for persistence"""
        if not self.base_directory:
            return
            
        try:
            # Create directory if it doesn't exist
            episodes_dir = os.path.join(self.base_directory, "episodes")
            os.makedirs(episodes_dir, exist_ok=True)
            
            # Save episode to JSON file
            episode_path = os.path.join(episodes_dir, f"{episode_id}.json")
            with open(episode_path, 'w') as f:
                # Convert episode to dictionary
                episode_dict = episode.model_dump()
                # Convert datetime objects to strings
                for key, value in episode_dict.items():
                    if isinstance(value, datetime):
                        episode_dict[key] = value.isoformat()
                
                json.dump(episode_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving episode {episode_id}: {e}")
            
    def _load_episodes(self) -> None:
        """Load episodes from disk"""
        if not self.base_directory:
            return
            
        try:
            # Check if episodes directory exists
            episodes_dir = os.path.join(self.base_directory, "episodes")
            if not os.path.exists(episodes_dir):
                return
                
            # Load all episode files
            episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]
            
            for filename in episode_files:
                try:
                    episode_path = os.path.join(episodes_dir, filename)
                    with open(episode_path, 'r') as f:
                        episode_data = json.load(f)
                        
                    # Convert string timestamps back to datetime
                    for key in ['timestamp', 'event_time', 'last_accessed']:
                        if key in episode_data and episode_data[key]:
                            try:
                                episode_data[key] = datetime.fromisoformat(episode_data[key])
                            except:
                                episode_data[key] = datetime.now()
                                
                    # Create episode object
                    episode = EpisodicMemory(**episode_data)
                    
                    # Extract episode ID from filename
                    episode_id = filename.split('.')[0]
                    
                    # Store in memory
                    self.episodes[episode_id] = episode
                    
                    # Add to indices
                    self._add_to_indices(episode_id, episode)
                    
                    # Update episode count
                    self.episode_count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading episode file {filename}: {e}")
                    
            logger.info(f"Loaded {self.episode_count} episodes from disk")
            
        except Exception as e:
            logger.error(f"Error loading episodes: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of episodic memory"""
        return {
            "episode_count": len(self.episodes),
            "important_events_count": len(self.important_events),
            "development_level": self.development_level,
            "max_capacity": self.max_episodes,
            "contexts": list(self.context_index.keys()),
            "narratives": list(self.narrative_index.keys()),
            "emotion_categories": list(self.emotion_index.keys())
        } 