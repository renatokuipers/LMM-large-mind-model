from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import numpy as np
import uuid
import os
import json
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.memory.models import EpisodicMemory, Memory
from lmm_project.utils.vector_store import VectorStore

class EpisodicMemoryModule(BaseModule):
    """
    Episodic memory system for experiences and events
    
    Episodic memory represents specific events, experiences, and 
    autobiographical information. This memory type is crucial for 
    self-identity and autobiographical knowledge.
    """
    # Episode storage
    episodes: Dict[str, EpisodicMemory] = Field(default_factory=dict)
    # Vector store for semantic search
    vector_store: Optional[VectorStore] = None
    # Narratives (collections of related episodes)
    narratives: Dict[str, List[str]] = Field(default_factory=dict)
    # Episode contexts (grouping by context)
    contexts: Dict[str, Set[str]] = Field(default_factory=dict)
    # Temporal ordering of episodes (by event time)
    temporal_index: List[Tuple[datetime, str]] = Field(default_factory=list)
    # Storage directory
    storage_dir: str = Field(default="storage/memories/episodic")
    # Forgetting rate (memories below this vividness may be forgotten)
    forgetting_rate: float = Field(default=0.02)
    # Time bias (recent memories are emphasized)
    time_bias: float = Field(default=0.8)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize episodic memory module"""
        super().__init__(
            module_id=module_id,
            module_type="episodic_memory",
            event_bus=event_bus,
            **data
        )
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            dimension=768,
            storage_dir="storage/embeddings/episodic"
        )
        
        # Try to load previous episodes
        self._load_episodes()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("experience_recorded", self._handle_experience_recorded)
            self.subscribe_to_message("episodic_query", self._handle_episodic_query)
            self.subscribe_to_message("memory_consolidation", self._handle_memory_consolidation)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process episodic memory operations
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform (add_episode, get_episode,
                         search_episodes, create_narrative, etc.)
            - Additional parameters depend on the operation
            
        Returns:
        Dictionary containing operation results
        """
        operation = input_data.get("operation", "")
        
        if operation == "add_episode":
            episode_data = input_data.get("episode", {})
            return self.add_episode(episode_data)
        
        elif operation == "get_episode":
            episode_id = input_data.get("episode_id", "")
            return self.get_episode(episode_id)
        
        elif operation == "search_episodes":
            query = input_data.get("query", "")
            return self.search_episodes(query)
        
        elif operation == "create_narrative":
            episodes = input_data.get("episodes", [])
            narrative_name = input_data.get("narrative_name", f"narrative_{uuid.uuid4().hex[:8]}")
            return self.create_narrative(episodes, narrative_name)
        
        elif operation == "get_recent_episodes":
            count = input_data.get("count", 5)
            return self.get_recent_episodes(count)
        
        elif operation == "get_episodes_by_context":
            context = input_data.get("context", "")
            return self.get_episodes_by_context(context)
            
        return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update episodic memory's developmental level
        
        As episodic memory develops:
        - Episode detail and vividness increases
        - Temporal ordering becomes more accurate
        - Narrative formation becomes more sophisticated
        - Emotional binding to memories improves
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development
        delta = self.development_level - prev_level
        
        # Improve forgetting rate
        forgetting_decrease = delta * 0.003
        self.forgetting_rate = max(0.001, self.forgetting_rate - forgetting_decrease)
        
        # Adjust time bias (more developed episodic memory can better maintain 
        # older memories, so we reduce recency bias)
        time_bias_decrease = delta * 0.05
        self.time_bias = max(0.3, self.time_bias - time_bias_decrease)
        
        return self.development_level
    
    def add_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new episode to episodic memory
        
        Parameters:
        episode_data: Dictionary containing episode data
            - content: Description of the episode
            - context: Where this memory took place
            - involved_entities: Optional list of involved entities
            - emotional_impact: Optional dict of emotions and intensities
            - importance: Optional importance value (0.0-1.0)
            - is_first_person: Optional flag for first-person perspective
            - vividness: Optional vividness value (0.0-1.0)
            
        Returns:
        Operation result
        """
        # Create episode ID if not provided
        if "id" not in episode_data:
            episode_data["id"] = str(uuid.uuid4())
            
        episode_id = episode_data["id"]
        
        # Ensure required fields
        if "content" not in episode_data:
            return {"status": "error", "message": "No content provided"}
            
        if "context" not in episode_data:
            episode_data["context"] = "unknown"
        
        # Default event time to now if not provided
        if "event_time" not in episode_data:
            episode_data["event_time"] = datetime.now()
        
        # Create EpisodicMemory object
        episode = EpisodicMemory(**episode_data)
        
        # Store episode
        self.episodes[episode_id] = episode
        
        # Add to context index
        context = episode.context
        if context not in self.contexts:
            self.contexts[context] = set()
        self.contexts[context].add(episode_id)
        
        # Add to temporal index
        self.temporal_index.append((episode.event_time, episode_id))
        self.temporal_index.sort(key=lambda x: x[0])
        
        # Add to narrative if specified
        narrative_id = episode_data.get("narrative_id")
        if narrative_id:
            if narrative_id not in self.narratives:
                self.narratives[narrative_id] = []
            self.narratives[narrative_id].append(episode_id)
        
        # Generate embedding if not provided
        if not episode.embedding:
            # Combine content and context for better embedding
            embedding_text = f"{episode.content} in {episode.context}"
            episode.embedding = self._generate_embedding(embedding_text)
            
            # Add to vector store if embedding exists
            if episode.embedding:
                self.vector_store.add(
                    embeddings=[episode.embedding],
                    metadata_list=[{
                        "id": episode_id,
                        "content": episode.content,
                        "context": episode.context
                    }]
                )
        
        # Save to disk
        self._save_episode(episode)
        
        # Publish event
        self.publish_message("episode_added", {
            "episode_id": episode_id,
            "content": episode.content,
            "context": episode.context
        })
        
        return {
            "status": "success",
            "episode_id": episode_id
        }
    
    def get_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        Get a specific episode by ID
        
        Parameters:
        episode_id: ID of the episode to retrieve
        
        Returns:
        Operation result containing episode data
        """
        # Check if episode exists
        if episode_id not in self.episodes:
            return {"status": "error", "message": f"Episode not found: {episode_id}"}
        
        episode = self.episodes[episode_id]
        
        # Update activation
        episode.update_activation(0.3)
        
        # Get episode data
        result = episode.model_dump()
        
        # Convert datetime objects to strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        
        # Get adjacent episodes in narratives
        if episode.narrative_id:
            narrative = self.narratives.get(episode.narrative_id, [])
            if narrative:
                try:
                    pos = narrative.index(episode_id)
                    result["previous_episode"] = narrative[pos-1] if pos > 0 else None
                    result["next_episode"] = narrative[pos+1] if pos < len(narrative)-1 else None
                except ValueError:
                    result["previous_episode"] = None
                    result["next_episode"] = None
        
        # Publish event
        self.publish_message("episode_retrieved", {
            "episode_id": episode_id,
            "content": episode.content
        })
        
        # Return episode data
        return {
            "status": "success",
            "episode": result,
            "episode_id": episode_id
        }
    
    def search_episodes(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for episodes by semantic similarity
        
        Parameters:
        query: Text query
        limit: Maximum number of results
        
        Returns:
        Operation result containing matching episodes
        """
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        if not query_embedding:
            return {"status": "error", "message": "Failed to generate embedding for query"}
        
        # Search vector store
        try:
            indices, distances, metadata = self.vector_store.search(
                query_embedding=query_embedding,
                k=limit
            )
            
            # Collect results
            results = []
            for idx, dist, meta in zip(indices, distances, metadata):
                episode_id = meta.get("id")
                if episode_id in self.episodes:
                    episode = self.episodes[episode_id]
                    # Update activation
                    episode.update_activation(0.2)
                    results.append({
                        "episode_id": episode_id,
                        "content": episode.content,
                        "context": episode.context,
                        "event_time": episode.event_time.isoformat(),
                        "similarity_score": 1.0 - min(1.0, float(dist))
                    })
            
            # Publish event
            self.publish_message("episode_search_results", {
                "query": query,
                "result_count": len(results)
            })
            
            return {
                "status": "success",
                "results": results,
                "query": query
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Search failed: {str(e)}"}
    
    def create_narrative(self, episode_ids: List[str], narrative_name: str) -> Dict[str, Any]:
        """
        Create a narrative from multiple episodes
        
        Parameters:
        episode_ids: List of episode IDs to include in the narrative
        narrative_name: Name for the narrative
        
        Returns:
        Operation result
        """
        if not episode_ids:
            return {"status": "error", "message": "No episodes provided"}
        
        # Check if all episodes exist
        missing_episodes = [eid for eid in episode_ids if eid not in self.episodes]
        if missing_episodes:
            return {"status": "error", "message": f"Episodes not found: {missing_episodes}"}
        
        # Create narrative ID
        narrative_id = f"narrative_{uuid.uuid4().hex[:8]}"
        
        # Add episodes to narrative
        self.narratives[narrative_id] = episode_ids
        
        # Update narrative ID and sequence position in episodes
        for i, episode_id in enumerate(episode_ids):
            episode = self.episodes[episode_id]
            episode.narrative_id = narrative_id
            episode.sequence_position = i
            self._save_episode(episode)
        
        # Save narratives
        self._save_narratives()
        
        # Publish event
        self.publish_message("narrative_created", {
            "narrative_id": narrative_id,
            "narrative_name": narrative_name,
            "episode_count": len(episode_ids)
        })
        
        return {
            "status": "success",
            "narrative_id": narrative_id,
            "narrative_name": narrative_name,
            "episodes": episode_ids
        }
    
    def get_recent_episodes(self, count: int = 5) -> Dict[str, Any]:
        """
        Get the most recent episodes
        
        Parameters:
        count: Number of episodes to retrieve
        
        Returns:
        Operation result containing recent episodes
        """
        if not self.temporal_index:
            return {"status": "error", "message": "No episodes available", "episodes": []}
        
        # Get the most recent episodes
        recent_indices = self.temporal_index[-count:]
        recent_indices.reverse()  # Most recent first
        
        recent_episodes = []
        for _, episode_id in recent_indices:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                recent_episodes.append({
                    "episode_id": episode_id,
                    "content": episode.content,
                    "context": episode.context,
                    "event_time": episode.event_time.isoformat()
                })
        
        return {
            "status": "success",
            "episodes": recent_episodes,
            "count": len(recent_episodes)
        }
    
    def get_episodes_by_context(self, context: str) -> Dict[str, Any]:
        """
        Get episodes by context
        
        Parameters:
        context: Context to filter by
        
        Returns:
        Operation result containing episodes in the context
        """
        if context not in self.contexts:
            return {"status": "error", "message": f"Context not found: {context}", "episodes": []}
        
        context_episode_ids = self.contexts[context]
        context_episodes = []
        
        for episode_id in context_episode_ids:
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                context_episodes.append({
                    "episode_id": episode_id,
                    "content": episode.content,
                    "event_time": episode.event_time.isoformat(),
                    "importance": episode.importance
                })
        
        # Sort by event time (most recent first)
        context_episodes.sort(key=lambda x: x["event_time"], reverse=True)
        
        return {
            "status": "success",
            "context": context,
            "episodes": context_episodes,
            "count": len(context_episodes)
        }
    
    def decay_episodes(self) -> Dict[str, Any]:
        """
        Apply memory decay to episodic memories
        
        Episodes with low vividness and importance may be forgotten
        based on the system's forgetting rate.
        
        Returns:
        Operation result
        """
        before_count = len(self.episodes)
        forgotten_count = 0
        
        # Identify candidates for forgetting
        to_forget = []
        
        for episode_id, episode in self.episodes.items():
            # Calculate forgetting probability
            forget_probability = self._calculate_forget_probability(episode)
            
            # Check if episode should be forgotten
            if np.random.random() < forget_probability:
                to_forget.append(episode_id)
        
        # Forget episodes
        for episode_id in to_forget:
            self._forget_episode(episode_id)
            forgotten_count += 1
        
        return {
            "status": "success",
            "before_count": before_count,
            "forgotten_count": forgotten_count,
            "after_count": len(self.episodes)
        }
    
    def count_episodes(self) -> int:
        """Count the number of stored episodes"""
        return len(self.episodes)
    
    def save_state(self) -> str:
        """
        Save the current state of episodic memory
        
        Returns:
        Path to saved state directory
        """
        # Save episodes
        for episode_id, episode in self.episodes.items():
            self._save_episode(episode)
        
        # Save narratives
        self._save_narratives()
        
        # Save contexts
        self._save_contexts()
        
        # Save temporal index
        self._save_temporal_index()
        
        # Save vector store
        self.vector_store.save()
        
        return self.storage_dir
    
    def _save_episode(self, episode: EpisodicMemory) -> None:
        """Save a single episode to disk"""
        try:
            episodes_dir = Path(self.storage_dir) / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            
            episode_path = episodes_dir / f"{episode.id}.json"
            with open(episode_path, "w") as f:
                # We need to convert the episode to a dict and handle datetime objects
                episode_dict = episode.model_dump()
                # Convert datetime to string
                for key, value in episode_dict.items():
                    if isinstance(value, datetime):
                        episode_dict[key] = value.isoformat()
                json.dump(episode_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving episode {episode.id}: {e}")
    
    def _save_narratives(self) -> None:
        """Save narratives to disk"""
        try:
            narrative_path = Path(self.storage_dir) / "narratives.json"
            with open(narrative_path, "w") as f:
                json.dump(self.narratives, f, indent=2)
        except Exception as e:
            print(f"Error saving narratives: {e}")
    
    def _save_contexts(self) -> None:
        """Save contexts to disk"""
        try:
            # Convert sets to lists for JSON serialization
            contexts_dict = {context: list(episodes) for context, episodes in self.contexts.items()}
            
            contexts_path = Path(self.storage_dir) / "contexts.json"
            with open(contexts_path, "w") as f:
                json.dump(contexts_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving contexts: {e}")
    
    def _save_temporal_index(self) -> None:
        """Save temporal index to disk"""
        try:
            # Convert datetime objects to strings
            temporal_index = [(dt.isoformat(), eid) for dt, eid in self.temporal_index]
            
            temporal_path = Path(self.storage_dir) / "temporal_index.json"
            with open(temporal_path, "w") as f:
                json.dump(temporal_index, f, indent=2)
        except Exception as e:
            print(f"Error saving temporal index: {e}")
    
    def _load_episodes(self) -> None:
        """Load episodes from disk"""
        try:
            # Load episodes
            episodes_dir = Path(self.storage_dir) / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in episodes_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        episode_data = json.load(f)
                        # Convert string back to datetime
                        if "timestamp" in episode_data and isinstance(episode_data["timestamp"], str):
                            episode_data["timestamp"] = datetime.fromisoformat(episode_data["timestamp"])
                        if "event_time" in episode_data and isinstance(episode_data["event_time"], str):
                            episode_data["event_time"] = datetime.fromisoformat(episode_data["event_time"])
                        if "last_accessed" in episode_data and episode_data["last_accessed"] and isinstance(episode_data["last_accessed"], str):
                            episode_data["last_accessed"] = datetime.fromisoformat(episode_data["last_accessed"])
                        
                        # Create episode object
                        episode = EpisodicMemory(**episode_data)
                        self.episodes[episode.id] = episode
                        
                        # Add to vector store if embedding exists
                        if episode.embedding:
                            self.vector_store.add(
                                embeddings=[episode.embedding],
                                metadata_list=[{
                                    "id": episode.id,
                                    "content": episode.content,
                                    "context": episode.context
                                }]
                            )
                except Exception as e:
                    print(f"Error loading episode from {file_path}: {e}")
            
            # Load narratives
            narrative_path = Path(self.storage_dir) / "narratives.json"
            if narrative_path.exists():
                with open(narrative_path, "r") as f:
                    self.narratives = json.load(f)
            
            # Load contexts
            contexts_path = Path(self.storage_dir) / "contexts.json"
            if contexts_path.exists():
                with open(contexts_path, "r") as f:
                    contexts_dict = json.load(f)
                    # Convert lists back to sets
                    self.contexts = {context: set(episodes) for context, episodes in contexts_dict.items()}
            
            # Load temporal index
            temporal_path = Path(self.storage_dir) / "temporal_index.json"
            if temporal_path.exists():
                with open(temporal_path, "r") as f:
                    temporal_data = json.load(f)
                    # Convert strings back to datetime
                    self.temporal_index = [(datetime.fromisoformat(dt), eid) for dt, eid in temporal_data]
            
            print(f"Loaded {len(self.episodes)} episodes from disk")
        except Exception as e:
            print(f"Error loading episodes: {e}")
    
    def _forget_episode(self, episode_id: str) -> None:
        """
        Forget (remove) an episode
        
        Parameters:
        episode_id: ID of the episode to forget
        """
        if episode_id not in self.episodes:
            return
            
        episode = self.episodes[episode_id]
        
        # Remove from episodes
        del self.episodes[episode_id]
        
        # Remove from context
        context = episode.context
        if context in self.contexts and episode_id in self.contexts[context]:
            self.contexts[context].remove(episode_id)
            if not self.contexts[context]:
                del self.contexts[context]
        
        # Remove from narrative
        narrative_id = episode.narrative_id
        if narrative_id and narrative_id in self.narratives:
            if episode_id in self.narratives[narrative_id]:
                self.narratives[narrative_id].remove(episode_id)
                if not self.narratives[narrative_id]:
                    del self.narratives[narrative_id]
        
        # Remove from temporal index
        self.temporal_index = [(dt, eid) for dt, eid in self.temporal_index if eid != episode_id]
        
        # Delete episode file
        self._delete_episode_file(episode_id)
        
        # Publish event
        self.publish_message("episode_forgotten", {
            "episode_id": episode_id,
            "content": episode.content
        })
    
    def _delete_episode_file(self, episode_id: str) -> None:
        """Delete an episode file from disk"""
        try:
            episode_path = Path(self.storage_dir) / "episodes" / f"{episode_id}.json"
            if episode_path.exists():
                episode_path.unlink()
        except Exception as e:
            print(f"Error deleting episode file {episode_id}: {e}")
    
    def _calculate_forget_probability(self, episode: EpisodicMemory) -> float:
        """Calculate the probability of forgetting an episode"""
        # Base forgetting probability from forgetting rate
        prob = self.forgetting_rate
        
        # Adjust based on episode properties
        
        # Importance reduces forgetting
        prob -= episode.importance * 0.5
        
        # Vividness reduces forgetting
        prob -= episode.vividness * 0.3
        
        # Recent access reduces forgetting
        if episode.last_accessed:
            days_since_access = (datetime.now() - episode.last_accessed).days
            recency_factor = 1.0 / (1.0 + np.exp(-0.1 * days_since_access + 3))
            prob += recency_factor * 0.2
        
        # Emotional impact reduces forgetting (strong emotions are remembered)
        emotional_strength = 0.0
        if episode.emotional_impact:
            emotional_values = [abs(v) for v in episode.emotional_impact.values()]
            if emotional_values:
                emotional_strength = max(emotional_values)
        prob -= emotional_strength * 0.4
        
        # Narrative episodes are less likely to be forgotten
        if episode.narrative_id:
            prob -= 0.15
        
        # Time bias (older episodes more likely to be forgotten)
        days_old = (datetime.now() - episode.event_time).days
        time_factor = 1.0 / (1.0 + np.exp(-0.05 * days_old + 3))
        prob += time_factor * self.time_bias * 0.3
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, prob))
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for text"""
        try:
            from lmm_project.utils.llm_client import LLMClient
            client = LLMClient()
            embedding = client.get_embedding(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    # Event handlers
    
    def _handle_experience_recorded(self, message: Message) -> None:
        """
        Handle experience recorded events
        
        When an experience is recorded, it should be added to episodic memory.
        """
        content = message.content
        experience = content.get("experience", "")
        
        if not experience:
            return
            
        # Create episode data
        episode_data = {
            "content": experience,
            "context": content.get("context", "unknown"),
            "event_time": content.get("event_time", datetime.now()),
            "importance": content.get("importance", 0.5)
        }
        
        # Add emotional impact if available
        if "emotions" in content:
            episode_data["emotional_impact"] = content["emotions"]
            
        # Add involved entities if available
        if "entities" in content:
            episode_data["involved_entities"] = content["entities"]
            
        # Add the episode
        self.add_episode(episode_data)
    
    def _handle_episodic_query(self, message: Message) -> None:
        """Handle episodic query events"""
        content = message.content
        query = content.get("query", "")
        
        if not query:
            return
            
        results = self.search_episodes(query)
        
        if self.event_bus and results.get("status") == "success":
            # Publish results
            self.publish_message("episodic_query_response", {
                "requester": message.sender,
                "results": results.get("results", []),
                "query": query
            })
    
    def _handle_memory_consolidation(self, message: Message) -> None:
        """
        Handle memory consolidation events
        
        During consolidation (e.g., during simulated sleep), episodic 
        memories may be strengthened, weakened, or organized into narratives.
        """
        content = message.content
        event = content.get("event", {})
        
        if not event:
            return
            
        # Apply changes to episodes
        memory_ids = event.get("memory_ids", [])
        strength_changes = event.get("strength_changes", {})
        
        for memory_id, change in strength_changes.items():
            if memory_id in self.episodes:
                episode = self.episodes[memory_id]
                
                # Adjust vividness and importance
                episode.vividness = max(0.0, min(1.0, episode.vividness + change * 0.5))
                episode.importance = max(0.0, min(1.0, episode.importance + change * 0.3))
                
                # Save episode
                self._save_episode(episode)
        
        # Apply decay to all episodes
        self.decay_episodes() 