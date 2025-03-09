"""
Memory Module

This module is responsible for storing and retrieving information across
different timeframes and contexts. It integrates multiple memory systems 
including working memory, episodic memory, and semantic memory.
"""

import logging
import time
import uuid
import os
import json
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
from collections import deque, OrderedDict
import numpy as np

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "memory",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "MemorySystem":
    """
    Factory function to create and return a memory module
    
    This function initializes and returns a complete memory system,
    with working memory, episodic memory, and semantic memory.
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication
        development_level: Initial developmental level for the system
        
    Returns:
        Initialized MemorySystem
    """
    return MemorySystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class WorkingMemory:
    """
    Short-term memory buffer with limited capacity
    
    Working memory holds a small amount of information in an active state
    for manipulation and use in ongoing cognitive processes.
    """
    def __init__(self, capacity: int = 4):
        """
        Initialize working memory
        
        Args:
            capacity: Maximum number of items that can be held
        """
        self.capacity = capacity
        self.items = OrderedDict()  # Using OrderedDict for LRU functionality
        self.access_timestamps = {}  # Track when items were last accessed
        
    def add_item(self, item_id: str, item_data: Dict[str, Any]) -> bool:
        """
        Add an item to working memory
        
        Args:
            item_id: Unique identifier for the item
            item_data: The data to store
            
        Returns:
            Whether the item was successfully added
        """
        # Check if already at capacity
        if len(self.items) >= self.capacity and item_id not in self.items:
            # Remove least recently used item
            self._remove_lru_item()
            
        # Add or update the item
        self.items[item_id] = item_data
        self.access_timestamps[item_id] = time.time()
        return True
        
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an item from working memory
        
        Args:
            item_id: Identifier of the item to retrieve
            
        Returns:
            The item data or None if not found
        """
        if item_id in self.items:
            # Update access timestamp
            self.access_timestamps[item_id] = time.time()
            return self.items[item_id]
        return None
        
    def _remove_lru_item(self):
        """Remove the least recently used item"""
        if not self.items:
            return
            
        # Find item with oldest timestamp
        oldest_id = min(self.access_timestamps, key=self.access_timestamps.get)
        
        # Remove it
        if oldest_id in self.items:
            del self.items[oldest_id]
            del self.access_timestamps[oldest_id]
            
    def get_all_items(self) -> List[Dict[str, Any]]:
        """Get all items currently in working memory"""
        return list(self.items.values())
        
    def clear(self):
        """Clear all items from working memory"""
        self.items.clear()
        self.access_timestamps.clear()
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of working memory"""
        return {
            "capacity": self.capacity,
            "current_usage": len(self.items),
            "items": list(self.items.keys())
        }

class EpisodicMemory:
    """
    Memory for specific events and experiences
    
    Episodic memory stores experiences with their temporal and contextual details.
    It develops from basic event storage to sophisticated autobiographical memory.
    """
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episodic memory
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.episodes = {}  # Dictionary of episodes by ID
        self.episode_timestamps = {}  # When episodes were created
        self.temporal_index = []  # Episodes in temporal order
        self.max_episodes = max_episodes
        self.episode_count = 0
        
    def store_episode(self, episode_data: Dict[str, Any]) -> str:
        """
        Store a new episode
        
        Args:
            episode_data: Data representing the episode
                Must include 'content' key
                
        Returns:
            ID of the stored episode
        """
        # Generate a unique ID for this episode
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        
        # Add timestamp if not provided
        if "timestamp" not in episode_data:
            episode_data["timestamp"] = time.time()
            
        # Store the episode
        self.episodes[episode_id] = episode_data
        self.episode_timestamps[episode_id] = episode_data["timestamp"]
        
        # Add to temporal index
        self._add_to_temporal_index(episode_id, episode_data["timestamp"])
        
        # Increment count
        self.episode_count += 1
        
        # Check if we need to prune
        if self.episode_count > self.max_episodes:
            self._prune_episodes()
            
        return episode_id
        
    def _add_to_temporal_index(self, episode_id: str, timestamp: float):
        """Add an episode to the temporal index in the correct position"""
        # Simple implementation: just maintain a sorted list
        # For larger systems, more sophisticated indexing would be needed
        self.temporal_index.append((timestamp, episode_id))
        self.temporal_index.sort()  # Sort by timestamp
        
    def retrieve_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific episode by ID
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            Episode data or None if not found
        """
        return self.episodes.get(episode_id)
        
    def retrieve_recent_episodes(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent episodes
        
        Args:
            count: Number of episodes to retrieve
            
        Returns:
            List of recent episodes
        """
        # Get the most recent episodes from the temporal index
        recent_ids = [ep_id for _, ep_id in reversed(self.temporal_index[-count:])]
        return [self.episodes[ep_id] for ep_id in recent_ids if ep_id in self.episodes]
        
    def retrieve_by_timeframe(
        self, 
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve episodes within a specific timeframe
        
        Args:
            start_time: Start of the timeframe (timestamp)
            end_time: End of the timeframe (timestamp)
            
        Returns:
            List of episodes within the timeframe
        """
        # Find episodes in the timeframe
        matching_ids = []
        for timestamp, ep_id in self.temporal_index:
            if start_time <= timestamp <= end_time:
                matching_ids.append(ep_id)
                
        return [self.episodes[ep_id] for ep_id in matching_ids if ep_id in self.episodes]
        
    def _prune_episodes(self):
        """Remove oldest episodes to stay within max limit"""
        # Find how many episodes to remove
        to_remove = self.episode_count - self.max_episodes
        if to_remove <= 0:
            return
            
        # Remove oldest episodes
        for timestamp, ep_id in self.temporal_index[:to_remove]:
            if ep_id in self.episodes:
                del self.episodes[ep_id]
                del self.episode_timestamps[ep_id]
                
        # Update temporal index and count
        self.temporal_index = self.temporal_index[to_remove:]
        self.episode_count -= to_remove
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of episodic memory"""
        return {
            "episode_count": self.episode_count,
            "max_episodes": self.max_episodes,
            "oldest_timestamp": self.temporal_index[0][0] if self.temporal_index else None,
            "newest_timestamp": self.temporal_index[-1][0] if self.temporal_index else None
        }

class SemanticMemory:
    """
    Memory for facts, concepts, and general knowledge
    
    Semantic memory stores conceptual knowledge independent of specific
    contexts or episodes. It develops from simple facts to complex knowledge networks.
    """
    def __init__(self, max_items: int = 10000):
        """
        Initialize semantic memory
        
        Args:
            max_items: Maximum number of items to store
        """
        self.concepts = {}  # Dictionary of concepts by ID
        self.concept_timestamps = {}  # When concepts were created/updated
        self.max_items = max_items
        
        # Simple retrieval indices
        self.label_index = {}  # Map labels to concept IDs
        self.type_index = {}  # Map concept types to concept IDs
        
        # For development tracking
        self.item_count = 0
        self.update_count = 0
        
    def store_concept(self, concept_data: Dict[str, Any]) -> str:
        """
        Store a concept in semantic memory
        
        Args:
            concept_data: Data representing the concept
                Should include 'label' and 'type' keys
                
        Returns:
            ID of the stored concept
        """
        # Extract key information
        label = concept_data.get("label", "")
        concept_type = concept_data.get("type", "general")
        
        # Check if this concept already exists (by label)
        concept_id = self.label_index.get(label.lower())
        
        # If it exists, update it
        if concept_id:
            old_type = self.concepts[concept_id].get("type", "")
            
            # Update the concept
            self.concepts[concept_id].update(concept_data)
            self.concept_timestamps[concept_id] = time.time()
            
            # Update type index if type changed
            if old_type != concept_type:
                # Remove from old type index
                if old_type in self.type_index and concept_id in self.type_index[old_type]:
                    self.type_index[old_type].remove(concept_id)
                
                # Add to new type index
                if concept_type not in self.type_index:
                    self.type_index[concept_type] = set()
                self.type_index[concept_type].add(concept_id)
                
            self.update_count += 1
            
        else:
            # Create a new concept
            concept_id = f"con_{uuid.uuid4().hex[:8]}"
            
            # Store the concept
            self.concepts[concept_id] = concept_data
            self.concept_timestamps[concept_id] = time.time()
            
            # Add to indices
            self.label_index[label.lower()] = concept_id
            
            if concept_type not in self.type_index:
                self.type_index[concept_type] = set()
            self.type_index[concept_type].add(concept_id)
            
            self.item_count += 1
            
            # Check if we need to prune
            if self.item_count > self.max_items:
                self._prune_concepts()
                
        return concept_id
        
    def retrieve_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a concept by its label
        
        Args:
            label: Label of the concept to retrieve
            
        Returns:
            Concept data or None if not found
        """
        concept_id = self.label_index.get(label.lower())
        if concept_id:
            return self.concepts.get(concept_id)
        return None
        
    def retrieve_by_type(self, concept_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve concepts of a specific type
        
        Args:
            concept_type: Type of concepts to retrieve
            limit: Maximum number to retrieve
            
        Returns:
            List of matching concepts
        """
        if concept_type not in self.type_index:
            return []
            
        # Get concept IDs of this type
        concept_ids = list(self.type_index[concept_type])[:limit]
        
        # Return the concepts
        return [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        
    def _prune_concepts(self):
        """Remove least recently used concepts to stay within max limit"""
        # Find how many concepts to remove
        to_remove = self.item_count - self.max_items
        if to_remove <= 0:
            return
            
        # Sort concepts by timestamp
        sorted_concepts = sorted(
            self.concept_timestamps.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest concepts
        for concept_id, _ in sorted_concepts[:to_remove]:
            if concept_id in self.concepts:
                # Get concept data for index removal
                concept = self.concepts[concept_id]
                label = concept.get("label", "").lower()
                concept_type = concept.get("type", "")
                
                # Remove from main storage
                del self.concepts[concept_id]
                del self.concept_timestamps[concept_id]
                
                # Remove from indices
                if label in self.label_index and self.label_index[label] == concept_id:
                    del self.label_index[label]
                    
                if concept_type in self.type_index and concept_id in self.type_index[concept_type]:
                    self.type_index[concept_type].remove(concept_id)
                    
        # Update count
        self.item_count -= to_remove
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of semantic memory"""
        return {
            "item_count": self.item_count,
            "update_count": self.update_count,
            "max_items": self.max_items,
            "type_counts": {ctype: len(cids) for ctype, cids in self.type_index.items()}
        }

class MemorySystem(BaseModule):
    """
    Integrated memory system with multiple memory types
    
    The memory system develops from simple storage and retrieval to 
    sophisticated organization, consolidation, and recall processes.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic memory storage",
        0.2: "Short-term working memory",
        0.4: "Episodic memory formation",
        0.6: "Semantic memory organization",
        0.8: "Memory consolidation",
        1.0: "Integrated memory systems"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the memory system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="memory_system",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize memory systems
        self.working_memory = WorkingMemory(capacity=3)
        self.episodic_memory = EpisodicMemory(max_episodes=100)
        self.semantic_memory = SemanticMemory(max_items=500)
        
        # Adjust memory parameters based on development level
        self._adjust_memory_for_development()
        
        # Subscribe to relevant message types
        if self.event_bus:
            self.subscribe_to_message("memory_store")
            self.subscribe_to_message("memory_retrieve")
            self.subscribe_to_message("perception_result")
            self.subscribe_to_message("attention_focus_update")
    
    def _adjust_memory_for_development(self):
        """Adjust memory parameters based on developmental level"""
        # Working memory capacity increases with development
        # Research suggests capacity grows from ~2-3 items to ~4-7 items
        self.working_memory.capacity = max(2, int(2 + self.development_level * 5))
        
        # Episodic memory capacity increases dramatically with development
        self.episodic_memory.max_episodes = int(100 + self.development_level * 900)
        
        # Semantic memory also expands with development
        self.semantic_memory.max_items = int(500 + self.development_level * 9500)
        
        logger.debug(f"Memory capacity updated: WM={self.working_memory.capacity}, " 
                    f"EM={self.episodic_memory.max_episodes}, SM={self.semantic_memory.max_items}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations
        
        Args:
            input_data: Dictionary containing operation details
                Required keys: 'operation'
                
        Returns:
            Operation result
        """
        # Get operation type
        operation = input_data.get("operation", "")
        memory_type = input_data.get("memory_type", "")
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        result = {
            "process_id": process_id,
            "timestamp": time.time(),
            "module_id": self.module_id,
            "operation": operation,
            "memory_type": memory_type,
            "development_level": self.development_level
        }
        
        # Early development - only working memory operations
        if self.development_level < 0.2 and memory_type not in ["working", ""]:
            result.update({
                "status": "error",
                "error": "Memory system not developed enough for this operation",
                "available_types": ["working"]
            })
            return result
            
        # Route operation to appropriate handler
        if operation == "store":
            # Store operation
            result.update(self._handle_store(input_data))
            
        elif operation == "retrieve":
            # Retrieve operation
            result.update(self._handle_retrieve(input_data))
            
        elif operation == "consolidate":
            # Consolidation operation (only at higher development levels)
            if self.development_level >= 0.8:
                result.update(self._handle_consolidate(input_data))
            else:
                result.update({
                    "status": "error",
                    "error": "Memory consolidation not yet developed"
                })
                
        else:
            # Unknown operation
            result.update({
                "status": "error",
                "error": f"Unknown memory operation: {operation}"
            })
            
        # Publish result
        if self.event_bus:
            self.publish_message(
                "memory_result",
                {"result": result}
            )
            
        return result
    
    def _handle_store(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory storage operations
        
        Args:
            input_data: Storage operation data
            
        Returns:
            Operation result
        """
        memory_type = input_data.get("memory_type", "working")
        content = input_data.get("content", {})
        
        if not content:
            return {
                "status": "error",
                "error": "No content provided for storage"
            }
            
        if memory_type == "working":
            # Store in working memory
            item_id = input_data.get("item_id", f"wm_{uuid.uuid4().hex[:8]}")
            success = self.working_memory.add_item(item_id, content)
            
            return {
                "status": "success" if success else "error",
                "item_id": item_id,
                "memory_type": "working"
            }
            
        elif memory_type == "episodic" and self.development_level >= 0.4:
            # Store in episodic memory
            episode_id = self.episodic_memory.store_episode(content)
            
            return {
                "status": "success",
                "episode_id": episode_id,
                "memory_type": "episodic"
            }
            
        elif memory_type == "semantic" and self.development_level >= 0.6:
            # Store in semantic memory
            concept_id = self.semantic_memory.store_concept(content)
            
            return {
                "status": "success",
                "concept_id": concept_id,
                "memory_type": "semantic"
            }
            
        else:
            return {
                "status": "error",
                "error": f"Memory type '{memory_type}' not available at development level {self.development_level}"
            }
    
    def _handle_retrieve(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory retrieval operations
        
        Args:
            input_data: Retrieval operation data
            
        Returns:
            Operation result with retrieved content
        """
        memory_type = input_data.get("memory_type", "working")
        
        if memory_type == "working":
            # Retrieve from working memory
            if "item_id" in input_data:
                # Retrieve specific item
                item_id = input_data["item_id"]
                item = self.working_memory.get_item(item_id)
                
                if item:
                    return {
                        "status": "success",
                        "item_id": item_id,
                        "content": item,
                        "memory_type": "working"
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Item '{item_id}' not found in working memory"
                    }
            else:
                # Retrieve all items
                items = self.working_memory.get_all_items()
                
                return {
                    "status": "success",
                    "items": items,
                    "count": len(items),
                    "memory_type": "working"
                }
                
        elif memory_type == "episodic" and self.development_level >= 0.4:
            # Retrieve from episodic memory
            if "episode_id" in input_data:
                # Retrieve specific episode
                episode_id = input_data["episode_id"]
                episode = self.episodic_memory.retrieve_episode(episode_id)
                
                if episode:
                    return {
                        "status": "success",
                        "episode_id": episode_id,
                        "content": episode,
                        "memory_type": "episodic"
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Episode '{episode_id}' not found in episodic memory"
                    }
            elif "timeframe" in input_data:
                # Retrieve by timeframe
                timeframe = input_data["timeframe"]
                start_time = timeframe.get("start", 0)
                end_time = timeframe.get("end", time.time())
                
                episodes = self.episodic_memory.retrieve_by_timeframe(start_time, end_time)
                
                return {
                    "status": "success",
                    "episodes": episodes,
                    "count": len(episodes),
                    "memory_type": "episodic",
                    "timeframe": timeframe
                }
            else:
                # Retrieve recent episodes
                count = input_data.get("count", 5)
                episodes = self.episodic_memory.retrieve_recent_episodes(count)
                
                return {
                    "status": "success",
                    "episodes": episodes,
                    "count": len(episodes),
                    "memory_type": "episodic"
                }
                
        elif memory_type == "semantic" and self.development_level >= 0.6:
            # Retrieve from semantic memory
            if "label" in input_data:
                # Retrieve by label
                label = input_data["label"]
                concept = self.semantic_memory.retrieve_by_label(label)
                
                if concept:
                    return {
                        "status": "success",
                        "label": label,
                        "content": concept,
                        "memory_type": "semantic"
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Concept '{label}' not found in semantic memory"
                    }
            elif "type" in input_data:
                # Retrieve by type
                concept_type = input_data["type"]
                limit = input_data.get("limit", 100)
                
                concepts = self.semantic_memory.retrieve_by_type(concept_type, limit)
                
                return {
                    "status": "success",
                    "type": concept_type,
                    "concepts": concepts,
                    "count": len(concepts),
                    "memory_type": "semantic"
                }
            else:
                return {
                    "status": "error",
                    "error": "No retrieval criteria specified for semantic memory"
                }
                
        else:
            return {
                "status": "error",
                "error": f"Memory type '{memory_type}' not available at development level {self.development_level}"
            }
    
    def _handle_consolidate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory consolidation operations
        
        This involves moving information between memory systems,
        such as from working memory to long-term storage.
        
        Args:
            input_data: Consolidation operation data
            
        Returns:
            Operation result
        """
        # Only available at higher development levels
        if self.development_level < 0.8:
            return {
                "status": "error",
                "error": "Memory consolidation not yet developed"
            }
            
        source_type = input_data.get("source_type", "")
        target_type = input_data.get("target_type", "")
        
        if not source_type or not target_type:
            return {
                "status": "error",
                "error": "Source and target memory types must be specified"
            }
            
        # Handle different consolidation pathways
        if source_type == "working" and target_type == "episodic":
            # Consolidate working memory to episodic memory
            items = self.working_memory.get_all_items()
            
            # Create a single episode from all working memory items
            if items:
                episode_data = {
                    "content": items,
                    "source": "working_memory",
                    "timestamp": time.time(),
                    "consolidation": True
                }
                
                episode_id = self.episodic_memory.store_episode(episode_data)
                
                return {
                    "status": "success",
                    "source_type": source_type,
                    "target_type": target_type,
                    "episode_id": episode_id,
                    "item_count": len(items)
                }
            else:
                return {
                    "status": "error",
                    "error": "No items in working memory to consolidate"
                }
                
        elif source_type == "episodic" and target_type == "semantic":
            # Extract semantic information from episodes
            if "episode_ids" in input_data:
                episode_ids = input_data["episode_ids"]
                episodes = [
                    self.episodic_memory.retrieve_episode(ep_id)
                    for ep_id in episode_ids
                    if self.episodic_memory.retrieve_episode(ep_id)
                ]
            else:
                # Use recent episodes
                count = input_data.get("count", 5)
                episodes = self.episodic_memory.retrieve_recent_episodes(count)
                
            # Extract concepts from episodes
            # This is a simplified approach - in a real system this would involve
            # sophisticated concept extraction and generalization
            concepts = []
            for episode in episodes:
                # Extract potential concepts based on episode content
                if isinstance(episode.get("content"), dict) and "label" in episode["content"]:
                    # If content already has a label field, it might be a concept
                    concepts.append(episode["content"])
                elif isinstance(episode.get("content"), list):
                    # If content is a list, check each item
                    for item in episode["content"]:
                        if isinstance(item, dict) and "label" in item:
                            concepts.append(item)
                            
            # Store extracted concepts
            concept_ids = []
            for concept in concepts:
                # Add a type if not present
                if "type" not in concept:
                    concept["type"] = "extracted"
                    
                concept_id = self.semantic_memory.store_concept(concept)
                concept_ids.append(concept_id)
                
            return {
                "status": "success",
                "source_type": source_type,
                "target_type": target_type,
                "concept_ids": concept_ids,
                "concept_count": len(concept_ids)
            }
            
        else:
            return {
                "status": "error",
                "error": f"Unsupported consolidation pathway: {source_type} to {target_type}"
            }
    
    def _handle_message(self, message: Message):
        """
        Handle incoming messages
        
        Args:
            message: The message to handle
        """
        if message.message_type == "memory_store":
            # Memory storage request
            if message.content:
                self.process_input({
                    "operation": "store",
                    "process_id": message.id,
                    **message.content
                })
                
        elif message.message_type == "memory_retrieve":
            # Memory retrieval request
            if message.content:
                self.process_input({
                    "operation": "retrieve",
                    "process_id": message.id,
                    **message.content
                })
                
        elif message.message_type == "perception_result":
            # Store perception results in working memory
            if message.content and "result" in message.content:
                result = message.content["result"]
                
                # Only store if development level is sufficient
                if self.development_level >= 0.2:
                    # Store in working memory
                    item_id = f"perception_{result.get('process_id', message.id)}"
                    self.working_memory.add_item(item_id, result)
                    
                    # At higher development, also store in episodic memory
                    if self.development_level >= 0.4:
                        # Only store significant perceptions in episodic memory
                        # Here we use a simple heuristic: if there are recognized patterns
                        if "patterns" in result and result["patterns"]:
                            self.episodic_memory.store_episode({
                                "content": result,
                                "source": "perception",
                                "timestamp": result.get("timestamp", message.timestamp)
                            })
                
        elif message.message_type == "attention_focus_update":
            # Store attention focus in working memory
            if message.content and "result" in message.content:
                result = message.content["result"]
                
                # Store current focus in working memory
                if result.get("current_focus"):
                    item_id = f"attention_{result.get('process_id', message.id)}"
                    self.working_memory.add_item(item_id, result["current_focus"])
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update development level
        prev_level = self.development_level
        new_level = super().update_development(amount)
        
        # If development level changed significantly, adjust memory systems
        if int(prev_level * 10) != int(new_level * 10):
            logger.info(f"Memory system upgraded to development level {new_level:.1f}")
            self._adjust_memory_for_development()
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add memory-specific state
        state.update({
            "working_memory": self.working_memory.get_state(),
            "episodic_memory": self.episodic_memory.get_state() if self.development_level >= 0.4 else "Not yet developed",
            "semantic_memory": self.semantic_memory.get_state() if self.development_level >= 0.6 else "Not yet developed"
        })
        
        return state 