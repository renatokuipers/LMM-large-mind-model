# Empty placeholder files 

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
from lmm_project.modules.memory.models import Memory, MemoryConsolidationEvent
from lmm_project.utils.vector_store import VectorStore

class LongTermMemory(BaseModule):
    """
    Long-term memory storage system
    
    Long-term memory provides persistent storage for memories that have
    been consolidated from working memory. It includes mechanisms for
    retrieval, forgetting, and consolidation.
    """
    # Memory storage
    memories: Dict[str, Memory] = Field(default_factory=dict)
    # Vector store for semantic search
    vector_store: Optional[VectorStore] = None
    # Embedding dimension
    embedding_dimension: int = Field(default=768)
    # Consolidation threshold (memories above this activation get consolidated)
    consolidation_threshold: float = Field(default=0.6)
    # Minimum time between consolidation events (seconds)
    consolidation_interval: float = Field(default=300)
    # Forgetting rate (memories below this importance may be forgotten)
    forgetting_rate: float = Field(default=0.01)
    # Last consolidation timestamp
    last_consolidation: datetime = Field(default_factory=datetime.now)
    # Storage directory
    storage_dir: str = Field(default="storage/memories")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize long-term memory module"""
        super().__init__(
            module_id=module_id,
            module_type="long_term_memory",
            event_bus=event_bus,
            **data
        )
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            dimension=self.embedding_dimension,
            storage_dir="storage/embeddings/memories"
        )
        
        # Try to load previous memories
        self._load_memories()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("working_memory_update", self._handle_working_memory_update)
            self.subscribe_to_message("memory_search", self._handle_memory_search)
            self.subscribe_to_message("consolidation_trigger", self._handle_consolidation_trigger)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform (store, retrieve, search, forget)
            - memory: Memory data for store operation
            - memory_id: Memory ID for retrieve/forget operations
            - query: Search query for search operation
            - embedding: Optional query embedding for search
            - limit: Optional result limit for search
            
        Returns:
        Dictionary containing operation results
        """
        operation = input_data.get("operation", "")
        
        if operation == "store":
            memory_data = input_data.get("memory", {})
            return self.store_memory(memory_data)
        
        elif operation == "retrieve":
            memory_id = input_data.get("memory_id", "")
            return self.retrieve_memory(memory_id)
        
        elif operation == "search":
            query = input_data.get("query", "")
            embedding = input_data.get("embedding")
            limit = input_data.get("limit", 5)
            return self.search_memories(query, embedding, limit)
        
        elif operation == "forget":
            memory_id = input_data.get("memory_id", "")
            return self.forget_memory(memory_id)
        
        elif operation == "consolidate":
            return self.consolidate_memories()
            
        return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update long-term memory's developmental level
        
        As long-term memory develops:
        - Capacity increases
        - Retrieval becomes more efficient
        - Consolidation threshold decreases
        - Forgetting becomes more selective
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development
        delta = self.development_level - prev_level
        
        # Improve consolidation threshold
        threshold_decrease = delta * 0.05
        self.consolidation_threshold = max(0.3, self.consolidation_threshold - threshold_decrease)
        
        # Decrease forgetting rate
        forgetting_decrease = delta * 0.002
        self.forgetting_rate = max(0.001, self.forgetting_rate - forgetting_decrease)
        
        return self.development_level
    
    def store_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory in long-term memory
        
        Parameters:
        memory_data: Dictionary containing memory data
        
        Returns:
        Operation result
        """
        # Create a new Memory instance
        memory_type = memory_data.get("type", "generic")
        
        if "id" not in memory_data:
            memory_data["id"] = str(uuid.uuid4())
            
        memory_id = memory_data["id"]
        
        # Create Memory object with appropriate type
        if memory_type == "memory":
            from lmm_project.modules.memory.models import Memory
            memory = Memory(**memory_data)
        else:
            # Default to base Memory type
            memory = Memory(**memory_data)
        
        # Generate embedding if not provided
        if not memory.embedding and "content" in memory_data:
            memory.embedding = self._generate_embedding(memory.content)
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Add to vector store if embedding exists
        if memory.embedding:
            self.vector_store.add(
                embeddings=[memory.embedding],
                metadata_list=[{"id": memory_id, "content": memory.content}]
            )
        
        # Save to disk
        self._save_memory(memory)
        
        # Publish event
        self.publish_message("memory_stored", {
            "memory_id": memory_id,
            "content": memory.content,
            "timestamp": memory.timestamp.isoformat()
        })
        
        return {
            "status": "success",
            "memory_id": memory_id
        }
    
    def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a memory by ID
        
        Parameters:
        memory_id: ID of the memory to retrieve
        
        Returns:
        Operation result containing memory data
        """
        # Check if memory exists
        if memory_id not in self.memories:
            return {"status": "error", "message": f"Memory not found: {memory_id}"}
        
        memory = self.memories[memory_id]
        
        # Update activation
        memory.update_activation(0.3)
        
        # Publish event
        self.publish_message("memory_retrieved", {
            "memory_id": memory_id,
            "content": memory.content,
            "importance": memory.importance
        })
        
        # Return memory data
        return {
            "status": "success",
            "memory": memory.model_dump(),
            "memory_id": memory_id
        }
    
    def search_memories(
        self, 
        query: str, 
        query_embedding: Optional[List[float]] = None, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for memories by semantic similarity
        
        Parameters:
        query: Text query
        query_embedding: Optional pre-generated embedding
        limit: Maximum number of results
        
        Returns:
        Operation result containing matching memories
        """
        # Generate embedding if not provided
        if not query_embedding:
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
                memory_id = meta.get("id")
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    # Update activation
                    memory.update_activation(0.2)
                    results.append({
                        "memory_id": memory_id,
                        "content": memory.content,
                        "importance": memory.importance,
                        "similarity_score": 1.0 - min(1.0, float(dist))
                    })
            
            # Publish event
            self.publish_message("memory_search_results", {
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
    
    def forget_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Forget (remove) a memory
        
        Parameters:
        memory_id: ID of the memory to forget
        
        Returns:
        Operation result
        """
        # Check if memory exists
        if memory_id not in self.memories:
            return {"status": "error", "message": f"Memory not found: {memory_id}"}
        
        # Get memory before removing
        memory = self.memories[memory_id]
        
        # Remove from memory store
        del self.memories[memory_id]
        
        # Remove from vector store
        # (Note: This is a simplified approach - in reality you'd need to track the index)
        # self.vector_store.delete([memory_id])
        
        # Remove from disk
        self._delete_memory_file(memory_id)
        
        # Publish event
        self.publish_message("memory_forgotten", {
            "memory_id": memory_id,
            "content": memory.content
        })
        
        return {
            "status": "success",
            "memory_id": memory_id
        }
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidate memories from working memory to long-term memory
        
        Returns:
        Consolidation result
        """
        now = datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds()
        
        # Check if it's time to consolidate
        if time_since_last < self.consolidation_interval:
            return {
                "status": "skipped", 
                "message": "Consolidation interval not reached",
                "next_consolidation": self.consolidation_interval - time_since_last
            }
        
        self.last_consolidation = now
        
        # Get items from working memory module
        if self.event_bus:
            # Request working memory items
            self.publish_message("working_memory_request", {
                "action": "get_items",
                "requester": self.module_id
            })
            
            # Note: In a real system, you'd handle the response asynchronously
            # For simplicity, we'll simulate it with a direct consolidation
            
            consolidation_event = MemoryConsolidationEvent(
                memory_ids=[],
                strength_changes={},
                reason="sleep"
            )
            
            # Publish consolidation event
            self.publish_message("memory_consolidation", {
                "event": consolidation_event.model_dump()
            })
            
            return {
                "status": "success",
                "consolidated_count": len(consolidation_event.memory_ids)
            }
        
        return {"status": "error", "message": "No event bus available for consolidation"}
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories in long-term storage"""
        return list(self.memories.values())
    
    def count_memories(self) -> int:
        """Count the number of stored memories"""
        return len(self.memories)
    
    def save_state(self) -> str:
        """
        Save the current state of long-term memory
        
        Returns:
        Path to saved state file
        """
        # Save vector store
        vector_path = self.vector_store.save()
        
        # Save memories to disk
        for memory_id, memory in self.memories.items():
            self._save_memory(memory)
        
        return self.storage_dir
    
    def _save_memory(self, memory: Memory) -> None:
        """Save a single memory to disk"""
        try:
            memory_path = Path(self.storage_dir) / f"{memory.id}.json"
            with open(memory_path, "w") as f:
                # We need to convert the memory to a dict and handle datetime objects
                memory_dict = memory.model_dump()
                # Convert datetime to string
                for key, value in memory_dict.items():
                    if isinstance(value, datetime):
                        memory_dict[key] = value.isoformat()
                json.dump(memory_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving memory {memory.id}: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk"""
        try:
            memory_path = Path(self.storage_dir)
            for file_path in memory_path.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        memory_data = json.load(f)
                        # Convert string back to datetime
                        if "timestamp" in memory_data and isinstance(memory_data["timestamp"], str):
                            memory_data["timestamp"] = datetime.fromisoformat(memory_data["timestamp"])
                        if "last_accessed" in memory_data and memory_data["last_accessed"] and isinstance(memory_data["last_accessed"], str):
                            memory_data["last_accessed"] = datetime.fromisoformat(memory_data["last_accessed"])
                        
                        # Create memory object
                        memory = Memory(**memory_data)
                        self.memories[memory.id] = memory
                        
                        # Add to vector store if embedding exists
                        if memory.embedding:
                            self.vector_store.add(
                                embeddings=[memory.embedding],
                                metadata_list=[{"id": memory.id, "content": memory.content}]
                            )
                except Exception as e:
                    print(f"Error loading memory from {file_path}: {e}")
            
            print(f"Loaded {len(self.memories)} memories from disk")
        except Exception as e:
            print(f"Error loading memories: {e}")
    
    def _delete_memory_file(self, memory_id: str) -> None:
        """Delete a memory file from disk"""
        try:
            memory_path = Path(self.storage_dir) / f"{memory_id}.json"
            if memory_path.exists():
                memory_path.unlink()
        except Exception as e:
            print(f"Error deleting memory file {memory_id}: {e}")
    
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
    
    def _handle_working_memory_update(self, message: Message) -> None:
        """
        Handle updates from working memory
        
        If an item is removed from working memory and has high activation,
        it may be consolidated to long-term memory.
        """
        content = message.content
        action = content.get("action")
        
        if action == "remove":
            # Item was removed from working memory
            item_id = content.get("item_id")
            item_content = content.get("content")
            
            if item_content and item_id:
                # Check if this should be consolidated
                activation = content.get("activation", 0.0)
                importance = content.get("importance", 0.5)
                
                if activation >= self.consolidation_threshold:
                    # Store in long-term memory
                    self.store_memory({
                        "content": item_content,
                        "importance": importance,
                        "source_id": item_id
                    })
    
    def _handle_memory_search(self, message: Message) -> None:
        """Handle memory search requests"""
        content = message.content
        query = content.get("query", "")
        limit = content.get("limit", 5)
        
        if query:
            results = self.search_memories(query, None, limit)
            
            if self.event_bus and results.get("status") == "success":
                # Publish results
                self.publish_message("memory_search_response", {
                    "requester": message.sender,
                    "results": results.get("results", []),
                    "query": query
                })
    
    def _handle_consolidation_trigger(self, message: Message) -> None:
        """Handle explicit consolidation triggers"""
        self.consolidate_memories() 
