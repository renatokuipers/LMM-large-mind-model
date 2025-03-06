"""
Memory Component

This module implements the memory systems of the Child's mind, including:
- Episodic memory (specific events and experiences)
- Semantic memory (facts and knowledge)
- Working memory (short-term processing)

The memory systems use vector embeddings and similarity search to provide
human-like memory retrieval with appropriate limitations and imperfections.
"""

import os
import uuid
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import faiss
from tqdm import tqdm
import requests

from ..utils.data_types import (
    Memory, EpisodicMemory, SemanticMemory, MemoryType,
    Emotion, EmotionType, DevelopmentalStage
)
from llm_module import LLMClient, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemorySystem:
    """
    The MemorySystem class manages different types of memory for the Child.
    
    It handles storage, retrieval, and forgetting of memories, as well as
    associations between different memories.
    """
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        embedding_client: Optional[LLMClient] = None,
        faiss_index_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the memory system.
        
        Args:
            embedding_dimension: Dimension of memory embeddings
            embedding_client: Client for generating embeddings
            faiss_index_path: Path to load/save FAISS indexes
            vector_db_path: Path to store vector database
            use_gpu: Whether to use GPU for FAISS operations
        """
        self.embedding_dimension = embedding_dimension
        self.embedding_client = embedding_client or LLMClient()
        self.faiss_index_path = faiss_index_path or "./data/faiss_indexes"
        self.vector_db_path = vector_db_path or "./data/vector_db"
        self.use_gpu = use_gpu
        
        # Create directory structure
        os.makedirs(self.faiss_index_path, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Initialize memory stores
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        
        # Initialize FAISS indexes for different memory types
        self._init_faiss_indexes()
        
        # Memory embeddings cache
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info("Memory system initialized")
    
    def _init_faiss_indexes(self):
        """Initialize or load FAISS indexes for memory retrieval."""
        # Create or load episodic memory index
        episodic_path = os.path.join(self.faiss_index_path, "episodic_memory.index")
        
        if os.path.exists(episodic_path):
            self.episodic_index = faiss.read_index(episodic_path)
            logger.info(f"Loaded episodic memory index from {episodic_path}")
        else:
            self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info("Created new episodic memory index")
        
        # Create or load semantic memory index
        semantic_path = os.path.join(self.faiss_index_path, "semantic_memory.index")
        
        if os.path.exists(semantic_path):
            self.semantic_index = faiss.read_index(semantic_path)
            logger.info(f"Loaded semantic memory index from {semantic_path}")
        else:
            self.semantic_index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info("Created new semantic memory index")
        
        # Move to GPU if available and requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.episodic_index = faiss.index_cpu_to_gpu(res, 0, self.episodic_index)
                self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index)
                logger.info("Moved memory indexes to GPU")
            except Exception as e:
                logger.warning(f"Failed to move indexes to GPU: {e}")
    
    def store_episodic_memory(self, memory: EpisodicMemory) -> str:
        """
        Store an episodic memory.
        
        Args:
            memory: The episodic memory to store
            
        Returns:
            The memory ID
        """
        if not memory.id:
            memory.id = str(uuid.uuid4())
        
        # Store the memory
        self.episodic_memories[memory.id] = memory
        
        # Generate and store embedding
        embedding = self._generate_embedding(memory.event_description)
        self.memory_embeddings[memory.id] = embedding
        
        # Add to FAISS index
        faiss.normalize_L2(embedding.reshape(1, -1))  # Normalize for cosine similarity
        self.episodic_index.add(embedding.reshape(1, -1))
        
        logger.debug(f"Stored episodic memory: {memory.id}")
        return memory.id
    
    def store_semantic_memory(self, memory: SemanticMemory) -> str:
        """
        Store a semantic memory.
        
        Args:
            memory: The semantic memory to store
            
        Returns:
            The memory ID
        """
        if not memory.id:
            memory.id = str(uuid.uuid4())
        
        # Store the memory
        self.semantic_memories[memory.id] = memory
        
        # Generate and store embedding for concept+definition
        content = f"{memory.concept}: {memory.definition}"
        embedding = self._generate_embedding(content)
        self.memory_embeddings[memory.id] = embedding
        
        # Add to FAISS index
        faiss.normalize_L2(embedding.reshape(1, -1))  # Normalize for cosine similarity
        self.semantic_index.add(embedding.reshape(1, -1))
        
        logger.debug(f"Stored semantic memory: {memory.id}")
        return memory.id
    
    def retrieve_episodic_memories(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.6,
        developmental_stage: DevelopmentalStage = DevelopmentalStage.EARLY_ADULTHOOD
    ) -> List[EpisodicMemory]:
        """
        Retrieve episodic memories related to a query.
        
        Args:
            query: The search query
            limit: Maximum number of memories to retrieve
            min_similarity: Minimum similarity threshold
            developmental_stage: The child's developmental stage (affects recall ability)
            
        Returns:
            List of relevant episodic memories
        """
        if not self.episodic_memories:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Adjust search parameters based on developmental stage
        adjusted_limit = self._adjust_recall_limit(limit, developmental_stage)
        adjusted_threshold = self._adjust_recall_threshold(min_similarity, developmental_stage)
        
        # Search for similar memories
        distances, indices = self.episodic_index.search(
            query_embedding.reshape(1, -1), 
            min(adjusted_limit, self.episodic_index.ntotal)
        )
        
        # Convert indices to memory objects
        results = []
        memory_ids = list(self.episodic_memories.keys())
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(memory_ids):
                continue
                
            # Convert distance to similarity (FAISS returns L2 distance)
            similarity = 1.0 / (1.0 + dist)
            
            if similarity < adjusted_threshold:
                continue
                
            memory_id = memory_ids[idx]
            memory = self.episodic_memories[memory_id]
            
            # Update access time and apply memory decay
            memory.last_accessed = datetime.now()
            self._apply_memory_decay(memory)
            
            # Only include if memory strength is sufficient
            if memory.strength >= 0.2:
                results.append(memory)
        
        logger.debug(f"Retrieved {len(results)} episodic memories for query: {query[:30]}...")
        return results
    
    def retrieve_semantic_memories(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.6,
        developmental_stage: DevelopmentalStage = DevelopmentalStage.EARLY_ADULTHOOD
    ) -> List[SemanticMemory]:
        """
        Retrieve semantic memories related to a query.
        
        Args:
            query: The search query
            limit: Maximum number of memories to retrieve
            min_similarity: Minimum similarity threshold
            developmental_stage: The child's developmental stage (affects recall ability)
            
        Returns:
            List of relevant semantic memories
        """
        if not self.semantic_memories:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Adjust search parameters based on developmental stage
        adjusted_limit = self._adjust_recall_limit(limit, developmental_stage)
        adjusted_threshold = self._adjust_recall_threshold(min_similarity, developmental_stage)
        
        # Search for similar memories
        distances, indices = self.semantic_index.search(
            query_embedding.reshape(1, -1), 
            min(adjusted_limit, self.semantic_index.ntotal)
        )
        
        # Convert indices to memory objects
        results = []
        memory_ids = list(self.semantic_memories.keys())
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(memory_ids):
                continue
                
            # Convert distance to similarity (FAISS returns L2 distance)
            similarity = 1.0 / (1.0 + dist)
            
            if similarity < adjusted_threshold:
                continue
                
            memory_id = memory_ids[idx]
            memory = self.semantic_memories[memory_id]
            
            # Update access time and apply memory decay
            memory.last_accessed = datetime.now()
            self._apply_memory_decay(memory)
            
            # Only include if confidence is sufficient
            if memory.confidence >= 0.3:
                results.append(memory)
        
        logger.debug(f"Retrieved {len(results)} semantic memories for query: {query[:30]}...")
        return results
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector from text.

        Args:
            text: The text to generate an embedding for

        Returns:
            A list of floats representing the embedding vector
        """
        try:
            response = requests.post(
                self.embedding_endpoint,
                json={"input": text, "model": "text-embedding-ada-002"}
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            
            # Check if the embedding dimension matches what we expect
            if len(embedding) != self.embedding_dimension:
                logger.warning(
                    f"Expected embedding dimension {self.embedding_dimension}, "
                    f"but got {len(embedding)}. Using zero vector."
                )
                return [0.0] * self.embedding_dimension
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    # Add compatibility method for tests
    def _get_embedding(self, text: str) -> List[float]:
        """Compatibility method for tests that calls _generate_embedding.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding vector
        """
        return self._generate_embedding(text)
    
    def _apply_memory_decay(self, memory: Memory):
        """
        Apply decay to a memory based on time since last access.
        
        Args:
            memory: The memory to decay
        """
        # Calculate time since creation in days
        days_since_creation = (datetime.now() - memory.created_at).days
        
        # Calculate time since last access in days
        days_since_access = (datetime.now() - memory.last_accessed).days
        
        # Base decay factor (higher = faster decay)
        base_decay = memory.decay_rate
        
        # Episodic memories decay faster than semantic
        type_factor = 1.2 if memory.type == MemoryType.EPISODIC else 0.7
        
        # Calculate decay amount
        decay_amount = base_decay * type_factor * (
            0.01 * days_since_creation + 0.03 * days_since_access
        )
        
        # Apply decay to memory strength
        memory.strength = max(0.0, memory.strength - decay_amount)
        
        # For semantic memories, also decay confidence
        if isinstance(memory, SemanticMemory):
            confidence_decay = decay_amount * 0.7  # Confidence decays more slowly
            memory.confidence = max(0.0, memory.confidence - confidence_decay)
    
    def _adjust_recall_limit(self, limit: int, stage: DevelopmentalStage) -> int:
        """
        Adjust recall limit based on developmental stage.
        
        Args:
            limit: The base limit
            stage: The developmental stage
            
        Returns:
            Adjusted limit
        """
        # Memory capacity increases with development
        stage_factors = {
            DevelopmentalStage.INFANCY: 0.2,  # Very limited recall
            DevelopmentalStage.EARLY_CHILDHOOD: 0.4,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.6,
            DevelopmentalStage.ADOLESCENCE: 0.8,
            DevelopmentalStage.EARLY_ADULTHOOD: 1.0  # Full capacity
        }
        
        factor = stage_factors.get(stage, 0.5)
        adjusted = max(1, int(limit * factor))
        return adjusted
    
    def _adjust_recall_threshold(self, threshold: float, stage: DevelopmentalStage) -> float:
        """
        Adjust recall similarity threshold based on developmental stage.
        
        Args:
            threshold: The base threshold
            stage: The developmental stage
            
        Returns:
            Adjusted threshold
        """
        # Higher stages have better recall (lower threshold)
        stage_adjustments = {
            DevelopmentalStage.INFANCY: 0.2,  # Needs very close match
            DevelopmentalStage.EARLY_CHILDHOOD: 0.15,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.1,
            DevelopmentalStage.ADOLESCENCE: 0.05,
            DevelopmentalStage.EARLY_ADULTHOOD: 0.0  # No adjustment
        }
        
        adjustment = stage_adjustments.get(stage, 0.1)
        adjusted = threshold + adjustment
        return min(0.95, adjusted)  # Cap at 0.95
    
    def consolidate_memories(self):
        """
        Consolidate memories by strengthening important ones and removing weak ones.
        This simulates sleep and memory consolidation processes.
        """
        logger.info("Consolidating memories...")
        
        # Strengthen memories with high emotional impact
        strengthened = 0
        for memory_id, memory in list(self.episodic_memories.items()):
            if isinstance(memory, EpisodicMemory) and abs(memory.emotional_valence) > 0.7:
                # Strong emotional memories are strengthened
                memory.strength = min(1.0, memory.strength + 0.1)
                memory.decay_rate = max(0.05, memory.decay_rate - 0.02)  # More resilient
                strengthened += 1
        
        # Remove weak memories
        removed_episodic = 0
        for memory_id, memory in list(self.episodic_memories.items()):
            if memory.strength < 0.1:
                del self.episodic_memories[memory_id]
                removed_episodic += 1
        
        removed_semantic = 0
        for memory_id, memory in list(self.semantic_memories.items()):
            if isinstance(memory, SemanticMemory) and memory.confidence < 0.2:
                del self.semantic_memories[memory_id]
                removed_semantic += 1
        
        logger.info(f"Memory consolidation: strengthened {strengthened}, "
                    f"removed {removed_episodic} episodic and {removed_semantic} semantic memories")
        
        # Rebuild indexes if significant changes occurred
        if removed_episodic > 0 or removed_semantic > 0:
            self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Rebuild FAISS indexes after significant memory changes."""
        logger.info("Rebuilding memory indexes...")
        
        # Rebuild episodic index
        self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
        for memory_id, memory in self.episodic_memories.items():
            if memory_id in self.memory_embeddings:
                embedding = self.memory_embeddings[memory_id].reshape(1, -1)
                faiss.normalize_L2(embedding)
                self.episodic_index.add(embedding)
        
        # Rebuild semantic index
        self.semantic_index = faiss.IndexFlatL2(self.embedding_dimension)
        for memory_id, memory in self.semantic_memories.items():
            if memory_id in self.memory_embeddings:
                embedding = self.memory_embeddings[memory_id].reshape(1, -1)
                faiss.normalize_L2(embedding)
                self.semantic_index.add(embedding)
        
        # Move to GPU if available and requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.episodic_index = faiss.index_cpu_to_gpu(res, 0, self.episodic_index)
                self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index)
            except Exception as e:
                logger.warning(f"Failed to move rebuilt indexes to GPU: {e}")
    
    def save_state(self):
        """Save memory state to disk."""
        logger.info("Saving memory system state...")
        
        # Save FAISS indexes
        episodic_path = os.path.join(self.faiss_index_path, "episodic_memory.index")
        semantic_path = os.path.join(self.faiss_index_path, "semantic_memory.index")
        
        # If using GPU, move to CPU for saving
        if self.use_gpu:
            cpu_episodic = faiss.index_gpu_to_cpu(self.episodic_index)
            cpu_semantic = faiss.index_gpu_to_cpu(self.semantic_index)
            faiss.write_index(cpu_episodic, episodic_path)
            faiss.write_index(cpu_semantic, semantic_path)
        else:
            faiss.write_index(self.episodic_index, episodic_path)
            faiss.write_index(self.semantic_index, semantic_path)
        
        # Save memory data
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Save episodic memories
        episodic_data = {}
        for memory_id, memory in self.episodic_memories.items():
            episodic_data[memory_id] = memory.model_dump()
            
        with open(os.path.join(self.vector_db_path, "episodic_memories.json"), 'w') as f:
            json.dump(episodic_data, f, indent=2, default=str)
        
        # Save semantic memories
        semantic_data = {}
        for memory_id, memory in self.semantic_memories.items():
            semantic_data[memory_id] = memory.model_dump()
            
        with open(os.path.join(self.vector_db_path, "semantic_memories.json"), 'w') as f:
            json.dump(semantic_data, f, indent=2, default=str)
        
        # Save embeddings
        embeddings_data = {}
        for memory_id, embedding in self.memory_embeddings.items():
            embeddings_data[memory_id] = embedding.tolist()
            
        with open(os.path.join(self.vector_db_path, "memory_embeddings.json"), 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info("Memory system state saved")
    
    def load_state(self):
        """Load memory state from disk."""
        logger.info("Loading memory system state...")
        
        # Load FAISS indexes (already done in __init__)
        
        # Load memory data
        try:
            # Load episodic memories
            episodic_path = os.path.join(self.vector_db_path, "episodic_memories.json")
            if os.path.exists(episodic_path):
                with open(episodic_path, 'r') as f:
                    episodic_data = json.load(f)
                    
                self.episodic_memories = {}
                for memory_id, memory_dict in episodic_data.items():
                    # Convert string dates back to datetime
                    if 'created_at' in memory_dict and isinstance(memory_dict['created_at'], str):
                        memory_dict['created_at'] = datetime.fromisoformat(memory_dict['created_at'].replace('Z', '+00:00'))
                    if 'last_accessed' in memory_dict and isinstance(memory_dict['last_accessed'], str):
                        memory_dict['last_accessed'] = datetime.fromisoformat(memory_dict['last_accessed'].replace('Z', '+00:00'))
                        
                    self.episodic_memories[memory_id] = EpisodicMemory(**memory_dict)
            
            # Load semantic memories
            semantic_path = os.path.join(self.vector_db_path, "semantic_memories.json")
            if os.path.exists(semantic_path):
                with open(semantic_path, 'r') as f:
                    semantic_data = json.load(f)
                    
                self.semantic_memories = {}
                for memory_id, memory_dict in semantic_data.items():
                    # Convert string dates back to datetime
                    if 'created_at' in memory_dict and isinstance(memory_dict['created_at'], str):
                        memory_dict['created_at'] = datetime.fromisoformat(memory_dict['created_at'].replace('Z', '+00:00'))
                    if 'last_accessed' in memory_dict and isinstance(memory_dict['last_accessed'], str):
                        memory_dict['last_accessed'] = datetime.fromisoformat(memory_dict['last_accessed'].replace('Z', '+00:00'))
                        
                    self.semantic_memories[memory_id] = SemanticMemory(**memory_dict)
            
            # Load embeddings
            embeddings_path = os.path.join(self.vector_db_path, "memory_embeddings.json")
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'r') as f:
                    embeddings_data = json.load(f)
                    
                self.memory_embeddings = {}
                for memory_id, embedding_list in embeddings_data.items():
                    self.memory_embeddings[memory_id] = np.array(embedding_list, dtype=np.float32)
            
            logger.info(f"Loaded {len(self.episodic_memories)} episodic and {len(self.semantic_memories)} semantic memories")
            
        except Exception as e:
            logger.error(f"Error loading memory state: {e}")
    
    def simulate_memory_exercise(self):
        """
        Simulate memory exercise to strengthen important memories.
        This can be used for 'studying' or rehearsal.
        """
        # Strengthen recently accessed memories
        recent_cutoff = datetime.now() - timedelta(days=1)
        strengthened = 0
        
        for memory in self.episodic_memories.values():
            if memory.last_accessed > recent_cutoff:
                memory.strength = min(1.0, memory.strength + 0.05)
                strengthened += 1
        
        for memory in self.semantic_memories.values():
            if memory.last_accessed > recent_cutoff:
                memory.strength = min(1.0, memory.strength + 0.05)
                memory.confidence = min(1.0, memory.confidence + 0.05)
                strengthened += 1
        
        logger.info(f"Memory exercise: strengthened {strengthened} recently accessed memories")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "episodic_count": len(self.episodic_memories),
            "semantic_count": len(self.semantic_memories),
            "episodic_index_size": self.episodic_index.ntotal,
            "semantic_index_size": self.semantic_index.ntotal,
            "average_episodic_strength": 0.0,
            "average_semantic_confidence": 0.0,
            "memory_embeddings_count": len(self.memory_embeddings)
        }
        
        if self.episodic_memories:
            stats["average_episodic_strength"] = sum(m.strength for m in self.episodic_memories.values()) / len(self.episodic_memories)
            
        if self.semantic_memories:
            stats["average_semantic_confidence"] = sum(m.confidence for m in self.semantic_memories.values()) / len(self.semantic_memories)
        
        return stats 