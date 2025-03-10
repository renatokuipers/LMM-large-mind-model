"""
Storage Module

This module provides unified storage capabilities for the LMM system including:
- Vector embeddings storage for semantic representation
- State persistence for saving and loading system states
- Experience logging for recording developmental experiences

These storage systems form the persistent memory backbone of the LMM,
allowing for long-term development and retrieval of past experiences.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Import storage components
from .vector_db import VectorDB
from .state_persistence import StatePersistence
from .experience_logger import ExperienceLogger

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Central manager for all storage functionality
    
    This class provides a unified interface for vector storage, state persistence,
    and experience logging, ensuring consistent configuration and access patterns.
    """
    
    def __init__(
        self,
        storage_dir: str = "storage",
        vector_dimension: int = 768,
        vector_index_type: str = "Flat",
        use_gpu: bool = True
    ):
        """
        Initialize the storage manager
        
        Args:
            storage_dir: Base directory for all storage
            vector_dimension: Dimension for vector embeddings
            vector_index_type: Type of FAISS index to use
            use_gpu: Whether to use GPU acceleration for vectors when available
        """
        self.storage_dir = storage_dir
        self._create_directories()
        
        # Initialize storage components
        self.vector_db = VectorDB(
            dimension=vector_dimension,
            index_type=vector_index_type,
            storage_dir=os.path.join(storage_dir, "embeddings"),
            use_gpu=use_gpu
        )
        
        self.state_persistence = StatePersistence(
            storage_dir=storage_dir
        )
        
        self.experience_logger = ExperienceLogger(
            storage_dir=storage_dir
        )
        
        logger.info(f"Storage manager initialized at {storage_dir}")
    
    def _create_directories(self) -> None:
        """Create necessary directories for storage"""
        base_dir = Path(self.storage_dir)
        base_dir.mkdir(exist_ok=True)
        
        for subdir in ["embeddings", "states", "experiences", "backups"]:
            (base_dir / subdir).mkdir(exist_ok=True)
    
    # -------------------------
    # Vector Storage Operations
    # -------------------------
    
    def store_embeddings(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Store vectors with associated metadata
        
        Args:
            vectors: Numpy array of vectors to store
            metadata_list: List of metadata dictionaries for each vector
            ids: Optional list of IDs for the vectors
            
        Returns:
            List of IDs for the stored vectors
        """
        return self.vector_db.add(vectors, metadata_list, ids)
    
    def generate_and_store_embeddings(
        self,
        texts: Union[str, List[str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        llm_client = None,
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ) -> List[str]:
        """
        Generate embeddings for texts and store them with metadata
        
        This method uses the LLMClient to generate embeddings and then
        stores them in the vector database.
        
        Args:
            texts: String or list of strings to generate embeddings for
            metadata_list: List of metadata dictionaries for each text
            ids: Optional list of IDs for the vectors
            llm_client: LLMClient instance to use for embedding generation
            embedding_model: Model to use for embedding generation
            
        Returns:
            List of IDs for the stored embeddings
        """
        if llm_client is None:
            from lmm_project.utils.llm_client import LLMClient
            llm_client = LLMClient()
        
        # Create default metadata if not provided
        if metadata_list is None:
            if isinstance(texts, str):
                metadata_list = [{"text": texts}]
            else:
                metadata_list = [{"text": text} for text in texts]
        elif isinstance(texts, str) and len(metadata_list) == 1:
            # Single text with metadata
            if "text" not in metadata_list[0]:
                metadata_list[0]["text"] = texts
        elif isinstance(texts, list):
            # Ensure metadata has text field and matches length
            if len(metadata_list) != len(texts):
                raise ValueError(f"Length of metadata_list ({len(metadata_list)}) must match length of texts ({len(texts)})")
            for i, meta in enumerate(metadata_list):
                if "text" not in meta:
                    meta["text"] = texts[i]
        
        # Generate embeddings
        try:
            embeddings = llm_client.get_embedding(texts, embedding_model)
            
            # Convert to numpy array
            if isinstance(texts, str):
                vectors = np.array([embeddings])
                metadata_list = [metadata_list[0]]
            else:
                vectors = np.array(embeddings)
            
            # Store embeddings
            return self.store_embeddings(vectors, metadata_list, ids)
        except Exception as e:
            logger.error(f"Error generating or storing embeddings: {str(e)}")
            raise
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Optional = None
    ) -> List[Dict[str, Any]]:
        """
        Find vectors similar to the query vector
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filter_fn: Optional function to filter results
            
        Returns:
            List of match dictionaries
        """
        return self.vector_db.search(query_vector, k, filter_fn)
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 10,
        filter_fn: Optional = None,
        llm_client = None,
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ) -> List[Dict[str, Any]]:
        """
        Find vectors similar to the embedding of the query text
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            filter_fn: Optional function to filter results
            llm_client: LLMClient instance to use for embedding generation
            embedding_model: Model to use for embedding generation
            
        Returns:
            List of match dictionaries
        """
        if llm_client is None:
            from lmm_project.utils.llm_client import LLMClient
            llm_client = LLMClient()
        
        try:
            # Generate embedding for query text
            query_embedding = llm_client.get_embedding(query_text, embedding_model)
            query_vector = np.array(query_embedding)
            
            # Search for similar vectors
            return self.search_similar(query_vector, k, filter_fn)
        except Exception as e:
            logger.error(f"Error in text similarity search: {str(e)}")
            raise
    
    def save_vector_index(self) -> str:
        """
        Save the current vector index to disk
        
        Returns:
            Path to the saved index
        """
        return self.vector_db.save()
    
    # -------------------------
    # State Persistence Operations
    # -------------------------
    
    def save_system_state(
        self,
        mind_state: Dict[str, Any],
        label: Optional[str] = None,
        description: Optional[str] = None,
        is_checkpoint: bool = False
    ) -> str:
        """
        Save the current state of the system
        
        Args:
            mind_state: Dictionary containing the system state
            label: Optional label for the state
            description: Optional description of the state
            is_checkpoint: Whether this is an important checkpoint
            
        Returns:
            ID of the saved state
        """
        return self.state_persistence.save_state(
            mind_state=mind_state,
            label=label,
            description=description,
            is_checkpoint=is_checkpoint
        )
    
    def load_system_state(
        self,
        state_id: Optional[str] = None,
        use_latest: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load a saved system state
        
        Args:
            state_id: ID of the state to load
            use_latest: Whether to load the latest state if no ID provided
            
        Returns:
            Loaded state dictionary or None if not found
        """
        return self.state_persistence.load_state(
            state_id=state_id,
            use_latest=use_latest
        )
    
    def list_saved_states(
        self,
        limit: int = 100,
        checkpoints_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List saved system states
        
        Args:
            limit: Maximum number of states to list
            checkpoints_only: Whether to only list checkpoints
            
        Returns:
            List of state information dictionaries
        """
        return self.state_persistence.list_states(
            limit=limit,
            checkpoints_only=checkpoints_only
        )
    
    # -------------------------
    # Experience Logging Operations
    # -------------------------
    
    def log_experience(
        self,
        experience_data: Dict[str, Any],
        experience_type: str,
        source: str,
        emotional_valence: str = "neutral",
        emotional_intensity: float = 0.5,
        importance_score: float = 0.5,
        tags: List[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Log an experience
        
        Args:
            experience_data: Dictionary containing experience data
            experience_type: Type of experience (e.g., "interaction", "perception")
            source: Source of the experience (e.g., "mother", "environment")
            emotional_valence: Emotional tone (e.g., "positive", "negative")
            emotional_intensity: Intensity of emotion (0.0 to 1.0)
            importance_score: Importance of the experience (0.0 to 1.0)
            tags: List of tags for the experience
            embedding: Optional vector embedding of the experience
            
        Returns:
            ID of the logged experience
        """
        return self.experience_logger.log_experience(
            experience_data=experience_data,
            experience_type=experience_type,
            source=source,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            importance_score=importance_score,
            tags=tags,
            embedding=embedding
        )
    
    def log_experience_with_embedding(
        self,
        experience_data: Dict[str, Any],
        experience_type: str,
        source: str,
        text_for_embedding: str,
        emotional_valence: str = "neutral",
        emotional_intensity: float = 0.5,
        importance_score: float = 0.5,
        tags: List[str] = None,
        llm_client = None,
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ) -> str:
        """
        Log an experience with automatically generated embedding
        
        This method generates an embedding for the provided text and then
        logs the experience with that embedding.
        
        Args:
            experience_data: Dictionary containing experience data
            experience_type: Type of experience (e.g., "interaction", "perception")
            source: Source of the experience (e.g., "mother", "environment")
            text_for_embedding: Text to generate embedding from
            emotional_valence: Emotional tone (e.g., "positive", "negative")
            emotional_intensity: Intensity of emotion (0.0 to 1.0)
            importance_score: Importance of the experience (0.0 to 1.0)
            tags: List of tags for the experience
            llm_client: LLMClient instance to use for embedding generation
            embedding_model: Model to use for embedding generation
            
        Returns:
            ID of the logged experience
        """
        if llm_client is None:
            from lmm_project.utils.llm_client import LLMClient
            llm_client = LLMClient()
        
        try:
            # Generate embedding
            embedding_vector = llm_client.get_embedding(text_for_embedding, embedding_model)
            embedding = np.array(embedding_vector)
            
            # Log experience with embedding
            return self.log_experience(
                experience_data=experience_data,
                experience_type=experience_type,
                source=source,
                emotional_valence=emotional_valence,
                emotional_intensity=emotional_intensity,
                importance_score=importance_score,
                tags=tags,
                embedding=embedding
            )
        except Exception as e:
            logger.error(f"Error generating embedding for experience: {str(e)}")
            # Fall back to logging without embedding
            logger.info("Falling back to logging experience without embedding")
            return self.log_experience(
                experience_data=experience_data,
                experience_type=experience_type,
                source=source,
                emotional_valence=emotional_valence,
                emotional_intensity=emotional_intensity,
                importance_score=importance_score,
                tags=tags
            )
    
    def query_experiences(
        self,
        experience_type: Optional[str] = None,
        source: Optional[str] = None,
        time_range: Optional[Tuple] = None,
        emotional_valence: Optional[str] = None,
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query experiences matching criteria
        
        Args:
            experience_type: Type of experience to query
            source: Source of experiences to query
            time_range: Tuple of (start_time, end_time)
            emotional_valence: Emotional tone to filter by
            min_importance: Minimum importance score
            tags: List of tags to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching experience dictionaries
        """
        return self.experience_logger.query_experiences(
            experience_type=experience_type,
            source=source,
            time_range=time_range,
            emotional_valence=emotional_valence,
            min_importance=min_importance,
            tags=tags,
            limit=limit
        )
    
    def get_experience_timeline(
        self,
        limit: int = 100,
        group_by_day: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get a timeline of experiences
        
        Args:
            limit: Maximum number of experiences
            group_by_day: Whether to group experiences by day
            
        Returns:
            List of experience information dictionaries
        """
        return self.experience_logger.get_experience_timeline(
            limit=limit,
            group_by_day=group_by_day
        )
    
    def close(self) -> None:
        """Close all storage components properly"""
        self.vector_db.save()
        self.state_persistence.close()
        self.experience_logger.close()
        logger.info("Storage manager closed")

def create_storage_manager(
    storage_dir: str = "storage",
    vector_dimension: int = 768,
    vector_index_type: str = "Flat",
    use_gpu: bool = True
) -> StorageManager:
    """
    Create and return a storage manager
    
    Args:
        storage_dir: Base directory for all storage
        vector_dimension: Dimension for vector embeddings
        vector_index_type: Type of FAISS index to use
        use_gpu: Whether to use GPU acceleration for vectors
        
    Returns:
        Initialized StorageManager
    """
    return StorageManager(
        storage_dir=storage_dir,
        vector_dimension=vector_dimension,
        vector_index_type=vector_index_type,
        use_gpu=use_gpu
    )

# Exports
__all__ = [
    # Classes
    'StorageManager',
    'VectorDB',
    'StatePersistence',
    'ExperienceLogger',
    
    # Functions
    'create_storage_manager'
] 
