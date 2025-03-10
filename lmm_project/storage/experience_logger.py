"""
Experience logger implementation for the LMM project.

This module provides mechanisms for logging and retrieving experiences
with embeddings for semantic search, temporal sequencing, and categorization.
"""
import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from pydantic import BaseModel, Field, root_validator

from lmm_project.storage.vector_db import VectorDB
from lmm_project.utils.llm_client import LLMClient
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("storage.experience_logger")


class ExperienceMetadata(BaseModel):
    """Metadata for an experience entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    experience_type: str
    source: str
    emotional_valence: str = Field(default="neutral")
    emotional_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class ExperienceLogger:
    """
    Logger for recording and retrieving experiences.
    
    Maintains a database of experiences with embeddings for semantic search,
    emotional tagging, and temporally-ordered retrieval for the development of
    episodic memory and self-concept.
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDB] = None,
        experience_collection: str = "experiences",
        storage_dir: str = "storage/experiences",
        embedding_dimension: int = 768
    ):
        """
        Initialize the experience logger.
        
        Parameters:
        vector_db: Optional existing vector database
        experience_collection: Collection name for experiences
        storage_dir: Directory to store experience data
        embedding_dimension: Dimension of experience embeddings
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or use provided vector database
        if vector_db is None:
            self.vector_db = VectorDB({
                "dimension": embedding_dimension,
                "storage_dir": str(self.storage_dir / "vectors")
            })
        else:
            self.vector_db = vector_db
            
        # Create experience collection if it doesn't exist
        self.experience_collection = experience_collection
        self.vector_db.create_collection(experience_collection)
        
        # Experience timeline (for sequential access)
        self.timeline_path = self.storage_dir / "timeline.json"
        self.timeline: List[Dict[str, Any]] = []
        self._load_timeline()
        
        # Initialize LLM client for generating embeddings
        self.llm_client = None  # Lazy initialization
        
        logger.info(f"Initialized ExperienceLogger with collection '{experience_collection}'")
    
    def _load_timeline(self) -> None:
        """Load experience timeline from disk."""
        if self.timeline_path.exists():
            try:
                with open(self.timeline_path, 'r') as f:
                    self.timeline = json.load(f)
                logger.info(f"Loaded {len(self.timeline)} experiences from timeline")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load experience timeline: {e}")
                self.timeline = []
    
    def _save_timeline(self) -> None:
        """Save experience timeline to disk."""
        try:
            with open(self.timeline_path, 'w') as f:
                json.dump(self.timeline, f)
            logger.debug(f"Saved {len(self.timeline)} experiences to timeline")
        except IOError as e:
            logger.error(f"Failed to save experience timeline: {e}")
    
    def _get_llm_client(self) -> LLMClient:
        """Get or initialize LLM client."""
        if self.llm_client is None:
            # Use the default URL from environment variables
            self.llm_client = LLMClient()
        return self.llm_client
    
    def log_experience(
        self,
        content: Dict[str, Any],
        experience_type: str,
        source: str,
        embedding_text: Optional[str] = None,
        emotional_valence: str = "neutral",
        emotional_intensity: float = 0.5,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Log an experience to the database.
        
        Parameters:
        content: Experience content
        experience_type: Type of experience
        source: Source of the experience
        embedding_text: Optional text to use for generating embedding
        emotional_valence: Emotional valence (positive, negative, neutral)
        emotional_intensity: Intensity of emotion (0.0-1.0)
        importance_score: Importance of the experience (0.0-1.0)
        tags: Optional list of tags
        embedding: Optional pre-computed embedding vector
        
        Returns:
        ID of the logged experience
        """
        # Create metadata
        metadata = ExperienceMetadata(
            experience_type=experience_type,
            source=source,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            importance_score=importance_score,
            tags=tags or []
        )
        
        # Generate embedding if needed
        if embedding is None and embedding_text is not None:
            try:
                client = self._get_llm_client()
                embedding = client.get_embedding(embedding_text)
                # Convert to numpy array if needed
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Use zero vector as fallback
                embedding = np.zeros(self.vector_db.config.dimension)
        
        # Use zero vector if still no embedding
        if embedding is None:
            embedding = np.zeros(self.vector_db.config.dimension)
            
        # Add experience metadata to content
        full_metadata = {
            "id": metadata.id,
            "timestamp": metadata.timestamp,
            "experience_type": metadata.experience_type,
            "source": metadata.source,
            "emotional_valence": metadata.emotional_valence,
            "emotional_intensity": metadata.emotional_intensity,
            "importance_score": metadata.importance_score,
            "tags": metadata.tags,
            "content": content
        }
        
        # Add to vector database
        ids = self.vector_db.add_vectors(
            vectors=[embedding],
            metadata_list=[full_metadata],
            collection=self.experience_collection
        )
        
        # Add to timeline
        timeline_entry = {
            "id": metadata.id,
            "vector_id": ids[0] if ids else None,
            "timestamp": metadata.timestamp,
            "experience_type": metadata.experience_type,
            "source": metadata.source,
            "importance_score": metadata.importance_score,
            "tags": metadata.tags
        }
        self.timeline.append(timeline_entry)
        self._save_timeline()
        
        logger.debug(f"Logged experience {metadata.id} of type {experience_type}")
        return metadata.id
    
    def search_similar_experiences(
        self,
        query_text: str,
        k: int = 10,
        filter_by_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for experiences by semantic similarity.
        
        Parameters:
        query_text: Text to search for
        k: Number of results to return
        filter_by_type: Optional experience type filter
        filter_by_source: Optional source filter
        min_importance: Minimum importance score
        
        Returns:
        List of matching experiences
        """
        # Create embedding for query
        try:
            client = self._get_llm_client()
            query_embedding = client.get_embedding(query_text)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
            
        # Create metadata filter
        metadata_filter = {}
        if filter_by_type:
            metadata_filter["experience_type"] = filter_by_type
        if filter_by_source:
            metadata_filter["source"] = filter_by_source
        if min_importance > 0.0:
            # This will be handled in post-processing as vector_db doesn't support numeric comparisons
            pass
            
        # Search vector database
        results = self.vector_db.search(
            query_vector=query_embedding,
            k=k * 2,  # Get more results for post-filtering
            collection=self.experience_collection,
            metadata_filter=metadata_filter
        )
        
        # Post-process results for importance filter
        filtered_results = []
        for result in results:
            # Filter by importance if needed
            if min_importance > 0.0:
                importance = result["metadata"].get("importance_score", 0.0)
                if importance < min_importance:
                    continue
                    
            # Format result
            filtered_results.append({
                "id": result["metadata"].get("id"),
                "timestamp": result["metadata"].get("timestamp"),
                "experience_type": result["metadata"].get("experience_type"),
                "source": result["metadata"].get("source"),
                "emotional_valence": result["metadata"].get("emotional_valence"),
                "emotional_intensity": result["metadata"].get("emotional_intensity"),
                "importance_score": result["metadata"].get("importance_score"),
                "tags": result["metadata"].get("tags", []),
                "content": result["metadata"].get("content", {}),
                "similarity": result["similarity"]
            })
            
        # Return limited results
        return filtered_results[:k]
    
    def get_experience_by_id(self, experience_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an experience by its ID.
        
        Parameters:
        experience_id: ID of the experience
        
        Returns:
        Experience data or None if not found
        """
        # Find the vector ID from timeline
        vector_id = None
        for entry in self.timeline:
            if entry["id"] == experience_id:
                vector_id = entry.get("vector_id")
                break
                
        if vector_id is None:
            logger.warning(f"Experience {experience_id} not found in timeline")
            return None
            
        # Get metadata from vector database
        metadata = self.vector_db.vector_store.get_metadata(vector_id)
        if not metadata:
            logger.warning(f"Experience {experience_id} metadata not found in vector database")
            return None
            
        # Format result
        result = {
            "id": metadata.get("id"),
            "timestamp": metadata.get("timestamp"),
            "experience_type": metadata.get("experience_type"),
            "source": metadata.get("source"),
            "emotional_valence": metadata.get("emotional_valence"),
            "emotional_intensity": metadata.get("emotional_intensity"),
            "importance_score": metadata.get("importance_score"),
            "tags": metadata.get("tags", []),
            "content": metadata.get("content", {})
        }
        
        return result
    
    def get_recent_experiences(
        self,
        limit: int = 10,
        filter_by_type: Optional[str] = None,
        filter_by_source: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent experiences.
        
        Parameters:
        limit: Maximum number of experiences to return
        filter_by_type: Optional experience type filter
        filter_by_source: Optional source filter
        min_importance: Minimum importance score
        
        Returns:
        List of recent experiences
        """
        # Copy and reverse timeline (newest first)
        timeline = list(reversed(self.timeline))
        
        # Filter timeline
        filtered_timeline = []
        for entry in timeline:
            if filter_by_type and entry["experience_type"] != filter_by_type:
                continue
                
            if filter_by_source and entry["source"] != filter_by_source:
                continue
                
            if min_importance > 0.0 and entry["importance_score"] < min_importance:
                continue
                
            # Fetch full experience
            experience = self.get_experience_by_id(entry["id"])
            if experience:
                filtered_timeline.append(experience)
                
            # Check limit
            if len(filtered_timeline) >= limit:
                break
                
        return filtered_timeline
    
    def delete_experience(self, experience_id: str) -> bool:
        """
        Delete an experience.
        
        Parameters:
        experience_id: ID of the experience to delete
        
        Returns:
        True if successfully deleted
        """
        # Find the vector ID from timeline
        vector_id = None
        timeline_index = None
        for i, entry in enumerate(self.timeline):
            if entry["id"] == experience_id:
                vector_id = entry.get("vector_id")
                timeline_index = i
                break
                
        if vector_id is None:
            logger.warning(f"Experience {experience_id} not found in timeline")
            return False
            
        # Delete from vector database
        if vector_id is not None:
            self.vector_db.delete_vectors([vector_id], collection=self.experience_collection)
            
        # Remove from timeline
        if timeline_index is not None:
            self.timeline.pop(timeline_index)
            self._save_timeline()
            
        logger.debug(f"Deleted experience {experience_id}")
        return True
    
    def add_tag(self, experience_id: str, tag: str) -> bool:
        """
        Add a tag to an experience.
        
        Parameters:
        experience_id: ID of the experience
        tag: Tag to add
        
        Returns:
        True if successfully added
        """
        # Find the vector ID from timeline
        vector_id = None
        for entry in self.timeline:
            if entry["id"] == experience_id:
                vector_id = entry.get("vector_id")
                
                # Add tag to timeline entry
                if tag not in entry["tags"]:
                    entry["tags"].append(tag)
                    
                break
                
        if vector_id is None:
            logger.warning(f"Experience {experience_id} not found in timeline")
            return False
            
        # Update metadata in vector database
        metadata = self.vector_db.vector_store.get_metadata(vector_id)
        if metadata:
            if "tags" not in metadata:
                metadata["tags"] = []
                
            if tag not in metadata["tags"]:
                metadata["tags"].append(tag)
                self.vector_db.vector_store.update_metadata(vector_id, metadata)
                
            self._save_timeline()
            return True
            
        return False
    
    def save(self) -> None:
        """Save all data to disk."""
        self.vector_db.save()
        self._save_timeline()
        logger.info("Saved experience logger data to disk")
