"""
Vector database implementation for the LMM project.

This module provides efficient storage and retrieval of vector embeddings
using FAISS indices with both CPU and GPU support. It enables semantic
search and similarity-based operations for the LMM system.
"""
import os
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import faiss
import torch
from pydantic import BaseModel, Field, model_validator

from lmm_project.utils.vector_store import VectorStore, get_vector_store
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("storage.vector_db")


class VectorDBConfig(BaseModel):
    """Configuration for the vector database."""
    dimension: int = Field(default=768, ge=1)
    index_type: str = Field(default="Flat")
    use_gpu: bool = Field(default=True)
    storage_dir: str = Field(default="storage/vectors")
    metadata_filename: str = Field(default="metadata.json")
    
    @model_validator(mode='after')
    def validate_storage_dir(self) -> 'VectorDBConfig':
        """Ensure storage directory exists."""
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
        return self


class VectorDB:
    """
    Vector database for storing and retrieving embeddings.
    
    This class provides a high-level interface to FAISS indices for efficient
    similarity search and vector operations, with automatic GPU acceleration
    when available.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector database.
        
        Parameters:
        config: Configuration dictionary
        """
        self.config = VectorDBConfig(**(config or {}))
        
        # Initialize vector store from utils
        self.vector_store = get_vector_store(
            dimension=self.config.dimension,
            index_type=self.config.index_type,
            use_gpu=self.config.use_gpu,
            storage_dir=self.config.storage_dir
        )
        
        # Track collections (named subsets of vectors)
        self.collections: Dict[str, List[int]] = {}
        self._load_collections()
        
        logger.info(
            f"Initialized VectorDB with {self.config.index_type} index, "
            f"dimension={self.config.dimension}, GPU={self.config.use_gpu}"
        )
    
    def _load_collections(self) -> None:
        """Load collection metadata from disk."""
        metadata_path = os.path.join(self.config.storage_dir, self.config.metadata_filename)
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.collections = json.load(f)
                logger.info(f"Loaded {len(self.collections)} collections from metadata")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load collections metadata: {e}")
                self.collections = {}
    
    def _save_collections(self) -> None:
        """Save collection metadata to disk."""
        metadata_path = os.path.join(self.config.storage_dir, self.config.metadata_filename)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.collections, f)
            logger.debug(f"Saved {len(self.collections)} collections to metadata")
        except IOError as e:
            logger.error(f"Failed to save collections metadata: {e}")
    
    def add_vectors(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        collection: Optional[str] = None
    ) -> List[int]:
        """
        Add vectors to the database.
        
        Parameters:
        vectors: Vectors to add (numpy array or list of lists)
        metadata_list: Metadata for each vector
        collection: Optional collection name to group vectors
        
        Returns:
        List of assigned IDs
        """
        # Add vectors to vector store
        ids = self.vector_store.add_vectors(vectors, metadata_list)
        
        # Track in collection if specified
        if collection:
            if collection not in self.collections:
                self.collections[collection] = []
            self.collections[collection].extend(ids)
            self._save_collections()
            
        logger.debug(f"Added {len(ids)} vectors{f' to collection {collection}' if collection else ''}")
        return ids
    
    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        k: int = 10,
        collection: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Parameters:
        query_vector: Vector to search for
        k: Number of results to return
        collection: Optional collection to limit search to
        metadata_filter: Optional filter for metadata fields
        
        Returns:
        List of results with ids, distances, and metadata
        """
        # Limit search to collection if specified
        search_filter = None
        if collection:
            if collection not in self.collections:
                logger.warning(f"Collection {collection} does not exist")
                return []
                
            collection_ids = set(self.collections[collection])
            
            # Create filter function to limit search to collection
            def id_filter(id: int, metadata: Dict[str, Any]) -> bool:
                return id in collection_ids
                
            # Combine with metadata filter if provided
            if metadata_filter:
                original_filter = id_filter
                
                def combined_filter(id: int, metadata: Dict[str, Any]) -> bool:
                    if not original_filter(id, metadata):
                        return False
                    
                    # Check all metadata filter conditions
                    for key, value in metadata_filter.items():
                        if key not in metadata or metadata[key] != value:
                            return False
                    return True
                
                search_filter = combined_filter
            else:
                search_filter = id_filter
        elif metadata_filter:
            # Use only metadata filter
            def meta_only_filter(id: int, metadata: Dict[str, Any]) -> bool:
                for key, value in metadata_filter.items():
                    if key not in metadata or metadata[key] != value:
                        return False
                return True
                
            search_filter = meta_only_filter
            
        # Perform search
        ids, distances, metadatas = self.vector_store.search(
            query_vector, 
            k=k,
            metadata_filter=search_filter
        )
        
        # Format results
        results = []
        for i, (id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
            results.append({
                "id": id,
                "distance": float(distance),
                "similarity": float(1.0 - min(distance, 1.0)),  # Convert distance to similarity
                "metadata": metadata
            })
            
        return results
    
    def delete_vectors(self, ids: List[int], collection: Optional[str] = None) -> None:
        """
        Delete vectors from the database.
        
        Parameters:
        ids: IDs of vectors to delete
        collection: Optional collection to remove from (without deleting vectors)
        """
        if collection:
            # Just remove from collection
            if collection in self.collections:
                self.collections[collection] = [id for id in self.collections[collection] if id not in set(ids)]
                self._save_collections()
                logger.debug(f"Removed {len(ids)} vectors from collection {collection}")
        else:
            # Delete from vector store
            self.vector_store.delete(ids)
            
            # Remove from all collections
            for coll_name in self.collections:
                self.collections[coll_name] = [id for id in self.collections[coll_name] if id not in set(ids)]
            self._save_collections()
            
            logger.debug(f"Deleted {len(ids)} vectors from database")
    
    def create_collection(self, name: str, ids: Optional[List[int]] = None) -> str:
        """
        Create a new collection.
        
        Parameters:
        name: Collection name
        ids: Optional initial vector IDs to include
        
        Returns:
        Collection name
        """
        # Generate unique name if not provided or already exists
        if not name or name in self.collections:
            base_name = name or "collection"
            name = f"{base_name}_{uuid.uuid4().hex[:8]}"
            
        # Initialize collection
        self.collections[name] = ids or []
        self._save_collections()
        
        logger.debug(f"Created collection {name} with {len(self.collections[name])} vectors")
        return name
    
    def delete_collection(self, name: str, delete_vectors: bool = False) -> bool:
        """
        Delete a collection.
        
        Parameters:
        name: Collection name
        delete_vectors: Whether to also delete the vectors in the collection
        
        Returns:
        True if successfully deleted
        """
        if name not in self.collections:
            logger.warning(f"Collection {name} does not exist")
            return False
            
        # Optionally delete vectors
        if delete_vectors:
            self.vector_store.delete(self.collections[name])
            
        # Remove collection
        del self.collections[name]
        self._save_collections()
        
        logger.debug(f"Deleted collection {name}")
        return True
    
    def save(self) -> str:
        """
        Save the vector database to disk.
        
        Returns:
        Path to saved index
        """
        # Save vector store
        index_path = self.vector_store.save(name="main")
        
        # Save collections
        self._save_collections()
        
        logger.info(f"Saved vector database to {index_path}")
        return index_path
    
    def get_collection_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all collections.
        
        Returns:
        Dictionary of collection information
        """
        result = {}
        for name, ids in self.collections.items():
            result[name] = {
                "name": name,
                "vector_count": len(ids),
                "ids": ids[:5] + (["..."] if len(ids) > 5 else [])
            }
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
        Dictionary of statistics
        """
        # Get metadata for all vectors
        metadata_list = []
        for i in range(self.vector_store.get_vector_count()):
            metadata = self.vector_store.get_metadata(i)
            if metadata:
                metadata_list.append(metadata)
        
        # Calculate statistics
        stats = {
            "vector_count": self.vector_store.get_vector_count(),
            "dimension": self.config.dimension,
            "index_type": self.config.index_type,
            "collection_count": len(self.collections),
            "metadata_count": len(metadata_list),
            "collections": self.get_collection_info()
        }
        
        return stats
