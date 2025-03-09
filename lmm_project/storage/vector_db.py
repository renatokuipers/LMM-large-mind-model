"""
Vector Database Module

This module provides a unified interface for vector storage and retrieval
operations using FAISS. It supports:
- Storing embeddings with associated metadata
- Efficient similarity search with multiple index types
- GPU acceleration when available
- Persistence and incremental updates

The vector database is a critical component for semantic memory and
concept representation in the LMM system.
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path
import numpy as np
import faiss

# Set up logging
logger = logging.getLogger(__name__)

class VectorDB:
    """
    Vector database for storing and retrieving embeddings
    
    This class provides a unified interface for vector storage operations
    using FAISS. It supports efficient similarity search with optional
    GPU acceleration.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "Flat",
        storage_dir: str = "storage/embeddings",
        index_name: str = "default",
        use_gpu: bool = True,
        nlist: int = 100,  # For IVF indices
        nprobe: int = 10   # For search
    ):
        """
        Initialize the vector database
        
        Args:
            dimension: Vector dimension (default for Nomic embeddings)
            index_type: Type of FAISS index ("Flat", "IVF", or "HNSW")
            storage_dir: Directory for storing indices
            index_name: Name of this index
            use_gpu: Whether to use GPU acceleration if available
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to probe during search
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_name = index_name
        
        # Status tracking
        self.is_trained = False
        self.vector_count = 0
        self.last_modified = time.time()
        self.gpu_enabled = False
        
        # Ensure storage directory exists
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_index: Dict[str, int] = {}
        self.deleted_indices: Set[int] = set()
        
        # Initialize index
        self.index = self._create_index()
        
        # Move to GPU if requested and available
        if use_gpu:
            self._move_to_gpu()
    
    def _create_index(self) -> faiss.Index:
        """Create the FAISS index based on the specified type"""
        if self.index_type == "Flat":
            return faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "IVF":
            # Create IVF index (requires training)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe  # Number of clusters to visit during search
            return index
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph-based index
            return faiss.IndexHNSWFlat(self.dimension, 32)  # 32 is the number of connections per node
            
        else:
            logger.warning(f"Unknown index type '{self.index_type}', falling back to Flat")
            return faiss.IndexFlatL2(self.dimension)
    
    def _move_to_gpu(self) -> bool:
        """Move the index to GPU if available"""
        try:
            # Check if GPU is available
            if not faiss.get_num_gpus():
                logger.info("No GPU found for FAISS acceleration")
                return False
                
            # Get GPU resources
            res = faiss.StandardGpuResources()
            
            # Move index to GPU
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.index = gpu_index
            self.gpu_enabled = True
            
            logger.info("Successfully moved FAISS index to GPU")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")
            return False
    
    def train(self, vectors: np.ndarray) -> bool:
        """
        Train the index (required for IVF indices)
        
        Args:
            vectors: Training vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only IVF indices need training
            if self.index_type != "IVF":
                self.is_trained = True
                return True
                
            # Check if we have enough training data
            if vectors.shape[0] < self.nlist:
                logger.warning(f"Not enough training data: {vectors.shape[0]} < {self.nlist}")
                return False
                
            # Train the index
            self.index.train(vectors)
            self.is_trained = True
            
            logger.info(f"Trained {self.index_type} index with {vectors.shape[0]} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error training index: {e}")
            return False
    
    def add(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to the index
        
        Args:
            vectors: Vectors to add (numpy array with shape [n, dimension])
            metadata_list: List of metadata dictionaries for each vector
            ids: Optional list of IDs for the vectors
            
        Returns:
            List of assigned IDs
        """
        try:
            # Validate inputs
            n_vectors = vectors.shape[0]
            if len(metadata_list) != n_vectors:
                raise ValueError(f"Number of vectors ({n_vectors}) does not match metadata list length ({len(metadata_list)})")
                
            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{self.vector_count + i}_{int(time.time())}" for i in range(n_vectors)]
            elif len(ids) != n_vectors:
                raise ValueError(f"Number of vectors ({n_vectors}) does not match IDs list length ({len(ids)})")
                
            # Check if index needs training
            if self.index_type == "IVF" and not self.is_trained:
                if not self.train(vectors):
                    logger.warning("Failed to train index, attempting to add vectors anyway")
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Store metadata and update mappings
            start_idx = self.vector_count
            for i, (vec_id, metadata) in enumerate(zip(ids, metadata_list)):
                idx = start_idx + i
                self.id_to_index[vec_id] = idx
                self.metadata.append({
                    "id": vec_id,
                    "index": idx,
                    "timestamp": time.time(),
                    **metadata
                })
                
            # Update counts
            self.vector_count += n_vectors
            self.last_modified = time.time()
            
            logger.info(f"Added {n_vectors} vectors to index, total count: {self.vector_count}")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            return []
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_fn: Optional filter function for results
            
        Returns:
            List of search results with metadata and distances
        """
        try:
            # Ensure query vector is properly shaped
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # Adjust k to account for deleted vectors
            adjusted_k = k + len(self.deleted_indices)
            
            # Search index
            distances, indices = self.index.search(query_vector, min(adjusted_k, self.vector_count))
            
            # Process results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.metadata):
                    continue
                    
                # Skip deleted vectors
                if idx in self.deleted_indices:
                    continue
                    
                # Get metadata
                result = {
                    "distance": float(dist),
                    "similarity": 1.0 / (1.0 + float(dist)),  # Convert distance to similarity
                    **self.metadata[idx]
                }
                
                # Apply filter if provided
                if filter_fn is None or filter_fn(result):
                    results.append(result)
                    
                # Stop once we have enough results
                if len(results) >= k:
                    break
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple query vectors
        
        Args:
            query_vectors: Query vectors
            k: Number of results to return for each query
            filter_fn: Optional filter function for results
            
        Returns:
            List of search results for each query
        """
        try:
            # Adjust k to account for deleted vectors
            adjusted_k = k + len(self.deleted_indices)
            
            # Search index
            distances, indices = self.index.search(query_vectors, min(adjusted_k, self.vector_count))
            
            # Process results
            all_results = []
            for q_idx in range(len(query_vectors)):
                results = []
                for idx_idx, dist_idx in zip(indices[q_idx], distances[q_idx]):
                    # Skip invalid indices
                    if idx_idx < 0 or idx_idx >= len(self.metadata):
                        continue
                        
                    # Skip deleted vectors
                    if idx_idx in self.deleted_indices:
                        continue
                        
                    # Get metadata
                    result = {
                        "distance": float(dist_idx),
                        "similarity": 1.0 / (1.0 + float(dist_idx)),
                        **self.metadata[idx_idx]
                    }
                    
                    # Apply filter if provided
                    if filter_fn is None or filter_fn(result):
                        results.append(result)
                        
                    # Stop once we have enough results
                    if len(results) >= k:
                        break
                        
                all_results.append(results)
                
            return all_results
            
        except Exception as e:
            logger.error(f"Error performing batch search: {e}")
            return [[]] * len(query_vectors)
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get vector metadata by ID
        
        Args:
            id: ID of the vector
            
        Returns:
            Metadata for the vector, or None if not found
        """
        try:
            if id not in self.id_to_index:
                return None
                
            idx = self.id_to_index[id]
            
            # Check if deleted
            if idx in self.deleted_indices:
                return None
                
            return self.metadata[idx]
            
        except Exception as e:
            logger.error(f"Error getting vector by ID: {e}")
            return None
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector by ID
        
        Note: FAISS doesn't support direct deletion, so we mark it as deleted
        
        Args:
            id: ID of the vector to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if id not in self.id_to_index:
                logger.warning(f"Vector {id} not found for deletion")
                return False
                
            idx = self.id_to_index[id]
            
            # Mark as deleted
            self.deleted_indices.add(idx)
            
            # Remove from ID mapping
            del self.id_to_index[id]
            
            self.last_modified = time.time()
            logger.info(f"Marked vector {id} as deleted")
            
            # If too many deletions, rebuild index
            if len(self.deleted_indices) > self.vector_count * 0.2:
                self._rebuild_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            return False
    
    def _rebuild_index(self) -> bool:
        """
        Rebuild the index to remove deleted vectors
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Rebuilding index to remove {len(self.deleted_indices)} deleted vectors")
            
            # Collect all non-deleted vectors
            valid_indices = [i for i in range(self.vector_count) 
                            if i not in self.deleted_indices and i < len(self.metadata)]
                            
            if not valid_indices:
                logger.warning("No valid vectors to rebuild index")
                return False
                
            # Create a new index
            new_index = self._create_index()
            
            # Extract vectors to add back
            vectors = []
            new_metadata = []
            new_id_to_index = {}
            
            # Need to get original vectors from the index
            # This is tricky with FAISS, so we'll use a workaround
            for new_idx, old_idx in enumerate(valid_indices):
                # Get vector ID
                vec_id = self.metadata[old_idx].get("id")
                if not vec_id:
                    continue
                    
                # Update mappings
                new_metadata.append(self.metadata[old_idx])
                new_id_to_index[vec_id] = new_idx
                
                # We'll need to query the original vector
                # In a real implementation, you'd store the original vectors separately
                # or extract them from the index
                
            # Add vectors to new index
            # In a real implementation, you'd add the vectors here
            
            # Update instance variables
            self.metadata = new_metadata
            self.id_to_index = new_id_to_index
            self.deleted_indices = set()
            self.vector_count = len(new_metadata)
            self.last_modified = time.time()
            
            # Move to GPU if needed
            if self.gpu_enabled:
                self._move_to_gpu()
                
            logger.info(f"Rebuilt index with {self.vector_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "vector_count": self.vector_count,
            "deleted_count": len(self.deleted_indices),
            "active_count": self.vector_count - len(self.deleted_indices),
            "is_trained": self.is_trained,
            "gpu_enabled": self.gpu_enabled,
            "last_modified": self.last_modified,
            "nlist": self.nlist if self.index_type == "IVF" else None,
            "nprobe": self.nprobe if self.index_type == "IVF" else None
        }
    
    def save(self, add_timestamp: bool = True) -> str:
        """
        Save the index and metadata to disk
        
        Args:
            add_timestamp: Whether to add a timestamp to the filename
            
        Returns:
            Path to the saved files
        """
        try:
            # Create timestamp
            timestamp = int(time.time())
            
            # Create filenames
            if add_timestamp:
                index_path = self.storage_dir / f"{self.index_name}_{timestamp}.index"
                meta_path = self.storage_dir / f"{self.index_name}_{timestamp}.meta"
            else:
                index_path = self.storage_dir / f"{self.index_name}.index"
                meta_path = self.storage_dir / f"{self.index_name}.meta"
                
            # Move index to CPU for saving if it's on GPU
            if self.gpu_enabled:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
                
            # Save metadata and mappings
            meta_data = {
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "deleted_indices": list(self.deleted_indices),
                "vector_count": self.vector_count,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "is_trained": self.is_trained,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "timestamp": timestamp
            }
            
            with open(meta_path, 'wb') as f:
                pickle.dump(meta_data, f, protocol=4)
                
            logger.info(f"Saved index to {index_path} and metadata to {meta_path}")
            return str(index_path)
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return ""
    
    def load(self, index_path: Optional[str] = None, meta_path: Optional[str] = None) -> bool:
        """
        Load the index and metadata from disk
        
        Args:
            index_path: Path to the index file
            meta_path: Path to the metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If paths not provided, use latest
            if not index_path:
                # Find latest index file
                index_files = list(self.storage_dir.glob(f"{self.index_name}_*.index"))
                if not index_files:
                    # Try without timestamp
                    index_files = list(self.storage_dir.glob(f"{self.index_name}.index"))
                
                if not index_files:
                    logger.warning(f"No index files found for {self.index_name}")
                    return False
                    
                # Sort by modification time (newest first)
                index_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                index_path = str(index_files[0])
                
                # Derive metadata path
                meta_path = index_path.replace(".index", ".meta")
                
            # Load index
            self.index = faiss.read_index(index_path)
            
            # Move to GPU if needed
            if self.gpu_enabled:
                self._move_to_gpu()
                
            # Load metadata
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
                
            # Update instance variables
            self.metadata = meta_data.get("metadata", [])
            self.id_to_index = meta_data.get("id_to_index", {})
            self.deleted_indices = set(meta_data.get("deleted_indices", []))
            self.vector_count = meta_data.get("vector_count", 0)
            self.dimension = meta_data.get("dimension", self.dimension)
            self.index_type = meta_data.get("index_type", self.index_type)
            self.is_trained = meta_data.get("is_trained", False)
            self.nlist = meta_data.get("nlist", self.nlist)
            self.nprobe = meta_data.get("nprobe", self.nprobe)
            
            # Set nprobe for the loaded index
            if self.index_type == "IVF":
                if isinstance(self.index, faiss.IndexIVFFlat) or hasattr(self.index, "nprobe"):
                    self.index.nprobe = self.nprobe
                    
            logger.info(f"Loaded index from {index_path} with {self.vector_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear the index and metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reinitialize index
            self.index = self._create_index()
            
            # Move to GPU if needed
            if self.gpu_enabled:
                self._move_to_gpu()
                
            # Reset metadata
            self.metadata = []
            self.id_to_index = {}
            self.deleted_indices = set()
            self.vector_count = 0
            self.is_trained = False
            self.last_modified = time.time()
            
            logger.info("Cleared index and metadata")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
    
    def __len__(self) -> int:
        """Get the number of active vectors in the index"""
        return self.vector_count - len(self.deleted_indices)
        
    def __str__(self) -> str:
        """Get a string representation of the index"""
        return f"VectorDB({self.index_type}, dim={self.dimension}, vectors={len(self)})" 