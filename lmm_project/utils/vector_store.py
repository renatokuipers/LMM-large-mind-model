import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from lmm_project.core.exceptions import StorageError
from lmm_project.utils.config_manager import get_config
from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.llm_client import LLMClient

# Initialize logger
logger = get_module_logger("vector_store")

# Get configuration
config = get_config()


class VectorStore:
    """
    Vector storage and retrieval system using FAISS.
    Supports GPU acceleration when available.
    """

    def __init__(
        self,
        dimension: int = None,
        index_type: str = "Flat",
        use_gpu: bool = None,
        storage_dir: str = None
    ):
        """
        Initialize vector store.
        
        Parameters:
        dimension: Dimension of vectors (default from config)
        index_type: FAISS index type ("Flat", "IVF", "HNSW")
        use_gpu: Whether to use GPU acceleration (default based on config)
        storage_dir: Directory to store indices
        """
        # Set dimension from config if not provided
        self.dimension = dimension or config.get_int("storage.vector_dimension", 768)
        
        # Set index type from config if not provided
        self.index_type = index_type or config.get_string("storage.vector_index_type", "Flat")
        
        # Use GPU based on config if not explicitly set
        if use_gpu is None:
            use_gpu = config.get_boolean("system.cuda_enabled", True)
        self.use_gpu = use_gpu and self._is_gpu_available()
        
        # Set storage directory
        self.storage_dir = storage_dir or config.get_string("storage_dir", "storage")
        os.makedirs(Path(self.storage_dir) / "vectors", exist_ok=True)
        
        # Initialize index and metadata
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        # Create the index
        self._create_index()
        
        logger.info(f"Vector store initialized (dimension: {self.dimension}, "
                    f"index_type: {self.index_type}, use_gpu: {self.use_gpu})")

    def _is_gpu_available(self) -> bool:
        """Check if FAISS GPU is available."""
        try:
            gpu_count = faiss.get_num_gpus()
            return gpu_count > 0
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}")
            return False

    def _create_index(self) -> None:
        """Create FAISS index based on configuration."""
        try:
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVF":
                # IVF index requires training, so we'll use a more basic one initially
                # and then train it when we have enough data
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                self.index.nprobe = 10  # Number of cells to visit during search
            elif self.index_type == "HNSW":
                # Hierarchical Navigable Small World graph
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 is the number of links per node
            else:
                logger.warning(f"Unknown index type: {self.index_type}, using Flat index")
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Move to GPU if available and requested
            if self.use_gpu:
                try:
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                    logger.info("Using GPU acceleration for vector store")
                except Exception as e:
                    logger.warning(f"Failed to use GPU acceleration: {str(e)}")
                    self.use_gpu = False
        
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            # Fall back to basic index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_type = "Flat"

    def add_vectors(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Parameters:
        vectors: Vectors to add (must be numpy array or list of lists)
        metadata_list: Optional metadata for each vector
        
        Returns:
        List of IDs for the added vectors
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            
            # Ensure vectors are float32
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            # Check dimensions
            if vectors.shape[1] != self.dimension:
                raise StorageError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
            
            # Training for IVF index if needed
            if self.index_type == "IVF" and not self.index.is_trained and vectors.shape[0] >= 100:
                logger.info("Training IVF index")
                if self.use_gpu:
                    # Need to move back to CPU for training
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    cpu_index.train(vectors)
                    # Move back to GPU
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
                else:
                    self.index.train(vectors)
            
            # Assign IDs
            start_id = self.next_id
            ids = np.arange(start_id, start_id + vectors.shape[0], dtype=np.int64)
            self.next_id = start_id + vectors.shape[0]
            
            # Add vectors
            self.index.add_with_ids(vectors, ids)
            
            # Add metadata if provided
            if metadata_list is not None:
                if len(metadata_list) != vectors.shape[0]:
                    raise StorageError(f"Metadata list length ({len(metadata_list)}) must match vector count ({vectors.shape[0]})")
                for i, idx in enumerate(ids):
                    self.metadata[int(idx)] = metadata_list[i]
            
            return ids.tolist()
            
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}")
            raise StorageError(f"Failed to add vectors: {str(e)}") from e

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Parameters:
        query_vector: Query vector
        k: Number of results to return
        metadata_filter: Optional filter for metadata
        
        Returns:
        Tuple of (ids, distances, metadata)
        """
        try:
            # Convert query to numpy array if needed
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array([query_vector], dtype=np.float32)
            else:
                # Ensure it's 2D
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                query_vector = query_vector.astype(np.float32)
            
            # Check dimension
            if query_vector.shape[1] != self.dimension:
                raise StorageError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vector.shape[1]}")
            
            # Search index
            distances, indices = self.index.search(query_vector, k)
            
            # Convert to lists
            ids = indices[0].tolist()
            dists = distances[0].tolist()
            
            # Get metadata and apply filter if needed
            meta_list = []
            filtered_ids = []
            filtered_dists = []
            
            for i, idx in enumerate(ids):
                # Skip if ID is -1 (can happen with some index types)
                if idx == -1:
                    continue
                
                # Get metadata
                meta = self.metadata.get(int(idx), {})
                
                # Apply filter if provided
                if metadata_filter is not None:
                    match = True
                    for key, value in metadata_filter.items():
                        if key not in meta or meta[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Keep this result
                filtered_ids.append(idx)
                filtered_dists.append(dists[i])
                meta_list.append(meta)
                
                # Stop if we have enough results
                if len(filtered_ids) >= k:
                    break
            
            return filtered_ids, filtered_dists, meta_list
            
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise StorageError(f"Failed to search vectors: {str(e)}") from e

    def delete(self, ids: List[int]) -> None:
        """
        Delete vectors by ID.
        Note: FAISS does not support direct deletion with all index types.
        For unsupported indices, this marks vectors as deleted in metadata.
        
        Parameters:
        ids: List of vector IDs to delete
        """
        try:
            # Check if direct removal is supported
            if hasattr(self.index, "remove_ids"):
                # Convert to numpy array
                id_array = np.array(ids, dtype=np.int64)
                self.index.remove_ids(id_array)
                
                # Remove from metadata
                for idx in ids:
                    if idx in self.metadata:
                        del self.metadata[idx]
            else:
                # Mark as deleted in metadata
                logger.warning("Direct deletion not supported for this index type, marking as deleted in metadata")
                for idx in ids:
                    if idx in self.metadata:
                        self.metadata[idx]["_deleted"] = True
        
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise StorageError(f"Failed to delete vectors: {str(e)}") from e

    def update_metadata(self, id: int, metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a vector.
        
        Parameters:
        id: Vector ID
        metadata: New metadata (will be merged with existing)
        """
        if id not in self.metadata:
            self.metadata[id] = {}
        
        # Update metadata
        self.metadata[id].update(metadata)

    def get_metadata(self, id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector.
        
        Parameters:
        id: Vector ID
        
        Returns:
        Metadata or None if not found
        """
        return self.metadata.get(id)

    def save(self, name: str) -> str:
        """
        Save the index and metadata to disk.
        
        Parameters:
        name: Name for the saved index
        
        Returns:
        Path to the saved index
        """
        try:
            # Create directory
            index_dir = Path(self.storage_dir) / "vectors"
            os.makedirs(index_dir, exist_ok=True)
            
            # Base path for files
            base_path = index_dir / name
            
            # Save metadata
            metadata_path = f"{base_path}.metadata"
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "metadata": self.metadata,
                    "next_id": self.next_id,
                    "dimension": self.dimension,
                    "index_type": self.index_type
                }, f)
            
            # Save index
            index_path = f"{base_path}.index"
            
            # If using GPU, move to CPU for saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            
            logger.info(f"Vector store saved to {base_path}")
            return str(base_path)
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise StorageError(f"Failed to save vector store: {str(e)}") from e

    @classmethod
    def load(cls, name: str, storage_dir: Optional[str] = None, use_gpu: Optional[bool] = None) -> "VectorStore":
        """
        Load an index and metadata from disk.
        
        Parameters:
        name: Name of the saved index
        storage_dir: Storage directory (default from config)
        use_gpu: Whether to use GPU (default based on config)
        
        Returns:
        VectorStore instance
        """
        try:
            # Get storage directory from config if not provided
            if storage_dir is None:
                storage_dir = config.get_string("storage_dir", "storage")
            
            index_dir = Path(storage_dir) / "vectors"
            
            # Base path for files
            base_path = index_dir / name
            
            # Load metadata
            metadata_path = f"{base_path}.metadata"
            if not os.path.exists(metadata_path):
                raise StorageError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            
            # Load index
            index_path = f"{base_path}.index"
            if not os.path.exists(index_path):
                raise StorageError(f"Index file not found: {index_path}")
            
            # Create instance
            instance = cls(
                dimension=data["dimension"],
                index_type=data["index_type"],
                use_gpu=use_gpu,
                storage_dir=storage_dir
            )
            
            # Load saved index
            index = faiss.read_index(index_path)
            
            # Move to GPU if requested
            if instance.use_gpu:
                try:
                    gpu_resources = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU: {str(e)}")
                    instance.use_gpu = False
            
            # Set index and metadata
            instance.index = index
            instance.metadata = data["metadata"]
            instance.next_id = data["next_id"]
            
            logger.info(f"Vector store loaded from {base_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise StorageError(f"Failed to load vector store: {str(e)}") from e


# Get embeddings using the LLM client
def get_embeddings(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Get embeddings for texts using the LLM client.
    
    Parameters:
    texts: Text or list of texts to embed
    
    Returns:
    Embedding vector or list of embedding vectors
    """
    client = LLMClient()
    embedding_model = config.get_string("DEFAULT_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5@q4_k_m")
    
    return client.get_embedding(texts, embedding_model)


# Create vector store singleton
_vector_store_instance = None


def get_vector_store(
    dimension: Optional[int] = None,
    index_type: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    storage_dir: Optional[str] = None
) -> VectorStore:
    """
    Get or create a singleton vector store instance.
    
    Parameters:
    dimension: Vector dimension
    index_type: Index type
    use_gpu: Whether to use GPU
    storage_dir: Storage directory
    
    Returns:
    VectorStore instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            dimension=dimension,
            index_type=index_type,
            use_gpu=use_gpu,
            storage_dir=storage_dir
        )
    
    return _vector_store_instance
