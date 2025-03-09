"""
Vector Store Utility

Provides functionality for generating, storing, and retrieving embeddings.
This module serves as a unified interface for embedding operations throughout
the LMM system.
"""

import os
import json
import numpy as np
import faiss
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from lmm_project.core.exceptions import StorageError
from lmm_project.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Global LLM client for embeddings - lazily initialized
_llm_client = None

def get_llm_client():
    """Get or initialize the LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(base_url="http://192.168.2.12:1234")
    return _llm_client

def get_embedding(text: Union[str, List[str]], model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m") -> np.ndarray:
    """
    Get embedding vector for text using the LLM API
    
    Args:
        text: Text or list of texts to generate embeddings for
        model: Embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    try:
        # Get LLM client
        client = get_llm_client()
        
        # Get embedding from API
        embedding = client.get_embedding(text, embedding_model=model)
        
        # Convert to numpy array if not already
        if isinstance(embedding, list):
            if isinstance(embedding[0], list):
                # Multiple embeddings
                return np.array(embedding)
            else:
                # Single embedding
                return np.array(embedding)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        
        # Return zero embedding as fallback
        # Determine dimension based on the model
        if "nomic" in model:
            dim = 768
        else:
            dim = 1536  # Default for OpenAI models
            
        # Return zero vector(s)
        if isinstance(text, list):
            return np.zeros((len(text), dim))
        else:
            return np.zeros(dim)

class VectorStore:
    """
    Vector store for embedding storage and retrieval
    
    This class provides methods for storing and retrieving embeddings
    using FAISS for efficient similarity search.
    """
    def __init__(
        self, 
        dimension: int = 768, 
        index_type: str = "Flat", 
        storage_dir: str = "storage/embeddings",
        use_gpu: bool = False
    ):
        """
        Initialize the vector store
        
        Parameters:
        dimension: Dimension of the embeddings
        index_type: Type of FAISS index to use
        storage_dir: Directory to store index files
        use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            # IVF index requires training, so we start with a base index
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.is_trained = False
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        # Move to GPU if requested and available
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("FAISS index moved to GPU")
            except Exception as e:
                print(f"Failed to use GPU: {e}")
                self.use_gpu = False
                
        # Metadata storage
        self.metadata: List[Dict[str, Any]] = []
        
    def add(self, embeddings: Union[List[List[float]], np.ndarray], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add embeddings to the index
        
        Parameters:
        embeddings: List of embedding vectors
        metadata_list: List of metadata dictionaries
        
        Returns:
        List of assigned IDs
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings and metadata entries must match")
            
        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings_array = np.array(embeddings, dtype=np.float32)
        else:
            embeddings_array = embeddings
            
        # Check if IVF index needs training
        if self.index_type == "IVF" and not self.index.is_trained:
            if len(embeddings_array) < 100:
                # Not enough data to train, use random data
                random_data = np.random.random((100, self.dimension)).astype(np.float32)
                self.index.train(random_data)
            else:
                self.index.train(embeddings_array)
                
        # Get starting ID
        start_id = len(self.metadata)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Add metadata
        for meta in metadata_list:
            # Add timestamp if not present
            if "timestamp" not in meta:
                meta["timestamp"] = datetime.now().isoformat()
            # Add ID
            meta["id"] = start_id + len(self.metadata)
            self.metadata.append(meta)
            
        # Return assigned IDs
        return list(range(start_id, start_id + len(metadata_list)))
    
    def search(self, query_embedding: Union[List[float], np.ndarray], k: int = 5) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Search for similar embeddings
        
        Parameters:
        query_embedding: Query embedding vector
        k: Number of results to return
        
        Returns:
        Tuple of (IDs, distances, metadata)
        """
        # Convert to numpy array if needed
        if isinstance(query_embedding, list):
            query_array = np.array([query_embedding], dtype=np.float32)
        else:
            query_array = query_embedding.reshape(1, -1)
            
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Get metadata
        result_metadata = []
        valid_indices = []
        valid_distances = []
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # FAISS may return -1 for not enough results
                valid_indices.append(int(idx))
                valid_distances.append(float(distances[0][i]))
                result_metadata.append(self.metadata[idx])
                
        return valid_indices, valid_distances, result_metadata
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the index and metadata to disk
        
        Parameters:
        filename: Optional filename to save to
        
        Returns:
        Path to saved files
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"index_{timestamp}"
                
            # Ensure it doesn't have an extension
            filename = Path(filename).stem
            
            # Save index
            index_path = self.storage_dir / f"{filename}.index"
            
            # Convert to CPU if on GPU
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
                
            # Save metadata
            metadata_path = self.storage_dir / f"{filename}.meta"
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
                
            # Save config
            config_path = self.storage_dir / f"{filename}.config"
            config = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "count": len(self.metadata),
                "saved_at": datetime.now().isoformat()
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            return str(index_path)
        except Exception as e:
            raise StorageError(f"Failed to save vector store: {e}")
    
    def load(self, filename: str) -> None:
        """
        Load the index and metadata from disk
        
        Parameters:
        filename: Path to the index file
        """
        try:
            # Handle with or without extension
            filepath = Path(filename)
            base_path = self.storage_dir / filepath.stem
            
            # Load index
            index_path = f"{base_path}.index"
            self.index = faiss.read_index(str(index_path))
            
            # Move to GPU if requested
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    print(f"Failed to use GPU: {e}")
                    self.use_gpu = False
                    
            # Load metadata
            metadata_path = f"{base_path}.meta"
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
                
            # Update dimension from loaded index
            self.dimension = self.index.d
            
            # Determine index type
            if isinstance(self.index, faiss.IndexFlatL2) or isinstance(self.index, faiss.GpuIndexFlatL2):
                self.index_type = "Flat"
            elif isinstance(self.index, faiss.IndexIVFFlat) or isinstance(self.index, faiss.GpuIndexIVFFlat):
                self.index_type = "IVF"
                
        except Exception as e:
            raise StorageError(f"Failed to load vector store: {e}")
    
    def get_item(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific item
        
        Parameters:
        idx: Index of the item
        
        Returns:
        Metadata dictionary or None if not found
        """
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None
    
    def update_metadata(self, idx: int, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific item
        
        Parameters:
        idx: Index of the item
        metadata: New metadata dictionary
        
        Returns:
        True if successful, False otherwise
        """
        if 0 <= idx < len(self.metadata):
            # Preserve ID
            metadata["id"] = self.metadata[idx]["id"]
            self.metadata[idx] = metadata
            return True
        return False
    
    def delete(self, indices: List[int]) -> bool:
        """
        Delete items from the index
        
        Note: FAISS doesn't support true deletion, so this is a soft delete
        that only removes metadata. The vectors remain in the index but
        won't be returned in search results.
        
        Parameters:
        indices: List of indices to delete
        
        Returns:
        True if successful
        """
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                # Mark as deleted in metadata
                self.metadata[idx]["deleted"] = True
                
        return True
    
    def clear(self) -> None:
        """
        Clear the index and metadata
        """
        # Recreate index
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.is_trained = False
            
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Failed to use GPU: {e}")
                self.use_gpu = False
                
        # Clear metadata
        self.metadata = []