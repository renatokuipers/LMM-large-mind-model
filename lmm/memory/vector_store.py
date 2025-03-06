"""
Vector Store module for the Large Mind Model (LMM).

This module implements vector storage and retrieval for the LMM's memory system,
using FAISS for efficient similarity search operations on embeddings.
"""
import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import faiss

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.mother.llm_client import LLMClient

logger = get_logger("lmm.memory.vector_store")

class VectorStore:
    """
    Vector store for the LMM's memory system.
    
    This class provides methods for storing, retrieving, and searching vector embeddings
    using FAISS, with support for GPU acceleration if available.
    """
    
    def __init__(self, dimension: int = 1024, use_gpu: Optional[bool] = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            use_gpu: Whether to use GPU acceleration (if None, uses config value)
        """
        config = get_config()
        self.dimension = dimension
        self.use_gpu = use_gpu if use_gpu is not None else config.memory.use_gpu
        self.gpu_device = config.memory.gpu_device
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Move index to GPU if requested and available
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, self.index)
                logger.info(f"Using GPU (device {self.gpu_device}) for vector operations")
            except Exception as e:
                logger.warning(f"Failed to use GPU for vector operations: {e}")
                logger.info("Falling back to CPU for vector operations")
                self.use_gpu = False
        
        # Initialize metadata storage
        self.metadata: List[Dict[str, Any]] = []
        
        # Initialize LLM client for embeddings
        self.llm_client = LLMClient()
        
        logger.info(f"Initialized vector store with dimension {self.dimension}")
    
    def add(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> int:
        """
        Add a text item to the vector store.
        
        Args:
            text: Text to add
            metadata: Optional metadata to associate with the text
            embedding: Optional pre-computed embedding
            
        Returns:
            Index of the added item
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.llm_client.get_embedding(text)
        
        # Convert embedding to numpy array
        embedding_np = np.array([embedding], dtype=np.float32)
        
        # Add to FAISS index
        index_id = self.index.ntotal
        self.index.add(embedding_np)
        
        # Store metadata
        item_metadata = metadata or {}
        item_metadata["text"] = text
        item_metadata["timestamp"] = datetime.now().isoformat()
        item_metadata["index_id"] = index_id
        
        self.metadata.append(item_metadata)
        
        logger.debug(f"Added item to vector store with index {index_id}")
        return index_id
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        query_embedding: Optional[List[float]] = None,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for items similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            query_embedding: Optional pre-computed query embedding
            filter_fn: Optional function to filter results
            
        Returns:
            List of dictionaries with search results
        """
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = self.llm_client.get_embedding(query)
        
        # Convert query embedding to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_np, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no result
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                result["similarity"] = 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity
                
                # Apply filter if provided
                if filter_fn is None or filter_fn(result):
                    results.append(result)
        
        logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def get(self, index_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an item from the vector store by index ID.
        
        Args:
            index_id: Index ID of the item
            
        Returns:
            Dictionary with the item data, or None if not found
        """
        if 0 <= index_id < len(self.metadata):
            return self.metadata[index_id]
        return None
    
    def count(self) -> int:
        """
        Get the number of items in the vector store.
        
        Returns:
            Number of items
        """
        return self.index.ntotal
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "index.faiss")
        
        # If using GPU, convert back to CPU for saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)
        
        logger.info(f"Saved vector store to {directory} with {self.count()} items")
    
    def load(self, directory: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            True if successful, False otherwise
        """
        # Check if files exist
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Vector store files not found in {directory}")
            return False
        
        try:
            # Load FAISS index
            cpu_index = faiss.read_index(index_path)
            
            # Move to GPU if requested
            if self.use_gpu:
                try:
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, cpu_index)
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU: {e}")
                    self.index = cpu_index
                    self.use_gpu = False
            else:
                self.index = cpu_index
            
            # Load metadata
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded vector store from {directory} with {self.count()} items")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the vector store."""
        # Reinitialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, self.index)
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        # Clear metadata
        self.metadata = []
        
        logger.info("Cleared vector store") 