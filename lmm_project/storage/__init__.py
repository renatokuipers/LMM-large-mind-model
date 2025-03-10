"""
Storage module for the LMM project.

This module provides persistence and storage mechanisms for the cognitive
architecture, including vector storage, state persistence, and experience logging.
"""

from lmm_project.storage.vector_db import (
    VectorDB,
    VectorDBConfig
)

from lmm_project.storage.state_persistence import (
    StatePersistence
)

from lmm_project.storage.experience_logger import (
    ExperienceLogger,
    ExperienceMetadata
)

# Storage manager singleton
_storage_manager_instance = None


class StorageManager:
    """
    Central manager for all storage systems in the LMM.
    
    Provides access to vector database, state persistence, and experience
    logging through a unified interface.
    """
    
    def __init__(
        self,
        storage_dir: str = "storage",
        vector_dimension: int = 768,
        vector_index_type: str = "Flat",
        use_gpu: bool = True
    ):
        """
        Initialize the storage manager.
        
        Parameters:
        storage_dir: Base directory for all storage
        vector_dimension: Dimension for vector embeddings
        vector_index_type: Type of FAISS index to use
        use_gpu: Whether to use GPU acceleration
        """
        self.storage_dir = storage_dir
        self.vector_dimension = vector_dimension
        self.vector_index_type = vector_index_type
        self.use_gpu = use_gpu
        
        # Initialize storage components
        self.vector_db = VectorDB({
            "dimension": vector_dimension,
            "index_type": vector_index_type,
            "use_gpu": use_gpu,
            "storage_dir": f"{storage_dir}/vectors"
        })
        
        self.state_persistence = StatePersistence(
            storage_dir=f"{storage_dir}/states"
        )
        
        self.experience_logger = ExperienceLogger(
            vector_db=self.vector_db,
            storage_dir=f"{storage_dir}/experiences",
            embedding_dimension=vector_dimension
        )
    
    def save_all(self) -> None:
        """Save all storage components to disk."""
        self.vector_db.save()
        self.experience_logger.save()
    
    def close(self) -> None:
        """Close storage components and free resources."""
        # Currently no specific close operations needed
        pass


def create_storage_manager(
    storage_dir: str = "storage",
    vector_dimension: int = 768,
    vector_index_type: str = "Flat",
    use_gpu: bool = True
) -> StorageManager:
    """
    Create the storage manager singleton.
    
    Parameters:
    storage_dir: Base directory for all storage
    vector_dimension: Dimension for vector embeddings
    vector_index_type: Type of FAISS index to use
    use_gpu: Whether to use GPU acceleration
    
    Returns:
    StorageManager instance
    """
    global _storage_manager_instance
    
    if _storage_manager_instance is None:
        _storage_manager_instance = StorageManager(
            storage_dir=storage_dir,
            vector_dimension=vector_dimension,
            vector_index_type=vector_index_type,
            use_gpu=use_gpu
        )
    
    return _storage_manager_instance


def get_storage_manager() -> StorageManager:
    """
    Get the storage manager singleton.
    
    Returns:
    StorageManager instance
    """
    global _storage_manager_instance
    
    if _storage_manager_instance is None:
        _storage_manager_instance = create_storage_manager()
    
    return _storage_manager_instance


__all__ = [
    # Vector database
    'VectorDB', 'VectorDBConfig',
    
    # State persistence
    'StatePersistence',
    
    # Experience logger
    'ExperienceLogger', 'ExperienceMetadata',
    
    # Central manager
    'StorageManager', 'create_storage_manager', 'get_storage_manager'
] 
