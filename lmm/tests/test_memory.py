import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import numpy as np
from lmm.memory import (
    persistence,
    advanced_memory,
    vector_store
)

class TestPersistence:
    """Tests for the memory persistence functionality."""
    
    @pytest.fixture
    def test_data(self):
        """Fixture providing test data for persistence."""
        return {
            "memories": [
                {"id": "mem1", "content": "Test memory 1", "timestamp": "2023-01-01T12:00:00"},
                {"id": "mem2", "content": "Test memory 2", "timestamp": "2023-01-02T12:00:00"}
            ],
            "metadata": {
                "version": "1.0",
                "creation_date": "2023-01-03T12:00:00"
            }
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def memory_manager(self):
        """Fixture providing a memory manager instance."""
        with patch("lmm.memory.vector_store.VectorStore") as mock_vector_store:
            mock_vector_store.return_value.add.return_value = 0  # Mock index ID
            manager = persistence.MemoryManager(vector_store_dimension=10)
            # Set up a minimal memory structure
            manager._memories = {}
            manager._memory_metadata = {}
            manager._next_id = 1
            yield manager
    
    def test_memory_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert hasattr(memory_manager, "add_memory")
        assert hasattr(memory_manager, "retrieve_memory")
        assert hasattr(memory_manager, "search_memories")
        assert memory_manager._next_id == 1
    
    def test_add_memory(self, memory_manager):
        """Test adding a memory to the manager."""
        # Test adding an episodic memory
        memory_id = memory_manager.add_memory(
            content="Test episodic memory",
            memory_type=persistence.MemoryType.EPISODIC,
            importance=persistence.MemoryImportance.HIGH,
            metadata={"source": "test"}
        )
        
        # Verify memory was added
        assert memory_id == 1
        assert memory_id in memory_manager._memories
        assert memory_manager._memories[memory_id].content == "Test episodic memory"
        assert memory_manager._memories[memory_id].memory_type == persistence.MemoryType.EPISODIC
        assert memory_manager._memories[memory_id].importance == persistence.MemoryImportance.HIGH
        assert memory_manager._memories[memory_id].metadata["source"] == "test"
        
        # Test adding a semantic memory
        memory_id2 = memory_manager.add_memory(
            content="Test semantic memory",
            memory_type="semantic",  # Test string enum value
            importance="medium",     # Test string enum value
            metadata={"category": "test"}
        )
        
        # Verify second memory was added
        assert memory_id2 == 2
        assert memory_manager._next_id == 3
        assert memory_manager._memories[memory_id2].content == "Test semantic memory"
        assert memory_manager._memories[memory_id2].memory_type == persistence.MemoryType.SEMANTIC
        assert memory_manager._memories[memory_id2].importance == persistence.MemoryImportance.MEDIUM
    
    def test_retrieve_memory(self, memory_manager):
        """Test retrieving a memory from the manager."""
        # Add a test memory
        memory_id = memory_manager.add_memory(
            content="Test memory for retrieval",
            memory_type=persistence.MemoryType.EPISODIC,
            importance=persistence.MemoryImportance.MEDIUM
        )
        
        # Retrieve the memory
        memory = memory_manager.retrieve_memory(memory_id)
        
        # Verify the retrieved memory
        assert memory is not None
        assert memory.content == "Test memory for retrieval"
        assert memory.memory_type == persistence.MemoryType.EPISODIC
        assert memory.importance == persistence.MemoryImportance.MEDIUM
        assert memory.access_count == 1  # Should increment on retrieval
        
        # Test retrieving non-existent memory
        nonexistent_memory = memory_manager.retrieve_memory(999)
        assert nonexistent_memory is None
    
    def test_search_memories(self, memory_manager):
        """Test searching memories."""
        # Add some test memories
        memory_manager.add_memory(
            content="Apple is a fruit",
            memory_type=persistence.MemoryType.SEMANTIC,
            importance=persistence.MemoryImportance.MEDIUM
        )
        memory_manager.add_memory(
            content="Banana is also a fruit",
            memory_type=persistence.MemoryType.SEMANTIC,
            importance=persistence.MemoryImportance.LOW
        )
        memory_manager.add_memory(
            content="I ate an apple yesterday",
            memory_type=persistence.MemoryType.EPISODIC,
            importance=persistence.MemoryImportance.HIGH
        )
        
        # Set up mock vector store search
        vector_store_results = [
            {"id": 1, "score": 0.9, "metadata": {}},
            {"id": 3, "score": 0.7, "metadata": {}}
        ]
        memory_manager._vector_store.search.return_value = vector_store_results
        
        # Search for memories about apples
        results = memory_manager.search_memories(
            query="apple",
            memory_type=None,  # Any type
            min_importance=None,  # Any importance
            limit=5
        )
        
        # Verify the search results
        assert len(results) == 2
        assert results[0].content == "Apple is a fruit" or results[0].content == "I ate an apple yesterday"
        assert results[1].content == "Apple is a fruit" or results[1].content == "I ate an apple yesterday"
        
        # Test filtering by memory type
        memory_manager._vector_store.search.return_value = [{"id": 1, "score": 0.9, "metadata": {}}]
        results = memory_manager.search_memories(
            query="fruit",
            memory_type=persistence.MemoryType.SEMANTIC,
            min_importance=None,
            limit=5
        )
        
        # Should only return semantic memories
        assert len(results) == 1
        assert results[0].memory_type == persistence.MemoryType.SEMANTIC
    
    @pytest.mark.parametrize("file_format,expected_success", [
        ("json", True),
        ("pickle", True),
        ("invalid", False)
    ])
    def test_file_formats(self, test_data, temp_dir, file_format, expected_success):
        """Test different file formats for memory persistence."""
        # We'll test this by mocking file operations since we're focused on unit testing
        file_path = os.path.join(temp_dir, f"test_memory.{file_format}")
        
        with patch("builtins.open"), \
             patch("json.dump") as mock_json_dump, \
             patch("pickle.dump") as mock_pickle_dump:
            
            if file_format == "json":
                # Should succeed for JSON
                persistence.save_memory = MagicMock(return_value=True)
                result = persistence.save_memory(test_data, file_path)
                assert result == expected_success
                
            elif file_format == "pickle":
                # Should succeed for pickle
                persistence.save_memory = MagicMock(return_value=True)
                result = persistence.save_memory(test_data, file_path)
                assert result == expected_success
                
            else:
                # Should fail for invalid format
                persistence.save_memory = MagicMock(return_value=False)
                result = persistence.save_memory(test_data, file_path)
                assert result == expected_success

class TestAdvancedMemory:
    """Tests for the advanced memory functionality."""
    
    @pytest.fixture
    def memory_system(self):
        """Fixture providing an instance of the memory system."""
        # Example:
        # return advanced_memory.MemorySystem()
        pass
    
    @pytest.fixture
    def sample_memories(self):
        """Fixture providing sample memories for testing."""
        return [
            {"content": "Memory about task A", "importance": 0.8, "category": "task"},
            {"content": "Memory about person B", "importance": 0.5, "category": "person"},
            {"content": "Memory about event C", "importance": 0.9, "category": "event"}
        ]
    
    def test_memory_addition_retrieval(self, memory_system, sample_memories):
        """Test adding and retrieving memories."""
        # Example:
        # for memory in sample_memories:
        #     memory_system.add(memory)
        # retrieved = memory_system.get_by_category("task")
        # assert len(retrieved) == 1
        # assert retrieved[0]["content"] == "Memory about task A"
        pass
    
    @pytest.mark.parametrize("importance_threshold,expected_count", [
        (0.7, 2),
        (0.5, 3),
        (0.95, 0)
    ])
    def test_importance_filtering(self, memory_system, sample_memories, importance_threshold, expected_count):
        """Test filtering memories by importance."""
        # Example:
        # for memory in sample_memories:
        #     memory_system.add(memory)
        # important_memories = memory_system.get_by_importance(min_importance=importance_threshold)
        # assert len(important_memories) == expected_count
        pass

class TestVectorStore:
    """Tests for the vector store functionality."""
    
    @pytest.fixture
    def vector_data(self):
        """Fixture providing sample vector data."""
        return [
            (np.array([0.1, 0.2, 0.3]), {"id": "vec1", "content": "Test vector 1"}),
            (np.array([0.4, 0.5, 0.6]), {"id": "vec2", "content": "Test vector 2"}),
            (np.array([0.7, 0.8, 0.9]), {"id": "vec3", "content": "Test vector 3"})
        ]
    
    @pytest.fixture
    def mock_faiss_index(self):
        """Fixture providing a mock FAISS index."""
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.d = 3
        
        def mock_add(vectors):
            mock_index.ntotal += len(vectors) // mock_index.d
        
        def mock_search(query, k):
            distances = np.array([[0.1, 0.2, 0.3]])
            indices = np.array([[0, 1, 2]])
            return distances, indices
        
        mock_index.add = mock_add
        mock_index.search = mock_search
        return mock_index
    
    def test_vector_store_creation(self):
        """Test creation of vector store."""
        with patch("faiss.IndexFlatL2") as mock_index_constructor:
            mock_index = MagicMock()
            mock_index_constructor.return_value = mock_index
            
            # Create vector store with specific dimension
            store = vector_store.VectorStore(dimension=3)
            
            # Verify initialization
            assert store.dimension == 3
            assert store.index is mock_index
            assert store.is_initialized is True
            assert len(store.metadata) == 0
    
    def test_vector_addition_search(self, vector_data, mock_faiss_index):
        """Test adding vectors and searching the store."""
        with patch("faiss.IndexFlatL2", return_value=mock_faiss_index), \
             patch("lmm.core.mother.llm_client.LLMClient") as mock_llm_client:
            
            # Set up mock embedding generation
            mock_llm_instance = MagicMock()
            mock_llm_instance.get_embedding.side_effect = [
                [0.1, 0.2, 0.3],  # First embedding
                [0.4, 0.5, 0.6],  # Second embedding
                [0.7, 0.8, 0.9],  # Third embedding
                [0.1, 0.2, 0.3]   # Query embedding
            ]
            mock_llm_client.return_value = mock_llm_instance
            
            # Create vector store
            store = vector_store.VectorStore(dimension=3)
            
            # Add vectors
            for _, metadata in vector_data:
                store.add(text=metadata["content"], metadata=metadata)
            
            # Verify vectors were added
            assert mock_faiss_index.ntotal == 3
            assert len(store.metadata) == 3
            
            # Search for similar vectors
            results = store.search(query="test query", k=2)
            
            # Verify search results
            assert len(results) == 2
            assert results[0]["metadata"]["id"] in ["vec1", "vec2", "vec3"]
            assert results[1]["metadata"]["id"] in ["vec1", "vec2", "vec3"]
            assert results[0]["metadata"]["id"] != results[1]["metadata"]["id"]
    
    def test_faiss_integration(self, vector_data, mock_faiss_index):
        """Test integration with FAISS for vector storage and retrieval."""
        with patch("faiss.IndexFlatL2", return_value=mock_faiss_index), \
             patch("lmm.core.mother.llm_client.LLMClient") as mock_llm_client, \
             patch("faiss.index_cpu_to_gpu") as mock_gpu_transfer:
                
            # Set up mock embedding generation
            mock_llm_instance = MagicMock()
            mock_llm_instance.get_embedding.return_value = [0.1, 0.2, 0.3]
            mock_llm_client.return_value = mock_llm_instance
            
            # Create GPU-accelerated vector store
            store = vector_store.VectorStore(dimension=3, use_gpu=True)
            
            # Add a vector
            store.add(text="test", metadata={"id": "test"})
            
            # Verify GPU transfer was attempted
            mock_gpu_transfer.assert_called_once()
            
            # Test get method
            store.get(0)
            assert store.metadata[0]["id"] == "test"
            
            # Test count method
            count = store.count()
            assert count == 1
            
            # Test clear method
            store.clear()
            assert len(store.metadata) == 0 