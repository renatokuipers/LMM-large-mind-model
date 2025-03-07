import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

@pytest.fixture(scope="session")
def temp_dir():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def sample_text_data():
    """Fixture providing sample text data for testing."""
    return [
        "This is the first sample text for testing purposes.",
        "Here is another sample with different words and structure.",
        "The third sample contains more technical terms like AI and machine learning.",
        "Sample number four discusses emotional aspects like happiness and sadness.",
        "The final sample is about social interactions between people."
    ]

@pytest.fixture
def sample_vector_data():
    """Fixture providing sample vector data for embedding tests."""
    # Generate 5 random vectors of dimension 10
    vectors = np.random.randn(5, 10)
    metadata = [
        {"id": f"vec{i}", "content": f"Vector content {i}", "tags": ["test", f"tag{i}"]}
        for i in range(5)
    ]
    return list(zip(vectors, metadata))

@pytest.fixture
def mock_faiss_index():
    """Fixture providing a mock FAISS index."""
    mock_index = MagicMock()
    mock_index.ntotal = 0
    mock_index.d = 10
    
    def mock_add(vectors):
        mock_index.ntotal += len(vectors) // mock_index.d
    
    def mock_search(query, k):
        # Return mock search results - indices and distances
        indices = np.array([[0, 1, 2]])
        distances = np.array([[0.1, 0.2, 0.3]])
        return distances, indices
    
    mock_index.add = mock_add
    mock_index.search = mock_search
    return mock_index

@pytest.fixture
def sample_config_data():
    """Fixture providing sample configuration data."""
    return {
        "model": {
            "name": "test-model",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 100
            }
        },
        "memory": {
            "persistence_dir": "memory_data",
            "max_entries": 1000
        },
        "logging": {
            "level": "INFO",
            "file": "app.log"
        }
    }

@pytest.fixture
def sample_time_series_data():
    """Fixture providing sample time series data for visualization tests."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=20),
        "value": np.random.randn(20).cumsum(),  # Random walk
        "category": ["A", "B"] * 10
    })

@pytest.fixture
def sample_memory_data():
    """Fixture providing sample memory data for testing."""
    return [
        {
            "id": f"mem{i}",
            "content": f"Memory content {i}",
            "timestamp": f"2023-01-{i+1:02d}T12:00:00",
            "importance": i/10,
            "tags": ["test", f"tag{i}"]
        }
        for i in range(10)
    ]

@pytest.fixture
def mock_llm_client():
    """Fixture providing a mock LLM client."""
    mock_client = MagicMock()
    
    def generate_text(prompt, **kwargs):
        # Return different responses based on the prompt
        if "hello" in prompt.lower():
            return "Hello! How can I assist you today?"
        elif "meaning" in prompt.lower():
            return "The meaning is whatever you make of it."
        else:
            return "I'm not sure how to respond to that."
    
    mock_client.generate = generate_text
    return mock_client

@pytest.fixture
def windows_paths():
    """Fixture providing sample Windows-style paths for testing path handling."""
    return {
        "absolute": "C:\\Users\\test\\Projects\\lmm",
        "relative": "data\\memories",
        "mixed_slashes": "data/memories\\files",
        "with_spaces": "C:\\Program Files\\LMM Project\\data"
    } 