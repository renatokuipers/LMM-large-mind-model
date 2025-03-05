import unittest
import os
import sys
import requests
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_child.cognition.cognitive_component import CognitiveComponent
from neural_child.memory.memory_component import MemoryComponent


class MockResponse:
    def __init__(self, json_data=None, status_code=200, raise_error=False):
        self.json_data = json_data
        self.status_code = status_code
        self.raise_error = raise_error
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.raise_error:
            raise requests.exceptions.HTTPError("API Error")


class TestEmbeddingAPI(unittest.TestCase):
    
    def setUp(self):
        # Use standard test configuration
        self.cognitive = CognitiveComponent(device="cpu")
        self.memory = MemoryComponent(device="cpu")
        
        # Create mock embedding data
        self.mock_embedding = np.random.randn(384).astype(np.float32).tolist()
        self.mock_response = {
            "data": [
                {
                    "embedding": self.mock_embedding,
                    "index": 0,
                    "object": "embedding"
                }
            ],
            "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
    
    @patch('requests.post')
    def test_cognitive_embedding_api_success(self, mock_post):
        """Test successful API call for cognitive component embeddings."""
        # Set up mock response
        mock_post.return_value = MockResponse(json_data=self.mock_response)
        
        # Test cognitive component's embedding creation
        input_data = {
            "mother_utterance": "Hello little one",
            "emotional_state": {"joy": 0.7},
            "context": {"location": "home"}
        }
        
        # Access private method for testing
        embedding = self.cognitive._create_input_embedding(
            input_data["mother_utterance"],
            input_data["emotional_state"],
            input_data["context"]
        )
        
        # Verify embedding shape
        self.assertEqual(embedding.shape[0], 1)  # Batch dimension
        self.assertEqual(embedding.shape[1], self.cognitive.input_size)
        
        # Verify API was called once
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_cognitive_embedding_api_failure(self, mock_post):
        """Test API failure handling for cognitive component embeddings."""
        # Set up mock response to fail
        mock_post.return_value = MockResponse(raise_error=True)
        
        # Test cognitive component's embedding creation with failing API
        input_data = {
            "mother_utterance": "Hello little one",
            "emotional_state": {"joy": 0.7},
            "context": {"location": "home"}
        }
        
        # Should not raise exception due to fallback behavior
        embedding = self.cognitive._create_input_embedding(
            input_data["mother_utterance"],
            input_data["emotional_state"],
            input_data["context"]
        )
        
        # Verify we got something back despite API failure
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape[0], 1)  # Batch dimension
        self.assertEqual(embedding.shape[1], self.cognitive.input_size)
    
    @patch('requests.post')
    def test_memory_embedding_api(self, mock_post):
        """Test memory component's embedding API usage."""
        # Set up mock response
        mock_post.return_value = MockResponse(json_data=self.mock_response)
        
        # Test memory embedding creation
        memory_item = {
            "type": "episodic",
            "description": "Mother said hello",
            "context": {"location": "home"}
        }
        
        # Access private method for testing
        embedding = self.memory._create_memory_embedding(memory_item)
        
        # Verify embedding shape
        self.assertEqual(len(embedding), self.memory.embedding_dimension)
        
        # Verify API was called once
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_memory_embedding_fallback(self, mock_post):
        """Test memory component's embedding fallback mechanism."""
        # Set up mock response to fail
        mock_post.return_value = MockResponse(raise_error=True)
        
        # Test memory embedding creation with failing API
        memory_item = {
            "type": "episodic",
            "description": "Mother said hello",
            "context": {"location": "home"}
        }
        
        # Should not raise exception due to fallback behavior
        embedding = self.memory._create_memory_embedding(memory_item)
        
        # Verify we got something back despite API failure
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), self.memory.embedding_dimension)
    
    @patch('requests.post')
    def test_dimension_handling(self, mock_post):
        """Test handling of embeddings with incorrect dimensions."""
        # Create oversized embedding
        oversized_embedding = np.random.randn(500).astype(np.float32).tolist()
        mock_response = {
            "data": [{"embedding": oversized_embedding, "index": 0}],
            "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m"
        }
        
        mock_post.return_value = MockResponse(json_data=mock_response)
        
        # Test memory embedding creation with oversized embedding
        memory_item = {
            "type": "episodic",
            "description": "Test dimension handling"
        }
        
        # Should truncate the embedding to correct dimension
        embedding = self.memory._create_memory_embedding(memory_item)
        
        # Verify embedding is the correct dimension
        self.assertEqual(len(embedding), self.memory.embedding_dimension)
        
        # Now test with undersized embedding
        undersized_embedding = np.random.randn(100).astype(np.float32).tolist()
        mock_response["data"][0]["embedding"] = undersized_embedding
        mock_post.return_value = MockResponse(json_data=mock_response)
        
        # Should pad the embedding to correct dimension
        embedding = self.memory._create_memory_embedding(memory_item)
        
        # Verify embedding is the correct dimension
        self.assertEqual(len(embedding), self.memory.embedding_dimension)


if __name__ == "__main__":
    unittest.main()