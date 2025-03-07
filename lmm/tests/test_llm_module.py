import pytest
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock
import importlib.util
import requests

# Import the llm_module from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import llm_module

class TestLLMModule:
    """Tests for the standalone LLM module functionality."""
    
    @pytest.fixture
    def llm_config(self):
        """Fixture providing LLM configuration."""
        return {
            "base_url": "http://test.api.local:1234",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100
        }
    
    @pytest.fixture
    def mock_response(self):
        """Fixture providing a mock LLM response."""
        return {
            "id": "test-response-id",
            "object": "completion",
            "created": 1677858242,
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from the mock LLM."
                    },
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
    
    @pytest.fixture
    def mock_embedding_response(self):
        """Fixture providing a mock embedding response."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0
                }
            ],
            "model": "text-embedding-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
    
    def test_llm_module_initialization(self, llm_config):
        """Test LLM module initialization."""
        # Initialize LLM client
        llm = llm_module.LLMClient(base_url=llm_config["base_url"])
        
        # Verify initialization
        assert llm.base_url == llm_config["base_url"]
        assert llm.headers == {"Content-Type": "application/json"}
    
    def test_text_generation(self, llm_config, mock_response):
        """Test text generation with the LLM module."""
        with patch("requests.post") as mock_post:
            # Configure mock response
            mock_post_response = MagicMock()
            mock_post_response.json.return_value = mock_response
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Create messages
            messages = [
                llm_module.Message(role="system", content="You are a helpful assistant."),
                llm_module.Message(role="user", content="Hello, how are you?")
            ]
            
            # Generate response
            response = llm.chat_completion(
                messages=messages,
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"]
            )
            
            # Verify response
            assert response == mock_response["choices"][0]["message"]["content"]
            
            # Verify request
            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            
            # Check endpoint
            assert kwargs["url"] == f"{llm_config['base_url']}/v1/chat/completions"
            
            # Check payload
            payload = kwargs["json"]
            assert payload["model"] == llm_config["model"]
            assert payload["temperature"] == llm_config["temperature"]
            assert payload["max_tokens"] == llm_config["max_tokens"]
            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "You are a helpful assistant."
    
    def test_structured_completion(self, llm_config, mock_response):
        """Test structured JSON completion."""
        with patch("requests.post") as mock_post:
            # Configure mock response
            mock_post_response = MagicMock()
            mock_post_response.json.return_value = {
                "id": "test-response-id",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"name": "John", "age": 30})
                        }
                    }
                ]
            }
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Create messages
            messages = [
                llm_module.Message(role="system", content="You are a helpful assistant that outputs JSON."),
                llm_module.Message(role="user", content="Provide a person's information in JSON format.")
            ]
            
            # Create JSON schema
            json_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name", "age"]
            }
            
            # Generate structured response
            response = llm.structured_completion(
                messages=messages,
                json_schema=json_schema,
                model=llm_config["model"]
            )
            
            # Verify response
            assert response["name"] == "John"
            assert response["age"] == 30
            
            # Verify request
            mock_post.assert_called_once()
            
            # Check payload
            payload = mock_post.call_args[1]["json"]
            assert "response_format" in payload
            assert payload["response_format"]["type"] == "json_object"
    
    def test_embedding_generation(self, llm_config, mock_embedding_response):
        """Test generating embeddings for text."""
        with patch("requests.post") as mock_post:
            # Configure mock response
            mock_post_response = MagicMock()
            mock_post_response.json.return_value = mock_embedding_response
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Generate embedding for single text
            embedding = llm.get_embedding("Test text")
            
            # Verify response
            assert isinstance(embedding, list)
            assert len(embedding) == 5
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Verify request
            mock_post.assert_called_once()
            
            # Check payload
            payload = mock_post.call_args[1]["json"]
            assert payload["input"] == "Test text"
            
            # Reset mock
            mock_post.reset_mock()
            
            # Configure mock response for multiple inputs
            mock_post_response.json.return_value = {
                "object": "list",
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ],
                "model": "text-embedding-model"
            }
            
            # Generate embeddings for multiple texts
            embeddings = llm.get_embedding(["Text 1", "Text 2"])
            
            # Verify response
            assert isinstance(embeddings, list)
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @pytest.mark.parametrize("temperature,max_tokens,expected_behavior", [
        (0.1, 50, "precise_short"),
        (0.7, 100, "balanced"),
        (1.0, 200, "creative_long")
    ])
    def test_parameter_influence(self, llm_config, mock_response, temperature, max_tokens, expected_behavior):
        """Test how different parameters influence LLM behavior."""
        with patch("requests.post") as mock_post:
            # Configure mock response
            mock_post_response = MagicMock()
            mock_post_response.json.return_value = mock_response
            mock_post_response.status_code = 200
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Create messages
            messages = [
                llm_module.Message(role="system", content="You are a helpful assistant."),
                llm_module.Message(role="user", content="Tell me about AI.")
            ]
            
            # Generate response with specific parameters
            llm.chat_completion(
                messages=messages,
                model=llm_config["model"],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Verify that parameters were correctly passed
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == temperature
            assert payload["max_tokens"] == max_tokens
    
    def test_error_handling(self, llm_config):
        """Test error handling in the LLM module."""
        with patch("requests.post") as mock_post:
            # Configure mock to raise an exception
            mock_post_response = MagicMock()
            mock_post_response.status_code = 400
            mock_post_response.json.return_value = {"error": "Invalid request"}
            mock_post_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad request")
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Create messages
            messages = [
                llm_module.Message(role="system", content="You are a helpful assistant."),
                llm_module.Message(role="user", content="Hello, how are you?")
            ]
            
            # Attempt to generate response
            with pytest.raises(requests.exceptions.HTTPError):
                llm.chat_completion(
                    messages=messages,
                    model=llm_config["model"]
                )
    
    def test_streaming_response(self, llm_config):
        """Test streaming response processing."""
        with patch("requests.post") as mock_post:
            # Create a mock streaming response
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            mock_post_response.iter_lines.return_value = [
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                b'data: {"choices":[{"delta":{"content":" world"}}]}',
                b'data: {"choices":[{"delta":{"content":"!"}}]}',
                b'data: [DONE]'
            ]
            mock_post.return_value = mock_post_response
            
            # Initialize client
            llm = llm_module.LLMClient(base_url=llm_config["base_url"])
            
            # Create messages
            messages = [
                llm_module.Message(role="user", content="Greet me")
            ]
            
            # Generate streaming response
            response = llm.chat_completion(
                messages=messages,
                model=llm_config["model"],
                stream=True
            )
            
            # Process the stream
            result = llm.process_stream(response)
            
            # Verify the result
            assert result == "Hello world!" 