# llm_module.py
import requests
import json
from typing import List, Dict, Union, Optional, Any, Callable
from dataclasses import dataclass
import logging
import threading
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str

class LLMClient:
    def __init__(self, base_url: str = "http://192.168.2.12:1234", timeout: float = 30.0):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the LLM API endpoint
            timeout: Default timeout for API requests in seconds
        """
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.default_timeout = timeout
        logger.info(f"Initialized LLM client with base URL: {base_url}")
    
    def _validate_messages(self, messages: List[Message]) -> List[Message]:
        """
        Validate and normalize message list.
        
        Args:
            messages: List of messages to validate
            
        Returns:
            Validated and normalized message list
        """
        if not messages:
            logger.warning("Empty messages array. Adding a default system message.")
            return [Message(role="system", content="You are a helpful AI assistant.")]
        
        # Check for empty content
        for i, msg in enumerate(messages):
            if not msg.content.strip():
                logger.warning(f"Message at index {i} has empty content. Adding placeholder.")
                messages[i] = Message(role=msg.role, content="Please provide information.")
        
        # Ensure the first message has a valid role
        if messages[0].role not in ["system", "user", "assistant"]:
            logger.warning(f"First message has invalid role: {messages[0].role}. Changing to 'system'.")
            messages[0] = Message(role="system", content=messages[0].content)
        
        return messages
    
    def _execute_with_timeout(self, func: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """
        Execute a function with a timeout.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            TimeoutError: If the function takes longer than the timeout
        """
        timeout_value = timeout or self.default_timeout
        result = [None]
        exception = [None]
        completed = [False]
        
        def worker():
            try:
                result[0] = func(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                exception[0] = e
                logger.error(f"Error in thread: {str(e)}")
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout_value)
        if not completed[0]:
            if thread.is_alive():
                logger.error(f"Operation timed out after {timeout_value} seconds")
                raise TimeoutError(f"Operation timed out after {timeout_value} seconds")
            elif exception[0]:
                logger.error(f"Operation failed: {str(exception[0])}")
                raise exception[0]
            else:
                logger.error("Unknown error occurred")
                raise RuntimeError("Unknown error occurred")
        
        if exception[0]:
            logger.error(f"Operation failed: {str(exception[0])}")
            raise exception[0]
            
        return result[0]

    def chat_completion(
        self,
        messages: List[Message],
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> Union[str, requests.Response]:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of messages for the conversation
            model: Model to use (default: qwen2.5-7b-instruct)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (-1 for model default)
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            
        Returns:
            Generated text or response object if streaming
            
        Raises:
            ValueError: For invalid arguments or API errors
            TimeoutError: If the request times out
            requests.RequestException: For network issues
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        logger.debug(f"Sending chat completion request to {endpoint}")
        
        # Validate messages
        messages = self._validate_messages(messages)
        logger.debug(f"Using {len(messages)} messages")
        
        # Prepare payload with optimizations for qwen2.5-7b-instruct
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "stream": stream
        }
        
        # Only add max_tokens if it's a positive value
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
            
        # Add specific parameters for qwen2.5-7b-instruct
        if "qwen" in model.lower():
            payload["top_p"] = 0.7  # Recommended for Qwen models
            if "repetition_penalty" not in payload:
                payload["repetition_penalty"] = 1.05  # Helps with code generation
        
        try:
            logger.info(f"Sending request with model={model}, temperature={temperature}, max_tokens={max_tokens}")
            
            def make_request():
                return requests.post(
                    endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=timeout or self.default_timeout
                )
            
            response = self._execute_with_timeout(make_request, timeout=timeout)
            logger.debug(f"Response status code: {response.status_code}")
            
            # Handle specific error status codes
            if response.status_code == 400:
                error_text = response.text
                logger.error(f"Bad request error: {error_text}")
                raise ValueError(f"LLM API returned 400 error: {error_text}")
            elif response.status_code == 404:
                logger.error(f"Model not found: {model}")
                raise ValueError(f"Model not found: {model}")
            elif response.status_code == 429:
                logger.error("Rate limit exceeded")
                raise ValueError("Rate limit exceeded, please try again later")
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            if stream:
                return response
                
            # Parse the response
            try:
                response_json = response.json()
            except json.JSONDecodeError:
                logger.error(f"JSON decode error with response text: {response.text}")
                raise ValueError(f"Failed to parse JSON response: {response.text}")
            
            # Validate response structure
            if "choices" not in response_json or not response_json["choices"]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'choices' field or empty choices")
                
            if "message" not in response_json["choices"][0]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'message' field in choices")
                
            if "content" not in response_json["choices"][0]["message"]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'content' field in message")
            
            content = response_json["choices"][0]["message"]["content"]
            logger.info(f"Successfully received response of length {len(content)}")
            return content
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout or self.default_timeout} seconds")
            raise TimeoutError(f"Request timed out after {timeout or self.default_timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during chat completion: {e}")
            raise

    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False,
        timeout: Optional[float] = None
    ) -> Union[Dict, requests.Response]:
        """
        Generate a structured JSON completion response.
        
        Args:
            messages: List of messages for the conversation
            json_schema: JSON schema for validating the response
            model: Model to use (default: qwen2.5-7b-instruct)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate (-1 for model default)
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON object or response object if streaming
            
        Raises:
            ValueError: For invalid arguments or API errors
            TimeoutError: If the request times out
            requests.RequestException: For network issues
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        logger.debug(f"Sending structured completion request to {endpoint}")
        
        # Validate messages
        messages = self._validate_messages(messages)
        logger.debug(f"Using {len(messages)} messages")
        
        # Add a system message hint if none exists
        if not any(msg.role == "system" for msg in messages):
            logger.debug("Adding system message hint for structured output")
            schema_hint = f"You must respond with JSON that matches this schema: {json.dumps(json_schema)}"
            messages.insert(0, Message(role="system", content=schema_hint))
        else:
            # Enhance existing system message with schema information
            for i, msg in enumerate(messages):
                if msg.role == "system":
                    enhanced_content = f"{msg.content}\n\nImportant: Your response must be valid JSON that follows this schema: {json.dumps(json_schema)}"
                    messages[i] = Message(role="system", content=enhanced_content)
                    break
        
        # Prepare payload
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema
            },
            "stream": stream
        }
        
        # Only add max_tokens if it's a positive value
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        
        # Add specific parameters for qwen2.5-7b-instruct
        if "qwen" in model.lower():
            payload["top_p"] = 0.9  # Higher for structured output
            payload["frequency_penalty"] = 0.1  # Helps with diverse responses
        
        try:
            logger.info(f"Sending structured request with temperature={temperature}, max_tokens={max_tokens}")
            
            def make_request():
                return requests.post(
                    endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=timeout or self.default_timeout
                )
            
            response = self._execute_with_timeout(make_request, timeout=timeout)
            logger.debug(f"Response status code: {response.status_code}")
            
            # Handle specific error status codes
            if response.status_code == 400:
                error_text = response.text
                logger.error(f"Bad request error: {error_text}")
                logger.error("This might indicate an issue with the messages format or JSON schema.")
                raise ValueError(f"LLM API returned 400 error: {error_text}")
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            if stream:
                return response
                
            # Parse the response
            try:
                response_json = response.json()
            except json.JSONDecodeError:
                logger.error(f"JSON decode error with response text: {response.text}")
                raise ValueError(f"Failed to parse JSON response: {response.text}")
            
            # Validate response structure
            if "choices" not in response_json or not response_json["choices"]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'choices' field or empty choices")
                
            if "message" not in response_json["choices"][0]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'message' field in choices")
                
            if "content" not in response_json["choices"][0]["message"]:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'content' field in message")
            
            content = response_json["choices"][0]["message"]["content"]
            logger.info(f"Successfully received JSON response of length {len(content)}")
            
            # Parse the content as JSON
            try:
                # Handle both direct JSON and string-encoded JSON
                if isinstance(content, dict):
                    # Some APIs return JSON directly
                    json_response = content
                    logger.debug("Received direct JSON object response")
                elif isinstance(content, str):
                    # Parse string-encoded JSON
                    content = content.strip()
                    # Handle cases where the API returns markdown-formatted JSON
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()
                    elif content.startswith("```") and content.endswith("```"):
                        content = content[3:-3].strip()
                    
                    json_response = json.loads(content)
                    logger.debug("Successfully parsed JSON from string response")
                else:
                    logger.error(f"Unexpected content type: {type(content)}")
                    raise ValueError(f"Unexpected content type: {type(content)}")
                
                return json_response
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Raw content: {content}")
                
                # Attempt to extract JSON from text in case it's embedded in other text
                try:
                    # Try to find JSON-like patterns
                    json_pattern = r'(\{.*\})'
                    matches = re.search(json_pattern, content, re.DOTALL)
                    if matches:
                        potential_json = matches.group(1)
                        logger.info("Found potential JSON pattern, attempting to parse")
                        json_response = json.loads(potential_json)
                        logger.info("Successfully extracted and parsed JSON")
                        return json_response
                except Exception as extraction_error:
                    logger.error(f"JSON extraction attempt failed: {extraction_error}")
                
                raise ValueError(f"Failed to parse structured JSON response: {e}")
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout or self.default_timeout} seconds")
            raise TimeoutError(f"Request timed out after {timeout or self.default_timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during structured completion: {e}")
            raise

    def get_embedding(
        self,
        texts: Union[str, List[str]],
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m",
        timeout: Optional[float] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for given input text(s).
        
        Args:
            texts: Input text or list of texts
            embedding_model: Model to use for embeddings
            timeout: Request timeout in seconds
            
        Returns:
            Single embedding vector or list of vectors
            
        Raises:
            ValueError: For invalid arguments or API errors
            TimeoutError: If the request times out
            requests.RequestException: For network issues
        """
        endpoint = f"{self.base_url}/v1/embeddings"
        logger.debug(f"Sending embedding request to {endpoint}")
        
        # Validate input
        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Input text cannot be empty")
        elif isinstance(texts, list):
            if not texts or not any(text.strip() for text in texts if isinstance(text, str)):
                raise ValueError("Input text list cannot be empty")
        else:
            raise ValueError(f"Invalid input type: {type(texts)}")
        
        payload = {
            "model": embedding_model,
            "input": texts
        }
        
        try:
            logger.info(f"Getting embeddings with model={embedding_model}")
            
            def make_request():
                return requests.post(
                    endpoint, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=timeout or self.default_timeout
                )
            
            response = self._execute_with_timeout(make_request, timeout=timeout)
            logger.debug(f"Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            try:
                response_json = response.json()
            except json.JSONDecodeError:
                logger.error(f"JSON decode error with response text: {response.text}")
                raise ValueError(f"Failed to parse JSON response: {response.text}")
            
            if "data" not in response_json:
                logger.error(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'data' field")
            
            embeddings_data = response_json["data"]
            
            # Handle single or multiple embeddings
            if isinstance(texts, str):
                if not embeddings_data or "embedding" not in embeddings_data[0]:
                    logger.error(f"Unexpected embeddings format: {embeddings_data}")
                    raise ValueError("Response missing embedding data")
                logger.info(f"Successfully received embedding vector")
                return embeddings_data[0]["embedding"]
            else:
                if not embeddings_data or any("embedding" not in item for item in embeddings_data):
                    logger.error(f"Unexpected embeddings format: {embeddings_data}")
                    raise ValueError("Response missing embedding data")
                logger.info(f"Successfully received {len(embeddings_data)} embedding vectors")
                return [item["embedding"] for item in embeddings_data]
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {timeout or self.default_timeout} seconds")
            raise TimeoutError(f"Request timed out after {timeout or self.default_timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding request: {e}")
            raise

    def process_stream(self, response: requests.Response) -> str:
        """
        Process a streaming response into accumulated text.
        
        Args:
            response: Streaming response object
            
        Returns:
            Accumulated text from the stream
        """
        accumulated_text = ""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_response = json.loads(line_text.replace('data: ', ''))
                            chunk = json_response.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            accumulated_text += chunk
                    except json.JSONDecodeError:
                        logger.warning("Could not decode JSON from stream line")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            
        return accumulated_text
    
    def stream_generator(self, response: requests.Response):
        """
        Generator that yields each token as it arrives from the stream.
        
        Args:
            response: Streaming response object
            
        Yields:
            Text chunks as they arrive
        """
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_response = json.loads(line_text.replace('data: ', ''))
                            chunk = json_response.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"Error processing stream chunk: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error in stream generator: {e}")

    def health_check(self, timeout: Optional[float] = 5.0) -> bool:
        """
        Check if the LLM API is available and responding.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            True if API is healthy, False otherwise
        """
        health_url = f"{self.base_url}/v1/health"
        try:
            logger.debug(f"Checking API health at {health_url}")
            response = requests.get(health_url, timeout=timeout or 5.0)
            is_healthy = response.status_code == 200
            logger.info(f"API health check {'succeeded' if is_healthy else 'failed'} with status {response.status_code}")
            return is_healthy
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False

# -------------------------
# Usage Example
# -------------------------
if __name__ == "__main__":
    client = LLMClient()

    # Verify API health
    if client.health_check():
        print("API is healthy")
    else:
        print("API health check failed")
        exit(1)

    # Chat completion usage example:
    messages = [
        Message(role="system", content="Always speak in rhymes."),
        Message(role="user", content="Tell me about your day.")
    ]
    chat_response = client.chat_completion(messages)
    print("\n\nChat Response:", chat_response)

    # Structured completion example:
    json_schema = {
        "name": "joke_response",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "joke": {"type": "string"}
            },
            "required": ["joke"] 
        }
    }
    messages = [
        Message(role="system", content="You are a helpful jokester."),
        Message(role="user", content="Tell me a joke.")
    ]

    structured_response = client.structured_completion(messages, json_schema)
    print("\n\nStructured Response:", structured_response)

    # Embedding usage example:
    embedding_response = client.get_embedding(["I feel happy today!"])
    print("\n\nEmbedding Response length:", len(embedding_response))