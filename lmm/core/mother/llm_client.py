"""
LLM client for the Mother module, based on the existing llm_module.py.
"""
import json
import requests
from typing import List, Dict, Union, Optional, Any
from dataclasses import dataclass

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger

logger = get_logger("lmm.mother.llm_client")

@dataclass
class Message:
    """A message in a conversation with the LLM."""
    role: str
    content: str

class LLMClient:
    """Client for interacting with the local LLM API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the LLM API. If None, uses the URL from config.
        """
        config = get_config()
        self.base_url = base_url or config.llm.base_url
        self.headers = {"Content-Type": "application/json"}
        logger.info(f"Initialized LLM client with base URL: {self.base_url}")

    # -------------------------
    # Chat Completion Methods
    # -------------------------
    def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, requests.Response]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of messages in the conversation
            model: Model to use for completion
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or response object if streaming
        """
        config = get_config()
        model = model or config.llm.chat_model
        temperature = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
        
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        logger.debug(f"Sending chat completion request with model: {model}")
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()

        if stream:
            return response
        return response.json()["choices"][0]["message"]["content"]

    # -------------------------
    # Structured JSON Completion
    # -------------------------
    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict, requests.Response]:
        """
        Generate a structured completion according to a JSON schema.
        
        Args:
            messages: List of messages in the conversation
            json_schema: JSON schema for the response
            model: Model to use for completion
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated JSON or response object if streaming
        """
        config = get_config()
        model = model or config.llm.chat_model
        temperature = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
        
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema
            },
            "stream": stream
        }
        
        logger.debug(f"Sending structured completion request with model: {model}")
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        
        if stream:
            return response
        
        content = response.json()["choices"][0]["message"]["content"]
        # Parse the JSON content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                return {"error": "Failed to parse JSON response", "raw_content": content}
        return content

    # -------------------------
    # Embedding Methods
    # -------------------------
    def get_embedding(
        self,
        texts: Union[str, List[str]],
        embedding_model: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for given input text(s).
        
        Args:
            texts: Text or list of texts to embed
            embedding_model: Model to use for embeddings
            
        Returns:
            Embedding vector(s)
        """
        config = get_config()
        embedding_model = embedding_model or config.llm.embedding_model
        
        endpoint = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": embedding_model,
            "input": texts
        }
        
        logger.debug(f"Sending embedding request with model: {embedding_model}")
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        
        embeddings_data = response.json()["data"]

        # Handle single or multiple embeddings
        if isinstance(texts, str):
            return embeddings_data[0]["embedding"]
        else:
            return [item["embedding"] for item in embeddings_data]

    # -------------------------
    # Streaming Helper
    # -------------------------
    def process_stream(self, response: requests.Response) -> str:
        """
        Process a streaming response.
        
        Args:
            response: Streaming response from the API
            
        Returns:
            Accumulated text from the stream
        """
        accumulated_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        json_str = line_text[6:]  # Remove 'data: ' prefix
                        if json_str.strip() == '[DONE]':
                            break
                        json_response = json.loads(json_str)
                        chunk = json_response.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        accumulated_text += chunk
                        yield chunk  # Yield each chunk for real-time processing
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing stream: {e}")
                    continue
        return accumulated_text 