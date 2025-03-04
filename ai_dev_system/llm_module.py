# llm_module.py
import os
import requests
import json
from typing import List, Dict, Union, Optional, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Representation of a chat message with role and content."""
    role: str
    content: str

class LLMClient:
    """Client for interacting with LLM APIs with flexible configuration."""
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialize the LLM client with configuration.
        
        Args:
            base_url: LLM API URL (defaults to env var LLM_API_URL or localhost)
            api_key: API key for authentication (defaults to env var LLM_API_KEY)
            default_model: Default model to use (defaults to env var LLM_DEFAULT_MODEL)
            timeout: Default request timeout in seconds
        """
        # Initialize configuration from arguments or environment variables
        self.base_url = (base_url or 
                         os.environ.get("LLM_API_URL", "http://localhost:1234")).rstrip('/')
        self.default_model = default_model or os.environ.get("LLM_DEFAULT_MODEL", "qwen2.5-7b-instruct")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        
        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            
        logger.info(f"LLM client initialized with URL: {self.base_url}")
        logger.info(f"Default model: {self.default_model}")
        
    def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        timeout: Optional[int] = None
    ) -> Union[str, requests.Response]:
        """
        Send a chat completion request to the LLM API.
        
        Args:
            messages: List of messages for the conversation
            model: Model to use (defaults to configured default_model)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (-1 for model default)
            stream: Whether to stream the response
            response_format: Optional response format (e.g., for JSON mode)
            stop: Optional stop sequences to end generation
            timeout: Request timeout in seconds
            
        Returns:
            If stream=False, returns the generated text as a string.
            If stream=True, returns the raw response object for streaming.
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model or self.default_model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if provided
        if response_format:
            payload["response_format"] = response_format
            
        if stop:
            payload["stop"] = stop
            
        request_timeout = timeout or self.timeout
        
        try:
            logger.info(f"Sending request to {endpoint}")
            logger.info(f"Model: {payload['model']}, Temperature: {temperature}")
                
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=request_timeout
            )
            response.raise_for_status()
            
            if stream:
                return response
            else:
                return response.json()["choices"][0]["message"]["content"]
                
        except requests.exceptions.RequestException as e:
            error_message = f"API request error: {str(e)}"
            logger.error(error_message)
            raise ConnectionError(error_message)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Send a structured completion request to get a JSON response.
        
        Args:
            messages: List of messages for the conversation
            json_schema: JSON schema for the expected response
            model: Model to use (defaults to configured default_model)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON response as a dictionary
        """
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema
        }
        
        try:
            response = self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format=response_format,
                timeout=timeout
            )
            
            if isinstance(response, str):
                return json.loads(response)
            else:
                raise TypeError("Expected string response, but got a different type")
                
        except json.JSONDecodeError as e:
            error_message = f"Failed to parse JSON response: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)

def process_stream(response: requests.Response) -> str:
    """
    Process a streaming response and return the accumulated text.
    
    Args:
        response: Streaming response from the LLM API
        
    Returns:
        Accumulated text from the stream
    """
    accumulated_text = ""
    for line in response.iter_lines():
        if not line:
            continue
            
        try:
            line_text = line.decode('utf-8')
            if not line_text.startswith('data: '):
                continue
                
            json_str = line_text.replace('data: ', '')
            
            # Check for [DONE] marker
            if json_str.strip() == '[DONE]':
                break
                
            json_response = json.loads(json_str)
            if not json_response.get("choices"):
                continue
                
            delta = json_response["choices"][0].get("delta", {})
            if "content" in delta:
                chunk = delta["content"]
                accumulated_text += chunk
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            
    return accumulated_text

def process_stream_with_callback(
    response: requests.Response, 
    callback: Callable[[str], None]
) -> str:
    """
    Process a streaming response with a callback for each chunk.
    
    Args:
        response: Streaming response from the LLM API
        callback: Function to call with each chunk of text
        
    Returns:
        Accumulated text from the stream
    """
    accumulated_text = ""
    for line in response.iter_lines():
        if not line:
            continue
            
        try:
            line_text = line.decode('utf-8')
            if not line_text.startswith('data: '):
                continue
                
            json_str = line_text.replace('data: ', '')
            
            # Check for [DONE] marker
            if json_str.strip() == '[DONE]':
                break
                
            json_response = json.loads(json_str)
            if not json_response.get("choices"):
                continue
                
            delta = json_response["choices"][0].get("delta", {})
            if "content" in delta:
                chunk = delta["content"]
                accumulated_text += chunk
                callback(chunk)
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            
    return accumulated_text

if __name__ == "__main__":
    # Set up basic logging for module testing
    logging.basicConfig(level=logging.ERROR)
    
    # Example usage
    client = LLMClient()
    messages = [
        Message(role="system", content="Always answer in rhymes."),
        Message(role="user", content="Introduce yourself.")
    ]
    response = client.chat_completion(messages, stream=False)
    print("Non-streaming response:", response)
    
    stream_response = client.chat_completion(messages, stream=True)
    streamed_text = process_stream(stream_response)
    print("Streamed response:", streamed_text)

    joke_schema = {
        "schema": {
            "type": "object",
            "properties": {
                "joke": {"type": "string"}
            },
            "required": ["joke"]
        }
    }
    
    structured_response = client.structured_completion(messages, joke_schema)
    print("Structured response:", structured_response)