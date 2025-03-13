# llm_module.py
import requests
import json
from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

class LLMClient:
    def __init__(self, base_url: str = "http://192.168.2.12:1234"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    # -------------------------
    # Chat Completion Methods
    # -------------------------
    def chat_completion(
        self,
        messages: List[Message],
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Union[str, requests.Response]:
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        # Debug print
        print(f"Sending chat completion request to {endpoint}")
        print(f"Number of messages: {len(messages)}")
        if len(messages) > 0:
            print(f"First message: {messages[0].role} - length {len(messages[0].content)}")
        
        # Validate messages to ensure we have at least one
        if not messages:
            print("WARNING: Empty messages array. Adding a default system message.")
            messages = [Message(role="system", content="You are a helpful AI assistant.")]
        
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            print(f"Sending request with temperature={temperature}, max_tokens={max_tokens}")
            response = requests.post(endpoint, headers=self.headers, json=payload)
            print(f"Response status code: {response.status_code}")
            
            # Check for specific error status codes
            if response.status_code == 400:
                print(f"Bad request error: {response.text}")
                print("This might indicate an issue with the messages format.")
                raise ValueError(f"LLM API returned 400 error: {response.text}")
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            if stream:
                return response
                
            # Parse the response
            response_json = response.json()
            
            # Check for empty or malformed response
            if "choices" not in response_json or not response_json["choices"]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'choices' field or empty choices")
                
            if "message" not in response_json["choices"][0]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'message' field in choices")
                
            if "content" not in response_json["choices"][0]["message"]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'content' field in message")
            
            content = response_json["choices"][0]["message"]["content"]
            print(f"Successfully received response of length {len(content)}")
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise
        except json.JSONDecodeError:
            print(f"JSON decode error with response text: {response.text}")
            raise ValueError(f"Failed to parse JSON response: {response.text}")
        except Exception as e:
            print(f"Unexpected error during chat completion: {e}")
            raise

    # -------------------------
    # Structured JSON Completion
    # -------------------------
    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Union[Dict, requests.Response]:
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        # Debug print
        print(f"Sending structured completion request to {endpoint}")
        print(f"Number of messages: {len(messages)}")
        if len(messages) > 0:
            print(f"First message: {messages[0].role} - length {len(messages[0].content)}")
        
        # Validate messages to ensure we have at least one
        if not messages:
            print("WARNING: Empty messages array. Adding a default system message.")
            messages = [Message(role="system", content="You are a helpful AI assistant.")]
        
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
        
        try:
            print(f"Sending structured request with temperature={temperature}, max_tokens={max_tokens}")
            response = requests.post(endpoint, headers=self.headers, json=payload)
            print(f"Response status code: {response.status_code}")
            
            # Check for specific error status codes
            if response.status_code == 400:
                print(f"Bad request error: {response.text}")
                print("This might indicate an issue with the messages format or JSON schema.")
                raise ValueError(f"LLM API returned 400 error: {response.text}")
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            if stream:
                return response
                
            # Parse the response
            response_json = response.json()
            
            # Check for empty or malformed response
            if "choices" not in response_json or not response_json["choices"]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'choices' field or empty choices")
                
            if "message" not in response_json["choices"][0]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'message' field in choices")
                
            if "content" not in response_json["choices"][0]["message"]:
                print(f"Unexpected response format: {response_json}")
                raise ValueError("Response missing 'content' field in message")
            
            content = response_json["choices"][0]["message"]["content"]
            print(f"Successfully received JSON response of length {len(content)}")
            
            # Parse the content as JSON
            try:
                json_response = json.loads(content)
                print(f"Successfully parsed JSON response with keys: {json_response.keys() if isinstance(json_response, dict) else 'non-dict'}")
                return json_response
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw content: {content}")
                raise ValueError(f"Failed to parse structured JSON response: {e}")
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise
        except json.JSONDecodeError:
            print(f"JSON decode error with response text: {response.text}")
            raise ValueError(f"Failed to parse JSON response: {response.text}")
        except Exception as e:
            print(f"Unexpected error during structured completion: {e}")
            raise

    # -------------------------
    # Embedding Methods
    # -------------------------
    def get_embedding(
        self,
        texts: Union[str, List[str]],
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for given input text(s)."""
        endpoint = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": embedding_model,
            "input": texts
        }
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
        accumulated_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                    chunk = json_response.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    accumulated_text += chunk
                except json.JSONDecodeError:
                    continue
        return accumulated_text
    
    def stream_generator(self, response: requests.Response):
        """Generator that yields each token as it arrives from the stream"""
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
                    print(f"Error processing stream chunk: {str(e)}")
                    continue

# -------------------------
# Usage Example (Embedding)
# -------------------------
if __name__ == "__main__":
    client = LLMClient()

    # Chat completion usage example:
    messages = [
        Message(role="system", content="Always speak in rhymes."),
        Message(role="user", content="Tell me about your day.")
    ]
    chat_response = client.chat_completion(messages)
    print("\n\nChat Response:", chat_response)

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
    print("\n\nEmbedding Response:", embedding_response)
