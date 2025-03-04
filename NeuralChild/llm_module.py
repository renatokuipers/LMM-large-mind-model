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
        """Initialize the LLM client with the base URL of the API."""
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}

    def chat_completion(
        self,
        messages: List[Message],
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = -1,
        stream: bool = False
    ) -> Union[str, requests.Response]:
        """Send a chat completion request to the API."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        if stream:
            return response
        else:
            return response.json()["choices"][0]["message"]["content"]

    def structured_completion(
        self,
        messages: List[Message],
        json_schema: Dict,
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 50,
    ) -> Dict:
        """Send a structured completion request to the API."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema
            },
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

def process_stream(response: requests.Response) -> str:
    """Process a streaming response."""
    accumulated_text = ""
    for line in response.iter_lines():
        if line:
            try:
                json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                if json_response.get("choices") and json_response["choices"][0].get("delta"):
                    delta = json_response["choices"][0]["delta"]
                    if "content" in delta:
                        accumulated_text += delta["content"]
            except json.JSONDecodeError:
                continue
    return accumulated_text

if __name__ == "__main__":
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
    structured_response = client.structured_completion(messages, joke_schema)
    print("Structured response:", structured_response)