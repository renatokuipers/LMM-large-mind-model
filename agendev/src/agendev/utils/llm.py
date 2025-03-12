"""
LLM provider utility for AgenDev.

This module contains the LLMProvider class for interacting with language models.
It uses the local LLM client implementation provided in llm_module.py.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union

from ..llm_module import LLMClient, Message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Provider for Language Model interactions.
    
    This class integrates with the existing LLMClient for local model usage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider.
        
        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        
        # Default to local provider
        self.provider = config.get("provider", "local")
        
        # Default model for local setup (use Qwen model by default as shown in llm_module.py)
        self.model = config.get("model", "qwen2.5-7b-instruct")
        
        # Get base URL for local LLM API
        self.base_url = config.get("endpoint", "http://192.168.2.12:1234")
        
        # Initialize the LLM client
        self.llm_client = LLMClient(base_url=self.base_url)
        
        logger.info(f"LLM provider initialized: {self.provider}, model: {self.model}, endpoint: {self.base_url}")
    
    async def generate_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: User prompt
            system_message: System message for the LLM
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop: Stop sequences
            
        Returns:
            Dictionary with the completion result
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append(Message(role="system", content=system_message))
            
            messages.append(Message(role="user", content=prompt))
            
            # Get response from LLM client
            completion = self.llm_client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None,
                stream=False
            )
            
            return {
                "success": True,
                "completion": completion
            }
            
        except Exception as e:
            logger.exception(f"Error generating completion: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_structured_completion(
        self,
        prompt: str,
        json_schema: Dict,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Generate a structured completion that conforms to a JSON schema.
        
        Args:
            prompt: User prompt
            json_schema: JSON schema for the response
            system_message: System message for the LLM
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with the structured completion result
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append(Message(role="system", content=system_message))
            
            messages.append(Message(role="user", content=prompt))
            
            # Get structured response from LLM client
            result = self.llm_client.structured_completion(
                messages=messages,
                json_schema=json_schema,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else None,
                stream=False
            )
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.exception(f"Error generating structured completion: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_embeddings(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: String or list of strings to generate embeddings for
            
        Returns:
            Dictionary with the embeddings result
        """
        try:
            # Get embeddings from LLM client
            embeddings = self.llm_client.get_embedding(
                texts=texts,
                embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
            )
            
            return {
                "success": True,
                "embeddings": embeddings
            }
            
        except Exception as e:
            logger.exception(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_llm(self, agent):
        """Set LLM integration for an agent."""
        from ..llm_integration import LLMIntegration, LLMConfig
        
        # Create LLM integration using the existing implementation
        llm_config = LLMConfig(
            model=self.model,
            temperature=0.7,
            max_tokens=-1
        )
        
        llm_integration = LLMIntegration(
            base_url=self.base_url,
            config=llm_config
        )
        
        # Initialize the agent with the LLM integration
        agent.initialize(llm_integration)
        
        return agent 