# knowledge_agent.py
"""Knowledge agent for providing information and best practices."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import asyncio
import json
import re
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

from .agent_base import Agent, AgentStatus


class KnowledgeQuery(BaseModel):
    """A query to the knowledge agent."""
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class KnowledgeResponse(BaseModel):
    """A response from the knowledge agent."""
    query: str
    response: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class KnowledgeAgent(Agent):
    """
    Knowledge agent for providing information and best practices.
    
    This agent is responsible for answering questions about technologies, 
    frameworks, libraries, best practices, etc.
    """
    
    def __init__(
        self,
        name: str = "Knowledge",
        description: str = "Provides information and best practices",
        agent_type: str = "knowledge"
    ):
        """Initialize the KnowledgeAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Track queries and responses - using private variables
        self._queries: List[KnowledgeQuery] = []
        self._responses: List[KnowledgeResponse] = []
    
    # Define properties for safer access
    @property
    def queries(self) -> List[KnowledgeQuery]:
        return self._queries
    
    @queries.setter
    def queries(self, value: List[KnowledgeQuery]):
        self._queries = value
    
    @property
    def responses(self) -> List[KnowledgeResponse]:
        return self._responses
    
    @responses.setter
    def responses(self, value: List[KnowledgeResponse]):
        self._responses = value
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process knowledge-related requests.
        
        Args:
            input_data: Input data for the knowledge request
            
        Returns:
            Dictionary with the knowledge response
        """
        # Check if LLM integration is available
        if not self.llm:
            raise ValueError("LLM integration not available")
        
        # Extract the action
        action = input_data.get("action", "")
        
        if action == "query":
            query = input_data.get("query", "")
            context = input_data.get("context", {})
            return await self._answer_query(query, context)
        elif action == "suggest_best_practices":
            topic = input_data.get("topic", "")
            return await self._suggest_best_practices(topic)
        elif action == "explain_code":
            code = input_data.get("code", "")
            language = input_data.get("language", "python")
            return await self._explain_code(code, language)
        elif action == "find_references":
            topic = input_data.get("topic", "")
            return await self._find_references(topic)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _answer_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a knowledge query.
        
        Args:
            query: The query to answer
            context: Additional context for the query
            
        Returns:
            Dictionary with the query response
        """
        if not query:
            return {
                "success": False,
                "error": "Empty query provided"
            }
        
        # Create query record
        knowledge_query = KnowledgeQuery(
            query=query,
            context=context
        )
        
        self.queries.append(knowledge_query)
        
        # Define the schema for the LLM response
        schema = {
            "name": "knowledge_response",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Detailed response to the query"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sources of information for the response"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the response (0-1)"
                    }
                },
                "required": ["response", "confidence"]
            }
        }
        
        # Create context string
        context_str = ""
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are a knowledgeable software development expert. Answer the following query
        with accurate, detailed information. If you're uncertain about any part of your answer,
        indicate your level of confidence.
        
        Query: {query}
        {context_str}
        
        Provide a detailed response with references to relevant sources when applicable.
        """
        
        try:
            # Call the LLM with the prompt
            response_data = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            # Create knowledge response
            knowledge_response = KnowledgeResponse(
                query=query,
                response=response_data.get("response", ""),
                sources=response_data.get("sources", []),
                confidence=response_data.get("confidence", 0.5)
            )
            
            self.responses.append(knowledge_response)
            
            return {
                "success": True,
                "response": knowledge_response.response,
                "sources": knowledge_response.sources,
                "confidence": knowledge_response.confidence
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to answer query: {str(e)}"
            }
    
    async def _suggest_best_practices(self, topic: str) -> Dict[str, Any]:
        """
        Suggest best practices for a given topic.
        
        Args:
            topic: Topic to suggest best practices for
            
        Returns:
            Dictionary with best practices
        """
        if not topic:
            return {
                "success": False,
                "error": "Empty topic provided"
            }
        
        # Define the schema for the LLM response
        schema = {
            "name": "best_practices",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "best_practices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "example": {"type": "string"},
                                "rationale": {"type": "string"}
                            },
                            "required": ["title", "description"]
                        }
                    },
                    "references": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["best_practices"]
            }
        }
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are an expert software developer. Provide a comprehensive list of best practices for 
        the following topic:
        
        Topic: {topic}
        
        For each best practice, provide:
        1. A concise title
        2. A detailed description
        3. An example (when applicable)
        4. The rationale behind the practice
        
        Also provide references to authoritative sources.
        """
        
        try:
            # Call the LLM with the prompt
            response_data = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            return {
                "success": True,
                "best_practices": response_data.get("best_practices", []),
                "references": response_data.get("references", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to suggest best practices: {str(e)}"
            }
    
    async def _explain_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Explain a code snippet.
        
        Args:
            code: Code snippet to explain
            language: Programming language of the code
            
        Returns:
            Dictionary with code explanation
        """
        if not code:
            return {
                "success": False,
                "error": "Empty code provided"
            }
        
        # Define the schema for the LLM response
        schema = {
            "name": "code_explanation",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Detailed explanation of the code"
                    },
                    "key_concepts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Key programming concepts used in the code"
                        }
                    },
                    "potential_issues": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Potential issues or improvements in the code"
                        }
                    }
                },
                "required": ["explanation"]
            }
        }
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are an expert software developer. Explain the following {language} code:
        
        ```{language}
        {code}
        ```
        
        Provide:
        1. A detailed explanation of what the code does
        2. Key programming concepts used in the code
        3. Potential issues or improvements (if any)
        
        Focus on clarity and accuracy in your explanation.
        """
        
        try:
            # Call the LLM with the prompt
            response_data = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            return {
                "success": True,
                "explanation": response_data.get("explanation", ""),
                "key_concepts": response_data.get("key_concepts", []),
                "potential_issues": response_data.get("potential_issues", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to explain code: {str(e)}"
            }
    
    async def _find_references(self, topic: str) -> Dict[str, Any]:
        """
        Find references for a given topic.
        
        Args:
            topic: Topic to find references for
            
        Returns:
            Dictionary with references
        """
        if not topic:
            return {
                "success": False,
                "error": "Empty topic provided"
            }
        
        # Define the schema for the LLM response
        schema = {
            "name": "references",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "source": {"type": "string"},
                                "url": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["title", "source"]
                        }
                    },
                    "summary": {"type": "string"}
                },
                "required": ["references", "summary"]
            }
        }
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are a software development researcher. Find authoritative references for the following topic:
        
        Topic: {topic}
        
        Provide a list of high-quality references including:
        - Official documentation
        - Reputable books
        - Academic papers
        - Well-known blog posts or articles
        
        For each reference, include:
        1. Title
        2. Source (e.g., organization, author)
        3. URL (if applicable)
        4. Brief description of the content
        
        Also provide a summary of the key resources.
        """
        
        try:
            # Call the LLM with the prompt
            response_data = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            return {
                "success": True,
                "references": response_data.get("references", []),
                "summary": response_data.get("summary", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to find references: {str(e)}"
            } 