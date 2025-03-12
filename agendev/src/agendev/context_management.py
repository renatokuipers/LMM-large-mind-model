# context_management.py
"""Embedding-based code representation and relationship mapping."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path
import os
import json
import numpy as np
from pydantic import BaseModel, Field, model_validator

from .utils.fs_utils import (
    resolve_path, 
    load_json, 
    save_json, 
    safe_save_json, 
    save_pickle, 
    load_pickle
)
from .llm_module import LLMClient, Message

class ContextElement(BaseModel):
    """Represents a single element in the context system."""
    id: UUID = Field(default_factory=uuid4)
    element_type: str  # "code", "comment", "function", "class", "file", etc.
    content: str
    source_file: Optional[str] = None
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Embedding vector (stored separately for efficiency)
    embedding_id: Optional[str] = None
    
    # Relationships
    parent_id: Optional[UUID] = None
    children_ids: List[UUID] = Field(default_factory=list)
    related_ids: Dict[str, List[UUID]] = Field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    def add_child(self, child_id: UUID) -> None:
        """Add a child to this element."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.updated_at = datetime.now()
    
    def add_relationship(self, relation_type: str, target_id: UUID) -> None:
        """Add a relationship to another element."""
        if relation_type not in self.related_ids:
            self.related_ids[relation_type] = []
        
        if target_id not in self.related_ids[relation_type]:
            self.related_ids[relation_type].append(target_id)
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this element."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

class CodeEmbedding(BaseModel):
    """Stores embedding vectors for context elements."""
    id: str
    vector: List[float]
    element_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)
    
    @model_validator(mode='after')
    def validate_vector(self) -> 'CodeEmbedding':
        """Ensure vector is properly formatted."""
        if not self.vector:
            raise ValueError("Embedding vector cannot be empty")
        return self

class ContextManager:
    """Manages code context using embeddings and relationships."""
    
    def __init__(self, 
                embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m",
                llm_base_url: str = "http://192.168.2.12:1234",
                context_dir: str = "artifacts/models/context"):
        """
        Initialize the context manager.
        
        Args:
            embedding_model: Model to use for generating embeddings
            llm_base_url: Base URL for the LLM service
            context_dir: Directory to store context data in
        """
        self.embedding_model = embedding_model
        self.llm_client = LLMClient(base_url=llm_base_url)
        self.context_dir = resolve_path(context_dir, create_parents=True)
        
        # Load existing elements and embeddings
        self.elements: Dict[UUID, ContextElement] = {}
        self.embeddings: Dict[str, CodeEmbedding] = {}
        
        self._load_context()
    
    def _load_context(self) -> None:
        """Load context data from disk."""
        elements_path = self.context_dir / "elements.json"
        if elements_path.exists():
            elements_data = load_json(elements_path)
            for element_dict in elements_data.get("elements", []):
                try:
                    element = ContextElement.model_validate(element_dict)
                    self.elements[element.id] = element
                except Exception as e:
                    print(f"Error loading context element: {e}")
        
        embeddings_path = self.context_dir / "embeddings.pkl"
        if embeddings_path.exists():
            self.embeddings = load_pickle(embeddings_path, default={})
    
    def _save_context(self) -> None:
        """Save context data to disk."""
        elements_data = {
            "elements": [element.model_dump() for element in self.elements.values()]
        }
        safe_save_json(elements_data, self.context_dir / "elements.json")
        save_pickle(self.embeddings, self.context_dir / "embeddings.pkl")
        
        # Save individual element files for easier access
        elements_dir = self.context_dir / "elements"
        os.makedirs(elements_dir, exist_ok=True)
        
        for element_id, element in self.elements.items():
            element_path = elements_dir / f"{element_id}.json"
            safe_save_json(element.model_dump(), element_path)
    
    def create_element(self, 
                     content: str, 
                     element_type: str, 
                     source_file: Optional[str] = None,
                     source_line_start: Optional[int] = None,
                     source_line_end: Optional[int] = None,
                     parent_id: Optional[UUID] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None,
                     generate_embedding: bool = True) -> UUID:
        """
        Create a new context element.
        
        Args:
            content: Text content of the element
            element_type: Type of element (code, comment, function, etc.)
            source_file: Path to the source file
            source_line_start: Starting line number in source
            source_line_end: Ending line number in source
            parent_id: UUID of parent element
            metadata: Additional metadata
            tags: List of tags
            generate_embedding: Whether to generate an embedding vector
            
        Returns:
            UUID of the created element
        """
        element = ContextElement(
            element_type=element_type,
            content=content,
            source_file=source_file,
            source_line_start=source_line_start,
            source_line_end=source_line_end,
            parent_id=parent_id,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Generate embedding if requested
        if generate_embedding:
            embedding_id = self._generate_embedding(element.id, content)
            element.embedding_id = embedding_id
        
        # Add as child to parent if parent exists
        if parent_id and parent_id in self.elements:
            self.elements[parent_id].add_child(element.id)
        
        # Store the element
        self.elements[element.id] = element
        
        # Save context data
        self._save_context()
        
        return element.id
    
    def _generate_embedding(self, element_id: UUID, content: str) -> str:
        """
        Generate an embedding vector for the content.
        
        Args:
            element_id: UUID of the element
            content: Text content to embed
            
        Returns:
            ID of the embedding
        """
        try:
            # Generate embedding using the LLM client
            vector = self.llm_client.get_embedding(content, embedding_model=self.embedding_model)
            
            # Create a unique ID for the embedding
            embedding_id = f"emb_{uuid4().hex[:8]}"
            
            # Store the embedding
            embedding = CodeEmbedding(
                id=embedding_id,
                vector=vector,
                element_id=element_id
            )
            
            self.embeddings[embedding_id] = embedding
            return embedding_id
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return ""
    
    def get_element(self, element_id: UUID) -> Optional[ContextElement]:
        """
        Get a context element by ID.
        
        Args:
            element_id: UUID of the element
            
        Returns:
            ContextElement or None if not found
        """
        return self.elements.get(element_id)
    
    def get_elements_by_type(self, element_type: str) -> List[ContextElement]:
        """
        Get all elements of a specific type.
        
        Args:
            element_type: Type of elements to retrieve
            
        Returns:
            List of matching elements
        """
        return [e for e in self.elements.values() if e.element_type == element_type]
    
    def get_elements_by_tag(self, tag: str) -> List[ContextElement]:
        """
        Get all elements with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching elements
        """
        return [e for e in self.elements.values() if tag in e.tags]
    
    def get_element_embedding(self, element_id: UUID) -> Optional[List[float]]:
        """
        Get the embedding vector for an element.
        
        Args:
            element_id: UUID of the element
            
        Returns:
            Embedding vector or None if not found
        """
        element = self.elements.get(element_id)
        if not element or not element.embedding_id:
            return None
        
        embedding = self.embeddings.get(element.embedding_id)
        if not embedding:
            return None
        
        return embedding.vector
    
    def get_children(self, element_id: UUID) -> List[ContextElement]:
        """
        Get all children of an element.
        
        Args:
            element_id: UUID of the parent element
            
        Returns:
            List of child elements
        """
        element = self.elements.get(element_id)
        if not element:
            return []
        
        return [self.elements.get(child_id) for child_id in element.children_ids 
                if child_id in self.elements]
    
    def get_related_elements(self, element_id: UUID, relation_type: Optional[str] = None) -> Dict[str, List[ContextElement]]:
        """
        Get elements related to the specified element.
        
        Args:
            element_id: UUID of the element
            relation_type: Optional type of relationship to filter by
            
        Returns:
            Dictionary mapping relation types to lists of elements
        """
        element = self.elements.get(element_id)
        if not element:
            return {}
        
        if relation_type:
            related_ids = element.related_ids.get(relation_type, [])
            return {relation_type: [self.elements.get(rel_id) for rel_id in related_ids 
                                   if rel_id in self.elements]}
        
        result = {}
        for rel_type, rel_ids in element.related_ids.items():
            result[rel_type] = [self.elements.get(rel_id) for rel_id in rel_ids 
                               if rel_id in self.elements]
        
        return result
    
    def get_top_elements(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get the most recently added elements in the context manager.
        
        Args:
            limit: Maximum number of elements to return
            
        Returns:
            List of dictionaries with element information
        """
        # Sort elements by creation time (newest first)
        sorted_elements = sorted(
            self.elements.values(), 
            key=lambda x: x.created_at if hasattr(x, 'created_at') else datetime.now(),
            reverse=True
        )
        
        # Return the first 'limit' elements
        result = []
        for element in sorted_elements[:limit]:
            result.append({
                "type": element.element_type,
                "name": element.metadata.get("name", "Unnamed") if element.metadata else "Unnamed",
                "file": element.source_file or "Unknown"
            })
        
        return result
    
    def find_similar_elements(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Tuple[ContextElement, float]]:
        """
        Find elements similar to the query text.
        
        Args:
            query: Text to compare against
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples containing (element, similarity_score)
        """
        # Generate embedding for the query
        try:
            query_vector = self.llm_client.get_embedding(query, embedding_model=self.embedding_model)
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return []
        
        # Calculate similarity with all embeddings
        similarities = []
        for emb_id, embedding in self.embeddings.items():
            element_id = embedding.element_id
            if element_id not in self.elements:
                continue
            
            similarity = self._calculate_similarity(query_vector, embedding.vector)
            if similarity >= threshold:
                similarities.append((self.elements[element_id], similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def index_file(self, file_path: Union[str, Path], 
                 element_types: List[str] = ["file", "class", "function", "method"]) -> List[UUID]:
        """
        Index a source code file, creating context elements.
        
        Args:
            file_path: Path to the file to index
            element_types: Types of elements to extract
            
        Returns:
            List of UUIDs of created elements
        """
        full_path = resolve_path(file_path)
        if not full_path.exists():
            return []
        
        # Read the file content
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Create a file-level element
        file_element_id = self.create_element(
            content=content,
            element_type="file",
            source_file=str(file_path),
            tags=["file", full_path.name, full_path.suffix[1:]]  # .py -> py
        )
        
        created_elements = [file_element_id]
        
        # Use LLM to extract code elements
        if "class" in element_types or "function" in element_types or "method" in element_types:
            elements = self._extract_code_elements(content, str(file_path), element_types)
            
            for elem in elements:
                element_id = self.create_element(
                    content=elem["content"],
                    element_type=elem["type"],
                    source_file=str(file_path),
                    source_line_start=elem.get("line_start"),
                    source_line_end=elem.get("line_end"),
                    parent_id=file_element_id,
                    metadata=elem.get("metadata", {}),
                    tags=[elem["type"], *elem.get("tags", [])]
                )
                created_elements.append(element_id)
        
        return created_elements
    
    def _extract_code_elements(self, content: str, file_path: str, element_types: List[str]) -> List[Dict]:
        """
        Extract code elements from content using LLM.
        
        Args:
            content: Source code content
            file_path: Path to the source file
            element_types: Types of elements to extract
            
        Returns:
            List of dictionaries describing the extracted elements
        """
        # Prepare the prompt for the LLM
        element_types_str = ", ".join(element_types)
        prompt = f"""
        Extract the following elements from this source code: {element_types_str}.
        For each element, provide:
        1. The element type
        2. The element name
        3. The full content including docstrings and comments
        4. The start and end line numbers
        5. Any relevant tags (e.g., 'class', 'public', 'private', etc.)
        
        Format the output as a JSON array of objects.
        
        Source file: {file_path}
        
        ```
        {content}
        ```
        """
        
        # Ask the LLM to extract elements
        messages = [
            Message(role="system", content="You are a code analysis assistant that extracts elements from source code."),
            Message(role="user", content=prompt)
        ]
        
        try:
            # Define the schema for structured output
            json_schema = {
                "name": "code_elements",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "name": {"type": "string"},
                                    "content": {"type": "string"},
                                    "line_start": {"type": "integer"},
                                    "line_end": {"type": "integer"},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "metadata": {"type": "object"}
                                },
                                "required": ["type", "name", "content"]
                            }
                        }
                    },
                    "required": ["elements"]
                }
            }
            
            response = self.llm_client.structured_completion(
                messages=messages,
                json_schema=json_schema,
                temperature=0.2,  # Low temperature for more precise extraction
                max_tokens=4000
            )
            
            # Parse the response
            if isinstance(response, dict) and "elements" in response:
                return response["elements"]
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if "elements" in parsed:
                        return parsed["elements"]
                except:
                    pass
            
            return []
            
        except Exception as e:
            print(f"Error extracting code elements: {e}")
            return []
    
    def generate_context_for_task(self, task_description: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate relevant context for a task.
        
        Args:
            task_description: Description of the task
            top_k: Maximum number of context elements to include
            
        Returns:
            Dictionary with relevant context
        """
        # Find similar elements
        similar_elements = self.find_similar_elements(task_description, top_k=top_k)
        
        context = {
            "task": task_description,
            "elements": [
                {
                    "id": str(elem.id),
                    "type": elem.element_type,
                    "content": elem.content,
                    "source_file": elem.source_file,
                    "relevance": score
                }
                for elem, score in similar_elements
            ]
        }
        
        return context
    
    def explain_code_relationships(self, element_id: UUID) -> str:
        """
        Use LLM to explain relationships between a code element and related elements.
        
        Args:
            element_id: UUID of the element
            
        Returns:
            Explanation of relationships
        """
        element = self.get_element(element_id)
        if not element:
            return "Element not found."
        
        # Get related elements
        related = self.get_related_elements(element_id)
        children = self.get_children(element_id)
        
        # Build context
        context = f"Element: {element.element_type} from {element.source_file}\n\n{element.content}\n\n"
        
        if children:
            context += "Children:\n"
            for child in children:
                context += f"- {child.element_type}: {child.content[:100]}...\n"
        
        for rel_type, elements in related.items():
            context += f"\n{rel_type.title()} relationships:\n"
            for rel_elem in elements:
                context += f"- {rel_elem.element_type}: {rel_elem.content[:100]}...\n"
        
        # Ask the LLM for an explanation
        messages = [
            Message(role="system", content="You are a code analysis assistant that explains relationships between code elements."),
            Message(role="user", content=f"Explain the relationships and dependencies in this code:\n\n{context}")
        ]
        
        try:
            explanation = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=1000
            )
            return explanation
        except Exception as e:
            return f"Error generating explanation: {e}"