"""Embedding-based code representation and relationship mapping."""
import os
import json
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .utils.fs_utils import (
    get_models_directory, 
    ensure_directory,
    save_json,
    load_json,
    save_binary,
    load_binary
)
from .llm_module import LLMClient, Message


class ContextItem(BaseModel):
    """Represents a single item in the context database."""
    id: UUID = Field(default_factory=uuid4)
    content_type: str  # "file", "code", "documentation", "conversation", etc.
    content_id: str  # Filename, code snippet ID, etc.
    content: str  # The actual text content
    embedding: Optional[List[float]] = None  # Vector embedding
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    # Relationships
    related_ids: List[UUID] = Field(default_factory=list)
    
    def set_embedding(self, embedding: List[float]) -> None:
        """Set the embedding for this context item."""
        self.embedding = embedding
        self.updated_at = datetime.now()
    
    def add_relationship(self, context_id: UUID) -> None:
        """Add a related context item."""
        if context_id not in self.related_ids:
            self.related_ids.append(context_id)
            self.updated_at = datetime.now()
    
    def remove_relationship(self, context_id: UUID) -> None:
        """Remove a related context item."""
        if context_id in self.related_ids:
            self.related_ids.remove(context_id)
            self.updated_at = datetime.now()


class ContextDatabase(BaseModel):
    """A database of context items with their embeddings."""
    items: Dict[UUID, ContextItem] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_item(self, item: ContextItem) -> UUID:
        """Add a context item to the database."""
        self.items[item.id] = item
        self.updated_at = datetime.now()
        return item.id
    
    def get_item(self, item_id: UUID) -> Optional[ContextItem]:
        """Get a context item from the database."""
        return self.items.get(item_id)
    
    def update_item(self, item: ContextItem) -> None:
        """Update a context item in the database."""
        self.items[item.id] = item
        self.updated_at = datetime.now()
    
    def remove_item(self, item_id: UUID) -> None:
        """Remove a context item from the database."""
        if item_id in self.items:
            del self.items[item_id]
            self.updated_at = datetime.now()


class ContextManager:
    """Manages context throughout the development process."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m"):
        """
        Initialize the context manager.
        
        Args:
            llm_client: LLM client for generating embeddings
            embedding_model: Model to use for embeddings
        """
        self.llm_client = llm_client or LLMClient()
        self.embedding_model = embedding_model
        self.context_db = self._load_database()
        
        # Initialize embedding matrix for similarity search
        self._update_embedding_matrix()
    
    def _load_database(self) -> ContextDatabase:
        """Load the context database from disk."""
        models_dir = get_models_directory()
        db_path = models_dir / 'context_db.json'
        
        if db_path.exists():
            db_data = load_json(db_path)
            if db_data:
                return ContextDatabase.model_validate(db_data)
        
        return ContextDatabase()
    
    def _save_database(self) -> None:
        """Save the context database to disk."""
        models_dir = get_models_directory()
        db_path = models_dir / 'context_db.json'
        
        save_json(self.context_db.model_dump(), db_path)
    
    def _update_embedding_matrix(self) -> None:
        """Update the embedding matrix for similarity search."""
        self.embedding_ids = []
        embeddings = []
        
        for item_id, item in self.context_db.items.items():
            if item.embedding is not None:
                self.embedding_ids.append(item_id)
                embeddings.append(item.embedding)
        
        if embeddings:
            self.embedding_matrix = np.array(embeddings)
        else:
            self.embedding_matrix = np.array([])
    
    def add_context(self, content: str, content_type: str, content_id: str, tags: List[str] = None) -> UUID:
        """
        Add a new context item.
        
        Args:
            content: The text content
            content_type: Type of content (file, code, etc.)
            content_id: Identifier for the content
            tags: Tags for categorizing the content
            
        Returns:
            ID of the new context item
        """
        item = ContextItem(
            content_type=content_type,
            content_id=content_id,
            content=content,
            tags=tags or []
        )
        
        # Generate embedding
        embedding = self.llm_client.get_embedding(content, self.embedding_model)
        item.set_embedding(embedding)
        
        # Add to database
        item_id = self.context_db.add_item(item)
        
        # Update embedding matrix
        self._update_embedding_matrix()
        
        # Save to disk
        self._save_database()
        
        return item_id
    
    def update_context(self, item_id: UUID, content: str = None, tags: List[str] = None) -> None:
        """
        Update an existing context item.
        
        Args:
            item_id: ID of the context item
            content: New content (if None, keeps existing content)
            tags: New tags (if None, keeps existing tags)
        """
        item = self.context_db.get_item(item_id)
        if not item:
            return
        
        updated = False
        
        if content is not None and content != item.content:
            item.content = content
            # Re-generate embedding
            embedding = self.llm_client.get_embedding(content, self.embedding_model)
            item.set_embedding(embedding)
            updated = True
        
        if tags is not None:
            item.tags = tags
            updated = True
        
        if updated:
            item.updated_at = datetime.now()
            self.context_db.update_item(item)
            self._update_embedding_matrix()
            self._save_database()
    
    def remove_context(self, item_id: UUID) -> None:
        """
        Remove a context item.
        
        Args:
            item_id: ID of the context item to remove
        """
        self.context_db.remove_item(item_id)
        self._update_embedding_matrix()
        self._save_database()
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[UUID, float]]:
        """
        Find context items similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.embedding_matrix.size:
            return []
        
        # Generate embedding for the query
        query_embedding = self.llm_client.get_embedding(query, self.embedding_model)
        
        # Calculate cosine similarity
        query_array = np.array(query_embedding)
        norm_query = np.linalg.norm(query_array)
        norm_matrix = np.linalg.norm(self.embedding_matrix, axis=1)
        
        # Avoid division by zero
        if norm_query == 0 or np.any(norm_matrix == 0):
            return []
        
        # Calculate cosine similarity
        dot_product = np.dot(self.embedding_matrix, query_array)
        similarity = dot_product / (norm_matrix * norm_query)
        
        # Get top-k results
        top_indices = np.argsort(-similarity)[:top_k]
        
        return [(self.embedding_ids[i], float(similarity[i])) for i in top_indices]
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> List[ContextItem]:
        """
        Get context items relevant to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of ContextItem objects
        """
        similar_items = self.search_similar(query, top_k)
        
        return [self.context_db.get_item(item_id) for item_id, _ in similar_items if self.context_db.get_item(item_id) is not None]
    
    def build_context_for_task(self, task_description: str, max_tokens: int = 4000) -> str:
        """
        Build a context string for a task.
        
        Args:
            task_description: Description of the task
            max_tokens: Maximum tokens for the context
            
        Returns:
            Context string for the task
        """
        relevant_items = self.get_relevant_context(task_description)
        
        context = f"Task: {task_description}\n\n"
        context += "Relevant context:\n\n"
        
        for item in relevant_items:
            item_context = f"--- {item.content_type}: {item.content_id} ---\n{item.content}\n\n"
            
            # Simple token count approximation (no exact tokenization)
            if len(context + item_context) / 4 > max_tokens:  # Rough estimate of 4 chars per token
                break
                
            context += item_context
        
        return context
    
    def add_file_to_context(self, file_path: Union[str, Path], tags: List[str] = None) -> UUID:
        """
        Add a file to the context database.
        
        Args:
            file_path: Path to the file
            tags: Tags for the file
            
        Returns:
            ID of the new context item
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.add_context(
            content=content,
            content_type="file",
            content_id=str(path),
            tags=tags or []
        )
    
    def add_code_snippet_to_context(self, code: str, snippet_id: str, tags: List[str] = None) -> UUID:
        """
        Add a code snippet to the context database.
        
        Args:
            code: The code snippet
            snippet_id: Identifier for the snippet
            tags: Tags for the snippet
            
        Returns:
            ID of the new context item
        """
        return self.add_context(
            content=code,
            content_type="code",
            content_id=snippet_id,
            tags=tags or []
        )
    
    def add_conversation_to_context(self, messages: List[Dict[str, str]], conversation_id: str, tags: List[str] = None) -> UUID:
        """
        Add a conversation to the context database.
        
        Args:
            messages: List of message dictionaries
            conversation_id: Identifier for the conversation
            tags: Tags for the conversation
            
        Returns:
            ID of the new context item
        """
        content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return self.add_context(
            content=content,
            content_type="conversation",
            content_id=conversation_id,
            tags=tags or []
        )