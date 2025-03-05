"""
Memory component for the Neural Child's mind.

This module contains the implementation of the memory component that handles
different types of memory (episodic, semantic, and procedural) for the simulated mind.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import random
import time
import json
from pathlib import Path
from collections import deque, defaultdict
import faiss
import numpy as np
import requests

from neural_child.mind.base import NeuralComponent

class MemoryComponent(NeuralComponent):
    """Memory component that handles different types of memory."""
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        output_size: int = 64,
        name: str = "memory_component",
        memory_capacity: int = 10000,
        embedding_api_url: str = "http://192.168.2.12:1234/v1/embeddings",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m",
        embedding_dimension: int = 384
    ):
        """Initialize the memory component.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Size of hidden layer
            output_size: Size of output vectors
            name: Name of the component
            memory_capacity: Maximum number of memories to store
            embedding_api_url: URL for the embedding API
            embedding_model: Model to use for embeddings
            embedding_dimension: Dimension of embedding vectors (default: 384 for Nomic embedding model)
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size, name=name)
        
        # Memory development metrics
        self.working_memory_capacity = 0.1  # Initial capacity (scales with age)
        self.long_term_memory_development = 0.05
        self.memory_consolidation_rate = 0.1
        self.memory_retrieval_accuracy = 0.05
        
        # Memory storage
        self.episodic_memories = []
        self.semantic_memories = {}
        self.procedural_memories = {}
        
        # Working memory (short-term)
        self.working_memory = deque(maxlen=5)
        
        # Memory capacity
        self.memory_capacity = memory_capacity
        
        # Embedding API
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        
        # Episodic memory index for similarity search
        self.embedding_dimension = embedding_dimension  # Dimension for text embeddings
        self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Tracking memory development
        self.total_experiences = 0
        self.memory_consolidations = 0
        self.successful_retrievals = 0
        self.failed_retrievals = 0
        
        # Initialize experience counter
        self.experience_counter = 0
        
        # Initialize emotional association tracking
        self.emotional_associations = defaultdict(lambda: defaultdict(float))
        
        # Initialize timestamp for memory consolidation
        self.last_consolidation_time = time.time()
        
        # Set of all concepts learned
        self.concepts = set()
        
        # Memory decay parameters
        self.decay_rate = 0.01  # Rate at which memories decay
        self.rehearsal_boost = 0.2  # Boost to memory strength when rehearsed
        
        # Memory formation thresholds
        self.emotional_significance_threshold = 0.3  # Threshold for emotional significance
        self.novelty_threshold = 0.2  # Threshold for novelty
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs.
        
        Args:
            inputs: Dictionary of inputs to the component
                - current_experience: Dictionary representing the current experience
                - emotional_state: Dictionary of the child's emotional state
                - developmental_stage: String representing the developmental stage
                - age_months: Float representing the age in months
                - query: Optional dictionary for memory retrieval
                
        Returns:
            Dictionary of outputs from the component
                - working_memory: List of items in working memory
                - retrieved_memories: List of retrieved memories (if query provided)
                - memory_development: Dictionary of memory development metrics
        """
        # Extract inputs
        current_experience = inputs.get("current_experience", {})
        emotional_state = inputs.get("emotional_state", {})
        developmental_stage = inputs.get("developmental_stage", "Prenatal")
        age_months = inputs.get("age_months", 0.0)
        query = inputs.get("query", None)
        
        # Update memory development metrics based on age
        self._update_memory_development(age_months, developmental_stage)
        
        # Process current experience
        if current_experience:
            self._process_experience(current_experience, emotional_state)
        
        # Retrieve memories if query provided
        retrieved_memories = []
        if query:
            retrieved_memories = self._retrieve_memories(query)
        
        # Apply memory decay
        self._apply_memory_decay()
        
        # Consolidate memories
        self._consolidate_memories()
        
        # Prepare outputs
        outputs = {
            "working_memory": list(self.working_memory),
            "retrieved_memories": retrieved_memories,
            "memory_development": {
                "working_memory_capacity": self.working_memory_capacity,
                "long_term_memory_development": self.long_term_memory_development,
                "memory_consolidation_rate": self.memory_consolidation_rate,
                "memory_retrieval_accuracy": self.memory_retrieval_accuracy
            }
        }
        
        # Update activation level
        self.update_activation(self.long_term_memory_development)
        
        return outputs
    
    def _process_experience(self, experience: Dict[str, Any], emotional_state: Dict[str, float]):
        """Process a new experience.
        
        Args:
            experience: Dictionary representing the experience
            emotional_state: Dictionary of the child's emotional state
        """
        # Skip if experience is empty
        if not experience:
            return
        
        # Add timestamp to experience
        experience_with_metadata = experience.copy()
        experience_with_metadata["timestamp"] = time.time()
        experience_with_metadata["memory_strength"] = 1.0  # Initial memory strength
        
        # Calculate emotional significance
        emotional_significance = max(emotional_state.values()) if emotional_state else 0.0
        experience_with_metadata["emotional_significance"] = emotional_significance
        
        # Calculate novelty
        novelty = self._calculate_novelty(experience)
        experience_with_metadata["novelty"] = novelty
        
        # Determine if experience should be stored in episodic memory
        should_store_episodic = (
            emotional_significance >= self.emotional_significance_threshold or
            novelty >= self.novelty_threshold or
            random.random() < self.long_term_memory_development  # Random chance based on development
        )
        
        # Store in episodic memory if significant
        if should_store_episodic:
            # Generate a unique ID for the memory
            memory_id = f"memory_{len(self.episodic_memories)}"
            experience_with_metadata["id"] = memory_id
            
            # Add to episodic memory
            self.episodic_memories.append(experience_with_metadata)
            
            try:
                # Create embedding for the memory
                embedding = self._create_memory_embedding(experience_with_metadata)
                
                # Verify embedding dimension
                if len(embedding) != self.embedding_dimension:
                    # Ensure correct dimension
                    if len(embedding) > self.embedding_dimension:
                        embedding = embedding[:self.embedding_dimension]
                    else:
                        padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                        embedding = np.concatenate([embedding, padding])
                
                # Add to FAISS index
                self.episodic_index.add(np.array([embedding], dtype=np.float32))
            except Exception as e:
                print(f"Error adding memory to FAISS index: {e}")
                # If there was an error adding to the index, consider rebuilding it
                if len(self.episodic_memories) > 1:  # Only if we have at least one other memory
                    print("Attempting to rebuild the index...")
                    self._rebuild_episodic_index()
            
            # Limit episodic memory size
            if len(self.episodic_memories) > self.memory_capacity:
                # Remove oldest memory
                self.episodic_memories.pop(0)
                # Rebuild FAISS index
                self._rebuild_episodic_index()
        
        # Add to working memory
        self.working_memory.append(experience_with_metadata)
        
        # Extract semantic information if possible
        self._extract_semantic_information(experience_with_metadata)
        
        # Extract procedural information if possible
        self._extract_procedural_information(experience_with_metadata)
    
    def _calculate_novelty(self, experience: Dict[str, Any]) -> float:
        """Calculate the novelty of an experience.
        
        Args:
            experience: Dictionary representing the experience
            
        Returns:
            Float representing the novelty (0.0 to 1.0)
        """
        # If no memories, everything is novel
        if not self.episodic_memories:
            return 1.0
        
        # Create embedding for the experience
        embedding = self._create_memory_embedding(experience)
        
        # Find most similar memory
        D, I = self.episodic_index.search(np.array([embedding], dtype=np.float32), 1)
        
        # Convert distance to novelty (higher distance = higher novelty)
        if len(D) > 0 and len(D[0]) > 0:
            # Normalize distance to 0-1 range (assuming max distance around 10)
            novelty = min(1.0, D[0][0] / 10.0)
            return novelty
        else:
            return 1.0  # If no similar memory found, it's completely novel
    
    def _create_memory_embedding(self, memory: Dict[str, Any]) -> np.ndarray:
        """Create an embedding vector for a memory using the embedding API.
        
        Args:
            memory: Memory dictionary containing content to embed
            
        Returns:
            Numpy array containing the embedding vector with dimension matching self.embedding_dimension
        """
        # Extract relevant content from the memory
        content_parts = []
        
        # Add memory type
        memory_type = memory.get("type", "unknown")
        content_parts.append(f"Type: {memory_type}")
        
        # Add content based on memory type
        if memory_type == "episodic":
            # For episodic memories, include the description and context
            if "description" in memory:
                content_parts.append(f"Description: {memory['description']}")
            
            if "context" in memory:
                context = memory["context"]
                if isinstance(context, dict):
                    context_str = ", ".join([f"{k}: {v}" for k, v in context.items() 
                                           if isinstance(v, (str, int, float))])
                    content_parts.append(f"Context: {context_str}")
                elif isinstance(context, str):
                    content_parts.append(f"Context: {context}")
            
            # Include emotional associations
            if "emotional_associations" in memory and isinstance(memory["emotional_associations"], dict):
                emotions_str = ", ".join([f"{emotion}: {value:.2f}" 
                                        for emotion, value in memory["emotional_associations"].items()])
                content_parts.append(f"Emotions: {emotions_str}")
                
        elif memory_type == "semantic":
            # For semantic memories, include the concept and definition
            if "concept" in memory:
                content_parts.append(f"Concept: {memory['concept']}")
            
            if "definition" in memory:
                content_parts.append(f"Definition: {memory['definition']}")
                
            # Include related concepts
            if "related_concepts" in memory and isinstance(memory["related_concepts"], list):
                related = ", ".join(memory["related_concepts"])
                content_parts.append(f"Related: {related}")
                
        elif memory_type == "procedural":
            # For procedural memories, include the action and outcome
            if "action" in memory:
                content_parts.append(f"Action: {memory['action']}")
            
            if "outcome" in memory:
                content_parts.append(f"Outcome: {memory['outcome']}")
                
            # Include steps if available
            if "steps" in memory and isinstance(memory["steps"], list):
                steps = " → ".join(memory["steps"])
                content_parts.append(f"Steps: {steps}")
        
        # Combine all parts into a single string
        content_text = " | ".join(content_parts)
        
        try:
            # Call the embedding API
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.embedding_model,
                "input": content_text
            }
            
            response = requests.post(self.embedding_api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Extract the embedding from the response
            embedding_data = response.json()
            embedding = embedding_data["data"][0]["embedding"]
            
            # Convert to numpy array
            embedding_np = np.array(embedding, dtype=np.float32)
            
            # Ensure the embedding dimension matches self.embedding_dimension
            if len(embedding_np) != self.embedding_dimension:
                if len(embedding_np) > self.embedding_dimension:
                    # Truncate if too large
                    embedding_np = embedding_np[:self.embedding_dimension]
                else:
                    # Pad with zeros if too small
                    padding = np.zeros(self.embedding_dimension - len(embedding_np), dtype=np.float32)
                    embedding_np = np.concatenate([embedding_np, padding])
            
            return embedding_np
            
        except Exception as e:
            # Fallback to a simpler method if API call fails
            print(f"Embedding API call failed: {e}. Using fallback embedding method.")
            
            # Create a deterministic embedding based on the hash of the content
            seed = hash(content_text) % 10000
            np.random.seed(seed)
            
            # Generate a random embedding with the correct dimension
            embedding = np.random.randn(self.embedding_dimension).astype(np.float32)
            
            # Normalize the embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
    
    def _rebuild_episodic_index(self):
        """Rebuild the FAISS index for episodic memories."""
        # Reset index
        self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Add all memories to index
        if len(self.episodic_memories) > 0:
            embeddings = []
            for memory in self.episodic_memories:
                try:
                    embedding = self._create_memory_embedding(memory)
                    # Verify embedding dimension
                    if len(embedding) != self.embedding_dimension:
                        # Ensure correct dimension
                        if len(embedding) > self.embedding_dimension:
                            embedding = embedding[:self.embedding_dimension]
                        else:
                            padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                            embedding = np.concatenate([embedding, padding])
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error creating embedding for memory during index rebuild: {e}")
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.episodic_index.add(embeddings_array)
    
    def _extract_semantic_information(self, experience: Dict[str, Any]):
        """Extract semantic information from an experience.
        
        Args:
            experience: Dictionary representing the experience
        """
        # Skip if experience doesn't have the right structure
        if not isinstance(experience, dict):
            return
        
        # Extract concepts and their attributes
        for key, value in experience.items():
            if key in ["timestamp", "memory_strength", "id", "emotional_significance", "novelty"]:
                continue
            
            # Only process if development is sufficient
            if random.random() > self.long_term_memory_development:
                continue
            
            # Create or update semantic memory for this concept
            if key not in self.semantic_memories:
                self.semantic_memories[key] = {
                    "instances": [],
                    "attributes": defaultdict(int),
                    "memory_strength": 1.0
                }
            
            # Add instance
            self.semantic_memories[key]["instances"].append(value)
            
            # Limit instances to prevent memory issues
            if len(self.semantic_memories[key]["instances"]) > 20:
                self.semantic_memories[key]["instances"] = self.semantic_memories[key]["instances"][-20:]
            
            # Update attributes if value is a dictionary
            if isinstance(value, dict):
                for attr_key, attr_value in value.items():
                    if isinstance(attr_value, (str, int, float, bool)):
                        self.semantic_memories[key]["attributes"][f"{attr_key}:{attr_value}"] += 1
            
            # Boost memory strength
            self.semantic_memories[key]["memory_strength"] = min(
                1.0, self.semantic_memories[key]["memory_strength"] + self.rehearsal_boost
            )
    
    def _extract_procedural_information(self, experience: Dict[str, Any]):
        """Extract procedural information from an experience.
        
        Args:
            experience: Dictionary representing the experience
        """
        # Skip if experience doesn't have the right structure
        if not isinstance(experience, dict) or "action" not in experience:
            return
        
        # Only process if development is sufficient
        if random.random() > self.long_term_memory_development:
            return
        
        action = experience.get("action", "")
        result = experience.get("result", "")
        
        if action:
            # Create or update procedural memory for this action
            if action not in self.procedural_memories:
                self.procedural_memories[action] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "contexts": [],
                    "memory_strength": 1.0
                }
            
            # Update success/failure counts
            if result == "success":
                self.procedural_memories[action]["success_count"] += 1
            elif result == "failure":
                self.procedural_memories[action]["failure_count"] += 1
            
            # Add context
            if "context" in experience:
                self.procedural_memories[action]["contexts"].append(experience["context"])
                
                # Limit contexts to prevent memory issues
                if len(self.procedural_memories[action]["contexts"]) > 10:
                    self.procedural_memories[action]["contexts"] = self.procedural_memories[action]["contexts"][-10:]
            
            # Boost memory strength
            self.procedural_memories[action]["memory_strength"] = min(
                1.0, self.procedural_memories[action]["memory_strength"] + self.rehearsal_boost
            )
    
    def _retrieve_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories based on a query.
        
        Args:
            query: Dictionary representing the query
                - type: Type of memory to retrieve ("episodic", "semantic", "procedural")
                - content: Content to search for
                - limit: Maximum number of memories to retrieve
                
        Returns:
            List of retrieved memories
        """
        memory_type = query.get("type", "episodic")
        content = query.get("content", {})
        limit = query.get("limit", 5)
        
        # Apply retrieval accuracy (chance of failing to retrieve)
        if random.random() > self.memory_retrieval_accuracy:
            return []
        
        if memory_type == "episodic":
            return self._retrieve_episodic_memories(content, limit)
        elif memory_type == "semantic":
            return self._retrieve_semantic_memories(content, limit)
        elif memory_type == "procedural":
            return self._retrieve_procedural_memories(content, limit)
        else:
            return []
    
    def _retrieve_episodic_memories(self, content: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Retrieve episodic memories based on content.
        
        Args:
            content: Content to search for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of retrieved memories
        """
        if not self.episodic_memories:
            return []
        
        # Create embedding for the query
        query_embedding = self._create_memory_embedding(content)
        
        # Find similar memories
        D, I = self.episodic_index.search(np.array([query_embedding], dtype=np.float32), min(limit, len(self.episodic_memories)))
        
        # Retrieve memories
        retrieved_memories = []
        for i in range(len(I[0])):
            memory_id = self.episodic_memories[I[0][i]]["id"]
            
            # Find memory with this ID
            for memory in self.episodic_memories:
                if memory.get("id") == memory_id:
                    # Apply memory strength filter
                    if memory.get("memory_strength", 0.0) > 0.2:
                        retrieved_memories.append(memory)
                        
                        # Boost memory strength (rehearsal effect)
                        memory["memory_strength"] = min(1.0, memory.get("memory_strength", 0.0) + self.rehearsal_boost)
                    break
        
        return retrieved_memories
    
    def _retrieve_semantic_memories(self, content: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Retrieve semantic memories based on content.
        
        Args:
            content: Content to search for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of retrieved memories
        """
        if not self.semantic_memories:
            return []
        
        retrieved_memories = []
        
        # Extract concepts to search for
        concepts = content.get("concepts", [])
        
        for concept in concepts:
            if concept in self.semantic_memories:
                memory = self.semantic_memories[concept]
                
                # Apply memory strength filter
                if memory.get("memory_strength", 0.0) > 0.2:
                    retrieved_memories.append({
                        "concept": concept,
                        "instances": memory["instances"],
                        "attributes": dict(memory["attributes"]),
                        "memory_strength": memory["memory_strength"]
                    })
                    
                    # Boost memory strength (rehearsal effect)
                    memory["memory_strength"] = min(1.0, memory.get("memory_strength", 0.0) + self.rehearsal_boost)
                    
                    if len(retrieved_memories) >= limit:
                        break
        
        return retrieved_memories
    
    def _retrieve_procedural_memories(self, content: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Retrieve procedural memories based on content.
        
        Args:
            content: Content to search for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of retrieved memories
        """
        if not self.procedural_memories:
            return []
        
        retrieved_memories = []
        
        # Extract actions to search for
        actions = content.get("actions", [])
        
        if isinstance(self.procedural_memories, dict):
            # Dictionary-based retrieval
            for action in actions:
                if action in self.procedural_memories:
                    memory = self.procedural_memories[action]
                    
                    # Apply memory strength filter
                    if memory.get("memory_strength", 0.0) > 0.2:
                        retrieved_memories.append({
                            "action": action,
                            "success_count": memory.get("success_count", 0),
                            "failure_count": memory.get("failure_count", 0),
                            "contexts": memory.get("contexts", []),
                            "memory_strength": memory.get("memory_strength", 0.0)
                        })
                        
                        # Boost memory strength (rehearsal effect)
                        memory["memory_strength"] = min(1.0, memory.get("memory_strength", 0.0) + self.rehearsal_boost)
                        
                        if len(retrieved_memories) >= limit:
                            break
        elif isinstance(self.procedural_memories, list):
            # List-based retrieval
            for memory in self.procedural_memories:
                if "action" in memory and memory["action"] in actions:
                    # Apply memory strength filter
                    if memory.get("memory_strength", 0.0) > 0.2:
                        retrieved_memories.append({
                            "action": memory["action"],
                            "success_count": memory.get("success_count", 0),
                            "failure_count": memory.get("failure_count", 0),
                            "contexts": memory.get("contexts", []),
                            "memory_strength": memory.get("memory_strength", 0.0)
                        })
                        
                        # Boost memory strength (rehearsal effect)
                        memory["memory_strength"] = min(1.0, memory.get("memory_strength", 0.0) + self.rehearsal_boost)
                        
                        if len(retrieved_memories) >= limit:
                            break
        
        return retrieved_memories
    
    def _apply_memory_decay(self):
        """Apply decay to all memories."""
        # Decay episodic memories
        for memory in self.episodic_memories:
            memory["memory_strength"] = max(0.0, memory.get("memory_strength", 0.0) - self.decay_rate)
        
        # Decay semantic memories
        for concept, memory in self.semantic_memories.items():
            memory["memory_strength"] = max(0.0, memory.get("memory_strength", 0.0) - self.decay_rate)
        
        # Decay procedural memories - handle whether it's a list or a dictionary
        if isinstance(self.procedural_memories, dict):
            for action, memory in self.procedural_memories.items():
                memory["memory_strength"] = max(0.0, memory.get("memory_strength", 0.0) - self.decay_rate)
        elif isinstance(self.procedural_memories, list):
            for memory in self.procedural_memories:
                memory["memory_strength"] = max(0.0, memory.get("memory_strength", 0.0) - self.decay_rate)
    
    def _consolidate_memories(self):
        """Consolidate memories (strengthen important ones, remove weak ones)."""
        # Only consolidate if development is sufficient
        if random.random() > self.memory_consolidation_rate:
            return
        
        # Consolidate episodic memories
        self.episodic_memories = [
            memory for memory in self.episodic_memories
            if memory.get("memory_strength", 0.0) > 0.1 or memory.get("emotional_significance", 0.0) > 0.5
        ]
        
        # Rebuild episodic index if memories were removed
        self._rebuild_episodic_index()
        
        # Consolidate semantic memories
        self.semantic_memories = {
            concept: memory for concept, memory in self.semantic_memories.items()
            if memory.get("memory_strength", 0.0) > 0.1
        }
        
        # Consolidate procedural memories
        if isinstance(self.procedural_memories, dict):
            self.procedural_memories = {
                action: memory for action, memory in self.procedural_memories.items()
                if memory.get("memory_strength", 0.0) > 0.1
            }
        elif isinstance(self.procedural_memories, list):
            self.procedural_memories = [
                memory for memory in self.procedural_memories
                if memory.get("memory_strength", 0.0) > 0.1
            ]
    
    def _update_memory_development(self, age_months: float, developmental_stage: str):
        """Update memory development metrics based on age and developmental stage.
        
        Args:
            age_months: Float representing the age in months
            developmental_stage: String representing the developmental stage
        """
        # Working memory capacity
        if age_months <= 12:
            # First year: minimal working memory
            self.working_memory_capacity = 1
        elif age_months <= 36:
            # 1-3 years: gradual increase
            self.working_memory_capacity = 1 + int((age_months - 12) / 8)
        elif age_months <= 72:
            # 3-6 years: continued increase
            self.working_memory_capacity = 4 + int((age_months - 36) / 12)
        else:
            # After 6 years: adult-like capacity
            self.working_memory_capacity = 7
        
        # Long-term memory development
        if age_months <= 24:
            # First 2 years: beginning of long-term memory
            self.long_term_memory_development = min(0.3, 0.1 + age_months * 0.008)
        elif age_months <= 60:
            # 2-5 years: rapid development
            self.long_term_memory_development = min(0.7, 0.3 + (age_months - 24) * 0.01)
        else:
            # After 5 years: refinement
            self.long_term_memory_development = min(1.0, 0.7 + (age_months - 60) * 0.003)
        
        # Memory consolidation rate
        if age_months <= 36:
            # First 3 years: slow consolidation
            self.memory_consolidation_rate = min(0.3, 0.1 + age_months * 0.005)
        elif age_months <= 96:
            # 3-8 years: improving consolidation
            self.memory_consolidation_rate = min(0.7, 0.3 + (age_months - 36) * 0.005)
        else:
            # After 8 years: adult-like consolidation
            self.memory_consolidation_rate = min(1.0, 0.7 + (age_months - 96) * 0.002)
        
        # Memory retrieval accuracy
        if age_months <= 24:
            # First 2 years: poor retrieval
            self.memory_retrieval_accuracy = min(0.3, 0.1 + age_months * 0.008)
        elif age_months <= 60:
            # 2-5 years: improving retrieval
            self.memory_retrieval_accuracy = min(0.7, 0.3 + (age_months - 24) * 0.01)
        else:
            # After 5 years: adult-like retrieval
            self.memory_retrieval_accuracy = min(0.95, 0.7 + (age_months - 60) * 0.004)
        
        # Update decay rate based on development
        self.decay_rate = max(0.001, 0.01 - (self.memory_consolidation_rate * 0.009))
    
    def get_memory_counts(self) -> Dict[str, int]:
        """Get counts of different types of memories.
        
        Returns:
            Dictionary of memory counts
        """
        return {
            "episodic": len(self.episodic_memories),
            "semantic": len(self.semantic_memories),
            "procedural": len(self.procedural_memories),
            "working": len(self.working_memory)
        }
    
    def get_memory_development_metrics(self) -> Dict[str, float]:
        """Get the memory development metrics.
        
        Returns:
            Dictionary of memory development metrics
        """
        return {
            "working_memory_capacity": self.working_memory_capacity,
            "long_term_memory_development": self.long_term_memory_development,
            "memory_consolidation_rate": self.memory_consolidation_rate,
            "memory_retrieval_accuracy": self.memory_retrieval_accuracy
        }
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Call parent save method
        super().save(directory)
        
        # Save additional state
        additional_state = {
            "working_memory_capacity": self.working_memory_capacity,
            "long_term_memory_development": self.long_term_memory_development,
            "memory_consolidation_rate": self.memory_consolidation_rate,
            "memory_retrieval_accuracy": self.memory_retrieval_accuracy,
            "memory_capacity": self.memory_capacity,
            "episodic_memories": self.episodic_memories,
            "semantic_memories": self.semantic_memories,
            "procedural_memories": self.procedural_memories,
            "working_memory": list(self.working_memory),
            "decay_rate": self.decay_rate,
            "rehearsal_boost": self.rehearsal_boost,
            "emotional_significance_threshold": self.emotional_significance_threshold,
            "novelty_threshold": self.novelty_threshold,
            "embedding_dimension": self.embedding_dimension
        }
        
        with open(directory / f"{self.name}_additional_state.json", "w") as f:
            json.dump(additional_state, f)
        
        # Save FAISS index
        faiss.write_index(self.episodic_index, str(directory / f"{self.name}_episodic_index.faiss"))
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Call parent load method
        super().load(directory)
        
        # Load additional state
        additional_state_path = directory / f"{self.name}_additional_state.json"
        if additional_state_path.exists():
            with open(additional_state_path, "r") as f:
                additional_state = json.load(f)
                self.working_memory_capacity = additional_state["working_memory_capacity"]
                self.long_term_memory_development = additional_state["long_term_memory_development"]
                self.memory_consolidation_rate = additional_state["memory_consolidation_rate"]
                self.memory_retrieval_accuracy = additional_state["memory_retrieval_accuracy"]
                self.memory_capacity = additional_state["memory_capacity"]
                self.episodic_memories = additional_state["episodic_memories"]
                self.semantic_memories = additional_state["semantic_memories"]
                
                # Handle procedural memories, ensuring they're stored as a dictionary
                if "procedural_memories" in additional_state:
                    if isinstance(additional_state["procedural_memories"], list):
                        # Convert to dictionary if it's a list
                        proc_memories = {}
                        for memory in additional_state["procedural_memories"]:
                            if "action" in memory:
                                proc_memories[memory["action"]] = memory
                        self.procedural_memories = proc_memories
                    else:
                        # Use as is if it's already a dictionary
                        self.procedural_memories = additional_state["procedural_memories"]
                else:
                    # Initialize as empty dictionary if missing
                    self.procedural_memories = {}
                self.working_memory = deque(additional_state["working_memory"], maxlen=5)
                self.decay_rate = additional_state["decay_rate"]
                self.rehearsal_boost = additional_state["rehearsal_boost"]
                self.emotional_significance_threshold = additional_state["emotional_significance_threshold"]
                self.novelty_threshold = additional_state["novelty_threshold"]
                # Load embedding_dimension if it exists, otherwise keep the default
                if "embedding_dimension" in additional_state:
                    self.embedding_dimension = additional_state["embedding_dimension"]
        
        # Load FAISS index
        index_path = directory / f"{self.name}_episodic_index.faiss"
        try:
            if index_path.exists():
                self.episodic_index = faiss.read_index(str(index_path))
                
                # Verify index dimension matches expected dimension
                if self.episodic_index.d != self.embedding_dimension:
                    print(f"Warning: Loaded index dimension ({self.episodic_index.d}) doesn't match expected dimension ({self.embedding_dimension})")
                    # Reset the index to the correct dimension and rebuild
                    self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
                    # And rebuild it from memories
                    self._rebuild_episodic_index()
            else:
                # If no index file exists, initialize a new one and build it
                self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
                if self.episodic_memories:
                    self._rebuild_episodic_index()
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Initializing a new index.")
            # If there's an error loading the index, initialize a new one
            self.episodic_index = faiss.IndexFlatL2(self.embedding_dimension)
            # And rebuild it from memories if any exist
            if self.episodic_memories:
                self._rebuild_episodic_index() 