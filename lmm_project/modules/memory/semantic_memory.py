from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import uuid
import os
import json
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.memory.models import SemanticMemory, Memory
from lmm_project.utils.vector_store import VectorStore

class SemanticMemoryModule(BaseModule):
    """
    Semantic memory system for knowledge, concepts, and facts
    
    Semantic memory represents factual knowledge and concepts that the 
    mind has learned, independent of specific episodes where they were 
    acquired. This module handles storage, organization, and retrieval
    of knowledge using a hierarchical concept network.
    """
    # Concept storage
    concepts: Dict[str, SemanticMemory] = Field(default_factory=dict)
    # Vector store for semantic search
    vector_store: Optional[VectorStore] = None
    # Concept relationships (mapping concept_id -> list of related concept_ids with strength)
    relationships: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # Domains for organizing concepts
    domains: Dict[str, Set[str]] = Field(default_factory=dict)
    # Knowledge confidence threshold (concepts below this are uncertain)
    confidence_threshold: float = Field(default=0.6)
    # Storage directory
    storage_dir: str = Field(default="storage/memories/semantic")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize semantic memory module"""
        super().__init__(
            module_id=module_id,
            module_type="semantic_memory",
            event_bus=event_bus,
            **data
        )
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            dimension=768,
            storage_dir="storage/embeddings/semantic"
        )
        
        # Try to load previous concepts
        self._load_concepts()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("memory_stored", self._handle_memory_stored)
            self.subscribe_to_message("instruction_received", self._handle_instruction)
            self.subscribe_to_message("semantic_query", self._handle_semantic_query)
            self.subscribe_to_message("concept_update", self._handle_concept_update)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process semantic memory operations
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform (add_concept, get_concept, 
                         search_concepts, relate_concepts, etc.)
            - Additional parameters depend on the operation
            
        Returns:
        Dictionary containing operation results
        """
        operation = input_data.get("operation", "")
        
        if operation == "add_concept":
            concept_data = input_data.get("concept", {})
            return self.add_concept(concept_data)
        
        elif operation == "get_concept":
            concept_id = input_data.get("concept_id", "")
            concept_name = input_data.get("concept_name", "")
            if concept_id:
                return self.get_concept_by_id(concept_id)
            elif concept_name:
                return self.get_concept_by_name(concept_name)
            else:
                return {"status": "error", "message": "No concept ID or name provided"}
        
        elif operation == "search_concepts":
            query = input_data.get("query", "")
            return self.search_concepts(query)
        
        elif operation == "relate_concepts":
            source_id = input_data.get("source_id", "")
            target_id = input_data.get("target_id", "")
            strength = input_data.get("strength", 0.5)
            return self.relate_concepts(source_id, target_id, strength)
        
        elif operation == "get_domain_concepts":
            domain = input_data.get("domain", "")
            return self.get_domain_concepts(domain)
            
        return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update semantic memory's developmental level
        
        As semantic memory develops:
        - Concept formation becomes more nuanced and abstract
        - Relationship detection improves
        - Knowledge integration becomes more sophisticated
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development
        delta = self.development_level - prev_level
        
        # Improve confidence threshold
        confidence_decrease = delta * 0.05
        self.confidence_threshold = max(0.3, self.confidence_threshold - confidence_decrease)
        
        return self.development_level
    
    def add_concept(self, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new concept to semantic memory
        
        Parameters:
        concept_data: Dictionary containing concept data
            - concept: The concept name or label
            - content: Concept description
            - confidence: Optional confidence level (0.0-1.0)
            - domain: Optional knowledge domain
            - related_concepts: Optional dict of related concepts {concept_id: strength}
            
        Returns:
        Operation result
        """
        # Create concept ID if not provided
        if "id" not in concept_data:
            concept_data["id"] = str(uuid.uuid4())
            
        concept_id = concept_data["id"]
        
        # Ensure concept field is set
        if "concept" not in concept_data:
            if "content" in concept_data:
                concept_data["concept"] = concept_data["content"]
            else:
                return {"status": "error", "message": "No concept name provided"}
        
        # Create SemanticMemory object
        concept = SemanticMemory(**concept_data)
        
        # Store concept
        self.concepts[concept_id] = concept
        
        # Add to appropriate domain
        domain = concept_data.get("domain")
        if domain:
            if domain not in self.domains:
                self.domains[domain] = set()
            self.domains[domain].add(concept_id)
        
        # Add relationships if provided
        related_concepts = concept_data.get("related_concepts", {})
        for related_id, strength in related_concepts.items():
            self.relate_concepts(concept_id, related_id, strength)
        
        # Generate embedding if not provided
        if not concept.embedding:
            # Combine concept name and content for better embedding
            embedding_text = f"{concept.concept}: {concept.content}"
            concept.embedding = self._generate_embedding(embedding_text)
            
            # Add to vector store if embedding exists
            if concept.embedding:
                self.vector_store.add(
                    embeddings=[concept.embedding],
                    metadata_list=[{
                        "id": concept_id,
                        "concept": concept.concept,
                        "content": concept.content
                    }]
                )
        
        # Save to disk
        self._save_concept(concept)
        
        # Publish event
        self.publish_message("concept_added", {
            "concept_id": concept_id,
            "concept": concept.concept,
            "content": concept.content,
            "confidence": concept.confidence
        })
        
        return {
            "status": "success",
            "concept_id": concept_id
        }
    
    def get_concept_by_id(self, concept_id: str) -> Dict[str, Any]:
        """
        Get a concept by ID
        
        Parameters:
        concept_id: ID of the concept to retrieve
        
        Returns:
        Operation result containing concept data
        """
        # Check if concept exists
        if concept_id not in self.concepts:
            return {"status": "error", "message": f"Concept not found: {concept_id}"}
        
        concept = self.concepts[concept_id]
        
        # Update activation
        concept.update_activation(0.3)
        
        # Get related concepts
        related_concepts = {}
        if concept_id in self.relationships:
            related_concepts = self.relationships[concept_id]
        
        # Publish event
        self.publish_message("concept_retrieved", {
            "concept_id": concept_id,
            "concept": concept.concept
        })
        
        # Return concept data with related concepts
        return {
            "status": "success",
            "concept_id": concept_id,
            "concept": concept.concept,
            "content": concept.content,
            "confidence": concept.confidence,
            "domain": concept.domain,
            "related_concepts": related_concepts
        }
    
    def get_concept_by_name(self, concept_name: str) -> Dict[str, Any]:
        """
        Get a concept by name
        
        Parameters:
        concept_name: Name of the concept to retrieve
        
        Returns:
        Operation result containing concept data
        """
        # Search for concept by name
        for concept_id, concept in self.concepts.items():
            if concept.concept.lower() == concept_name.lower():
                return self.get_concept_by_id(concept_id)
        
        # If not found, try a partial match
        for concept_id, concept in self.concepts.items():
            if concept_name.lower() in concept.concept.lower():
                return self.get_concept_by_id(concept_id)
        
        # If still not found, try a semantic search
        search_results = self.search_concepts(concept_name)
        if search_results.get("status") == "success" and search_results.get("results"):
            # Return the top result
            top_result = search_results["results"][0]
            return self.get_concept_by_id(top_result["concept_id"])
        
        return {"status": "error", "message": f"Concept not found: {concept_name}"}
    
    def search_concepts(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for concepts by semantic similarity
        
        Parameters:
        query: Text query
        limit: Maximum number of results
        
        Returns:
        Operation result containing matching concepts
        """
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        if not query_embedding:
            return {"status": "error", "message": "Failed to generate embedding for query"}
        
        # Search vector store
        try:
            indices, distances, metadata = self.vector_store.search(
                query_embedding=query_embedding,
                k=limit
            )
            
            # Collect results
            results = []
            for idx, dist, meta in zip(indices, distances, metadata):
                concept_id = meta.get("id")
                if concept_id in self.concepts:
                    concept = self.concepts[concept_id]
                    # Update activation
                    concept.update_activation(0.2)
                    results.append({
                        "concept_id": concept_id,
                        "concept": concept.concept,
                        "content": concept.content,
                        "confidence": concept.confidence,
                        "similarity_score": 1.0 - min(1.0, float(dist))
                    })
            
            # Publish event
            self.publish_message("concept_search_results", {
                "query": query,
                "result_count": len(results)
            })
            
            return {
                "status": "success",
                "results": results,
                "query": query
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Search failed: {str(e)}"}
    
    def relate_concepts(self, source_id: str, target_id: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        Create or update a relationship between concepts
        
        Parameters:
        source_id: Source concept ID
        target_id: Target concept ID
        strength: Relationship strength (0.0-1.0)
        
        Returns:
        Operation result
        """
        # Check if concepts exist
        if source_id not in self.concepts:
            return {"status": "error", "message": f"Source concept not found: {source_id}"}
        
        if target_id not in self.concepts:
            return {"status": "error", "message": f"Target concept not found: {target_id}"}
        
        # Initialize relationship dictionaries if needed
        if source_id not in self.relationships:
            self.relationships[source_id] = {}
        
        if target_id not in self.relationships:
            self.relationships[target_id] = {}
        
        # Create bidirectional relationship
        self.relationships[source_id][target_id] = strength
        self.relationships[target_id][source_id] = strength
        
        # Update related_concepts field in concepts
        source_concept = self.concepts[source_id]
        target_concept = self.concepts[target_id]
        
        source_concept.related_concepts[target_id] = strength
        target_concept.related_concepts[source_id] = strength
        
        # Save concepts
        self._save_concept(source_concept)
        self._save_concept(target_concept)
        
        # Save relationships
        self._save_relationships()
        
        # Publish event
        self.publish_message("concepts_related", {
            "source_id": source_id,
            "target_id": target_id,
            "source_concept": source_concept.concept,
            "target_concept": target_concept.concept,
            "strength": strength
        })
        
        return {
            "status": "success",
            "source_id": source_id,
            "target_id": target_id,
            "strength": strength
        }
    
    def get_domain_concepts(self, domain: str) -> Dict[str, Any]:
        """
        Get all concepts in a domain
        
        Parameters:
        domain: Domain name
        
        Returns:
        Operation result containing concepts in the domain
        """
        if domain not in self.domains:
            return {"status": "error", "message": f"Domain not found: {domain}", "concepts": []}
        
        domain_concept_ids = self.domains[domain]
        domain_concepts = []
        
        for concept_id in domain_concept_ids:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                domain_concepts.append({
                    "concept_id": concept_id,
                    "concept": concept.concept,
                    "content": concept.content,
                    "confidence": concept.confidence
                })
        
        return {
            "status": "success",
            "domain": domain,
            "concepts": domain_concepts,
            "count": len(domain_concepts)
        }
    
    def update_concept_confidence(
        self, 
        concept_id: str, 
        confidence_delta: float
    ) -> Dict[str, Any]:
        """
        Update confidence in a concept
        
        Parameters:
        concept_id: ID of the concept to update
        confidence_delta: Change in confidence (-1.0 to 1.0)
        
        Returns:
        Operation result
        """
        if concept_id not in self.concepts:
            return {"status": "error", "message": f"Concept not found: {concept_id}"}
        
        concept = self.concepts[concept_id]
        old_confidence = concept.confidence
        
        # Update confidence
        concept.confidence = max(0.0, min(1.0, concept.confidence + confidence_delta))
        
        # Save concept
        self._save_concept(concept)
        
        # Publish event
        self.publish_message("concept_confidence_updated", {
            "concept_id": concept_id,
            "concept": concept.concept,
            "old_confidence": old_confidence,
            "new_confidence": concept.confidence,
            "delta": confidence_delta
        })
        
        return {
            "status": "success",
            "concept_id": concept_id,
            "old_confidence": old_confidence,
            "new_confidence": concept.confidence
        }
    
    def count_concepts(self) -> int:
        """Count the number of stored concepts"""
        return len(self.concepts)
    
    def save_state(self) -> str:
        """
        Save the current state of semantic memory
        
        Returns:
        Path to saved state directory
        """
        # Save concepts
        for concept_id, concept in self.concepts.items():
            self._save_concept(concept)
        
        # Save relationships
        self._save_relationships()
        
        # Save domains
        self._save_domains()
        
        # Save vector store
        self.vector_store.save()
        
        return self.storage_dir
    
    def _save_concept(self, concept: SemanticMemory) -> None:
        """Save a single concept to disk"""
        try:
            concepts_dir = Path(self.storage_dir) / "concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            
            concept_path = concepts_dir / f"{concept.id}.json"
            with open(concept_path, "w") as f:
                # We need to convert the concept to a dict and handle datetime objects
                concept_dict = concept.model_dump()
                # Convert datetime to string
                for key, value in concept_dict.items():
                    if isinstance(value, datetime):
                        concept_dict[key] = value.isoformat()
                json.dump(concept_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving concept {concept.id}: {e}")
    
    def _save_relationships(self) -> None:
        """Save concept relationships to disk"""
        try:
            relationship_path = Path(self.storage_dir) / "relationships.json"
            with open(relationship_path, "w") as f:
                json.dump(self.relationships, f, indent=2)
        except Exception as e:
            print(f"Error saving relationships: {e}")
    
    def _save_domains(self) -> None:
        """Save concept domains to disk"""
        try:
            # Convert sets to lists for JSON serialization
            domains_dict = {domain: list(concepts) for domain, concepts in self.domains.items()}
            
            domains_path = Path(self.storage_dir) / "domains.json"
            with open(domains_path, "w") as f:
                json.dump(domains_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving domains: {e}")
    
    def _load_concepts(self) -> None:
        """Load concepts from disk"""
        try:
            # Load concepts
            concepts_dir = Path(self.storage_dir) / "concepts"
            concepts_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in concepts_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        concept_data = json.load(f)
                        # Convert string back to datetime
                        if "timestamp" in concept_data and isinstance(concept_data["timestamp"], str):
                            concept_data["timestamp"] = datetime.fromisoformat(concept_data["timestamp"])
                        if "last_accessed" in concept_data and concept_data["last_accessed"] and isinstance(concept_data["last_accessed"], str):
                            concept_data["last_accessed"] = datetime.fromisoformat(concept_data["last_accessed"])
                        
                        # Create concept object
                        concept = SemanticMemory(**concept_data)
                        self.concepts[concept.id] = concept
                        
                        # Add to vector store if embedding exists
                        if concept.embedding:
                            self.vector_store.add(
                                embeddings=[concept.embedding],
                                metadata_list=[{
                                    "id": concept.id,
                                    "concept": concept.concept,
                                    "content": concept.content
                                }]
                            )
                except Exception as e:
                    print(f"Error loading concept from {file_path}: {e}")
            
            # Load relationships
            relationship_path = Path(self.storage_dir) / "relationships.json"
            if relationship_path.exists():
                with open(relationship_path, "r") as f:
                    self.relationships = json.load(f)
            
            # Load domains
            domains_path = Path(self.storage_dir) / "domains.json"
            if domains_path.exists():
                with open(domains_path, "r") as f:
                    domains_dict = json.load(f)
                    # Convert lists back to sets
                    self.domains = {domain: set(concepts) for domain, concepts in domains_dict.items()}
            
            print(f"Loaded {len(self.concepts)} concepts from disk")
        except Exception as e:
            print(f"Error loading concepts: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for text"""
        try:
            from lmm_project.utils.llm_client import LLMClient
            client = LLMClient()
            embedding = client.get_embedding(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    # Event handlers
    
    def _handle_memory_stored(self, message: Message) -> None:
        """
        Handle memory stored events
        
        When memories are stored in long-term memory, we check if they
        might represent knowledge that should be added to semantic memory.
        """
        content = message.content
        memory_id = content.get("memory_id")
        memory_content = content.get("content")
        
        if not memory_content or not memory_id:
            return
            
        # Check if this memory might contain factual knowledge
        # This is a simplistic approach - in a real system, you'd use 
        # NLP to detect factual statements
        if (
            "is" in memory_content.lower() or 
            "are" in memory_content.lower() or
            "means" in memory_content.lower() or
            "defined as" in memory_content.lower()
        ):
            # This might be a fact - extract it
            self._extract_concept_from_memory(memory_id, memory_content)
    
    def _handle_instruction(self, message: Message) -> None:
        """
        Handle instruction events
        
        The mother might directly teach facts or concepts.
        """
        content = message.content
        instruction = content.get("instruction", "")
        
        if not instruction:
            return
            
        # Check if this instruction contains knowledge
        # This is a simplistic approach
        if (
            "is" in instruction.lower() or 
            "are" in instruction.lower() or
            "means" in instruction.lower() or
            "defined as" in instruction.lower()
        ):
            # This might be a fact - extract it
            self._extract_concept_from_instruction(instruction)
    
    def _handle_semantic_query(self, message: Message) -> None:
        """Handle semantic query events"""
        content = message.content
        query = content.get("query", "")
        
        if not query:
            return
            
        results = self.search_concepts(query)
        
        if self.event_bus and results.get("status") == "success":
            # Publish results
            self.publish_message("semantic_query_response", {
                "requester": message.sender,
                "results": results.get("results", []),
                "query": query
            })
    
    def _handle_concept_update(self, message: Message) -> None:
        """Handle concept update events"""
        content = message.content
        concept_id = content.get("concept_id")
        confidence_delta = content.get("confidence_delta")
        
        if concept_id and confidence_delta is not None:
            self.update_concept_confidence(concept_id, confidence_delta)
    
    def _extract_concept_from_memory(self, memory_id: str, memory_content: str) -> None:
        """
        Extract potential concepts from a memory
        
        This is a simplified implementation - in a real system, you'd use 
        more sophisticated NLP to extract concepts and relationships.
        """
        # Simple heuristic: look for "X is Y" patterns
        content_lower = memory_content.lower()
        
        # Check for "X is Y" pattern
        if " is " in content_lower:
            parts = memory_content.split(" is ", 1)
            if len(parts) == 2:
                concept_name = parts[0].strip()
                concept_description = parts[1].strip()
                
                # Create concept data
                concept_data = {
                    "concept": concept_name,
                    "content": f"{concept_name} is {concept_description}",
                    "confidence": 0.7,  # Moderate confidence in extracted concepts
                    "source_type": "experience"
                }
                
                # Add the concept
                self.add_concept(concept_data)
    
    def _extract_concept_from_instruction(self, instruction: str) -> None:
        """
        Extract concepts from direct instruction
        
        This is simplified - real implementation would use more sophisticated NLP.
        """
        # Simple heuristic: look for "X is Y" patterns
        instruction_lower = instruction.lower()
        
        # Check for "X is Y" pattern
        if " is " in instruction_lower:
            parts = instruction.split(" is ", 1)
            if len(parts) == 2:
                concept_name = parts[0].strip()
                concept_description = parts[1].strip()
                
                # Create concept data
                concept_data = {
                    "concept": concept_name,
                    "content": f"{concept_name} is {concept_description}",
                    "confidence": 0.9,  # Higher confidence for direct instruction
                    "source_type": "instruction"
                }
                
                # Add the concept
                self.add_concept(concept_data) 