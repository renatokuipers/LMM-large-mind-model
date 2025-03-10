# TODO: Implement the SelfConcept class to maintain beliefs about the self
# This component should be able to:
# - Represent knowledge and beliefs about the self
# - Organize self-knowledge into domains (abilities, traits, etc.)
# - Update self-concept based on experiences and feedback
# - Maintain consistency in self-representation

# TODO: Implement developmental progression in self-concept:
# - Simple categorical self-recognition in early stages
# - Concrete trait descriptions in childhood
# - Social comparison and ideal self in adolescence
# - Complex, nuanced self-understanding in adulthood

# TODO: Create mechanisms for:
# - Self-schema formation: Organize self-knowledge by domain
# - Self-evaluation: Assess self-attributes against standards
# - Identity integration: Maintain coherence across domains
# - Self-verification: Seek confirmation of existing self-views

# TODO: Implement different self-concept domains:
# - Ability domain: Beliefs about capabilities and skills
# - Social domain: Representations of social roles and identities
# - Physical domain: Beliefs about physical attributes
# - Psychological domain: Understanding of internal states and traits

# TODO: Connect to memory and social systems
# Self-concept should draw on autobiographical memory
# and incorporate social feedback and comparisons

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

import numpy as np
import torch
from collections import deque
from pydantic import ValidationError

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
from lmm_project.modules.identity.models import SelfAttribute, SelfConcept as SelfConceptModel
from lmm_project.modules.identity.neural_net import SelfConceptNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class SelfConcept(BaseModule):
    """
    Represents beliefs and knowledge about oneself
    
    This module maintains and updates beliefs about the self in various domains,
    integrating new information into a coherent self-concept.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic self-recognition",
        0.2: "Simple categorical self-descriptions",
        0.4: "Concrete trait descriptions across domains",
        0.6: "Social comparison and ideal self-development",
        0.8: "Integration of self across domains and contexts",
        1.0: "Complex, nuanced self-understanding with temporal stability"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the self-concept module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize self-concept
        self.self_concept = SelfConceptModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = SelfConceptNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize attribute embeddings for similarity search
        self.attribute_embeddings = {}
        
        # Initialize with basic domains based on development level
        self._adjust_domains_for_development()
    
    def _adjust_domains_for_development(self):
        """Adjust available self-concept domains based on development level"""
        # Initialize domain structure
        domains = []
        
        # Basic domains available at initial development
        if self.development_level >= 0.0:
            domains.extend([
                "physical",  # Basic physical characteristics
                "preferences"  # Simple likes and dislikes
            ])
            
        # Additional domains that become available with development
        if self.development_level >= 0.2:
            domains.extend([
                "abilities",  # What I can and cannot do
                "social"  # Relationships with others
            ])
            
        if self.development_level >= 0.4:
            domains.extend([
                "emotions",  # Emotional tendencies
                "personality",  # Trait-based descriptions
                "academic"  # Knowledge and learning abilities
            ])
            
        if self.development_level >= 0.6:
            domains.extend([
                "values",  # Personal values and ethics
                "goals",  # Future-oriented aspirations
                "competence"  # Areas of mastery
            ])
            
        if self.development_level >= 0.8:
            domains.extend([
                "ideal_self",  # Who I want to be
                "moral",  # Moral character
                "belief_systems",  # Worldview and beliefs
                "life_narrative"  # Self as a story
            ])
        
        # Initialize domains in self-concept if not already present
        for domain in domains:
            if domain not in self.self_concept.domains:
                self.self_concept.domains[domain] = []
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the self-concept module
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dict with processing results
        """
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "status": "error",
                "message": "Input must be a dictionary"
            }
        
        # Extract process ID if provided
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract operation
        operation = input_data.get("operation", "")
        
        # Dispatch to appropriate handler
        if operation == "add_attribute":
            return self._add_attribute(input_data, process_id)
        elif operation == "update_attribute":
            return self._update_attribute(input_data, process_id)
        elif operation == "query_self":
            return self._query_self(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _add_attribute(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Add a new self-attribute
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check required fields
        required_fields = ["domain", "content"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "status": "error",
                    "message": f"Missing required field: {field}",
                    "process_id": process_id
                }
        
        # Get domain
        domain = input_data["domain"]
        
        # Check if this domain is available at current development level
        if domain not in self.self_concept.domains:
            # If domain isn't recognized, check development level
            self._adjust_domains_for_development()
            
            # If still not available, return error
            if domain not in self.self_concept.domains:
                return {
                    "status": "error",
                    "message": f"Domain '{domain}' not available at current development level",
                    "process_id": process_id
                }
        
        try:
            # Create attribute
            attribute_id = str(uuid.uuid4())
            
            # Get optional fields with defaults
            confidence = input_data.get("confidence", 0.5)
            importance = input_data.get("importance", 0.5)
            valence = input_data.get("valence", 0.0)
            evidence = input_data.get("evidence", [])
            sources = input_data.get("sources", [])
            
            # Create attribute
            attribute = SelfAttribute(
                attribute_id=attribute_id,
                domain=domain,
                content=input_data["content"],
                confidence=confidence,
                importance=importance,
                valence=valence,
                evidence=evidence,
                sources=sources
            )
            
            # Add to self-concept
            self.self_concept.add_attribute(attribute)
            
            # Create embedding for this attribute
            features = self._extract_features(f"{domain}: {attribute.content}")
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    domain=domain
                )
                self.attribute_embeddings[attribute_id] = output["attribute_encoding"].cpu().squeeze(0)
            
            # Update self-esteem if valence is provided
            self._update_self_esteem()
            
            return {
                "status": "success",
                "message": "Attribute added successfully",
                "attribute_id": attribute_id,
                "process_id": process_id
            }
            
        except ValidationError as e:
            return {
                "status": "error",
                "message": f"Validation error: {str(e)}",
                "process_id": process_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding attribute: {str(e)}",
                "process_id": process_id
            }
    
    def _update_attribute(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Update an existing self-attribute
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check for attribute ID
        if "attribute_id" not in input_data:
            return {
                "status": "error",
                "message": "Missing attribute_id",
                "process_id": process_id
            }
            
        attribute_id = input_data["attribute_id"]
        
        # Check if attribute exists
        if attribute_id not in self.self_concept.attributes:
            return {
                "status": "error",
                "message": f"Attribute not found: {attribute_id}",
                "process_id": process_id
            }
            
        # Get attribute
        attribute = self.self_concept.attributes[attribute_id]
        
        # Track if updated
        updated = False
        
        # Update fields
        if "content" in input_data:
            attribute.content = input_data["content"]
            updated = True
            
        if "domain" in input_data:
            new_domain = input_data["domain"]
            
            # Check if domain is available
            if new_domain not in self.self_concept.domains:
                self._adjust_domains_for_development()
                
                if new_domain not in self.self_concept.domains:
                    return {
                        "status": "error",
                        "message": f"Domain '{new_domain}' not available at current development level",
                        "process_id": process_id
                    }
            
            # Update domain
            old_domain = attribute.domain
            attribute.domain = new_domain
            
            # Update domain lists
            if attribute_id in self.self_concept.domains[old_domain]:
                self.self_concept.domains[old_domain].remove(attribute_id)
            
            if attribute_id not in self.self_concept.domains[new_domain]:
                self.self_concept.domains[new_domain].append(attribute_id)
                
            updated = True
            
        if "confidence" in input_data:
            attribute.confidence = max(0.0, min(1.0, float(input_data["confidence"])))
            updated = True
            
        if "importance" in input_data:
            attribute.importance = max(0.0, min(1.0, float(input_data["importance"])))
            updated = True
            
        if "valence" in input_data:
            attribute.valence = max(-1.0, min(1.0, float(input_data["valence"])))
            updated = True
            
        if "evidence" in input_data:
            if isinstance(input_data["evidence"], list):
                attribute.evidence = input_data["evidence"]
                updated = True
                
        if "sources" in input_data:
            if isinstance(input_data["sources"], list):
                attribute.sources = input_data["sources"]
                updated = True
        
        if updated:
            # Update timestamp
            attribute.updated_at = datetime.now()
            
            # Update in self-concept
            self.self_concept.attributes[attribute_id] = attribute
            self.self_concept.last_updated = datetime.now()
            
            # Update embedding if content or domain changed
            if "content" in input_data or "domain" in input_data:
                features = self._extract_features(f"{attribute.domain}: {attribute.content}")
                with torch.no_grad():
                    output = self.network(
                        features.to(self.device),
                        domain=attribute.domain
                    )
                    self.attribute_embeddings[attribute_id] = output["attribute_encoding"].cpu().squeeze(0)
            
            # Update self-esteem if valence is updated
            if "valence" in input_data:
                self._update_self_esteem()
            
            return {
                "status": "success",
                "message": "Attribute updated successfully",
                "process_id": process_id
            }
        else:
            return {
                "status": "error",
                "message": "No fields updated",
                "process_id": process_id
            }
    
    def _query_self(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query information about the self-concept
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return entire self-concept summary
            return {
                "status": "success",
                "domains": list(self.self_concept.domains.keys()),
                "attribute_count": len(self.self_concept.attributes),
                "global_self_esteem": self.self_concept.global_self_esteem,
                "clarity": self.self_concept.clarity,
                "stability": self.self_concept.stability,
                "complexity": self.self_concept.complexity,
                "process_id": process_id
            }
        
        elif query_type == "domain":
            # Get domain-specific attributes
            if "domain" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing domain for domain query",
                    "process_id": process_id
                }
                
            domain = input_data["domain"]
            
            if domain not in self.self_concept.domains:
                return {
                    "status": "error",
                    "message": f"Domain not found: {domain}",
                    "process_id": process_id
                }
                
            # Get attributes in this domain
            attributes = []
            for attr_id in self.self_concept.domains[domain]:
                if attr_id in self.self_concept.attributes:
                    attr = self.self_concept.attributes[attr_id]
                    attributes.append({
                        "attribute_id": attr_id,
                        "content": attr.content,
                        "confidence": attr.confidence,
                        "importance": attr.importance,
                        "valence": attr.valence
                    })
                    
            return {
                "status": "success",
                "domain": domain,
                "attributes": attributes,
                "process_id": process_id
            }
        
        elif query_type == "similarity":
            # Find attributes similar to a query
            if "query" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing query for similarity search",
                    "process_id": process_id
                }
                
            query = input_data["query"]
            domain = input_data.get("domain")  # Optional domain filter
            
            # Extract features from query
            features = self._extract_features(query)
            
            # Get encoding from network
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    domain=domain
                )
                query_encoding = output["attribute_encoding"].cpu().squeeze(0)
            
            # Find similar attributes
            similar_attributes = []
            
            if len(self.attribute_embeddings) > 0:
                # Calculate similarity to existing attributes
                similarities = {}
                for attr_id, embedding in self.attribute_embeddings.items():
                    # Skip if domain filter is applied and doesn't match
                    if domain and self.self_concept.attributes[attr_id].domain != domain:
                        continue
                        
                    similarity = torch.cosine_similarity(
                        query_encoding.unsqueeze(0),
                        embedding.unsqueeze(0)
                    ).item()
                    similarities[attr_id] = similarity
                    
                # Get top 5 similar attributes
                sorted_attrs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                for attr_id, similarity in sorted_attrs[:5]:
                    if similarity > 0.6:  # Only include reasonably similar results
                        attr = self.self_concept.attributes[attr_id]
                        similar_attributes.append({
                            "attribute_id": attr_id,
                            "domain": attr.domain,
                            "content": attr.content,
                            "confidence": attr.confidence,
                            "importance": attr.importance,
                            "similarity": similarity
                        })
                        
            return {
                "status": "success",
                "similar_attributes": similar_attributes,
                "process_id": process_id
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _update_self_esteem(self):
        """
        Update the global self-esteem based on attribute valence and importance
        """
        if not self.self_concept.attributes:
            # No attributes yet, keep default
            return
        
        # Calculate weighted average of valence, weighted by importance
        total_importance = 0.0
        weighted_valence_sum = 0.0
        
        for attr_id, attr in self.self_concept.attributes.items():
            # Convert valence from [-1, 1] to [0, 1] for self-esteem
            valence_positive = (attr.valence + 1) / 2
            weighted_valence_sum += valence_positive * attr.importance
            total_importance += attr.importance
        
        if total_importance > 0:
            # Calculate weighted average
            self.self_concept.global_self_esteem = weighted_valence_sum / total_importance
        
        # Also update clarity, stability and complexity based on development
        # These become more pronounced with development
        self.self_concept.clarity = 0.3 + (self.development_level * 0.7)
        self.self_concept.stability = 0.2 + (self.development_level * 0.8)
        self.self_concept.complexity = 0.1 + (self.development_level * 0.9)
        
        # Adjust for number of domains and attributes
        if len(self.self_concept.domains) > 0:
            domain_factor = min(1.0, len(self.self_concept.domains) / 10)
            self.self_concept.complexity = (self.self_concept.complexity + domain_factor) / 2
    
    def _extract_features(self, data) -> torch.Tensor:
        """
        Extract features from data using the neural network
        
        Args:
            data: Data to extract features from
            
        Returns:
            Tensor of features
        """
        # Simple feature extraction
        text = str(data)
        words = text.split()
        
        # Create a simple word embedding
        embedding = torch.zeros(min(len(words), 128), dtype=torch.float32)
        
        for i, word in enumerate(words[:embedding.size(0)]):
            # Simple hash-based embedding
            hash_val = hash(word) % 10000
            embedding[i] = (hash_val / 10000) * 2 - 1
            
        # Pad if needed
        if embedding.size(0) < 128:
            padding = torch.zeros(128 - embedding.size(0), dtype=torch.float32)
            embedding = torch.cat([embedding, padding])
            
        return embedding.unsqueeze(0)  # Add batch dimension
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        
        # Update development level
        self.development_level = max(0.0, min(1.0, self.development_level + amount))
        
        # Update neural network
        self.network.set_development_level(self.development_level)
        
        # Check if crossed a milestone
        for level in sorted(self.development_milestones.keys()):
            if old_level < level <= self.development_level:
                milestone = self.development_milestones[level]
                
                # Publish milestone event if we have an event bus
                if self.event_bus:
                    self.event_bus.publish({
                        "sender": self.module_id,
                        "message_type": "development_milestone",
                        "content": {
                            "module": "self_concept",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Log milestone
                print(f"Self-Concept Development Milestone: {milestone} (level {level})")
                
                # Update domains for new development level
                self._adjust_domains_for_development()
                
        # Update self-esteem and other metrics with new development level
        self._update_self_esteem()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the self-concept module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "self_concept": self.self_concept.dict(),
            "developmental_level": self.development_level,
            "attribute_count": len(self.self_concept.attributes),
            "domain_count": len(self.self_concept.domains)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        # Convert embeddings to lists for serialization
        attribute_embeddings_serialized = {}
        for attr_id, embedding in self.attribute_embeddings.items():
            attribute_embeddings_serialized[attr_id] = embedding.numpy().tolist()
            
        return {
            "module_id": self.module_id,
            "self_concept": self.self_concept.dict(),
            "developmental_level": self.development_level,
            "attribute_embeddings": attribute_embeddings_serialized
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state to load
        """
        # Load basic state
        self.module_id = state["module_id"]
        self.development_level = state["developmental_level"]
        
        # Update network development level
        self.network.set_development_level(self.development_level)
        
        # Load self-concept
        if "self_concept" in state:
            try:
                concept_data = state["self_concept"]
                self.self_concept = SelfConceptModel(**concept_data)
            except Exception as e:
                print(f"Error loading self-concept: {e}")
        
        # Load embeddings
        if "attribute_embeddings" in state:
            self.attribute_embeddings = {}
            for attr_id, embedding_list in state["attribute_embeddings"].items():
                self.attribute_embeddings[attr_id] = torch.tensor(embedding_list, dtype=torch.float32)
