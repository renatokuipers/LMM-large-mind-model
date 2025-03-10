# TODO: Implement the SemanticProcessing class to extract meaning from language
# This component should be able to:
# - Understand the meaning of words in context
# - Extract relationships between concepts in language
# - Interpret literal and non-literal language
# - Build semantic representations of sentences and discourse

# TODO: Implement developmental progression in semantic processing:
# - Simple direct meanings in early stages
# - Growing comprehension of relationships in childhood
# - Basic figurative language in later childhood
# - Complex abstractions and nuance in adolescence/adulthood

# TODO: Create mechanisms for:
# - Semantic composition: Combine word meanings into phrase meanings
# - Contextual interpretation: Adjust meanings based on context
# - Reference resolution: Determine what pronouns and references point to
# - Implication extraction: Infer unstated meanings and entailments

# TODO: Implement different semantic phenomena:
# - Polysemy: Multiple related meanings of words
# - Metaphor and simile: Figurative comparisons
# - Pragmatics: Social and contextual aspects of meaning
# - Entailment: Logical relationships between statements

# TODO: Connect to conceptual knowledge and memory
# Semantic processing should leverage conceptual knowledge
# and store extracted meanings in memory

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
from lmm_project.modules.language.models import SemanticModel, LanguageNeuralState
from lmm_project.modules.language.neural_net import SemanticNetwork, get_device
from lmm_project.utils.llm_client import LLMClient

class SemanticProcessing(BaseModule):
    """
    Processes and extracts meaning from language
    
    This module is responsible for understanding the semantic content
    of language, including concepts, relationships, and contextual meaning.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic concept recognition",
        0.2: "Simple word meanings",
        0.4: "Semantic categorization",
        0.6: "Relational understanding",
        0.8: "Contextual meaning",
        1.0: "Abstract semantics"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the semantic processing module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize semantic model
        self.semantic_model = SemanticModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = SemanticNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.semantic_processing_development = self.development_level
        
        # Initialize with basic semantics based on development level
        self._initialize_semantic_concepts()
        
        # Recent inputs queue (for tracking recent semantic processing)
        self.recent_inputs = deque(maxlen=100)
        
        # For embedding generation when needed
        self.llm_client = LLMClient()
    
    def _initialize_semantic_concepts(self):
        """Initialize basic semantic concepts based on development level"""
        # Basic concepts at earliest stages
        basic_concepts = {
            "mama": {
                "category": "people",
                "features": {"person": True, "female": True, "caregiver": True}
            },
            "dada": {
                "category": "people",
                "features": {"person": True, "male": True, "caregiver": True}
            },
            "milk": {
                "category": "food",
                "features": {"liquid": True, "white": True, "drink": True}
            },
            "dog": {
                "category": "animals",
                "features": {"animal": True, "furry": True, "bark": True}
            }
        }
        
        # Add concepts with confidence based on development level
        for concept, info in basic_concepts.items():
            # Only add if not already present
            if concept not in self.semantic_model.concept_network:
                self.semantic_model.concept_network[concept] = {
                    "concept_id": str(uuid.uuid4()),
                    "category": info["category"],
                    "features": info["features"],
                    "related_concepts": [],
                    "confidence": 0.7 * max(0.3, self.development_level)
                }
                
                # Add to semantic categories
                if info["category"] in self.semantic_model.semantic_categories:
                    if concept not in self.semantic_model.semantic_categories[info["category"]]:
                        self.semantic_model.semantic_categories[info["category"]].append(concept)
                else:
                    self.semantic_model.semantic_categories[info["category"]] = [concept]
                
                # Initialize concept embedding if development level is sufficient
                if self.development_level >= 0.3:
                    self._generate_concept_embedding(concept)
        
        # Add more complex concepts with increased development
        if self.development_level >= 0.4:
            # More advanced concepts
            advanced_concepts = {
                "happy": {
                    "category": "emotions",
                    "features": {"feeling": True, "positive": True}
                },
                "sad": {
                    "category": "emotions",
                    "features": {"feeling": True, "negative": True}
                },
                "big": {
                    "category": "properties",
                    "features": {"size": True, "large": True}
                },
                "small": {
                    "category": "properties",
                    "features": {"size": True, "tiny": True}
                }
            }
            
            # Add advanced concepts
            for concept, info in advanced_concepts.items():
                if concept not in self.semantic_model.concept_network:
                    self.semantic_model.concept_network[concept] = {
                        "concept_id": str(uuid.uuid4()),
                        "category": info["category"],
                        "features": info["features"],
                        "related_concepts": [],
                        "confidence": 0.6 * ((self.development_level - 0.4) / 0.6)
                    }
                    
                    # Add to semantic categories
                    if info["category"] in self.semantic_model.semantic_categories:
                        if concept not in self.semantic_model.semantic_categories[info["category"]]:
                            self.semantic_model.semantic_categories[info["category"]].append(concept)
                    else:
                        self.semantic_model.semantic_categories[info["category"]] = [concept]
                    
                    # Generate concept embedding
                    self._generate_concept_embedding(concept)
        
        # Add contextual meanings with higher development
        if self.development_level >= 0.6:
            # Simple contextual meanings
            contexts = {
                "hot": {
                    "food": {"meaning": "high temperature", "features": {"temperature": "high"}},
                    "weather": {"meaning": "high outdoor temperature", "features": {"outdoor": True, "temperature": "high"}}
                },
                "good": {
                    "food": {"meaning": "tasty", "features": {"taste": "pleasant"}},
                    "behavior": {"meaning": "well-behaved", "features": {"obedient": True, "pleasant": True}}
                }
            }
            
            # Add contextual meanings
            for word, context_info in contexts.items():
                if word not in self.semantic_model.contextual_meanings:
                    self.semantic_model.contextual_meanings[word] = {}
                    
                for context, meaning in context_info.items():
                    self.semantic_model.contextual_meanings[word][context] = {
                        "meaning": meaning["meaning"],
                        "features": meaning["features"],
                        "confidence": 0.6 * ((self.development_level - 0.6) / 0.4)
                    }
    
    def _generate_concept_embedding(self, concept: str):
        """
        Generate and store embedding for a concept
        
        Args:
            concept: The concept to generate embedding for
        """
        try:
            # Use LLM client to get embedding
            embedding = self.llm_client.get_embedding(concept)
            
            # Handle different return formats
            if isinstance(embedding, list):
                if isinstance(embedding[0], list):
                    # Handle nested list output
                    self.semantic_model.concept_embeddings[concept] = embedding[0]
                else:
                    # Handle flat list output
                    self.semantic_model.concept_embeddings[concept] = embedding
        except Exception as e:
            # Fall back to simplified embedding if LLM client fails
            print(f"Warning: Failed to get embedding for '{concept}': {e}")
            
            # Create a simple hash-based embedding
            self.semantic_model.concept_embeddings[concept] = [(hash(concept + str(i)) % 10000) / 10000 for i in range(64)]
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the semantic processing module
        
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
        operation = input_data.get("operation", "understand")
        
        # Dispatch to appropriate handler
        if operation == "understand":
            return self._understand_meaning(input_data, process_id)
        elif operation == "learn_concept":
            return self._learn_concept(input_data, process_id)
        elif operation == "relate_concepts":
            return self._relate_concepts(input_data, process_id)
        elif operation == "query_semantics":
            return self._query_semantics(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _understand_meaning(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Understand the meaning of a word or phrase
        
        Args:
            input_data: Input data dictionary including text to understand
            process_id: Process identifier
            
        Returns:
            Dict with understanding results
        """
        # Check for text to understand
        if "text" not in input_data:
            return {
                "status": "error",
                "message": "Missing text for semantic understanding",
                "process_id": process_id
            }
        
        text = input_data["text"]
        context = input_data.get("context")
        
        # Extract features from text
        text_features = self._extract_features(text)
        text_features = text_features.to(self.device)
        
        # Process through network
        with torch.no_grad():
            if context:
                # Process with context
                context_features = self._extract_features(context)
                context_features = context_features.to(self.device)
                
                output = self.network(
                    text_features, 
                    operation="contextualize",
                    context=context_features
                )
                
                understanding_depth = output["context_effect"].cpu().item()
                text_embedding = output["contextualized_embedding"].cpu().squeeze(0)
            else:
                # Process without context
                output = self.network(text_features, operation="understand")
                understanding_depth = output["understanding_depth"].cpu().item()
                text_embedding = output["semantic_embedding"].cpu().squeeze(0) if "semantic_embedding" in output else None
        
        # Identify relevant concepts
        relevant_concepts = []
        
        # Simple word matching for concepts
        words = text.lower().split()
        for word in words:
            if word in self.semantic_model.concept_network:
                relevant_concepts.append({
                    "concept": word,
                    "category": self.semantic_model.concept_network[word]["category"],
                    "confidence": self.semantic_model.concept_network[word]["confidence"]
                })
        
        # If we have embeddings, find similar concepts by embedding similarity
        if text_embedding is not None and self.development_level >= 0.5:
            similarities = {}
            
            for concept, embedding in self.semantic_model.concept_embeddings.items():
                if concept in [c["concept"] for c in relevant_concepts]:
                    continue  # Skip already matched concepts
                
                # Convert to tensor
                concept_tensor = torch.tensor(embedding, dtype=torch.float32)
                
                # Ensure tensors have same dimensions
                if text_embedding.shape != concept_tensor.shape:
                    continue
                
                # Calculate similarity
                similarity = torch.cosine_similarity(
                    text_embedding.unsqueeze(0),
                    concept_tensor.unsqueeze(0)
                ).item()
                
                if similarity > 0.6:  # Threshold for relevance
                    similarities[concept] = similarity
            
            # Get top similar concepts
            for concept, similarity in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]:
                if concept in self.semantic_model.concept_network:
                    relevant_concepts.append({
                        "concept": concept,
                        "category": self.semantic_model.concept_network[concept]["category"],
                        "confidence": similarity * 0.8,  # Reduce confidence for similarity-based matches
                        "similarity_match": True
                    })
        
        # Check for contextual meanings
        contextual_meanings = []
        
        if context and self.development_level >= 0.6:
            for word in words:
                if word in self.semantic_model.contextual_meanings:
                    # Look for matching context
                    context_words = context.lower().split()
                    for context_type, meaning in self.semantic_model.contextual_meanings[word].items():
                        if context_type in context_words:
                            contextual_meanings.append({
                                "word": word,
                                "context": context_type,
                                "meaning": meaning["meaning"],
                                "confidence": meaning["confidence"]
                            })
        
        # Record activation in neural state
        self.neural_state.add_activation("semantic_processing", {
            'operation': 'understand',
            'text': text,
            'relevant_concepts_count': len(relevant_concepts),
            'understanding_depth': understanding_depth
        })
        
        # Record in recent inputs
        self.recent_inputs.append({
            "type": "semantic_understanding",
            "text": text,
            "context": context,
            "timestamp": datetime.now()
        })
        
        # Return understanding results
        return {
            "status": "success",
            "text": text,
            "understanding_depth": understanding_depth,
            "relevant_concepts": relevant_concepts,
            "contextual_meanings": contextual_meanings,
            "development_level": self.development_level,
            "process_id": process_id
        }
    
    def _learn_concept(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Learn a new concept or update an existing one
        
        Args:
            input_data: Input data dictionary including concept information
            process_id: Process identifier
            
        Returns:
            Dict with learning results
        """
        # Check for required fields
        if "concept" not in input_data:
            return {
                "status": "error",
                "message": "Missing concept for learning",
                "process_id": process_id
            }
        
        concept = input_data["concept"]
        category = input_data.get("category", "uncategorized")
        features = input_data.get("features", {})
        
        # Check if concept already exists
        if concept in self.semantic_model.concept_network:
            # Update existing concept
            existing = self.semantic_model.concept_network[concept]
            
            # Update category if specified
            if category != "uncategorized":
                old_category = existing["category"]
                existing["category"] = category
                
                # Update semantic categories
                if old_category in self.semantic_model.semantic_categories and concept in self.semantic_model.semantic_categories[old_category]:
                    self.semantic_model.semantic_categories[old_category].remove(concept)
                
                if category in self.semantic_model.semantic_categories:
                    if concept not in self.semantic_model.semantic_categories[category]:
                        self.semantic_model.semantic_categories[category].append(concept)
                else:
                    self.semantic_model.semantic_categories[category] = [concept]
            
            # Update features if provided
            if features:
                for feature, value in features.items():
                    existing["features"][feature] = value
            
            # Increase confidence
            existing["confidence"] = min(1.0, existing["confidence"] + 0.05)
            
            # Generate or update embedding if development level is sufficient
            if self.development_level >= 0.3 and concept not in self.semantic_model.concept_embeddings:
                self._generate_concept_embedding(concept)
            
            # Record activation in neural state
            self.neural_state.add_activation("semantic_processing", {
                'operation': 'update_concept',
                'concept': concept,
                'confidence': existing["confidence"]
            })
            
            return {
                "status": "success",
                "message": "Updated existing concept",
                "concept": concept,
                "confidence": existing["confidence"],
                "process_id": process_id
            }
        else:
            # Create new concept
            concept_obj = {
                "concept_id": str(uuid.uuid4()),
                "category": category,
                "features": features,
                "related_concepts": [],
                "confidence": 0.5  # Initial confidence
            }
            
            # Development level affects initial confidence
            concept_obj["confidence"] *= max(0.5, self.development_level)
            
            # Add to concept network
            self.semantic_model.concept_network[concept] = concept_obj
            
            # Add to semantic categories
            if category in self.semantic_model.semantic_categories:
                if concept not in self.semantic_model.semantic_categories[category]:
                    self.semantic_model.semantic_categories[category].append(concept)
            else:
                self.semantic_model.semantic_categories[category] = [concept]
            
            # Generate embedding if development level is sufficient
            if self.development_level >= 0.3:
                self._generate_concept_embedding(concept)
            
            # Record activation in neural state
            self.neural_state.add_activation("semantic_processing", {
                'operation': 'learn_concept',
                'concept': concept,
                'confidence': concept_obj["confidence"]
            })
            
            return {
                "status": "success",
                "message": "Learned new concept",
                "concept": concept,
                "confidence": concept_obj["confidence"],
                "process_id": process_id
            }
    
    def _relate_concepts(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Create or identify relationships between concepts
        
        Args:
            input_data: Input data dictionary including concepts to relate
            process_id: Process identifier
            
        Returns:
            Dict with relation results
        """
        # Check for required fields
        if "concept1" not in input_data or "concept2" not in input_data:
            return {
                "status": "error",
                "message": "Missing concept1 or concept2 for relation",
                "process_id": process_id
            }
        
        concept1 = input_data["concept1"]
        concept2 = input_data["concept2"]
        relation_type = input_data.get("relation_type")
        
        # Check if both concepts exist
        if concept1 not in self.semantic_model.concept_network or concept2 not in self.semantic_model.concept_network:
            missing = []
            if concept1 not in self.semantic_model.concept_network:
                missing.append(concept1)
            if concept2 not in self.semantic_model.concept_network:
                missing.append(concept2)
                
            return {
                "status": "error",
                "message": f"Concepts not found: {', '.join(missing)}",
                "process_id": process_id
            }
        
        # Check development level for relations
        if self.development_level < 0.4:
            return {
                "status": "undeveloped",
                "message": "Concept relations require higher development level (0.4+)",
                "current_level": self.development_level,
                "process_id": process_id
            }
        
        # Get concept embeddings if available
        if concept1 in self.semantic_model.concept_embeddings and concept2 in self.semantic_model.concept_embeddings:
            embedding1 = torch.tensor(self.semantic_model.concept_embeddings[concept1], dtype=torch.float32).unsqueeze(0)
            embedding2 = torch.tensor(self.semantic_model.concept_embeddings[concept2], dtype=torch.float32).unsqueeze(0)
            
            # Process through network to identify relationship
            with torch.no_grad():
                output = self.network(
                    embedding1.to(self.device),
                    operation="relate",
                    second_concept=embedding2.to(self.device)
                )
                
                relation_clarity = float(output["relation_clarity"].cpu().item())
                
                # If relation type not specified, try to determine from network
                if not relation_type and "relation_probs" in output:
                    relation_probs = output["relation_probs"].cpu().numpy()[0]
                    relation_types = ["is_a", "has_a", "part_of", "similar_to", "opposite_of", "used_for", "located_at", "made_of", "caused_by", "member_of"]
                    
                    # Get most likely relation type
                    max_idx = np.argmax(relation_probs)
                    if max_idx < len(relation_types):
                        relation_type = relation_types[max_idx]
        else:
            # No embeddings available, use simpler approach
            relation_clarity = 0.5
        
        # Add to related concepts
        if concept2 not in self.semantic_model.concept_network[concept1]["related_concepts"]:
            self.semantic_model.concept_network[concept1]["related_concepts"].append(concept2)
        
        if concept1 not in self.semantic_model.concept_network[concept2]["related_concepts"]:
            self.semantic_model.concept_network[concept2]["related_concepts"].append(concept1)
        
        # Record activation in neural state
        self.neural_state.add_activation("semantic_processing", {
            'operation': 'relate_concepts',
            'concept1': concept1,
            'concept2': concept2,
            'relation_type': relation_type,
            'relation_clarity': relation_clarity
        })
        
        # Return relation results
        result = {
            "status": "success",
            "concept1": concept1,
            "concept2": concept2,
            "relation_clarity": relation_clarity,
            "process_id": process_id
        }
        
        if relation_type:
            result["relation_type"] = relation_type
            
        return result
    
    def _query_semantics(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query semantic information
        
        Args:
            input_data: Input data dictionary including query parameters
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return summary of semantic knowledge
            return {
                "status": "success",
                "concept_count": len(self.semantic_model.concept_network),
                "categories": list(self.semantic_model.semantic_categories.keys()),
                "contextual_meanings_count": len(self.semantic_model.contextual_meanings),
                "development_level": self.development_level,
                "process_id": process_id
            }
        
        elif query_type == "concept":
            # Check for concept
            if "concept" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing concept for concept query",
                    "process_id": process_id
                }
                
            concept = input_data["concept"]
            
            # Check if concept exists
            if concept not in self.semantic_model.concept_network:
                return {
                    "status": "error",
                    "message": f"Concept not found: {concept}",
                    "process_id": process_id
                }
                
            # Get concept information
            concept_info = self.semantic_model.concept_network[concept]
            
            result = {
                "status": "success",
                "concept": concept,
                "category": concept_info["category"],
                "features": concept_info["features"],
                "related_concepts": concept_info["related_concepts"],
                "confidence": concept_info["confidence"],
                "process_id": process_id
            }
            
            # Add embedding info if available but don't return the full embedding
            if concept in self.semantic_model.concept_embeddings:
                result["has_embedding"] = True
                
            # Add contextual meanings if available
            if concept in self.semantic_model.contextual_meanings:
                result["contextual_meanings"] = {
                    context: meaning["meaning"] 
                    for context, meaning in self.semantic_model.contextual_meanings[concept].items()
                }
                
            return result
        
        elif query_type == "category":
            # Check for category
            if "category" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing category for category query",
                    "process_id": process_id
                }
                
            category = input_data["category"]
            
            # Check if category exists
            if category not in self.semantic_model.semantic_categories:
                return {
                    "status": "error",
                    "message": f"Category not found: {category}",
                    "available_categories": list(self.semantic_model.semantic_categories.keys()),
                    "process_id": process_id
                }
                
            # Get concepts in category
            category_concepts = []
            for concept in self.semantic_model.semantic_categories[category]:
                if concept in self.semantic_model.concept_network:
                    category_concepts.append({
                        "concept": concept,
                        "confidence": self.semantic_model.concept_network[concept]["confidence"]
                    })
                    
            return {
                "status": "success",
                "category": category,
                "concepts": category_concepts,
                "concept_count": len(category_concepts),
                "process_id": process_id
            }
            
        elif query_type == "similar":
            # Check for query concept
            if "concept" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing concept for similarity query",
                    "process_id": process_id
                }
                
            concept = input_data["concept"]
            
            # Check development level
            if self.development_level < 0.5:
                return {
                    "status": "undeveloped",
                    "message": "Concept similarity query requires higher development level (0.5+)",
                    "current_level": self.development_level,
                    "process_id": process_id
                }
                
            # Check if concept exists and has embedding
            if concept not in self.semantic_model.concept_network:
                return {
                    "status": "error",
                    "message": f"Concept not found: {concept}",
                    "process_id": process_id
                }
                
            if concept not in self.semantic_model.concept_embeddings:
                return {
                    "status": "error",
                    "message": f"No embedding available for concept: {concept}",
                    "process_id": process_id
                }
                
            # Find similar concepts by embedding similarity
            similarities = {}
            target_embedding = torch.tensor(self.semantic_model.concept_embeddings[concept], dtype=torch.float32)
            
            for other_concept, embedding in self.semantic_model.concept_embeddings.items():
                if other_concept != concept:
                    # Convert to tensor
                    other_tensor = torch.tensor(embedding, dtype=torch.float32)
                    
                    # Ensure tensors have same dimensions
                    if target_embedding.shape != other_tensor.shape:
                        continue
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(
                        target_embedding.unsqueeze(0),
                        other_tensor.unsqueeze(0)
                    ).item()
                    
                    similarities[other_concept] = similarity
                    
            # Get top similar concepts
            top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "status": "success",
                "concept": concept,
                "similar_concepts": top_similar,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _extract_features(self, text: str) -> torch.Tensor:
        """
        Extract features from text
        
        Args:
            text: The text to extract features from
            
        Returns:
            Tensor of features
        """
        # Simple feature extraction
        words = text.lower().split()
        
        # Create a simple embedding for the text
        embedding = torch.zeros(128, dtype=torch.float32)
        
        for i, word in enumerate(words[:min(len(words), 20)]):  # Limit to 20 words
            # Get position in embedding
            pos = (hash(word) + i) % 120  # Keep a few positions for special features
            embedding[pos] = 1.0
            
            # Add emphasis on first and last words
            if i == 0:
                embedding[120] = 1.0  # First word marker
            if i == len(words) - 1:
                embedding[121] = 1.0  # Last word marker
                
            # Add known concept marker if word is a known concept
            if word in self.semantic_model.concept_network:
                embedding[122] = 1.0
                
                # Add category information
                category = self.semantic_model.concept_network[word]["category"]
                category_hash = hash(category) % 5
                embedding[123 + category_hash] = 1.0
        
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
        
        # Update neural state
        self.neural_state.semantic_processing_development = self.development_level
        self.neural_state.last_updated = datetime.now()
        
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
                            "module": "semantic_processing",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Update semantic concepts for new development level
                self._initialize_semantic_concepts()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the semantic processing module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "concept_count": len(self.semantic_model.concept_network),
            "category_count": len(self.semantic_model.semantic_categories),
            "embedding_count": len(self.semantic_model.concept_embeddings),
            "contextual_meanings_count": len(self.semantic_model.contextual_meanings)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        # Convert embeddings to lists for serialization (they should already be lists,
        # but ensure they're in a serializable format)
        concept_embeddings_serialized = {}
        for concept, embedding in self.semantic_model.concept_embeddings.items():
            if isinstance(embedding, torch.Tensor):
                concept_embeddings_serialized[concept] = embedding.numpy().tolist()
            else:
                concept_embeddings_serialized[concept] = embedding
                
        # Create a copy of the semantic model to modify
        semantic_model_dict = self.semantic_model.dict()
        semantic_model_dict["concept_embeddings"] = concept_embeddings_serialized
        
        return {
            "module_id": self.module_id,
            "semantic_model": semantic_model_dict,
            "developmental_level": self.development_level,
            "neural_state": {
                "development": self.neural_state.semantic_processing_development,
                "accuracy": self.neural_state.semantic_processing_accuracy
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state to load
        """
        # Load module ID
        self.module_id = state["module_id"]
        
        # Load development level
        self.development_level = state["developmental_level"]
        self.network.set_development_level(self.development_level)
        
        # Load semantic model
        if "semantic_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.semantic_model = parse_obj_as(SemanticModel, state["semantic_model"])
            except Exception as e:
                print(f"Error loading semantic model: {e}")
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            self.neural_state.semantic_processing_development = ns.get("development", self.development_level)
            self.neural_state.semantic_processing_accuracy = ns.get("accuracy", 0.5)
