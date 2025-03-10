# TODO: Implement the ConceptCombination class to generate novel concepts
# This component should be able to:
# - Blend existing concepts to create new ones
# - Identify compatible conceptual properties for combination
# - Resolve conflicts when combining incompatible properties
# - Generate novel inferences from combined concepts

# TODO: Implement developmental progression in concept combination:
# - Simple property transfer in early stages
# - Basic blending of compatible concepts in childhood
# - Complex integration of diverse concepts in adolescence
# - Sophisticated conceptual blending with emergent properties in adulthood

# TODO: Create mechanisms for:
# - Property mapping: Identify corresponding properties between concepts
# - Blend space creation: Generate new conceptual spaces from inputs
# - Conflict resolution: Handle contradictory properties in combined concepts
# - Emergent property inference: Derive new properties not present in source concepts

# TODO: Implement different combination strategies:
# - Property intersection: Retain only common properties
# - Property union: Retain all properties from both concepts
# - Selective projection: Strategically select properties to transfer
# - Emergent combination: Create entirely new properties

# TODO: Connect to memory and language systems
# Concept combination should draw from semantic memory
# and be influenced by linguistic knowledge

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.creativity.models import ConceptCombinationState, Concept, CreativeOutput
from lmm_project.modules.creativity.neural_net import ConceptCombiner

class ConceptCombination(BaseModule):
    """
    Combines existing concepts to create new ones
    
    This module creates new concepts by combining existing ones
    through various patterns such as blending, property transfer,
    and analogical mapping.
    
    Developmental progression:
    - Simple associations in early stages
    - Basic property transfers in childhood
    - Conceptual blending in adolescence
    - Complex analogical reasoning in adulthood
    """
    
    # Developmental milestones for concept combination
    development_milestones = {
        0.0: "simple_association",      # Basic association of concepts
        0.25: "property_transfer",      # Transferring properties between concepts
        0.5: "conceptual_blending",     # Blending concepts to form new ones
        0.75: "analogical_mapping",     # Using analogies to map concepts
        0.9: "creative_abstraction"     # Creating abstract concepts from concrete ones
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the concept combination module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="concept_combination", event_bus=event_bus)
        
        # Initialize state
        self.state = ConceptCombinationState()
        
        # Initialize neural network for concept combination
        self.input_dim = 128  # Default dimension
        self.network = ConceptCombiner(
            concept_dim=self.input_dim,
            hidden_dim=256,
            output_dim=self.input_dim
        )
        
        # Initialize concept combination patterns with usage frequencies
        self.state.combination_patterns = {
            "association": 1.0,  # Available at all development levels
            "property_transfer": 0.0,  # Will be enabled with development
            "blend": 0.0,  # Will be enabled with development
            "analogy": 0.0  # Will be enabled with development
        }
        
        # Subscribe to relevant events
        if self.event_bus:
            self.event_bus.subscribe("concept_created", self._handle_concept)
            self.event_bus.subscribe("combination_request", self._handle_combination_request)
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to combine concepts
        
        Args:
            input_data: Dictionary containing concepts to combine and parameters
            
        Returns:
            Dictionary with the results of concept combination
        """
        # Extract input information
        concepts = input_data.get("concepts", [])
        pattern = input_data.get("pattern", self._select_combination_pattern())
        context = input_data.get("context", {})
        
        # Validate input
        if not concepts or len(concepts) < 2:
            return {
                "status": "error",
                "message": "At least two concepts required for combination",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
            
        # Load concepts (either directly provided or by ID)
        concept_objects = []
        for concept in concepts:
            if isinstance(concept, dict) and "concept_id" in concept:
                # Concept directly provided
                concept_obj = Concept(**concept)
                # Cache for later use
                self.state.concept_cache[concept_obj.concept_id] = concept_obj
                concept_objects.append(concept_obj)
            elif isinstance(concept, str):
                # Concept ID provided, check cache
                if concept in self.state.concept_cache:
                    concept_objects.append(self.state.concept_cache[concept])
                else:
                    # TODO: In a real system, you might need to fetch from a database here
                    # For now, return an error
                    return {
                        "status": "error",
                        "message": f"Concept {concept} not found in cache",
                        "module_id": self.module_id,
                        "module_type": self.module_type
                    }
            else:
                return {
                    "status": "error",
                    "message": "Invalid concept format",
                    "module_id": self.module_id,
                    "module_type": self.module_type
                }
                
        # Create concept embeddings (simplified - in a real system you would use actual embeddings)
        # Here we'll just create random tensors as placeholders
        concept_embeddings = []
        for concept in concept_objects:
            # In a real system, this would use the concept's features to create an embedding
            # For now, create a random tensor
            embedding = torch.randn(1, self.input_dim)
            concept_embeddings.append(embedding)
            
        # Apply the appropriate combination pattern
        result = self._combine_concepts(concept_objects, concept_embeddings, pattern, context)
        
        # Update state with new combination
        if result["status"] == "success":
            # Add to combinations
            self.state.combinations[result["combination_id"]] = {
                "concepts": [c.concept_id for c in concept_objects],
                "pattern": pattern,
                "result": result["concept"].model_dump(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update recent combinations
            self.state.recent_combinations.append(result["combination_id"])
            # Keep list at a reasonable size
            if len(self.state.recent_combinations) > 20:
                self.state.recent_combinations = self.state.recent_combinations[-20:]
                
            # Update pattern usage
            if pattern in self.state.combination_patterns:
                self.state.combination_patterns[pattern] += 0.1
                
            # Create creative output
            creative_output = CreativeOutput(
                content=result["concept"].model_dump(),
                output_type="combined_concept",
                novelty_score=result.get("novelty_score", 0.5),
                coherence_score=result.get("coherence_score", 0.5),
                usefulness_score=result.get("usefulness_score", 0.5),
                source_components=[self.module_id]
            )
            
            # Publish the creative output
            if self.event_bus:
                from lmm_project.core.message import Message
                
                self.event_bus.publish(
                    Message(
                        sender="concept_combination",
                        message_type="creative_output",
                        content=creative_output.model_dump()
                    )
                )
                
        # Publish the combination result
        if self.event_bus:
            self.event_bus.publish(
                Message(
                    sender="concept_combination",
                    message_type="combination_result",
                    content=result
                )
            )
                
        return result
        
    def update_development(self, amount: float) -> float:
        """
        Update the module's developmental level
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # Update neural network if available
        if hasattr(self, 'network') and hasattr(self.network, 'update_development'):
            self.network.update_development(amount)
        
        # Adjust combination patterns based on development
        self._adjust_combination_patterns()
        
        return new_level
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_association"
        for level, name in sorted(self.development_milestones.items()):
            if self.development_level >= level:
                milestone = name
        return milestone
        
    def _select_combination_pattern(self) -> str:
        """Select a combination pattern based on development level and pattern frequencies"""
        # Filter patterns by availability (non-zero frequency)
        available_patterns = {k: v for k, v in self.state.combination_patterns.items() if v > 0}
        
        if not available_patterns:
            return "association"  # Default fallback
            
        # Select pattern based on probabilities
        patterns = list(available_patterns.keys())
        frequencies = list(available_patterns.values())
        total = sum(frequencies)
        probabilities = [f / total for f in frequencies]
        
        return np.random.choice(patterns, p=probabilities)
        
    def _combine_concepts(self, 
                         concepts: List[Concept], 
                         embeddings: List[torch.Tensor],
                         pattern: str, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine concepts using a specified pattern
        
        Args:
            concepts: List of concepts to combine
            embeddings: Tensor embeddings of concepts
            pattern: Combination pattern to use
            context: Additional context for combination
            
        Returns:
            Dictionary with combination results
        """
        # Initialize result
        result = {
            "status": "success",
            "module_id": self.module_id,
            "module_type": self.module_type,
            "pattern": pattern,
            "combination_id": str(uuid.uuid4())
        }
        
        try:
            # Get concept names for the new concept name
            concept_names = [c.name for c in concepts]
            
            # Process based on pattern
            if pattern == "association":
                # Simple association of concepts
                new_concept = self._association_combination(concepts, embeddings)
                
                # Validate
                if new_concept is None:
                    return {
                        "status": "error",
                        "message": "Failed to create association",
                        "module_id": self.module_id,
                        "module_type": self.module_type
                    }
                    
                result["concept"] = new_concept
                result["novelty_score"] = 0.3
                result["coherence_score"] = 0.7
                
            elif pattern == "property_transfer":
                # Transfer properties from one concept to another
                if len(concepts) < 2:
                    return {
                        "status": "error",
                        "message": "Property transfer requires at least 2 concepts",
                        "module_id": self.module_id,
                        "module_type": self.module_type
                    }
                    
                new_concept = self._property_transfer_combination(concepts, embeddings)
                
                result["concept"] = new_concept
                result["novelty_score"] = 0.5
                result["coherence_score"] = 0.6
                
            elif pattern == "blend":
                # Conceptual blending
                if len(concepts) < 2:
                    return {
                        "status": "error",
                        "message": "Blending requires at least 2 concepts",
                        "module_id": self.module_id,
                        "module_type": self.module_type
                    }
                    
                new_concept = self._blend_combination(concepts, embeddings)
                
                result["concept"] = new_concept
                result["novelty_score"] = 0.7
                result["coherence_score"] = 0.5
                
            elif pattern == "analogy":
                # Analogical mapping
                if len(concepts) < 3:
                    return {
                        "status": "error",
                        "message": "Analogy requires at least 3 concepts",
                        "module_id": self.module_id,
                        "module_type": self.module_type
                    }
                    
                new_concept = self._analogy_combination(concepts, embeddings)
                
                result["concept"] = new_concept
                result["novelty_score"] = 0.8
                result["coherence_score"] = 0.4
                
            else:
                # Unknown pattern
                return {
                    "status": "error",
                    "message": f"Unknown combination pattern: {pattern}",
                    "module_id": self.module_id,
                    "module_type": self.module_type
                }
                
            # Calculate usefulness score based on development level
            result["usefulness_score"] = 0.3 + 0.5 * self.development_level
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error combining concepts: {str(e)}",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
    
    def _association_combination(self, 
                                concepts: List[Concept], 
                                embeddings: List[torch.Tensor]) -> Concept:
        """Simple association of concepts"""
        # Get concept names
        names = [c.name for c in concepts]
        new_name = " & ".join(names)
        
        # Combine features (union of all features)
        new_features = {}
        for concept in concepts:
            new_features.update(concept.features)
            
        # Get associations (union of all associations)
        new_associations = []
        for concept in concepts:
            new_associations.extend(concept.associations)
            
        # Remove duplicates
        new_associations = list(set(new_associations))
        
        # Add original concepts as associations
        for concept in concepts:
            if concept.concept_id not in new_associations:
                new_associations.append(concept.concept_id)
        
        # Create new concept
        return Concept(
            name=new_name,
            features=new_features,
            associations=new_associations,
            source=f"association({', '.join(names)})"
        )
        
    def _property_transfer_combination(self, 
                                      concepts: List[Concept], 
                                      embeddings: List[torch.Tensor]) -> Concept:
        """Transfer properties from one concept to another"""
        # Use the first concept as base, second as modifier
        base_concept = concepts[0]
        modifier_concept = concepts[1]
        
        # Create a new name
        new_name = f"{modifier_concept.name}-like {base_concept.name}"
        
        # Start with base concept features
        new_features = dict(base_concept.features)
        
        # Use neural network to determine which features to transfer
        with torch.no_grad():
            _, attention_dict = self.network([embeddings[0], embeddings[1]], "property_transfer")
            
        if attention_dict.get("attention") is not None:
            attention = attention_dict["attention"].item()
        else:
            attention = 0.5  # Default if attention is not available
            
        # Select features to transfer based on attention
        transfer_features = {}
        for key, value in modifier_concept.features.items():
            if np.random.random() < attention:
                transfer_features[key] = value
                
        # Add transferred features, overwriting existing ones
        new_features.update(transfer_features)
        
        # Combine associations
        new_associations = list(base_concept.associations)
        for assoc in modifier_concept.associations:
            if assoc not in new_associations:
                new_associations.append(assoc)
                
        # Add original concepts as associations
        for concept in concepts:
            if concept.concept_id not in new_associations:
                new_associations.append(concept.concept_id)
        
        # Create new concept
        return Concept(
            name=new_name,
            features=new_features,
            associations=new_associations,
            source=f"property_transfer({base_concept.name}, {modifier_concept.name})"
        )
        
    def _blend_combination(self, 
                          concepts: List[Concept], 
                          embeddings: List[torch.Tensor]) -> Concept:
        """Blend two concepts to create a new one"""
        # Get the two concepts to blend
        concept1 = concepts[0]
        concept2 = concepts[1]
        
        # Create a blended name
        name_parts1 = concept1.name.split()
        name_parts2 = concept2.name.split()
        
        # Try to create a portmanteau if single words
        if len(name_parts1) == 1 and len(name_parts2) == 1:
            # Simple portmanteau: first half of first word + second half of second word
            half1 = len(name_parts1[0]) // 2
            half2 = len(name_parts2[0]) // 2
            new_name = name_parts1[0][:half1+1] + name_parts2[0][half2:]
        else:
            # Otherwise use both names
            new_name = f"{concept1.name}-{concept2.name} Blend"
            
        # Use neural network to blend concepts
        with torch.no_grad():
            combined_embedding, attention_dict = self.network([embeddings[0], embeddings[1]], "blend")
            
        # Blend features based on concept similarity
        new_features = {}
        
        # Add features from both concepts
        # In a real implementation, this would use the neural network output to create new features
        for key, value in concept1.features.items():
            new_features[key] = value
            
        for key, value in concept2.features.items():
            # If feature exists in both, create a blend
            if key in new_features:
                if isinstance(value, (int, float)) and isinstance(new_features[key], (int, float)):
                    # Average numerical values
                    new_features[key] = (new_features[key] + value) / 2
                else:
                    # For non-numeric, use the second value
                    new_features[key] = value
            else:
                # Add new feature
                new_features[key] = value
        
        # Create new emergent features (more likely with higher development)
        if np.random.random() < self.development_level:
            # Add an emergent feature that wasn't in either concept
            new_features["emergent_property"] = "Created through conceptual blending"
        
        # Combine associations
        new_associations = list(set(concept1.associations + concept2.associations))
        
        # Add original concepts as associations
        for concept in concepts:
            if concept.concept_id not in new_associations:
                new_associations.append(concept.concept_id)
        
        # Create new concept
        return Concept(
            name=new_name,
            features=new_features,
            associations=new_associations,
            source=f"blend({concept1.name}, {concept2.name})"
        )
        
    def _analogy_combination(self, 
                            concepts: List[Concept], 
                            embeddings: List[torch.Tensor]) -> Concept:
        """Create a concept through analogical mapping"""
        # For analogy we need at least 3 concepts: A is to B as C is to ?
        if len(concepts) < 3:
            raise ValueError("Analogy requires at least 3 concepts")
            
        concept_a = concepts[0]
        concept_b = concepts[1]
        concept_c = concepts[2]
        
        # Create a name for the new concept
        new_name = f"D in [{concept_a.name}:{concept_b.name}::{concept_c.name}:D]"
        
        # Use neural network to create analogy
        with torch.no_grad():
            combined_embedding, relation_dict = self.network(
                [embeddings[0], embeddings[1], embeddings[2]], 
                "analogy"
            )
            
        # Start with features from C
        new_features = dict(concept_c.features)
        
        # Identify differences between A and B
        diff_features = {}
        for key, value in concept_b.features.items():
            if key in concept_a.features:
                # Feature exists in both
                if isinstance(value, (int, float)) and isinstance(concept_a.features[key], (int, float)):
                    # Calculate numerical difference
                    diff = value - concept_a.features[key]
                    diff_features[key] = diff
                else:
                    # For non-numeric, note the change
                    diff_features[key] = (concept_a.features[key], value)
            else:
                # Feature in B but not in A (addition)
                diff_features[key] = ("added", value)
                
        for key in concept_a.features:
            if key not in concept_b.features:
                # Feature in A but not in B (removal)
                diff_features[key] = ("removed", concept_a.features[key])
        
        # Apply differences to C to create D
        for key, diff in diff_features.items():
            if key in new_features:
                if isinstance(diff, (int, float)) and isinstance(new_features[key], (int, float)):
                    # Apply numerical difference
                    new_features[key] += diff
                elif isinstance(diff, tuple) and diff[0] == "removed":
                    # Remove feature
                    del new_features[key]
                elif isinstance(diff, tuple) and isinstance(diff[0], (int, float, str)):
                    # Update with new value based on the relationship
                    new_features[key] = diff[1]
            else:
                # Feature not in C
                if isinstance(diff, tuple) and diff[0] == "added":
                    # Add the feature
                    new_features[key] = diff[1]
        
        # Combine associations, focusing on B and C
        new_associations = list(set(concept_b.associations + concept_c.associations))
        
        # Add original concepts as associations
        for concept in concepts:
            if concept.concept_id not in new_associations:
                new_associations.append(concept.concept_id)
        
        # Create new concept
        return Concept(
            name=new_name,
            features=new_features,
            associations=new_associations,
            source=f"analogy({concept_a.name}, {concept_b.name}, {concept_c.name})"
        )
    
    def _handle_concept(self, message: Message) -> None:
        """Handle concept creation messages"""
        if isinstance(message.content, dict):
            concept_data = message.content
            # Cache the concept for potential future use
            concept = Concept(**concept_data)
            self.state.concept_cache[concept.concept_id] = concept
            
    def _handle_combination_request(self, message: Message) -> None:
        """Handle concept combination requests"""
        if isinstance(message.content, dict):
            # Process the combination request
            result = self.process_input(message.content)
            
            # Publish result if successful
            if result["status"] == "success" and self.event_bus:
                self.event_bus.publish(
                    Message(
                        sender="concept_combination",
                        message_type="combination_result",
                        content=result
                    )
                )

    def _adjust_combination_patterns(self) -> None:
        """
        Adjust available combination patterns based on development level
        
        As development increases, more sophisticated combination patterns become available:
        - Association (always available)
        - Property transfer (available at development level 0.25+)
        - Conceptual blending (available at development level 0.5+)
        - Analogical mapping (available at development level 0.75+)
        """
        # Enable property transfer at development level 0.25+
        if self.development_level >= 0.25:
            self.state.combination_patterns["property_transfer"] = 0.5
        else:
            self.state.combination_patterns["property_transfer"] = 0.0
            
        # Enable conceptual blending at development level 0.5+
        if self.development_level >= 0.5:
            self.state.combination_patterns["blend"] = 0.5
        else:
            self.state.combination_patterns["blend"] = 0.0
            
        # Enable analogical mapping at development level 0.75+
        if self.development_level >= 0.75:
            self.state.combination_patterns["analogy"] = 0.5
        else:
            self.state.combination_patterns["analogy"] = 0.0
