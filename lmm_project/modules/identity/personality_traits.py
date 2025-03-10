# TODO: Implement the PersonalityTraits class to represent stable behavior patterns
# This component should be able to:
# - Represent consistent patterns of thinking, feeling, and behaving
# - Develop traits gradually through experience
# - Maintain trait stability while allowing for growth and change
# - Express traits through behavior in context-appropriate ways

# TODO: Implement developmental progression in personality traits:
# - Simple temperamental tendencies in early stages
# - Growing behavioral consistencies in childhood
# - Trait consolidation in adolescence
# - Stable yet nuanced personality in adulthood

# TODO: Create mechanisms for:
# - Trait extraction: Identify patterns across behaviors
# - Trait integration: Organize traits into coherent dimensions
# - Trait expression: Apply traits to guide behavior
# - Trait adaptation: Adjust expression based on context

# TODO: Implement trait frameworks:
# - Consider using established models (Big Five, etc.)
# - Include traits for thinking styles (analytical, intuitive, etc.)
# - Include traits for emotional tendencies (reactive, stable, etc.)
# - Include traits for behavioral patterns (cautious, impulsive, etc.)

# TODO: Connect to behavior generation and social systems
# Traits should influence behavior production and
# should develop through social interactions

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
from datetime import datetime
import numpy as np
from pydantic import ValidationError

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
from lmm_project.modules.identity.models import (
    PersonalityTrait, 
    TraitDimension, 
    PersonalityProfile
)
from lmm_project.modules.identity.neural_net import PersonalityNetwork, get_device

class PersonalityTraits(BaseModule):
    """
    Manages personality traits and their expression
    
    This module represents stable patterns of thinking, feeling, and behaving
    that develop gradually and become increasingly consistent over time.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic temperamental tendencies",
        0.2: "Simple behavioral consistencies",
        0.4: "Emerging trait patterns across situations",
        0.6: "Consolidated traits with some situational flexibility",
        0.8: "Stable trait dimensions with contextual adaptation",
        1.0: "Fully developed personality with balance of stability and adaptation"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the personality traits module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize personality profile
        self.profile = PersonalityProfile()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = PersonalityNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize trait embeddings for similarity search
        self.trait_embeddings = {}
        self.dimension_embeddings = {}
        
        # Initialize with basic traits and dimensions based on development level
        self._initialize_basic_traits()
    
    def _initialize_basic_traits(self):
        """Initialize basic traits based on development level"""
        # Initialize with Big Five dimensions at a minimum
        dimensions = [
            {
                "name": "Extraversion",
                "description": "Tendency to seek stimulation and engage with others",
                "positive_pole": "Extraverted, outgoing, energetic, sociable",
                "negative_pole": "Introverted, reserved, solitary, quiet",
                "score": 0.5
            },
            {
                "name": "Neuroticism",
                "description": "Tendency to experience negative emotions",
                "positive_pole": "Emotionally stable, calm, secure, resilient",
                "negative_pole": "Anxious, irritable, moody, insecure",
                "score": 0.5
            },
            {
                "name": "Agreeableness",
                "description": "Tendency to be compassionate toward others",
                "positive_pole": "Cooperative, compassionate, kind, trusting",
                "negative_pole": "Critical, suspicious, uncooperative, challenging",
                "score": 0.5
            },
            {
                "name": "Conscientiousness",
                "description": "Tendency to show self-discipline and aim for achievement",
                "positive_pole": "Organized, disciplined, careful, responsible",
                "negative_pole": "Spontaneous, careless, disorganized, impulsive",
                "score": 0.5
            },
            {
                "name": "Openness",
                "description": "Tendency to appreciate novelty and variety",
                "positive_pole": "Creative, curious, open-minded, imaginative",
                "negative_pole": "Conventional, practical, focused, traditional",
                "score": 0.5
            }
        ]
        
        # Create each dimension
        for dim_info in dimensions:
            dimension = TraitDimension(
                dimension_id=str(uuid.uuid4()),
                name=dim_info["name"],
                description=dim_info["description"],
                positive_pole=dim_info["positive_pole"],
                negative_pole=dim_info["negative_pole"],
                score=dim_info["score"]
            )
            
            # Add to profile
            self.profile.add_dimension(dimension)
            
            # Create initial embedding for this dimension
            features = self._extract_features(f"{dimension.name}: {dimension.description}")
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    operation="extract_traits"
                )
                self.dimension_embeddings[dimension.dimension_id] = output["trait_encoding"].cpu().squeeze(0)
        
        # If development level is high enough, add some basic traits
        if self.development_level >= 0.2:
            # Create a few basic traits based on the dimensions
            for dim_id, dimension in self.profile.dimensions.items():
                if dimension.name == "Extraversion":
                    self._create_initial_trait("Sociable", 0.6, dimension.dimension_id, is_positive=True)
                    self._create_initial_trait("Quiet", 0.4, dimension.dimension_id, is_positive=False)
                elif dimension.name == "Neuroticism":
                    self._create_initial_trait("Calm", 0.6, dimension.dimension_id, is_positive=True)
                    self._create_initial_trait("Anxious", 0.4, dimension.dimension_id, is_positive=False)
                elif dimension.name == "Agreeableness":
                    self._create_initial_trait("Kind", 0.6, dimension.dimension_id, is_positive=True)
                    self._create_initial_trait("Critical", 0.4, dimension.dimension_id, is_positive=False)
                elif dimension.name == "Conscientiousness":
                    self._create_initial_trait("Organized", 0.6, dimension.dimension_id, is_positive=True)
                    self._create_initial_trait("Impulsive", 0.4, dimension.dimension_id, is_positive=False)
                elif dimension.name == "Openness":
                    self._create_initial_trait("Curious", 0.6, dimension.dimension_id, is_positive=True)
                    self._create_initial_trait("Traditional", 0.4, dimension.dimension_id, is_positive=False)
    
    def _create_initial_trait(self, name: str, score: float, dimension_id: str, is_positive: bool):
        """
        Create an initial trait for a dimension
        
        Args:
            name: Trait name
            score: Trait score (0.0 to 1.0)
            dimension_id: ID of the dimension this trait belongs to
            is_positive: Whether this trait is on the positive or negative pole
        """
        dimension = self.profile.dimensions[dimension_id]
        
        # Create description based on dimension's poles
        if is_positive:
            description = f"Tendency to be {name.lower()}, related to {dimension.name.lower()}"
        else:
            description = f"Tendency to be {name.lower()}, opposite of high {dimension.name.lower()}"
        
        # Create the trait
        trait = PersonalityTrait(
            trait_id=str(uuid.uuid4()),
            name=name,
            description=description,
            score=score,
            stability=0.3 + (self.development_level * 0.3),  # More stable at higher development
            dimension=dimension_id
        )
        
        # Add to profile
        self.profile.add_trait(trait)
        
        # Create initial embedding for this trait
        features = self._extract_features(f"{trait.name}: {trait.description}")
        with torch.no_grad():
            output = self.network(
                features.to(self.device),
                operation="extract_traits"
            )
            self.trait_embeddings[trait.trait_id] = output["trait_encoding"].cpu().squeeze(0)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the personality traits module
        
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
        if operation == "add_trait":
            return self._add_trait(input_data, process_id)
        elif operation == "update_trait":
            return self._update_trait(input_data, process_id)
        elif operation == "add_dimension":
            return self._add_dimension(input_data, process_id)
        elif operation == "update_dimension":
            return self._update_dimension(input_data, process_id)
        elif operation == "extract_traits":
            return self._extract_traits(input_data, process_id)
        elif operation == "query_traits":
            return self._query_traits(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _add_trait(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Add a new personality trait
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check required fields
        required_fields = ["name", "description", "score"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "status": "error",
                    "message": f"Missing required field: {field}",
                    "process_id": process_id
                }
        
        try:
            # Create trait
            trait_id = str(uuid.uuid4())
            
            # Get dimension if provided
            dimension = input_data.get("dimension")
            
            # Create trait
            trait = PersonalityTrait(
                trait_id=trait_id,
                name=input_data["name"],
                description=input_data["description"],
                score=float(input_data["score"]),
                stability=input_data.get("stability", 0.5),
                behavioral_instances=input_data.get("behavioral_instances", []),
                opposing_trait=input_data.get("opposing_trait"),
                dimension=dimension
            )
            
            # Add to profile
            self.profile.add_trait(trait)
            
            # Create embedding for this trait
            features = self._extract_features(f"{trait.name}: {trait.description}")
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    operation="extract_traits"
                )
                self.trait_embeddings[trait_id] = output["trait_encoding"].cpu().squeeze(0)
            
            return {
                "status": "success",
                "message": "Trait added successfully",
                "trait_id": trait_id,
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
                "message": f"Error adding trait: {str(e)}",
                "process_id": process_id
            }
    
    def _update_trait(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Update an existing personality trait
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check for trait ID
        if "trait_id" not in input_data:
            return {
                "status": "error",
                "message": "Missing trait_id",
                "process_id": process_id
            }
            
        trait_id = input_data["trait_id"]
        
        # Check if trait exists
        if trait_id not in self.profile.traits:
            return {
                "status": "error",
                "message": f"Trait not found: {trait_id}",
                "process_id": process_id
            }
            
        # Get trait
        trait = self.profile.traits[trait_id]
        
        # Track if updated
        updated = False
        
        # Update fields
        if "name" in input_data:
            trait.name = input_data["name"]
            updated = True
            
        if "description" in input_data:
            trait.description = input_data["description"]
            updated = True
            
        if "score" in input_data:
            trait.score = max(0.0, min(1.0, float(input_data["score"])))
            updated = True
            
        if "stability" in input_data:
            trait.stability = max(0.0, min(1.0, float(input_data["stability"])))
            updated = True
            
        if "behavioral_instances" in input_data:
            if isinstance(input_data["behavioral_instances"], list):
                trait.behavioral_instances = input_data["behavioral_instances"]
                updated = True
                
        if "opposing_trait" in input_data:
            trait.opposing_trait = input_data["opposing_trait"]
            updated = True
            
        if "dimension" in input_data:
            trait.dimension = input_data["dimension"]
            updated = True
        
        if updated:
            # Update timestamp
            trait.updated_at = datetime.now()
            
            # Update in profile
            self.profile.traits[trait_id] = trait
            self.profile.last_updated = datetime.now()
            
            # Update embedding if name or description changed
            if "name" in input_data or "description" in input_data:
                features = self._extract_features(f"{trait.name}: {trait.description}")
                with torch.no_grad():
                    output = self.network(
                        features.to(self.device),
                        operation="extract_traits"
                    )
                    self.trait_embeddings[trait_id] = output["trait_encoding"].cpu().squeeze(0)
            
            return {
                "status": "success",
                "message": "Trait updated successfully",
                "process_id": process_id
            }
        else:
            return {
                "status": "error",
                "message": "No fields updated",
                "process_id": process_id
            }
    
    def _add_dimension(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Add a new trait dimension
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check required fields
        required_fields = ["name", "description", "positive_pole", "negative_pole"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "status": "error",
                    "message": f"Missing required field: {field}",
                    "process_id": process_id
                }
        
        try:
            # Create dimension
            dimension_id = str(uuid.uuid4())
            
            # Create dimension
            dimension = TraitDimension(
                dimension_id=dimension_id,
                name=input_data["name"],
                description=input_data["description"],
                positive_pole=input_data["positive_pole"],
                negative_pole=input_data["negative_pole"],
                score=input_data.get("score", 0.5)
            )
            
            # Add to profile
            self.profile.add_dimension(dimension)
            
            # Create embedding for this dimension
            features = self._extract_features(f"{dimension.name}: {dimension.description}")
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    operation="extract_traits"
                )
                self.dimension_embeddings[dimension_id] = output["trait_encoding"].cpu().squeeze(0)
            
            return {
                "status": "success",
                "message": "Dimension added successfully",
                "dimension_id": dimension_id,
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
                "message": f"Error adding dimension: {str(e)}",
                "process_id": process_id
            }
    
    def _update_dimension(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Update an existing trait dimension
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check for dimension ID
        if "dimension_id" not in input_data:
            return {
                "status": "error",
                "message": "Missing dimension_id",
                "process_id": process_id
            }
            
        dimension_id = input_data["dimension_id"]
        
        # Check if dimension exists
        if dimension_id not in self.profile.dimensions:
            return {
                "status": "error",
                "message": f"Dimension not found: {dimension_id}",
                "process_id": process_id
            }
            
        # Get dimension
        dimension = self.profile.dimensions[dimension_id]
        
        # Track if updated
        updated = False
        
        # Update fields
        if "name" in input_data:
            dimension.name = input_data["name"]
            updated = True
            
        if "description" in input_data:
            dimension.description = input_data["description"]
            updated = True
            
        if "positive_pole" in input_data:
            dimension.positive_pole = input_data["positive_pole"]
            updated = True
            
        if "negative_pole" in input_data:
            dimension.negative_pole = input_data["negative_pole"]
            updated = True
            
        if "score" in input_data:
            dimension.score = max(0.0, min(1.0, float(input_data["score"])))
            updated = True
        
        if updated:
            # Update timestamp
            dimension.updated_at = datetime.now()
            
            # Update in profile
            self.profile.dimensions[dimension_id] = dimension
            self.profile.last_updated = datetime.now()
            
            # Update embedding if name or description changed
            if "name" in input_data or "description" in input_data:
                features = self._extract_features(f"{dimension.name}: {dimension.description}")
                with torch.no_grad():
                    output = self.network(
                        features.to(self.device),
                        operation="extract_traits"
                    )
                    self.dimension_embeddings[dimension_id] = output["trait_encoding"].cpu().squeeze(0)
            
            return {
                "status": "success",
                "message": "Dimension updated successfully",
                "process_id": process_id
            }
        else:
            return {
                "status": "error",
                "message": "No fields updated",
                "process_id": process_id
            }
    
    def _extract_traits(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Extract personality traits from behavioral data
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Check for behavior description
        if "behavior" not in input_data:
            return {
                "status": "error",
                "message": "Missing behavior description",
                "process_id": process_id
            }
            
        behavior = input_data["behavior"]
        
        # Extract features
        features = self._extract_features(behavior)
        
        # Process through network
        with torch.no_grad():
            output = self.network(
                features.to(self.device),
                operation="extract_traits"
            )
            
            trait_encoding = output["trait_encoding"].cpu().squeeze(0)
            
        # Find most similar traits
        similar_traits = []
        
        if len(self.trait_embeddings) > 0:
            # Calculate similarity to existing traits
            similarities = {}
            for trait_id, embedding in self.trait_embeddings.items():
                similarity = torch.cosine_similarity(
                    trait_encoding.unsqueeze(0),
                    embedding.unsqueeze(0)
                ).item()
                similarities[trait_id] = similarity
                
            # Get top 3 similar traits
            sorted_traits = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for trait_id, similarity in sorted_traits[:3]:
                if similarity > 0.7:  # Only include if reasonably similar
                    trait = self.profile.traits[trait_id]
                    similar_traits.append({
                        "trait_id": trait_id,
                        "name": trait.name,
                        "similarity": similarity
                    })
        
        # Update behavioral instances for the most similar trait
        if len(similar_traits) > 0 and similar_traits[0]["similarity"] > 0.85:
            most_similar_trait_id = similar_traits[0]["trait_id"]
            trait = self.profile.traits[most_similar_trait_id]
            
            # Add behavior as an instance if not too many already
            if len(trait.behavioral_instances) < 20:  # Limit number of instances
                trait.behavioral_instances.append(behavior)
                trait.updated_at = datetime.now()
                self.profile.traits[most_similar_trait_id] = trait
            
            # Update trait score with small adjustment toward extremes
            current_score = trait.score
            if current_score > 0.5:
                # Strengthen high score traits
                new_score = current_score + (0.02 * (1.0 - trait.stability))
                trait.score = min(1.0, new_score)
            else:
                # Strengthen low score traits
                new_score = current_score - (0.02 * (1.0 - trait.stability))
                trait.score = max(0.0, new_score)
                
            self.profile.traits[most_similar_trait_id] = trait
        
        # If at higher development level, consider new trait formation
        if self.development_level >= 0.6 and len(similar_traits) == 0:
            # Extract potential trait name and description
            try:
                # Use network to generate trait properties
                context_tensor = torch.cat([
                    features, 
                    torch.zeros(1, features.size(1) - features.size(0)).to(features.device)
                ], dim=0)
                
                trait_properties = self.network(
                    context_tensor.to(self.device),
                    operation="generate_trait"
                )
                
                # If confidence is high enough, create a new trait
                if "trait_confidence" in trait_properties and trait_properties["trait_confidence"].item() > 0.7:
                    # Get development-appropriate stability
                    stability = 0.3 + (self.development_level * 0.4)
                    
                    # Create trait
                    self._add_trait({
                        "name": f"New Trait {len(self.profile.traits) + 1}",
                        "description": f"Pattern extracted from behavior: {behavior[:50]}...",
                        "score": 0.7,  # Start with moderately high score
                        "stability": stability,
                        "behavioral_instances": [behavior]
                    }, str(uuid.uuid4()))
            except Exception as e:
                # Silently fail trait generation - it's an advanced feature
                pass
        
        return {
            "status": "success",
            "similar_traits": similar_traits,
            "process_id": process_id
        }
    
    def _query_traits(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query personality traits
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with processing results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return all traits
            traits = []
            for trait_id, trait in self.profile.traits.items():
                traits.append({
                    "trait_id": trait_id,
                    "name": trait.name,
                    "description": trait.description,
                    "score": trait.score,
                    "stability": trait.stability,
                    "dimension": trait.dimension
                })
                
            return {
                "status": "success",
                "traits": traits,
                "process_id": process_id
            }
        
        elif query_type == "by_dimension":
            # Get dimension
            if "dimension_id" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing dimension_id for by_dimension query",
                    "process_id": process_id
                }
                
            dimension_id = input_data["dimension_id"]
            
            # Get traits for this dimension
            traits = []
            for trait_id, trait in self.profile.traits.items():
                if trait.dimension == dimension_id:
                    traits.append({
                        "trait_id": trait_id,
                        "name": trait.name,
                        "description": trait.description,
                        "score": trait.score,
                        "stability": trait.stability
                    })
                    
            if not traits:
                return {
                    "status": "success",
                    "message": f"No traits found for dimension {dimension_id}",
                    "traits": [],
                    "process_id": process_id
                }
                
            return {
                "status": "success",
                "traits": traits,
                "process_id": process_id
            }
        
        elif query_type == "similar":
            # Get query
            if "query" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing query for similar query",
                    "process_id": process_id
                }
                
            query = input_data["query"]
            
            # Extract features
            features = self._extract_features(query)
            
            # Process through network
            with torch.no_grad():
                output = self.network(
                    features.to(self.device),
                    operation="extract_traits"
                )
                
                query_encoding = output["trait_encoding"].cpu().squeeze(0)
                
            # Find most similar traits
            similar_traits = []
            
            if len(self.trait_embeddings) > 0:
                # Calculate similarity to existing traits
                similarities = {}
                for trait_id, embedding in self.trait_embeddings.items():
                    similarity = torch.cosine_similarity(
                        query_encoding.unsqueeze(0),
                        embedding.unsqueeze(0)
                    ).item()
                    similarities[trait_id] = similarity
                    
                # Get top 5 similar traits
                sorted_traits = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                for trait_id, similarity in sorted_traits[:5]:
                    trait = self.profile.traits[trait_id]
                    similar_traits.append({
                        "trait_id": trait_id,
                        "name": trait.name,
                        "description": trait.description,
                        "score": trait.score,
                        "similarity": similarity
                    })
                    
            return {
                "status": "success",
                "similar_traits": similar_traits,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
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
                            "module": "personality_traits",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Log milestone
                print(f"Personality Traits Development Milestone: {milestone} (level {level})")
                
                # Perform developmental updates
                if level == 0.4:
                    # At this level, traits become more stable
                    for trait_id, trait in self.profile.traits.items():
                        trait.stability = min(0.7, trait.stability + 0.2)
                        self.profile.traits[trait_id] = trait
                        
                elif level == 0.6:
                    # At this level, traits organize into more coherent dimensions
                    self.profile.integration = min(0.8, self.profile.integration + 0.3)
                    self.profile.differentiation = min(0.8, self.profile.differentiation + 0.3)
                    
                elif level == 0.8:
                    # At this level, traits become highly stable
                    for trait_id, trait in self.profile.traits.items():
                        trait.stability = min(0.9, trait.stability + 0.2)
                        self.profile.traits[trait_id] = trait
                        
                    self.profile.stability = min(0.9, self.profile.stability + 0.3)
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the personality traits module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "personality_profile": self.profile.dict(),
            "developmental_level": self.development_level,
            "trait_count": len(self.profile.traits),
            "dimension_count": len(self.profile.dimensions)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        # Convert embeddings to lists for serialization
        trait_embeddings_serialized = {}
        for trait_id, embedding in self.trait_embeddings.items():
            trait_embeddings_serialized[trait_id] = embedding.numpy().tolist()
            
        dimension_embeddings_serialized = {}
        for dim_id, embedding in self.dimension_embeddings.items():
            dimension_embeddings_serialized[dim_id] = embedding.numpy().tolist()
            
        return {
            "module_id": self.module_id,
            "personality_profile": self.profile.dict(),
            "developmental_level": self.development_level,
            "trait_embeddings": trait_embeddings_serialized,
            "dimension_embeddings": dimension_embeddings_serialized
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
        
        # Load personality profile
        if "personality_profile" in state:
            try:
                profile_data = state["personality_profile"]
                self.profile = PersonalityProfile(**profile_data)
            except Exception as e:
                print(f"Error loading personality profile: {e}")
        
        # Load embeddings
        if "trait_embeddings" in state:
            self.trait_embeddings = {}
            for trait_id, embedding_list in state["trait_embeddings"].items():
                self.trait_embeddings[trait_id] = torch.tensor(embedding_list, dtype=torch.float32)
                
        if "dimension_embeddings" in state:
            self.dimension_embeddings = {}
            for dim_id, embedding_list in state["dimension_embeddings"].items():
                self.dimension_embeddings[dim_id] = torch.tensor(embedding_list, dtype=torch.float32)
