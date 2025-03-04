# semantic_network.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
import numpy as np
from pydantic import BaseModel, Field, field_validator
import networkx as nx
from scipy.spatial.distance import cosine
import json
from pathlib import Path
from enum import Enum
import os

logger = logging.getLogger("SemanticNetwork")

class SemanticRelation(str, Enum):
    """Types of semantic relations between concepts"""
    IS_A = "is_a"                  # Hypernym/hyponym (dog is_a animal)
    HAS_PROPERTY = "has_property"  # Object-property (ball is_round)
    PART_OF = "part_of"            # Meronym/holonym (arm part_of body)
    USED_FOR = "used_for"          # Function (spoon used_for eating)
    SIMILAR_TO = "similar_to"      # Similarity (happy similar_to glad)
    OPPOSITE_OF = "opposite_of"    # Antonym (hot opposite_of cold)
    LOCATED_AT = "located_at"      # Spatial (toy located_at room)
    ASSOCIATED = "associated"      # General association (cat associated with purr)

class ConceptNode(BaseModel):
    """A concept in the semantic network"""
    word: str
    category: Optional[str] = None
    properties: Dict[str, float] = Field(default_factory=dict)
    first_encountered: datetime = Field(default_factory=datetime.now)
    last_activated: datetime = Field(default_factory=datetime.now)
    activation_count: int = Field(0, ge=0)
    activation_level: float = Field(0.0, ge=0.0, le=1.0)
    emotional_associations: Dict[str, float] = Field(default_factory=dict)
    
    def update_activation(self, activation_intensity: float = 0.5) -> None:
        """Update activation level and related metrics"""
        # Apply time-based decay to current activation
        current_time = datetime.now()
        time_diff = (current_time - self.last_activated).total_seconds()
        decay_factor = max(0.0, 1.0 - (time_diff / 3600))  # More decay over time
        
        # Calculate new activation level
        self.activation_level = min(1.0, (self.activation_level * decay_factor) + activation_intensity)
        self.last_activated = current_time
        self.activation_count += 1
    
    def add_property(self, property_name: str, strength: float = 0.5) -> None:
        """Add or update a property associated with this concept"""
        if property_name in self.properties:
            # Blend existing and new association
            current = self.properties[property_name]
            self.properties[property_name] = (current * 0.7) + (strength * 0.3)
        else:
            self.properties[property_name] = strength
    
    def add_emotional_association(self, emotion: str, intensity: float = 0.5) -> None:
        """Add or update an emotional association"""
        if emotion in self.emotional_associations:
            # Blend existing and new association
            current = self.emotional_associations[emotion]
            self.emotional_associations[emotion] = (current * 0.7) + (intensity * 0.3)
        else:
            self.emotional_associations[emotion] = intensity
    
    def get_age_days(self) -> float:
        """Get the age of this concept in days"""
        return (datetime.now() - self.first_encountered).total_seconds() / 86400

class SemanticLink(BaseModel):
    """A connection between two concepts in the semantic network"""
    source: str
    target: str
    relation_type: SemanticRelation
    strength: float = Field(0.5, ge=0.0, le=1.0)
    first_formed: datetime = Field(default_factory=datetime.now)
    last_activated: datetime = Field(default_factory=datetime.now)
    activation_count: int = Field(0, ge=0)
    
    def update_strength(self, increment: float = 0.1) -> None:
        """Update the strength of this semantic link"""
        self.strength = min(1.0, self.strength + increment)
        self.last_activated = datetime.now()
        self.activation_count += 1
    
    def decay_strength(self, days_elapsed: float = 1.0) -> None:
        """Apply decay to link strength over time"""
        decay_factor = min(0.99, 1.0 - (0.01 * days_elapsed))
        self.strength *= decay_factor

class SemanticNetwork:
    """Manages the semantic network representing relationships between concepts"""
    
    def __init__(self, data_dir: Path = Path("./data/semantic_network")):
        """Initialize the semantic network"""
        self.data_dir = data_dir
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Initialize network
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, ConceptNode] = {}
        self.links: List[SemanticLink] = []
        
        # Try to initialize word embeddings if available
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.embeddings_available = True
            logger.info("Word embeddings available through spaCy")
        except (ImportError, OSError):
            logger.warning("spaCy not available; semantic similarity will be limited")
            self.nlp = None
            self.embeddings_available = False
        
        # Initialize some basic semantic categories
        self.semantic_categories = {
            "animal": ["dog", "cat", "bird", "fish", "horse"],
            "food": ["milk", "apple", "cookie", "banana", "juice"],
            "toy": ["ball", "doll", "block", "book", "teddy"],
            "person": ["mommy", "daddy", "baby", "friend", "doctor"],
            "action": ["eat", "sleep", "play", "run", "jump"],
            "body_part": ["head", "hand", "foot", "eye", "ear"],
            "clothing": ["shoe", "sock", "hat", "shirt", "pants"],
            "vehicle": ["car", "bus", "train", "bike", "airplane"],
            "household": ["bed", "chair", "table", "door", "window"],
            "color": ["red", "blue", "green", "yellow", "black"]
        }
        
        logger.info("Semantic network initialized")
    
    def add_concept(self, word: str, category: Optional[str] = None, 
                  properties: Optional[Dict[str, float]] = None,
                  emotions: Optional[Dict[str, float]] = None) -> ConceptNode:
        """Add a new concept to the semantic network"""
        word = word.lower().strip()
        
        # Check if concept already exists
        if word in self.concepts:
            # Update existing concept
            concept = self.concepts[word]
            concept.update_activation(0.5)
            
            # Update category if provided and not already set
            if category and not concept.category:
                concept.category = category
                
            # Update properties if provided
            if properties:
                for prop, strength in properties.items():
                    concept.add_property(prop, strength)
                    
            # Update emotional associations if provided
            if emotions:
                for emotion, intensity in emotions.items():
                    concept.add_emotional_association(emotion, intensity)
                    
            return concept
            
        # Create new concept
        concept = ConceptNode(
            word=word,
            category=category,
            properties=properties or {},
            emotional_associations=emotions or {}
        )
        
        # Add to network
        self.concepts[word] = concept
        self.graph.add_node(word, data=concept.model_dump())
        
        # Try to infer category if not provided
        if not category:
            # Check if word belongs to one of our predefined categories
            for cat, examples in self.semantic_categories.items():
                if word in examples:
                    concept.category = cat
                    break
            
            # Try to infer category through word embeddings
            if self.embeddings_available and self.nlp:
                self._infer_category_from_embeddings(concept)
        
        logger.info(f"Added new concept to semantic network: '{word}'")
        return concept
    
    def _infer_category_from_embeddings(self, concept: ConceptNode) -> None:
        """Try to infer category using word embeddings"""
        if not self.nlp:
            return
            
        word = concept.word
        word_vector = self.nlp(word).vector
        
        if np.all(word_vector == 0):  # Check if zero vector (word not in vocabulary)
            return
            
        # Compare with category prototypes
        best_category = None
        best_similarity = -1.0
        
        for category, examples in self.semantic_categories.items():
            # Calculate average vector for category examples
            vectors = []
            for example in examples:
                example_vector = self.nlp(example).vector
                if not np.all(example_vector == 0):
                    vectors.append(example_vector)
                    
            if not vectors:
                continue
                
            category_vector = np.mean(vectors, axis=0)
            
            # Calculate cosine similarity
            similarity = 1 - cosine(word_vector, category_vector)
            
            if similarity > best_similarity and similarity > 0.4:  # Threshold
                best_similarity = similarity
                best_category = category
        
        if best_category:
            concept.category = best_category
            logger.info(f"Inferred category '{best_category}' for concept '{word}'")
    
    def add_relation(self, source: str, target: str, relation_type: SemanticRelation, 
                    strength: float = 0.5) -> Optional[SemanticLink]:
        """Add a semantic relation between two concepts"""
        source = source.lower().strip()
        target = target.lower().strip()
        
        # Ensure both concepts exist
        if source not in self.concepts:
            self.add_concept(source)
        if target not in self.concepts:
            self.add_concept(target)
        
        # Check if relation already exists
        for link in self.links:
            if link.source == source and link.target == target and link.relation_type == relation_type:
                # Update existing link
                link.update_strength(0.1)
                return link
        
        # Create new link
        link = SemanticLink(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=strength
        )
        
        # Add to network
        self.links.append(link)
        self.graph.add_edge(source, target, relation=relation_type.value, strength=strength, 
                           data=link.model_dump())
        
        logger.info(f"Added semantic relation: '{source}' {relation_type} '{target}'")
        return link
    
    def process_text(self, text: str, context: Optional[str] = None, 
                    emotional_state: Optional[Dict[str, float]] = None) -> None:
        """Process text to extract and update semantic information"""
        if not self.nlp:
            logger.warning("Cannot process text for semantic relations without spaCy")
            return
            
        doc = self.nlp(text)
        
        # Extract nouns as potential concepts
        nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
        
        # Extract adjectives for properties
        adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
        
        # Extract verbs for actions
        verbs = [token.text.lower() for token in doc if token.pos_ == "VERB"]
        
        # Add concepts for nouns
        for noun in nouns:
            self.add_concept(noun, emotions=emotional_state)
        
        # Add properties based on adjective-noun pairs
        for token in doc:
            if token.pos_ == "NOUN":
                noun = token.text.lower()
                
                # Find adjectives modifying this noun
                for child in token.children:
                    if child.pos_ == "ADJ":
                        adj = child.text.lower()
                        
                        # Add property to concept
                        if noun in self.concepts:
                            self.concepts[noun].add_property(adj, 0.7)
                            
                            # Also add has_property relation
                            self.add_relation(noun, adj, SemanticRelation.HAS_PROPERTY)
        
        # Add relations based on syntactic dependencies
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Subject-verb relation (doer of action)
                subject = token.text.lower()
                verb = token.head.text.lower()
                
                if subject in self.concepts and verb in self.concepts:
                    self.add_relation(subject, verb, SemanticRelation.USED_FOR)
            
            elif token.dep_ == "dobj" and token.head.pos_ == "VERB":
                # Verb-object relation (action applied to object)
                verb = token.head.text.lower()
                obj = token.text.lower()
                
                if verb in self.concepts and obj in self.concepts:
                    self.add_relation(verb, obj, SemanticRelation.USED_FOR)
        
        # Extract entities for IS_A relations
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"]:
                entity_text = ent.text.lower()
                entity_type = ent.label_.lower()
                
                # Add concept with category
                self.add_concept(entity_text, category=entity_type)
                
                # Add IS_A relation
                self.add_relation(entity_text, entity_type, SemanticRelation.IS_A)
        
        # Process co-occurrence for associated concepts
        if len(nouns) > 1:
            for i in range(len(nouns)):
                for j in range(i+1, len(nouns)):
                    if nouns[i] != nouns[j]:
                        # Add symmetric association between co-occurring nouns
                        self.add_relation(nouns[i], nouns[j], SemanticRelation.ASSOCIATED, 0.3)
                        self.add_relation(nouns[j], nouns[i], SemanticRelation.ASSOCIATED, 0.3)
    
    def explicitly_teach_relation(self, source: str, relation_type: SemanticRelation, 
                                 target: str, definition: Optional[str] = None) -> None:
        """Explicitly teach a semantic relation to the child"""
        source = source.lower().strip()
        target = target.lower().strip()
        
        # Add both concepts if they don't exist
        if source not in self.concepts:
            self.add_concept(source)
        if target not in self.concepts:
            self.add_concept(target)
        
        # Add relation with higher strength for explicit teaching
        link = self.add_relation(source, target, relation_type, 0.8)
        
        # If definition provided, update source concept properties
        if definition and source in self.concepts:
            if self.nlp:
                # Extract adjectives from definition as properties
                definition_doc = self.nlp(definition)
                adjectives = [token.text.lower() for token in definition_doc if token.pos_ == "ADJ"]
                
                for adj in adjectives:
                    self.concepts[source].add_property(adj, 0.7)
        
        logger.info(f"Explicitly taught relation: '{source}' {relation_type} '{target}'")
    
    def spread_activation(self, seed_concepts: List[str], activation_level: float = 1.0, 
                         decay_factor: float = 0.5, max_depth: int = 2) -> List[str]:
        """Spread activation through the network from seed concepts"""
        if not seed_concepts:
            return []
            
        # Convert to lowercase
        seed_concepts = [c.lower().strip() for c in seed_concepts]
        
        # Filter to concepts that exist in the network
        valid_seeds = [c for c in seed_concepts if c in self.concepts]
        
        if not valid_seeds:
            return []
        
        # Set initial activation for seed concepts
        for concept in valid_seeds:
            self.concepts[concept].update_activation(activation_level)
        
        # Spread activation
        activated_concepts = set(valid_seeds)
        frontier = valid_seeds.copy()
        
        for depth in range(max_depth):
            new_frontier = []
            
            for concept in frontier:
                # Get outgoing edges
                if concept not in self.graph:
                    continue
                    
                for neighbor in self.graph.neighbors(concept):
                    # Skip already activated concepts
                    if neighbor in activated_concepts:
                        continue
                        
                    # Get edge data
                    edge_data = self.graph.get_edge_data(concept, neighbor)
                    if not edge_data:
                        continue
                        
                    # Calculate spreading activation
                    edge_strength = edge_data.get('strength', 0.5)
                    propagated_activation = activation_level * edge_strength * (decay_factor ** depth)
                    
                    # Apply activation if above threshold
                    if propagated_activation > 0.1:
                        self.concepts[neighbor].update_activation(propagated_activation)
                        activated_concepts.add(neighbor)
                        new_frontier.append(neighbor)
            
            frontier = new_frontier
            if not frontier:
                break
        
        # Return list of activated concepts
        return list(activated_concepts)
    
    def find_related_concepts(self, concept: str, relation_type: Optional[SemanticRelation] = None, 
                            min_strength: float = 0.3, max_results: int = 10) -> List[Tuple[str, float]]:
        """Find concepts related to the given concept by specified relation type"""
        concept = concept.lower().strip()
        
        if concept not in self.graph:
            return []
        
        related = []
        
        # Get outgoing edges
        for neighbor in self.graph.neighbors(concept):
            edge_data = self.graph.get_edge_data(concept, neighbor)
            if not edge_data:
                continue
                
            # Check relation type if specified
            if relation_type and edge_data.get('relation') != relation_type.value:
                continue
                
            # Check strength threshold
            strength = edge_data.get('strength', 0.0)
            if strength < min_strength:
                continue
                
            related.append((neighbor, strength))
        
        # Sort by strength and limit results
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_results]
    
    def find_path(self, source: str, target: str, max_length: int = 3) -> List[Tuple[str, str, SemanticRelation]]:
        """Find a path between two concepts"""
        source = source.lower().strip()
        target = target.lower().strip()
        
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            # Use NetworkX to find shortest path
            path = nx.shortest_path(self.graph, source=source, target=target, weight=None)
            
            if len(path) > max_length + 1:
                return []
                
            # Convert path to list of relations
            relations = []
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                edge_data = self.graph.get_edge_data(from_node, to_node)
                relation = edge_data.get('relation', 'associated')
                relations.append((from_node, to_node, relation))
                
            return relations
            
        except nx.NetworkXNoPath:
            return []
    
    def get_concept_neighborhood(self, concept: str, radius: int = 1) -> Dict[str, Any]:
        """Get local neighborhood of a concept for visualization"""
        concept = concept.lower().strip()
        
        if concept not in self.graph:
            return {"nodes": [], "edges": []}
        
        # Extract subgraph
        neighborhood = nx.ego_graph(self.graph, concept, radius=radius, undirected=True)
        
        # Format for visualization
        nodes = []
        for node in neighborhood.nodes():
            concept_data = self.concepts.get(node)
            if concept_data:
                nodes.append({
                    "id": node,
                    "category": concept_data.category or "unknown",
                    "activation": concept_data.activation_level,
                    "age_days": concept_data.get_age_days()
                })
                
        edges = []
        for u, v, data in neighborhood.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": data.get('relation', 'associated'),
                "strength": data.get('strength', 0.5)
            })
                
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def apply_decay(self, days_elapsed: float = 1.0) -> None:
        """Apply natural decay to all concepts and links"""
        # Decay concept activation
        for concept in self.concepts.values():
            concept.activation_level *= 0.9  # Activation decays relatively quickly
        
        # Decay link strength
        for link in self.links:
            link.decay_strength(days_elapsed)
    
    def save_state(self, filename: Optional[str] = None) -> None:
        """Save the state of the semantic network"""
        if not filename:
            filename = f"semantic_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = self.data_dir / filename
        
        # Prepare data for serialization
        state = {
            "concepts": {word: concept.model_dump() for word, concept in self.concepts.items()},
            "links": [link.model_dump() for link in self.links]
        }
        
        # Convert datetime objects to strings
        state_json = json.dumps(state, default=lambda o: o.isoformat() if isinstance(o, datetime) else o)
        
        with open(filepath, 'w') as f:
            f.write(state_json)
            
        logger.info(f"Semantic network saved to {filepath}")
    
    def load_state(self, filepath: Path) -> None:
        """Load the state of the semantic network"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return
            
        with open(filepath, 'r') as f:
            state = json.loads(f.read())
            
        # Restore concepts
        self.concepts = {}
        for word, concept_data in state["concepts"].items():
            # Convert string dates back to datetime
            for date_field in ['first_encountered', 'last_activated']:
                if date_field in concept_data and isinstance(concept_data[date_field], str):
                    concept_data[date_field] = datetime.fromisoformat(concept_data[date_field])
                    
            self.concepts[word] = ConceptNode(**concept_data)
            
        # Restore links
        self.links = []
        for link_data in state["links"]:
            # Convert string dates back to datetime
            for date_field in ['first_formed', 'last_activated']:
                if date_field in link_data and isinstance(link_data[date_field], str):
                    link_data[date_field] = datetime.fromisoformat(link_data[date_field])
                    
            self.links.append(SemanticLink(**link_data))
            
        # Rebuild graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for word, concept in self.concepts.items():
            self.graph.add_node(word, data=concept.model_dump())
            
        # Add edges
        for link in self.links:
            self.graph.add_edge(link.source, link.target, 
                              relation=link.relation_type,
                              strength=link.strength, 
                              data=link.model_dump())
            
        logger.info(f"Semantic network loaded from {filepath}")
        logger.info(f"Loaded {len(self.concepts)} concepts and {len(self.links)} links")