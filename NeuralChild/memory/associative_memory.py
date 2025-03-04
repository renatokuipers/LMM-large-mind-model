# associative_memory.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypeVar, Generic
import logging
import random
import json
import os
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum, auto
import networkx as nx

# Import from memory manager and long-term memory
from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority
from memory.long_term_memory import LongTermMemory, KnowledgeDomain

# Set up logging
logger = logging.getLogger("AssociativeMemory")

class AssociationType(str, Enum):
    """Types of associations between concepts"""
    SEMANTIC = "semantic"       # Related by meaning
    TEMPORAL = "temporal"       # Related by time of occurrence
    CAUSAL = "causal"           # Cause and effect relationships
    SIMILARITY = "similarity"   # Similar concepts
    CONTRAST = "contrast"       # Contrasting/opposite concepts
    HIERARCHICAL = "hierarchical"  # Is-a, part-of relationships
    EMOTIONAL = "emotional"     # Linked by emotional response

class AssociationStrength(Enum):
    """Strength of associations"""
    WEAK = auto()       # Barely linked
    MODERATE = auto()   # Moderately linked
    STRONG = auto()     # Strongly linked
    INTRINSIC = auto()  # Fundamentally linked, inseparable

class AssociationLink(BaseModel):
    """A link between two concepts in associative memory"""
    source_id: str = Field(..., description="Source memory ID")
    target_id: str = Field(..., description="Target memory ID")
    association_type: AssociationType = Field(..., description="Type of association")
    strength: AssociationStrength = Field(AssociationStrength.MODERATE, description="Strength of association")
    strength_value: float = Field(0.5, ge=0.0, le=1.0, description="Numeric strength value")
    bidirectional: bool = Field(False, description="Whether the association works both ways")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activated: datetime = Field(default_factory=datetime.now)
    activation_count: int = Field(0, ge=0)
    context: Optional[str] = Field(None, description="Context where this association was formed")
    emotional_valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional tone of association")
    
    def activate(self) -> None:
        """Activate this association"""
        self.activation_count += 1
        self.last_activated = datetime.now()
    
    def strengthen(self, amount: float = 0.1) -> None:
        """Strengthen the association"""
        self.strength_value = min(1.0, self.strength_value + amount)
        
        # Update strength category
        if self.strength_value > 0.9:
            self.strength = AssociationStrength.INTRINSIC
        elif self.strength_value > 0.7:
            self.strength = AssociationStrength.STRONG
        elif self.strength_value > 0.4:
            self.strength = AssociationStrength.MODERATE
        else:
            self.strength = AssociationStrength.WEAK
            
        self.activate()
    
    def weaken(self, amount: float = 0.1) -> None:
        """Weaken the association"""
        self.strength_value = max(0.0, self.strength_value - amount)
        
        # Update strength category
        if self.strength_value < 0.2:
            self.strength = AssociationStrength.WEAK
        elif self.strength_value < 0.5:
            self.strength = AssociationStrength.MODERATE
        elif self.strength_value < 0.8:
            self.strength = AssociationStrength.STRONG
        else:
            self.strength = AssociationStrength.INTRINSIC
    
    def apply_decay(self, days_elapsed: float, rate: float = 0.01) -> None:
        """Apply decay based on time elapsed"""
        # Skip for intrinsic associations
        if self.strength == AssociationStrength.INTRINSIC:
            return
            
        # Calculate decay based on strength - stronger associations decay more slowly
        if self.strength == AssociationStrength.STRONG:
            decay_factor = rate * 0.5 * days_elapsed
        elif self.strength == AssociationStrength.MODERATE:
            decay_factor = rate * days_elapsed
        else:  # WEAK
            decay_factor = rate * 2.0 * days_elapsed
            
        # Cap maximum decay
        decay_factor = min(0.5, decay_factor)
        
        # Apply decay
        self.strength_value *= (1.0 - decay_factor)
        
        # Update strength category
        if self.strength_value < 0.2:
            self.strength = AssociationStrength.WEAK
        elif self.strength_value < 0.5:
            self.strength = AssociationStrength.MODERATE
        elif self.strength_value < 0.8:
            self.strength = AssociationStrength.STRONG

class ConceptNode(BaseModel):
    """A concept in the associative network"""
    memory_id: str = Field(..., description="ID of the memory item this represents")
    activation_level: float = Field(0.0, ge=0.0, le=1.0, description="Current activation level")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance of this concept")
    associations: Dict[str, str] = Field(default_factory=dict, description="Association IDs by target ID")
    centrality: float = Field(0.0, ge=0.0, description="Centrality in the network")
    categories: List[str] = Field(default_factory=list, description="Categories this concept belongs to")
    last_activated: datetime = Field(default_factory=datetime.now)
    
    def update_activation(self, amount: float) -> None:
        """Update activation level"""
        self.activation_level = min(1.0, self.activation_level + amount)
        self.last_activated = datetime.now()
    
    def decay_activation(self, rate: float = 0.1) -> None:
        """Apply decay to activation level"""
        self.activation_level *= (1.0 - rate)

class AssociativeMemory:
    """Connection-making system for concepts"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize associative memory
        
        Args:
            data_dir: Directory for persistent storage
        """
        self.data_dir = data_dir or Path("./data/memory/associative")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.concepts: Dict[str, ConceptNode] = {}  # memory_id -> ConceptNode
        self.associations: Dict[str, AssociationLink] = {}  # association_id -> AssociationLink
        
        # Indices for faster access
        self.source_index: Dict[str, Set[str]] = {}  # source_id -> association_ids
        self.target_index: Dict[str, Set[str]] = {}  # target_id -> association_ids
        self.type_index: Dict[AssociationType, Set[str]] = {  # type -> association_ids
            assoc_type: set() for assoc_type in AssociationType
        }
        
        # Network graph for analysis
        self.graph = nx.DiGraph()
        
        # References to other memory systems
        self.memory_manager: Optional[MemoryManager] = None
        self.long_term_memory: Optional[LongTermMemory] = None
        
        # Spreading activation parameters
        self.activation_decay = 0.3  # How quickly activation decays with distance
        self.activation_threshold = 0.2  # Minimum activation to spread
        
        logger.info("Associative memory initialized")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def set_long_term_memory(self, long_term_memory: LongTermMemory) -> None:
        """Set the long-term memory reference"""
        self.long_term_memory = long_term_memory
    
    def add_association(self, memory_item: MemoryItem) -> str:
        """Add a new association to memory"""
        memory_id = memory_item.id
        
        # Check if we already have an association for this memory
        if memory_id in self.associations:
            return memory_id
        
        # Extract source and target from memory content
        # This assumes the memory content has a specific format
        # In a real implementation, this would use more sophisticated extraction
        content = memory_item.content
        source_id = None
        target_id = None
        association_type = AssociationType.SEMANTIC
        bidirectional = False
        
        if isinstance(content, dict):
            source_id = content.get("source_id")
            target_id = content.get("target_id")
            association_type_str = content.get("type", "semantic")
            try:
                association_type = AssociationType(association_type_str)
            except ValueError:
                association_type = AssociationType.SEMANTIC
            
            bidirectional = content.get("bidirectional", False)
            context = content.get("context")
            emotional_valence = content.get("emotional_valence", 0.0)
        else:
            # No structured content, can't create association
            logger.warning(f"Cannot create association from unstructured content: {memory_id}")
            return memory_id
        
        # Skip if source or target is missing
        if not source_id or not target_id:
            logger.warning(f"Missing source or target for association: {memory_id}")
            return memory_id
        
        # Create the association
        strength_value = memory_item.attributes.salience
        
        # Determine strength category
        if strength_value > 0.9:
            strength = AssociationStrength.INTRINSIC
        elif strength_value > 0.7:
            strength = AssociationStrength.STRONG
        elif strength_value > 0.4:
            strength = AssociationStrength.MODERATE
        else:
            strength = AssociationStrength.WEAK
        
        association = AssociationLink(
            source_id=source_id,
            target_id=target_id,
            association_type=association_type,
            strength=strength,
            strength_value=strength_value,
            bidirectional=bidirectional,
            context=context if 'context' in locals() else None,
            emotional_valence=emotional_valence if 'emotional_valence' in locals() else 0.0
        )
        
        # Store the association
        self.associations[memory_id] = association
        
        # Update indices
        if source_id not in self.source_index:
            self.source_index[source_id] = set()
        self.source_index[source_id].add(memory_id)
        
        if target_id not in self.target_index:
            self.target_index[target_id] = set()
        self.target_index[target_id].add(memory_id)
        
        self.type_index[association_type].add(memory_id)
        
        # Create or update nodes in the network
        self._ensure_concept_exists(source_id)
        self._ensure_concept_exists(target_id)
        
        # Update the concept's associations
        self.concepts[source_id].associations[target_id] = memory_id
        
        # Add to graph
        self.graph.add_edge(
            source_id, 
            target_id, 
            id=memory_id, 
            type=association_type.value,
            strength=strength_value,
            bidirectional=bidirectional
        )
        
        # If bidirectional, add reverse link in graph
        if bidirectional:
            self.graph.add_edge(
                target_id,
                source_id,
                id=f"{memory_id}_reverse",
                type=association_type.value,
                strength=strength_value * 0.9,  # Slightly weaker in reverse
                bidirectional=True
            )
        
        logger.info(f"Added association {memory_id} from {source_id} to {target_id} ({association_type.value})")
        return memory_id
    
    def _ensure_concept_exists(self, memory_id: str) -> None:
        """Ensure a concept exists in the network"""
        if memory_id in self.concepts:
            return
        
        # Create a new concept node
        importance = 0.5  # Default importance
        
        # If long-term memory is available, get importance from there
        if self.long_term_memory and memory_id in self.long_term_memory.memories:
            importance = self.long_term_memory.memories[memory_id].importance
        
        concept = ConceptNode(
            memory_id=memory_id,
            importance=importance
        )
        
        self.concepts[memory_id] = concept
        self.graph.add_node(memory_id, importance=importance)
    
    def create_association(self, source_id: str, target_id: str, strength: float = 0.5,
                          association_type: AssociationType = AssociationType.SEMANTIC,
                          bidirectional: bool = False, context: Optional[str] = None) -> Optional[str]:
        """Create a new association between concepts"""
        # Skip if either concept doesn't exist in memory manager
        if not self.memory_manager:
            logger.warning("Cannot create association: no memory manager")
            return None
        
        if self.memory_manager.retrieve(source_id) is None:
            logger.warning(f"Cannot create association: source {source_id} not found")
            return None
            
        if self.memory_manager.retrieve(target_id) is None:
            logger.warning(f"Cannot create association: target {target_id} not found")
            return None
        
        # Check if an association already exists
        if source_id in self.source_index:
            for assoc_id in self.source_index[source_id]:
                assoc = self.associations[assoc_id]
                if assoc.target_id == target_id and assoc.association_type == association_type:
                    # Association already exists, strengthen it
                    assoc.strengthen(0.1)
                    return assoc_id
        
        # Create association content
        association_content = {
            "source_id": source_id,
            "target_id": target_id,
            "type": association_type.value,
            "bidirectional": bidirectional,
            "context": context
        }
        
        # Create memory attributes
        attributes = MemoryAttributes(
            salience=strength,
            confidence=0.9,  # High confidence for direct associations
        )
        
        # Create a memory item for this association
        memory_id = f"assoc_{source_id}_{target_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if self.memory_manager:
            # Store the association in memory manager
            memory_id = self.memory_manager.store(
                content=association_content,
                memory_type=MemoryType.ASSOCIATIVE,
                tags=["association", association_type.value],
                emotional_valence=0.0,
                emotional_intensity=0.0,
                salience=strength
            )
            
            # Add the association to our own tracking
            memory_item = self.memory_manager.retrieve(memory_id)
            if memory_item:
                return self.add_association(memory_item)
        
        logger.warning("Failed to create association in memory manager")
        return None
    
    def get_associations(self, memory_id: str, direction: str = "outgoing",
                        min_strength: float = 0.0) -> List[str]:
        """Get associations for a memory
        
        Args:
            memory_id: ID of the memory to get associations for
            direction: 'outgoing', 'incoming', or 'both'
            min_strength: Minimum strength to include
            
        Returns:
            List of association IDs
        """
        result = []
        
        # Get outgoing associations
        if direction in ["outgoing", "both"]:
            if memory_id in self.source_index:
                for assoc_id in self.source_index[memory_id]:
                    if self.associations[assoc_id].strength_value >= min_strength:
                        result.append(assoc_id)
        
        # Get incoming associations
        if direction in ["incoming", "both"]:
            if memory_id in self.target_index:
                for assoc_id in self.target_index[memory_id]:
                    if self.associations[assoc_id].strength_value >= min_strength:
                        result.append(assoc_id)
        
        return result
    
    def get_associated_concepts(self, memory_id: str, direction: str = "outgoing",
                              min_strength: float = 0.0) -> List[str]:
        """Get concepts associated with a memory
        
        Args:
            memory_id: ID of the memory to get associations for
            direction: 'outgoing', 'incoming', or 'both'
            min_strength: Minimum strength to include
            
        Returns:
            List of associated memory IDs
        """
        result = []
        
        # Get outgoing connections
        if direction in ["outgoing", "both"]:
            if memory_id in self.source_index:
                for assoc_id in self.source_index[memory_id]:
                    assoc = self.associations[assoc_id]
                    if assoc.strength_value >= min_strength:
                        result.append(assoc.target_id)
        
        # Get incoming connections
        if direction in ["incoming", "both"]:
            if memory_id in self.target_index:
                for assoc_id in self.target_index[memory_id]:
                    assoc = self.associations[assoc_id]
                    if assoc.strength_value >= min_strength:
                        result.append(assoc.source_id)
        
        return result
    
    def activate_association(self, association_id: str) -> None:
        """Activate an association"""
        if association_id not in self.associations:
            return
        
        # Get the association
        association = self.associations[association_id]
        
        # Update activation count and timestamp
        association.activate()
        
        # Strengthen the association slightly with use
        association.strengthen(0.05)
        
        # Activate the connected concepts
        if association.source_id in self.concepts:
            self.concepts[association.source_id].update_activation(0.3)
        
        if association.target_id in self.concepts:
            self.concepts[association.target_id].update_activation(0.3)
    
    def update_association(self, association_id: str, memory_item: Optional[MemoryItem] = None) -> bool:
        """Update an association"""
        if association_id not in self.associations:
            return False
        
        # Get the association
        association = self.associations[association_id]
        
        # Update activation count and timestamp
        association.activate()
        
        # Update from memory item if provided
        if memory_item:
            content = memory_item.content
            
            if isinstance(content, dict):
                # Update context if provided
                if "context" in content:
                    association.context = content["context"]
                
                # Update emotional valence if provided
                if "emotional_valence" in content:
                    association.emotional_valence = content["emotional_valence"]
                
                # Update bidirectional flag if provided
                if "bidirectional" in content:
                    old_bidirectional = association.bidirectional
                    new_bidirectional = content["bidirectional"]
                    
                    if old_bidirectional != new_bidirectional:
                        association.bidirectional = new_bidirectional
                        
                        # Update graph
                        if new_bidirectional:
                            # Add reverse edge
                            self.graph.add_edge(
                                association.target_id,
                                association.source_id,
                                id=f"{association_id}_reverse",
                                type=association.association_type.value,
                                strength=association.strength_value * 0.9,
                                bidirectional=True
                            )
                        else:
                            # Remove reverse edge
                            if self.graph.has_edge(association.target_id, association.source_id):
                                self.graph.remove_edge(association.target_id, association.source_id)
            
            # Update strength based on memory salience
            strength_value = memory_item.attributes.salience
            if abs(association.strength_value - strength_value) > 0.1:
                association.strength_value = strength_value
                
                # Update strength category
                if strength_value > 0.9:
                    association.strength = AssociationStrength.INTRINSIC
                elif strength_value > 0.7:
                    association.strength = AssociationStrength.STRONG
                elif strength_value > 0.4:
                    association.strength = AssociationStrength.MODERATE
                else:
                    association.strength = AssociationStrength.WEAK
                
                # Update graph
                if self.graph.has_edge(association.source_id, association.target_id):
                    self.graph[association.source_id][association.target_id]["strength"] = strength_value
                
                if association.bidirectional and self.graph.has_edge(association.target_id, association.source_id):
                    self.graph[association.target_id][association.source_id]["strength"] = strength_value * 0.9
        
        return True
    
    def remove_association(self, association_id: str) -> bool:
        """Remove an association"""
        if association_id not in self.associations:
            return False
        
        # Get the association
        association = self.associations[association_id]
        
        # Remove from indices
        if association.source_id in self.source_index:
            self.source_index[association.source_id].discard(association_id)
        
        if association.target_id in self.target_index:
            self.target_index[association.target_id].discard(association_id)
        
        self.type_index[association.association_type].discard(association_id)
        
        # Remove from concepts
        if association.source_id in self.concepts and association.target_id in self.concepts[association.source_id].associations:
            del self.concepts[association.source_id].associations[association.target_id]
        
        # Remove from graph
        if self.graph.has_edge(association.source_id, association.target_id):
            self.graph.remove_edge(association.source_id, association.target_id)
        
        # Remove reverse edge if bidirectional
        if association.bidirectional and self.graph.has_edge(association.target_id, association.source_id):
            self.graph.remove_edge(association.target_id, association.source_id)
        
        # Remove from associations
        del self.associations[association_id]
        
        logger.info(f"Removed association {association_id}")
        return True
    
    def spread_activation(self, seed_concepts: List[str], activation_value: float = 1.0,
                         max_depth: int = 2) -> Dict[str, float]:
        """Spread activation from seed concepts through the network
        
        Args:
            seed_concepts: List of concept IDs to start activation from
            activation_value: Initial activation value
            max_depth: Maximum steps to spread activation
            
        Returns:
            Dictionary of memory_id -> activation_level
        """
        # Reset all concept activations
        for concept in self.concepts.values():
            concept.activation_level = 0.0
        
        # Initialize active concepts with seeds
        active_concepts = {}
        for concept_id in seed_concepts:
            if concept_id in self.concepts:
                self.concepts[concept_id].update_activation(activation_value)
                active_concepts[concept_id] = activation_value
        
        # Spread activation for each depth level
        for depth in range(max_depth):
            # Calculate decay for this level
            level_decay = self.activation_decay ** (depth + 1)
            
            # Process current active concepts
            next_active = {}
            for concept_id, activation in active_concepts.items():
                # Skip if below threshold
                if activation < self.activation_threshold:
                    continue
                
                # Get outgoing associations
                if concept_id in self.source_index:
                    for assoc_id in self.source_index[concept_id]:
                        assoc = self.associations[assoc_id]
                        target_id = assoc.target_id
                        
                        # Calculate propagated activation
                        propagated = activation * assoc.strength_value * level_decay
                        
                        # Update target concept
                        if target_id in self.concepts:
                            self.concepts[target_id].update_activation(propagated)
                            
                            # Add to next active set
                            if target_id in next_active:
                                next_active[target_id] += propagated
                            else:
                                next_active[target_id] = propagated
                
                # Also spread through incoming if bidirectional
                if concept_id in self.target_index:
                    for assoc_id in self.target_index[concept_id]:
                        assoc = self.associations[assoc_id]
                        
                        # Only follow bidirectional links backward
                        if not assoc.bidirectional:
                            continue
                            
                        source_id = assoc.source_id
                        
                        # Calculate propagated activation (weaker in reverse)
                        propagated = activation * assoc.strength_value * level_decay * 0.8
                        
                        # Update source concept
                        if source_id in self.concepts:
                            self.concepts[source_id].update_activation(propagated)
                            
                            # Add to next active set
                            if source_id in next_active:
                                next_active[source_id] += propagated
                            else:
                                next_active[source_id] = propagated
            
            # Update active concepts for next iteration
            active_concepts = next_active
            
            # Stop if no active concepts left
            if not active_concepts:
                break
        
        # Collect results
        results = {}
        for concept_id, concept in self.concepts.items():
            if concept.activation_level > 0:
                results[concept_id] = concept.activation_level
        
        return results
    
    def find_associations_by_type(self, assoc_type: AssociationType,
                                min_strength: float = 0.3,
                                max_results: int = 50) -> List[str]:
        """Find associations of a specific type"""
        if assoc_type not in self.type_index:
            return []
        
        # Get all associations of this type
        assoc_ids = list(self.type_index[assoc_type])
        
        # Filter by strength
        filtered = [assoc_id for assoc_id in assoc_ids
                   if self.associations[assoc_id].strength_value >= min_strength]
        
        # Sort by strength (descending)
        sorted_ids = sorted(filtered, 
                          key=lambda aid: self.associations[aid].strength_value,
                          reverse=True)
        
        # Return up to max_results
        return sorted_ids[:max_results]
    
    def find_path(self, start_id: str, end_id: str, max_length: int = 5) -> List[str]:
        """Find a path between two concepts
        
        Args:
            start_id: Starting concept ID
            end_id: Ending concept ID
            max_length: Maximum path length
            
        Returns:
            List of association IDs in the path, or empty list if no path found
        """
        if not self.graph.has_node(start_id) or not self.graph.has_node(end_id):
            return []
        
        try:
            # Find shortest path in graph
            path_nodes = nx.shortest_path(self.graph, start_id, end_id, weight=None)
            
            # Check if path is too long
            if len(path_nodes) > max_length + 1:
                return []
            
            # Convert to list of association IDs
            path_associations = []
            for i in range(len(path_nodes) - 1):
                source = path_nodes[i]
                target = path_nodes[i+1]
                
                # Find the association ID for this edge
                edge_data = self.graph.get_edge_data(source, target)
                if edge_data and "id" in edge_data:
                    assoc_id = edge_data["id"]
                    if assoc_id in self.associations:
                        path_associations.append(assoc_id)
            
            return path_associations
            
        except nx.NetworkXNoPath:
            return []
    
    def find_unexpected_associations(self, num_results: int = 10) -> List[str]:
        """Find unexpected or creative associations
        
        This looks for associations between concepts in different domains
        or categories that have moderate strength.
        
        Returns:
            List of association IDs
        """
        candidates = []
        
        # Look for associations with emotional valence
        for assoc_id, assoc in self.associations.items():
            if abs(assoc.emotional_valence) > 0.3 and assoc.strength_value > 0.3:
                unexpected_score = abs(assoc.emotional_valence) * assoc.strength_value
                candidates.append((assoc_id, unexpected_score))
        
        # Look for cross-domain associations using long-term memory
        if self.long_term_memory:
            for assoc_id, assoc in self.associations.items():
                source_id = assoc.source_id
                target_id = assoc.target_id
                
                source_domain = None
                target_domain = None
                
                # Get domains if available
                if source_id in self.long_term_memory.memories:
                    source_domain = self.long_term_memory.memories[source_id].domain
                
                if target_id in self.long_term_memory.memories:
                    target_domain = self.long_term_memory.memories[target_id].domain
                
                # Check if cross-domain and has meaningful strength
                if (source_domain and target_domain and source_domain != target_domain and
                    assoc.strength_value > 0.4 and assoc.strength_value < 0.8):
                    unexpected_score = assoc.strength_value * 0.8
                    candidates.append((assoc_id, unexpected_score))
        
        # Look for temporal associations with unusual patterns
        temporal_assocs = self.find_associations_by_type(AssociationType.TEMPORAL)
        for assoc_id in temporal_assocs:
            assoc = self.associations[assoc_id]
            
            # Check for unusual temporal associations with emotional content
            if abs(assoc.emotional_valence) > 0.5 and assoc.strength_value > 0.3:
                unexpected_score = abs(assoc.emotional_valence) * assoc.strength_value * 0.7
                candidates.append((assoc_id, unexpected_score))
        
        # Sort by unexpectedness score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return [assoc_id for assoc_id, _ in candidates[:num_results]]
    
    def analyze_concept_centrality(self) -> Dict[str, float]:
        """Analyze centrality of concepts in the network
        
        Returns:
            Dictionary mapping concept IDs to centrality scores
        """
        # Use eigenvector centrality for importance in the network
        try:
            centrality = nx.eigenvector_centrality(self.graph, weight="strength")
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fall back to degree centrality if eigenvector centrality fails
            centrality = nx.degree_centrality(self.graph)
        
        # Update concept centrality values
        for concept_id, score in centrality.items():
            if concept_id in self.concepts:
                self.concepts[concept_id].centrality = score
        
        return centrality
    
    def get_central_concepts(self, top_n: int = 10) -> List[str]:
        """Get the most central concepts in the network"""
        # Analyze centrality if needed
        if all(concept.centrality == 0.0 for concept in self.concepts.values()):
            self.analyze_concept_centrality()
        
        # Sort concepts by centrality
        sorted_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1].centrality,
            reverse=True
        )
        
        # Return top N
        return [concept_id for concept_id, _ in sorted_concepts[:top_n]]
    
    def apply_decay(self, days_elapsed: float = 1.0) -> List[str]:
        """Apply decay to all associations
        
        Returns:
            List of associations that have decayed below threshold
        """
        decayed_below_threshold = []
        
        for assoc_id, assoc in list(self.associations.items()):
            # Apply decay
            assoc.apply_decay(days_elapsed)
            
            # Check if decayed below threshold
            if assoc.strength_value < 0.1:
                decayed_below_threshold.append(assoc_id)
        
        return decayed_below_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the associative memory"""
        # Count associations by type
        type_counts = {assoc_type.value: len(assoc_ids) 
                     for assoc_type, assoc_ids in self.type_index.items()}
        
        # Count concepts and associations
        concept_count = len(self.concepts)
        association_count = len(self.associations)
        
        # Calculate average strength
        if self.associations:
            avg_strength = sum(assoc.strength_value for assoc in self.associations.values()) / len(self.associations)
        else:
            avg_strength = 0.0
        
        # Calculate network density
        if concept_count > 1:
            possible_connections = concept_count * (concept_count - 1)
            actual_connections = self.graph.number_of_edges()
            density = actual_connections / possible_connections
        else:
            density = 0.0
        
        # Find most connected concepts
        if self.concepts:
            most_connected = max(self.concepts.values(), 
                               key=lambda c: len(c.associations))
            most_connected_id = most_connected.memory_id
            most_connected_count = len(most_connected.associations)
        else:
            most_connected_id = None
            most_connected_count = 0
        
        return {
            "concept_count": concept_count,
            "association_count": association_count,
            "associations_by_type": type_counts,
            "avg_association_strength": avg_strength,
            "network_density": density,
            "most_connected_concept": most_connected_id,
            "most_connected_count": most_connected_count
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the state of associative memory to disk"""
        if filepath is None:
            filepath = self.data_dir / f"associative_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        concepts_data = {}
        for concept_id, concept in self.concepts.items():
            concepts_data[concept_id] = {
                "memory_id": concept.memory_id,
                "activation_level": concept.activation_level,
                "importance": concept.importance,
                "associations": concept.associations,
                "centrality": concept.centrality,
                "categories": concept.categories,
                "last_activated": concept.last_activated.isoformat()
            }
        
        associations_data = {}
        for assoc_id, assoc in self.associations.items():
            associations_data[assoc_id] = {
                "source_id": assoc.source_id,
                "target_id": assoc.target_id,
                "association_type": assoc.association_type.value,
                "strength": assoc.strength.name,
                "strength_value": assoc.strength_value,
                "bidirectional": assoc.bidirectional,
                "created_at": assoc.created_at.isoformat(),
                "last_activated": assoc.last_activated.isoformat(),
                "activation_count": assoc.activation_count,
                "context": assoc.context,
                "emotional_valence": assoc.emotional_valence
            }
        
        # Prepare indices for serialization
        source_index_data = {source_id: list(assoc_ids) 
                           for source_id, assoc_ids in self.source_index.items()}
        
        target_index_data = {target_id: list(assoc_ids)
                           for target_id, assoc_ids in self.target_index.items()}
        
        type_index_data = {assoc_type.value: list(assoc_ids)
                         for assoc_type, assoc_ids in self.type_index.items()}
        
        state = {
            "concepts": concepts_data,
            "associations": associations_data,
            "source_index": source_index_data,
            "target_index": target_index_data,
            "type_index": type_index_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "concept_count": len(self.concepts),
                "association_count": len(self.associations)
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Associative memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of associative memory from disk"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.concepts.clear()
            self.associations.clear()
            self.source_index.clear()
            self.target_index.clear()
            self.type_index = {assoc_type: set() for assoc_type in AssociationType}
            self.graph = nx.DiGraph()
            
            # Load concepts
            for concept_id, concept_data in state.get("concepts", {}).items():
                self.concepts[concept_id] = ConceptNode(
                    memory_id=concept_data["memory_id"],
                    activation_level=concept_data["activation_level"],
                    importance=concept_data["importance"],
                    associations=concept_data["associations"],
                    centrality=concept_data["centrality"],
                    categories=concept_data["categories"],
                    last_activated=datetime.fromisoformat(concept_data["last_activated"])
                )
                
                # Add to graph
                self.graph.add_node(concept_id, importance=concept_data["importance"])
            
            # Load associations
            for assoc_id, assoc_data in state.get("associations", {}).items():
                self.associations[assoc_id] = AssociationLink(
                    source_id=assoc_data["source_id"],
                    target_id=assoc_data["target_id"],
                    association_type=AssociationType(assoc_data["association_type"]),
                    strength=AssociationStrength[assoc_data["strength"]],
                    strength_value=assoc_data["strength_value"],
                    bidirectional=assoc_data["bidirectional"],
                    created_at=datetime.fromisoformat(assoc_data["created_at"]),
                    last_activated=datetime.fromisoformat(assoc_data["last_activated"]),
                    activation_count=assoc_data["activation_count"],
                    context=assoc_data["context"],
                    emotional_valence=assoc_data["emotional_valence"]
                )
                
                # Add to graph
                self.graph.add_edge(
                    assoc_data["source_id"],
                    assoc_data["target_id"],
                    id=assoc_id,
                    type=assoc_data["association_type"],
                    strength=assoc_data["strength_value"],
                    bidirectional=assoc_data["bidirectional"]
                )
                
                # Add reverse edge if bidirectional
                if assoc_data["bidirectional"]:
                    self.graph.add_edge(
                        assoc_data["target_id"],
                        assoc_data["source_id"],
                        id=f"{assoc_id}_reverse",
                        type=assoc_data["association_type"],
                        strength=assoc_data["strength_value"] * 0.9,
                        bidirectional=True
                    )
            
            # Load indices
            for source_id, assoc_ids in state.get("source_index", {}).items():
                self.source_index[source_id] = set(assoc_ids)
            
            for target_id, assoc_ids in state.get("target_index", {}).items():
                self.target_index[target_id] = set(assoc_ids)
            
            for type_str, assoc_ids in state.get("type_index", {}).items():
                self.type_index[AssociationType(type_str)] = set(assoc_ids)
            
            logger.info(f"Loaded associative memory state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading associative memory state: {str(e)}")
            return False