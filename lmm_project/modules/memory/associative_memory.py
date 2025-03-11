import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime
import time
import numpy as np
from collections import defaultdict, Counter

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.vector_store import get_embeddings, VectorStore
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType
from lmm_project.neural_substrate.neural_network import NeuralNetwork, NetworkType
from lmm_project.neural_substrate.neural_cluster import ClusterType
from lmm_project.neural_substrate.hebbian_learning import HebbianLearner, HebbianRule

from .models import (
    MemoryType,
    MemoryStatus,
    AssociativeLink,
    MemoryEvent,
    MemoryQuery,
    MemoryConfig
)

# Initialize logger
logger = get_module_logger("modules.memory.associative_memory")

class AssociativeMemory:
    """
    Manages associations between memory items.
    
    The associative memory system creates, strengthens, weakens, and retrieves
    connections between memory items based on co-activation, semantic similarity,
    and explicit associations. It supports various types of associations and
    implements Hebbian-like learning for connection reinforcement.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[MemoryConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the associative memory system.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the memory system
            developmental_age: Current developmental age of the mind
        """
        self._config = config or MemoryConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Storage for associations
        self._links: Dict[UUID, AssociativeLink] = {}
        
        # Index for fast lookup of associations by source and target
        self._source_index: Dict[UUID, Set[UUID]] = defaultdict(set)
        self._target_index: Dict[UUID, Set[UUID]] = defaultdict(set)
        self._type_index: Dict[str, Set[UUID]] = defaultdict(set)
        
        # Recent associations for tracking
        self._recent_associations = []
        
        # Association type statistics
        self._association_type_counts = Counter()
        
        # Neural network for association learning
        self._neural_network = None
        self._initialize_neural_network()
        
        # Hebbian learner for developing associations
        self._hebbian_learner = self._initialize_hebbian_learner()
        
        # Co-activation tracking for automatic association formation
        self._activation_history: Dict[UUID, List[datetime]] = defaultdict(list)
        self._co_activation_threshold = max(0.3, self._config.min_association_strength)
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Associative memory initialized with age {developmental_age}")
    
    def _initialize_neural_network(self) -> None:
        """Initialize neural network for association processing if enabled"""
        if not self._config.use_neural_networks:
            logger.info("Neural networks disabled for associative memory")
            return
            
        try:
            # Create a neural network for associative memory
            self._neural_network = NeuralNetwork(
                network_id="associative_memory_network",
                config={
                    "network_type": NetworkType.MODULAR,
                    "input_size": self._config.embedding_dimension * 2,  # For pairs of items
                    "output_size": 8,  # Association strength and type predictions
                    "hidden_layers": [self._config.embedding_dimension, 64],
                    "learning_rate": self._config.network_learn_rate,
                    "plasticity_enabled": True,
                    "cluster_sizes": [
                        self._config.embedding_dimension, 
                        self._config.embedding_dimension // 2,
                        64
                    ],
                    "cluster_types": [
                        ClusterType.FEED_FORWARD, 
                        ClusterType.RECURRENT,
                        ClusterType.COMPETITIVE
                    ]
                }
            )
            logger.info("Neural network initialized for associative memory processing")
        except Exception as e:
            logger.error(f"Failed to initialize neural network: {e}")
            self._neural_network = None
    
    def _initialize_hebbian_learner(self) -> Optional[HebbianLearner]:
        """Initialize Hebbian learning for association reinforcement"""
        try:
            hebbian_learner = HebbianLearner(
                learning_rate=max(0.01, 0.05 * (1 + self._developmental_age * 0.5)),
                rule=HebbianRule.BASIC
            )
            logger.info("Hebbian learner initialized for associative memory")
            return hebbian_learner
        except Exception as e:
            logger.error(f"Failed to initialize Hebbian learner: {e}")
            return None
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("create_association", self._handle_create_association)
        self._event_bus.subscribe("reinforce_association", self._handle_reinforce_association)
        self._event_bus.subscribe("memory_accessed", self._handle_memory_access)
        self._event_bus.subscribe("get_associations", self._handle_association_query)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _handle_create_association(self, event: Message) -> None:
        """
        Handle a create association event.
        
        Args:
            event: The event containing association data
        """
        try:
            association_data = event.data
            
            # Extract association details
            source_id = association_data.get("source_id")
            target_id = association_data.get("target_id")
            association_type = association_data.get("association_type", "generic")
            strength = association_data.get("strength", 0.5)
            bidirectional = association_data.get("bidirectional", False)
            features = association_data.get("features", {})
            
            # Validate IDs
            if not (source_id and target_id):
                logger.warning("Create association event missing source_id or target_id")
                return
                
            # Convert string IDs to UUIDs if needed
            if isinstance(source_id, str):
                source_id = UUID(source_id)
            if isinstance(target_id, str):
                target_id = UUID(target_id)
                
            # Create the association
            link = self.create_link(
                source_id=source_id,
                target_id=target_id,
                association_type=association_type,
                strength=strength,
                bidirectional=bidirectional,
                features=features
            )
            
            if link:
                # Publish response
                self._publish_memory_event("association_created", {
                    "link_id": str(link.id),
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "association_type": association_type,
                    "strength": link.strength,
                    "request_id": event.data.get("request_id")
                })
        except Exception as e:
            logger.error(f"Error handling create association event: {e}")
    
    def _handle_reinforce_association(self, event: Message) -> None:
        """
        Handle a reinforce association event.
        
        Args:
            event: The event containing association reinforcement data
        """
        try:
            # Check if we're reinforcing by ID or by source/target pair
            link_id = event.data.get("link_id")
            source_id = event.data.get("source_id")
            target_id = event.data.get("target_id")
            amount = event.data.get("amount", 0.1)
            
            # Apply developmental effects to reinforcement amount
            # Younger minds have more rapid strengthening
            development_factor = max(1.0, 1.5 - self._developmental_age * 0.5)
            adjusted_amount = amount * development_factor
            
            result = False
            
            if link_id:
                # Reinforce by link ID
                if isinstance(link_id, str):
                    link_id = UUID(link_id)
                result = self.reinforce_link(link_id, adjusted_amount)
            elif source_id and target_id:
                # Reinforce by source/target pair
                if isinstance(source_id, str):
                    source_id = UUID(source_id)
                if isinstance(target_id, str):
                    target_id = UUID(target_id)
                result = self.reinforce_link_between(source_id, target_id, adjusted_amount)
            else:
                logger.warning("Reinforce association event missing link_id or source_id/target_id pair")
                return
                
            # Publish response
            self._publish_memory_event("association_reinforced", {
                "success": result,
                "request_id": event.data.get("request_id"),
                "amount": adjusted_amount
            })
        except Exception as e:
            logger.error(f"Error handling reinforce association event: {e}")
    
    def _handle_memory_access(self, event: Message) -> None:
        """
        Handle a memory access event for auto-association tracking.
        
        Args:
            event: The event containing memory access data
        """
        try:
            memory_id = event.data.get("memory_id")
            if not memory_id:
                return
                
            # Convert string ID to UUID if needed
            if isinstance(memory_id, str):
                memory_id = UUID(memory_id)
                
            # Record activation for co-activation tracking
            current_time = datetime.now()
            self._activation_history[memory_id].append(current_time)
            
            # Keep only recent activations (last 60 seconds)
            cutoff_time = current_time - datetime.timedelta(seconds=60)
            self._activation_history[memory_id] = [
                t for t in self._activation_history[memory_id] 
                if t >= cutoff_time
            ]
            
            # Check for potential automatic associations based on co-activation
            self._auto_reinforce_for_memory(memory_id)
            
            # Trigger neural processing if enabled
            if self._neural_network and event.data.get("embedding"):
                self._process_memory_activation(memory_id, event.data.get("embedding"))
        except Exception as e:
            logger.error(f"Error handling memory access event: {e}")
    
    def _handle_association_query(self, event: Message) -> None:
        """
        Handle an association query event.
        
        Args:
            event: The event containing association query data
        """
        try:
            memory_id = event.data.get("memory_id")
            min_strength = event.data.get("min_strength", self._config.min_association_strength)
            association_type = event.data.get("association_type")
            direction = event.data.get("direction", "both")  # "outgoing", "incoming", or "both"
            
            if not memory_id:
                logger.warning("Association query event missing memory_id")
                return
                
            # Convert string ID to UUID if needed
            if isinstance(memory_id, str):
                memory_id = UUID(memory_id)
                
            # Get associations based on direction
            if direction == "outgoing":
                links = self.get_forward_links(memory_id, min_strength, association_type)
            elif direction == "incoming":
                links = self.get_backward_links(memory_id, min_strength, association_type)
            else:  # "both" or any other value
                links = self.get_links_for_memory(memory_id, min_strength, association_type)
                
            # Format response data
            link_data = []
            for link in links:
                link_data.append({
                    "link_id": str(link.id),
                    "source_id": str(link.source_id),
                    "target_id": str(link.target_id),
                    "association_type": link.association_type,
                    "strength": link.strength,
                    "created_at": link.created_at.isoformat(),
                    "last_reinforced": link.last_reinforced.isoformat(),
                    "reinforcement_count": link.reinforcement_count,
                    "bidirectional": link.bidirectional,
                    "features": link.features
                })
                
            # Publish response
            self._publish_memory_event("association_query_result", {
                "memory_id": str(memory_id),
                "links": link_data,
                "count": len(link_data),
                "request_id": event.data.get("request_id")
            })
        except Exception as e:
            logger.error(f"Error handling association query event: {e}")
    
    def _handle_age_update(self, event: Message) -> None:
        """
        Handle a developmental age update.
        
        Args:
            event: The event containing the new age
        """
        try:
            new_age = event.data.get("age")
            if new_age is not None:
                self.update_developmental_age(float(new_age))
        except Exception as e:
            logger.error(f"Error handling age update event: {e}")
    
    def _process_memory_activation(self, memory_id: UUID, embedding: List[float]) -> None:
        """
        Process memory activation through neural network for association learning.
        
        Args:
            memory_id: ID of the activated memory
            embedding: Vector embedding of the memory
        """
        if not self._neural_network:
            return
            
        try:
            # Find recent activations to compare with
            current_time = datetime.now()
            recent_activations = []
            
            for other_id, timestamps in self._activation_history.items():
                if other_id == memory_id:
                    continue
                    
                # Check if any activations within last 5 seconds
                recent_timestamps = [t for t in timestamps if (current_time - t).total_seconds() < 5]
                if recent_timestamps:
                    recent_activations.append(other_id)
            
            # Process each recent co-activation through neural network
            for other_id in recent_activations:
                # This is a simplification - in a full implementation, we'd retrieve
                # the actual embeddings for both memories and process them
                
                # Check if a link already exists
                existing_link = self._find_link(memory_id, other_id)
                if existing_link:
                    # Reinforce existing link
                    self.reinforce_link(existing_link.id, 0.05)
                else:
                    # Consider creating a new link if co-activation is frequent enough
                    link_created = self._check_create_link_from_coactivation(memory_id, other_id)
                    if link_created:
                        logger.debug(f"Created association from co-activation: {memory_id} -> {other_id}")
        except Exception as e:
            logger.error(f"Error processing memory activation: {e}")
    
    def create_link(
        self,
        source_id: UUID,
        target_id: UUID,
        association_type: str,
        strength: float = 0.5,
        bidirectional: bool = False,
        features: Optional[Dict[str, Any]] = None
    ) -> Optional[AssociativeLink]:
        """
        Create an association between two memory items.
        
        Args:
            source_id: ID of the source memory item
            target_id: ID of the target memory item
            association_type: Type of association
            strength: Initial strength of the association (0-1)
            bidirectional: Whether the association is bidirectional
            features: Additional features of the association
            
        Returns:
            The created association link, or None if creation failed
        """
        # Prevent self-loops
        if source_id == target_id:
            logger.warning("Cannot create association from item to itself")
            return None
        
        try:
            # Check if link already exists
            existing_link = self._find_link(source_id, target_id)
            if existing_link:
                # Update existing link
                existing_link.strength = max(existing_link.strength, strength)
                existing_link.last_reinforced = datetime.now()
                existing_link.reinforcement_count += 1
                existing_link.bidirectional = bidirectional or existing_link.bidirectional
                
                # Update features if provided
                if features:
                    existing_link.features.update(features)
                
                logger.debug(f"Updated existing association {existing_link.id}")
                return existing_link
            
            # Apply developmental adjustment to initial strength
            # Younger minds form weaker initial associations that require more reinforcement
            dev_factor = min(1.0, 0.7 + self._developmental_age * 0.3)
            adjusted_strength = strength * dev_factor
            
            # Create new link
            link = AssociativeLink(
                source_id=source_id,
                target_id=target_id,
                association_type=association_type,
                strength=adjusted_strength,
                bidirectional=bidirectional,
                features=features or {}
            )
            
            # Store link
            self._links[link.id] = link
            
            # Update indexes
            self._source_index[source_id].add(link.id)
            self._target_index[target_id].add(link.id)
            self._type_index[association_type].add(link.id)
            
            # Update statistics
            self._association_type_counts[association_type] += 1
            
            # Add to recent associations
            self._recent_associations.append(link.id)
            if len(self._recent_associations) > 20:
                self._recent_associations.pop(0)
            
            # Create bidirectional link if requested
            if bidirectional:
                # Avoid recursive bidirectional creation
                reverse_link = AssociativeLink(
                    source_id=target_id,
                    target_id=source_id,
                    association_type=association_type,
                    strength=adjusted_strength,
                    bidirectional=True,  # Mark as bidirectional
                    features=features or {}
                )
                
                # Store reverse link
                self._links[reverse_link.id] = reverse_link
                
                # Update indexes
                self._source_index[target_id].add(reverse_link.id)
                self._target_index[source_id].add(reverse_link.id)
                self._type_index[association_type].add(reverse_link.id)
            
            # Publish event
            self._publish_memory_event("association_created", {
                "link_id": str(link.id),
                "source_id": str(source_id),
                "target_id": str(target_id),
                "association_type": association_type,
                "strength": link.strength,
                "bidirectional": bidirectional
            })
            
            logger.debug(f"Created association {link.id} from {source_id} to {target_id}")
            return link
        except Exception as e:
            logger.error(f"Error creating association: {e}")
            return None
    
    def reinforce_link(self, link_id: UUID, amount: float = 0.1) -> bool:
        """
        Reinforce an association link.
        
        Args:
            link_id: ID of the link to reinforce
            amount: Amount to increase strength by
            
        Returns:
            True if successfully reinforced, False otherwise
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(link_id, str):
                link_id = UUID(link_id)
                
            if link_id in self._links:
                # Apply developmental adjustment to reinforcement amount
                # Younger minds have more plastic associations
                dev_factor = max(1.0, 1.5 - self._developmental_age * 0.5)
                adjusted_amount = amount * dev_factor
                
                # Reinforce link
                self._links[link_id].reinforce(adjusted_amount)
                
                # Update Hebbian learning if available
                if self._hebbian_learner:
                    # This is simplified; ideally we would retrieve the actual
                    # neural activations associated with both memories
                    self._hebbian_learner.apply_hebbian_update(
                        pre_activation=1.0,
                        post_activation=1.0,
                        connection_strength=self._links[link_id].strength
                    )
                
                # Check if bidirectional and reinforce reverse link too
                if self._links[link_id].bidirectional:
                    source_id = self._links[link_id].source_id
                    target_id = self._links[link_id].target_id
                    
                    # Find reverse link
                    for rev_link_id in self._source_index.get(target_id, set()):
                        if (rev_link_id in self._links and 
                            self._links[rev_link_id].target_id == source_id):
                            # Reinforce reverse link
                            self._links[rev_link_id].reinforce(adjusted_amount)
                            break
                
                logger.debug(f"Reinforced association {link_id} by {adjusted_amount}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error reinforcing link {link_id}: {e}")
            return False
    
    def reinforce_link_between(
        self,
        source_id: UUID,
        target_id: UUID,
        amount: float = 0.1
    ) -> bool:
        """
        Reinforce association between two memory items.
        
        Args:
            source_id: ID of source memory item
            target_id: ID of target memory item
            amount: Amount to increase strength by
            
        Returns:
            True if successfully reinforced, False otherwise
        """
        try:
            # Find link between source and target
            link = self._find_link(source_id, target_id)
            
            if link:
                # Reinforce existing link
                return self.reinforce_link(link.id, amount)
            else:
                # No link exists, create a weak one
                initial_strength = max(0.1, min(0.3, amount * 2))
                new_link = self.create_link(
                    source_id=source_id,
                    target_id=target_id,
                    association_type="automatic",
                    strength=initial_strength
                )
                
                return new_link is not None
        except Exception as e:
            logger.error(f"Error reinforcing link between {source_id} and {target_id}: {e}")
            return False
    
    def get_links_for_memory(
        self,
        memory_id: UUID,
        min_strength: float = 0.0,
        association_type: Optional[str] = None
    ) -> List[AssociativeLink]:
        """
        Get all association links for a memory item.
        
        Args:
            memory_id: ID of the memory item
            min_strength: Minimum association strength
            association_type: Optional type filter
            
        Returns:
            List of association links
        """
        try:
            results = []
            
            # Get outgoing links
            outgoing_links = self.get_forward_links(memory_id, min_strength, association_type)
            results.extend(outgoing_links)
            
            # Get incoming links
            incoming_links = self.get_backward_links(memory_id, min_strength, association_type)
            
            # Combine results (avoiding duplicates from bidirectional links)
            seen_ids = {link.id for link in results}
            for link in incoming_links:
                if link.id not in seen_ids:
                    results.append(link)
                    seen_ids.add(link.id)
            
            return results
        except Exception as e:
            logger.error(f"Error getting links for memory {memory_id}: {e}")
            return []
    
    def get_forward_links(
        self,
        source_id: UUID,
        min_strength: float = 0.0,
        association_type: Optional[str] = None
    ) -> List[AssociativeLink]:
        """
        Get outgoing association links from a memory item.
        
        Args:
            source_id: ID of the source memory item
            min_strength: Minimum association strength
            association_type: Optional type filter
            
        Returns:
            List of outgoing association links
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(source_id, str):
                source_id = UUID(source_id)
                
            results = []
            
            # Get all link IDs from source index
            link_ids = self._source_index.get(source_id, set())
            
            # Filter links by strength and type
            for link_id in link_ids:
                if link_id in self._links:
                    link = self._links[link_id]
                    
                    # Check strength threshold
                    if link.strength < min_strength:
                        continue
                        
                    # Check association type if specified
                    if association_type and link.association_type != association_type:
                        continue
                        
                    results.append(link)
            
            # Sort by strength, strongest first
            results.sort(key=lambda x: x.strength, reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error getting forward links for {source_id}: {e}")
            return []
    
    def get_backward_links(
        self,
        target_id: UUID,
        min_strength: float = 0.0,
        association_type: Optional[str] = None
    ) -> List[AssociativeLink]:
        """
        Get incoming association links to a memory item.
        
        Args:
            target_id: ID of the target memory item
            min_strength: Minimum association strength
            association_type: Optional type filter
            
        Returns:
            List of incoming association links
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(target_id, str):
                target_id = UUID(target_id)
                
            results = []
            
            # Get all link IDs from target index
            link_ids = self._target_index.get(target_id, set())
            
            # Filter links by strength and type
            for link_id in link_ids:
                if link_id in self._links:
                    link = self._links[link_id]
                    
                    # Check strength threshold
                    if link.strength < min_strength:
                        continue
                        
                    # Check association type if specified
                    if association_type and link.association_type != association_type:
                        continue
                        
                    results.append(link)
            
            # Sort by strength, strongest first
            results.sort(key=lambda x: x.strength, reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error getting backward links for {target_id}: {e}")
            return []
    
    def get_links_by_type(
        self,
        association_type: str,
        min_strength: float = 0.0
    ) -> List[AssociativeLink]:
        """
        Get association links by type.
        
        Args:
            association_type: Type of association to find
            min_strength: Minimum association strength
            
        Returns:
            List of association links of the specified type
        """
        try:
            results = []
            
            # Get all link IDs from type index
            link_ids = self._type_index.get(association_type, set())
            
            # Filter links by strength
            for link_id in link_ids:
                if link_id in self._links and self._links[link_id].strength >= min_strength:
                    results.append(self._links[link_id])
            
            # Sort by strength, strongest first
            results.sort(key=lambda x: x.strength, reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error getting links by type {association_type}: {e}")
            return []
    
    def get_link_count(self) -> int:
        """
        Get total number of association links.
        
        Returns:
            Number of association links
        """
        return len(self._links)
    
    def get_link_type_statistics(self) -> Dict[str, int]:
        """
        Get statistics about association types.
        
        Returns:
            Dictionary mapping association types to counts
        """
        return dict(self._association_type_counts)
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the associative memory system.
        
        Args:
            new_age: New developmental age
        """
        if new_age < 0:
            logger.warning(f"Invalid developmental age: {new_age}")
            return
            
        old_age = self._developmental_age
        self._developmental_age = new_age
        
        # Update learning parameters
        if self._hebbian_learner:
            # Adjust learning rate based on age
            # Younger minds have faster learning but more decay
            self._hebbian_learner.learning_rate = max(0.01, 0.05 * (1 + new_age * 0.5))
            self._hebbian_learner.decay_rate = max(0.001, 0.01 * (1 - new_age * 0.3))
        
        # Update co-activation threshold based on age
        # Younger minds form associations more easily but with lower strength
        self._co_activation_threshold = max(
            0.2, 
            self._config.min_association_strength * (1 + new_age * 0.5)
        )
        
        logger.info(f"Associative memory developmental age updated from {old_age:.2f} to {new_age:.2f}")
    
    def _find_link(self, source_id: UUID, target_id: UUID) -> Optional[AssociativeLink]:
        """
        Find association link between two memory items.
        
        Args:
            source_id: ID of source memory item
            target_id: ID of target memory item
            
        Returns:
            Association link if found, None otherwise
        """
        try:
            # Get all outgoing links from source
            for link_id in self._source_index.get(source_id, set()):
                if link_id in self._links and self._links[link_id].target_id == target_id:
                    return self._links[link_id]
                    
            # Check bidirectional links from target to source as well
            for link_id in self._source_index.get(target_id, set()):
                if (link_id in self._links and
                    self._links[link_id].target_id == source_id and
                    self._links[link_id].bidirectional):
                    return self._links[link_id]
                    
            return None
        except Exception as e:
            logger.error(f"Error finding link between {source_id} and {target_id}: {e}")
            return None
    
    def _check_create_link_from_coactivation(self, memory_id1: UUID, memory_id2: UUID) -> bool:
        """
        Check if a link should be created from co-activation and create if appropriate.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            
        Returns:
            True if a link was created, False otherwise
        """
        try:
            # Check if link already exists in either direction
            existing_link = self._find_link(memory_id1, memory_id2)
            if existing_link:
                # Reinforce existing link
                self.reinforce_link(existing_link.id, 0.05)
                return False
                
            # Check co-activation history to determine if link should be created
            timestamps1 = self._activation_history.get(memory_id1, [])
            timestamps2 = self._activation_history.get(memory_id2, [])
            
            if not timestamps1 or not timestamps2:
                return False
                
            # Calculate co-activation score
            # More co-activations within short time windows increases score
            co_activation_score = 0.0
            
            for t1 in timestamps1:
                for t2 in timestamps2:
                    # Calculate time difference in seconds
                    time_diff = abs((t1 - t2).total_seconds())
                    
                    # Only consider activations within short time window
                    if time_diff <= 5.0:
                        # Closer activations get higher score
                        co_activation_score += max(0.0, 1.0 - (time_diff / 5.0))
            
            # Scale score based on history length
            co_activation_score /= max(1, min(len(timestamps1), len(timestamps2)))
            
            # Check if score exceeds threshold
            should_create = co_activation_score >= self._co_activation_threshold
            
            if should_create:
                # Create link with strength based on co-activation score
                initial_strength = max(0.1, min(0.5, co_activation_score))
                new_link = self.create_link(
                    source_id=memory_id1,
                    target_id=memory_id2,
                    association_type="co_activation",
                    strength=initial_strength
                )
                
                return new_link is not None
                
            return False
        except Exception as e:
            logger.error(f"Error checking co-activation link: {e}")
            return False
    
    def _auto_reinforce_for_memory(self, memory_id: UUID) -> None:
        """
        Automatically reinforce associations for a memory.
        
        Args:
            memory_id: ID of memory that was accessed
        """
        try:
            # Get all associations for this memory
            links = self.get_links_for_memory(memory_id)
            
            if not links:
                return
                
            # Apply small reinforcement to all associations
            # using developmental age as a factor
            dev_factor = min(1.0, 0.5 + self._developmental_age * 0.5)
            for link in links:
                # Age-appropriate boost amount
                boost_amount = 0.02 * dev_factor
                
                # Boost is higher for stronger links (rich get richer)
                boost_amount *= (0.5 + link.strength * 0.5)
                
                # Apply reinforcement
                self.reinforce_link(link.id, boost_amount)
        except Exception as e:
            logger.error(f"Error auto-reinforcing for memory {memory_id}: {e}")
    
    def _publish_memory_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Publish a memory event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        event = MemoryEvent(
            event_type=event_type,
            source_module="memory.associative_memory",
            payload=payload
        )
        
        # Publish to event bus
        self._event_bus.publish(event_type, event.dict())
        
        # Also publish as message
        message = Message(
            type=MessageType.MEMORY,
            source="memory.associative_memory",
            recipient=Recipient.BROADCAST,
            content={
                "event_type": event_type,
                **payload
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
