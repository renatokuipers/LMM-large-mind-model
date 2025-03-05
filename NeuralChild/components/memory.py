"""
Memory component for the NeuralChild project.

This module defines the MemoryComponent class that handles different types
of memory systems in the neural child, including working memory, episodic
memory, semantic memory, and procedural memory.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import random
import math
import time
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
from enum import Enum
from pydantic import BaseModel, Field

from .base import NeuralComponent, ConnectionType
from ..config import DevelopmentalStage


class MemoryType(str, Enum):
    """Types of memory."""
    WORKING = "working"  # Short-term, limited capacity
    EPISODIC = "episodic"  # Autobiographical events
    SEMANTIC = "semantic"  # Facts and concepts
    PROCEDURAL = "procedural"  # Skills and how to do things
    EMOTIONAL = "emotional"  # Emotional memories


class MemoryStatus(str, Enum):
    """Status of a memory."""
    ACTIVE = "active"  # Currently being processed
    RECENT = "recent"  # Recently processed, easily accessible
    CONSOLIDATED = "consolidated"  # Long-term storage
    FADING = "fading"  # Beginning to be forgotten
    FORGOTTEN = "forgotten"  # No longer accessible


class MemoryAssociation(BaseModel):
    """Association between memories."""
    target_id: str
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    association_type: str  # e.g., "temporal", "semantic", "emotional"


class Memory(BaseModel):
    """Representation of a single memory."""
    id: str
    content: Dict[str, Any]  # The actual memory content
    memory_type: MemoryType
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: float  # Timestamp of creation
    last_accessed: float  # Timestamp of last access
    access_count: int = 0  # How often accessed
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)  # Emotional charge
    importance: float = Field(default=0.5, ge=0.0, le=1.0)  # How important the memory is
    development_stage: DevelopmentalStage  # Stage when formed
    context_tags: List[str] = Field(default_factory=list)  # Tags for context
    associations: List[MemoryAssociation] = Field(default_factory=list)  # Associated memories
    
    # Embedding representation (for semantic retrieval)
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class WorkingMemory(BaseModel):
    """Working memory system with limited capacity."""
    capacity: int  # Maximum number of items
    items: List[Memory] = Field(default_factory=list)  # Current items
    focus_of_attention: Optional[str] = None  # ID of currently focused memory
    
    class Config:
        arbitrary_types_allowed = True


class EpisodicMemory(BaseModel):
    """Episodic memory system for autobiographical events."""
    memories: Dict[str, Memory] = Field(default_factory=dict)
    recent_memories: List[str] = Field(default_factory=list)  # IDs of recent memories
    
    class Config:
        arbitrary_types_allowed = True


class SemanticMemory(BaseModel):
    """Semantic memory system for facts and concepts."""
    concepts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    facts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    relations: Dict[str, List[Tuple[str, str, float]]] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class ProceduralMemory(BaseModel):
    """Procedural memory system for skills and actions."""
    skills: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class MemoryComponent(NeuralComponent):
    """
    Component that handles memory systems in the neural child.
    
    This component models different types of memory and their
    development over time.
    """
    
    def __init__(
        self,
        development_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
        component_id: Optional[str] = None
    ):
        """
        Initialize the memory component.
        
        Args:
            development_stage: Current developmental stage
            component_id: Optional ID (generated if not provided)
        """
        super().__init__(
            name="Memory",
            activation_threshold=0.2,
            activation_decay_rate=0.05,
            learning_rate=0.05,
            development_stage=development_stage,
            component_id=component_id
        )
        
        # Initialize memory systems
        self._initialize_memory_systems()
        
        # Memory management parameters
        self.metadata.update({
            # Forgetting curves
            "working_memory_decay_rate": 0.2,  # Rate of decay for working memory
            "episodic_memory_decay_rate": 0.01,  # Rate of decay for episodic memory
            "semantic_memory_decay_rate": 0.001,  # Rate of decay for semantic memory
            
            # Retrieval strength factors
            "recency_factor": 0.7,  # How much recent access boosts retrieval
            "frequency_factor": 0.3,  # How much frequent access boosts retrieval
            "emotional_factor": 0.5,  # How much emotional charge boosts retrieval
            "importance_factor": 0.6,  # How much importance boosts retrieval
            
            # Memory formation thresholds
            "working_to_episodic_threshold": 0.5,  # Threshold for transfer to episodic
            "episodic_to_semantic_threshold": 0.7,  # Threshold for transfer to semantic
            
            # Consolidation parameters
            "consolidation_rate": 0.1,  # Rate of memory consolidation
            "consolidation_threshold": 3,  # Access count for consolidation
            
            # Association parameters
            "association_formation_threshold": 0.3,  # Threshold for forming associations
            "association_strength_increment": 0.1,  # Increment for association strength
            
            # Memory embedding dimension
            "embedding_dim": 32  # Dimension of memory embeddings
        })
        
        # Memory stats
        self.stats = {
            "total_memories_formed": 0,
            "total_memories_recalled": 0,
            "working_memory_capacity_used": 0,
            "episodic_memory_count": 0,
            "semantic_memory_count": 0,
            "procedural_memory_count": 0,
            "avg_retrieval_time": 0.0
        }
    
    def _initialize_memory_systems(self) -> None:
        """Initialize memory systems based on developmental stage."""
        # Working memory capacity depends on developmental stage
        working_memory_capacity = {
            DevelopmentalStage.PRENATAL: 1,
            DevelopmentalStage.INFANCY: 2,
            DevelopmentalStage.EARLY_CHILDHOOD: 3,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 5,
            DevelopmentalStage.ADOLESCENCE: 7,
            DevelopmentalStage.EARLY_ADULTHOOD: 7,
            DevelopmentalStage.MID_ADULTHOOD: 7
        }.get(self.development_stage, 1)
        
        # Initialize systems
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        
        # Memory sequence counter (for generating IDs)
        self.memory_counter = 0
    
    def _on_stage_transition(
        self, 
        old_stage: DevelopmentalStage, 
        new_stage: DevelopmentalStage
    ) -> None:
        """
        Handle developmental stage transitions.
        
        Args:
            old_stage: Previous developmental stage
            new_stage: New developmental stage
        """
        # Call parent method
        super()._on_stage_transition(old_stage, new_stage)
        
        # Update working memory capacity
        new_capacity = {
            DevelopmentalStage.PRENATAL: 1,
            DevelopmentalStage.INFANCY: 2,
            DevelopmentalStage.EARLY_CHILDHOOD: 3,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 5,
            DevelopmentalStage.ADOLESCENCE: 7,
            DevelopmentalStage.EARLY_ADULTHOOD: 7,
            DevelopmentalStage.MID_ADULTHOOD: 7
        }.get(new_stage, 1)
        
        # If capacity increased, update
        if new_capacity > self.working_memory.capacity:
            self.working_memory.capacity = new_capacity
        
        # Update decay rates based on stage
        if new_stage in [DevelopmentalStage.ADOLESCENCE, DevelopmentalStage.EARLY_ADULTHOOD, DevelopmentalStage.MID_ADULTHOOD]:
            # Better memory retention in later stages
            self.metadata["working_memory_decay_rate"] = 0.15
            self.metadata["episodic_memory_decay_rate"] = 0.005
        elif new_stage in [DevelopmentalStage.MIDDLE_CHILDHOOD]:
            # Moderate memory retention
            self.metadata["working_memory_decay_rate"] = 0.18
            self.metadata["episodic_memory_decay_rate"] = 0.008
        
        # Update association formation
        if new_stage in [DevelopmentalStage.EARLY_ADULTHOOD, DevelopmentalStage.MID_ADULTHOOD]:
            # Better at forming associations
            self.metadata["association_formation_threshold"] = 0.2
            self.metadata["association_strength_increment"] = 0.15
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations.
        
        Args:
            inputs: Dictionary containing:
                - 'operation': Memory operation to perform
                - 'data': Data for the operation
                - Other operation-specific parameters
                
        Returns:
            Dictionary containing operation results
        """
        operation = inputs.get('operation', 'encode')
        
        # Activate based on operation complexity
        activation_map = {
            'encode': 0.3,  # Storing new memory
            'retrieve': 0.4,  # Retrieving memory
            'associate': 0.5,  # Creating associations
            'consolidate': 0.6,  # Consolidating memories
            'forget': 0.2  # Forgetting memories
        }
        
        self.activate(activation_map.get(operation, 0.3))
        
        # If not activated enough, return minimal results
        if self.activation < self.activation_threshold:
            return {
                'success': False,
                'message': 'Memory component not sufficiently activated',
                'operation': operation
            }
        
        # Dispatch to appropriate operation handler
        if operation == 'encode':
            result = self._encode_memory(inputs)
        elif operation == 'retrieve':
            result = self._retrieve_memory(inputs)
        elif operation == 'associate':
            result = self._associate_memories(inputs)
        elif operation == 'consolidate':
            result = self._consolidate_memories(inputs)
        elif operation == 'forget':
            result = self._forget_memories(inputs)
        else:
            result = {
                'success': False,
                'message': f'Unknown operation: {operation}',
                'operation': operation
            }
        
        # Update memory systems (housekeeping)
        self._update_memory_systems()
        
        # Update component confidence based on operation success
        if result.get('success', False):
            # Success rate is 1.0 for successful operations
            self.update_confidence(1.0)
        else:
            # Success rate is 0.5 for failed operations - not complete failure
            self.update_confidence(0.5)
        
        return result
    
    def _encode_memory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode a new memory.
        
        Args:
            inputs: Dictionary containing:
                - 'data': Memory content
                - 'memory_type': Type of memory
                - 'emotional_valence': Emotional charge
                - 'importance': Importance level
                - 'context_tags': Context tags
                
        Returns:
            Dictionary with encoding results
        """
        data = inputs.get('data', {})
        memory_type = inputs.get('memory_type', MemoryType.EPISODIC)
        emotional_valence = inputs.get('emotional_valence', 0.0)
        importance = inputs.get('importance', 0.5)
        context_tags = inputs.get('context_tags', [])
        
        # Create memory ID
        memory_id = f"mem_{self.memory_counter}"
        self.memory_counter += 1
        
        # Get current time
        current_time = time.time()
        
        # Create embedding if semantic memory
        embedding = None
        if memory_type == MemoryType.SEMANTIC:
            # This is a simplified version - in a real system, we'd use a
            # proper embedding model like word2vec or a neural network
            # Here we're just creating a random vector as a placeholder
            embedding = [random.uniform(-1, 1) for _ in range(self.metadata["embedding_dim"])]
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            content=data,
            memory_type=memory_type,
            status=MemoryStatus.ACTIVE,
            created_at=current_time,
            last_accessed=current_time,
            emotional_valence=emotional_valence,
            importance=importance,
            development_stage=self.development_stage,
            context_tags=context_tags,
            embedding=embedding
        )
        
        # Handle based on memory type
        if memory_type == MemoryType.WORKING:
            # Check if working memory is full
            if len(self.working_memory.items) >= self.working_memory.capacity:
                # Remove least important item
                self.working_memory.items.sort(key=lambda m: m.importance)
                removed_memory = self.working_memory.items.pop(0)
                
                # Possibly transfer to episodic memory if important enough
                if (removed_memory.importance >= self.metadata["working_to_episodic_threshold"] or
                        removed_memory.emotional_valence >= 0.7 or
                        removed_memory.emotional_valence <= -0.7):
                    removed_memory.status = MemoryStatus.RECENT
                    self.episodic_memory.memories[removed_memory.id] = removed_memory
                    self.episodic_memory.recent_memories.append(removed_memory.id)
            
            # Add to working memory
            self.working_memory.items.append(memory)
            
            # Update focus of attention
            self.working_memory.focus_of_attention = memory_id
            
        elif memory_type == MemoryType.EPISODIC:
            # Add to episodic memory
            memory.status = MemoryStatus.RECENT
            self.episodic_memory.memories[memory_id] = memory
            self.episodic_memory.recent_memories.append(memory_id)
            
            # Limit recent memories list
            if len(self.episodic_memory.recent_memories) > 10:
                self.episodic_memory.recent_memories.pop(0)
            
        elif memory_type == MemoryType.SEMANTIC:
            # Add to semantic memory
            
            # If it's a concept
            if 'concept_name' in data:
                concept_name = data['concept_name']
                self.semantic_memory.concepts[concept_name] = {
                    'name': concept_name,
                    'attributes': data.get('attributes', {}),
                    'memory_id': memory_id,
                    'created_at': current_time
                }
            
            # If it's a fact
            elif 'fact' in data:
                fact = data['fact']
                fact_id = f"fact_{len(self.semantic_memory.facts)}"
                self.semantic_memory.facts[fact_id] = {
                    'fact': fact,
                    'context': data.get('context', {}),
                    'memory_id': memory_id,
                    'created_at': current_time
                }
            
            # If it defines relations
            if 'relations' in data:
                for relation in data['relations']:
                    if len(relation) >= 3:
                        source = relation[0]
                        relation_type = relation[1]
                        target = relation[2]
                        strength = relation[3] if len(relation) > 3 else 1.0
                        
                        if relation_type not in self.semantic_memory.relations:
                            self.semantic_memory.relations[relation_type] = []
                        
                        self.semantic_memory.relations[relation_type].append((source, target, strength))
        
        elif memory_type == MemoryType.PROCEDURAL:
            # Add to procedural memory
            if 'skill_name' in data:
                skill_name = data['skill_name']
                self.procedural_memory.skills[skill_name] = {
                    'name': skill_name,
                    'steps': data.get('steps', []),
                    'proficiency': data.get('proficiency', 0.1),
                    'memory_id': memory_id,
                    'created_at': current_time
                }
        
        # Update stats
        self.stats["total_memories_formed"] += 1
        if memory_type == MemoryType.EPISODIC:
            self.stats["episodic_memory_count"] = len(self.episodic_memory.memories)
        elif memory_type == MemoryType.SEMANTIC:
            self.stats["semantic_memory_count"] = len(self.semantic_memory.concepts) + len(self.semantic_memory.facts)
        elif memory_type == MemoryType.PROCEDURAL:
            self.stats["procedural_memory_count"] = len(self.procedural_memory.skills)
        
        return {
            'success': True,
            'memory_id': memory_id,
            'memory_type': memory_type,
            'message': f'Memory encoded successfully',
            'operation': 'encode'
        }
    
    def _retrieve_memory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve memory by ID, query, or association.
        
        Args:
            inputs: Dictionary containing one of:
                - 'memory_id': Specific memory ID to retrieve
                - 'query': Query to search for
                - 'by_association': ID to find associations of
                - 'memory_type': Type of memory to retrieve
                - 'context_tags': Context tags to filter by
                
        Returns:
            Dictionary with retrieval results
        """
        start_time = time.time()
        
        memory_id = inputs.get('memory_id')
        query = inputs.get('query')
        by_association = inputs.get('by_association')
        memory_type = inputs.get('memory_type')
        context_tags = inputs.get('context_tags', [])
        
        # Track retrieval success
        success = False
        retrieved_memories = []
        
        # Case 1: Retrieve by ID
        if memory_id:
            # Check working memory
            for memory in self.working_memory.items:
                if memory.id == memory_id:
                    # Update access stats
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    
                    # Add to results
                    retrieved_memories.append(memory)
                    success = True
                    break
            
            # Check episodic memory
            if not success and memory_id in self.episodic_memory.memories:
                memory = self.episodic_memory.memories[memory_id]
                
                # Update access stats
                memory.last_accessed = time.time()
                memory.access_count += 1
                
                # Add to results
                retrieved_memories.append(memory)
                success = True
                
                # If memory was consolidated, transfer to working memory if space
                if memory.status == MemoryStatus.CONSOLIDATED:
                    if len(self.working_memory.items) < self.working_memory.capacity:
                        self.working_memory.items.append(memory)
                        self.working_memory.focus_of_attention = memory_id
        
        # Case 2: Retrieve by query
        elif query:
            # This is a simplified search implementation
            # In a real system, this would use more sophisticated retrieval methods
            
            matching_memories = []
            
            # Search context tags first
            if context_tags:
                # Get memories with matching context tags
                for memory in self.episodic_memory.memories.values():
                    if any(tag in memory.context_tags for tag in context_tags):
                        matching_memories.append(memory)
            
            # If memory type specified, filter by type
            if memory_type and matching_memories:
                matching_memories = [m for m in matching_memories if m.memory_type == memory_type]
            
            # Further filter by query text match
            query_lower = query.lower()
            filtered_memories = []
            
            for memory in matching_memories:
                # Check if query appears in any string values in content
                found = False
                for key, value in memory.content.items():
                    if isinstance(value, str) and query_lower in value.lower():
                        found = True
                        break
                
                if found:
                    filtered_memories.append(memory)
            
            # Sort by relevance (combination of recency and importance)
            if filtered_memories:
                # Calculate relevance score for each memory
                current_time = time.time()
                scored_memories = []
                
                for memory in filtered_memories:
                    # Recency score (higher for more recent)
                    recency_score = 1.0 / max(1.0, (current_time - memory.last_accessed) / 86400.0)  # Days
                    
                    # Importance score
                    importance_score = memory.importance
                    
                    # Emotional score (absolute value of emotional valence)
                    emotional_score = abs(memory.emotional_valence)
                    
                    # Combined score
                    combined_score = (
                        recency_score * self.metadata["recency_factor"] +
                        importance_score * self.metadata["importance_factor"] +
                        emotional_score * self.metadata["emotional_factor"]
                    )
                    
                    scored_memories.append((memory, combined_score))
                
                # Sort by score (descending)
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                
                # Update access stats for top results and add to retrieved_memories
                for memory, _ in scored_memories[:5]:  # Limit to top 5
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    retrieved_memories.append(memory)
                
                success = bool(retrieved_memories)
        
        # Case 3: Retrieve by association
        elif by_association:
            # Find memories associated with the given memory ID
            source_memory = None
            
            # Check working memory
            for memory in self.working_memory.items:
                if memory.id == by_association:
                    source_memory = memory
                    break
            
            # Check episodic memory
            if not source_memory and by_association in self.episodic_memory.memories:
                source_memory = self.episodic_memory.memories[by_association]
            
            if source_memory:
                # Collect associated memory IDs
                associated_ids = [assoc.target_id for assoc in source_memory.associations]
                
                # Retrieve associated memories
                for mem_id in associated_ids:
                    # Check working memory
                    for memory in self.working_memory.items:
                        if memory.id == mem_id:
                            memory.last_accessed = time.time()
                            memory.access_count += 1
                            retrieved_memories.append(memory)
                            break
                    
                    # Check episodic memory
                    if mem_id in self.episodic_memory.memories:
                        memory = self.episodic_memory.memories[mem_id]
                        memory.last_accessed = time.time()
                        memory.access_count += 1
                        retrieved_memories.append(memory)
                
                success = bool(retrieved_memories)
        
        # Calculate retrieval time
        retrieval_time = time.time() - start_time
        
        # Update stats
        if success:
            self.stats["total_memories_recalled"] += len(retrieved_memories)
            # Update average retrieval time (exponential moving average)
            if self.stats["avg_retrieval_time"] == 0:
                self.stats["avg_retrieval_time"] = retrieval_time
            else:
                self.stats["avg_retrieval_time"] = 0.9 * self.stats["avg_retrieval_time"] + 0.1 * retrieval_time
            
            # Focus attention on the first retrieved memory
            if retrieved_memories:
                self.working_memory.focus_of_attention = retrieved_memories[0].id
        
        return {
            'success': success,
            'memories': [memory.dict() for memory in retrieved_memories],
            'count': len(retrieved_memories),
            'retrieval_time': retrieval_time,
            'message': 'Memory retrieval successful' if success else 'No matching memories found',
            'operation': 'retrieve'
        }
    
    def _associate_memories(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Associate two memories.
        
        Args:
            inputs: Dictionary containing:
                - 'source_id': Source memory ID
                - 'target_id': Target memory ID
                - 'association_type': Type of association
                - 'strength': Association strength
                
        Returns:
            Dictionary with association results
        """
        source_id = inputs.get('source_id')
        target_id = inputs.get('target_id')
        association_type = inputs.get('association_type', 'general')
        strength = inputs.get('strength', 0.5)
        
        # Find source memory
        source_memory = None
        
        # Check working memory
        for memory in self.working_memory.items:
            if memory.id == source_id:
                source_memory = memory
                break
        
        # Check episodic memory
        if not source_memory and source_id in self.episodic_memory.memories:
            source_memory = self.episodic_memory.memories[source_id]
        
        # Find target memory
        target_memory = None
        
        # Check working memory
        for memory in self.working_memory.items:
            if memory.id == target_id:
                target_memory = memory
                break
        
        # Check episodic memory
        if not target_memory and target_id in self.episodic_memory.memories:
            target_memory = self.episodic_memory.memories[target_id]
        
        # If both memories found, create association
        if source_memory and target_memory:
            # Check if association already exists
            existing_association = None
            for assoc in source_memory.associations:
                if assoc.target_id == target_id:
                    existing_association = assoc
                    break
            
            if existing_association:
                # Update existing association
                new_strength = min(1.0, existing_association.strength + self.metadata["association_strength_increment"])
                existing_association.strength = new_strength
            else:
                # Create new association
                new_association = MemoryAssociation(
                    target_id=target_id,
                    strength=strength,
                    association_type=association_type
                )
                source_memory.associations.append(new_association)
            
            # Bidirectional association (optional, based on association type)
            if association_type in ['temporal', 'semantic', 'spatial']:
                # Check if reverse association already exists
                existing_reverse = None
                for assoc in target_memory.associations:
                    if assoc.target_id == source_id:
                        existing_reverse = assoc
                        break
                
                if existing_reverse:
                    # Update existing association
                    new_strength = min(1.0, existing_reverse.strength + self.metadata["association_strength_increment"])
                    existing_reverse.strength = new_strength
                else:
                    # Create reverse association
                    reverse_association = MemoryAssociation(
                        target_id=source_id,
                        strength=strength * 0.8,  # Slightly weaker in reverse
                        association_type=association_type
                    )
                    target_memory.associations.append(reverse_association)
            
            return {
                'success': True,
                'source_id': source_id,
                'target_id': target_id,
                'association_type': association_type,
                'message': 'Memories associated successfully',
                'operation': 'associate'
            }
        else:
            return {
                'success': False,
                'message': 'One or both memories not found',
                'operation': 'associate'
            }
    
    def _consolidate_memories(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate memories from working to long-term storage.
        
        Args:
            inputs: Dictionary containing:
                - 'memory_ids': Optional list of specific memory IDs to consolidate
                
        Returns:
            Dictionary with consolidation results
        """
        specific_ids = inputs.get('memory_ids', [])
        consolidated_count = 0
        
        # If specific IDs provided, only consolidate those
        if specific_ids:
            # For each ID, find in working memory and consolidate
            for memory_id in specific_ids:
                for i, memory in enumerate(self.working_memory.items):
                    if memory.id == memory_id:
                        # Only consolidate if accessed enough or important
                        if (memory.access_count >= self.metadata["consolidation_threshold"] or
                                memory.importance >= self.metadata["working_to_episodic_threshold"]):
                            # Move to episodic memory
                            memory.status = MemoryStatus.CONSOLIDATED
                            self.episodic_memory.memories[memory.id] = memory
                            
                            # Remove from working memory
                            self.working_memory.items.pop(i)
                            
                            consolidated_count += 1
                            
                            # If this was the focus of attention, clear it
                            if self.working_memory.focus_of_attention == memory_id:
                                self.working_memory.focus_of_attention = None
                                
                                # Set new focus if working memory not empty
                                if self.working_memory.items:
                                    self.working_memory.focus_of_attention = self.working_memory.items[-1].id
                            
                            break
        else:
            # Consolidate all eligible memories in working memory
            # Start from the end to avoid index issues when removing
            i = len(self.working_memory.items) - 1
            while i >= 0:
                memory = self.working_memory.items[i]
                
                # Check if eligible for consolidation
                if (memory.access_count >= self.metadata["consolidation_threshold"] or
                        memory.importance >= self.metadata["working_to_episodic_threshold"]):
                    # Move to episodic memory
                    memory.status = MemoryStatus.CONSOLIDATED
                    self.episodic_memory.memories[memory.id] = memory
                    
                    # If this was the focus of attention, clear it
                    if self.working_memory.focus_of_attention == memory.id:
                        self.working_memory.focus_of_attention = None
                    
                    # Remove from working memory
                    self.working_memory.items.pop(i)
                    
                    consolidated_count += 1
                
                i -= 1
            
            # Set new focus if working memory not empty and focus was cleared
            if self.working_memory.items and not self.working_memory.focus_of_attention:
                self.working_memory.focus_of_attention = self.working_memory.items[-1].id
        
        # Update stats
        self.stats["episodic_memory_count"] = len(self.episodic_memory.memories)
        
        return {
            'success': True,
            'consolidated_count': consolidated_count,
            'working_memory_remaining': len(self.working_memory.items),
            'message': f'Consolidated {consolidated_count} memories',
            'operation': 'consolidate'
        }
    
    def _forget_memories(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply forgetting to memories based on decay rates.
        
        Args:
            inputs: Dictionary containing:
                - 'memory_type': Optional type of memory to forget
                - 'older_than_days': Optional age threshold
                
        Returns:
            Dictionary with forgetting results
        """
        memory_type = inputs.get('memory_type')
        older_than_days = inputs.get('older_than_days', 0)
        
        forgotten_count = 0
        fading_count = 0
        
        current_time = time.time()
        threshold_time = current_time - (older_than_days * 86400)  # Convert days to seconds
        
        # Apply forgetting to episodic memories
        if not memory_type or memory_type == MemoryType.EPISODIC:
            # Process each memory
            for memory_id, memory in list(self.episodic_memory.memories.items()):
                # Skip if not old enough
                if older_than_days > 0 and memory.created_at > threshold_time:
                    continue
                
                # Calculate time since last access
                time_since_access = current_time - memory.last_accessed
                
                # Calculate decay based on time and importance
                # Memories with high importance decay more slowly
                decay_factor = self.metadata["episodic_memory_decay_rate"] * (1.0 - memory.importance * 0.5)
                
                # Calculate probability of forgetting
                # This increases with time since last access
                forgetting_prob = decay_factor * (time_since_access / 86400.0)  # Days
                
                # Emotional memories are forgotten more slowly
                if abs(memory.emotional_valence) > 0.7:
                    forgetting_prob *= 0.5
                
                # If memory is already fading, increase the probability
                if memory.status == MemoryStatus.FADING:
                    forgetting_prob *= 2.0
                
                # Determine if memory should be forgotten or start fading
                if forgetting_prob > 0.8:
                    # Forget the memory
                    memory.status = MemoryStatus.FORGOTTEN
                    forgotten_count += 1
                    
                    # Actually remove forgotten memories after a while
                    if time_since_access > 30 * 86400:  # 30 days
                        del self.episodic_memory.memories[memory_id]
                
                elif forgetting_prob > 0.4 and memory.status != MemoryStatus.FADING:
                    # Mark memory as fading
                    memory.status = MemoryStatus.FADING
                    fading_count += 1
        
        # Update stats
        self.stats["episodic_memory_count"] = len(self.episodic_memory.memories)
        
        return {
            'success': True,
            'forgotten_count': forgotten_count,
            'fading_count': fading_count,
            'message': f'Forgot {forgotten_count} memories, {fading_count} are fading',
            'operation': 'forget'
        }
    
    def _update_memory_systems(self) -> None:
        """Update memory systems (housekeeping operations)."""
        # Update working memory stats
        self.stats["working_memory_capacity_used"] = len(self.working_memory.items)
        
        # Limit recent memories list
        if len(self.episodic_memory.recent_memories) > 10:
            self.episodic_memory.recent_memories = self.episodic_memory.recent_memories[-10:]
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID (convenience method).
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory object if found, None otherwise
        """
        # Check working memory
        for memory in self.working_memory.items:
            if memory.id == memory_id:
                return memory
        
        # Check episodic memory
        if memory_id in self.episodic_memory.memories:
            return self.episodic_memory.memories[memory_id]
        
        return None
    
    def get_current_focus(self) -> Optional[Memory]:
        """
        Get the current focus of attention.
        
        Returns:
            Memory object for current focus, None if no focus
        """
        if not self.working_memory.focus_of_attention:
            return None
        
        # Find the focus memory
        for memory in self.working_memory.items:
            if memory.id == self.working_memory.focus_of_attention:
                return memory
        
        # If not found in working memory, clear the focus
        self.working_memory.focus_of_attention = None
        return None
    
    def set_focus(self, memory_id: str) -> bool:
        """
        Set the focus of attention.
        
        Args:
            memory_id: Memory ID to focus on
            
        Returns:
            Whether focus was successfully set
        """
        # Check if memory exists in working memory
        for memory in self.working_memory.items:
            if memory.id == memory_id:
                self.working_memory.focus_of_attention = memory_id
                return True
        
        # If not in working memory, try to retrieve from episodic memory
        if memory_id in self.episodic_memory.memories:
            memory = self.episodic_memory.memories[memory_id]
            
            # Only add to working memory if there's space
            if len(self.working_memory.items) < self.working_memory.capacity:
                self.working_memory.items.append(memory)
                self.working_memory.focus_of_attention = memory_id
                return True
            else:
                # Remove least important item to make space
                self.working_memory.items.sort(key=lambda m: m.importance)
                self.working_memory.items.pop(0)
                
                # Add memory and set focus
                self.working_memory.items.append(memory)
                self.working_memory.focus_of_attention = memory_id
                return True
        
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return self.stats
    
    def create_simple_memory(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        context_tags: List[str] = None
    ) -> Optional[str]:
        """
        Create a simple memory (convenience method).
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            emotional_valence: Emotional charge
            context_tags: Context tags
            
        Returns:
            Memory ID if created successfully, None otherwise
        """
        result = self.process({
            'operation': 'encode',
            'data': content,
            'memory_type': memory_type,
            'importance': importance,
            'emotional_valence': emotional_valence,
            'context_tags': context_tags or []
        })
        
        if result.get('success', False):
            return result.get('memory_id')
        
        return None