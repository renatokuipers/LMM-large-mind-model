# memory_manager.py
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Type, TypeVar, Generic
import logging
import json
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from enum import Enum, auto
import os

# Import config for memory settings
from config import get_config

# Set up logging
logger = logging.getLogger("MemoryManager")

class MemoryType(str, Enum):
    """Types of memory in the system"""
    WORKING = "working"       # Short-term active processing
    EPISODIC = "episodic"     # Event-based memories of experiences
    ASSOCIATIVE = "associative"  # Connections between concepts
    LONG_TERM = "long_term"   # Persistent knowledge storage
    EMOTIONAL = "emotional"   # Emotional memory traces

class MemoryPriority(Enum):
    """Priority levels for memory operations"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class MemoryAccess(Enum):
    """Types of memory access operations"""
    READ = auto()
    WRITE = auto()
    UPDATE = auto()
    DELETE = auto()

class MemoryAttributes(BaseModel):
    """Common attributes for all memory items"""
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(0, ge=0)
    emotional_valence: float = Field(0.0, ge=-1.0, le=1.0)  # -1 (negative) to 1 (positive)
    emotional_intensity: float = Field(0.0, ge=0.0, le=1.0)  # How strong the emotional connection is
    salience: float = Field(0.5, ge=0.0, le=1.0)  # How important/notable the memory is
    confidence: float = Field(1.0, ge=0.0, le=1.0)  # How confident we are in this memory
    
    def update_access(self) -> None:
        """Update the access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1

T = TypeVar('T', bound=BaseModel)

class MemoryItem(BaseModel, Generic[T]):
    """A generic memory item that can hold different types of content"""
    id: str = Field(..., description="Unique identifier for this memory item")
    memory_type: MemoryType = Field(..., description="Type of memory")
    attributes: MemoryAttributes = Field(default_factory=MemoryAttributes)
    content: T = Field(..., description="Content of the memory item")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    associations: Dict[str, float] = Field(default_factory=dict, description="Associations to other memory items")
    
    def decay(self, rate: float = 0.01) -> None:
        """Apply memory decay based on time since last access"""
        # Memory items decay differently based on their type
        if self.memory_type == MemoryType.WORKING:
            # Working memory decays quickly
            self.attributes.salience *= (1.0 - rate * 2.0)
        elif self.memory_type == MemoryType.EPISODIC:
            # Episodic memory decays based on emotional intensity
            decay_factor = rate * (1.0 - self.attributes.emotional_intensity * 0.5)
            self.attributes.salience *= (1.0 - decay_factor)
        elif self.memory_type == MemoryType.LONG_TERM:
            # Long-term memory decays very slowly
            self.attributes.salience *= (1.0 - rate * 0.1)
        else:
            # Default decay
            self.attributes.salience *= (1.0 - rate)
        
        # Handle associations decay
        for key in list(self.associations.keys()):
            self.associations[key] *= (1.0 - rate * 0.5)
            # Remove weak associations
            if self.associations[key] < 0.1:
                del self.associations[key]

class MemoryQuery(BaseModel):
    """Model for querying memory"""
    keywords: Optional[List[str]] = Field(None, description="Keywords to search for")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    memory_types: Optional[List[MemoryType]] = Field(None, description="Types of memory to search")
    time_range: Optional[Tuple[datetime, datetime]] = Field(None, description="Time range to search within")
    emotional_valence: Optional[Tuple[float, float]] = Field(None, description="Emotional valence range")
    salience_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum salience to include")
    max_results: int = Field(10, ge=1, description="Maximum number of results to return")
    context_bias: Optional[Dict[str, float]] = Field(None, description="Context to bias search towards")

class MemorySearchResult(BaseModel):
    """Results from a memory search"""
    items: List[Any] = Field(default_factory=list)  # We use Any here since the items are heterogeneous
    relevance_scores: Dict[str, float] = Field(default_factory=dict)
    query_match_details: Dict[str, Any] = Field(default_factory=dict)

class MemoryStats(BaseModel):
    """Statistics about the memory system"""
    total_items: int = Field(0, ge=0)
    items_by_type: Dict[MemoryType, int] = Field(default_factory=dict)
    avg_salience: float = Field(0.0, ge=0.0, le=1.0)
    strongest_associations: List[Tuple[str, str, float]] = Field(default_factory=list)
    memory_age_distribution: Dict[str, int] = Field(default_factory=dict)
    total_associations: int = Field(0, ge=0)

class MemoryManager:
    """Central coordinator for all memory subsystems"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the memory manager"""
        self.config = get_config().memory
        self.data_dir = data_dir or Path("./data/memory")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Memory storage - in a real implementation, these would be
        # actual instances of specialized memory systems
        self.memories: Dict[str, MemoryItem] = {}
        
        # Memory indices for faster retrieval
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> memory IDs
        self.type_index: Dict[MemoryType, Set[str]] = {  # memory type -> memory IDs
            memory_type: set() for memory_type in MemoryType
        }
        self.time_index: Dict[str, List[str]] = {}  # time bucket -> memory IDs
        
        # Memory subsystems will be loaded dynamically when needed
        self._working_memory = None
        self._episodic_memory = None
        self._associative_memory = None
        self._long_term_memory = None
        
        logger.info("Memory manager initialized")
    
    def _get_working_memory(self):
        """Lazy-load working memory subsystem"""
        if self._working_memory is None:
            # This will be imported here to avoid circular imports
            from memory.working_memory import WorkingMemory
            self._working_memory = WorkingMemory()
        return self._working_memory
    
    def _get_episodic_memory(self):
        """Lazy-load episodic memory subsystem"""
        if self._episodic_memory is None:
            # This will be imported here to avoid circular imports
            from memory.episodic_memory import EpisodicMemory
            self._episodic_memory = EpisodicMemory()
        return self._episodic_memory
    
    def _get_associative_memory(self):
        """Lazy-load associative memory subsystem"""
        if self._associative_memory is None:
            # This will be imported here to avoid circular imports
            from memory.associative_memory import AssociativeMemory
            self._associative_memory = AssociativeMemory()
        return self._associative_memory
    
    def _get_long_term_memory(self):
        """Lazy-load long-term memory subsystem"""
        if self._long_term_memory is None:
            # This will be imported here to avoid circular imports
            from memory.long_term_memory import LongTermMemory
            self._long_term_memory = LongTermMemory()
        return self._long_term_memory
    
    def store(self, 
              content: Any, 
              memory_type: MemoryType, 
              tags: Optional[List[str]] = None,
              emotional_valence: float = 0.0,
              emotional_intensity: float = 0.0,
              salience: float = 0.5) -> str:
        """Store a new memory item"""
        # Generate a unique ID
        memory_id = f"{memory_type.value}_{len(self.memories)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create memory attributes
        attributes = MemoryAttributes(
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            salience=salience
        )
        
        # Create the memory item
        memory_item = MemoryItem(
            id=memory_id,
            memory_type=memory_type,
            attributes=attributes,
            content=content,
            tags=tags or []
        )
        
        # Store in the appropriate subsystem
        if memory_type == MemoryType.WORKING:
            self._get_working_memory().add_item(memory_item)
        elif memory_type == MemoryType.EPISODIC:
            self._get_episodic_memory().add_memory(memory_item)
        elif memory_type == MemoryType.ASSOCIATIVE:
            self._get_associative_memory().add_association(memory_item)
        elif memory_type == MemoryType.LONG_TERM:
            self._get_long_term_memory().store(memory_item)
        
        # Store in our central index
        self.memories[memory_id] = memory_item
        
        # Update indices
        self._update_indices(memory_item)
        
        logger.info(f"Stored new {memory_type.value} memory with ID {memory_id}")
        return memory_id
    
    def _update_indices(self, memory_item: MemoryItem) -> None:
        """Update memory indices for faster retrieval"""
        # Update tag index
        for tag in memory_item.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(memory_item.id)
        
        # Update type index
        self.type_index[memory_item.memory_type].add(memory_item.id)
        
        # Update time index
        time_bucket = memory_item.attributes.created_at.strftime("%Y-%m-%d")
        if time_bucket not in self.time_index:
            self.time_index[time_bucket] = []
        self.time_index[time_bucket].append(memory_item.id)
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID"""
        if memory_id not in self.memories:
            return None
        
        memory_item = self.memories[memory_id]
        memory_item.attributes.update_access()
        
        # Retrieve from appropriate subsystem for any additional processing
        if memory_item.memory_type == MemoryType.WORKING:
            self._get_working_memory().access_item(memory_id)
        elif memory_item.memory_type == MemoryType.EPISODIC:
            self._get_episodic_memory().recall_memory(memory_id)
        elif memory_item.memory_type == MemoryType.ASSOCIATIVE:
            self._get_associative_memory().activate_association(memory_id)
        elif memory_item.memory_type == MemoryType.LONG_TERM:
            self._get_long_term_memory().retrieve(memory_id)
        
        return memory_item
    
    def search(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching the query"""
        # Start with all memories
        candidate_ids = set(self.memories.keys())
        
        # Filter by memory type
        if query.memory_types:
            type_ids = set()
            for memory_type in query.memory_types:
                type_ids.update(self.type_index.get(memory_type, set()))
            candidate_ids = candidate_ids.intersection(type_ids)
        
        # Filter by tags
        if query.tags:
            tag_ids = set()
            for tag in query.tags:
                tag_ids.update(self.tag_index.get(tag, set()))
            candidate_ids = candidate_ids.intersection(tag_ids)
        
        # Filter by time range
        if query.time_range:
            start_time, end_time = query.time_range
            time_ids = set()
            for memory_id in candidate_ids:
                memory = self.memories[memory_id]
                if start_time <= memory.attributes.created_at <= end_time:
                    time_ids.add(memory_id)
            candidate_ids = time_ids
        
        # Filter by emotional valence
        if query.emotional_valence:
            min_valence, max_valence = query.emotional_valence
            valence_ids = set()
            for memory_id in candidate_ids:
                memory = self.memories[memory_id]
                if min_valence <= memory.attributes.emotional_valence <= max_valence:
                    valence_ids.add(memory_id)
            candidate_ids = valence_ids
        
        # Filter by salience threshold
        salience_ids = set()
        for memory_id in candidate_ids:
            memory = self.memories[memory_id]
            if memory.attributes.salience >= query.salience_threshold:
                salience_ids.add(memory_id)
        candidate_ids = salience_ids
        
        # Calculate relevance scores
        relevance_scores = {}
        for memory_id in candidate_ids:
            memory = self.memories[memory_id]
            
            # Base score is the memory's salience
            score = memory.attributes.salience
            
            # Adjust for context bias if provided
            if query.context_bias:
                context_score = 0.0
                for context_key, context_weight in query.context_bias.items():
                    # Check if the context key is in tags or associations
                    if context_key in memory.tags:
                        context_score += context_weight * 0.5
                    if context_key in memory.associations:
                        context_score += context_weight * memory.associations[context_key]
                
                score += context_score
            
            # Adjust for keyword matches if provided
            if query.keywords:
                keyword_score = 0.0
                for keyword in query.keywords:
                    # Simple string matching - in a real implementation would use embeddings
                    if hasattr(memory.content, "lower") and isinstance(memory.content, str):
                        if keyword.lower() in memory.content.lower():
                            keyword_score += 0.3
                    
                    # Check in tags too
                    if keyword in memory.tags:
                        keyword_score += 0.2
                
                score += keyword_score
            
            relevance_scores[memory_id] = min(1.0, score)  # Cap at 1.0
        
        # Sort by relevance and limit results
        sorted_ids = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
        result_ids = sorted_ids[:query.max_results]
        
        # Build result
        result_items = [self.retrieve(memory_id) for memory_id in result_ids]
        result = MemorySearchResult(
            items=[item for item in result_items if item is not None],  # Filter out any None values
            relevance_scores={memory_id: relevance_scores[memory_id] for memory_id in result_ids},
            query_match_details={
                "total_candidates": len(candidate_ids),
                "returned_results": len(result_ids),
                "filter_criteria": {
                    "memory_types": [mt.value for mt in query.memory_types] if query.memory_types else None,
                    "tags": query.tags,
                    "time_range": [t.isoformat() for t in query.time_range] if query.time_range else None,
                    "emotional_valence": query.emotional_valence,
                    "salience_threshold": query.salience_threshold
                }
            }
        )
        
        return result
    
    def update(self, memory_id: str, 
              updates: Dict[str, Any], 
              priority: MemoryPriority = MemoryPriority.MEDIUM) -> bool:
        """Update a memory item"""
        if memory_id not in self.memories:
            return False
        
        memory_item = self.memories[memory_id]
        memory_item.attributes.update_access()
        
        # Apply updates
        updated = False
        
        # Update attributes
        if "emotional_valence" in updates:
            memory_item.attributes.emotional_valence = updates["emotional_valence"]
            updated = True
        
        if "emotional_intensity" in updates:
            memory_item.attributes.emotional_intensity = updates["emotional_intensity"]
            updated = True
            
        if "salience" in updates:
            memory_item.attributes.salience = updates["salience"]
            updated = True
            
        if "confidence" in updates:
            memory_item.attributes.confidence = updates["confidence"]
            updated = True
        
        # Update tags
        if "tags" in updates:
            # Remove memory ID from old tag indices
            for tag in memory_item.tags:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
            
            memory_item.tags = updates["tags"]
            
            # Update tag index with new tags
            for tag in memory_item.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(memory_id)
                
            updated = True
        
        # Update associations
        if "associations" in updates:
            memory_item.associations = updates["associations"]
            updated = True
        
        # Update content if provided
        if "content" in updates:
            memory_item.content = updates["content"]
            updated = True
        
        # Update in the appropriate subsystem
        if updated:
            if memory_item.memory_type == MemoryType.WORKING:
                self._get_working_memory().update_item(memory_id, memory_item)
            elif memory_item.memory_type == MemoryType.EPISODIC:
                self._get_episodic_memory().update_memory(memory_id, memory_item)
            elif memory_item.memory_type == MemoryType.ASSOCIATIVE:
                self._get_associative_memory().update_association(memory_id, memory_item)
            elif memory_item.memory_type == MemoryType.LONG_TERM:
                self._get_long_term_memory().update(memory_id, memory_item)
            
            logger.info(f"Updated memory {memory_id} with priority {priority.name}")
            
        return updated
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory item"""
        if memory_id not in self.memories:
            return False
        
        memory_item = self.memories[memory_id]
        
        # Remove from indices
        for tag in memory_item.tags:
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        if memory_item.memory_type in self.type_index:
            self.type_index[memory_item.memory_type].discard(memory_id)
        
        for time_bucket, ids in self.time_index.items():
            if memory_id in ids:
                ids.remove(memory_id)
        
        # Delete from the appropriate subsystem
        if memory_item.memory_type == MemoryType.WORKING:
            self._get_working_memory().remove_item(memory_id)
        elif memory_item.memory_type == MemoryType.EPISODIC:
            self._get_episodic_memory().delete_memory(memory_id)
        elif memory_item.memory_type == MemoryType.ASSOCIATIVE:
            self._get_associative_memory().remove_association(memory_id)
        elif memory_item.memory_type == MemoryType.LONG_TERM:
            self._get_long_term_memory().delete(memory_id)
        
        # Delete from central storage
        del self.memories[memory_id]
        
        logger.info(f"Deleted memory {memory_id}")
        return True
    
    def associate(self, source_id: str, target_id: str, strength: float = 0.5) -> bool:
        """Create or strengthen an association between two memory items"""
        if source_id not in self.memories or target_id not in self.memories:
            return False
        
        # Update association in source memory
        source_memory = self.memories[source_id]
        source_memory.associations[target_id] = strength
        source_memory.attributes.update_access()
        
        # Update association in target memory (bidirectional, but potentially weaker)
        target_memory = self.memories[target_id]
        current_strength = target_memory.associations.get(source_id, 0.0)
        new_strength = max(current_strength, strength * 0.8)  # Slightly weaker backlink
        target_memory.associations[source_id] = new_strength
        target_memory.attributes.update_access()
        
        # Update in the associative memory subsystem
        self._get_associative_memory().create_association(source_id, target_id, strength)
        
        logger.info(f"Associated memory {source_id} with {target_id} at strength {strength}")
        return True
    
    def consolidate(self) -> int:
        """Consolidate working memory into long-term memory"""
        working_memory = self._get_working_memory()
        long_term_memory = self._get_long_term_memory()
        
        # Get items from working memory that should be consolidated
        consolidation_candidates = working_memory.get_consolidation_candidates()
        
        consolidated_count = 0
        for item_id in consolidation_candidates:
            if item_id in self.memories:
                item = self.memories[item_id]
                
                # Create a new long-term memory from the working memory item
                lt_id = self.store(
                    content=item.content,
                    memory_type=MemoryType.LONG_TERM,
                    tags=item.tags,
                    emotional_valence=item.attributes.emotional_valence,
                    emotional_intensity=item.attributes.emotional_intensity,
                    salience=item.attributes.salience
                )
                
                # Copy associations
                for target_id, strength in item.associations.items():
                    self.associate(lt_id, target_id, strength)
                
                # Update consolidation metadata
                self.update(lt_id, {"tags": item.tags + ["consolidated"]})
                
                # Remove from working memory
                working_memory.remove_item(item_id)
                consolidated_count += 1
        
        logger.info(f"Consolidated {consolidated_count} items from working memory to long-term memory")
        return consolidated_count
    
    def apply_memory_decay(self, decay_rate: Optional[float] = None) -> None:
        """Apply decay to all memories"""
        if decay_rate is None:
            decay_rate = self.config.long_term_decay_rate
            
        for memory_id, memory_item in list(self.memories.items()):
            memory_item.decay(decay_rate)
            
            # Remove memories that have decayed below threshold
            if memory_item.attributes.salience < 0.01:
                self.delete(memory_id)
        
        logger.info(f"Applied memory decay with rate {decay_rate}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get statistics about the memory system"""
        # Count items by type
        items_by_type = {memory_type: len(ids) for memory_type, ids in self.type_index.items()}
        
        # Calculate average salience
        salience_values = [memory.attributes.salience for memory in self.memories.values()]
        avg_salience = sum(salience_values) / max(1, len(salience_values))
        
        # Find strongest associations
        all_associations = []
        for source_id, memory in self.memories.items():
            for target_id, strength in memory.associations.items():
                if target_id in self.memories:
                    all_associations.append((source_id, target_id, strength))
        
        # Sort by strength
        all_associations.sort(key=lambda x: x[2], reverse=True)
        strongest_associations = all_associations[:10]  # Top 10
        
        # Memory age distribution
        age_distribution = {}
        current_time = datetime.now()
        for memory in self.memories.values():
            age_days = (current_time - memory.attributes.created_at).days
            age_bucket = f"{age_days // 10 * 10}-{(age_days // 10 * 10) + 9} days"
            age_distribution[age_bucket] = age_distribution.get(age_bucket, 0) + 1
        
        return MemoryStats(
            total_items=len(self.memories),
            items_by_type=items_by_type,
            avg_salience=avg_salience,
            strongest_associations=strongest_associations,
            memory_age_distribution=age_distribution,
            total_associations=len(all_associations)
        )
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the memory state to disk"""
        if filepath is None:
            filepath = self.data_dir / f"memory_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        serializable_memories = {}
        for memory_id, memory in self.memories.items():
            # Convert content to a serializable format if possible
            content = memory.content
            if hasattr(content, "model_dump"):
                content = content.model_dump()
            elif not isinstance(content, (str, int, float, bool, list, dict, type(None))):
                # Convert to string if not directly serializable
                content = str(content)
            
            serializable_memories[memory_id] = {
                "id": memory.id,
                "memory_type": memory.memory_type.value,
                "attributes": {
                    "created_at": memory.attributes.created_at.isoformat(),
                    "last_accessed": memory.attributes.last_accessed.isoformat(),
                    "access_count": memory.attributes.access_count,
                    "emotional_valence": memory.attributes.emotional_valence,
                    "emotional_intensity": memory.attributes.emotional_intensity,
                    "salience": memory.attributes.salience,
                    "confidence": memory.attributes.confidence
                },
                "content": content,
                "tags": memory.tags,
                "associations": memory.associations
            }
        
        # Create state dictionary
        state = {
            "memories": serializable_memories,
            "tag_index": {tag: list(memory_ids) for tag, memory_ids in self.tag_index.items()},
            "type_index": {memory_type.value: list(memory_ids) for memory_type, memory_ids in self.type_index.items()},
            "time_index": self.time_index,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_memories": len(self.memories),
                "version": "1.0"
            }
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        logger.info(f"Memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the memory state from disk"""
        if not os.path.exists(filepath):
            logger.error(f"Memory state file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.memories.clear()
            self.tag_index.clear()
            self.type_index = {memory_type: set() for memory_type in MemoryType}
            self.time_index.clear()
            
            # Load memories
            for memory_id, memory_data in state.get("memories", {}).items():
                # Recreate the attributes
                attributes = MemoryAttributes(
                    created_at=datetime.fromisoformat(memory_data["attributes"]["created_at"]),
                    last_accessed=datetime.fromisoformat(memory_data["attributes"]["last_accessed"]),
                    access_count=memory_data["attributes"]["access_count"],
                    emotional_valence=memory_data["attributes"]["emotional_valence"],
                    emotional_intensity=memory_data["attributes"]["emotional_intensity"],
                    salience=memory_data["attributes"]["salience"],
                    confidence=memory_data["attributes"]["confidence"]
                )
                
                # Recreate the memory item
                memory_item = MemoryItem(
                    id=memory_data["id"],
                    memory_type=MemoryType(memory_data["memory_type"]),
                    attributes=attributes,
                    content=memory_data["content"],
                    tags=memory_data["tags"],
                    associations=memory_data["associations"]
                )
                
                # Add to memory storage
                self.memories[memory_id] = memory_item
            
            # Recreate indices
            for tag, memory_ids in state.get("tag_index", {}).items():
                self.tag_index[tag] = set(memory_ids)
            
            for memory_type_str, memory_ids in state.get("type_index", {}).items():
                self.type_index[MemoryType(memory_type_str)] = set(memory_ids)
            
            self.time_index = state.get("time_index", {})
            
            logger.info(f"Loaded memory state from {filepath} with {len(self.memories)} memories")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory state: {str(e)}")
            return False
    
    def clear_all_memories(self) -> None:
        """Clear all memories - use with caution!"""
        self.memories.clear()
        self.tag_index.clear()
        self.type_index = {memory_type: set() for memory_type in MemoryType}
        self.time_index.clear()
        
        logger.warning("Cleared all memories from memory manager")