"""
Memory Module

This module handles memory storage, retrieval, and management.
It serves as the persistent storage system for the LMM, enabling
experience accumulation, knowledge retention, and adaptive memory
organization based on importance, recency, and emotional salience.

Components:
    - WorkingMemory: Manages short-term active memory
    - LongTermMemory: Manages persistent storage of memories
    - EpisodicMemory: Stores event-based memories (experiences)
    - SemanticMemory: Stores factual knowledge and concepts
    - AssociativeMemory: Manages links between related memories
    - MemoryNetwork: Neural networks for memory operations

The memory module develops in sophistication with age, gradually
improving in capacity, retention, and retrieval capabilities as
the mind matures.
"""

from typing import Optional, Dict, List, Any, Set, Tuple, Union
from uuid import UUID

from lmm_project.core.event_bus import EventBus
from lmm_project.utils.logging_utils import get_module_logger

from .models import (
    MemoryType,
    MemoryStatus,
    MemoryStrength,
    EmotionalValence,
    MemoryItem,
    WorkingMemoryItem,
    EpisodicMemoryItem,
    SemanticMemoryItem,
    AssociativeLink,
    MemoryQuery,
    MemoryRetrievalResult,
    ConsolidationCandidate,
    MemoryEvent,
    MemoryState,
    MemoryConfig
)

from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .associative_memory import AssociativeMemory
from .neural_net import MemoryNetwork

# Initialize logger
logger = get_module_logger("modules.memory")

# Module instances
_working_memory = None
_long_term_memory = None
_episodic_memory = None
_semantic_memory = None
_associative_memory = None
_memory_network = None
_current_config = None
_current_age = 0.0
_initialized = False

def initialize(
    event_bus: EventBus,
    config: Optional[MemoryConfig] = None,
    developmental_age: float = 0.0
) -> None:
    """
    Initialize the memory module.
    
    Args:
        event_bus: The event bus for communication
        config: Configuration for the memory module
        developmental_age: Current developmental age of the mind
    """
    global _working_memory, _long_term_memory, _episodic_memory
    global _semantic_memory, _associative_memory, _memory_network
    global _current_config, _current_age, _initialized
    
    # Store configuration and age
    _current_config = config or MemoryConfig()
    _current_age = developmental_age
    
    # Create components
    _working_memory = WorkingMemory(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _long_term_memory = LongTermMemory(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _episodic_memory = EpisodicMemory(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _semantic_memory = SemanticMemory(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _associative_memory = AssociativeMemory(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _memory_network = MemoryNetwork(
        event_bus=event_bus,
        config=_current_config,
        developmental_age=_current_age
    )
    
    _initialized = True
    
    logger.info(f"Memory module initialized with age {developmental_age}")

def store_working_memory(
    content: Dict[str, Any],
    source: Optional[str] = None,
    importance: float = 0.5,
    emotional_valence: float = EmotionalValence.NEUTRAL,
    tags: Optional[Set[str]] = None
) -> Optional[WorkingMemoryItem]:
    """
    Store an item in working memory.
    
    Args:
        content: The content to store
        source: Source of the memory
        importance: Importance of the item
        emotional_valence: Emotional valence of the memory
        tags: Optional tags for categorization
        
    Returns:
        The created working memory item or None if failed
    """
    _ensure_initialized()
    return _working_memory.store_item(
        content=content,
        source=source,
        importance=importance,
        emotional_valence=emotional_valence,
        tags=tags or set()
    )

def store_episodic_memory(
    content: Dict[str, Any],
    actors: List[str] = None,
    location: Optional[str] = None,
    narrative: Optional[str] = None,
    emotional_valence: float = EmotionalValence.NEUTRAL,
    tags: Optional[Set[str]] = None
) -> Optional[EpisodicMemoryItem]:
    """
    Store an episodic memory (experience).
    
    Args:
        content: The content of the experience
        actors: People/entities involved in the experience
        location: Where the experience occurred
        narrative: Text description of the experience
        emotional_valence: Emotional tone of the memory
        tags: Optional tags for categorization
        
    Returns:
        The created episodic memory item or None if failed
    """
    _ensure_initialized()
    return _episodic_memory.store_memory(
        content=content,
        actors=actors or [],
        location=location,
        narrative=narrative,
        emotional_valence=emotional_valence,
        tags=tags or set()
    )

def store_semantic_memory(
    concept: str,
    definition: str,
    properties: Dict[str, Any] = None,
    examples: List[str] = None,
    related_concepts: List[str] = None,
    confidence: float = 0.7,
    tags: Optional[Set[str]] = None
) -> Optional[SemanticMemoryItem]:
    """
    Store a semantic memory (factual knowledge).
    
    Args:
        concept: The concept name
        definition: Definition of the concept
        properties: Properties of the concept
        examples: Examples of the concept
        related_concepts: Related concepts
        confidence: Confidence in this knowledge
        tags: Optional tags for categorization
        
    Returns:
        The created semantic memory item or None if failed
    """
    _ensure_initialized()
    return _semantic_memory.store_memory(
        concept=concept,
        definition=definition,
        properties=properties or {},
        examples=examples or [],
        related_concepts=related_concepts or [],
        confidence=confidence,
        tags=tags or set()
    )

def create_association(
    source_id: UUID,
    target_id: UUID,
    association_type: str,
    strength: float = 0.5
) -> Optional[AssociativeLink]:
    """
    Create an association between two memory items.
    
    Args:
        source_id: ID of the source memory item
        target_id: ID of the target memory item
        association_type: Type of association
        strength: Initial strength of association
        
    Returns:
        The created associative link or None if failed
    """
    _ensure_initialized()
    return _associative_memory.create_link(
        source_id=source_id,
        target_id=target_id,
        association_type=association_type,
        strength=strength
    )

def retrieve_memory(query: MemoryQuery) -> MemoryRetrievalResult:
    """
    Retrieve memories based on a query.
    
    Args:
        query: The query for memory retrieval
        
    Returns:
        Results of the memory retrieval
    """
    _ensure_initialized()
    result = _long_term_memory.retrieve_memories(query)
    
    # For relevant memories, update their access time and reinforce
    for memory_id in result.memory_items:
        _long_term_memory.reinforce_memory(memory_id)
    
    return result

def get_associations(memory_id: UUID, min_strength: float = 0.3) -> List[AssociativeLink]:
    """
    Get associations for a memory item.
    
    Args:
        memory_id: ID of the memory item
        min_strength: Minimum association strength
        
    Returns:
        List of associations for the memory item
    """
    _ensure_initialized()
    return _associative_memory.get_links_for_memory(memory_id, min_strength)

def get_working_memory_items() -> List[WorkingMemoryItem]:
    """
    Get current working memory items.
    
    Returns:
        List of items in working memory
    """
    _ensure_initialized()
    return _working_memory.get_items()

def get_state() -> MemoryState:
    """
    Get the current state of the memory module.
    
    Returns:
        Current memory state
    """
    _ensure_initialized()
    
    wm_items = _working_memory.get_items()
    
    return MemoryState(
        working_memory_capacity=_current_config.working_memory_capacity,
        working_memory_used=len(wm_items),
        working_memory_items={item.id: item.dict() for item in wm_items},
        episodic_memory_count=_episodic_memory.get_count(),
        semantic_memory_count=_semantic_memory.get_count(),
        associative_link_count=_associative_memory.get_link_count(),
        recent_retrievals=_long_term_memory.get_recent_retrievals(),
        developmental_age=_current_age
    )

def reinforce_memory(memory_id: UUID, amount: float = 0.1) -> bool:
    """
    Reinforce a memory to strengthen it.
    
    Args:
        memory_id: ID of the memory to reinforce
        amount: Amount to reinforce by
        
    Returns:
        Whether the reinforcement was successful
    """
    _ensure_initialized()
    return _long_term_memory.reinforce_memory(memory_id, amount)

def update_age(new_age: float) -> None:
    """
    Update the developmental age of the memory module.
    
    Args:
        new_age: The new developmental age
    """
    global _current_age
    
    _ensure_initialized()
    _current_age = new_age
    
    # Update components
    _working_memory.update_developmental_age(new_age)
    _long_term_memory.update_developmental_age(new_age)
    _episodic_memory.update_developmental_age(new_age)
    _semantic_memory.update_developmental_age(new_age)
    _associative_memory.update_developmental_age(new_age)
    _memory_network.update_developmental_age(new_age)
    
    logger.info(f"Memory module age updated to {new_age}")

def forget_memory(memory_id: UUID) -> bool:
    """
    Explicitly forget a memory.
    
    Args:
        memory_id: ID of the memory to forget
        
    Returns:
        Whether the memory was forgotten
    """
    _ensure_initialized()
    return _long_term_memory.forget_memory(memory_id)

def process_consolidation() -> int:
    """
    Trigger memory consolidation process.
    
    Returns:
        Number of memories consolidated
    """
    _ensure_initialized()
    candidates = _working_memory.get_consolidation_candidates()
    return _long_term_memory.consolidate_memories(candidates)

def _ensure_initialized() -> None:
    """
    Ensure the memory module is initialized.
    
    Raises:
        RuntimeError: If the module is not initialized
    """
    if not _initialized:
        raise RuntimeError(
            "Memory module not initialized. Call initialize() first."
        )

# Export public API
__all__ = [
    # Classes
    'WorkingMemory',
    'LongTermMemory',
    'EpisodicMemory',
    'SemanticMemory',
    'AssociativeMemory',
    'MemoryNetwork',
    
    # Models
    'MemoryType',
    'MemoryStatus',
    'MemoryStrength',
    'EmotionalValence',
    'MemoryItem',
    'WorkingMemoryItem',
    'EpisodicMemoryItem',
    'SemanticMemoryItem',
    'AssociativeLink',
    'MemoryQuery',
    'MemoryRetrievalResult',
    'MemoryEvent',
    'MemoryState',
    'MemoryConfig',
    
    # Functions
    'initialize',
    'store_working_memory',
    'store_episodic_memory',
    'store_semantic_memory',
    'create_association',
    'retrieve_memory',
    'get_associations',
    'get_working_memory_items',
    'get_state',
    'reinforce_memory',
    'update_age',
    'forget_memory',
    'process_consolidation'
]
