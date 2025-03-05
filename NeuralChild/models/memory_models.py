from typing import Dict, List, Optional, Union, Literal, Any, TypeVar, Generic
from enum import Enum, auto
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import numpy as np
from typing import Generic, TypeVar

# Define a type variable for generic memory content
T = TypeVar('T')

class MemoryType(str, Enum):
    """Types of memory"""
    WORKING = "working"            # Currently active and manipulated
    EPISODIC = "episodic"          # Event-based memories
    SEMANTIC = "semantic"          # Factual knowledge
    PROCEDURAL = "procedural"      # How to do things
    EMOTIONAL = "emotional"        # Emotion-linked memories
    ASSOCIATIVE = "associative"    # Connection-based memories
    AUTOBIOGRAPHICAL = "autobiographical"  # Self-related memories

class MemoryStrength(float, Enum):
    """Memory strength levels"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class EmotionalValence(str, Enum):
    """Emotional coloring of memories"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class MemoryAccessibility(float, Enum):
    """How easily a memory can be accessed"""
    INACCESSIBLE = 0.0      # Cannot be recalled
    VERY_DIFFICULT = 0.2    # Only with significant triggers/effort
    DIFFICULT = 0.4         # Requires prompting
    MODERATE = 0.6          # Can be recalled with some effort
    EASY = 0.8              # Readily available
    AUTOMATIC = 1.0         # Comes to mind without effort

class MemoryStage(str, Enum):
    """Stages of memory processing"""
    ENCODING = "encoding"       # Initial formation
    CONSOLIDATION = "consolidation"  # Moving to long-term
    STABLE = "stable"           # Long-term storage
    RECONSOLIDATION = "reconsolidation"  # After recall
    DECLINING = "declining"     # Fading
    FORGOTTEN = "forgotten"     # No longer accessible

class MemoryAttributes(BaseModel):
    """Attributes that describe a memory's properties"""
    strength: float = Field(MemoryStrength.MODERATE, ge=0.0, le=1.0)
    emotional_valence: EmotionalValence = Field(EmotionalValence.NEUTRAL)
    emotional_intensity: float = Field(0.5, ge=0.0, le=1.0)
    accessibility: float = Field(MemoryAccessibility.MODERATE, ge=0.0, le=1.0)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    stage: MemoryStage = Field(MemoryStage.ENCODING)
    rehearsal_count: int = Field(0, ge=0)
    last_accessed: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)

class MemoryItem(BaseModel, Generic[T]):
    """A generic memory item that can hold different types of content"""
    id: UUID = Field(default_factory=uuid4)
    memory_type: MemoryType = Field(..., description="Type of memory")
    attributes: MemoryAttributes = Field(default_factory=MemoryAttributes)
    content: T = Field(..., description="Content of the memory item")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    associations: Dict[str, float] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic lookup")
    
    class Config:
        arbitrary_types_allowed = True

class WorkingMemoryConfig(BaseModel):
    """Configuration for working memory"""
    capacity: int = Field(7, description="Number of items that can be held simultaneously")
    decay_rate: float = Field(0.1, description="Rate at which items decay if not refreshed")
    max_duration: float = Field(60.0, description="Maximum time in seconds an item stays without rehearsal")
    attention_boost: float = Field(0.3, description="How much attention extends memory duration")

class WorkingMemory(BaseModel):
    """Model for working memory"""
    items: Dict[UUID, MemoryItem] = Field(default_factory=dict)
    config: WorkingMemoryConfig = Field(default_factory=WorkingMemoryConfig)
    current_load: int = Field(0, description="Current number of items in working memory")
    capacity_utilization: float = Field(0.0, description="Fraction of capacity used")
    
    @model_validator(mode='after')
    def update_metrics(self):
        """Update the memory metrics"""
        self.current_load = len(self.items)
        self.capacity_utilization = self.current_load / self.config.capacity if self.config.capacity > 0 else 0.0
        return self
    
    def add_item(self, item: MemoryItem) -> bool:
        """Add an item to working memory, potentially displacing older items"""
        if len(self.items) >= self.config.capacity:
            # Find the oldest item
            oldest_id = min(self.items.items(), 
                           key=lambda x: x[1].attributes.last_accessed).key
            self.items.pop(oldest_id)
        
        self.items[item.id] = item
        self.update_metrics()
        return True
    
    def get_item(self, item_id: UUID) -> Optional[MemoryItem]:
        """Retrieve an item, refreshing its access time"""
        if item_id in self.items:
            self.items[item_id].attributes.last_accessed = datetime.now()
            return self.items[item_id]
        return None
    
    def remove_item(self, item_id: UUID) -> bool:
        """Remove an item from working memory"""
        if item_id in self.items:
            self.items.pop(item_id)
            self.update_metrics()
            return True
        return False
    
    def update_decay(self, elapsed_seconds: float) -> List[UUID]:
        """Update decay for all items, return IDs of items that should be removed"""
        current_time = datetime.now()
        items_to_remove = []
        
        for item_id, item in self.items.items():
            time_diff = (current_time - item.attributes.last_accessed).total_seconds()
            if time_diff > self.config.max_duration:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            self.items.pop(item_id)
        
        self.update_metrics()
        return items_to_remove

class EpisodicMemoryConfig(BaseModel):
    """Configuration for episodic memory"""
    consolidation_threshold: float = Field(0.6, description="Threshold for memory consolidation")
    emotional_weight: float = Field(0.7, description="How much emotions affect memory strength")
    decay_rate: float = Field(0.01, description="Base rate of memory decay")
    coherence_factor: float = Field(0.5, description="How coherently episodes are stored")

class Episode(BaseModel):
    """An episodic memory representing an event sequence"""
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Brief description of the episode")
    memory_items: List[UUID] = Field(default_factory=list, description="Memory items in this episode")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    people_involved: List[str] = Field(default_factory=list)
    emotional_summary: Dict[str, float] = Field(default_factory=dict)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    coherence: float = Field(0.7, ge=0.0, le=1.0, description="How coherent the episode is")
    is_consolidated: bool = Field(False, description="Whether this episode is consolidated to long-term")
    accessibility: float = Field(0.8, ge=0.0, le=1.0, description="How accessible this memory is")
    last_accessed: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for this episode")

class EpisodicMemory(BaseModel):
    """Model for episodic memory"""
    episodes: Dict[UUID, Episode] = Field(default_factory=dict)
    current_episode: Optional[UUID] = None
    config: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    
    def create_episode(self, title: str, location: Optional[str] = None,
                      people_involved: List[str] = None) -> UUID:
        """Create a new episode"""
        episode = Episode(
            title=title,
            location=location,
            people_involved=people_involved or []
        )
        self.episodes[episode.id] = episode
        self.current_episode = episode.id
        return episode.id
    
    def add_to_episode(self, episode_id: UUID, memory_item_id: UUID) -> bool:
        """Add a memory item to an episode"""
        if episode_id in self.episodes:
            if memory_item_id not in self.episodes[episode_id].memory_items:
                self.episodes[episode_id].memory_items.append(memory_item_id)
            return True
        return False
    
    def end_episode(self, episode_id: UUID) -> bool:
        """Mark an episode as ended"""
        if episode_id in self.episodes:
            self.episodes[episode_id].end_time = datetime.now()
            if self.current_episode == episode_id:
                self.current_episode = None
            return True
        return False
    
    def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """Retrieve an episode, updating its access time"""
        if episode_id in self.episodes:
            self.episodes[episode_id].last_accessed = datetime.now()
            return self.episodes[episode_id]
        return None
    
    def update_episode_emotion(self, episode_id: UUID, emotion: str, intensity: float) -> bool:
        """Update the emotional summary of an episode"""
        if episode_id in self.episodes:
            self.episodes[episode_id].emotional_summary[emotion] = intensity
            return True
        return False

class LongTermMemoryConfig(BaseModel):
    """Configuration for long-term memory"""
    consolidation_time: float = Field(3600.0, description="Time in seconds for consolidation")
    retrieval_difficulty_factor: float = Field(0.3, description="Base difficulty for retrieving memories")
    forgetting_curve_factor: float = Field(0.1, description="How quickly memories fade")
    semantic_link_strength: float = Field(0.6, description="Strength of semantic connections")
    emotional_persistence_factor: float = Field(0.8, description="How persistent emotional memories are")

class LongTermMemoryDomain(str, Enum):
    """Domains of long-term memory"""
    PERSONAL = "personal"      # Self-related knowledge
    SOCIAL = "social"          # People-related knowledge
    PROCEDURAL = "procedural"  # Skills and how-to knowledge
    DECLARATIVE = "declarative"  # Facts and information
    EMOTIONAL = "emotional"    # Emotion-linked memories
    LINGUISTIC = "linguistic"  # Language-related knowledge
    VALUES = "values"          # Beliefs, values, cultural knowledge

class LongTermMemory(BaseModel):
    """Model for long-term memory storage"""
    items: Dict[UUID, MemoryItem] = Field(default_factory=dict)
    episodes: Dict[UUID, Episode] = Field(default_factory=dict)
    domain_indices: Dict[LongTermMemoryDomain, List[UUID]] = Field(
        default_factory=lambda: {domain: [] for domain in LongTermMemoryDomain}
    )
    config: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)
    
    def add_item(self, item: MemoryItem, domain: LongTermMemoryDomain) -> bool:
        """Add an item to long-term memory"""
        self.items[item.id] = item
        if item.id not in self.domain_indices[domain]:
            self.domain_indices[domain].append(item.id)
        return True
    
    def add_episode(self, episode: Episode) -> bool:
        """Add a consolidated episode to long-term memory"""
        self.episodes[episode.id] = episode
        episode.is_consolidated = True
        return True
    
    def get_item(self, item_id: UUID) -> Optional[MemoryItem]:
        """Retrieve an item from long-term memory"""
        if item_id in self.items:
            self.items[item_id].attributes.last_accessed = datetime.now()
            self.items[item_id].attributes.rehearsal_count += 1
            return self.items[item_id]
        return None
    
    def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """Retrieve an episode from long-term memory"""
        if episode_id in self.episodes:
            self.episodes[episode_id].last_accessed = datetime.now()
            return self.episodes[episode_id]
        return None
    
    def get_domain_items(self, domain: LongTermMemoryDomain) -> List[MemoryItem]:
        """Get all items in a specific memory domain"""
        return [self.items[item_id] for item_id in self.domain_indices[domain] 
                if item_id in self.items]
    
    def update_memory_strength(self, item_id: UUID, strength_delta: float) -> bool:
        """Update the strength of a memory (e.g., after retrieval)"""
        if item_id in self.items:
            current_strength = self.items[item_id].attributes.strength
            new_strength = max(0.0, min(1.0, current_strength + strength_delta))
            self.items[item_id].attributes.strength = new_strength
            return True
        return False

class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation processes"""
    working_to_episodic_threshold: float = Field(0.5, description="Threshold for moving from working to episodic")
    episodic_to_longterm_threshold: float = Field(0.7, description="Threshold for episodic to long-term")
    consolidation_rate: float = Field(0.1, description="Base rate of consolidation")
    sleep_consolidation_boost: float = Field(0.3, description="Extra consolidation during sleep")
    emotional_consolidation_factor: float = Field(0.4, description="How emotions affect consolidation")
    importance_weight: float = Field(0.6, description="Effect of importance on consolidation")

class ConsolidationStatus(str, Enum):
    """Status of memory consolidation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ConsolidationTask(BaseModel):
    """A task for memory consolidation"""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID = Field(..., description="ID of memory to consolidate")
    source_type: Literal["working", "episodic"] = Field(..., description="Type of source memory")
    target_type: Literal["episodic", "longterm"] = Field(..., description="Type of target memory")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Priority of this consolidation")
    status: ConsolidationStatus = Field(ConsolidationStatus.PENDING)
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress of consolidation")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def update_progress(self, progress_amount: float) -> bool:
        """Update the progress of consolidation"""
        if self.status == ConsolidationStatus.PENDING:
            self.status = ConsolidationStatus.IN_PROGRESS
        
        self.progress = min(1.0, self.progress + progress_amount)
        
        if self.progress >= 1.0:
            self.status = ConsolidationStatus.COMPLETED
            self.completed_at = datetime.now()
        
        return self.status == ConsolidationStatus.COMPLETED

class ConsolidationSystem(BaseModel):
    """System for managing memory consolidation processes"""
    tasks: Dict[UUID, ConsolidationTask] = Field(default_factory=dict)
    config: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    is_sleeping: bool = Field(False, description="Whether sleep consolidation is active")
    
    def create_task(self, source_id: UUID, source_type: Literal["working", "episodic"],
                   target_type: Literal["episodic", "longterm"], priority: float = 0.5) -> UUID:
        """Create a new consolidation task"""
        task = ConsolidationTask(
            source_id=source_id,
            source_type=source_type,
            target_type=target_type,
            priority=priority
        )
        self.tasks[task.id] = task
        return task.id
    
    def process_tasks(self, processing_time: float) -> List[UUID]:
        """Process consolidation tasks for a given amount of time, return completed tasks"""
        # Sort tasks by priority
        sorted_tasks = sorted(
            self.tasks.items(), 
            key=lambda x: x[1].priority if x[1].status != ConsolidationStatus.COMPLETED else -1, 
            reverse=True
        )
        
        # Determine consolidation rate
        rate = self.config.consolidation_rate
        if self.is_sleeping:
            rate += self.config.sleep_consolidation_boost
        
        # Process tasks
        completed_tasks = []
        
        for task_id, task in sorted_tasks:
            if task.status in [ConsolidationStatus.PENDING, ConsolidationStatus.IN_PROGRESS]:
                progress_amount = rate * processing_time * task.priority
                completed = task.update_progress(progress_amount)
                if completed:
                    completed_tasks.append(task_id)
        
        return completed_tasks
    
    def start_sleep_consolidation(self) -> None:
        """Start sleep-based memory consolidation"""
        self.is_sleeping = True
    
    def end_sleep_consolidation(self) -> None:
        """End sleep-based memory consolidation"""
        self.is_sleeping = False
    
    def get_task_status(self, task_id: UUID) -> Optional[ConsolidationStatus]:
        """Get the status of a consolidation task"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None