# episodic_memory.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypeVar, Generic
import logging
import json
import os
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum, auto
import heapq
import random

# Import from memory manager and other memory components
from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority
from memory.long_term_memory import LongTermMemory, KnowledgeDomain
from memory.working_memory import WorkingMemory
from memory.associative_memory import AssociativeMemory, AssociationType

# Set up logging
logger = logging.getLogger("EpisodicMemory")

class EpisodeType(str, Enum):
    """Types of episodic memories"""
    INTERACTION = "interaction"  # Interactions with others
    OBSERVATION = "observation"  # Observations of environment
    REFLECTION = "reflection"    # Internal reflection/thought
    EMOTION = "emotion"          # Emotional experience
    ACHIEVEMENT = "achievement"  # Achievement or milestone
    FAILURE = "failure"          # Failure or setback

class EpisodeImportance(Enum):
    """Importance levels for episodic memories"""
    TRIVIAL = auto()     # Everyday, forgettable
    MINOR = auto()       # Somewhat memorable
    SIGNIFICANT = auto() # Important but not life-changing
    MAJOR = auto()       # Very important, life episode
    DEFINING = auto()    # Identity-defining moments

class TimeContext(BaseModel):
    """Temporal context for an episode"""
    timestamp: datetime = Field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    temporal_references: List[str] = Field(default_factory=list)  # e.g., "morning", "after dinner"
    sequence_position: Optional[int] = None  # Position in a sequence of events
    developmental_age: Optional[float] = None  # Age in days when this occurred

class SpatialContext(BaseModel):
    """Spatial context for an episode"""
    location: Optional[str] = None
    environment: Optional[str] = None
    spatial_references: List[str] = Field(default_factory=list)  # e.g., "near", "behind"

class ParticipantReference(BaseModel):
    """Reference to a participant in an episode"""
    entity_id: str = Field(..., description="ID of the entity")
    role: str = Field(..., description="Role in the episode")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance to the episode")
    relationship: Optional[str] = None  # Relationship to self

class EpisodeContent(BaseModel):
    """Content of an episodic memory"""
    summary: str = Field(..., description="Brief summary of the episode")
    details: Optional[str] = None  # More detailed description
    dialogue: Optional[List[Dict[str, str]]] = None  # Dialogue exchanges
    actions: Optional[List[str]] = None  # Actions that occurred
    outcomes: Optional[List[str]] = None  # Results of the episode
    interpretations: Optional[List[str]] = None  # Personal interpretations

class EmotionalImpact(BaseModel):
    """Emotional impact of an episode"""
    primary_emotion: str = Field(..., description="Primary emotion experienced")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity of emotion")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence")
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)  # Other emotions
    current_relevance: float = Field(0.5, ge=0.0, le=1.0)  # How relevant to current emotional state

class EpisodicMemoryItem(BaseModel):
    """An episodic memory item with context and content"""
    memory_id: str = Field(..., description="ID of the memory item")
    episode_type: EpisodeType = Field(..., description="Type of episode")
    importance: EpisodeImportance = Field(EpisodeImportance.MINOR, description="Importance level")
    rehearsal_count: int = Field(0, ge=0, description="How many times recalled")
    
    # Context
    time_context: TimeContext = Field(default_factory=TimeContext)
    spatial_context: Optional[SpatialContext] = None
    participants: List[ParticipantReference] = Field(default_factory=list)
    
    # Content
    content: EpisodeContent = Field(..., description="Episode content")
    emotional_impact: EmotionalImpact = Field(..., description="Emotional impact")
    
    # Related memories
    related_episodes: List[str] = Field(default_factory=list)  # IDs of related episodes
    related_concepts: List[str] = Field(default_factory=list)  # IDs of related semantic concepts
    
    # Retrieval metadata
    last_retrieved: datetime = Field(default_factory=datetime.now)
    retrieval_probability: float = Field(0.5, ge=0.0, le=1.0)  # Probability of spontaneous recall
    
    def rehearse(self) -> None:
        """Rehearse this memory, making it more accessible"""
        self.rehearsal_count += 1
        self.last_retrieved = datetime.now()
        
        # Increase retrieval probability with each rehearsal, with diminishing returns
        self.retrieval_probability = min(0.95, self.retrieval_probability + (0.1 * (1.0 - self.retrieval_probability)))
    
    def add_related_episode(self, episode_id: str) -> None:
        """Add a related episode"""
        if episode_id not in self.related_episodes:
            self.related_episodes.append(episode_id)
    
    def add_related_concept(self, concept_id: str) -> None:
        """Add a related concept"""
        if concept_id not in self.related_concepts:
            self.related_concepts.append(concept_id)
    
    def update_emotional_relevance(self, current_emotions: Dict[str, float]) -> float:
        """Update emotional relevance based on current emotional state
        
        Returns:
            float: The calculated relevance score
        """
        relevance = 0.0
        
        # Check primary emotion match
        if self.emotional_impact.primary_emotion in current_emotions:
            relevance += current_emotions[self.emotional_impact.primary_emotion] * 0.6
        
        # Check secondary emotions
        for emotion, intensity in self.emotional_impact.secondary_emotions.items():
            if emotion in current_emotions:
                relevance += current_emotions[emotion] * intensity * 0.3
        
        # Cap relevance at 1.0
        relevance = min(1.0, relevance)
        
        # Update emotional relevance
        self.emotional_impact.current_relevance = relevance
        
        return relevance
    
    def calculate_retrieval_probability(self, time_factor: float = 0.5) -> float:
        """Calculate probability of retrieval based on multiple factors
        
        Args:
            time_factor: Weight of time decay in calculation (0-1)
            
        Returns:
            float: Retrieval probability (0-1)
        """
        # Calculate time decay
        days_since_retrieval = (datetime.now() - self.last_retrieved).days
        time_decay = max(0.0, 1.0 - (days_since_retrieval * 0.01))
        
        # Calculate rehearsal factor
        rehearsal_factor = min(1.0, 0.3 + (self.rehearsal_count * 0.05))
        
        # Calculate importance factor
        importance_values = {
            EpisodeImportance.TRIVIAL: 0.2,
            EpisodeImportance.MINOR: 0.4,
            EpisodeImportance.SIGNIFICANT: 0.6,
            EpisodeImportance.MAJOR: 0.8,
            EpisodeImportance.DEFINING: 1.0
        }
        importance_factor = importance_values[self.importance]
        
        # Calculate emotional factor
        emotional_factor = self.emotional_impact.intensity * 0.8
        
        # Combine factors with weights
        probability = (
            (time_decay * time_factor) +
            (rehearsal_factor * 0.2) +
            (importance_factor * 0.2) +
            (emotional_factor * 0.1)
        ) / (time_factor + 0.2 + 0.2 + 0.1)
        
        # Update stored probability
        self.retrieval_probability = probability
        
        return probability

class EpisodicMemory:
    """Event-based memory of experiences"""
    
    def __init__(self, data_dir: Optional[Path] = None, capacity: int = 1000):
        """Initialize episodic memory
        
        Args:
            data_dir: Directory for persistent storage
            capacity: Maximum number of episodes to store
        """
        self.data_dir = data_dir or Path("./data/memory/episodic")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes: Dict[str, EpisodicMemoryItem] = {}
        self.capacity = capacity
        
        # Indices for efficient retrieval
        self.temporal_index: Dict[str, List[str]] = {}  # time bucket -> memory IDs
        self.type_index: Dict[EpisodeType, List[str]] = {
            episode_type: [] for episode_type in EpisodeType
        }
        self.participant_index: Dict[str, List[str]] = {}  # entity_id -> memory IDs
        self.emotion_index: Dict[str, List[str]] = {}  # emotion -> memory IDs
        
        # References to other memory systems
        self.memory_manager: Optional[MemoryManager] = None
        self.long_term_memory: Optional[LongTermMemory] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.associative_memory: Optional[AssociativeMemory] = None
        
        logger.info(f"Episodic memory initialized with capacity {capacity}")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def set_other_memory_systems(self, 
                               long_term_memory: Optional[LongTermMemory] = None,
                               working_memory: Optional[WorkingMemory] = None,
                               associative_memory: Optional[AssociativeMemory] = None) -> None:
        """Set references to other memory systems"""
        self.long_term_memory = long_term_memory
        self.working_memory = working_memory
        self.associative_memory = associative_memory
    
    def add_memory(self, memory_item: MemoryItem) -> str:
        """Add a memory item as an episodic memory"""
        memory_id = memory_item.id
        
        # Check if already exists
        if memory_id in self.episodes:
            logger.info(f"Episode already exists: {memory_id}")
            return memory_id
        
        # Check capacity
        if len(self.episodes) >= self.capacity:
            self._free_space()
        
        # Extract episodic details from memory content
        content = memory_item.content
        episode_type = EpisodeType.INTERACTION  # Default
        summary = ""
        details = None
        
        # Parse based on content type
        if isinstance(content, dict):
            # Extract episode type
            if "type" in content:
                try:
                    episode_type = EpisodeType(content["type"])
                except ValueError:
                    episode_type = EpisodeType.INTERACTION
            
            # Extract content fields
            summary = content.get("summary", "")
            details = content.get("details")
            dialogue = content.get("dialogue")
            actions = content.get("actions")
            outcomes = content.get("outcomes")
            interpretations = content.get("interpretations")
            
            # Extract participants
            participants = []
            if "participants" in content:
                for participant in content["participants"]:
                    if isinstance(participant, dict):
                        entity_id = participant.get("entity_id", participant.get("id", "unknown"))
                        role = participant.get("role", "unknown")
                        importance = participant.get("importance", 0.5)
                        relationship = participant.get("relationship")
                        
                        participants.append(ParticipantReference(
                            entity_id=entity_id,
                            role=role,
                            importance=importance,
                            relationship=relationship
                        ))
            
            # Extract temporal context
            time_context = TimeContext()
            if "time_context" in content:
                tc = content["time_context"]
                if isinstance(tc, dict):
                    # Use memory timestamp as default
                    time_context.timestamp = memory_item.attributes.created_at
                    
                    # Extract duration if provided as minutes
                    if "duration_minutes" in tc:
                        minutes = tc["duration_minutes"]
                        if isinstance(minutes, (int, float)):
                            time_context.duration = timedelta(minutes=minutes)
                    
                    # Extract other fields
                    if "temporal_references" in tc:
                        time_context.temporal_references = tc["temporal_references"]
                    if "sequence_position" in tc:
                        time_context.sequence_position = tc["sequence_position"]
                    if "developmental_age" in tc:
                        time_context.developmental_age = tc["developmental_age"]
            
            # Extract spatial context
            spatial_context = None
            if "spatial_context" in content:
                sc = content["spatial_context"]
                if isinstance(sc, dict):
                    spatial_context = SpatialContext(
                        location=sc.get("location"),
                        environment=sc.get("environment"),
                        spatial_references=sc.get("spatial_references", [])
                    )
            
            # Extract emotional impact
            emotional_impact = None
            if "emotional_impact" in content:
                ei = content["emotional_impact"]
                if isinstance(ei, dict):
                    primary_emotion = ei.get("primary_emotion", "neutral")
                    intensity = ei.get("intensity", 0.5)
                    valence = ei.get("valence", 0.0)
                    secondary_emotions = ei.get("secondary_emotions", {})
                    
                    emotional_impact = EmotionalImpact(
                        primary_emotion=primary_emotion,
                        intensity=intensity,
                        valence=valence,
                        secondary_emotions=secondary_emotions
                    )
            
            # If no emotional impact provided, create from memory attributes
            if not emotional_impact:
                valence = memory_item.attributes.emotional_valence
                intensity = memory_item.attributes.emotional_intensity
                
                # Determine primary emotion based on valence
                if valence > 0.3:
                    primary_emotion = "joy"
                elif valence < -0.3:
                    primary_emotion = "sadness"
                else:
                    primary_emotion = "neutral"
                
                emotional_impact = EmotionalImpact(
                    primary_emotion=primary_emotion,
                    intensity=intensity,
                    valence=valence
                )
            
            # Extract related memories
            related_episodes = content.get("related_episodes", [])
            related_concepts = content.get("related_concepts", [])
            
            # Extract importance
            importance = EpisodeImportance.MINOR  # Default
            if "importance" in content:
                importance_str = content["importance"]
                if isinstance(importance_str, str):
                    try:
                        importance = EpisodeImportance[importance_str.upper()]
                    except KeyError:
                        # If string doesn't match enum, interpret based on value
                        if importance_str.lower() in ["high", "major", "very important"]:
                            importance = EpisodeImportance.MAJOR
                        elif importance_str.lower() in ["medium", "moderate", "significant"]:
                            importance = EpisodeImportance.SIGNIFICANT
                        elif importance_str.lower() in ["low", "minor"]:
                            importance = EpisodeImportance.MINOR
                        elif importance_str.lower() in ["trivial", "minimal"]:
                            importance = EpisodeImportance.TRIVIAL
                        elif importance_str.lower() in ["critical", "defining", "core"]:
                            importance = EpisodeImportance.DEFINING
        elif isinstance(content, str):
            # Simple string content - treat as summary
            summary = content
            
            # Create minimal episode data
            emotional_impact = EmotionalImpact(
                primary_emotion="neutral",
                intensity=memory_item.attributes.emotional_intensity,
                valence=memory_item.attributes.emotional_valence
            )
            
            importance = EpisodeImportance.MINOR
            
            # Determine episode type from tags if available
            if hasattr(memory_item, "tags") and memory_item.tags:
                for tag in memory_item.tags:
                    if tag in [et.value for et in EpisodeType]:
                        episode_type = EpisodeType(tag)
                        break
        else:
            # Unsupported content type
            logger.warning(f"Unsupported content type for episodic memory: {type(content)}")
            return memory_id
        
        # Determine importance from emotional intensity if not explicitly set
        if importance == EpisodeImportance.MINOR and isinstance(emotional_impact, EmotionalImpact):
            if emotional_impact.intensity > 0.8:
                importance = EpisodeImportance.MAJOR
            elif emotional_impact.intensity > 0.6:
                importance = EpisodeImportance.SIGNIFICANT
        
        # Create episode content
        episode_content = EpisodeContent(
            summary=summary,
            details=details,
            dialogue=dialogue if 'dialogue' in locals() else None,
            actions=actions if 'actions' in locals() else None,
            outcomes=outcomes if 'outcomes' in locals() else None,
            interpretations=interpretations if 'interpretations' in locals() else None
        )
        
        # Create episodic memory item
        episodic_item = EpisodicMemoryItem(
            memory_id=memory_id,
            episode_type=episode_type,
            importance=importance,
            time_context=time_context if 'time_context' in locals() else TimeContext(),
            spatial_context=spatial_context if 'spatial_context' in locals() else None,
            participants=participants if 'participants' in locals() else [],
            content=episode_content,
            emotional_impact=emotional_impact,
            related_episodes=related_episodes if 'related_episodes' in locals() else [],
            related_concepts=related_concepts if 'related_concepts' in locals() else []
        )
        
        # Store the episode
        self.episodes[memory_id] = episodic_item
        
        # Update indices
        self._update_indices(episodic_item)
        
        # Create associations for related memories
        self._create_associations(episodic_item)
        
        logger.info(f"Added episodic memory: {memory_id} ({episode_type.value})")
        return memory_id
    
    def _update_indices(self, episode: EpisodicMemoryItem) -> None:
        """Update memory indices for efficient retrieval"""
        memory_id = episode.memory_id
        
        # Update temporal index
        time_bucket = episode.time_context.timestamp.strftime("%Y-%m-%d")
        if time_bucket not in self.temporal_index:
            self.temporal_index[time_bucket] = []
        self.temporal_index[time_bucket].append(memory_id)
        
        # Update type index
        self.type_index[episode.episode_type].append(memory_id)
        
        # Update participant index
        for participant in episode.participants:
            entity_id = participant.entity_id
            if entity_id not in self.participant_index:
                self.participant_index[entity_id] = []
            self.participant_index[entity_id].append(memory_id)
        
        # Update emotion index
        primary_emotion = episode.emotional_impact.primary_emotion
        if primary_emotion not in self.emotion_index:
            self.emotion_index[primary_emotion] = []
        self.emotion_index[primary_emotion].append(memory_id)
        
        # Also index secondary emotions
        for emotion in episode.emotional_impact.secondary_emotions:
            if emotion not in self.emotion_index:
                self.emotion_index[emotion] = []
            self.emotion_index[emotion].append(memory_id)
    
    def _create_associations(self, episode: EpisodicMemoryItem) -> None:
        """Create associations for related memories"""
        if not self.associative_memory:
            return
        
        memory_id = episode.memory_id
        
        # Create associations for related episodes
        for related_id in episode.related_episodes:
            if related_id in self.episodes:
                self.associative_memory.create_association(
                    source_id=memory_id,
                    target_id=related_id,
                    strength=0.7,
                    association_type=AssociationType.TEMPORAL,
                    bidirectional=True
                )
        
        # Create associations for related concepts
        for concept_id in episode.related_concepts:
            self.associative_memory.create_association(
                source_id=memory_id,
                target_id=concept_id,
                strength=0.6,
                association_type=AssociationType.SEMANTIC,
                bidirectional=False
            )
        
        # Create associations for participants
        for participant in episode.participants:
            entity_id = participant.entity_id
            self.associative_memory.create_association(
                source_id=memory_id,
                target_id=entity_id,
                strength=participant.importance,
                association_type=AssociationType.SEMANTIC,
                bidirectional=False
            )
    
    def _free_space(self) -> None:
        """Free space by removing least important memories"""
        if len(self.episodes) < self.capacity:
            return
        
        # Calculate scores for all episodes
        episode_scores = {}
        current_time = datetime.now()
        
        for memory_id, episode in self.episodes.items():
            # Calculate recency (higher is more recent)
            days_old = (current_time - episode.time_context.timestamp).days
            recency_score = max(0.0, 1.0 - (days_old / 365.0))  # Scale by year
            
            # Get importance score
            importance_values = {
                EpisodeImportance.TRIVIAL: 0.2,
                EpisodeImportance.MINOR: 0.4,
                EpisodeImportance.SIGNIFICANT: 0.6,
                EpisodeImportance.MAJOR: 0.8,
                EpisodeImportance.DEFINING: 1.0
            }
            importance_score = importance_values[episode.importance]
            
            # Calculate recall frequency score
            recall_score = min(1.0, episode.rehearsal_count / 10.0)
            
            # Calculate emotional intensity score
            emotional_score = episode.emotional_impact.intensity
            
            # Combined score with weighted components
            score = (
                recency_score * 0.2 +
                importance_score * 0.4 +
                recall_score * 0.2 +
                emotional_score * 0.2
            )
            
            episode_scores[memory_id] = score
        
        # Sort by score (ascending) and remove lowest-scoring episodes
        to_remove = sorted(episode_scores.keys(), key=lambda k: episode_scores[k])[:5]
        
        for memory_id in to_remove:
            self._remove_from_indices(self.episodes[memory_id])
            del self.episodes[memory_id]
            logger.info(f"Removed low-scoring episodic memory to free space: {memory_id}")
    
    def _remove_from_indices(self, episode: EpisodicMemoryItem) -> None:
        """Remove an episode from all indices"""
        memory_id = episode.memory_id
        
        # Remove from temporal index
        time_bucket = episode.time_context.timestamp.strftime("%Y-%m-%d")
        if time_bucket in self.temporal_index and memory_id in self.temporal_index[time_bucket]:
            self.temporal_index[time_bucket].remove(memory_id)
        
        # Remove from type index
        if memory_id in self.type_index[episode.episode_type]:
            self.type_index[episode.episode_type].remove(memory_id)
        
        # Remove from participant index
        for participant in episode.participants:
            entity_id = participant.entity_id
            if entity_id in self.participant_index and memory_id in self.participant_index[entity_id]:
                self.participant_index[entity_id].remove(memory_id)
        
        # Remove from emotion index
        primary_emotion = episode.emotional_impact.primary_emotion
        if primary_emotion in self.emotion_index and memory_id in self.emotion_index[primary_emotion]:
            self.emotion_index[primary_emotion].remove(memory_id)
        
        # Remove from secondary emotion indices
        for emotion in episode.emotional_impact.secondary_emotions:
            if emotion in self.emotion_index and memory_id in self.emotion_index[emotion]:
                self.emotion_index[emotion].remove(memory_id)
    
    def recall_memory(self, memory_id: str) -> Optional[EpisodicMemoryItem]:
        """Recall an episodic memory"""
        if memory_id not in self.episodes:
            return None
        
        # Get the episode
        episode = self.episodes[memory_id]
        
        # Update rehearsal count and timestamp
        episode.rehearse()
        
        # Update retrieval probability
        episode.calculate_retrieval_probability()
        
        # If associative memory is available, activate associations
        if self.associative_memory:
            for related_id in episode.related_episodes:
                if related_id in self.episodes:
                    self.associative_memory.activate_association(f"assoc_{memory_id}_{related_id}")
            
            for concept_id in episode.related_concepts:
                self.associative_memory.activate_association(f"assoc_{memory_id}_{concept_id}")
        
        logger.info(f"Recalled episodic memory: {memory_id}")
        return episode
    
    def update_memory(self, memory_id: str, memory_item: Optional[MemoryItem] = None) -> bool:
        """Update an episodic memory"""
        if memory_id not in self.episodes:
            return False
        
        # Get the episode
        episode = self.episodes[memory_id]
        
        # Update rehearsal info
        episode.rehearse()
        
        # Update from memory item if provided
        if memory_item and isinstance(memory_item.content, dict):
            content = memory_item.content
            
            # Update content fields if provided
            if "summary" in content:
                episode.content.summary = content["summary"]
            if "details" in content:
                episode.content.details = content["details"]
            if "dialogue" in content:
                episode.content.dialogue = content["dialogue"]
            if "actions" in content:
                episode.content.actions = content["actions"]
            if "outcomes" in content:
                episode.content.outcomes = content["outcomes"]
            if "interpretations" in content:
                episode.content.interpretations = content["interpretations"]
            
            # Update related memories if provided
            if "related_episodes" in content:
                episode.related_episodes = content["related_episodes"]
            if "related_concepts" in content:
                episode.related_concepts = content["related_concepts"]
            
            # Update importance if provided
            if "importance" in content:
                importance_str = content["importance"]
                if isinstance(importance_str, str):
                    try:
                        episode.importance = EpisodeImportance[importance_str.upper()]
                    except KeyError:
                        # Map string values to enum values
                        if importance_str.lower() in ["high", "major", "very important"]:
                            episode.importance = EpisodeImportance.MAJOR
                        elif importance_str.lower() in ["medium", "moderate", "significant"]:
                            episode.importance = EpisodeImportance.SIGNIFICANT
                        elif importance_str.lower() in ["low", "minor"]:
                            episode.importance = EpisodeImportance.MINOR
                        elif importance_str.lower() in ["trivial", "minimal"]:
                            episode.importance = EpisodeImportance.TRIVIAL
                        elif importance_str.lower() in ["critical", "defining", "core"]:
                            episode.importance = EpisodeImportance.DEFINING
            
            # Update emotional impact if provided
            if "emotional_impact" in content:
                ei = content["emotional_impact"]
                if isinstance(ei, dict):
                    if "primary_emotion" in ei:
                        episode.emotional_impact.primary_emotion = ei["primary_emotion"]
                    if "intensity" in ei:
                        episode.emotional_impact.intensity = ei["intensity"]
                    if "valence" in ei:
                        episode.emotional_impact.valence = ei["valence"]
                    if "secondary_emotions" in ei:
                        episode.emotional_impact.secondary_emotions = ei["secondary_emotions"]
        
        # Recreate associations with updated data
        if self.associative_memory:
            self._create_associations(episode)
        
        logger.info(f"Updated episodic memory: {memory_id}")
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete an episodic memory"""
        if memory_id not in self.episodes:
            return False
        
        # Get the episode
        episode = self.episodes[memory_id]
        
        # Remove from indices
        self._remove_from_indices(episode)
        
        # Remove the episode
        del self.episodes[memory_id]
        
        logger.info(f"Deleted episodic memory: {memory_id}")
        return True
    
    def find_memories_by_type(self, episode_type: EpisodeType, limit: int = 10) -> List[str]:
        """Find memories of a specific type"""
        if episode_type not in self.type_index:
            return []
        
        # Get memory IDs for this type
        memory_ids = self.type_index[episode_type]
        
        # Sort by recency
        sorted_ids = sorted(
            memory_ids,
            key=lambda mid: self.episodes[mid].time_context.timestamp if mid in self.episodes else datetime.min,
            reverse=True
        )
        
        return sorted_ids[:limit]
    
    def find_memories_by_emotion(self, emotion: str, limit: int = 10) -> List[str]:
        """Find memories with a specific emotion"""
        if emotion not in self.emotion_index:
            return []
        
        # Get memory IDs for this emotion
        memory_ids = self.emotion_index[emotion]
        
        # Sort by intensity
        sorted_ids = []
        for mid in memory_ids:
            if mid in self.episodes:
                episode = self.episodes[mid]
                intensity = 0.0
                
                if episode.emotional_impact.primary_emotion == emotion:
                    intensity = episode.emotional_impact.intensity
                elif emotion in episode.emotional_impact.secondary_emotions:
                    intensity = episode.emotional_impact.secondary_emotions[emotion]
                
                sorted_ids.append((mid, intensity))
        
        sorted_ids.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in sorted_ids[:limit]]
    
    def find_memories_by_participant(self, entity_id: str, limit: int = 10) -> List[str]:
        """Find memories involving a specific participant"""
        if entity_id not in self.participant_index:
            return []
        
        # Get memory IDs for this participant
        memory_ids = self.participant_index[entity_id]
        
        # Sort by importance of participant in the memory
        sorted_ids = []
        for mid in memory_ids:
            if mid in self.episodes:
                episode = self.episodes[mid]
                importance = 0.0
                
                for participant in episode.participants:
                    if participant.entity_id == entity_id:
                        importance = participant.importance
                        break
                
                sorted_ids.append((mid, importance))
        
        sorted_ids.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in sorted_ids[:limit]]
    
    def find_memories_by_timeframe(self, start_time: datetime, end_time: datetime,
                                 limit: int = 10) -> List[str]:
        """Find memories within a specific timeframe"""
        relevant_ids = []
        
        # Find all relevant time buckets
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            time_bucket = current_date.strftime("%Y-%m-%d")
            if time_bucket in self.temporal_index:
                relevant_ids.extend(self.temporal_index[time_bucket])
            current_date += timedelta(days=1)
        
        # Filter by exact time range
        filtered_ids = []
        for mid in set(relevant_ids):
            if mid in self.episodes:
                timestamp = self.episodes[mid].time_context.timestamp
                if start_time <= timestamp <= end_time:
                    filtered_ids.append(mid)
        
        # Sort by recency (most recent first)
        sorted_ids = sorted(
            filtered_ids,
            key=lambda mid: self.episodes[mid].time_context.timestamp,
            reverse=True
        )
        
        return sorted_ids[:limit]
    
    def find_related_memories(self, memory_id: str, limit: int = 10) -> List[str]:
        """Find memories related to a specific memory"""
        if memory_id not in self.episodes:
            return []
        
        episode = self.episodes[memory_id]
        
        # Start with explicitly related episodes
        related_ids = episode.related_episodes.copy()
        
        # Find temporally adjacent memories
        timestamp = episode.time_context.timestamp
        time_bucket = timestamp.strftime("%Y-%m-%d")
        
        if time_bucket in self.temporal_index:
            for mid in self.temporal_index[time_bucket]:
                if mid != memory_id and mid not in related_ids and mid in self.episodes:
                    # Check temporal proximity
                    other_timestamp = self.episodes[mid].time_context.timestamp
                    time_diff = abs((timestamp - other_timestamp).total_seconds())
                    if time_diff < 3600:  # Within an hour
                        related_ids.append(mid)
        
        # Find memories with the same participants
        for participant in episode.participants:
            entity_id = participant.entity_id
            if entity_id in self.participant_index:
                for mid in self.participant_index[entity_id]:
                    if mid != memory_id and mid not in related_ids and mid in self.episodes:
                        related_ids.append(mid)
        
        # Find memories with similar emotional impact
        primary_emotion = episode.emotional_impact.primary_emotion
        if primary_emotion in self.emotion_index:
            for mid in self.emotion_index[primary_emotion]:
                if mid != memory_id and mid not in related_ids and mid in self.episodes:
                    related_ids.append(mid)
        
        # Score related memories for relevance
        scored_ids = []
        for mid in related_ids[:50]:  # Limit initial candidates
            if mid in self.episodes:
                other_episode = self.episodes[mid]
                score = 0.0
                
                # Score based on temporal proximity
                time_diff_days = abs((timestamp - other_episode.time_context.timestamp).total_seconds()) / 86400
                temporal_score = max(0.0, 1.0 - (time_diff_days / 7.0))  # Scale by week
                score += temporal_score * 0.3
                
                # Score based on emotional similarity
                if other_episode.emotional_impact.primary_emotion == primary_emotion:
                    emotional_score = 0.5 + (other_episode.emotional_impact.intensity * 0.5)
                    score += emotional_score * 0.3
                
                # Score based on participant overlap
                participant_overlap = 0.0
                other_participants = {p.entity_id for p in other_episode.participants}
                episode_participants = {p.entity_id for p in episode.participants}
                
                if episode_participants and other_participants:
                    overlap_count = len(episode_participants.intersection(other_participants))
                    total_count = len(episode_participants.union(other_participants))
                    if total_count > 0:
                        participant_overlap = overlap_count / total_count
                
                score += participant_overlap * 0.4
                
                scored_ids.append((mid, score))
        
        # Sort by relevance score
        scored_ids.sort(key=lambda x: x[1], reverse=True)
        
        return [mid for mid, _ in scored_ids[:limit]]
    
    def retrieve_spontaneously(self, current_cues: Dict[str, Any],
                             limit: int = 3) -> List[str]:
        """Spontaneously retrieve memories based on current cues
        
        This simulates the spontaneous recall that happens in human memory.
        
        Args:
            current_cues: Dictionary with cues like emotions, participants, etc.
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory IDs
        """
        candidates = set()
        scores = {}
        
        # Extract cues
        current_emotions = current_cues.get("emotions", {})
        current_participants = current_cues.get("participants", [])
        current_context = current_cues.get("context", "")
        current_location = current_cues.get("location", "")
        
        # Find memories matching emotional cues
        for emotion, intensity in current_emotions.items():
            if emotion in self.emotion_index:
                for mid in self.emotion_index[emotion]:
                    if mid in self.episodes:
                        candidates.add(mid)
        
        # Find memories matching participant cues
        for entity_id in current_participants:
            if entity_id in self.participant_index:
                for mid in self.participant_index[entity_id]:
                    if mid in self.episodes:
                        candidates.add(mid)
        
        # Find relevant recent memories
        recent_memories = []
        for time_bucket, memory_ids in sorted(self.temporal_index.items(), reverse=True)[:10]:
            recent_memories.extend(memory_ids)
        
        # Add some recent memories to candidates
        for mid in recent_memories[:20]:
            if mid in self.episodes:
                candidates.add(mid)
        
        # Score candidate memories for relevance to current cues
        for mid in candidates:
            episode = self.episodes[mid]
            score = 0.0
            
            # Base probability from calculated retrieval probability
            base_prob = episode.retrieval_probability
            score += base_prob * 0.3
            
            # Score based on emotional relevance
            emotional_relevance = episode.update_emotional_relevance(current_emotions)
            score += emotional_relevance * 0.3
            
            # Score based on participant overlap
            participant_score = 0.0
            episode_participants = {p.entity_id for p in episode.participants}
            if current_participants and episode_participants:
                overlap_count = len(set(current_participants).intersection(episode_participants))
                if overlap_count > 0:
                    participant_score = 0.5 + (overlap_count / len(current_participants) * 0.5)
            
            score += participant_score * 0.2
            
            # Score based on location match
            location_score = 0.0
            if current_location and episode.spatial_context and episode.spatial_context.location:
                if current_location == episode.spatial_context.location:
                    location_score = 0.8
            
            score += location_score * 0.1
            
            # Context keyword match
            context_score = 0.0
            if current_context and (episode.content.summary or episode.content.details):
                summary = episode.content.summary or ""
                details = episode.content.details or ""
                full_text = (summary + " " + details).lower()
                
                # Check for context keywords in memory text
                context_keywords = current_context.lower().split()
                for keyword in context_keywords:
                    if len(keyword) > 3 and keyword in full_text:
                        context_score += 0.2
                
                context_score = min(0.8, context_score)
            
            score += context_score * 0.1
            
            # Add random factor to simulate unpredictability of memory
            score += random.uniform(0.0, 0.1)
            
            scores[mid] = score
        
        # Sort by score and take top memories
        sorted_memories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top N memories that exceed threshold
        retrieval_threshold = 0.3
        result = []
        
        for mid, score in sorted_memories:
            if score >= retrieval_threshold and len(result) < limit:
                result.append(mid)
                # Rehearse this memory since it was spontaneously recalled
                self.episodes[mid].rehearse()
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory"""
        # Count memories by type
        type_counts = {episode_type.value: len(memory_ids) 
                     for episode_type, memory_ids in self.type_index.items()}
        
        # Count memories by emotional impact
        emotion_counts = {emotion: len(memory_ids) 
                        for emotion, memory_ids in self.emotion_index.items()}
        
        # Get top participants by memory count
        participant_counts = {entity_id: len(memory_ids) 
                            for entity_id, memory_ids in self.participant_index.items()}
        
        top_participants = sorted(participant_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate average rehearsal count
        avg_rehearsal = 0.0
        if self.episodes:
            avg_rehearsal = sum(episode.rehearsal_count for episode in self.episodes.values()) / len(self.episodes)
        
        # Count memories by importance
        importance_counts = {importance.name: 0 for importance in EpisodeImportance}
        for episode in self.episodes.values():
            importance_counts[episode.importance.name] += 1
        
        # Count memories by recency
        current_time = datetime.now()
        recency_counts = {
            "today": 0,
            "this_week": 0,
            "this_month": 0,
            "this_year": 0,
            "older": 0
        }
        
        for episode in self.episodes.values():
            days_ago = (current_time - episode.time_context.timestamp).days
            if days_ago < 1:
                recency_counts["today"] += 1
            elif days_ago < 7:
                recency_counts["this_week"] += 1
            elif days_ago < 30:
                recency_counts["this_month"] += 1
            elif days_ago < 365:
                recency_counts["this_year"] += 1
            else:
                recency_counts["older"] += 1
        
        return {
            "total_memories": len(self.episodes),
            "memories_by_type": type_counts,
            "memories_by_emotion": emotion_counts,
            "top_participants": dict(top_participants),
            "average_rehearsal_count": avg_rehearsal,
            "memories_by_importance": importance_counts,
            "memories_by_recency": recency_counts,
            "capacity": self.capacity,
            "capacity_used_percent": (len(self.episodes) / self.capacity) * 100
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the state of episodic memory to disk"""
        if filepath is None:
            filepath = self.data_dir / f"episodic_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        episodes_data = {}
        for memory_id, episode in self.episodes.items():
            # Convert Pydantic model to dict
            episodes_data[memory_id] = episode.model_dump()
            
            # Convert datetime objects to ISO format strings
            episodes_data[memory_id]["time_context"]["timestamp"] = episode.time_context.timestamp.isoformat()
            episodes_data[memory_id]["last_retrieved"] = episode.last_retrieved.isoformat()
        
        # Prepare indices for serialization
        temporal_index_data = {time_bucket: memory_ids 
                             for time_bucket, memory_ids in self.temporal_index.items()}
        
        type_index_data = {episode_type.value: memory_ids 
                         for episode_type, memory_ids in self.type_index.items()}
        
        participant_index_data = {entity_id: memory_ids 
                                for entity_id, memory_ids in self.participant_index.items()}
        
        emotion_index_data = {emotion: memory_ids 
                            for emotion, memory_ids in self.emotion_index.items()}
        
        state = {
            "episodes": episodes_data,
            "temporal_index": temporal_index_data,
            "type_index": type_index_data,
            "participant_index": participant_index_data,
            "emotion_index": emotion_index_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "episode_count": len(self.episodes),
                "capacity": self.capacity
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Episodic memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of episodic memory from disk"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.episodes.clear()
            self.temporal_index.clear()
            self.type_index = {episode_type: [] for episode_type in EpisodeType}
            self.participant_index.clear()
            self.emotion_index.clear()
            
            # Load episodes
            for memory_id, episode_data in state.get("episodes", {}).items():
                # Convert timestamps back to datetime
                if "time_context" in episode_data and "timestamp" in episode_data["time_context"]:
                    episode_data["time_context"]["timestamp"] = datetime.fromisoformat(
                        episode_data["time_context"]["timestamp"]
                    )
                
                if "last_retrieved" in episode_data:
                    episode_data["last_retrieved"] = datetime.fromisoformat(episode_data["last_retrieved"])
                
                # Convert episode_type and importance from strings to enums
                if "episode_type" in episode_data:
                    episode_data["episode_type"] = EpisodeType(episode_data["episode_type"])
                
                if "importance" in episode_data:
                    episode_data["importance"] = EpisodeImportance[episode_data["importance"]]
                
                # Create EpisodicMemoryItem
                self.episodes[memory_id] = EpisodicMemoryItem(**episode_data)
            
            # Load indices
            self.temporal_index = state.get("temporal_index", {})
            
            # Convert type_index keys from strings to enums
            for type_str, memory_ids in state.get("type_index", {}).items():
                self.type_index[EpisodeType(type_str)] = memory_ids
            
            self.participant_index = state.get("participant_index", {})
            self.emotion_index = state.get("emotion_index", {})
            
            # Load capacity if present
            if "metadata" in state and "capacity" in state["metadata"]:
                self.capacity = state["metadata"]["capacity"]
            
            logger.info(f"Loaded episodic memory state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading episodic memory state: {str(e)}")
            return False