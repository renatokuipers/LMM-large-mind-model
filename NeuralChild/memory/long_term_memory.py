# long_term_memory.py
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

# Import from memory manager
from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority

# Set up logging
logger = logging.getLogger("LongTermMemory")

class MemoryStrength(Enum):
    """Strength levels for long-term memories"""
    WEAK = auto()      # Barely remembered, could easily be forgotten
    MODERATE = auto()  # Somewhat stable, but might fade with time
    STRONG = auto()    # Well-remembered, resistant to forgetting
    PERMANENT = auto() # Foundational memories that will not be forgotten

class KnowledgeDomain(str, Enum):
    """Domains of knowledge in long-term memory"""
    PERSONAL = "personal"           # Personal experiences, preferences, identity
    SOCIAL = "social"               # Knowledge about people and relationships
    PROCEDURAL = "procedural"       # How to do things
    DECLARATIVE = "declarative"     # Facts, concepts, information
    EMOTIONAL = "emotional"         # Emotional patterns and responses
    LINGUISTIC = "linguistic"       # Language knowledge
    VALUES = "values"               # Moral and ethical principles
    CULTURAL = "cultural"           # Cultural norms and understanding

class LongTermMemoryItem(BaseModel):
    """A long-term memory item with additional metadata"""
    memory_id: str = Field(..., description="Reference to the item in memory manager")
    strength: MemoryStrength = Field(MemoryStrength.MODERATE, description="Memory strength")
    domain: Optional[KnowledgeDomain] = Field(None, description="Knowledge domain")
    certainty: float = Field(0.8, ge=0.0, le=1.0, description="How certain we are of this memory")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance to the self-concept")
    reinforcement_count: int = Field(0, ge=0, description="How many times this has been reinforced")
    last_reinforced: datetime = Field(default_factory=datetime.now)
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Resistance to modification")
    
    def reinforce(self) -> None:
        """Reinforce this memory, making it stronger"""
        self.reinforcement_count += 1
        self.last_reinforced = datetime.now()
        
        # Increase stability with each reinforcement, with diminishing returns
        stability_gain = 0.1 * (1.0 - self.stability)
        self.stability = min(1.0, self.stability + stability_gain)
        
        # Update strength based on reinforcement count
        if self.reinforcement_count >= 10 and self.stability > 0.8:
            self.strength = MemoryStrength.PERMANENT
        elif self.reinforcement_count >= 5 and self.stability > 0.6:
            self.strength = MemoryStrength.STRONG
        elif self.reinforcement_count >= 2:
            self.strength = MemoryStrength.MODERATE
    
    def update_certainty(self, new_evidence: float) -> None:
        """Update certainty based on new evidence or experience"""
        # New evidence is weighted by stability - stable memories resist change
        resistance_factor = self.stability
        
        # Calculate weighted average between current certainty and new evidence
        self.certainty = (self.certainty * resistance_factor) + (new_evidence * (1.0 - resistance_factor))
    
    def apply_decay(self, rate: float = 0.001) -> None:
        """Apply memory decay based on strength"""
        # Permanent memories don't decay
        if self.strength == MemoryStrength.PERMANENT:
            return
        
        # Calculate time-based decay
        days_since_reinforcement = (datetime.now() - self.last_reinforced).days
        
        # Different decay rates based on memory strength
        if self.strength == MemoryStrength.WEAK:
            decay_factor = rate * 3.0 * days_since_reinforcement
        elif self.strength == MemoryStrength.MODERATE:
            decay_factor = rate * 1.0 * days_since_reinforcement
        else:  # STRONG
            decay_factor = rate * 0.3 * days_since_reinforcement
        
        # Apply decay to certainty and importance
        decay_factor = min(0.5, decay_factor)  # Cap maximum decay
        self.certainty *= (1.0 - decay_factor)
        
        # Importance decays much more slowly
        self.importance *= (1.0 - (decay_factor * 0.1))
        
        # Stability decays very slowly
        self.stability *= (1.0 - (decay_factor * 0.05))
        
        # Downgrade strength if certainty falls too low
        if self.certainty < 0.3 and self.strength == MemoryStrength.STRONG:
            self.strength = MemoryStrength.MODERATE
        elif self.certainty < 0.2 and self.strength == MemoryStrength.MODERATE:
            self.strength = MemoryStrength.WEAK

class BeliefSystem(BaseModel):
    """A system of related beliefs and values"""
    name: str = Field(..., description="Name of this belief system")
    core_values: Dict[str, float] = Field(default_factory=dict, description="Core values with importance")
    beliefs: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Beliefs with certainty")
    consistency: float = Field(0.8, ge=0.0, le=1.0, description="Internal consistency of this belief system")
    formation_date: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_value(self, value: str, importance: float) -> None:
        """Add or update a core value"""
        self.core_values[value] = importance
        self.last_updated = datetime.now()
    
    def add_belief(self, category: str, belief: str, certainty: float) -> None:
        """Add or update a belief"""
        if category not in self.beliefs:
            self.beliefs[category] = {}
        
        self.beliefs[category][belief] = certainty
        self.last_updated = datetime.now()
    
    def evaluate_consistency(self) -> float:
        """Evaluate the internal consistency of this belief system"""
        # For a real system, this would use more sophisticated analysis
        # For now, we'll use a simplified approach
        
        # Count the number of beliefs
        belief_count = sum(len(beliefs) for beliefs in self.beliefs.values())
        
        if belief_count == 0:
            return 1.0  # No beliefs, so no inconsistency
        
        # For simplicity, assume beliefs are consistent unless marked otherwise
        # In a real system, this would check for logical contradictions
        self.consistency = 0.9  # Default high consistency
        return self.consistency

class PersonalIdentity(BaseModel):
    """Core identity and self-concept"""
    self_attributes: Dict[str, float] = Field(default_factory=dict, description="Attributes with certainty")
    core_memories: List[str] = Field(default_factory=list, description="Formative memory IDs")
    relationships: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Key relationships")
    preferences: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Preferences by category")
    belief_systems: Dict[str, BeliefSystem] = Field(default_factory=dict, description="Belief systems")
    
    def add_self_attribute(self, attribute: str, certainty: float) -> None:
        """Add or update a self-attribute"""
        self.self_attributes[attribute] = certainty
    
    def add_core_memory(self, memory_id: str) -> None:
        """Add a core memory to identity"""
        if memory_id not in self.core_memories:
            self.core_memories.append(memory_id)
    
    def remove_core_memory(self, memory_id: str) -> bool:
        """Remove a core memory from identity"""
        if memory_id in self.core_memories:
            self.core_memories.remove(memory_id)
            return True
        return False
    
    def add_relationship(self, person: str, data: Dict[str, Any]) -> None:
        """Add or update a relationship"""
        self.relationships[person] = data
    
    def add_preference(self, category: str, item: str, strength: float) -> None:
        """Add or update a preference"""
        if category not in self.preferences:
            self.preferences[category] = {}
        
        self.preferences[category][item] = strength
    
    def add_belief_system(self, belief_system: BeliefSystem) -> None:
        """Add a belief system"""
        self.belief_systems[belief_system.name] = belief_system
    
    def get_self_concept(self) -> Dict[str, Any]:
        """Get a consolidated view of self-concept"""
        # Extract the most important self-attributes
        important_attributes = {k: v for k, v in self.self_attributes.items() if v > 0.7}
        
        # Extract strongest preferences
        strong_preferences = {}
        for category, prefs in self.preferences.items():
            strong_preferences[category] = {k: v for k, v in prefs.items() if v > 0.7}
        
        # Extract core values from belief systems
        core_values = {}
        for belief_system in self.belief_systems.values():
            for value, importance in belief_system.core_values.items():
                if importance > 0.8:
                    core_values[value] = importance
        
        return {
            "attributes": important_attributes,
            "preferences": strong_preferences,
            "values": core_values,
            "core_memory_count": len(self.core_memories),
            "relationship_count": len(self.relationships)
        }

class LongTermMemory:
    """Persistent knowledge storage system"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize long-term memory
        
        Args:
            data_dir: Directory for persistent storage
        """
        self.data_dir = data_dir or Path("./data/memory/long_term")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.memories: Dict[str, LongTermMemoryItem] = {}
        self.identity = PersonalIdentity()
        self.domain_index: Dict[KnowledgeDomain, Set[str]] = {
            domain: set() for domain in KnowledgeDomain
        }
        self.memory_manager: Optional[MemoryManager] = None
        
        # Index by importance for quick access to crucial memories
        self.importance_threshold = 0.7  # Threshold for "important" memories
        self.important_memories: Set[str] = set()
        
        logger.info("Long-term memory initialized")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def store(self, memory_item: MemoryItem, domain: Optional[KnowledgeDomain] = None,
              importance: float = 0.5, is_core_memory: bool = False) -> str:
        """Store a memory in long-term memory"""
        memory_id = memory_item.id
        
        # Determine memory strength based on emotional attributes
        if memory_item.attributes.emotional_intensity > 0.8 or importance > 0.8:
            strength = MemoryStrength.STRONG
        elif memory_item.attributes.emotional_intensity > 0.4 or importance > 0.5:
            strength = MemoryStrength.MODERATE
        else:
            strength = MemoryStrength.WEAK
        
        # Create long-term memory item
        lt_memory = LongTermMemoryItem(
            memory_id=memory_id,
            strength=strength,
            domain=domain,
            certainty=memory_item.attributes.confidence,
            importance=importance,
            stability=min(0.8, memory_item.attributes.salience + 0.3)  # Initial stability
        )
        
        # Store the memory
        self.memories[memory_id] = lt_memory
        
        # Update domain index
        if domain:
            self.domain_index[domain].add(memory_id)
        
        # Add to important memories if applicable
        if importance >= self.importance_threshold:
            self.important_memories.add(memory_id)
        
        # Add to identity if it's a core memory
        if is_core_memory:
            self.identity.add_core_memory(memory_id)
        
        logger.info(f"Stored memory in long-term memory: {memory_id}")
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory from long-term memory"""
        if memory_id not in self.memories:
            return None
        
        # Reinforce the memory
        self.memories[memory_id].reinforce()
        
        # Get the actual memory from the memory manager
        if self.memory_manager:
            memory_item = self.memory_manager.retrieve(memory_id)
            if memory_item:
                logger.info(f"Retrieved memory from long-term memory: {memory_id}")
                return memory_item
        
        logger.warning(f"Could not retrieve memory {memory_id} from memory manager")
        return None
    
    def update(self, memory_id: str, memory_item: Optional[MemoryItem] = None,
              new_certainty: Optional[float] = None, new_importance: Optional[float] = None) -> bool:
        """Update a memory in long-term memory"""
        if memory_id not in self.memories:
            return False
        
        lt_memory = self.memories[memory_id]
        
        # Update metadata
        lt_memory.reinforce()
        
        # Update certainty if provided
        if new_certainty is not None:
            lt_memory.update_certainty(new_certainty)
        
        # Update importance if provided
        if new_importance is not None:
            lt_memory.importance = new_importance
            
            # Update important memories index
            if new_importance >= self.importance_threshold:
                self.important_memories.add(memory_id)
            elif memory_id in self.important_memories:
                self.important_memories.remove(memory_id)
        
        logger.info(f"Updated memory in long-term memory: {memory_id}")
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory from long-term memory"""
        if memory_id not in self.memories:
            return False
        
        lt_memory = self.memories[memory_id]
        
        # Remove from domain index
        if lt_memory.domain:
            self.domain_index[lt_memory.domain].discard(memory_id)
        
        # Remove from important memories
        self.important_memories.discard(memory_id)
        
        # Remove from identity if it's a core memory
        self.identity.remove_core_memory(memory_id)
        
        # Remove from memories
        del self.memories[memory_id]
        
        logger.info(f"Deleted memory from long-term memory: {memory_id}")
        return True
    
    def get_memories_by_domain(self, domain: KnowledgeDomain, min_certainty: float = 0.0) -> List[str]:
        """Get all memories in a specific domain"""
        memory_ids = self.domain_index.get(domain, set())
        
        # Filter by certainty if needed
        if min_certainty > 0.0:
            return [memory_id for memory_id in memory_ids 
                   if memory_id in self.memories and self.memories[memory_id].certainty >= min_certainty]
        
        return list(memory_ids)
    
    def get_important_memories(self, top_n: Optional[int] = None) -> List[str]:
        """Get important memories"""
        # Rank by importance
        ranked = [(memory_id, self.memories[memory_id].importance)
                 for memory_id in self.important_memories
                 if memory_id in self.memories]
        
        # Sort by importance, descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return all or top N
        if top_n:
            return [memory_id for memory_id, _ in ranked[:top_n]]
        return [memory_id for memory_id, _ in ranked]
    
    def apply_memory_decay(self, rate: float = 0.001) -> List[str]:
        """Apply decay to all memories and return IDs of memories that decayed below threshold"""
        decayed_below_threshold = []
        
        for memory_id, lt_memory in list(self.memories.items()):
            # Apply decay
            lt_memory.apply_decay(rate)
            
            # Check if memory has decayed below threshold
            if lt_memory.certainty < 0.1 and lt_memory.strength == MemoryStrength.WEAK:
                # This memory has faded too much
                decayed_below_threshold.append(memory_id)
        
        return decayed_below_threshold
    
    def register_belief(self, category: str, belief: str, certainty: float,
                       belief_system_name: str = "default") -> None:
        """Register a belief in a belief system"""
        # Get or create belief system
        if belief_system_name not in self.identity.belief_systems:
            self.identity.belief_systems[belief_system_name] = BeliefSystem(name=belief_system_name)
        
        belief_system = self.identity.belief_systems[belief_system_name]
        
        # Add belief
        belief_system.add_belief(category, belief, certainty)
        belief_system.evaluate_consistency()
        
        logger.info(f"Registered belief '{belief}' in system '{belief_system_name}'")
    
    def register_value(self, value: str, importance: float,
                      belief_system_name: str = "default") -> None:
        """Register a value in a belief system"""
        # Get or create belief system
        if belief_system_name not in self.identity.belief_systems:
            self.identity.belief_systems[belief_system_name] = BeliefSystem(name=belief_system_name)
        
        belief_system = self.identity.belief_systems[belief_system_name]
        
        # Add value
        belief_system.add_value(value, importance)
        
        logger.info(f"Registered value '{value}' in system '{belief_system_name}'")
    
    def register_preference(self, category: str, item: str, strength: float) -> None:
        """Register a preference"""
        self.identity.add_preference(category, item, strength)
        logger.info(f"Registered preference for '{item}' in category '{category}'")
    
    def register_self_attribute(self, attribute: str, certainty: float) -> None:
        """Register a self-attribute"""
        self.identity.add_self_attribute(attribute, certainty)
        logger.info(f"Registered self-attribute '{attribute}'")
    
    def register_relationship(self, person: str, data: Dict[str, Any]) -> None:
        """Register a relationship"""
        self.identity.add_relationship(person, data)
        logger.info(f"Registered relationship with '{person}'")
    
    def get_belief_certainty(self, belief: str, category: Optional[str] = None,
                           belief_system_name: Optional[str] = None) -> float:
        """Get certainty for a specific belief"""
        # Search in specific belief system if provided
        if belief_system_name:
            if belief_system_name not in self.identity.belief_systems:
                return 0.0
            
            belief_system = self.identity.belief_systems[belief_system_name]
            
            # Search in specific category if provided
            if category:
                if category not in belief_system.beliefs:
                    return 0.0
                return belief_system.beliefs[category].get(belief, 0.0)
            
            # Search across all categories
            for cat_beliefs in belief_system.beliefs.values():
                if belief in cat_beliefs:
                    return cat_beliefs[belief]
            
            return 0.0
        
        # Search across all belief systems
        max_certainty = 0.0
        
        for belief_system in self.identity.belief_systems.values():
            for cat, beliefs in belief_system.beliefs.items():
                if category and cat != category:
                    continue
                
                if belief in beliefs:
                    max_certainty = max(max_certainty, beliefs[belief])
        
        return max_certainty
    
    def get_value_importance(self, value: str, belief_system_name: Optional[str] = None) -> float:
        """Get importance for a specific value"""
        # Search in specific belief system if provided
        if belief_system_name:
            if belief_system_name not in self.identity.belief_systems:
                return 0.0
            
            belief_system = self.identity.belief_systems[belief_system_name]
            return belief_system.core_values.get(value, 0.0)
        
        # Search across all belief systems
        max_importance = 0.0
        
        for belief_system in self.identity.belief_systems.values():
            if value in belief_system.core_values:
                max_importance = max(max_importance, belief_system.core_values[value])
        
        return max_importance
    
    def get_preference_strength(self, category: str, item: str) -> float:
        """Get strength of a specific preference"""
        if category not in self.identity.preferences:
            return 0.0
        
        return self.identity.preferences[category].get(item, 0.0)
    
    def get_identity_state(self) -> Dict[str, Any]:
        """Get the current state of identity"""
        # Get self-concept summary
        self_concept = self.identity.get_self_concept()
        
        # Get belief systems summary
        belief_systems_summary = {}
        for name, system in self.identity.belief_systems.items():
            belief_systems_summary[name] = {
                "value_count": len(system.core_values),
                "belief_count": sum(len(beliefs) for beliefs in system.beliefs.values()),
                "consistency": system.consistency,
                "age_days": (datetime.now() - system.formation_date).days
            }
        
        # Count memories by domain
        domain_counts = {domain.value: len(memory_ids) for domain, memory_ids in self.domain_index.items()}
        
        return {
            "self_concept": self_concept,
            "belief_systems": belief_systems_summary,
            "domain_counts": domain_counts,
            "important_memory_count": len(self.important_memories),
            "total_memory_count": len(self.memories)
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the state of long-term memory to disk"""
        if filepath is None:
            filepath = self.data_dir / f"long_term_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        memories_data = {}
        for memory_id, lt_memory in self.memories.items():
            memories_data[memory_id] = {
                "memory_id": memory_id,
                "strength": lt_memory.strength.name,
                "domain": lt_memory.domain.value if lt_memory.domain else None,
                "certainty": lt_memory.certainty,
                "importance": lt_memory.importance,
                "reinforcement_count": lt_memory.reinforcement_count,
                "last_reinforced": lt_memory.last_reinforced.isoformat(),
                "stability": lt_memory.stability
            }
        
        # Convert identity to serializable format
        identity_data = self.identity.model_dump()
        
        # Convert datetime objects in belief systems
        for belief_system in identity_data["belief_systems"].values():
            belief_system["formation_date"] = datetime.fromisoformat(belief_system["formation_date"]).isoformat()
            belief_system["last_updated"] = datetime.fromisoformat(belief_system["last_updated"]).isoformat()
        
        # Prepare domain index
        domain_index_data = {domain.value: list(memory_ids) for domain, memory_ids in self.domain_index.items()}
        
        state = {
            "memories": memories_data,
            "identity": identity_data,
            "domain_index": domain_index_data,
            "important_memories": list(self.important_memories),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "memory_count": len(self.memories)
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Long-term memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of long-term memory from disk"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.memories.clear()
            self.identity = PersonalIdentity()
            self.domain_index = {domain: set() for domain in KnowledgeDomain}
            self.important_memories.clear()
            
            # Load memories
            for memory_id, memory_data in state.get("memories", {}).items():
                lt_memory = LongTermMemoryItem(
                    memory_id=memory_data["memory_id"],
                    strength=MemoryStrength[memory_data["strength"]],
                    domain=KnowledgeDomain(memory_data["domain"]) if memory_data["domain"] else None,
                    certainty=memory_data["certainty"],
                    importance=memory_data["importance"],
                    reinforcement_count=memory_data["reinforcement_count"],
                    last_reinforced=datetime.fromisoformat(memory_data["last_reinforced"]),
                    stability=memory_data["stability"]
                )
                
                self.memories[memory_id] = lt_memory
            
            # Load identity
            identity_data = state.get("identity", {})
            self.identity = PersonalIdentity.model_validate(identity_data)
            
            # Load domain index
            domain_index_data = state.get("domain_index", {})
            for domain_str, memory_ids in domain_index_data.items():
                self.domain_index[KnowledgeDomain(domain_str)] = set(memory_ids)
            
            # Load important memories
            self.important_memories = set(state.get("important_memories", []))
            
            logger.info(f"Loaded long-term memory state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading long-term memory state: {str(e)}")
            return False