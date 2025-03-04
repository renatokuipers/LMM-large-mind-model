# associative_memory.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import numpy as np
from pathlib import Path
import json
import os
import random

from memory.memory_manager import MemoryItem, MemoryType, MemoryAttributes, MemoryManager, MemoryPriority

logger = logging.getLogger("AssociativeMemory")

class AssociativeLink:
    """Link between concepts in associative memory"""
    def __init__(
        self,
        source: str,
        target: str,
        strength: float = 0.5,
        link_type: str = "semantic",
        created_at: Optional[datetime] = None
    ):
        self.source = source
        self.target = target
        self.strength = strength
        self.link_type = link_type
        self.created_at = created_at or datetime.now()
        self.last_activated = self.created_at
        self.activation_count = 0
    
    def activate(self, boost: float = 0.1) -> None:
        """Activate this link, increasing its strength"""
        self.strength = min(1.0, self.strength + boost)
        self.last_activated = datetime.now()
        self.activation_count += 1
    
    def decay(self, rate: float = 0.01) -> None:
        """Apply decay to this link's strength"""
        self.strength *= (1.0 - rate)

class AssociativeMemory:
    """Manages associations between concepts and memories"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize associative memory"""
        self.data_dir = data_dir or Path("./data/memory/associative")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.links: Dict[str, AssociativeLink] = {}  # link_id -> AssociativeLink
        self.source_index: Dict[str, Set[str]] = {}  # source_id -> set of link_ids
        self.target_index: Dict[str, Set[str]] = {}  # target_id -> set of link_ids
        self.memory_manager: Optional[MemoryManager] = None
        
        # Metrics
        self.total_activations = 0
        
        logger.info("Associative memory initialized")
    
    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set the memory manager reference"""
        self.memory_manager = memory_manager
    
    def add_association(self, memory_item: MemoryItem) -> str:
        """Add an association from a memory item"""
        link_id = memory_item.id
        
        # Extract source and target from content
        content = memory_item.content
        source = content.get("source", "unknown")
        target = content.get("target", "unknown")
        strength = content.get("strength", 0.5)
        link_type = content.get("type", "semantic")
        
        # Create the link
        link = AssociativeLink(
            source=source,
            target=target,
            strength=strength,
            link_type=link_type
        )
        
        # Store the link
        self.links[link_id] = link
        
        # Update indices
        if source not in self.source_index:
            self.source_index[source] = set()
        self.source_index[source].add(link_id)
        
        if target not in self.target_index:
            self.target_index[target] = set()
        self.target_index[target].add(link_id)
        
        logger.info(f"Added association {link_id}: {source} -> {target}")
        return link_id
    
    def create_association(self, source: str, target: str, 
                         strength: float = 0.5, link_type: str = "semantic") -> str:
        """Create a new association directly"""
        # Generate a unique ID
        link_id = f"assoc_{len(self.links)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the link
        link = AssociativeLink(
            source=source,
            target=target,
            strength=strength,
            link_type=link_type
        )
        
        # Store the link
        self.links[link_id] = link
        
        # Update indices
        if source not in self.source_index:
            self.source_index[source] = set()
        self.source_index[source].add(link_id)
        
        if target not in self.target_index:
            self.target_index[target] = set()
        self.target_index[target].add(link_id)
        
        logger.info(f"Created association {link_id}: {source} -> {target}")
        return link_id
    
    def get_associations_from(self, source: str, min_strength: float = 0.0) -> List[Tuple[str, float, str]]:
        """Get all associations from a source"""
        if source not in self.source_index:
            return []
        
        result = []
        for link_id in self.source_index[source]:
            link = self.links.get(link_id)
            if link and link.strength >= min_strength:
                result.append((link.target, link.strength, link.link_type))
        
        # Sort by strength, descending
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def get_associations_to(self, target: str, min_strength: float = 0.0) -> List[Tuple[str, float, str]]:
        """Get all associations to a target"""
        if target not in self.target_index:
            return []
        
        result = []
        for link_id in self.target_index[target]:
            link = self.links.get(link_id)
            if link and link.strength >= min_strength:
                result.append((link.source, link.strength, link.link_type))
        
        # Sort by strength, descending
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def activate_association(self, link_id: str) -> bool:
        """Activate an association, increasing its strength"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        link.activate()
        self.total_activations += 1
        
        logger.debug(f"Activated association {link_id}: {link.source} -> {link.target}")
        return True
    
    def activate_concept(self, concept: str, activation_strength: float = 0.5) -> List[str]:
        """Activate a concept and spread activation to connected concepts"""
        activated_concepts = [concept]
        
        # Get outgoing links
        if concept in self.source_index:
            for link_id in self.source_index[concept]:
                link = self.links.get(link_id)
                if link:
                    # Activate the link
                    original_strength = link.strength
                    link.activate(boost=activation_strength * 0.1)
                    
                    # If the link is strong enough, activate the target
                    if link.strength > 0.3:
                        # Recursive activation with diminished strength
                        target_activation = activation_strength * link.strength * 0.7
                        if target_activation > 0.1 and link.target not in activated_concepts:
                            activated_concepts.append(link.target)
                            target_activated = self.activate_concept(link.target, target_activation)
                            activated_concepts.extend(target_activated)
        
        return list(set(activated_concepts))  # Remove duplicates
    
    def update_association(self, link_id: str, memory_item: Optional[MemoryItem] = None) -> bool:
        """Update an association"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        
        # If memory_item is provided, update from its content
        if memory_item and hasattr(memory_item, "content"):
            content = memory_item.content
            if "strength" in content:
                link.strength = content["strength"]
            if "type" in content:
                link.link_type = content["type"]
        
        logger.debug(f"Updated association {link_id}")
        return True
    
    def remove_association(self, link_id: str) -> bool:
        """Remove an association"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        
        # Remove from indices
        if link.source in self.source_index:
            self.source_index[link.source].discard(link_id)
        if link.target in self.target_index:
            self.target_index[link.target].discard(link_id)
        
        # Remove the link
        del self.links[link_id]
        
        logger.info(f"Removed association {link_id}")
        return True
    
    def apply_decay(self, rate: float = 0.01) -> None:
        """Apply decay to all associations"""
        for link in self.links.values():
            link.decay(rate)
        
        logger.debug(f"Applied decay with rate {rate} to all associations")
    
    def get_strongest_associations(self, limit: int = 10) -> List[Tuple[str, str, float, str]]:
        """Get the strongest associations"""
        # Sort links by strength
        sorted_links = sorted(self.links.values(), key=lambda x: x.strength, reverse=True)
        
        # Return top N
        return [(link.source, link.target, link.strength, link.link_type) 
                for link in sorted_links[:limit]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the associative memory"""
        link_count = len(self.links)
        concept_count = len(set(self.source_index.keys()) | set(self.target_index.keys()))
        
        # Count links by type
        link_types = {}
        for link in self.links.values():
            if link.link_type not in link_types:
                link_types[link.link_type] = 0
            link_types[link.link_type] += 1
        
        # Get average strength
        avg_strength = sum(link.strength for link in self.links.values()) / max(1, link_count)
        
        return {
            "link_count": link_count,
            "concept_count": concept_count,
            "link_types": link_types,
            "avg_strength": avg_strength,
            "total_activations": self.total_activations
        }
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save the state of associative memory"""
        if filepath is None:
            filepath = self.data_dir / f"associative_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare state for serialization
        links_data = {}
        for link_id, link in self.links.items():
            links_data[link_id] = {
                "source": link.source,
                "target": link.target,
                "strength": link.strength,
                "link_type": link.link_type,
                "created_at": link.created_at.isoformat(),
                "last_activated": link.last_activated.isoformat(),
                "activation_count": link.activation_count
            }
        
        state = {
            "links": links_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "link_count": len(self.links),
                "total_activations": self.total_activations
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Associative memory state saved to {filepath}")
    
    def load_state(self, filepath: Path) -> bool:
        """Load the state of associative memory"""
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.links.clear()
            self.source_index.clear()
            self.target_index.clear()
            
            # Load links
            for link_id, link_data in state.get("links", {}).items():
                link = AssociativeLink(
                    source=link_data["source"],
                    target=link_data["target"],
                    strength=link_data["strength"],
                    link_type=link_data["link_type"],
                    created_at=datetime.fromisoformat(link_data["created_at"])
                )
                link.last_activated = datetime.fromisoformat(link_data["last_activated"])
                link.activation_count = link_data["activation_count"]
                
                # Store the link
                self.links[link_id] = link
                
                # Update indices
                if link.source not in self.source_index:
                    self.source_index[link.source] = set()
                self.source_index[link.source].add(link_id)
                
                if link.target not in self.target_index:
                    self.target_index[link.target] = set()
                self.target_index[link.target].add(link_id)
            
            # Load metadata
            if "metadata" in state:
                self.total_activations = state["metadata"].get("total_activations", 0)
            
            logger.info(f"Loaded associative memory state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading associative memory state: {str(e)}")
            return False