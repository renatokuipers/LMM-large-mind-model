from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import uuid
import os
import json
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.memory.models import Memory, AssociativeLink
from lmm_project.utils.vector_store import VectorStore

class AssociativeMemoryModule(BaseModule):
    """
    Associative memory system for linking memories
    
    Associative memory creates and manages connections between different 
    memories, allowing for the spread of activation and emergent 
    associations between concepts, episodes, and other mental content.
    """
    # Association storage
    associations: Dict[str, AssociativeLink] = Field(default_factory=dict)
    # Memory source index (memory_id -> list of association_ids where memory is source)
    source_index: Dict[str, Set[str]] = Field(default_factory=dict)
    # Memory target index (memory_id -> list of association_ids where memory is target)
    target_index: Dict[str, Set[str]] = Field(default_factory=dict)
    # Association types index (type -> list of association_ids)
    type_index: Dict[str, Set[str]] = Field(default_factory=dict)
    # Hebbian learning rate (how quickly associations strengthen)
    hebbian_rate: float = Field(default=0.01)
    # Association decay rate (how quickly associations weaken when unused)
    decay_rate: float = Field(default=0.001)
    # Storage directory
    storage_dir: str = Field(default="storage/memories/associations")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize associative memory module"""
        super().__init__(
            module_id=module_id,
            module_type="associative_memory",
            event_bus=event_bus,
            **data
        )
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Try to load previous associations
        self._load_associations()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("memory_stored", self._handle_memory_stored)
            self.subscribe_to_message("memory_retrieved", self._handle_memory_retrieved)
            self.subscribe_to_message("concept_added", self._handle_concept_added)
            self.subscribe_to_message("episode_added", self._handle_episode_added)
            self.subscribe_to_message("association_query", self._handle_association_query)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process associative memory operations
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform (associate, get_associations,
                         spread_activation, find_path, etc.)
            - Additional parameters depend on the operation
            
        Returns:
        Dictionary containing operation results
        """
        operation = input_data.get("operation", "")
        
        if operation == "associate":
            source_id = input_data.get("source_id", "")
            target_id = input_data.get("target_id", "")
            link_type = input_data.get("link_type", "general")
            strength = input_data.get("strength", 0.5)
            return self.associate(source_id, target_id, link_type, strength)
        
        elif operation == "get_associations":
            memory_id = input_data.get("memory_id", "")
            return self.get_associations(memory_id)
        
        elif operation == "spread_activation":
            source_id = input_data.get("source_id", "")
            activation = input_data.get("activation", 0.5)
            depth = input_data.get("depth", 2)
            return self.spread_activation(source_id, activation, depth)
        
        elif operation == "find_path":
            source_id = input_data.get("source_id", "")
            target_id = input_data.get("target_id", "")
            return self.find_path(source_id, target_id)
        
        elif operation == "get_association_by_type":
            link_type = input_data.get("link_type", "")
            limit = input_data.get("limit", 10)
            return self.get_associations_by_type(link_type, limit)
            
        return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update associative memory's developmental level
        
        As associative memory develops:
        - Association formation becomes more sophisticated
        - Connections become more stable
        - Pattern recognition improves
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development
        delta = self.development_level - prev_level
        
        # Improve hebbian learning rate
        hebbian_increase = delta * 0.01
        self.hebbian_rate = min(0.1, self.hebbian_rate + hebbian_increase)
        
        # Decrease decay rate (associations become more stable)
        decay_decrease = delta * 0.0005
        self.decay_rate = max(0.0001, self.decay_rate - decay_decrease)
        
        return self.development_level
    
    def associate(
        self, 
        source_id: str, 
        target_id: str, 
        link_type: str = "general", 
        strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create or strengthen an association between two memories
        
        Parameters:
        source_id: Source memory ID
        target_id: Target memory ID
        link_type: Type of association
        strength: Initial association strength (0.0-1.0)
        
        Returns:
        Operation result
        """
        # Check if source and target are different
        if source_id == target_id:
            return {"status": "error", "message": "Cannot associate a memory with itself"}
        
        # Check if association already exists
        existing_association = self._find_association(source_id, target_id)
        
        if existing_association:
            # Strengthen existing association
            association = self.associations[existing_association]
            old_strength = association.strength
            association.update_strength(self.hebbian_rate)
            association.activation_count += 1
            
            # Save association
            self._save_association(association)
            
            # Publish event
            self.publish_message("association_strengthened", {
                "association_id": existing_association,
                "source_id": source_id,
                "target_id": target_id,
                "old_strength": old_strength,
                "new_strength": association.strength
            })
            
            return {
                "status": "success",
                "association_id": existing_association,
                "operation": "strengthened",
                "old_strength": old_strength,
                "new_strength": association.strength
            }
        else:
            # Create new association
            association_id = f"assoc_{uuid.uuid4().hex[:8]}"
            
            association = AssociativeLink(
                source_id=source_id,
                target_id=target_id,
                strength=strength,
                link_type=link_type,
                formed_at=datetime.now(),
                activation_count=1
            )
            
            # Store association
            self.associations[association_id] = association
            
            # Update indices
            if source_id not in self.source_index:
                self.source_index[source_id] = set()
            self.source_index[source_id].add(association_id)
            
            if target_id not in self.target_index:
                self.target_index[target_id] = set()
            self.target_index[target_id].add(association_id)
            
            if link_type not in self.type_index:
                self.type_index[link_type] = set()
            self.type_index[link_type].add(association_id)
            
            # Save association
            self._save_association(association, association_id)
            
            # Publish event
            self.publish_message("association_created", {
                "association_id": association_id,
                "source_id": source_id,
                "target_id": target_id,
                "link_type": link_type,
                "strength": strength
            })
            
            return {
                "status": "success",
                "association_id": association_id,
                "operation": "created",
                "strength": strength
            }
    
    def get_associations(self, memory_id: str) -> Dict[str, Any]:
        """
        Get all associations for a memory
        
        Parameters:
        memory_id: Memory ID to get associations for
        
        Returns:
        Operation result containing associated memories
        """
        # Check if memory exists in indices
        if memory_id not in self.source_index and memory_id not in self.target_index:
            return {
                "status": "error", 
                "message": f"No associations found for memory: {memory_id}",
                "outgoing": [],
                "incoming": []
            }
        
        # Get outgoing associations (memory is source)
        outgoing_assocs = self.source_index.get(memory_id, set())
        outgoing = []
        
        for assoc_id in outgoing_assocs:
            if assoc_id in self.associations:
                assoc = self.associations[assoc_id]
                outgoing.append({
                    "association_id": assoc_id,
                    "target_id": assoc.target_id,
                    "link_type": assoc.link_type,
                    "strength": assoc.strength
                })
        
        # Get incoming associations (memory is target)
        incoming_assocs = self.target_index.get(memory_id, set())
        incoming = []
        
        for assoc_id in incoming_assocs:
            if assoc_id in self.associations:
                assoc = self.associations[assoc_id]
                incoming.append({
                    "association_id": assoc_id,
                    "source_id": assoc.source_id,
                    "link_type": assoc.link_type,
                    "strength": assoc.strength
                })
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "outgoing": sorted(outgoing, key=lambda x: x["strength"], reverse=True),
            "incoming": sorted(incoming, key=lambda x: x["strength"], reverse=True),
            "total_associations": len(outgoing) + len(incoming)
        }
    
    def spread_activation(
        self, 
        source_id: str, 
        activation: float = 0.5, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Spread activation from a source memory
        
        Parameters:
        source_id: Starting memory ID
        activation: Initial activation level
        depth: How many steps to spread activation
        
        Returns:
        Operation result containing activated memories
        """
        if activation <= 0.0 or depth <= 0:
            return {"status": "error", "message": "Invalid activation or depth"}
        
        # Track visited nodes and their activation
        activations = {source_id: activation}
        visited = set()
        
        # Queue of nodes to process (memory_id, current_depth, current_activation)
        queue = [(source_id, 0, activation)]
        
        while queue:
            current_id, current_depth, current_activation = queue.pop(0)
            
            # Skip if already visited or max depth reached
            if current_id in visited or current_depth >= depth:
                continue
                
            visited.add(current_id)
            
            # Get outgoing associations
            outgoing_assocs = self.source_index.get(current_id, set())
            
            for assoc_id in outgoing_assocs:
                if assoc_id in self.associations:
                    assoc = self.associations[assoc_id]
                    
                    # Calculate propagated activation
                    propagated = current_activation * assoc.strength
                    
                    # Skip weak activations
                    if propagated < 0.1:
                        continue
                        
                    target_id = assoc.target_id
                    
                    # Update target activation (take maximum if already activated)
                    if target_id in activations:
                        activations[target_id] = max(activations[target_id], propagated)
                    else:
                        activations[target_id] = propagated
                    
                    # Add to queue for further propagation
                    if current_depth + 1 < depth:
                        queue.append((target_id, current_depth + 1, propagated))
            
            # Get incoming associations
            incoming_assocs = self.target_index.get(current_id, set())
            
            for assoc_id in incoming_assocs:
                if assoc_id in self.associations:
                    assoc = self.associations[assoc_id]
                    
                    # Calculate propagated activation (slightly weaker for incoming)
                    propagated = current_activation * assoc.strength * 0.8
                    
                    # Skip weak activations
                    if propagated < 0.1:
                        continue
                        
                    source_id = assoc.source_id
                    
                    # Update source activation (take maximum if already activated)
                    if source_id in activations:
                        activations[source_id] = max(activations[source_id], propagated)
                    else:
                        activations[source_id] = propagated
                    
                    # Add to queue for further propagation
                    if current_depth + 1 < depth:
                        queue.append((source_id, current_depth + 1, propagated))
        
        # Remove the original source memory
        del activations[source_id]
        
        # Sort by activation level
        sorted_activations = [
            {"memory_id": mid, "activation": act}
            for mid, act in activations.items()
        ]
        sorted_activations.sort(key=lambda x: x["activation"], reverse=True)
        
        return {
            "status": "success",
            "source_id": source_id,
            "activated_memories": sorted_activations,
            "activation_count": len(sorted_activations)
        }
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> Dict[str, Any]:
        """
        Find a path between two memories
        
        Parameters:
        source_id: Starting memory ID
        target_id: Target memory ID
        max_depth: Maximum path length to consider
        
        Returns:
        Operation result containing the path if found
        """
        if source_id == target_id:
            return {
                "status": "success",
                "path_found": True,
                "path": [{"memory_id": source_id}],
                "path_length": 0,
                "path_strength": 1.0
            }
        
        # Use BFS to find shortest path
        visited = set()
        queue = [
            (source_id, [], 1.0)  # (current_id, path, path_strength)
        ]
        
        while queue:
            current_id, path, path_strength = queue.pop(0)
            
            # Skip if already visited or path too long
            if current_id in visited or len(path) >= max_depth:
                continue
                
            visited.add(current_id)
            
            # Get outgoing associations
            outgoing_assocs = self.source_index.get(current_id, set())
            
            for assoc_id in outgoing_assocs:
                if assoc_id in self.associations:
                    assoc = self.associations[assoc_id]
                    next_id = assoc.target_id
                    
                    # Calculate new path strength
                    new_strength = path_strength * assoc.strength
                    
                    # Create new path
                    new_path = path + [{
                        "memory_id": current_id, 
                        "association": {
                            "id": assoc_id,
                            "type": assoc.link_type,
                            "strength": assoc.strength
                        }
                    }]
                    
                    # Check if target found
                    if next_id == target_id:
                        # Complete path
                        final_path = new_path + [{"memory_id": target_id}]
                        
                        return {
                            "status": "success",
                            "path_found": True,
                            "path": final_path,
                            "path_length": len(final_path) - 1,
                            "path_strength": new_strength
                        }
                    
                    # Add to queue
                    queue.append((next_id, new_path, new_strength))
        
        # Check incoming direction (reverse path finding) if no path found
        visited = set()
        queue = [
            (target_id, [], 1.0)  # (current_id, path, path_strength)
        ]
        
        while queue:
            current_id, path, path_strength = queue.pop(0)
            
            # Skip if already visited or path too long
            if current_id in visited or len(path) >= max_depth:
                continue
                
            visited.add(current_id)
            
            # Get incoming associations
            incoming_assocs = self.target_index.get(current_id, set())
            
            for assoc_id in incoming_assocs:
                if assoc_id in self.associations:
                    assoc = self.associations[assoc_id]
                    next_id = assoc.source_id
                    
                    # Calculate new path strength
                    new_strength = path_strength * assoc.strength
                    
                    # Create new path (reverse order)
                    new_path = [{
                        "memory_id": current_id, 
                        "association": {
                            "id": assoc_id,
                            "type": assoc.link_type,
                            "strength": assoc.strength
                        }
                    }] + path
                    
                    # Check if source found
                    if next_id == source_id:
                        # Complete path (reverse order)
                        final_path = [{"memory_id": source_id}] + new_path
                        
                        return {
                            "status": "success",
                            "path_found": True,
                            "path": final_path,
                            "path_length": len(final_path) - 1,
                            "path_strength": new_strength
                        }
                    
                    # Add to queue
                    queue.append((next_id, new_path, new_strength))
        
        return {
            "status": "success",
            "path_found": False,
            "max_depth_searched": max_depth,
            "message": f"No path found between {source_id} and {target_id} within depth {max_depth}"
        }
    
    def get_associations_by_type(self, link_type: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get associations by type
        
        Parameters:
        link_type: Type of association to find
        limit: Maximum number of associations to return
        
        Returns:
        Operation result containing associations of the specified type
        """
        if link_type not in self.type_index:
            return {
                "status": "error",
                "message": f"No associations found of type: {link_type}",
                "associations": []
            }
        
        assoc_ids = self.type_index.get(link_type, set())
        associations = []
        
        for assoc_id in assoc_ids:
            if assoc_id in self.associations:
                assoc = self.associations[assoc_id]
                associations.append({
                    "association_id": assoc_id,
                    "source_id": assoc.source_id,
                    "target_id": assoc.target_id,
                    "strength": assoc.strength,
                    "activation_count": assoc.activation_count
                })
        
        # Sort by strength and limit results
        associations.sort(key=lambda x: x["strength"], reverse=True)
        associations = associations[:limit]
        
        return {
            "status": "success",
            "link_type": link_type,
            "associations": associations,
            "total_found": len(assoc_ids),
            "returned": len(associations)
        }
    
    def decay_associations(self) -> Dict[str, Any]:
        """
        Apply decay to associations over time
        
        Less frequently used associations weaken over time.
        
        Returns:
        Operation result
        """
        before_count = len(self.associations)
        decayed_count = 0
        removed_count = 0
        
        to_remove = []
        
        for assoc_id, assoc in self.associations.items():
            # Calculate time-based decay
            days_since_formed = (datetime.now() - assoc.formed_at).days
            
            # Base decay amount
            decay_amount = self.decay_rate
            
            # Adjust decay based on activation count (frequently used associations decay slower)
            activation_factor = 1.0 / (1.0 + 0.1 * assoc.activation_count)
            decay_amount *= activation_factor
            
            # Apply decay
            if decay_amount > 0:
                old_strength = assoc.strength
                assoc.strength = max(0.0, assoc.strength - decay_amount)
                
                # If decayed significantly, count it
                if old_strength - assoc.strength > 0.01:
                    decayed_count += 1
                
                # If strength falls below threshold, mark for removal
                if assoc.strength < 0.05:
                    to_remove.append(assoc_id)
                else:
                    # Save updated association
                    self._save_association(assoc, assoc_id)
        
        # Remove weak associations
        for assoc_id in to_remove:
            self._remove_association(assoc_id)
            removed_count += 1
        
        return {
            "status": "success",
            "before_count": before_count,
            "decayed_count": decayed_count,
            "removed_count": removed_count,
            "after_count": len(self.associations)
        }
    
    def count_associations(self) -> int:
        """Count the number of stored associations"""
        return len(self.associations)
    
    def save_state(self) -> str:
        """
        Save the current state of associative memory
        
        Returns:
        Path to saved state directory
        """
        # Save associations
        for assoc_id, assoc in self.associations.items():
            self._save_association(assoc, assoc_id)
        
        # Save indices
        self._save_indices()
        
        return self.storage_dir
    
    def _find_association(self, source_id: str, target_id: str) -> Optional[str]:
        """Find an existing association between source and target"""
        # Check source index
        source_assocs = self.source_index.get(source_id, set())
        
        for assoc_id in source_assocs:
            if assoc_id in self.associations:
                assoc = self.associations[assoc_id]
                if assoc.target_id == target_id:
                    return assoc_id
        
        return None
    
    def _save_association(self, association: AssociativeLink, association_id: Optional[str] = None) -> None:
        """Save a single association to disk"""
        if association_id is None:
            # Try to find the association ID
            for aid, assoc in self.associations.items():
                if assoc == association:
                    association_id = aid
                    break
            
            if association_id is None:
                # If still not found, can't save
                return
        
        try:
            assocs_dir = Path(self.storage_dir) / "associations"
            assocs_dir.mkdir(parents=True, exist_ok=True)
            
            assoc_path = assocs_dir / f"{association_id}.json"
            with open(assoc_path, "w") as f:
                # Convert to dict and handle datetime
                assoc_dict = association.model_dump()
                # Convert datetime to string
                for key, value in assoc_dict.items():
                    if isinstance(value, datetime):
                        assoc_dict[key] = value.isoformat()
                json.dump(assoc_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving association {association_id}: {e}")
    
    def _save_indices(self) -> None:
        """Save indices to disk"""
        try:
            # Convert sets to lists for JSON serialization
            source_dict = {src: list(assocs) for src, assocs in self.source_index.items()}
            target_dict = {tgt: list(assocs) for tgt, assocs in self.target_index.items()}
            type_dict = {typ: list(assocs) for typ, assocs in self.type_index.items()}
            
            indices_dir = Path(self.storage_dir) / "indices"
            indices_dir.mkdir(parents=True, exist_ok=True)
            
            with open(indices_dir / "source_index.json", "w") as f:
                json.dump(source_dict, f, indent=2)
                
            with open(indices_dir / "target_index.json", "w") as f:
                json.dump(target_dict, f, indent=2)
                
            with open(indices_dir / "type_index.json", "w") as f:
                json.dump(type_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving indices: {e}")
    
    def _load_associations(self) -> None:
        """Load associations from disk"""
        try:
            # Load associations
            assocs_dir = Path(self.storage_dir) / "associations"
            assocs_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in assocs_dir.glob("*.json"):
                try:
                    assoc_id = file_path.stem
                    with open(file_path, "r") as f:
                        assoc_data = json.load(f)
                        # Convert string back to datetime
                        if "formed_at" in assoc_data and isinstance(assoc_data["formed_at"], str):
                            assoc_data["formed_at"] = datetime.fromisoformat(assoc_data["formed_at"])
                        
                        # Create association object
                        association = AssociativeLink(**assoc_data)
                        self.associations[assoc_id] = association
                except Exception as e:
                    print(f"Error loading association from {file_path}: {e}")
            
            # Load indices
            indices_dir = Path(self.storage_dir) / "indices"
            indices_dir.mkdir(parents=True, exist_ok=True)
            
            # Source index
            source_path = indices_dir / "source_index.json"
            if source_path.exists():
                with open(source_path, "r") as f:
                    source_dict = json.load(f)
                    # Convert lists back to sets
                    self.source_index = {src: set(assocs) for src, assocs in source_dict.items()}
            
            # Target index
            target_path = indices_dir / "target_index.json"
            if target_path.exists():
                with open(target_path, "r") as f:
                    target_dict = json.load(f)
                    # Convert lists back to sets
                    self.target_index = {tgt: set(assocs) for tgt, assocs in target_dict.items()}
            
            # Type index
            type_path = indices_dir / "type_index.json"
            if type_path.exists():
                with open(type_path, "r") as f:
                    type_dict = json.load(f)
                    # Convert lists back to sets
                    self.type_index = {typ: set(assocs) for typ, assocs in type_dict.items()}
            
            # Rebuild indices if loaded associations but indices are empty
            if self.associations and (not self.source_index or not self.target_index or not self.type_index):
                self._rebuild_indices()
                
            print(f"Loaded {len(self.associations)} associations from disk")
        except Exception as e:
            print(f"Error loading associations: {e}")
    
    def _rebuild_indices(self) -> None:
        """Rebuild indices from associations"""
        self.source_index = {}
        self.target_index = {}
        self.type_index = {}
        
        for assoc_id, assoc in self.associations.items():
            # Source index
            if assoc.source_id not in self.source_index:
                self.source_index[assoc.source_id] = set()
            self.source_index[assoc.source_id].add(assoc_id)
            
            # Target index
            if assoc.target_id not in self.target_index:
                self.target_index[assoc.target_id] = set()
            self.target_index[assoc.target_id].add(assoc_id)
            
            # Type index
            if assoc.link_type not in self.type_index:
                self.type_index[assoc.link_type] = set()
            self.type_index[assoc.link_type].add(assoc_id)
    
    def _remove_association(self, association_id: str) -> None:
        """Remove an association from memory and disk"""
        if association_id not in self.associations:
            return
            
        assoc = self.associations[association_id]
        
        # Remove from source index
        if assoc.source_id in self.source_index:
            if association_id in self.source_index[assoc.source_id]:
                self.source_index[assoc.source_id].remove(association_id)
                if not self.source_index[assoc.source_id]:
                    del self.source_index[assoc.source_id]
        
        # Remove from target index
        if assoc.target_id in self.target_index:
            if association_id in self.target_index[assoc.target_id]:
                self.target_index[assoc.target_id].remove(association_id)
                if not self.target_index[assoc.target_id]:
                    del self.target_index[assoc.target_id]
        
        # Remove from type index
        if assoc.link_type in self.type_index:
            if association_id in self.type_index[assoc.link_type]:
                self.type_index[assoc.link_type].remove(association_id)
                if not self.type_index[assoc.link_type]:
                    del self.type_index[assoc.link_type]
        
        # Remove from associations dict
        del self.associations[association_id]
        
        # Remove file
        try:
            assoc_path = Path(self.storage_dir) / "associations" / f"{association_id}.json"
            if assoc_path.exists():
                assoc_path.unlink()
        except Exception as e:
            print(f"Error deleting association file {association_id}: {e}")
    
    # Event handlers
    
    def _handle_memory_stored(self, message: Message) -> None:
        """
        Handle memory stored events
        
        When a memory is stored, we may create associations between it and 
        recent or related memories.
        """
        content = message.content
        memory_id = content.get("memory_id")
        memory_content = content.get("content")
        
        if not memory_id or not memory_content:
            return
            
        # This is a simplified approach - in a real system, you would:
        # 1. Analyze memory content to find related concepts/episodes
        # 2. Create meaningful associations based on content similarity
        # 3. Adjust strength based on relevance
        
        # For demonstration, let's create weak associations with a few recent memories
        self._associate_with_recent(memory_id)
    
    def _handle_memory_retrieved(self, message: Message) -> None:
        """
        Handle memory retrieved events
        
        When a memory is retrieved, we strengthen associations to it.
        """
        content = message.content
        memory_id = content.get("memory_id")
        
        if not memory_id:
            return
            
        # Get associations involving this memory
        source_assocs = self.source_index.get(memory_id, set())
        target_assocs = self.target_index.get(memory_id, set())
        
        # Strengthen each association slightly
        for assoc_id in source_assocs.union(target_assocs):
            if assoc_id in self.associations:
                assoc = self.associations[assoc_id]
                assoc.update_strength(self.hebbian_rate * 0.3)  # Smaller strengthening
                self._save_association(assoc, assoc_id)
    
    def _handle_concept_added(self, message: Message) -> None:
        """
        Handle concept added events
        
        When a new concept is added, we may create associations with related concepts.
        """
        content = message.content
        concept_id = content.get("concept_id")
        concept_content = content.get("content", "")
        
        if not concept_id:
            return
            
        # This is a simplified approach - in a real system, you'd use NLP
        # to identify related concepts and create appropriate associations
        
        # For demonstration, associate with recent concepts
        self._associate_with_recent(concept_id, link_type="conceptual")
    
    def _handle_episode_added(self, message: Message) -> None:
        """
        Handle episode added events
        
        When a new episode is added, we may create temporal associations with previous episodes.
        """
        content = message.content
        episode_id = content.get("episode_id")
        context = content.get("context", "")
        
        if not episode_id:
            return
            
        # Associate with recent episodes in the same context
        self._associate_with_context(episode_id, context, link_type="sequential")
    
    def _handle_association_query(self, message: Message) -> None:
        """Handle association query events"""
        content = message.content
        query_type = content.get("query_type", "")
        
        if query_type == "get_associations":
            memory_id = content.get("memory_id")
            if memory_id:
                result = self.get_associations(memory_id)
                
                if self.event_bus and result["status"] == "success":
                    self.publish_message("association_query_response", {
                        "requester": message.sender,
                        "result": result,
                        "memory_id": memory_id
                    })
        
        elif query_type == "spread_activation":
            source_id = content.get("source_id")
            activation = content.get("activation", 0.5)
            depth = content.get("depth", 2)
            
            if source_id:
                result = self.spread_activation(source_id, activation, depth)
                
                if self.event_bus and result["status"] == "success":
                    self.publish_message("association_query_response", {
                        "requester": message.sender,
                        "result": result,
                        "query_type": "spread_activation"
                    })
    
    def _associate_with_recent(self, memory_id: str, limit: int = 3, link_type: str = "temporal") -> None:
        """Create associations with recent memories"""
        # Get a few recent source memories (excluding the current one)
        recent_sources = [
            src for src in self.source_index.keys() 
            if src != memory_id
        ]
        
        # Sort by recency (if we had timestamps) and limit
        # For simplicity, we'll just take a few random ones
        if recent_sources:
            import random
            sample_size = min(limit, len(recent_sources))
            selected = random.sample(recent_sources, sample_size)
            
            # Create weak associations
            for src_id in selected:
                self.associate(
                    source_id=memory_id, 
                    target_id=src_id, 
                    link_type=link_type, 
                    strength=0.3  # Weak initial association
                )
    
    def _associate_with_context(self, memory_id: str, context: str, link_type: str = "contextual") -> None:
        """Create associations with memories in the same context"""
        # This would typically be implemented using context information from episodic memory
        # For simplicity, we'll just use a weak random association if other memories exist
        
        if len(self.associations) > 10:
            # Associate with a random existing memory
            import random
            other_id = random.choice(list(self.source_index.keys()))
            
            if other_id != memory_id:
                self.associate(
                    source_id=memory_id, 
                    target_id=other_id, 
                    link_type=link_type, 
                    strength=0.25  # Weak contextual association
                ) 