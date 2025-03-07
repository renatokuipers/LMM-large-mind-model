"""
Memory module for the Large Mind Model (LMM).

This module handles the memory functions of the LMM, including
storing and retrieving memories, memory consolidation, and forgetting.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import random

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.memory.persistence import MemoryManager, MemoryType, MemoryImportance, Memory
from lmm.memory.advanced_memory import AdvancedMemoryManager, MemoryStrength, MemoryActivation
from lmm.core.mind_modules.base import MindModule
from lmm.core.development.stages import DevelopmentalStage

logger = get_logger("lmm.mind_modules.memory")

class MemoryModule(MindModule):
    """
    Memory module for storing and retrieving memories.
    
    This module manages episodic, semantic, procedural, and emotional memories,
    implementing various memory processes such as consolidation, forgetting, and retrieval.
    """
    
    def __init__(self):
        """Initialize the Memory Module with advanced memory capabilities."""
        super().__init__("Memory")
        
        # Initialize advanced memory manager
        self.memory_manager = AdvancedMemoryManager()
        
        # Track recently processed content to avoid duplicates
        self.recent_content_hash = set()
        
        # Memory consolidation scheduling
        self.last_consolidation = datetime.now()
        self.consolidation_interval = 3600  # 1 hour in seconds
        
        # Context tracking
        self.current_context: Dict[str, Any] = {}
        
        logger.info("Initialized Memory Module with advanced memory capabilities")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for memory-related operations.

        Args:
            input_data: Dictionary containing operation and parameters
                - operation: One of 'store', 'retrieve', 'associate', 'consolidate', 'forget'
                - parameters: Operation-specific parameters

        Returns:
            Dictionary with operation results
        """
        operation = input_data.get("operation", "")
        parameters = input_data.get("parameters", {})
        stage = input_data.get("developmental_stage", DevelopmentalStage.PRENATAL.value)
        
        # Update memory parameters based on developmental stage
        self.memory_manager.update_developmental_parameters(stage)
        
        results = {"success": False, "operation": operation}
        
        # Perform operation
        if operation == "store":
            results = self._store_memory(parameters)
        elif operation == "retrieve":
            results = self._retrieve_memory(parameters)
        elif operation == "search":
            results = self._search_memories(parameters)
        elif operation == "associate":
            results = self._associate_memories(parameters)
        elif operation == "consolidate":
            results = self._consolidate_memories(parameters)
        elif operation == "forget":
            results = self._forget_memories(parameters)
        elif operation == "update_context":
            results = self._update_context(parameters)
        elif operation == "get_stats":
            results = self._get_memory_stats()
        elif operation == "get_working_memory":
            results = self._get_working_memory()
        elif operation == "get_memory_graph":
            results = self._get_memory_graph(parameters)
        else:
            results["error"] = f"Unknown operation: {operation}"
        
        # Perform scheduled memory consolidation if enough time has passed
        self._check_scheduled_consolidation()
        
        return results
    
    def _store_memory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a memory.
        
        Args:
            parameters: Dictionary with memory parameters
                - content: Memory content
                - memory_type: Type of memory (episodic, semantic, procedural, emotional)
                - importance: Importance level (low, medium, high, critical)
                - context_tags: Optional list of context tags
                - related_memories: Optional list of related memory IDs
                - metadata: Optional additional metadata
                
        Returns:
            Dictionary with operation results
        """
        content = parameters.get("content", "")
        memory_type = parameters.get("memory_type", MemoryType.EPISODIC.value)
        importance = parameters.get("importance", MemoryImportance.MEDIUM.value)
        context_tags = parameters.get("context_tags", [])
        related_memories = parameters.get("related_memories", [])
        metadata = parameters.get("metadata", {})
        
        # Add current context tags if available
        if self.current_context.get("tags"):
            context_tags.extend(self.current_context.get("tags", []))
        
        # Skip if content is empty
        if not content:
            return {"success": False, "operation": "store", "error": "Empty content"}
        
        # Skip duplicates (naive content hashing)
        content_hash = hash(content)
        if content_hash in self.recent_content_hash:
            return {"success": False, "operation": "store", "error": "Duplicate content"}
        
        try:
            # Store memory with advanced manager
            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                context_tags=context_tags,
                related_memories=related_memories,
                metadata=metadata
            )
            
            # Add to recent content hash (with limited size)
            self.recent_content_hash.add(content_hash)
            if len(self.recent_content_hash) > 100:
                self.recent_content_hash.pop()
            
            logger.info(f"Stored memory ID {memory_id} of type {memory_type}")
            
            return {
                "success": True,
                "operation": "store",
                "memory_id": memory_id,
                "memory_type": memory_type
            }
        except ValueError as e:
            logger.error(f"Vector dimension error while storing memory: {str(e)}")
            return {
                "success": False,
                "operation": "store",
                "error": f"Vector store error: {str(e)}",
                "content_preview": content[:50] + "..." if len(content) > 50 else content
            }
        except Exception as e:
            logger.error(f"Unexpected error storing memory: {str(e)}")
            return {
                "success": False,
                "operation": "store",
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _retrieve_memory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve a memory by ID.
        
        Args:
            parameters: Dictionary with retrieval parameters
                - memory_id: ID of the memory to retrieve
                
        Returns:
            Dictionary with operation results
        """
        memory_id = parameters.get("memory_id")
        
        if memory_id is None:
            return {"success": False, "operation": "retrieve", "error": "Missing memory_id"}
        
        # Retrieve memory with activation effects
        memory = self.memory_manager.retrieve_memory(memory_id)
        
        if not memory:
            return {"success": False, "operation": "retrieve", "error": "Memory not found"}
        
        memory_dict = {
            "id": memory.vector_store_id,
            "content": memory.content,
            "type": memory.memory_type.value,
            "importance": memory.importance.value,
            "created_at": memory.created_at.isoformat(),
            "metadata": memory.metadata
        }
        
        # If reconstruction effects were applied, add a note
        if memory.metadata.get("reconstructed"):
            memory_dict["reconstructed"] = True
            memory_dict["confidence"] = memory.metadata.get("confidence", 0.5)
        
        return {
            "success": True,
            "operation": "retrieve",
            "memory": memory_dict
        }
    
    def _search_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for memories.
        
        Args:
            parameters: Dictionary with search parameters
                - query: Search query
                - memory_type: Optional type filter
                - min_importance: Optional minimum importance level
                - limit: Optional result limit
                - min_activation: Optional minimum activation level
                - context_tags: Optional context tags to filter by
                - retrieval_strategy: Optional strategy (vector, graph, context, combined)
                
        Returns:
            Dictionary with operation results
        """
        query = parameters.get("query", "")
        memory_type = parameters.get("memory_type")
        min_importance = parameters.get("min_importance")
        limit = parameters.get("limit", 5)
        min_activation = parameters.get("min_activation", 0.0)
        context_tags = parameters.get("context_tags", self.current_context.get("tags", []))
        retrieval_strategy = parameters.get("retrieval_strategy", "combined")
        
        if not query:
            return {"success": False, "operation": "search", "error": "Empty query"}
        
        try:
            # Search memories with enhanced retrieval
            memories = self.memory_manager.search_memories(
                query=query,
                memory_type=memory_type,
                min_importance=min_importance,
                context_tags=context_tags,
                min_activation=min_activation,
                limit=limit,
                retrieval_strategy=retrieval_strategy
            )
            
            # Convert to dictionary format
            memory_dicts = []
            for memory in memories:
                memory_dict = {
                    "id": memory.vector_store_id,
                    "content": memory.content,
                    "type": memory.memory_type.value,
                    "importance": memory.importance.value,
                    "created_at": memory.created_at.isoformat(),
                    "metadata": memory.metadata
                }
                
                # Add retrieval score if available
                if hasattr(memory, 'retrieval_score'):
                    memory_dict["retrieval_score"] = memory.retrieval_score
                
                memory_dicts.append(memory_dict)
            
            return {
                "success": True,
                "operation": "search",
                "memories": memory_dicts,
                "count": len(memory_dicts)
            }
        except ValueError as e:
            logger.error(f"Vector dimension error while searching memories: {str(e)}")
            return {
                "success": False,
                "operation": "search",
                "error": f"Vector store error: {str(e)}",
                "memories": [],
                "count": 0
            }
        except Exception as e:
            logger.error(f"Unexpected error searching memories: {str(e)}")
            return {
                "success": False,
                "operation": "search",
                "error": f"Unexpected error: {str(e)}",
                "memories": [],
                "count": 0
            }
    
    def _associate_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an association between two memories.
        
        Args:
            parameters: Dictionary with association parameters
                - memory_id1: First memory ID
                - memory_id2: Second memory ID
                - strength: Optional association strength (0.0-1.0)
                
        Returns:
            Dictionary with operation results
        """
        memory_id1 = parameters.get("memory_id1")
        memory_id2 = parameters.get("memory_id2")
        strength = parameters.get("strength", 0.5)
        
        if memory_id1 is None or memory_id2 is None:
            return {"success": False, "operation": "associate", "error": "Missing memory IDs"}
        
        # Associate memories
        success = self.memory_manager.associate_memories(memory_id1, memory_id2, strength)
        
        if not success:
            return {"success": False, "operation": "associate", "error": "Association failed"}
        
        return {
            "success": True,
            "operation": "associate",
            "memory_id1": memory_id1,
            "memory_id2": memory_id2,
            "strength": strength
        }
    
    def _consolidate_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate memories.
        
        Args:
            parameters: Dictionary with consolidation parameters
                - force: Optional boolean to force consolidation
                
        Returns:
            Dictionary with operation results
        """
        force = parameters.get("force", False)
        
        # Consolidate memories
        consolidated_count = self.memory_manager.consolidate_memories(force)
        
        # Update last consolidation time
        self.last_consolidation = datetime.now()
        
        return {
            "success": True,
            "operation": "consolidate",
            "consolidated_count": consolidated_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def _forget_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actively forget memories.
        
        Args:
            parameters: Dictionary with forgetting parameters
                - older_than_days: Optional minimum age in days
                - memory_type: Optional type filter
                - max_importance: Optional maximum importance level
                - max_strength: Optional maximum memory strength
                
        Returns:
            Dictionary with operation results
        """
        older_than_days = parameters.get("older_than_days")
        memory_type = parameters.get("memory_type")
        max_importance = parameters.get("max_importance", MemoryImportance.LOW.value)
        max_strength = parameters.get("max_strength", 0.3)
        
        # Forget memories matching criteria
        forgotten_count = self.memory_manager.forget_memories(
            older_than_days=older_than_days,
            memory_type=memory_type,
            max_importance=max_importance,
            max_strength=max_strength
        )
        
        return {
            "success": True,
            "operation": "forget",
            "forgotten_count": forgotten_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current context for memory operations.
        
        Args:
            parameters: Dictionary with context parameters
                - tags: List of context tags
                - metadata: Additional context metadata
                
        Returns:
            Dictionary with operation results
        """
        tags = parameters.get("tags", [])
        metadata = parameters.get("metadata", {})
        
        # Update current context
        self.current_context = {
            "tags": tags,
            "metadata": metadata,
            "updated_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "operation": "update_context",
            "context": self.current_context
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = self.memory_manager.get_memory_stats()
        
        return {
            "success": True,
            "operation": "get_stats",
            "stats": stats
        }
    
    def _get_working_memory(self) -> Dict[str, Any]:
        """
        Get the current contents of working memory.
        
        Returns:
            Dictionary with working memory contents
        """
        contents = self.memory_manager.get_working_memory_contents()
        
        return {
            "success": True,
            "operation": "get_working_memory",
            "contents": contents,
            "capacity": self.memory_manager.working_memory_capacity,
            "usage": len(contents)
        }
    
    def _check_scheduled_consolidation(self) -> None:
        """
        Check if memory consolidation should be performed.
        
        This simulates the natural memory consolidation that occurs over time.
        """
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        
        if time_since_last >= self.consolidation_interval:
            # Perform consolidation
            self.memory_manager.consolidate_memories()
            
            # Update last consolidation time
            self.last_consolidation = datetime.now()
            
            logger.info("Performed scheduled memory consolidation")
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get the current status of the memory module.
        
        Returns:
            Dictionary with module status
        """
        stats = self.memory_manager.get_memory_stats()
        working_memory = self.memory_manager.get_working_memory_contents()
        
        return {
            "name": self.name,
            "status": "active",
            "memory_counts": {
                "total": stats.get("total_memories", 0),
                "by_type": stats.get("memory_types", {}),
                "by_importance": stats.get("memory_importance", {})
            },
            "working_memory": {
                "capacity": self.memory_manager.working_memory_capacity,
                "usage": len(working_memory),
                "contents": [item["content"][:50] + "..." for item in working_memory]
            },
            "memory_strength": stats.get("strength_distribution", {}),
            "memory_activation": stats.get("activation_distribution", {})
        }
    
    def _get_memory_graph(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the memory graph data for visualization.
        
        Args:
            parameters: Dictionary with parameters
                - limit: Maximum number of nodes to include
                
        Returns:
            Dictionary with operation results
        """
        limit = parameters.get("limit", 100)
        
        # Get memory graph data from advanced memory manager
        graph_data = self.memory_manager.get_memory_graph(limit=limit)
        
        return {
            "success": True,
            "operation": "get_memory_graph",
            "graph": graph_data
        } 