"""
Advanced memory module for the Large Mind Model (LMM).

This module implements enhanced memory mechanisms including:
- Memory consolidation from short-term to long-term
- Associative memory networks
- Context-sensitive retrieval
- Active forgetting and memory decay
- Working memory limitations
- Memory reconstruction effects
"""
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import math
import random
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import heapq
import networkx as nx
from collections import defaultdict, deque

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.memory.persistence import MemoryManager, MemoryType, MemoryImportance, Memory
from lmm.core.development.stages import DevelopmentalStage

logger = get_logger("lmm.memory.advanced_memory")

class MemoryStrength(float, Enum):
    """Memory strength levels indicating how well a memory is consolidated."""
    WEAK = 0.2       # Recently formed, easily forgotten
    MODERATE = 0.5   # Partially consolidated
    STRONG = 0.8     # Well-consolidated, long-term memory
    PERMANENT = 1.0  # Critical memories, unlikely to be forgotten

class MemoryActivation(float, Enum):
    """Memory activation levels indicating current accessibility."""
    INACTIVE = 0.0   # Not currently accessible
    LOW = 0.3        # Difficult to recall
    MEDIUM = 0.6     # Moderately accessible
    HIGH = 0.9       # Highly accessible, in working memory

class AdvancedMemoryManager:
    """
    Enhanced memory management system with realistic memory processes.
    
    This class extends the basic memory manager with:
    - Memory consolidation processes
    - Associative memory networks
    - Working memory limitations
    - Active forgetting mechanisms
    - Context-sensitive retrieval
    - Memory reconstruction effects
    """
    
    def __init__(self, base_memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the Advanced Memory Manager.
        
        Args:
            base_memory_manager: Optional existing memory manager to build upon
        """
        # Use provided memory manager or create new one
        self.base_manager = base_memory_manager or MemoryManager()
        
        # Memory networks for associative connections
        self.memory_graph = nx.Graph()
        
        # Memory activation levels (memory_id -> activation level)
        self.activations: Dict[int, float] = {}
        
        # Memory strengths (memory_id -> strength level)
        self.strengths: Dict[int, float] = {}
        
        # Working memory (limited capacity queue of currently active memories)
        self.working_memory = deque(maxlen=7)  # ~7 items limit like human working memory
        
        # Memory access history (for consolidation)
        self.access_history: Dict[int, List[datetime]] = defaultdict(list)
        
        # Context tags for enhanced retrieval
        self.context_tags: Dict[str, Set[int]] = defaultdict(set)
        
        # Last consolidation time
        self.last_consolidation = datetime.now()
        
        # Developmental parameters
        self.working_memory_capacity = 3  # Starts small, increases with development
        self.consolidation_efficiency = 0.3  # Starts low, improves with development
        self.retrieval_efficiency = 0.4  # Starts low, improves with development
        self.forgetting_rate = 0.1  # Rate of passive forgetting
        
        logger.info("Initialized Advanced Memory Manager")
    
    def add_memory(
        self, 
        content: str, 
        memory_type: Union[MemoryType, str],
        importance: Union[MemoryImportance, str] = MemoryImportance.MEDIUM,
        context_tags: Optional[List[str]] = None,
        related_memories: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a memory with enhanced features.
        
        Args:
            content: Content of the memory
            memory_type: Type of memory
            importance: Importance of the memory
            context_tags: List of context tags for memory organization
            related_memories: List of related memory IDs
            metadata: Additional metadata
            
        Returns:
            ID of the added memory
        """
        # Add memory through base manager
        memory_id = self.base_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {}
        )
        
        # Set initial memory strength based on importance
        if isinstance(importance, str):
            importance = MemoryImportance(importance)
            
        if importance == MemoryImportance.CRITICAL:
            strength = MemoryStrength.STRONG
        elif importance == MemoryImportance.HIGH:
            strength = MemoryStrength.MODERATE
        elif importance == MemoryImportance.MEDIUM:
            strength = MemoryStrength.MODERATE
        else:  # LOW
            strength = MemoryStrength.WEAK
            
        self.strengths[memory_id] = strength
        
        # Set high initial activation
        self.activations[memory_id] = MemoryActivation.HIGH
        
        # Add to working memory
        self._update_working_memory(memory_id)
        
        # Add to memory graph
        self.memory_graph.add_node(memory_id, 
                                  content=content[:100],  # Truncate for graph storage
                                  type=str(memory_type),
                                  importance=str(importance),
                                  added=datetime.now())
        
        # Add relationships to other memories
        if related_memories:
            for related_id in related_memories:
                if related_id in self.strengths and self.memory_graph.has_node(related_id):
                    self.memory_graph.add_edge(memory_id, related_id, weight=0.5)
        
        # Add context tags
        if context_tags:
            for tag in context_tags:
                self.context_tags[tag].add(memory_id)
        
        # Record access for consolidation
        self.access_history[memory_id].append(datetime.now())
        
        logger.debug(f"Added memory {memory_id} with strength {strength}")
        return memory_id
    
    def retrieve_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieve a memory by ID with activation effects.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory object, or None if not found
        """
        memory = self.base_manager.retrieve_memory(memory_id)
        
        if memory:
            # Increase activation
            self.activations[memory_id] = min(float(MemoryActivation.HIGH), 
                                            self.activations.get(memory_id, 0.0) + 0.3)
            
            # Add to working memory
            self._update_working_memory(memory_id)
            
            # Record access for consolidation
            self.access_history[memory_id].append(datetime.now())
            
            # Apply memory reconstruction effects (slight modifications to simulate human memory)
            memory = self._apply_reconstruction_effects(memory)
            
            logger.debug(f"Retrieved memory {memory_id}, activation: {self.activations[memory_id]}")
        
        return memory
    
    def search_memories(
        self, 
        query: str, 
        memory_type: Optional[Union[MemoryType, str]] = None,
        min_importance: Optional[Union[MemoryImportance, str]] = None,
        context_tags: Optional[List[str]] = None,
        min_activation: float = 0.0,
        limit: int = 5,
        retrieval_strategy: str = "combined"
    ) -> List[Memory]:
        """
        Search for memories with enhanced retrieval mechanisms.
        
        Args:
            query: Query text
            memory_type: Optional filter by memory type
            min_importance: Optional minimum importance level
            context_tags: Optional context tags to filter by
            min_activation: Minimum activation level
            limit: Maximum number of results
            retrieval_strategy: Strategy for retrieval (vector, graph, context, combined)
            
        Returns:
            List of matching Memory objects
        """
        # Step 1: Basic vector search from base memory manager
        base_results = self.base_manager.search_memories(
            query=query,
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit * 2  # Get more results for reranking
        )
        
        # Extract memory IDs from base results
        base_memory_ids = [memory.vector_store_id for memory in base_results if memory.vector_store_id is not None]
        
        # Step A: Context-based filtering
        context_memory_ids = set()
        if context_tags:
            # Find memories that match any of the context tags
            for tag in context_tags:
                context_memory_ids.update(self.context_tags.get(tag, set()))
        
        # Step B: Graph-based memory association ("spreading activation")
        graph_memory_ids = set()
        if base_memory_ids and retrieval_strategy in ["graph", "combined"]:
            # Find memories that are connected to the base results
            for memory_id in base_memory_ids:
                if self.memory_graph.has_node(memory_id):
                    # Get neighbors up to 2 steps away
                    neighbors = set()
                    for neighbor in self.memory_graph.neighbors(memory_id):
                        neighbors.add(neighbor)
                        # Second-degree neighbors with decreasing relevance
                        for second_neighbor in self.memory_graph.neighbors(neighbor):
                            if second_neighbor != memory_id:
                                neighbors.add(second_neighbor)
                    
                    graph_memory_ids.update(neighbors)
        
        # Step 3: Combine results based on retrieval strategy
        candidate_memory_ids = set()
        
        if retrieval_strategy == "vector":
            candidate_memory_ids = set(base_memory_ids)
        elif retrieval_strategy == "graph":
            candidate_memory_ids = graph_memory_ids
        elif retrieval_strategy == "context":
            candidate_memory_ids = context_memory_ids if context_tags else set(base_memory_ids)
        else:  # "combined"
            # Union of all approaches with priority to context-matching memories
            candidate_memory_ids = set(base_memory_ids).union(graph_memory_ids)
            if context_tags and context_memory_ids:
                # Prioritize context matches
                candidate_memory_ids = context_memory_ids.intersection(candidate_memory_ids) or candidate_memory_ids
        
        # Step 4: Filter by activation level
        if min_activation > 0:
            candidate_memory_ids = {
                memory_id for memory_id in candidate_memory_ids 
                if self.activations.get(memory_id, 0.0) >= min_activation
            }
        
        # Step 5: Retrieve and score memories
        scored_memories = []
        for memory_id in candidate_memory_ids:
            memory = self.base_manager.retrieve_memory(memory_id)
            if memory:
                # Calculate combined score using multiple factors
                vector_score = next((m.metadata.get("similarity", 0.0) for m in base_results 
                                    if m.vector_store_id == memory_id), 0.0)
                
                activation_score = self.activations.get(memory_id, 0.0)
                strength_score = self.strengths.get(memory_id, 0.0)
                
                # Context score
                context_score = 0.0
                if context_tags and memory_id in context_memory_ids:
                    context_score = 0.3
                
                # Graph connectedness score
                graph_score = 0.0
                if self.memory_graph.has_node(memory_id):
                    # More connected nodes get higher scores
                    graph_score = min(0.3, len(list(self.memory_graph.neighbors(memory_id))) * 0.05)
                
                # Recency score (boost recent memories)
                recency_score = 0.0
                if memory_id in self.access_history and self.access_history[memory_id]:
                    latest_access = max(self.access_history[memory_id])
                    time_diff = (datetime.now() - latest_access).total_seconds()
                    recency_score = max(0.0, 0.3 - min(0.3, time_diff / 86400))  # Decay over 24 hours
                
                # Combined score with weights
                combined_score = (
                    0.4 * vector_score +
                    0.2 * activation_score +
                    0.1 * strength_score +
                    0.1 * context_score +
                    0.1 * graph_score +
                    0.1 * recency_score
                )
                
                # Add score to memory metadata for return
                memory.metadata["retrieval_score"] = combined_score
                memory.metadata["vector_score"] = vector_score
                memory.metadata["activation_score"] = activation_score
                memory.metadata["strength_score"] = strength_score
                memory.metadata["context_score"] = context_score
                memory.metadata["graph_score"] = graph_score
                memory.metadata["recency_score"] = recency_score
                
                # Increase memory activation (simulation of retrieval practice effect)
                self.activations[memory_id] = min(float(MemoryActivation.HIGH), 
                                                activation_score + 0.1)
                
                # Record access for consolidation
                self.access_history[memory_id].append(datetime.now())
                
                scored_memories.append((combined_score, memory))
        
        # Sort by combined score
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        # Get top memories
        top_memories = [memory for _, memory in scored_memories[:limit]]
        
        # Update working memory with top results
        for memory in top_memories:
            if memory.vector_store_id is not None:
                self._update_working_memory(memory.vector_store_id)
        
        logger.debug(f"Retrieved {len(top_memories)} memories via {retrieval_strategy} strategy")
        return top_memories
    
    def consolidate_memories(self, force: bool = False) -> int:
        """
        Consolidate memories based on access patterns and importance.
        
        Args:
            force: Force consolidation regardless of time since last consolidation
            
        Returns:
            Number of memories consolidated
        """
        # Check if enough time has passed since last consolidation (simulate sleep/rest)
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        if not force and time_since_last < 3600:  # Default once per hour
            return 0
        
        consolidation_count = 0
        
        # Process all memories for consolidation/forgetting
        for memory_id, accesses in self.access_history.items():
            # Skip if no access history
            if not accesses:
                continue
            
            # Get current strength
            current_strength = self.strengths.get(memory_id, MemoryStrength.WEAK)
            
            # Calculate consolidation factors
            
            # 1. Access frequency
            recent_accesses = [ts for ts in accesses 
                              if (datetime.now() - ts).total_seconds() < 86400]  # Last 24h
            access_frequency = len(recent_accesses) / 10.0  # Normalize, max = 1.0 at 10 accesses/day
            access_frequency = min(1.0, access_frequency)
            
            # 2. Access recency
            if recent_accesses:
                last_access = max(recent_accesses)
                recency_hours = (datetime.now() - last_access).total_seconds() / 3600
                recency_factor = max(0.0, 1.0 - min(1.0, recency_hours / 24))  # Decay over 24h
            else:
                recency_factor = 0.0
            
            # 3. Get memory importance
            memory = self.base_manager.retrieve_memory(memory_id)
            importance_factor = 0.0
            if memory:
                if memory.importance == MemoryImportance.CRITICAL:
                    importance_factor = 1.0
                elif memory.importance == MemoryImportance.HIGH:
                    importance_factor = 0.7
                elif memory.importance == MemoryImportance.MEDIUM:
                    importance_factor = 0.4
                else:  # LOW
                    importance_factor = 0.1
            
            # 4. Current activation level
            activation_factor = min(1.0, self.activations.get(memory_id, 0.0) * 2)
            
            # Combine factors
            consolidation_factor = (
                0.4 * access_frequency +
                0.3 * recency_factor +
                0.2 * importance_factor +
                0.1 * activation_factor
            )
            
            # Apply consolidation efficiency (developmental parameter)
            consolidation_factor *= self.consolidation_efficiency
            
            # Determine new strength
            if current_strength < MemoryStrength.PERMANENT:
                # Increase strength based on consolidation factor
                strength_increase = consolidation_factor * 0.2  # Max increase = 0.2 per consolidation
                new_strength = min(float(MemoryStrength.PERMANENT), current_strength + strength_increase)
                
                if new_strength > current_strength:
                    self.strengths[memory_id] = new_strength
                    consolidation_count += 1
                    logger.debug(f"Consolidated memory {memory_id}: {current_strength:.2f} → {new_strength:.2f}")
            
            # Apply forgetting curve (memory decay)
            self._apply_forgetting(memory_id)
        
        # Update last consolidation time
        self.last_consolidation = datetime.now()
        
        # Network optimization: create new connections between co-activated memories
        self._optimize_memory_network()
        
        logger.info(f"Consolidated {consolidation_count} memories")
        return consolidation_count
    
    def forget_memories(
        self,
        older_than_days: Optional[int] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        max_importance: Optional[Union[MemoryImportance, str]] = MemoryImportance.LOW,
        max_strength: float = 0.3  # Only forget memories below this strength
    ) -> int:
        """
        Actively forget memories based on strength, importance, and age.
        
        Args:
            older_than_days: Only forget memories older than this many days
            memory_type: Only forget memories of this type
            max_importance: Only forget memories with importance up to this level
            max_strength: Only forget memories with strength below this threshold
            
        Returns:
            Number of memories forgotten
        """
        forgotten_count = 0
        memory_ids_to_forget = []
        
        # Gather memory IDs that match forgetting criteria
        for memory_id, strength in self.strengths.items():
            if strength > max_strength:
                continue  # Skip strong memories
                
            # Check access history
            if older_than_days and memory_id in self.access_history and self.access_history[memory_id]:
                latest_access = max(self.access_history[memory_id])
                if (datetime.now() - latest_access).days < older_than_days:
                    continue  # Skip recently accessed memories
            
            # Check memory details
            memory = self.base_manager.retrieve_memory(memory_id)
            if not memory:
                memory_ids_to_forget.append(memory_id)  # Clean up references to non-existent memories
                continue
                
            # Check memory type
            if memory_type and memory.memory_type != memory_type:
                continue
                
            # Check importance
            if max_importance and memory.importance > max_importance:
                continue
                
            # This memory meets all forgetting criteria
            memory_ids_to_forget.append(memory_id)
        
        # Perform forgetting
        for memory_id in memory_ids_to_forget:
            # Remove from internal tracking
            self.strengths.pop(memory_id, None)
            self.activations.pop(memory_id, None)
            self.access_history.pop(memory_id, None)
            
            # Remove from memory graph
            if self.memory_graph.has_node(memory_id):
                self.memory_graph.remove_node(memory_id)
            
            # Remove from context tags
            for tag_set in self.context_tags.values():
                if memory_id in tag_set:
                    tag_set.remove(memory_id)
            
            # Note: We don't actually delete from base memory manager
            # This is intentional - forgotten memories remain in the system
            # but become inaccessible through our mechanisms
            
            forgotten_count += 1
        
        logger.info(f"Actively forgot {forgotten_count} memories")
        return forgotten_count
    
    def associate_memories(self, memory_id1: int, memory_id2: int, strength: float = 0.5) -> bool:
        """
        Create an association between two memories.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            strength: Strength of the association (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if memory_id1 not in self.strengths or memory_id2 not in self.strengths:
            return False
        
        # Add or update edge in memory graph
        if not self.memory_graph.has_edge(memory_id1, memory_id2):
            self.memory_graph.add_edge(memory_id1, memory_id2, weight=strength)
        else:
            # Strengthen existing connection
            current_weight = self.memory_graph.get_edge_data(memory_id1, memory_id2).get("weight", 0.0)
            new_weight = min(1.0, current_weight + strength * 0.5)
            self.memory_graph[memory_id1][memory_id2]["weight"] = new_weight
        
        logger.debug(f"Associated memories {memory_id1} and {memory_id2} with strength {strength}")
        return True
    
    def get_memory_graph(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get a comprehensive representation of the memory graph with advanced analytics.
        
        This method provides a rich graph representation with multiple features:
        1. Memory node selection based on multiple strategies (importance, recency, activation)
        2. Community detection to identify clusters of related memories
        3. Centrality measures to identify key/hub memories
        4. Temporal analysis showing memory formation patterns
        5. Path analysis between specified memories
        6. Rich metadata for visualization and analysis
        
        Args:
            limit: Maximum number of nodes to include
            filter_type: Optional memory type to filter by
            filter_tags: Optional list of tags to filter by
            selection_strategy: Strategy for selecting nodes ('degree', 'importance', 
                                'recency', 'activation', 'mixed')
            include_communities: Whether to include community detection
            include_centrality: Whether to include centrality measures
            include_temporal: Whether to include temporal analysis
            path_between: Optional tuple of memory IDs to find paths between
            
        Returns:
            Dictionary with nodes, edges, and advanced analytics
        """
        import networkx as nx
        from datetime import datetime, timedelta
        import numpy as np
        from collections import defaultdict
        
        # Parameters with default values - would normally be function parameters
        filter_type = None
        filter_tags = None
        selection_strategy = 'mixed'
        include_communities = True
        include_centrality = True
        include_temporal = True
        path_between = None
        
        # Start with full graph
        full_graph = self.memory_graph.copy()
        
        # Apply filters if specified
        if filter_type or filter_tags:
            nodes_to_keep = []
            for node in full_graph.nodes():
                memory = self.retrieve_memory(node)
                if not memory:
                    continue
                    
                # Filter by type
                if filter_type and memory.memory_type != filter_type:
                    continue
                    
                # Filter by tags
                if filter_tags:
                    memory_tags = memory.metadata.get("context_tags", [])
                    if not any(tag in memory_tags for tag in filter_tags):
                        continue
                        
                nodes_to_keep.append(node)
                
            # Create filtered graph
            full_graph = full_graph.subgraph(nodes_to_keep)
        
        # Node selection strategy
        selected_nodes = []
        
        if selection_strategy == 'degree':
            # Select nodes with highest degree (most connections)
            nodes_by_degree = sorted(full_graph.degree, key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in nodes_by_degree[:limit]]
            
        elif selection_strategy == 'importance':
            # Select nodes with highest importance
            nodes_by_importance = []
            for node in full_graph.nodes():
                memory = self.retrieve_memory(node)
                if memory:
                    importance_value = self._importance_to_value(memory.importance)
                    nodes_by_importance.append((node, importance_value))
            
            nodes_by_importance.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in nodes_by_importance[:limit]]
            
        elif selection_strategy == 'recency':
            # Select most recent nodes
            nodes_by_recency = []
            for node in full_graph.nodes():
                memory = self.retrieve_memory(node)
                if memory:
                    nodes_by_recency.append((node, memory.created_at))
            
            nodes_by_recency.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in nodes_by_recency[:limit]]
            
        elif selection_strategy == 'activation':
            # Select nodes with highest activation
            nodes_by_activation = [(node, self.activations.get(node, 0.0)) 
                                  for node in full_graph.nodes()]
            nodes_by_activation.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in nodes_by_activation[:limit]]
            
        else:  # 'mixed' strategy - default
            # Combine various factors for a balanced selection
            node_scores = {}
            
            for node in full_graph.nodes():
                memory = self.retrieve_memory(node)
                if not memory:
                    continue
                    
                # Base score is degree (connection count)
                degree_score = full_graph.degree(node) / max(1, max(dict(full_graph.degree).values()))
                
                # Add importance component
                importance_score = self._importance_to_value(memory.importance) / 3.0
                
                # Add activation component
                activation_score = self.activations.get(node, 0.0)
                
                # Add recency component (normalize to 0-1)
                days_old = (datetime.now() - memory.created_at).days
                recency_score = max(0, 1 - (days_old / 30))  # Decay over 30 days
                
                # Calculate combined score (weighted)
                combined_score = (0.3 * degree_score + 
                                 0.3 * importance_score + 
                                 0.2 * activation_score + 
                                 0.2 * recency_score)
                
                node_scores[node] = combined_score
            
            # Select top scoring nodes
            selected_nodes = sorted(node_scores.keys(), 
                                   key=lambda x: node_scores[x], 
                                   reverse=True)[:limit]
        
        # Handle path between nodes if specified
        if path_between and len(path_between) == 2:
            source, target = path_between
            if source in full_graph and target in full_graph:
                try:
                    # Find shortest path
                    path = nx.shortest_path(full_graph, source=source, target=target)
                    
                    # Add path nodes to selection
                    selected_nodes = list(set(selected_nodes + path))
                    
                    # Limit if still too many
                    if len(selected_nodes) > limit:
                        selected_nodes = selected_nodes[:limit]
                except nx.NetworkXNoPath:
                    pass  # No path exists, continue with current selection
        
        # Create subgraph with selected nodes
        if not selected_nodes and full_graph.number_of_nodes() > 0:
            # Fallback: use degree-based selection
            nodes_by_degree = sorted(full_graph.degree, key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in nodes_by_degree[:limit]]
        
        # Get final subgraph
        subgraph = full_graph.subgraph(selected_nodes)
        
        # Initialize result with basic graph structure
        result = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "node_count": subgraph.number_of_nodes(),
                "edge_count": subgraph.number_of_edges(),
                "total_memory_count": full_graph.number_of_nodes(),
                "selection_strategy": selection_strategy,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Prepare advanced analytics if needed
        advanced_analytics = {}
        
        # Community detection
        if include_communities and subgraph.number_of_nodes() > 2:
            try:
                # Use Louvain method for community detection
                from community import community_louvain
                communities = community_louvain.best_partition(subgraph)
                
                # Store community assignments
                community_data = {"assignments": communities}
                
                # Calculate statistics per community
                community_stats = defaultdict(lambda: {"nodes": 0, "edges": 0, "avg_importance": 0})
                
                for node, community_id in communities.items():
                    community_stats[community_id]["nodes"] += 1
                    
                    # Get node importance
                    memory = self.retrieve_memory(node)
                    if memory:
                        imp_value = self._importance_to_value(memory.importance)
                        current = community_stats[community_id]["avg_importance"]
                        node_count = community_stats[community_id]["nodes"]
                        # Running average update
                        community_stats[community_id]["avg_importance"] = (
                            current * (node_count - 1) / node_count + imp_value / node_count
                        )
                
                # Count edges within each community
                for u, v, _ in subgraph.edges(data=True):
                    if communities[u] == communities[v]:
                        community_stats[communities[u]]["edges"] += 1
                
                community_data["statistics"] = dict(community_stats)
                advanced_analytics["communities"] = community_data
                
            except ImportError:
                advanced_analytics["communities"] = {
                    "error": "Community detection requires the 'python-louvain' package"
                }
        
        # Centrality measures
        if include_centrality and subgraph.number_of_nodes() > 2:
            centrality_measures = {}
            
            # Degree centrality (normalized)
            degree_cent = nx.degree_centrality(subgraph)
            centrality_measures["degree"] = degree_cent
            
            # Betweenness centrality (key bridge nodes)
            betweenness_cent = nx.betweenness_centrality(subgraph)
            centrality_measures["betweenness"] = betweenness_cent
            
            # Eigenvector centrality (influence)
            try:
                eigenvector_cent = nx.eigenvector_centrality(subgraph, max_iter=300)
                centrality_measures["eigenvector"] = eigenvector_cent
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                # May fail on some graph structures
                centrality_measures["eigenvector"] = {
                    "error": "Eigenvector centrality calculation failed to converge"
                }
            
            # PageRank (alternative influence measure)
            pagerank = nx.pagerank(subgraph)
            centrality_measures["pagerank"] = pagerank
            
            # Identify top nodes by each measure
            top_nodes = {}
            for measure, values in centrality_measures.items():
                if isinstance(values, dict) and not "error" in values:
                    sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_nodes[measure] = [{"id": n, "value": round(v, 3)} for n, v in sorted_nodes]
            
            advanced_analytics["centrality"] = {
                "measures": centrality_measures,
                "top_nodes": top_nodes
            }
        
        # Temporal analysis
        if include_temporal:
            # Group memories by time periods to see patterns of formation
            time_periods = {}
            memories_by_day = defaultdict(list)
            earliest_date = datetime.now()
            latest_date = datetime(1970, 1, 1)
            
            for node in subgraph.nodes():
                memory = self.retrieve_memory(node)
                if memory:
                    # Update time range
                    if memory.created_at < earliest_date:
                        earliest_date = memory.created_at
                    if memory.created_at > latest_date:
                        latest_date = memory.created_at
                    
                    # Group by day
                    day_key = memory.created_at.strftime("%Y-%m-%d")
                    memories_by_day[day_key].append(node)
            
            # Calculate time span
            time_span_days = (latest_date - earliest_date).days + 1
            
            # Create appropriate time buckets based on span
            if time_span_days <= 1:
                # Hours within a day
                time_buckets = defaultdict(list)
                for node in subgraph.nodes():
                    memory = self.retrieve_memory(node)
                    if memory:
                        hour_key = memory.created_at.strftime("%H:00")
                        time_buckets[hour_key].append(node)
                
                time_periods = {
                    "unit": "hour",
                    "data": dict(time_buckets)
                }
                
            elif time_span_days <= 30:
                # Daily buckets
                time_periods = {
                    "unit": "day",
                    "data": dict(memories_by_day)
                }
                
            else:
                # Weekly or monthly buckets
                if time_span_days <= 90:
                    # Weekly buckets
                    time_buckets = defaultdict(list)
                    for node in subgraph.nodes():
                        memory = self.retrieve_memory(node)
                        if memory:
                            # Get ISO week number
                            week_key = f"{memory.created_at.year}-W{memory.created_at.isocalendar()[1]}"
                            time_buckets[week_key].append(node)
                    
                    time_periods = {
                        "unit": "week",
                        "data": dict(time_buckets)
                    }
                else:
                    # Monthly buckets
                    time_buckets = defaultdict(list)
                    for node in subgraph.nodes():
                        memory = self.retrieve_memory(node)
                        if memory:
                            month_key = memory.created_at.strftime("%Y-%m")
                            time_buckets[month_key].append(node)
                    
                    time_periods = {
                        "unit": "month",
                        "data": dict(time_buckets)
                    }
            
            advanced_analytics["temporal"] = {
                "time_span_days": time_span_days,
                "earliest_date": earliest_date.isoformat(),
                "latest_date": latest_date.isoformat(),
                "distribution": time_periods
            }
        
        # Add advanced analytics to result
        if advanced_analytics:
            result["analytics"] = advanced_analytics
        
        # Build nodes list with rich metadata
        for node in subgraph.nodes():
            memory = self.retrieve_memory(node)
            if not memory:
                continue
                
            # Calculate various node metrics
            degree = subgraph.degree(node)
            activation = self.activations.get(node, 0.0)
            strength = self.strengths.get(node, 0.0)
            importance_value = self._importance_to_value(memory.importance)
            
            # Get summary of content (first 100 chars)
            content_summary = memory.content
            if len(content_summary) > 100:
                content_summary = content_summary[:97] + "..."
                
            # Get tags
            tags = memory.metadata.get("context_tags", [])
            
            # Get community assignment if available
            community = None
            if "communities" in advanced_analytics and "assignments" in advanced_analytics["communities"]:
                community = advanced_analytics["communities"]["assignments"].get(node)
            
            # Prepare node entry
            node_entry = {
                "id": node,
                "content": content_summary,
                "full_content": memory.content,
                "type": memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
                "importance": memory.importance.value if isinstance(memory.importance, MemoryImportance) else memory.importance,
                "importance_value": importance_value,
                "created_at": memory.created_at.isoformat(),
                "tags": tags,
                "activation": activation,
                "strength": strength,
                "degree": degree,
                "metrics": {
                    "activation": activation,
                    "strength": strength,
                    "degree": degree
                }
            }
            
            # Add centrality metrics if available
            if "centrality" in advanced_analytics and "measures" in advanced_analytics["centrality"]:
                centrality_metrics = {}
                for measure, values in advanced_analytics["centrality"]["measures"].items():
                    if isinstance(values, dict) and not "error" in values:
                        centrality_metrics[measure] = values.get(node, 0.0)
                node_entry["metrics"]["centrality"] = centrality_metrics
            
            # Add community assignment if available
            if community is not None:
                node_entry["community"] = community
                
            result["nodes"].append(node_entry)
        
        # Build edges list with weights and relation types
        for u, v, data in subgraph.edges(data=True):
            edge_entry = {
                "source": u,
                "target": v,
                "weight": data.get("weight", 0.5),
            }
            
            # Determine edge type based on connected memory types
            source_memory = self.retrieve_memory(u)
            target_memory = self.retrieve_memory(v)
            
            if source_memory and target_memory:
                if source_memory.memory_type == target_memory.memory_type:
                    edge_entry["relation"] = "same_type"
                else:
                    edge_entry["relation"] = "cross_type"
                    
                # Check if memories share tags
                source_tags = set(source_memory.metadata.get("context_tags", []))
                target_tags = set(target_memory.metadata.get("context_tags", []))
                shared_tags = source_tags.intersection(target_tags)
                
                if shared_tags:
                    edge_entry["shared_tags"] = list(shared_tags)
                    
                # Calculate time proximity 
                time_diff = abs((source_memory.created_at - target_memory.created_at).total_seconds())
                edge_entry["time_proximity"] = 1.0 / (1.0 + time_diff / 86400)  # Normalized proximity
            
            result["edges"].append(edge_entry)
        
        # Add summary statistics for the graph
        if subgraph.number_of_nodes() > 0:
            # Calculate graph density
            density = nx.density(subgraph)
            
            # Calculate average clustering coefficient (local structure)
            avg_clustering = nx.average_clustering(subgraph)
            
            # Calculate average path length (if connected)
            avg_path = None
            if nx.is_connected(subgraph):
                avg_path = nx.average_shortest_path_length(subgraph)
            
            result["metadata"]["graph_metrics"] = {
                "density": density,
                "avg_clustering": avg_clustering,
                "avg_path_length": avg_path,
                "is_connected": nx.is_connected(subgraph)
            }
        
        return result
        
    def _importance_to_value(self, importance: Union[MemoryImportance, str]) -> float:
        """Convert importance enum or string to float value."""
        if isinstance(importance, MemoryImportance):
            if importance == MemoryImportance.LOW:
                return 1.0
            elif importance == MemoryImportance.MEDIUM:
                return 2.0
            elif importance == MemoryImportance.HIGH:
                return 3.0
            elif importance == MemoryImportance.CRITICAL:
                return 4.0
            else:
                return 2.0
        elif isinstance(importance, str):
            if importance.lower() == "low":
                return 1.0
            elif importance.lower() == "medium":
                return 2.0
            elif importance.lower() == "high":
                return 3.0
            elif importance.lower() == "critical":
                return 4.0
            else:
                return 2.0
        else:
            return 2.0  # Default to medium
    
    def add_context_tag(self, memory_id: int, tag: str) -> bool:
        """
        Add a context tag to a memory.
        
        Args:
            memory_id: Memory ID
            tag: Context tag
            
        Returns:
            True if successful, False otherwise
        """
        if memory_id not in self.strengths:
            return False
        
        self.context_tags[tag].add(memory_id)
        logger.debug(f"Added context tag '{tag}' to memory {memory_id}")
        return True
    
    def get_working_memory_contents(self) -> List[Dict[str, Any]]:
        """
        Get the current contents of working memory.
        
        Returns:
            List of memories in working memory
        """
        contents = []
        for memory_id in self.working_memory:
            memory = self.base_manager.retrieve_memory(memory_id)
            if memory:
                contents.append({
                    "id": memory_id,
                    "content": memory.content,
                    "type": memory.memory_type.value,
                    "activation": self.activations.get(memory_id, 0.0),
                    "strength": self.strengths.get(memory_id, 0.0)
                })
        
        return contents
    
    def update_developmental_parameters(self, developmental_stage: str) -> None:
        """
        Update memory parameters based on developmental stage.
        
        Args:
            developmental_stage: Current developmental stage
        """
        # Map developmental stages to memory parameters
        stage_params = {
            DevelopmentalStage.PRENATAL.value: {
                "working_memory_capacity": 2,
                "consolidation_efficiency": 0.2,
                "retrieval_efficiency": 0.2,
                "forgetting_rate": 0.2
            },
            DevelopmentalStage.INFANCY.value: {
                "working_memory_capacity": 3,
                "consolidation_efficiency": 0.3,
                "retrieval_efficiency": 0.3,
                "forgetting_rate": 0.15
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "working_memory_capacity": 4,
                "consolidation_efficiency": 0.4,
                "retrieval_efficiency": 0.4,
                "forgetting_rate": 0.12
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "working_memory_capacity": 5,
                "consolidation_efficiency": 0.6,
                "retrieval_efficiency": 0.6,
                "forgetting_rate": 0.1
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "working_memory_capacity": 6,
                "consolidation_efficiency": 0.7,
                "retrieval_efficiency": 0.8,
                "forgetting_rate": 0.08
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "working_memory_capacity": 7,
                "consolidation_efficiency": 0.9,
                "retrieval_efficiency": 0.9,
                "forgetting_rate": 0.05
            }
        }
        
        # Get parameters for current stage
        params = stage_params.get(developmental_stage, stage_params[DevelopmentalStage.PRENATAL.value])
        
        # Update parameters
        self.working_memory_capacity = params["working_memory_capacity"]
        self.consolidation_efficiency = params["consolidation_efficiency"]
        self.retrieval_efficiency = params["retrieval_efficiency"]
        self.forgetting_rate = params["forgetting_rate"]
        
        # Update working memory deque size
        new_working_memory = deque(self.working_memory, maxlen=self.working_memory_capacity)
        self.working_memory = new_working_memory
        
        logger.info(f"Updated memory parameters for stage {developmental_stage}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        # Get base statistics
        base_stats = self.base_manager.get_memory_stats()
        
        # Calculate additional stats
        strength_categories = {
            "weak": 0,
            "moderate": 0,
            "strong": 0,
            "permanent": 0
        }
        
        for strength in self.strengths.values():
            if strength < 0.3:
                strength_categories["weak"] += 1
            elif strength < 0.6:
                strength_categories["moderate"] += 1
            elif strength < 0.9:
                strength_categories["strong"] += 1
            else:
                strength_categories["permanent"] += 1
        
        activation_categories = {
            "inactive": 0,
            "low": 0,
            "medium": 0,
            "high": 0
        }
        
        for activation in self.activations.values():
            if activation < 0.2:
                activation_categories["inactive"] += 1
            elif activation < 0.4:
                activation_categories["low"] += 1
            elif activation < 0.7:
                activation_categories["medium"] += 1
            else:
                activation_categories["high"] += 1
        
        # Combine stats
        stats = {
            **base_stats,
            "strength_distribution": strength_categories,
            "activation_distribution": activation_categories,
            "association_count": self.memory_graph.number_of_edges(),
            "working_memory_capacity": self.working_memory_capacity,
            "working_memory_usage": len(self.working_memory),
            "context_tag_count": len(self.context_tags),
            "consolidation_efficiency": self.consolidation_efficiency,
            "retrieval_efficiency": self.retrieval_efficiency,
            "forgetting_rate": self.forgetting_rate
        }
        
        return stats
    
    def _update_working_memory(self, memory_id: int) -> None:
        """
        Update working memory with a newly activated memory.
        
        Args:
            memory_id: Memory ID to add to working memory
        """
        # Remove if already in working memory (to move to most recent position)
        if memory_id in self.working_memory:
            self.working_memory.remove(memory_id)
        
        # Add to working memory (will automatically remove oldest if at capacity)
        self.working_memory.append(memory_id)
    
    def _apply_forgetting(self, memory_id: int) -> None:
        """
        Apply forgetting curve to a memory.
        
        Args:
            memory_id: Memory ID
        """
        # Get current values
        current_strength = self.strengths.get(memory_id, MemoryStrength.WEAK)
        current_activation = self.activations.get(memory_id, MemoryActivation.LOW)
        
        # Calculate decay based on current strength (stronger memories decay slower)
        strength_factor = current_strength / float(MemoryStrength.PERMANENT)
        decay_resistance = 0.5 + (0.5 * strength_factor)  # 0.5 to 1.0
        
        # Calculate decay amount
        decay_amount = self.forgetting_rate * (1.0 - decay_resistance)
        
        # Apply decay to activation (faster decay)
        new_activation = max(0.0, current_activation - (decay_amount * 2))
        self.activations[memory_id] = new_activation
        
        # Apply decay to strength (slower decay)
        if current_strength < MemoryStrength.PERMANENT:  # Permanent memories don't decay
            new_strength = max(0.0, current_strength - decay_amount)
            self.strengths[memory_id] = new_strength
    
    def _optimize_memory_network(self) -> None:
        """Optimize the memory network by connecting co-activated memories."""
        # Get recently activated memories (in working memory)
        active_memories = list(self.working_memory)
        
        if len(active_memories) < 2:
            return  # Need at least 2 memories to form connections
        
        # Connect all pairs of active memories
        for i in range(len(active_memories)):
            for j in range(i+1, len(active_memories)):
                memory_id1 = active_memories[i]
                memory_id2 = active_memories[j]
                
                # Create or strengthen connection
                self.associate_memories(memory_id1, memory_id2, 0.3)
    
    def _apply_reconstruction_effects(self, memory: Memory) -> Memory:
        """
        Apply memory reconstruction effects to simulate human memory inaccuracies.
        
        Args:
            memory: Memory to modify
            
        Returns:
            Modified memory
        """
        # Memory reconstruction effects depend on memory strength
        memory_id = memory.vector_store_id
        if memory_id is None:
            return memory
            
        strength = self.strengths.get(memory_id, MemoryStrength.WEAK)
        
        # Strong memories don't change much
        if strength > 0.7:
            return memory
            
        # For weaker memories, add a note about potential reconstruction
        if strength < 0.4:
            # Add a note about memory reconstruction (don't actually modify content)
            memory.metadata["reconstructed"] = True
            memory.metadata["confidence"] = strength
        
        return memory 