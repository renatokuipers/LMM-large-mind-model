"""Pydantic models for planning and simulation data."""
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator

from .task_models import Task, TaskStatus


class PlanningAlgorithm(str, Enum):
    """Supported planning algorithms."""
    MCTS = "mcts"  # Monte Carlo Tree Search
    A_STAR = "a_star"  # A* search


class PlanningHeuristic(str, Enum):
    """Heuristic functions used in planning."""
    TIME_ESTIMATE = "time_estimate"  # Based on time estimates
    RISK_LEVEL = "risk_level"  # Based on risk levels
    SUCCESS_PROBABILITY = "success_probability"  # Based on success probabilities
    DEPENDENCY_COUNT = "dependency_count"  # Based on the number of dependencies
    MULTI_FACTOR = "multi_factor"  # Combination of multiple factors


class TreeNode(BaseModel):
    """Represents a node in the Monte Carlo search tree."""
    id: UUID = Field(default_factory=uuid4)
    task_id: Optional[UUID] = None
    parent_id: Optional[UUID] = None
    children_ids: List[UUID] = Field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    value: float = 0.0
    
    # Node state
    completed_tasks: List[UUID] = Field(default_factory=list)
    remaining_tasks: List[UUID] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    def ucb1_value(self, exploration_weight: float = 1.0, parent_visits: int = 1) -> float:
        """Calculate the UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * (parent_visits / self.visits) ** 0.5
        
        return exploitation + exploration


class SimulationResult(BaseModel):
    """Results of a Monte Carlo simulation."""
    simulation_id: UUID = Field(default_factory=uuid4)
    start_node_id: UUID
    end_node_id: UUID
    
    # Performance metrics
    completion_time: float  # Estimated time to complete all tasks
    success_probability: float  # Overall probability of success
    risk_score: float  # Cumulative risk score
    
    # Task sequence
    task_sequence: List[UUID]  # Order of task execution
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('success_probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate that the probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v


class SearchTree(BaseModel):
    """Represents the entire search tree for Monte Carlo Tree Search."""
    id: UUID = Field(default_factory=uuid4)
    root_id: UUID
    nodes: Dict[UUID, TreeNode] = Field(default_factory=dict)
    best_path: List[UUID] = Field(default_factory=list)
    
    # Search parameters
    exploration_weight: float = 1.0
    max_simulations: int = 1000
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_node(self, node: TreeNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        self.updated_at = datetime.now()
    
    def get_node(self, node_id: UUID) -> Optional[TreeNode]:
        """Get a node from the tree."""
        return self.nodes.get(node_id)
    
    def get_best_path(self) -> List[TreeNode]:
        """Get the nodes in the best path."""
        return [self.nodes[node_id] for node_id in self.best_path if node_id in self.nodes]


class PathfindingGraph(BaseModel):
    """Represents a graph for A* pathfinding."""
    id: UUID = Field(default_factory=uuid4)
    nodes: Dict[UUID, Any] = Field(default_factory=dict)  # Task IDs mapped to A* node data
    edges: Dict[UUID, List[UUID]] = Field(default_factory=dict)  # Task ID to list of dependent task IDs
    
    # Pathfinding parameters
    heuristic: PlanningHeuristic = PlanningHeuristic.MULTI_FACTOR
    weight_factors: Dict[str, float] = Field(default_factory=dict)
    
    # Results
    optimal_path: List[UUID] = Field(default_factory=list)
    path_cost: float = 0.0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_node(self, task_id: UUID, node_data: Any) -> None:
        """Add a node to the graph."""
        self.nodes[task_id] = node_data
        if task_id not in self.edges:
            self.edges[task_id] = []
        self.updated_at = datetime.now()
    
    def add_edge(self, from_task_id: UUID, to_task_id: UUID) -> None:
        """Add an edge between nodes."""
        if from_task_id not in self.edges:
            self.edges[from_task_id] = []
        if to_task_id not in self.edges[from_task_id]:
            self.edges[from_task_id].append(to_task_id)
        self.updated_at = datetime.now()
    
    def get_neighbors(self, task_id: UUID) -> List[UUID]:
        """Get all neighbors of a node."""
        return self.edges.get(task_id, [])


class PlanningSession(BaseModel):
    """Represents a planning session that tracks planning results."""
    id: UUID = Field(default_factory=uuid4)
    algorithm: PlanningAlgorithm
    
    # Planning artifacts
    search_tree_id: Optional[UUID] = None  # For MCTS
    pathfinding_graph_id: Optional[UUID] = None  # For A*
    
    # Results
    recommended_task_sequence: List[UUID] = Field(default_factory=list)
    estimated_completion_time: float = 0.0
    estimated_success_probability: float = 0.5
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('estimated_success_probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate that the probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v