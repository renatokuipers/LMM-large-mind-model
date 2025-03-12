# planning_models.py
"""Pydantic models for planning and simulation data."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class SearchNodeType(str, Enum):
    """Types of nodes in search algorithms."""
    TASK = "task"
    SEQUENCE = "sequence"
    DECISION = "decision"
    SCENARIO = "scenario"

class PlanningPhase(str, Enum):
    """Phases of the planning process."""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration" 
    EXPLOITATION = "exploitation"
    FINALIZATION = "finalization"

class SimulationResult(str, Enum):
    """Possible outcomes of a simulation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNKNOWN = "unknown"

class SearchNode(BaseModel):
    """Base model for search algorithm nodes."""
    id: UUID = Field(default_factory=uuid4)
    node_type: SearchNodeType
    parent_id: Optional[UUID] = None
    children_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Search algorithm metrics
    visits: int = 0
    depth: int = 0
    
    def mark_visited(self) -> None:
        """Increment the visit count for this node."""
        self.visits += 1

class MCTSNode(SearchNode):
    """Monte Carlo Tree Search node with UCB1 exploration."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    # Node values and scores
    value: float = 0.0
    exploration_score: float = 0.0
    exploitation_score: float = 0.0
    
    # Task-specific information
    task_id: Optional[UUID] = None
    task_sequence: List[UUID] = Field(default_factory=list)
    completed_tasks: Set[UUID] = Field(default_factory=set)
    
    @property
    def total_simulations(self) -> int:
        """Total number of simulations run from this node."""
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        """Win rate for this node."""
        return self.wins / self.total_simulations if self.total_simulations > 0 else 0.0
    
    def update_scores(self, exploration_weight: float = 1.0) -> None:
        """Update exploitation and exploration scores."""
        # Exploitation score is win rate
        self.exploitation_score = self.win_rate
        
        # UCB1 exploration formula
        parent_visits = 1  # Avoid division by zero
        if hasattr(self, 'parent') and self.parent is not None:
            parent_visits = max(1, self.parent.visits)
            
        if self.visits == 0:
            self.exploration_score = float('inf')
        else:
            import math
            self.exploration_score = exploration_weight * math.sqrt(
                math.log(parent_visits) / self.visits
            )
            
        # Combined score for node selection
        self.value = self.exploitation_score + self.exploration_score
        
    def add_simulation_result(self, result: SimulationResult) -> None:
        """Add a simulation result to this node."""
        if result == SimulationResult.SUCCESS:
            self.wins += 1
        elif result == SimulationResult.FAILURE:
            self.losses += 1
        else:
            self.draws += 1
        
        self.update_scores()

class AStarNode(SearchNode):
    """A* search algorithm node."""
    g_score: float = Field(0.0, description="Cost from start to current node")
    h_score: float = Field(0.0, description="Heuristic estimate to goal")
    f_score: float = Field(0.0, description="Total estimated cost (g + h)")
    
    # Task-specific information
    task_id: Optional[UUID] = None
    task_sequence: List[UUID] = Field(default_factory=list)
    completed_tasks: Set[UUID] = Field(default_factory=set)
    remaining_tasks: Set[UUID] = Field(default_factory=set)
    
    def update_scores(self, new_g_score: float, new_h_score: float) -> None:
        """Update the node's f, g, and h scores."""
        self.g_score = new_g_score
        self.h_score = new_h_score
        self.f_score = new_g_score + new_h_score

class SimulationConfig(BaseModel):
    """Configuration for a planning simulation."""
    max_iterations: int = 1000
    exploration_weight: float = 1.414  # UCT constant (sqrt(2))
    max_depth: int = 20
    time_limit_seconds: float = 30.0
    task_priorities: Dict[UUID, float] = Field(default_factory=dict)
    risk_weights: Dict[UUID, float] = Field(default_factory=dict)
    
    # A* parameters
    heuristic_weight: float = 1.0
    
    # Planning phases
    phase: PlanningPhase = PlanningPhase.INITIALIZATION
    phase_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "exploration_to_exploitation": 0.5,
            "exploitation_to_finalization": 0.9
        }
    )

class SimulationStep(BaseModel):
    """Represents a single step in a simulation."""
    id: UUID = Field(default_factory=uuid4)
    simulation_id: UUID
    step_number: int
    node_id: UUID
    action_taken: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    reward: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class SimulationSession(BaseModel):
    """Records a complete simulation session."""
    id: UUID = Field(default_factory=uuid4)
    config: SimulationConfig
    root_node_id: UUID
    iteration_count: int = 0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    success_rate: float = 0.0
    
    steps: List[SimulationStep] = Field(default_factory=list)
    
    def record_step(self, node_id: UUID, action: Dict[str, Any], 
                   state_before: Dict[str, Any], state_after: Dict[str, Any],
                   reward: float = 0.0) -> UUID:
        """Record a step in the simulation."""
        step = SimulationStep(
            simulation_id=self.id,
            step_number=len(self.steps) + 1,
            node_id=node_id,
            action_taken=action,
            state_before=state_before,
            state_after=state_after,
            reward=reward
        )
        self.steps.append(step)
        return step.id
    
    def complete_simulation(self, success_rate: float) -> None:
        """Mark the simulation as complete and record results."""
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success_rate = success_rate
        self.iteration_count = len(self.steps)

class PlanSnapshot(BaseModel):
    """A snapshot of the current development plan."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    plan_version: int
    task_sequence: List[UUID]
    expected_duration_hours: float
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    risk_assessment: Dict[str, float]
    simulation_id: Optional[UUID] = None
    
    generated_by: str = "mcts"  # Algorithm used to generate this plan
    description: str = ""
    
    @model_validator(mode='after')
    def validate_plan(self) -> 'PlanSnapshot':
        """Validate the plan structure."""
        if not self.task_sequence:
            raise ValueError("Plan must include at least one task")
        return self

class PlanningHistory(BaseModel):
    """Tracks the evolution of development plans over time."""
    snapshots: List[PlanSnapshot] = Field(default_factory=list)
    current_plan_id: Optional[UUID] = None
    
    def add_snapshot(self, snapshot: PlanSnapshot) -> UUID:
        """Add a plan snapshot and set it as current if it's the first one."""
        self.snapshots.append(snapshot)
        if self.current_plan_id is None:
            self.current_plan_id = snapshot.id
        return snapshot.id
    
    def set_current_plan(self, plan_id: UUID) -> bool:
        """Set the current active plan."""
        for snapshot in self.snapshots:
            if snapshot.id == plan_id:
                self.current_plan_id = plan_id
                return True
        return False
    
    def get_current_plan(self) -> Optional[PlanSnapshot]:
        """Get the current active plan."""
        if self.current_plan_id is None:
            return None
            
        for snapshot in self.snapshots:
            if snapshot.id == self.current_plan_id:
                return snapshot
                
        return None
    
    def get_latest_plan(self) -> Optional[PlanSnapshot]:
        """Get the most recently added plan snapshot."""
        if not self.snapshots:
            return None
        
        # Return the most recently created snapshot
        return sorted(self.snapshots, key=lambda x: x.timestamp, reverse=True)[0]
    
    def get_plan_evolution(self) -> List[Tuple[datetime, float, float]]:
        """Get the evolution of plan confidence and risk over time."""
        return [
            (s.timestamp, s.confidence_score, sum(s.risk_assessment.values()) / len(s.risk_assessment) 
             if s.risk_assessment else 0.0)
            for s in sorted(self.snapshots, key=lambda x: x.timestamp)
        ]