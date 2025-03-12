# search_algorithms.py
"""Implementation of MCTS and A* for development planning."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple, Callable
import math
import random
import time
import logging
from uuid import UUID, uuid4
from datetime import datetime
from heapq import heappush, heappop
import numpy as np

from .models.planning_models import (
    SearchNode, MCTSNode, AStarNode, SimulationConfig, 
    SimulationSession, PlanningPhase, SimulationResult
)
from .models.task_models import Task, TaskStatus, TaskGraph
from .llm_integration import LLMIntegration, LLMConfig
from .utils.fs_utils import safe_save_json, resolve_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTSPlanner:
    """Monte Carlo Tree Search implementation for development planning."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        llm_integration: Optional[LLMIntegration] = None,
        config: Optional[SimulationConfig] = None,
        session_id: Optional[UUID] = None
    ):
        """
        Initialize the MCTS planner.
        
        Args:
            task_graph: The graph of tasks and dependencies
            llm_integration: Optional LLM integration for evaluations
            config: Configuration for the simulation
            session_id: Optional ID for the simulation session
        """
        self.task_graph = task_graph
        self.llm_integration = llm_integration
        self.config = config or SimulationConfig()
        
        # Initialize nodes dictionary
        self.nodes: Dict[UUID, MCTSNode] = {}
        
        # Create root node
        self.root_node = self._create_root_node()
        
        # Initialize simulation session
        self.session = SimulationSession(
            id=session_id or uuid4(),
            config=self.config,
            root_node_id=self.root_node.id
        )
        
        # Track statistics
        self.total_simulations = 0
        self.successful_simulations = 0
        
        # Save session directory
        self.session_dir = resolve_path(f"planning/search_trees/{self.session.id}", create_parents=True)
    
    def _create_root_node(self) -> MCTSNode:
        """Create the root node for the search tree."""
        # Get all planned tasks
        planned_tasks = {
            task_id: task for task_id, task in self.task_graph.tasks.items()
            if task.status == TaskStatus.PLANNED
        }
        
        # Create root node with no completed tasks
        root = MCTSNode(
            node_type="task",
            task_sequence=[],
            completed_tasks=set(),
            depth=0
        )
        
        self.nodes[root.id] = root
        return root
    
    def run_simulation(self, iterations: Optional[int] = None, time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Run MCTS simulation for a specified number of iterations or time limit.
        
        Args:
            iterations: Maximum number of iterations to run
            time_limit: Maximum time to run in seconds
            
        Returns:
            Simulation results
        """
        max_iterations = iterations or self.config.max_iterations
        max_time = time_limit or self.config.time_limit_seconds
        
        start_time = time.time()
        iteration = 0
        
        while (
            iteration < max_iterations and 
            (time.time() - start_time) < max_time
        ):
            # Run one iteration of MCTS
            result = self._run_iteration()
            iteration += 1
            
            # Update statistics
            self.total_simulations += 1
            if result == SimulationResult.SUCCESS:
                self.successful_simulations += 1
            
            # Update session
            self.session.iteration_count = iteration
            
            # Check if we should move to the next planning phase
            self._update_planning_phase(iteration, max_iterations)
            
            # Periodically save progress
            if iteration % 10 == 0:
                self._save_progress()
        
        # Complete the simulation
        elapsed_time = time.time() - start_time
        success_rate = self.successful_simulations / max(1, self.total_simulations)
        
        self.session.complete_simulation(success_rate)
        self._save_progress()
        
        # Return results
        return {
            "iterations": iteration,
            "elapsed_time": elapsed_time,
            "success_rate": success_rate,
            "best_sequence": self._get_best_sequence(),
            "session_id": str(self.session.id)
        }
    
    def _run_iteration(self) -> SimulationResult:
        """
        Run a single iteration of the MCTS algorithm.
        
        Returns:
            Result of the simulation
        """
        # 1. Selection: Select a promising node
        node = self._select_node(self.root_node)
        
        # 2. Expansion: Expand the selected node
        if node.visits > 0 and len(self._get_available_tasks(node)) > 0:
            node = self._expand_node(node)
        
        # 3. Simulation: Perform a rollout from the node
        result = self._rollout(node)
        
        # 4. Backpropagation: Update the path with the result
        self._backpropagate(node, result)
        
        return result
    
    def _select_node(self, node: MCTSNode) -> MCTSNode:
        """
        Select a promising node for expansion using UCB1.
        
        Args:
            node: Current node
            
        Returns:
            Selected node
        """
        # If node is a leaf node or hasn't been visited, return it
        if not node.children_ids or node.visits == 0:
            return node
        
        # Choose exploration/exploitation balance based on planning phase
        exploration_weight = self.config.exploration_weight
        if self.config.phase == PlanningPhase.EXPLOITATION:
            exploration_weight *= 0.5
        elif self.config.phase == PlanningPhase.FINALIZATION:
            exploration_weight *= 0.2
        
        best_score = float('-inf')
        best_child = None
        
        for child_id in node.children_ids:
            if child_id not in self.nodes:
                continue
                
            child = self.nodes[child_id]
            
            # Update scores with current exploration weight
            child.update_scores(exploration_weight)
            
            if child.value > best_score:
                best_score = child.value
                best_child = child
        
        # If no valid child found, return current node
        if best_child is None:
            return node
        
        # Recursive selection
        return self._select_node(best_child)
    
    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """
        Expand a node by adding a child.
        
        Args:
            node: Node to expand
            
        Returns:
            Newly created child node
        """
        # Get available tasks
        available_tasks = self._get_available_tasks(node)
        
        if not available_tasks:
            return node
        
        # Choose a random task
        task_id = random.choice(list(available_tasks))
        task = self.task_graph.tasks[task_id]
        
        # Create a new node
        new_task_sequence = node.task_sequence.copy() + [task_id]
        new_completed_tasks = node.completed_tasks.copy()
        new_completed_tasks.add(task_id)
        
        child = MCTSNode(
            node_type="task",
            parent_id=node.id,
            task_id=task_id,
            task_sequence=new_task_sequence,
            completed_tasks=new_completed_tasks,
            depth=node.depth + 1
        )
        
        # Add to nodes dictionary
        self.nodes[child.id] = child
        
        # Add child to parent
        if child.id not in node.children_ids:
            node.children_ids.append(child.id)
        
        return child
    
    def _rollout(self, node: MCTSNode) -> SimulationResult:
        """
        Perform a random rollout from the node.
        
        Args:
            node: Starting node for rollout
            
        Returns:
            Result of the simulation
        """
        # Copy current state
        completed_tasks = node.completed_tasks.copy()
        depth = node.depth
        
        # Continue rollout until we complete all tasks or reach max depth
        while depth < self.config.max_depth:
            # Get available tasks
            available_tasks = {
                task_id: task for task_id, task in self.task_graph.tasks.items()
                if task_id not in completed_tasks and 
                all(dep_id in completed_tasks for dep_id in task.dependencies)
            }
            
            # If no more tasks available
            if not available_tasks:
                # Check if we completed all planned tasks
                all_planned_tasks = {
                    task_id for task_id, task in self.task_graph.tasks.items()
                    if task.status == TaskStatus.PLANNED
                }
                
                if all_planned_tasks.issubset(completed_tasks):
                    return SimulationResult.SUCCESS
                else:
                    return SimulationResult.PARTIAL_SUCCESS
            
            # Choose a random task
            task_id = random.choice(list(available_tasks.keys()))
            
            # Add to completed tasks
            completed_tasks.add(task_id)
            depth += 1
        
        # If we reached max depth without completing all tasks
        return SimulationResult.PARTIAL_SUCCESS
    
    def _backpropagate(self, node: MCTSNode, result: SimulationResult) -> None:
        """
        Backpropagate the result up the tree.
        
        Args:
            node: Starting node for backpropagation
            result: Result of the simulation
        """
        current = node
        while current is not None:
            # Increment visit count
            current.mark_visited()
            
            # Add result
            current.add_simulation_result(result)
            
            # Move to parent
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                current = None
    
    def _get_available_tasks(self, node: MCTSNode) -> Dict[UUID, Task]:
        """
        Get tasks that can be executed next given the current state.
        
        Args:
            node: Current node
            
        Returns:
            Dictionary of available tasks
        """
        return {
            task_id: task for task_id, task in self.task_graph.tasks.items()
            if task_id not in node.completed_tasks and
            task.status == TaskStatus.PLANNED and
            all(dep_id in node.completed_tasks for dep_id in task.dependencies)
        }
    
    def _update_planning_phase(self, iteration: int, max_iterations: int) -> None:
        """Update the planning phase based on progress."""
        progress = iteration / max_iterations
        
        # Transition through phases based on progress
        if progress >= self.config.phase_thresholds["exploitation_to_finalization"]:
            self.config.phase = PlanningPhase.FINALIZATION
        elif progress >= self.config.phase_thresholds["exploration_to_exploitation"]:
            self.config.phase = PlanningPhase.EXPLOITATION
        else:
            self.config.phase = PlanningPhase.EXPLORATION
    
    def _get_best_sequence(self) -> List[UUID]:
        """
        Get the best task sequence found by MCTS.
        
        Returns:
            List of task IDs in the recommended sequence
        """
        # Start with the root
        current = self.root_node
        sequence = []
        
        # Follow the path with highest win rate
        while current.children_ids:
            best_win_rate = -1
            best_child = None
            
            for child_id in current.children_ids:
                if child_id not in self.nodes:
                    continue
                    
                child = self.nodes[child_id]
                
                if child.win_rate > best_win_rate:
                    best_win_rate = child.win_rate
                    best_child = child
            
            if best_child is None:
                break
                
            if best_child.task_id:
                sequence.append(best_child.task_id)
                
            current = best_child
        
        return sequence
    
    def _save_progress(self) -> None:
        """Save the current state of the simulation."""
        # Save session data
        session_path = self.session_dir / "session.json"
        safe_save_json(self.session.model_dump(), session_path)
        
        # Save top nodes (root + children)
        nodes_path = self.session_dir / "nodes.json"
        
        # Only save root and its immediate children to avoid huge files
        nodes_to_save = {
            str(self.root_node.id): self.root_node.model_dump()
        }
        
        for child_id in self.root_node.children_ids:
            if child_id in self.nodes:
                nodes_to_save[str(child_id)] = self.nodes[child_id].model_dump()
        
        safe_save_json(nodes_to_save, nodes_path)
        
        # Save best sequence
        sequence_path = self.session_dir / "best_sequence.json"
        sequence_data = {
            "task_ids": [str(task_id) for task_id in self._get_best_sequence()],
            "win_rate": self.successful_simulations / max(1, self.total_simulations),
            "total_simulations": self.total_simulations
        }
        safe_save_json(sequence_data, sequence_path)

class AStarPathfinder:
    """A* search implementation for finding optimal task sequences."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        heuristic_function: Optional[Callable[[UUID, Set[UUID]], float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the A* pathfinder.
        
        Args:
            task_graph: The graph of tasks and dependencies
            heuristic_function: Optional custom heuristic function
            config: Optional configuration
        """
        self.task_graph = task_graph
        self.heuristic_function = heuristic_function or self._default_heuristic
        self.config = config or {
            "heuristic_weight": 1.0,
            "max_iterations": 1000
        }
        
        # Nodes and results
        self.nodes: Dict[UUID, AStarNode] = {}
        self.came_from: Dict[UUID, UUID] = {}
        self.g_score: Dict[UUID, float] = {}
        self.f_score: Dict[UUID, float] = {}
    
    def find_path(
        self, 
        start_tasks: Optional[List[UUID]] = None,
        goal_condition: Optional[Callable[[Set[UUID]], bool]] = None
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Find the optimal path through the task graph.
        
        Args:
            start_tasks: Optional list of starting tasks
            goal_condition: Optional custom goal condition
            
        Returns:
            Tuple of (path, metadata)
        """
        # Initialize
        if start_tasks is None:
            # Default to tasks with no dependencies
            start_tasks = [
                task_id for task_id, task in self.task_graph.tasks.items()
                if not task.dependencies and task.status == TaskStatus.PLANNED
            ]
        
        if goal_condition is None:
            # Default goal is to complete all planned tasks
            planned_tasks = {
                task_id for task_id, task in self.task_graph.tasks.items()
                if task.status == TaskStatus.PLANNED
            }
            goal_condition = lambda completed: planned_tasks.issubset(completed)
        
        # Create start node
        start_node = AStarNode(
            node_type="sequence",
            task_sequence=[],
            completed_tasks=set(),
            remaining_tasks={
                task_id for task_id, task in self.task_graph.tasks.items()
                if task.status == TaskStatus.PLANNED
            }
        )
        self.nodes[start_node.id] = start_node
        
        # Open and closed sets
        open_set = [(0, start_node.id)]  # Priority queue (f_score, node_id)
        closed_set = set()
        
        # Initialize scores
        self.g_score = {start_node.id: 0}
        self.f_score = {start_node.id: self._calculate_heuristic(start_node.id)}
        
        # Search variables
        iterations = 0
        path_found = False
        goal_node_id = None
        
        # A* search
        while open_set and iterations < self.config["max_iterations"]:
            iterations += 1
            
            # Get node with lowest f_score
            current_f, current_id = heappop(open_set)
            
            if current_id in closed_set:
                continue
                
            current = self.nodes[current_id]
            
            # Check if we've reached the goal
            if goal_condition(current.completed_tasks):
                path_found = True
                goal_node_id = current_id
                break
            
            # Mark as processed
            closed_set.add(current_id)
            
            # Get available next tasks
            available_tasks = {
                task_id: task for task_id, task in self.task_graph.tasks.items()
                if task_id not in current.completed_tasks and
                task.status == TaskStatus.PLANNED and
                all(dep_id in current.completed_tasks for dep_id in task.dependencies)
            }
            
            # Try each available task
            for task_id, task in available_tasks.items():
                # Create new node
                new_task_sequence = current.task_sequence.copy() + [task_id]
                new_completed_tasks = current.completed_tasks.copy()
                new_completed_tasks.add(task_id)
                new_remaining_tasks = current.remaining_tasks.copy()
                new_remaining_tasks.remove(task_id)
                
                neighbor = AStarNode(
                    node_type="sequence",
                    parent_id=current_id,
                    task_id=task_id,
                    task_sequence=new_task_sequence,
                    completed_tasks=new_completed_tasks,
                    remaining_tasks=new_remaining_tasks,
                    depth=current.depth + 1
                )
                
                # Skip if we've already processed this exact state
                state_key = frozenset(new_completed_tasks)
                if any(n.completed_tasks == new_completed_tasks for n in self.nodes.values() if n.id in closed_set):
                    continue
                
                # Add to nodes dictionary
                self.nodes[neighbor.id] = neighbor
                
                # Calculate scores
                tentative_g_score = self.g_score[current_id] + task.estimated_duration_hours
                
                if neighbor.id not in self.g_score or tentative_g_score < self.g_score[neighbor.id]:
                    # This is a better path
                    self.came_from[neighbor.id] = current_id
                    self.g_score[neighbor.id] = tentative_g_score
                    
                    h_score = self._calculate_heuristic(neighbor.id)
                    self.f_score[neighbor.id] = tentative_g_score + h_score
                    
                    # Update node scores
                    neighbor.update_scores(tentative_g_score, h_score)
                    
                    # Add to open set
                    heappush(open_set, (self.f_score[neighbor.id], neighbor.id))
        
        # Reconstruct path if found
        path = []
        metadata = {
            "iterations": iterations,
            "nodes_explored": len(closed_set),
            "path_found": path_found
        }
        
        if path_found and goal_node_id:
            path = self._reconstruct_path(goal_node_id)
            metadata["path_length"] = len(path)
            metadata["estimated_duration"] = sum(
                self.task_graph.tasks[task_id].estimated_duration_hours
                for task_id in path
            )
        
        return path, metadata
    
    def _reconstruct_path(self, goal_id: UUID) -> List[UUID]:
        """
        Reconstruct the path from start to goal.
        
        Args:
            goal_id: ID of the goal node
            
        Returns:
            List of task IDs in the optimal sequence
        """
        path = []
        current_id = goal_id
        
        while current_id in self.nodes:
            current = self.nodes[current_id]
            if current.task_id:
                path.append(current.task_id)
            
            if current_id not in self.came_from:
                break
                
            current_id = self.came_from[current_id]
        
        return list(reversed(path))
    
    def _calculate_heuristic(self, node_id: UUID) -> float:
        """
        Calculate the heuristic value for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Heuristic value
        """
        node = self.nodes[node_id]
        
        if not node.remaining_tasks:
            return 0.0
            
        # Use the custom heuristic function if provided, otherwise use default
        return self.heuristic_function(node_id, node.remaining_tasks) * self.config["heuristic_weight"]
    
    def _default_heuristic(self, node_id: UUID, remaining_tasks: Set[UUID]) -> float:
        """
        Default heuristic function based on remaining task duration.
        
        Args:
            node_id: ID of the node
            remaining_tasks: Set of remaining task IDs
            
        Returns:
            Heuristic value
        """
        if not remaining_tasks:
            return 0.0
            
        # Sum up the estimated duration of all remaining tasks
        return sum(
            self.task_graph.tasks[task_id].estimated_duration_hours
            for task_id in remaining_tasks
            if task_id in self.task_graph.tasks
        )
    
    def critical_path_heuristic(self, node_id: UUID, remaining_tasks: Set[UUID]) -> float:
        """
        Heuristic based on the critical path through remaining tasks.
        
        Args:
            node_id: ID of the node
            remaining_tasks: Set of remaining task IDs
            
        Returns:
            Heuristic value
        """
        if not remaining_tasks:
            return 0.0
            
        # Create a subgraph of remaining tasks
        subgraph = TaskGraph()
        
        for task_id in remaining_tasks:
            if task_id in self.task_graph.tasks:
                subgraph.tasks[task_id] = self.task_graph.tasks[task_id]
        
        # Find dependencies within the subgraph
        for task_id, task in subgraph.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in subgraph.tasks:
                    # This dependency is within the subgraph
                    dep_task = self.task_graph.tasks[dep_id]
                    
                    # Create dependency object
                    dependency = {
                        "source_id": dep_id,
                        "target_id": task_id,
                        "dependency_type": "blocks",
                        "strength": 1.0
                    }
                    
                    subgraph.dependencies.append(dependency)
        
        # Get the critical path
        critical_path = subgraph.get_critical_path()
        
        # Sum up durations along the critical path
        return sum(
            self.task_graph.tasks[task_id].estimated_duration_hours
            for task_id in critical_path
            if task_id in self.task_graph.tasks
        )