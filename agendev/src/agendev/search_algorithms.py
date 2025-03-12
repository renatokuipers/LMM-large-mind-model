"""Implementation of MCTS and A* for development planning."""
import math
import random
import heapq
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import logging
from collections import defaultdict

import numpy as np

from .models.task_models import Task, TaskStatus, RiskLevel, TaskGraph
from .models.planning_models import (
    TreeNode, SearchTree, SimulationResult, PathfindingGraph,
    PlanningHeuristic, PlanningAlgorithm, PlanningSession
)
from .utils.fs_utils import (
    get_search_trees_directory, get_pathfinding_directory,
    save_json, load_json
)
from .probability_modeling import (
    calculate_task_success_probability,
    calculate_sequence_success_probability
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_algorithms")


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search for development planning."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        exploration_weight: float = 1.0,
        max_simulations: int = 1000,
        max_simulation_depth: int = 50,
        simulation_count_per_iteration: int = 10,
        save_interval: int = 100
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            task_graph: Graph of tasks and dependencies
            exploration_weight: Exploration vs exploitation weight (UCB1)
            max_simulations: Maximum number of simulations to run
            max_simulation_depth: Maximum depth of each simulation
            simulation_count_per_iteration: Number of simulations per iteration
            save_interval: How often to save the search tree
        """
        self.task_graph = task_graph
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_simulation_depth = max_simulation_depth
        self.simulation_count_per_iteration = simulation_count_per_iteration
        self.save_interval = save_interval
        
        # Create a new search tree
        self.tree = self._create_search_tree()
        
        # Track simulation count
        self.simulation_count = 0
    
    def _create_search_tree(self) -> SearchTree:
        """Create a new search tree with root node."""
        # Create root node
        root_node = TreeNode(
            task_id=None,  # Root has no task
            parent_id=None,
            children_ids=[],
            visits=0,
            value=0.0,
            completed_tasks=[],
            remaining_tasks=[task_id for task_id in self.task_graph.tasks.keys()]
        )
        
        # Create search tree
        tree = SearchTree(
            root_id=root_node.id,
            nodes={root_node.id: root_node},
            best_path=[],
            exploration_weight=self.exploration_weight,
            max_simulations=self.max_simulations
        )
        
        return tree
    
    def run(self, notification_callback=None) -> SearchTree:
        """
        Run the MCTS algorithm.
        
        Args:
            notification_callback: Optional callback for progress notifications
            
        Returns:
            The final search tree
        """
        while self.simulation_count < self.max_simulations:
            # Run a batch of simulations
            for _ in range(min(self.simulation_count_per_iteration, 
                              self.max_simulations - self.simulation_count)):
                # Select node to expand
                node_id = self._select(self.tree.root_id)
                
                # Expand the node
                child_node_id = self._expand(node_id)
                
                # If no child node was created, simulate from the current node
                simulate_node_id = child_node_id if child_node_id is not None else node_id
                
                # Simulate from the selected node
                simulation_result = self._simulate(simulate_node_id)
                
                # Backpropagate the results
                self._backpropagate(simulate_node_id, simulation_result.success_probability)
                
                # Increment simulation count
                self.simulation_count += 1
            
            # Update the best path
            self._update_best_path()
            
            # Save the tree periodically
            if self.simulation_count % self.save_interval == 0:
                self._save_tree()
                
                # Notify progress if callback provided
                if notification_callback:
                    progress_percentage = min(100, int(self.simulation_count / self.max_simulations * 100))
                    notification_callback(
                        f"MCTS progress: {progress_percentage}% complete. "
                        f"({self.simulation_count}/{self.max_simulations} simulations)"
                    )
        
        # Final save of the tree
        self._save_tree()
        
        # Final update of the best path
        self._update_best_path()
        
        return self.tree
    
    def _select(self, node_id: UUID) -> UUID:
        """
        Select a node to expand using UCB1.
        
        Args:
            node_id: Starting node ID
            
        Returns:
            Selected node ID
        """
        while True:
            node = self.tree.nodes[node_id]
            
            # If node is not fully expanded, return it
            if not node.children_ids or len(node.remaining_tasks) == 0:
                return node_id
            
            # Check if there are any unvisited children
            unvisited = [
                child_id for child_id in node.children_ids
                if self.tree.nodes[child_id].visits == 0
            ]
            
            if unvisited:
                # Return a random unvisited child
                return random.choice(unvisited)
            
            # Otherwise, select the child with the highest UCB1 value
            parent_visits = node.visits
            best_score = -float('inf')
            best_child_id = None
            
            for child_id in node.children_ids:
                child = self.tree.nodes[child_id]
                ucb1_score = child.ucb1_value(
                    exploration_weight=self.exploration_weight,
                    parent_visits=parent_visits
                )
                
                if ucb1_score > best_score:
                    best_score = ucb1_score
                    best_child_id = child_id
            
            if best_child_id is None:
                # No valid children, return current node
                return node_id
            
            node_id = best_child_id
    
    def _expand(self, node_id: UUID) -> Optional[UUID]:
        """
        Expand a node by adding a child.
        
        Args:
            node_id: Node to expand
            
        Returns:
            ID of the new child node, or None if no expansion possible
        """
        node = self.tree.nodes[node_id]
        
        # If there are no remaining tasks, can't expand
        if not node.remaining_tasks:
            return None
        
        # Find tasks that can be executed next (not blocked by dependencies)
        available_tasks = []
        for task_id in node.remaining_tasks:
            task = self.task_graph.tasks[task_id]
            
            # Check if task dependencies are met
            dependencies_met = True
            for dep in task.dependencies:
                if dep.task_id not in node.completed_tasks:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                available_tasks.append(task_id)
        
        # If no available tasks, can't expand
        if not available_tasks:
            return None
        
        # Randomly select a task for this expansion
        task_id = random.choice(available_tasks)
        
        # Create a new node for this task
        new_remaining = [t_id for t_id in node.remaining_tasks if t_id != task_id]
        new_completed = node.completed_tasks + [task_id]
        
        child_node = TreeNode(
            task_id=task_id,
            parent_id=node_id,
            children_ids=[],
            visits=0,
            value=0.0,
            completed_tasks=new_completed,
            remaining_tasks=new_remaining
        )
        
        # Add the child to the tree
        self.tree.add_node(child_node)
        
        # Update the parent's children list
        node.children_ids.append(child_node.id)
        self.tree.nodes[node_id] = node
        
        return child_node.id
    
    def _simulate(self, node_id: UUID) -> SimulationResult:
        """
        Run a simulation from a node to estimate its value.
        
        Args:
            node_id: Starting node ID
            
        Returns:
            Simulation result
        """
        node = self.tree.nodes[node_id]
        
        # Start with the current state
        completed_tasks = node.completed_tasks.copy()
        remaining_tasks = node.remaining_tasks.copy()
        task_sequence = completed_tasks.copy()  # Tasks completed so far
        
        # Estimate base completion time from already completed tasks
        completion_time = sum(
            self.task_graph.tasks[t_id].estimated_hours
            for t_id in completed_tasks
            if t_id in self.task_graph.tasks
        )
        
        # Simulate until all tasks are completed or max depth reached
        depth = 0
        while remaining_tasks and depth < self.max_simulation_depth:
            # Find tasks that can be executed next
            available_tasks = []
            for task_id in remaining_tasks:
                task = self.task_graph.tasks[task_id]
                
                # Check if task dependencies are met
                dependencies_met = True
                for dep in task.dependencies:
                    if dep.task_id not in completed_tasks:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    available_tasks.append(task_id)
            
            # If no available tasks, break
            if not available_tasks:
                break
            
            # Randomly select a task
            task_id = random.choice(available_tasks)
            task = self.task_graph.tasks[task_id]
            
            # Add task to sequence and update state
            task_sequence.append(task_id)
            completed_tasks.append(task_id)
            remaining_tasks.remove(task_id)
            
            # Update completion time
            completion_time += task.estimated_hours
            
            depth += 1
        
        # Calculate success probability for this sequence
        success_probability = calculate_sequence_success_probability(
            self.task_graph, task_sequence
        )
        
        # Calculate risk score based on task risk levels
        risk_score = sum(
            self._risk_level_to_score(self.task_graph.tasks[t_id].risk_level)
            for t_id in task_sequence
            if t_id in self.task_graph.tasks
        ) / max(1, len(task_sequence))
        
        # Create simulation result
        result = SimulationResult(
            simulation_id=uuid4(),
            start_node_id=node_id,
            end_node_id=node_id,  # Same as start for now
            completion_time=completion_time,
            success_probability=success_probability,
            risk_score=risk_score,
            task_sequence=task_sequence
        )
        
        return result
    
    def _backpropagate(self, node_id: UUID, value: float) -> None:
        """
        Backpropagate simulation results up the tree.
        
        Args:
            node_id: Node ID to start backpropagation from
            value: Value to backpropagate
        """
        while node_id is not None:
            node = self.tree.nodes[node_id]
            
            # Update node statistics
            node.visits += 1
            node.value += value
            
            # Update in tree
            self.tree.nodes[node_id] = node
            
            # Move to parent
            node_id = node.parent_id
    
    def _update_best_path(self) -> None:
        """Update the best path in the search tree."""
        # Start from the root
        path = []
        node_id = self.tree.root_id
        
        # Find the path with highest value
        while node_id is not None:
            node = self.tree.nodes[node_id]
            path.append(node_id)
            
            # If no children, we're done
            if not node.children_ids:
                break
            
            # Find the child with the highest average value
            best_value = -float('inf')
            best_child_id = None
            
            for child_id in node.children_ids:
                child = self.tree.nodes[child_id]
                if child.visits == 0:
                    continue
                
                avg_value = child.value / child.visits
                
                if avg_value > best_value:
                    best_value = avg_value
                    best_child_id = child_id
            
            # If no valid children, we're done
            if best_child_id is None:
                break
            
            node_id = best_child_id
        
        # Update the best path in the tree
        self.tree.best_path = path
    
    def _save_tree(self) -> None:
        """Save the search tree to disk."""
        tree_dir = get_search_trees_directory()
        tree_path = tree_dir / f"mcts_tree_{self.tree.id}.json"
        
        save_json(self.tree.model_dump(), tree_path)
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """
        Convert a risk level to a numeric score.
        
        Args:
            risk_level: Risk level to convert
            
        Returns:
            Numeric risk score
        """
        risk_scores = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.VERY_HIGH: 1.0
        }
        
        return risk_scores.get(risk_level, 0.5)
    
    def get_recommended_sequence(self) -> List[UUID]:
        """
        Get the recommended task sequence based on the search.
        
        Returns:
            List of task IDs in recommended order
        """
        # Use the best path to determine the task sequence
        task_sequence = []
        
        for node_id in self.tree.best_path:
            node = self.tree.nodes[node_id]
            if node.task_id is not None:  # Skip the root node which has no task
                task_sequence.append(node.task_id)
        
        return task_sequence


class AStarPathfinding:
    """A* pathfinding for development planning."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        heuristic: PlanningHeuristic = PlanningHeuristic.MULTI_FACTOR,
        weight_factors: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the A* pathfinding algorithm.
        
        Args:
            task_graph: Graph of tasks and dependencies
            heuristic: Heuristic function to use
            weight_factors: Weight factors for the multi-factor heuristic
        """
        self.task_graph = task_graph
        self.heuristic = heuristic
        
        # Default weight factors
        self.weight_factors = weight_factors or {
            "time_estimate": 0.4,
            "risk_level": 0.2,
            "success_probability": 0.3,
            "dependency_count": 0.1
        }
        
        # Create a new pathfinding graph
        self.graph = PathfindingGraph(
            heuristic=heuristic,
            weight_factors=self.weight_factors
        )
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the pathfinding graph from the task graph."""
        # Add nodes for all tasks
        for task_id, task in self.task_graph.tasks.items():
            # Create node data with task properties relevant for A*
            node_data = {
                "estimated_hours": task.estimated_hours,
                "risk_level": task.risk_level.value,
                "success_probability": task.success_probability,
                "priority": task.priority.value
            }
            
            self.graph.add_node(task_id, node_data)
        
        # Add edges based on dependencies
        for task_id, task in self.task_graph.tasks.items():
            # For each dependency
            for dep in task.dependencies:
                # Add an edge from dependency to task
                # (task depends on dep.task_id, so dep.task_id must be done before task)
                self.graph.add_edge(dep.task_id, task_id)
    
    def find_path(
        self,
        start_tasks: List[UUID] = None,
        goal_condition: Callable[[Set[UUID]], bool] = None
    ) -> List[UUID]:
        """
        Find an optimal path through the task graph.
        
        Args:
            start_tasks: Tasks to start from (defaults to tasks with no dependencies)
            goal_condition: Function to check if the goal is reached
                (defaults to all tasks completed)
            
        Returns:
            List of task IDs in optimal order
        """
        # Default start tasks: tasks with no dependencies
        if start_tasks is None:
            start_tasks = []
            for task_id, task in self.task_graph.tasks.items():
                if not task.dependencies:
                    start_tasks.append(task_id)
        
        # Default goal condition: all tasks completed
        if goal_condition is None:
            all_tasks = set(self.task_graph.tasks.keys())
            goal_condition = lambda completed: completed == all_tasks
        
        # Initialize A* algorithm
        start_state = frozenset()  # No tasks completed initially
        open_set = [(0, 0, start_state, [])]  # (f_score, g_score, state, path)
        closed_set = set()
        g_scores = {start_state: 0}  # Cost from start to state
        f_scores = {start_state: self._heuristic(start_state)}  # Estimated total cost
        
        # While there are nodes to explore
        while open_set:
            # Get the node with the lowest f_score
            _, g_score, current_state, path = heapq.heappop(open_set)
            
            # If goal reached, return the path
            if goal_condition(current_state):
                self.graph.optimal_path = path
                self.graph.path_cost = g_score
                return path
            
            # Skip if already processed
            if current_state in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current_state)
            
            # Get available next tasks
            available_tasks = self._get_available_tasks(current_state)
            
            # For each available task
            for task_id in available_tasks:
                # New state with this task completed
                new_state = frozenset(current_state | {task_id})
                
                # Skip if already processed
                if new_state in closed_set:
                    continue
                
                # Calculate new g_score (cost from start)
                task = self.task_graph.tasks[task_id]
                new_g_score = g_score + task.estimated_hours
                
                # If this path is better than any previous path to this state
                if new_state not in g_scores or new_g_score < g_scores[new_state]:
                    # Update scores
                    g_scores[new_state] = new_g_score
                    f_scores[new_state] = new_g_score + self._heuristic(new_state)
                    
                    # Add to open set
                    new_path = path + [task_id]
                    heapq.heappush(open_set, (f_scores[new_state], new_g_score, new_state, new_path))
        
        # No path found
        logger.warning("A* pathfinding: No path found")
        return []
    
    def _get_available_tasks(self, completed_tasks: Set[UUID]) -> List[UUID]:
        """
        Get tasks that can be executed next.
        
        Args:
            completed_tasks: Set of completed task IDs
            
        Returns:
            List of available task IDs
        """
        available_tasks = []
        
        for task_id, task in self.task_graph.tasks.items():
            # Skip if already completed
            if task_id in completed_tasks:
                continue
            
            # Check if dependencies are met
            dependencies_met = True
            for dep in task.dependencies:
                if dep.task_id not in completed_tasks:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                available_tasks.append(task_id)
        
        return available_tasks
    
    def _heuristic(self, state: Set[UUID]) -> float:
        """
        Calculate heuristic value for a state.
        
        Args:
            state: Current state (set of completed task IDs)
            
        Returns:
            Heuristic value
        """
        # Calculate based on the selected heuristic
        if self.heuristic == PlanningHeuristic.TIME_ESTIMATE:
            return self._time_estimate_heuristic(state)
        elif self.heuristic == PlanningHeuristic.RISK_LEVEL:
            return self._risk_level_heuristic(state)
        elif self.heuristic == PlanningHeuristic.SUCCESS_PROBABILITY:
            return self._success_probability_heuristic(state)
        elif self.heuristic == PlanningHeuristic.DEPENDENCY_COUNT:
            return self._dependency_count_heuristic(state)
        else:  # MULTI_FACTOR
            return self._multi_factor_heuristic(state)
    
    def _time_estimate_heuristic(self, state: Set[UUID]) -> float:
        """
        Heuristic based on estimated time to complete remaining tasks.
        
        Args:
            state: Current state
            
        Returns:
            Heuristic value
        """
        remaining_time = 0.0
        
        for task_id, task in self.task_graph.tasks.items():
            if task_id not in state:
                remaining_time += task.estimated_hours
        
        return remaining_time
    
    def _risk_level_heuristic(self, state: Set[UUID]) -> float:
        """
        Heuristic based on risk levels of remaining tasks.
        
        Args:
            state: Current state
            
        Returns:
            Heuristic value
        """
        risk_sum = 0.0
        count = 0
        
        for task_id, task in self.task_graph.tasks.items():
            if task_id not in state:
                risk_sum += self._risk_level_to_score(task.risk_level)
                count += 1
        
        return risk_sum * 10.0 if count > 0 else 0.0
    
    def _success_probability_heuristic(self, state: Set[UUID]) -> float:
        """
        Heuristic based on success probability of remaining tasks.
        
        Args:
            state: Current state
            
        Returns:
            Heuristic value
        """
        # Inverse of average success probability (lower probability = higher cost)
        prob_sum = 0.0
        count = 0
        
        for task_id, task in self.task_graph.tasks.items():
            if task_id not in state:
                prob_sum += task.success_probability
                count += 1
        
        avg_prob = prob_sum / count if count > 0 else 1.0
        return (1.0 - avg_prob) * 100.0  # Scale to be comparable with other heuristics
    
    def _dependency_count_heuristic(self, state: Set[UUID]) -> float:
        """
        Heuristic based on dependency counts.
        
        Args:
            state: Current state
            
        Returns:
            Heuristic value
        """
        dependency_count = 0
        
        for task_id, task in self.task_graph.tasks.items():
            if task_id not in state:
                for dep in task.dependencies:
                    if dep.task_id not in state:
                        dependency_count += 1
        
        return dependency_count
    
    def _multi_factor_heuristic(self, state: Set[UUID]) -> float:
        """
        Multi-factor heuristic combining several factors.
        
        Args:
            state: Current state
            
        Returns:
            Heuristic value
        """
        time_factor = self._time_estimate_heuristic(state) * self.weight_factors["time_estimate"]
        risk_factor = self._risk_level_heuristic(state) * self.weight_factors["risk_level"]
        prob_factor = self._success_probability_heuristic(state) * self.weight_factors["success_probability"]
        dep_factor = self._dependency_count_heuristic(state) * self.weight_factors["dependency_count"]
        
        return time_factor + risk_factor + prob_factor + dep_factor
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """
        Convert a risk level to a numeric score.
        
        Args:
            risk_level: Risk level to convert
            
        Returns:
            Numeric risk score
        """
        risk_scores = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.VERY_HIGH: 1.0
        }
        
        return risk_scores.get(risk_level, 0.5)
    
    def save_graph(self) -> None:
        """Save the pathfinding graph to disk."""
        graph_dir = get_pathfinding_directory()
        graph_path = graph_dir / f"astar_graph_{self.graph.id}.json"
        
        save_json(self.graph.model_dump(), graph_path)


class PlanningService:
    """Service for planning and optimizing task sequences."""
    
    def __init__(self, task_graph: TaskGraph):
        """
        Initialize the planning service.
        
        Args:
            task_graph: Graph of tasks and dependencies
        """
        self.task_graph = task_graph
    
    def create_planning_session(
        self,
        algorithm: PlanningAlgorithm,
        **algorithm_params
    ) -> PlanningSession:
        """
        Create a new planning session.
        
        Args:
            algorithm: Planning algorithm to use
            **algorithm_params: Additional parameters for the algorithm
            
        Returns:
            New planning session
        """
        session = PlanningSession(
            algorithm=algorithm
        )
        
        # Run the planning algorithm
        if algorithm == PlanningAlgorithm.MCTS:
            search = MonteCarloTreeSearch(
                task_graph=self.task_graph,
                **algorithm_params
            )
            tree = search.run()
            
            session.search_tree_id = tree.id
            session.recommended_task_sequence = search.get_recommended_sequence()
            
            # Calculate estimated completion time
            session.estimated_completion_time = sum(
                self.task_graph.tasks[task_id].estimated_hours
                for task_id in session.recommended_task_sequence
                if task_id in self.task_graph.tasks
            )
            
            # Calculate estimated success probability
            session.estimated_success_probability = calculate_sequence_success_probability(
                self.task_graph, session.recommended_task_sequence
            )
            
        elif algorithm == PlanningAlgorithm.A_STAR:
            pathfinding = AStarPathfinding(
                task_graph=self.task_graph,
                **algorithm_params
            )
            path = pathfinding.find_path()
            pathfinding.save_graph()
            
            session.pathfinding_graph_id = pathfinding.graph.id
            session.recommended_task_sequence = path
            session.estimated_completion_time = pathfinding.graph.path_cost
            
            # Calculate estimated success probability
            session.estimated_success_probability = calculate_sequence_success_probability(
                self.task_graph, session.recommended_task_sequence
            )
        
        return session