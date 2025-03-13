# search_algorithms.py
"""Implementation of MCTS and A* for development planning."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple, Callable
import math
import random
import time
import logging
import functools
from uuid import UUID, uuid4
from datetime import datetime
from heapq import heappush, heappop
import numpy as np
import os
from pathlib import Path

from .models.planning_models import (
    SearchNode, MCTSNode, AStarNode, SimulationConfig, 
    SimulationSession, PlanningPhase, SimulationResult
)
from .models.task_models import Task, TaskStatus, TaskGraph, TaskType
from .llm_integration import LLMIntegration, LLMConfig
from .utils.fs_utils import safe_save_json, resolve_path, load_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project type constants for domain-specific knowledge
PROJECT_TYPE_WEB_APP = "web_app"
PROJECT_TYPE_CLI = "cli"
PROJECT_TYPE_LIBRARY = "library"
PROJECT_TYPE_UNKNOWN = "unknown"

# Define complexity thresholds for depth limiting
COMPLEXITY_LOW = 2.0
COMPLEXITY_MEDIUM = 4.0
COMPLEXITY_HIGH = 6.0
COMPLEXITY_VERY_HIGH = 8.0

class ProjectTypeDetector:
    """Detects the type of project based on task graph analysis."""
    
    @staticmethod
    def detect_project_type(task_graph: TaskGraph) -> str:
        """
        Analyze task graph to determine the project type.
        
        Args:
            task_graph: The graph of tasks to analyze
            
        Returns:
            Project type string
        """
        web_indicators = [
            "html", "css", "javascript", "frontend", "backend", "api",
            "route", "endpoint", "react", "angular", "vue", "dom", "browser",
            "responsive", "ui", "ux", "interface", "web", "http", "rest"
        ]
        
        cli_indicators = [
            "command", "terminal", "console", "shell", "cli", "argument",
            "option", "flag", "stdin", "stdout", "stderr", "pipe", "script"
        ]
        
        library_indicators = [
            "library", "package", "module", "function", "method", "class",
            "abstraction", "interface", "api", "dependency", "import"
        ]
        
        # Count indicators in task titles and descriptions
        web_count, cli_count, lib_count = 0, 0, 0
        
        for task in task_graph.tasks.values():
            text = (task.title + " " + task.description).lower()
            
            for indicator in web_indicators:
                if indicator in text:
                    web_count += 1
                    
            for indicator in cli_indicators:
                if indicator in text:
                    cli_count += 1
                    
            for indicator in library_indicators:
                if indicator in text:
                    lib_count += 1
        
        # Determine project type based on highest count
        counts = [(web_count, PROJECT_TYPE_WEB_APP), 
                 (cli_count, PROJECT_TYPE_CLI), 
                 (lib_count, PROJECT_TYPE_LIBRARY)]
        
        highest_count, project_type = max(counts, key=lambda x: x[0])
        
        # If no clear indicators, return unknown
        if highest_count == 0:
            return PROJECT_TYPE_UNKNOWN
            
        return project_type
    
    @staticmethod
    def analyze_project_complexity(task_graph: TaskGraph) -> float:
        """
        Calculate the complexity of the project based on task estimates.
        
        Args:
            task_graph: The graph of tasks to analyze
            
        Returns:
            Complexity score (higher means more complex)
        """
        if not task_graph.tasks:
            return 0.0
            
        # Calculate average complexity and standard deviation
        complexities = [task.estimated_complexity for task in task_graph.tasks.values()]
        avg_complexity = sum(complexities) / len(complexities)
        
        # Calculate dependency density
        total_dependencies = sum(len(task.dependencies) for task in task_graph.tasks.values())
        max_possible_dependencies = len(task_graph.tasks) * (len(task_graph.tasks) - 1)
        dependency_density = total_dependencies / max_possible_dependencies if max_possible_dependencies > 0 else 0
        
        # Combine factors
        return avg_complexity * (1 + dependency_density * 2)

# Cache decorator for planning results
def cache_planning_result(func):
    """Decorator to cache planning results."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based on arguments
        try:
            # For MCTSPlanner.run_simulation, use task_graph and config as key
            if func.__name__ == "run_simulation" and isinstance(args[0], MCTSPlanner):
                planner = args[0]
                key = (
                    id(planner.task_graph),
                    planner.config.max_iterations,
                    planner.config.time_limit_seconds
                )
            # For AStarPathfinder.find_path, use task_graph as key
            elif func.__name__ == "find_path" and isinstance(args[0], AStarPathfinder):
                pathfinder = args[0]
                key = id(pathfinder.task_graph)
            else:
                # If not a recognized function, don't cache
                return func(*args, **kwargs)
                
            # Check if result is cached and not expired
            timestamp, result = cache.get(key, (None, None))
            if timestamp and (time.time() - timestamp < 300):  # 5-minute cache
                logger.info(f"Using cached result for {func.__name__}")
                return result
                
            # Run the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache[key] = (time.time(), result)
            
            # Limit cache size to avoid memory issues
            if len(cache) > 20:
                # Remove oldest entries
                oldest_key = min(cache.keys(), key=lambda k: cache[k][0])
                del cache[oldest_key]
                
            return result
        except Exception as e:
            logger.warning(f"Error in cache_planning_result: {e}")
            # If anything goes wrong with caching, just run the original function
            return func(*args, **kwargs)
    
    # Add a method to clear the cache
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper

class MCTSPlanner:
    """Monte Carlo Tree Search implementation for development planning."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        llm_integration: Optional[LLMIntegration] = None,
        config: Optional[SimulationConfig] = None,
        session_id: Optional[UUID] = None,
        project_type: Optional[str] = None
    ):
        """
        Initialize the MCTS planner.
        
        Args:
            task_graph: The graph of tasks and dependencies
            llm_integration: Optional LLM integration for evaluations
            config: Configuration for the simulation
            session_id: Optional ID for the simulation session
            project_type: Optional project type for domain-specific knowledge
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
        
        # Detect project type if not provided
        self.project_type = project_type or ProjectTypeDetector.detect_project_type(task_graph)
        
        # Calculate project complexity for depth limiting
        self.project_complexity = ProjectTypeDetector.analyze_project_complexity(task_graph)
        
        # Apply depth limiting based on project complexity
        self._adjust_max_depth_by_complexity()
        
        # Track statistics
        self.total_simulations = 0
        self.successful_simulations = 0
        
        # Save session directory
        self.session_dir = resolve_path(f"planning/search_trees/{self.session.id}", create_parents=True)
        
        # Cache for frequently accessed data
        self._task_type_cache = {}
        
        logger.info(f"Initialized MCTS Planner with project type: {self.project_type}, " 
                   f"complexity: {self.project_complexity:.2f}")
    
    def _adjust_max_depth_by_complexity(self) -> None:
        """Adjust maximum search depth based on project complexity."""
        base_depth = self.config.max_depth
        
        if self.project_complexity <= COMPLEXITY_LOW:
            # Simple projects can use full depth
            self.config.max_depth = base_depth
        elif self.project_complexity <= COMPLEXITY_MEDIUM:
            # Medium complexity projects use 80% of depth
            self.config.max_depth = int(base_depth * 0.8)
        elif self.project_complexity <= COMPLEXITY_HIGH:
            # High complexity projects use 60% of depth
            self.config.max_depth = int(base_depth * 0.6)
        else:
            # Very high complexity projects use 40% of depth
            self.config.max_depth = int(base_depth * 0.4)
            
        # Ensure minimum depth
        self.config.max_depth = max(5, self.config.max_depth)
        
        logger.info(f"Adjusted max depth to {self.config.max_depth} for complexity {self.project_complexity:.2f}")
    
    def _get_task_type(self, task_id: UUID) -> str:
        """Get the type of a task with caching."""
        if task_id in self._task_type_cache:
            return self._task_type_cache[task_id]
            
        if task_id in self.task_graph.tasks:
            task_type = self.task_graph.tasks[task_id].task_type.value
            self._task_type_cache[task_id] = task_type
            return task_type
            
        return "unknown"
    
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
    
    @cache_planning_result
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
            "session_id": str(self.session.id),
            "project_type": self.project_type,
            "project_complexity": self.project_complexity
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
        
        # Domain-specific adjustments
        if self.project_type == PROJECT_TYPE_WEB_APP:
            # For web apps, favor early foundation tasks
            exploration_weight *= self._get_web_app_exploration_factor(node)
        elif self.project_type == PROJECT_TYPE_CLI:
            # For CLI tools, favor early core command structure
            exploration_weight *= self._get_cli_exploration_factor(node)
        
        best_score = float('-inf')
        best_child = None
        
        for child_id in node.children_ids:
            if child_id not in self.nodes:
                continue
                
            child = self.nodes[child_id]
            
            # Apply domain-specific heuristics to update scores
            self._apply_domain_heuristics(child)
            
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
    
    def _get_web_app_exploration_factor(self, node: MCTSNode) -> float:
        """
        Get the exploration factor for web app projects.
        
        Args:
            node: Current node
            
        Returns:
            Exploration factor multiplier
        """
        # Prioritize certain task types based on completed tasks
        completed_types = [self._get_task_type(task_id) for task_id in node.completed_tasks]
        
        # Check if foundation tasks are completed
        has_setup = any(typ == TaskType.PLANNING.value for typ in completed_types)
        has_docs = any(typ == TaskType.DOCUMENTATION.value for typ in completed_types)
        
        if not has_setup:
            # Encourage exploration for planning tasks
            return 1.5
        elif not has_docs and node.depth > 3:
            # Encourage documentation after some progress
            return 1.2
            
        return 1.0
    
    def _get_cli_exploration_factor(self, node: MCTSNode) -> float:
        """
        Get the exploration factor for CLI projects.
        
        Args:
            node: Current node
            
        Returns:
            Exploration factor multiplier
        """
        # For CLI projects, early implementation of command structure is important
        if node.depth < 2:
            return 1.3
            
        # After initial structure, balance is more important
        return 1.0
    
    def _apply_domain_heuristics(self, node: MCTSNode) -> None:
        """
        Apply domain-specific heuristics to node scoring.
        
        Args:
            node: Node to apply heuristics to
        """
        if not node.task_id:
            return
            
        # Skip if task is not in graph
        if node.task_id not in self.task_graph.tasks:
            return
            
        task = self.task_graph.tasks[node.task_id]
        
        # Base adjustment is 1.0 (no change)
        adjustment = 1.0
        
        if self.project_type == PROJECT_TYPE_WEB_APP:
            # Web app heuristics
            if task.task_type == TaskType.PLANNING and node.depth < 3:
                # Prioritize early planning
                adjustment += 0.2
            elif "api" in task.title.lower() and node.depth < 5:
                # Prioritize early API development
                adjustment += 0.15
            elif task.task_type == TaskType.IMPLEMENTATION and "ui" in task.title.lower():
                # UI implementation typically comes after API
                if node.depth > 3:
                    adjustment += 0.1
                else:
                    adjustment -= 0.1
            elif task.task_type == TaskType.TEST:
                # Tests usually come after implementation
                if node.depth < 3:
                    adjustment -= 0.2
                else:
                    adjustment += 0.1
            
        elif self.project_type == PROJECT_TYPE_CLI:
            # CLI heuristics
            if "command" in task.title.lower() and node.depth < 3:
                # Prioritize early command structure
                adjustment += 0.2
            elif task.task_type == TaskType.DOCUMENTATION and node.depth > 5:
                # Documentation is more important later
                adjustment += 0.15
            
        # Apply the adjustment to the exploitation score
        node.exploitation_score *= adjustment
    
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
        
        # Apply domain-specific task selection
        if self.project_type == PROJECT_TYPE_WEB_APP:
            # For web apps, use weighted selection based on task type
            return self._expand_web_app_node(node, available_tasks)
        elif self.project_type == PROJECT_TYPE_CLI:
            # For CLI tools, prefer core functionality first
            return self._expand_cli_node(node, available_tasks)
        else:
            # Default expansion with random selection
            task_id = random.choice(list(available_tasks))
            return self._create_child_node(node, task_id)
    
    def _expand_web_app_node(self, node: MCTSNode, available_tasks: Dict[UUID, Task]) -> MCTSNode:
        """
        Expand a node with web app domain knowledge.
        
        Args:
            node: Node to expand
            available_tasks: Dictionary of available tasks
            
        Returns:
            Newly created child node
        """
        # Define weights for different task types for web apps
        type_weights = {
            TaskType.PLANNING.value: 5.0 if node.depth < 2 else 0.5,
            TaskType.IMPLEMENTATION.value: 3.0,
            TaskType.TEST.value: 1.0 if node.depth < 3 else 2.0,
            TaskType.DOCUMENTATION.value: 0.5 if node.depth < 3 else 2.0,
            TaskType.REFACTOR.value: 0.2 if node.depth < 4 else 1.0,
            TaskType.BUGFIX.value: 0.1 if node.depth < 4 else 1.5
        }
        
        # Calculate weights for available tasks
        task_weights = {}
        for task_id, task in available_tasks.items():
            base_weight = type_weights.get(task.task_type.value, 1.0)
            
            # Adjust weight based on dependencies
            dependency_factor = 1.0 + (len(task.dependencies) * 0.1)
            
            # Adjust weight based on title keywords for web apps
            title_lower = task.title.lower()
            if "api" in title_lower or "backend" in title_lower:
                base_weight *= 1.3 if node.depth < 3 else 0.9
            elif "ui" in title_lower or "frontend" in title_lower:
                base_weight *= 0.8 if node.depth < 3 else 1.2
            elif "database" in title_lower or "model" in title_lower:
                base_weight *= 1.5 if node.depth < 4 else 0.8
            
            task_weights[task_id] = base_weight * dependency_factor
        
        # Use weighted random selection
        total_weight = sum(task_weights.values())
        if total_weight == 0:
            # Fallback to uniform selection
            task_id = random.choice(list(available_tasks))
        else:
            # Weighted selection
            rand_val = random.uniform(0, total_weight)
            cumulative = 0
            for task_id, weight in task_weights.items():
                cumulative += weight
                if cumulative >= rand_val:
                    break
        
        return self._create_child_node(node, task_id)
    
    def _expand_cli_node(self, node: MCTSNode, available_tasks: Dict[UUID, Task]) -> MCTSNode:
        """
        Expand a node with CLI domain knowledge.
        
        Args:
            node: Node to expand
            available_tasks: Dictionary of available tasks
            
        Returns:
            Newly created child node
        """
        # For CLI applications, prioritize core command structure first
        for task_id, task in available_tasks.items():
            title_lower = task.title.lower()
            if ("command" in title_lower or "cli" in title_lower or "arg" in title_lower) and node.depth < 3:
                return self._create_child_node(node, task_id)
        
        # Next, prioritize implementation tasks
        implementation_tasks = {
            task_id: task for task_id, task in available_tasks.items()
            if task.task_type == TaskType.IMPLEMENTATION
        }
        
        if implementation_tasks and node.depth < 5:
            task_id = random.choice(list(implementation_tasks))
            return self._create_child_node(node, task_id)
        
        # Then, select randomly from the remaining tasks
        task_id = random.choice(list(available_tasks))
        return self._create_child_node(node, task_id)
    
    def _create_child_node(self, parent: MCTSNode, task_id: UUID) -> MCTSNode:
        """
        Create a child node for the specified task.
        
        Args:
            parent: Parent node
            task_id: ID of the task for the new node
            
        Returns:
            Newly created child node
        """
        # Create a new node
        new_task_sequence = parent.task_sequence.copy() + [task_id]
        new_completed_tasks = parent.completed_tasks.copy()
        new_completed_tasks.add(task_id)
        
        child = MCTSNode(
            node_type="task",
            parent_id=parent.id,
            task_id=task_id,
            task_sequence=new_task_sequence,
            completed_tasks=new_completed_tasks,
            depth=parent.depth + 1
        )
        
        # Add to nodes dictionary
        self.nodes[child.id] = child
        
        # Add child to parent
        if child.id not in parent.children_ids:
            parent.children_ids.append(child.id)
        
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
        
        # Use domain-specific rollout policy if applicable
        if self.project_type == PROJECT_TYPE_WEB_APP:
            return self._web_app_rollout(node, completed_tasks, depth)
        elif self.project_type == PROJECT_TYPE_CLI:
            return self._cli_rollout(node, completed_tasks, depth)
        else:
            # Default rollout policy
            return self._default_rollout(node, completed_tasks, depth)
    
    def _default_rollout(self, node: MCTSNode, completed_tasks: Set[UUID], depth: int) -> SimulationResult:
        """Default rollout policy for any project type."""
        # Continue rollout until we complete all tasks or reach max depth
        max_depth = self.config.max_depth
        
        while depth < max_depth:
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
    
    def _web_app_rollout(self, node: MCTSNode, completed_tasks: Set[UUID], depth: int) -> SimulationResult:
        """Web app specific rollout policy."""
        # For web apps, use a more structured approach
        max_depth = self.config.max_depth
        
        # Track stages of development
        planning_done = any(
            task_id in completed_tasks and 
            self.task_graph.tasks[task_id].task_type == TaskType.PLANNING
            for task_id in self.task_graph.tasks
            if task_id in completed_tasks
        )
        
        while depth < max_depth:
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
            
            # Priority selection for web apps
            selected_task_id = None
            
            # First ensure planning is done
            if not planning_done:
                planning_tasks = {
                    task_id: task for task_id, task in available_tasks.items()
                    if task.task_type == TaskType.PLANNING
                }
                if planning_tasks:
                    selected_task_id = random.choice(list(planning_tasks.keys()))
                    planning_done = True
            
            # Then prioritize backend/API tasks
            if not selected_task_id:
                backend_tasks = {
                    task_id: task for task_id, task in available_tasks.items()
                    if "api" in task.title.lower() or "backend" in task.title.lower()
                }
                if backend_tasks and depth < 5:
                    selected_task_id = random.choice(list(backend_tasks.keys()))
            
            # Then UI/frontend tasks
            if not selected_task_id:
                frontend_tasks = {
                    task_id: task for task_id, task in available_tasks.items()
                    if "ui" in task.title.lower() or "frontend" in task.title.lower()
                }
                if frontend_tasks and depth >= 3:
                    selected_task_id = random.choice(list(frontend_tasks.keys()))
            
            # Fall back to random selection
            if not selected_task_id:
                selected_task_id = random.choice(list(available_tasks.keys()))
            
            # Add to completed tasks
            completed_tasks.add(selected_task_id)
            depth += 1
        
        # If we reached max depth without completing all tasks
        return SimulationResult.PARTIAL_SUCCESS
    
    def _cli_rollout(self, node: MCTSNode, completed_tasks: Set[UUID], depth: int) -> SimulationResult:
        """CLI app specific rollout policy."""
        # For CLI apps, focus on command structure first
        max_depth = self.config.max_depth
        
        # Track progress through CLI development phases
        command_structure_done = False
        
        while depth < max_depth:
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
            
            # Priority selection for CLI apps
            selected_task_id = None
            
            # First ensure command structure is done
            if not command_structure_done and depth < 3:
                command_tasks = {
                    task_id: task for task_id, task in available_tasks.items()
                    if "command" in task.title.lower() or "cli" in task.title.lower()
                }
                if command_tasks:
                    selected_task_id = random.choice(list(command_tasks.keys()))
                    command_structure_done = True
            
            # Then focus on implementation
            if not selected_task_id:
                impl_tasks = {
                    task_id: task for task_id, task in available_tasks.items()
                    if task.task_type == TaskType.IMPLEMENTATION
                }
                if impl_tasks:
                    selected_task_id = random.choice(list(impl_tasks.keys()))
            
            # Fall back to random selection
            if not selected_task_id:
                selected_task_id = random.choice(list(available_tasks.keys()))
            
            # Add to completed tasks
            completed_tasks.add(selected_task_id)
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
            "total_simulations": self.total_simulations,
            "project_type": self.project_type,
            "project_complexity": self.project_complexity
        }
        safe_save_json(sequence_data, sequence_path)

    def save_session(self) -> None:
        """Save the current session state to disk."""
        if not self.session_dir.exists():
            os.makedirs(self.session_dir, exist_ok=True)
        
        # Save session metadata
        session_data = {
            "id": self.session.id,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_simulations": self.total_simulations,
                "successful_simulations": self.successful_simulations
            },
            "config": {
                "max_iterations": self.config.max_iterations,
                "exploration_weight": self.config.exploration_weight,
                "max_depth": self.config.max_depth
            },
            "project_info": {
                "type": self.project_type,
                "complexity": self.project_complexity
            }
        }
        
        # Use safe_save_json instead of save_json
        safe_save_json(session_data, self.session_dir / "session.json")
        
        # Save tree nodes
        self.save_tree()
    
    def save_tree(self) -> None:
        """Save the current tree state to disk."""
        if not self.session_dir.exists():
            os.makedirs(self.session_dir, exist_ok=True)
        
        # Serialize the nodes
        serialized_nodes = {}
        for node_id, node in self.nodes.items():
            serialized_nodes[str(node_id)] = {
                "id": str(node.id),
                "parent_id": str(node.parent_id) if node.parent_id else None,
                "state": [str(task_id) for task_id in node.state] if hasattr(node, 'state') else [],
                "untried_actions": [str(task_id) for task_id in node.untried_actions] if hasattr(node, 'untried_actions') else [],
                "children": [str(child_id) for child_id in node.children_ids],
                "visits": node.visits,
                "value": node.value,
                "wins": node.wins,
                "losses": node.losses,
                "draws": node.draws
            }
        
        # Save nodes
        safe_save_json(serialized_nodes, self.session_dir / "nodes.json")
        
        # Save best sequence if available
        if self._get_best_sequence():
            sequence_data = {
                "sequence": [str(task_id) for task_id in self._get_best_sequence()],
                "value": self.successful_simulations / max(1, self.total_simulations),
                "timestamp": datetime.now().isoformat()
            }
            safe_save_json(sequence_data, self.session_dir / "best_sequence.json")

class AStarPathfinder:
    """A* search implementation for finding optimal task sequences."""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        heuristic_function: Optional[Callable[[UUID, Set[UUID]], float]] = None,
        config: Optional[Dict[str, Any]] = None,
        project_type: Optional[str] = None
    ):
        """
        Initialize the A* pathfinder.
        
        Args:
            task_graph: The graph of tasks and dependencies
            heuristic_function: Optional custom heuristic function
            config: Optional configuration
            project_type: Optional project type for optimized heuristics
        """
        self.task_graph = task_graph
        self.config = config or {
            "heuristic_weight": 1.0,
            "max_iterations": 1000,
            "use_caching": True,
            "depth_limit": None
        }
        
        # Detect project type if not provided
        self.project_type = project_type or ProjectTypeDetector.detect_project_type(task_graph)
        
        # Calculate project complexity
        self.project_complexity = ProjectTypeDetector.analyze_project_complexity(task_graph)
        
        # Initialize heuristic function based on project type
        if heuristic_function:
            self.heuristic_function = heuristic_function
        else:
            if self.project_type == PROJECT_TYPE_WEB_APP:
                self.heuristic_function = self._web_app_heuristic
            elif self.project_type == PROJECT_TYPE_CLI:
                self.heuristic_function = self._cli_app_heuristic
            else:
                self.heuristic_function = self._default_heuristic
        
        # Apply depth limiting based on project complexity
        self._adjust_depth_limit_by_complexity()
        
        # Nodes and results
        self.nodes: Dict[UUID, AStarNode] = {}
        self.came_from: Dict[UUID, UUID] = {}
        self.g_score: Dict[UUID, float] = {}
        self.f_score: Dict[UUID, float] = {}
        
        # Cache for heuristic calculations
        self._heuristic_cache: Dict[Tuple[UUID, frozenset], float] = {}
        
        logger.info(f"Initialized A* Pathfinder with project type: {self.project_type}, "
                   f"complexity: {self.project_complexity:.2f}")
    
    def _adjust_depth_limit_by_complexity(self) -> None:
        """Adjust depth limit based on project complexity."""
        if self.config.get("depth_limit") is not None:
            # Already set, don't override
            return
            
        if self.project_complexity <= COMPLEXITY_LOW:
            # Simple projects can use higher limit
            self.config["depth_limit"] = 50
        elif self.project_complexity <= COMPLEXITY_MEDIUM:
            # Medium complexity projects
            self.config["depth_limit"] = 30
        elif self.project_complexity <= COMPLEXITY_HIGH:
            # High complexity projects
            self.config["depth_limit"] = 20
        else:
            # Very high complexity projects
            self.config["depth_limit"] = 15
            
        logger.info(f"Set depth limit to {self.config['depth_limit']} for complexity {self.project_complexity:.2f}")
    
    @cache_planning_result
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
        # Clear caches for a fresh run
        self._heuristic_cache.clear()
        
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
        
        # Depth tracking
        current_depth = 0
        depth_limit = self.config.get("depth_limit")
        
        # A* search
        while open_set and iterations < self.config["max_iterations"]:
            iterations += 1
            
            # Get node with lowest f_score
            current_f, current_id = heappop(open_set)
            
            if current_id in closed_set:
                continue
                
            current = self.nodes[current_id]
            current_depth = current.depth
            
            # Check if we've reached depth limit
            if depth_limit and current_depth >= depth_limit:
                logger.info(f"Reached depth limit {depth_limit}, terminating search")
                break
            
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
                base_cost = task.estimated_duration_hours
                
                # Apply domain-specific cost adjustments
                adjusted_cost = self._adjust_cost_by_domain(task_id, current.completed_tasks, base_cost)
                tentative_g_score = self.g_score[current_id] + adjusted_cost
                
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
            
            # Periodically clean up the heuristic cache if it gets too large
            if self.config.get("use_caching", True) and len(self._heuristic_cache) > 1000:
                self._heuristic_cache = {}
        
        # Reconstruct path if found
        path = []
        metadata = {
            "iterations": iterations,
            "nodes_explored": len(closed_set),
            "path_found": path_found,
            "project_type": self.project_type,
            "project_complexity": self.project_complexity,
            "depth_reached": current_depth
        }
        
        if path_found and goal_node_id:
            path = self._reconstruct_path(goal_node_id)
            metadata["path_length"] = len(path)
            metadata["estimated_duration"] = sum(
                self.task_graph.tasks[task_id].estimated_duration_hours
                for task_id in path
            )
        
        return path, metadata
    
    def _adjust_cost_by_domain(self, task_id: UUID, completed_tasks: Set[UUID], base_cost: float) -> float:
        """
        Adjust task cost based on domain-specific knowledge.
        
        Args:
            task_id: ID of the task
            completed_tasks: Set of already completed tasks
            base_cost: Base cost for the task
            
        Returns:
            Adjusted cost
        """
        if task_id not in self.task_graph.tasks:
            return base_cost
            
        task = self.task_graph.tasks[task_id]
        adjustment = 1.0
        
        if self.project_type == PROJECT_TYPE_WEB_APP:
            # Web app specific cost adjustments
            title_lower = task.title.lower()
            
            # Backend/API tasks should come before frontend
            if ("api" in title_lower or "backend" in title_lower) and len(completed_tasks) < 3:
                adjustment = 0.8  # Reduce cost to prioritize
            
            # UI tasks should come after backend
            if ("ui" in title_lower or "frontend" in title_lower):
                backend_count = sum(1 for t_id in completed_tasks if 
                                  "api" in self.task_graph.tasks[t_id].title.lower() 
                                  if t_id in self.task_graph.tasks)
                if backend_count == 0:
                    adjustment = 1.3  # Increase cost to deprioritize
                else:
                    adjustment = 0.9  # Slightly reduce cost
            
            # Documentation should come later
            if task.task_type == TaskType.DOCUMENTATION and len(completed_tasks) < 5:
                adjustment = 1.2  # Increase cost to deprioritize
                
        elif self.project_type == PROJECT_TYPE_CLI:
            # CLI specific cost adjustments
            title_lower = task.title.lower()
            
            # Command structure should be early
            if ("command" in title_lower or "cli" in title_lower) and len(completed_tasks) < 3:
                adjustment = 0.7  # Reduce cost to prioritize
            
            # Implementation should follow command structure
            if task.task_type == TaskType.IMPLEMENTATION:
                command_count = sum(1 for t_id in completed_tasks if 
                                  "command" in self.task_graph.tasks[t_id].title.lower() 
                                  if t_id in self.task_graph.tasks)
                if command_count > 0:
                    adjustment = 0.8  # Reduce cost to prioritize
        
        return base_cost * adjustment
    
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
        
        # Use caching if enabled
        if self.config.get("use_caching", True):
            cache_key = (node_id, frozenset(node.remaining_tasks))
            if cache_key in self._heuristic_cache:
                return self._heuristic_cache[cache_key]
            
        # Calculate the heuristic
        heuristic_value = self.heuristic_function(node_id, node.remaining_tasks) * self.config["heuristic_weight"]
        
        # Cache the result
        if self.config.get("use_caching", True):
            cache_key = (node_id, frozenset(node.remaining_tasks))
            self._heuristic_cache[cache_key] = heuristic_value
            
        return heuristic_value
    
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
    
    def _web_app_heuristic(self, node_id: UUID, remaining_tasks: Set[UUID]) -> float:
        """
        Web app specific heuristic that prioritizes backend before frontend.
        
        Args:
            node_id: ID of the node
            remaining_tasks: Set of remaining task IDs
            
        Returns:
            Heuristic value
        """
        if not remaining_tasks:
            return 0.0
            
        # Start with basic duration estimate
        base_estimate = self._default_heuristic(node_id, remaining_tasks)
        
        # Check for web app specific patterns
        node = self.nodes[node_id]
        completed_tasks = node.completed_tasks
        
        # Group tasks by type
        backend_tasks = set()
        frontend_tasks = set()
        other_tasks = set()
        
        for task_id in remaining_tasks:
            if task_id not in self.task_graph.tasks:
                continue
                
            task = self.task_graph.tasks[task_id]
            title_lower = task.title.lower()
            
            if "api" in title_lower or "backend" in title_lower or "server" in title_lower:
                backend_tasks.add(task_id)
            elif "ui" in title_lower or "frontend" in title_lower or "client" in title_lower:
                frontend_tasks.add(task_id)
            else:
                other_tasks.add(task_id)
        
        # Calculate adjustments
        adjustment = 1.0
        
        # If there are backend tasks remaining but frontend tasks already done,
        # that's not ideal - increase heuristic estimate
        backend_completed = any("api" in self.task_graph.tasks[t_id].title.lower() 
                             for t_id in completed_tasks 
                             if t_id in self.task_graph.tasks)
        
        if frontend_tasks and not backend_completed and backend_tasks:
            adjustment += 0.2
        
        # Adjust heuristic based on critical dependencies
        for task_id in remaining_tasks:
            if task_id not in self.task_graph.tasks:
                continue
                
            task = self.task_graph.tasks[task_id]
            # If many tasks depend on this one, prioritize it
            if len(task.dependents) > 2:
                adjustment -= 0.05 * len(task.dependents)
        
        # Ensure adjustment doesn't go negative
        adjustment = max(0.8, adjustment)
        
        return base_estimate * adjustment
    
    def _cli_app_heuristic(self, node_id: UUID, remaining_tasks: Set[UUID]) -> float:
        """
        CLI app specific heuristic that prioritizes command structure first.
        
        Args:
            node_id: ID of the node
            remaining_tasks: Set of remaining task IDs
            
        Returns:
            Heuristic value
        """
        if not remaining_tasks:
            return 0.0
            
        # Start with basic duration estimate
        base_estimate = self._default_heuristic(node_id, remaining_tasks)
        
        # Check for CLI app specific patterns
        node = self.nodes[node_id]
        completed_tasks = node.completed_tasks
        
        # Group tasks by type
        command_tasks = set()
        implementation_tasks = set()
        other_tasks = set()
        
        for task_id in remaining_tasks:
            if task_id not in self.task_graph.tasks:
                continue
                
            task = self.task_graph.tasks[task_id]
            title_lower = task.title.lower()
            
            if "command" in title_lower or "cli" in title_lower or "argument" in title_lower:
                command_tasks.add(task_id)
            elif task.task_type == TaskType.IMPLEMENTATION:
                implementation_tasks.add(task_id)
            else:
                other_tasks.add(task_id)
        
        # Calculate adjustments
        adjustment = 1.0
        
        # If command structure tasks are still pending, prioritize them
        if command_tasks and len(completed_tasks) < 3:
            adjustment -= 0.2
        
        # Adjust heuristic based on dependencies
        for task_id in remaining_tasks:
            if task_id not in self.task_graph.tasks:
                continue
                
            task = self.task_graph.tasks[task_id]
            # If many tasks depend on this one, prioritize it
            if len(task.dependents) > 1:
                adjustment -= 0.05 * len(task.dependents)
        
        # Ensure adjustment doesn't go negative
        adjustment = max(0.7, adjustment)
        
        return base_estimate * adjustment
    
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