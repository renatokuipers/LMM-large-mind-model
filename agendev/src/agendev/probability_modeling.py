# probability_modeling.py
"""Bayesian models for task completion probability."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple
import math
import random
import time
import logging
import json
import functools
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from pathlib import Path

from .models.task_models import Task, TaskStatus, TaskPriority, TaskRisk, TaskGraph, TaskType
from .models.planning_models import SimulationResult
from .llm_integration import LLMIntegration, LLMConfig, Message
from .utils.fs_utils import safe_save_json, resolve_path, load_json

# Use the same project type detector from search_algorithms
from .search_algorithms import ProjectTypeDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project type constants for domain-specific knowledge
PROJECT_TYPE_WEB_APP = "web_app"
PROJECT_TYPE_CLI = "cli"
PROJECT_TYPE_LIBRARY = "library"
PROJECT_TYPE_UNKNOWN = "unknown"

class ProbabilityDistribution:
    """Base class for probability distributions."""
    
    def sample(self) -> float:
        """Sample a value from the distribution."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def mean(self) -> float:
        """Get the mean of the distribution."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def variance(self) -> float:
        """Get the variance of the distribution."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get the confidence interval for the distribution."""
        raise NotImplementedError("Subclasses must implement this method")

class BetaDistribution(ProbabilityDistribution):
    """Beta distribution for modeling probabilities."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the beta distribution.
        
        Args:
            alpha: Alpha parameter (successes + 1)
            beta: Beta parameter (failures + 1)
        """
        self.alpha = max(0.01, alpha)  # Avoid zero
        self.beta = max(0.01, beta)    # Avoid zero
    
    def sample(self) -> float:
        """Sample a value from the beta distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def mean(self) -> float:
        """Get the mean of the beta distribution."""
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        """Get the variance of the beta distribution."""
        denominator = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        return (self.alpha * self.beta) / denominator
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get the confidence interval for the beta distribution.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Use percentile method
        lower_percentile = (1 - confidence) / 2
        upper_percentile = 1 - lower_percentile
        
        try:
            from scipy import stats
            lower = stats.beta.ppf(lower_percentile, self.alpha, self.beta)
            upper = stats.beta.ppf(upper_percentile, self.alpha, self.beta)
            return (lower, upper)
        except ImportError:
            # Fallback if SciPy is not available
            samples = [self.sample() for _ in range(1000)]
            samples.sort()
            lower_idx = int(lower_percentile * 1000)
            upper_idx = int(upper_percentile * 1000)
            return (samples[lower_idx], samples[upper_idx])
    
    def update(self, successes: int, failures: int) -> None:
        """
        Update the distribution with new observations.
        
        Args:
            successes: Number of successes observed
            failures: Number of failures observed
        """
        self.alpha += successes
        self.beta += failures
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean(),
            "variance": self.variance()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> BetaDistribution:
        """Create from dictionary representation."""
        return cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0)
        )

class TaskProbabilityModel:
    """Models task completion probabilities using Bayesian reasoning."""
    
    def __init__(
        self,
        task_graph: Optional[TaskGraph] = None,
        llm_integration: Optional[LLMIntegration] = None,
        model_id: Optional[str] = None
    ):
        """
        Initialize the task probability model.
        
        Args:
            task_graph: Optional task graph
            llm_integration: Optional LLM integration for generating priors
            model_id: Optional identifier for the model
        """
        self.task_graph = task_graph
        self.llm_integration = llm_integration
        self.model_id = model_id or f"model_{int(time.time())}"
        
        # Probability distributions for each task
        self.task_distributions: Dict[UUID, BetaDistribution] = {}
        
        # Dependency risk factors
        self.dependency_risks: Dict[Tuple[UUID, UUID], float] = {}
        
        # Statistics tracking
        self.simulation_results: Dict[UUID, List[SimulationResult]] = {}
        self.task_attempts: Dict[UUID, int] = {}
        self.task_successes: Dict[UUID, int] = {}
        
        # Initialize model directory
        self.model_dir = resolve_path(f"planning/simulations/{self.model_id}", create_parents=True)
        
        # Initialize distributions if task graph is provided
        if task_graph:
            self._initialize_distributions()
    
    def _initialize_distributions(self) -> None:
        """Initialize probability distributions for all tasks."""
        if not self.task_graph:
            return
            
        for task_id, task in self.task_graph.tasks.items():
            # Create initial distribution based on task metadata
            self.task_distributions[task_id] = self._create_prior_distribution(task)
            
            # Initialize statistics
            self.simulation_results[task_id] = []
            self.task_attempts[task_id] = 0
            self.task_successes[task_id] = 0
    
    def _create_prior_distribution(self, task: Task) -> BetaDistribution:
        """
        Create a prior distribution for a task based on its metadata.
        
        Args:
            task: The task to create a prior for
            
        Returns:
            Beta distribution representing prior beliefs
        """
        # Start with neutral prior
        alpha = 1.0
        beta = 1.0
        
        # Adjust based on risk level
        if task.risk == TaskRisk.LOW:
            alpha += 3.0  # More likely to succeed
            beta += 1.0
        elif task.risk == TaskRisk.MEDIUM:
            alpha += 2.0
            beta += 2.0
        elif task.risk == TaskRisk.HIGH:
            alpha += 1.0
            beta += 2.0
        elif task.risk == TaskRisk.CRITICAL:
            alpha += 1.0
            beta += 3.0  # More likely to fail
        
        # Adjust based on complexity
        complexity_factor = task.estimated_complexity
        if complexity_factor > 1.0:
            # More complex tasks are harder
            beta += (complexity_factor - 1.0) * 0.5
        
        # If LLM integration is available, refine with LLM
        if self.llm_integration:
            alpha, beta = self._refine_prior_with_llm(task, alpha, beta)
        
        return BetaDistribution(alpha, beta)
    
    def _refine_prior_with_llm(self, task: Task, alpha: float, beta: float) -> Tuple[float, float]:
        """
        Use LLM to refine the prior distribution.
        
        Args:
            task: The task to refine
            alpha: Initial alpha value
            beta: Initial beta value
            
        Returns:
            Tuple of (refined_alpha, refined_beta)
        """
        if not self.llm_integration:
            return alpha, beta
            
        try:
            # Create prompt for the LLM
            prompt = f"""
            Task: {task.title}
            Description: {task.description}
            Risk Level: {task.risk.value}
            Complexity: {task.estimated_complexity}
            Estimated Duration: {task.estimated_duration_hours} hours
            
            Based on the above information, estimate the probability of successful completion of this task.
            Provide your estimate as a percentage between 0 and 100.
            
            Also, estimate your confidence in this prediction on a scale of 1-10, where:
            1 = Very low confidence (high uncertainty)
            10 = Very high confidence (low uncertainty)
            """
            
            # Define the schema for structured output
            json_schema = {
                "name": "task_probability",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "success_probability": {
                            "type": "number",
                            "description": "Estimated probability of success (0-100)",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in the estimate (1-10)",
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["success_probability", "confidence"]
                }
            }
            
            # IMPORTANT FIX: Don't clear the context, as this may cause empty messages in some scenarios
            # Get structured response from LLM
            response = self.llm_integration.structured_query(
                prompt=prompt,
                json_schema=json_schema,
                clear_context=False,  # Changed from True to False
                save_to_context=True  # Changed from False to True
            )
            
            # Extract values
            success_prob = response.get("success_probability", 50) / 100.0  # Convert to 0-1
            confidence = response.get("confidence", 5) / 10.0  # Convert to 0-1
            
            # Higher confidence = stronger prior
            strength = 1.0 + confidence * 9.0  # Scale to 1-10
            
            # Convert to alpha/beta
            new_alpha = 1.0 + success_prob * strength
            new_beta = 1.0 + (1.0 - success_prob) * strength
            
            # Blend with original prior (giving more weight to LLM)
            alpha = 0.3 * alpha + 0.7 * new_alpha
            beta = 0.3 * beta + 0.7 * new_beta
            
        except Exception as e:
            logger.warning(f"Error refining prior with LLM: {e}")
        
        return alpha, beta
    
    def get_task_probability(self, task_id: UUID) -> float:
        """
        Get the probability of successful completion for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Probability of success (0-1)
        """
        if task_id not in self.task_distributions:
            # If no distribution exists, create one
            if self.task_graph and task_id in self.task_graph.tasks:
                self.task_distributions[task_id] = self._create_prior_distribution(self.task_graph.tasks[task_id])
            else:
                # Default to 0.5 if task not found
                return 0.5
        
        return self.task_distributions[task_id].mean()
    
    def get_confidence_interval(self, task_id: UUID, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get the confidence interval for task completion probability.
        
        Args:
            task_id: ID of the task
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if task_id not in self.task_distributions:
            # If no distribution exists, create one
            if self.task_graph and task_id in self.task_graph.tasks:
                self.task_distributions[task_id] = self._create_prior_distribution(self.task_graph.tasks[task_id])
            else:
                # Default wide interval if task not found
                return (0.25, 0.75)
        
        return self.task_distributions[task_id].confidence_interval(confidence)
    
    def update_from_simulation(self, task_id: UUID, result: SimulationResult) -> None:
        """
        Update the probability model with a simulation result.
        
        Args:
            task_id: ID of the task
            result: Result of the simulation
        """
        if task_id not in self.task_distributions:
            # If no distribution exists, create one
            if self.task_graph and task_id in self.task_graph.tasks:
                self.task_distributions[task_id] = self._create_prior_distribution(self.task_graph.tasks[task_id])
            else:
                # Skip if task not found
                return
        
        # Track the result
        if task_id not in self.simulation_results:
            self.simulation_results[task_id] = []
        self.simulation_results[task_id].append(result)
        
        # Update statistics
        self.task_attempts[task_id] = self.task_attempts.get(task_id, 0) + 1
        
        # Update distribution
        if result == SimulationResult.SUCCESS:
            self.task_successes[task_id] = self.task_successes.get(task_id, 0) + 1
            self.task_distributions[task_id].update(1, 0)
        elif result == SimulationResult.FAILURE:
            self.task_distributions[task_id].update(0, 1)
        else:
            # Partial success counts as half a success
            self.task_successes[task_id] = self.task_successes.get(task_id, 0) + 0.5
            self.task_distributions[task_id].update(0.5, 0.5)
    
    def update_dependency_risk(self, source_id: UUID, target_id: UUID, risk_factor: float) -> None:
        """
        Update the risk factor for a dependency.
        
        Args:
            source_id: ID of the source task
            target_id: ID of the target task
            risk_factor: Risk factor (0-1) where higher means more risk
        """
        self.dependency_risks[(source_id, target_id)] = max(0.0, min(1.0, risk_factor))
    
    def calculate_sequence_probability(self, task_sequence: List[UUID]) -> float:
        """
        Calculate the joint probability of a task sequence succeeding.
        
        Args:
            task_sequence: Sequence of task IDs
            
        Returns:
            Joint probability of sequence success
        """
        if not task_sequence:
            return 1.0
            
        # Initialize with 1.0 (certainty)
        joint_prob = 1.0
        
        # Track completed tasks
        completed_tasks = set()
        
        for task_id in task_sequence:
            # Get base task probability
            task_prob = self.get_task_probability(task_id)
            
            # Adjust for dependencies
            dependency_factor = 1.0
            
            for completed_id in completed_tasks:
                # Check if there is a dependency
                risk = self.dependency_risks.get((completed_id, task_id), 0.0)
                
                # Apply risk factor
                if risk > 0:
                    dependency_factor *= (1.0 - risk * 0.5)
            
            # Apply dependency adjustment
            adjusted_prob = task_prob * dependency_factor
            
            # Update joint probability
            joint_prob *= adjusted_prob
            
            # Add to completed tasks
            completed_tasks.add(task_id)
        
        return joint_prob
    
    def sample_sequence_outcome(self, task_sequence: List[UUID]) -> List[Tuple[UUID, SimulationResult]]:
        """
        Sample outcomes for a task sequence.
        
        Args:
            task_sequence: Sequence of task IDs
            
        Returns:
            List of (task_id, result) tuples
        """
        outcomes = []
        completed_tasks = set()
        
        for task_id in task_sequence:
            # Get base task distribution
            if task_id not in self.task_distributions:
                if self.task_graph and task_id in self.task_graph.tasks:
                    self.task_distributions[task_id] = self._create_prior_distribution(self.task_graph.tasks[task_id])
                else:
                    # Use default
                    self.task_distributions[task_id] = BetaDistribution(1.0, 1.0)
            
            distribution = self.task_distributions[task_id]
            
            # Sample from distribution
            success_prob = distribution.sample()
            
            # Adjust for dependencies
            for completed_id in completed_tasks:
                risk = self.dependency_risks.get((completed_id, task_id), 0.0)
                if risk > 0:
                    success_prob *= (1.0 - risk * 0.5)
            
            # Determine outcome
            random_value = random.random()
            if random_value < success_prob:
                result = SimulationResult.SUCCESS
            elif random_value < success_prob + (1 - success_prob) * 0.5:
                result = SimulationResult.PARTIAL_SUCCESS
            else:
                result = SimulationResult.FAILURE
            
            outcomes.append((task_id, result))
            
            # If task failed, subsequent tasks may be affected
            if result == SimulationResult.FAILURE:
                # Exit early if critical failure
                if self.task_graph and task_id in self.task_graph.tasks:
                    task = self.task_graph.tasks[task_id]
                    if task.risk == TaskRisk.CRITICAL:
                        break
            
            # Add to completed tasks if at least partial success
            if result != SimulationResult.FAILURE:
                completed_tasks.add(task_id)
        
        return outcomes
    
    def save(self) -> str:
        """
        Save the probability model to disk.
        
        Returns:
            Path to the saved model
        """
        model_path = self.model_dir / "model.json"
        
        # Prepare model data
        model_data = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "task_distributions": {
                str(task_id): dist.to_dict()
                for task_id, dist in self.task_distributions.items()
            },
            "dependency_risks": {
                f"{source}_{target}": risk
                for (source, target), risk in self.dependency_risks.items()
            },
            "task_attempts": {
                str(task_id): attempts
                for task_id, attempts in self.task_attempts.items()
            },
            "task_successes": {
                str(task_id): successes
                for task_id, successes in self.task_successes.items()
            }
        }
        
        # Save to disk
        safe_save_json(model_data, model_path)
        
        # Save simulation results
        results_path = self.model_dir / "simulation_results.json"
        results_data = {
            str(task_id): [result.value for result in results]
            for task_id, results in self.simulation_results.items()
        }
        safe_save_json(results_data, results_path)
        
        return str(model_path)
    
    @classmethod
    def load(cls, model_id: str, task_graph: Optional[TaskGraph] = None) -> TaskProbabilityModel:
        """
        Load a probability model from disk.
        
        Args:
            model_id: ID of the model to load
            task_graph: Optional task graph
            
        Returns:
            Loaded probability model
        """
        model_dir = resolve_path(f"planning/simulations/{model_id}")
        model_path = model_dir / "model.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model data
        model_data = load_json(model_path)
        
        # Create model
        model = cls(task_graph=task_graph, model_id=model_id)
        
        # Load distributions
        for task_id_str, dist_data in model_data.get("task_distributions", {}).items():
            task_id = UUID(task_id_str)
            model.task_distributions[task_id] = BetaDistribution.from_dict(dist_data)
        
        # Load dependency risks
        for key, risk in model_data.get("dependency_risks", {}).items():
            source_str, target_str = key.split("_")
            model.dependency_risks[(UUID(source_str), UUID(target_str))] = risk
        
        # Load statistics
        for task_id_str, attempts in model_data.get("task_attempts", {}).items():
            model.task_attempts[UUID(task_id_str)] = attempts
            
        for task_id_str, successes in model_data.get("task_successes", {}).items():
            model.task_successes[UUID(task_id_str)] = successes
        
        # Load simulation results
        results_path = model_dir / "simulation_results.json"
        if results_path.exists():
            results_data = load_json(results_path)
            for task_id_str, results in results_data.items():
                model.simulation_results[UUID(task_id_str)] = [
                    SimulationResult(result) for result in results
                ]
        
        return model
    
    def generate_task_report(self, task_id: UUID) -> Dict[str, Any]:
        """
        Generate a detailed report for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task report data
        """
        if task_id not in self.task_distributions:
            return {"error": "Task not found in probability model"}
            
        # Get task details
        task_name = "Unknown"
        task_description = ""
        if self.task_graph and task_id in self.task_graph.tasks:
            task = self.task_graph.tasks[task_id]
            task_name = task.title
            task_description = task.description
        
        # Get probability distribution
        distribution = self.task_distributions[task_id]
        success_prob = distribution.mean()
        confidence_interval = distribution.confidence_interval()
        
        # Get simulation statistics
        attempts = self.task_attempts.get(task_id, 0)
        successes = self.task_successes.get(task_id, 0)
        success_rate = successes / max(1, attempts) if attempts > 0 else float('nan')
        
        # Count result types
        results = self.simulation_results.get(task_id, [])
        result_counts = {
            SimulationResult.SUCCESS.value: 0,
            SimulationResult.PARTIAL_SUCCESS.value: 0,
            SimulationResult.FAILURE.value: 0
        }
        for result in results:
            result_counts[result.value] = result_counts.get(result.value, 0) + 1
        
        # Generate report
        report = {
            "task_id": str(task_id),
            "task_name": task_name,
            "task_description": task_description,
            "probability": {
                "success_probability": success_prob,
                "confidence_interval": confidence_interval,
                "alpha": distribution.alpha,
                "beta": distribution.beta,
                "variance": distribution.variance()
            },
            "simulations": {
                "total_attempts": attempts,
                "total_successes": successes,
                "success_rate": success_rate,
                "result_counts": result_counts
            },
            "dependencies": {
                "risk_factors": {
                    str(dep_id): risk
                    for (source, dep_id), risk in self.dependency_risks.items()
                    if source == task_id
                }
            }
        }
        
        return report

class ProjectRiskModel:
    """Models overall project risk using task probability models."""
    
    def __init__(
        self,
        task_probability_model: TaskProbabilityModel,
        task_graph: TaskGraph
    ):
        """
        Initialize the project risk model.
        
        Args:
            task_probability_model: Model for task probabilities
            task_graph: Task dependency graph
        """
        self.task_model = task_probability_model
        self.task_graph = task_graph
        
        # Cache for computed paths
        self.critical_path_cache: Optional[List[UUID]] = None
    
    def calculate_project_success_probability(self) -> float:
        """
        Calculate the overall probability of project success.
        
        Returns:
            Probability of project success (0-1)
        """
        # Get the critical path
        critical_path = self._get_critical_path()
        
        # Calculate the joint probability of the critical path
        return self.task_model.calculate_sequence_probability(critical_path)
    
    def _get_critical_path(self) -> List[UUID]:
        """
        Get the critical path through the task graph.
        
        Returns:
            List of task IDs on the critical path
        """
        if self.critical_path_cache is not None:
            return self.critical_path_cache
            
        self.critical_path_cache = self.task_graph.get_critical_path()
        return self.critical_path_cache
    
    def identify_risk_hotspots(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify high-risk tasks that could jeopardize the project.
        
        Args:
            threshold: Probability threshold for considering a task high-risk
            
        Returns:
            List of high-risk task details
        """
        critical_path = self._get_critical_path()
        hotspots = []
        
        for task_id in critical_path:
            prob = self.task_model.get_task_probability(task_id)
            
            if prob < threshold:
                # This is a risk hotspot
                task_name = "Unknown"
                if task_id in self.task_graph.tasks:
                    task_name = self.task_graph.tasks[task_id].title
                
                hotspots.append({
                    "task_id": str(task_id),
                    "task_name": task_name,
                    "success_probability": prob,
                    "confidence_interval": self.task_model.get_confidence_interval(task_id),
                    "is_on_critical_path": True
                })
        
        # Also check non-critical-path tasks with high dependencies
        for task_id, task in self.task_graph.tasks.items():
            if task_id in critical_path:
                continue  # Already checked
                
            # Check if this task has many dependents
            if len(task.dependents) >= 2:
                prob = self.task_model.get_task_probability(task_id)
                
                if prob < threshold:
                    hotspots.append({
                        "task_id": str(task_id),
                        "task_name": task.title,
                        "success_probability": prob,
                        "confidence_interval": self.task_model.get_confidence_interval(task_id),
                        "is_on_critical_path": False,
                        "dependent_count": len(task.dependents)
                    })
        
        # Sort by probability (ascending)
        return sorted(hotspots, key=lambda x: x["success_probability"])
    
    def monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to estimate project completion.
        
        Args:
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with simulation results
        """
        completion_times = []
        success_counts = 0
        task_failures = {}
        
        # Get all planned tasks
        planned_tasks = [
            task_id for task_id, task in self.task_graph.tasks.items()
            if task.status == TaskStatus.PLANNED
        ]
        
        # Track the critical path
        critical_path = self._get_critical_path()
        
        for _ in range(num_simulations):
            # Reset for this simulation
            completion_time = 0
            all_succeeded = True
            completed_tasks = set()
            
            # Process tasks in topological order
            remaining_tasks = planned_tasks.copy()
            
            while remaining_tasks:
                # Find tasks that can be executed
                executable = []
                for task_id in remaining_tasks:
                    task = self.task_graph.tasks[task_id]
                    if all(dep_id in completed_tasks for dep_id in task.dependencies):
                        executable.append(task_id)
                
                if not executable:
                    # Deadlock - cannot complete all tasks
                    all_succeeded = False
                    break
                
                # Process executable tasks in parallel
                max_duration = 0
                for task_id in executable:
                    # Sample outcome
                    outcomes = self.task_model.sample_sequence_outcome([task_id])
                    if not outcomes:
                        continue
                        
                    _, result = outcomes[0]
                    
                    if result == SimulationResult.FAILURE:
                        # Task failed
                        all_succeeded = False
                        task_failures[task_id] = task_failures.get(task_id, 0) + 1
                    else:
                        # Task succeeded (fully or partially)
                        completed_tasks.add(task_id)
                        
                        # Add to completion time
                        task = self.task_graph.tasks[task_id]
                        task_duration = task.estimated_duration_hours
                        
                        # Partial success takes longer
                        if result == SimulationResult.PARTIAL_SUCCESS:
                            task_duration *= 1.5
                            
                        max_duration = max(max_duration, task_duration)
                
                # Update total completion time
                completion_time += max_duration
                
                # Remove processed tasks
                for task_id in executable:
                    if task_id in remaining_tasks:
                        remaining_tasks.remove(task_id)
            
            # Record results
            if all_succeeded:
                success_counts += 1
                completion_times.append(completion_time)
        
        # Calculate statistics
        success_rate = success_counts / num_simulations
        
        # Process completion times
        if completion_times:
            completion_times.sort()
            mean_time = sum(completion_times) / len(completion_times)
            p10 = completion_times[int(0.1 * len(completion_times))]
            p50 = completion_times[int(0.5 * len(completion_times))]
            p90 = completion_times[int(0.9 * len(completion_times))]
        else:
            mean_time = p10 = p50 = p90 = 0
        
        # Get failure points
        failure_points = [
            {
                "task_id": str(task_id),
                "task_name": self.task_graph.tasks[task_id].title if task_id in self.task_graph.tasks else "Unknown",
                "failure_count": count,
                "failure_rate": count / num_simulations,
                "is_on_critical_path": task_id in critical_path
            }
            for task_id, count in sorted(task_failures.items(), key=lambda x: x[1], reverse=True)
            if count > 0
        ]
        
        return {
            "simulations": num_simulations,
            "success_rate": success_rate,
            "completion_time": {
                "mean": mean_time,
                "p10": p10,
                "p50": p50,
                "p90": p90
            },
            "failure_points": failure_points[:10]  # Top 10 failure points
        }