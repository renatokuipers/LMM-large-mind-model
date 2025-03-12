"""Bayesian models for task completion probability."""
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from uuid import UUID, uuid4
from datetime import datetime
import logging
import json
import numpy as np
from pathlib import Path

from .models.task_models import Task, TaskGraph, TaskType, RiskLevel, TaskStatus
from .utils.fs_utils import get_workspace_subdirectory, save_json, load_json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("probability_modeling")


class PriorDistribution:
    """Represents a prior probability distribution for a task."""
    
    def __init__(
        self,
        mean: float,
        variance: float,
        alpha: float = None,
        beta: float = None
    ):
        """
        Initialize a prior distribution.
        
        Args:
            mean: Mean of the distribution
            variance: Variance of the distribution
            alpha: Alpha parameter for Beta distribution
            beta: Beta parameter for Beta distribution
        """
        self.mean = mean
        self.variance = variance
        
        # If alpha and beta aren't provided, calculate them from mean and variance
        if alpha is None or beta is None:
            self.alpha, self.beta = self._calculate_alpha_beta(mean, variance)
        else:
            self.alpha = alpha
            self.beta = beta
    
    def _calculate_alpha_beta(self, mean: float, variance: float) -> Tuple[float, float]:
        """
        Calculate alpha and beta parameters for a Beta distribution.
        
        Args:
            mean: Mean of the distribution
            variance: Variance of the distribution
            
        Returns:
            Tuple of (alpha, beta)
        """
        # Ensure mean is within [0, 1]
        mean = max(0.01, min(0.99, mean))
        
        # Ensure variance is valid for this mean
        max_variance = mean * (1 - mean)
        variance = min(variance, max_variance * 0.99)
        
        # Calculate alpha and beta
        t = mean * (1 - mean) / variance - 1
        alpha = mean * t
        beta = (1 - mean) * t
        
        return alpha, beta
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """
        Sample from the distribution.
        
        Args:
            n: Number of samples
            
        Returns:
            Sampled value(s)
        """
        samples = np.random.beta(self.alpha, self.beta, n)
        
        if n == 1:
            return float(samples[0])
        return [float(x) for x in samples]
    
    def update(self, successes: int, failures: int) -> 'PriorDistribution':
        """
        Update the distribution with new observations.
        
        Args:
            successes: Number of successful observations
            failures: Number of failed observations
            
        Returns:
            Updated distribution
        """
        new_alpha = self.alpha + successes
        new_beta = self.beta + failures
        
        new_mean = new_alpha / (new_alpha + new_beta)
        new_variance = (new_alpha * new_beta) / ((new_alpha + new_beta)**2 * (new_alpha + new_beta + 1))
        
        return PriorDistribution(
            mean=new_mean,
            variance=new_variance,
            alpha=new_alpha,
            beta=new_beta
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert the distribution to a dictionary."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "alpha": self.alpha,
            "beta": self.beta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PriorDistribution':
        """Create a distribution from a dictionary."""
        return cls(
            mean=data["mean"],
            variance=data["variance"],
            alpha=data["alpha"],
            beta=data["beta"]
        )


class TaskProbabilityModel:
    """Model for estimating task success probabilities."""
    
    def __init__(self, task_graph: Optional[TaskGraph] = None):
        """
        Initialize the task probability model.
        
        Args:
            task_graph: Graph of tasks and dependencies
        """
        self.task_graph = task_graph
        self.prior_distributions: Dict[UUID, PriorDistribution] = {}
        self.posterior_distributions: Dict[UUID, PriorDistribution] = {}
        self.observations: Dict[UUID, Tuple[int, int]] = {}  # (successes, failures)
        
        # Initialize the model
        if task_graph:
            self._initialize_priors()
    
    def _initialize_priors(self) -> None:
        """Initialize prior distributions for all tasks."""
        for task_id, task in self.task_graph.tasks.items():
            # Generate a prior based on task properties
            prior = self._generate_prior(task)
            self.prior_distributions[task_id] = prior
            self.posterior_distributions[task_id] = prior
            self.observations[task_id] = (0, 0)
    
    def _generate_prior(self, task: Task) -> PriorDistribution:
        """
        Generate a prior distribution for a task.
        
        Args:
            task: Task to generate prior for
            
        Returns:
            Prior distribution
        """
        # Start with a base mean from the task's success_probability
        base_mean = task.success_probability
        
        # Adjust based on risk level
        risk_adjustment = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.0,
            RiskLevel.HIGH: -0.1,
            RiskLevel.VERY_HIGH: -0.2
        }
        
        mean = base_mean + risk_adjustment.get(task.risk_level, 0.0)
        mean = max(0.1, min(0.9, mean))  # Ensure between 0.1 and 0.9
        
        # Set variance based on confidence in the estimate
        # Higher risk = higher uncertainty = higher variance
        variance_by_risk = {
            RiskLevel.LOW: 0.01,
            RiskLevel.MEDIUM: 0.02,
            RiskLevel.HIGH: 0.04,
            RiskLevel.VERY_HIGH: 0.06
        }
        
        variance = variance_by_risk.get(task.risk_level, 0.02)
        
        # Create and return the prior
        return PriorDistribution(mean=mean, variance=variance)
    
    def update_with_observation(self, task_id: UUID, success: bool) -> None:
        """
        Update the model with a task observation.
        
        Args:
            task_id: ID of the observed task
            success: Whether the task was successful
        """
        if task_id not in self.posterior_distributions:
            logger.warning(f"Task {task_id} not found in model")
            return
        
        # Update observation counts
        successes, failures = self.observations[task_id]
        if success:
            successes += 1
        else:
            failures += 1
        self.observations[task_id] = (successes, failures)
        
        # Update posterior distribution
        prior = self.prior_distributions[task_id]
        self.posterior_distributions[task_id] = prior.update(successes, failures)
    
    def get_task_probability(self, task_id: UUID, use_posterior: bool = True) -> float:
        """
        Get the estimated success probability for a task.
        
        Args:
            task_id: ID of the task
            use_posterior: Whether to use the posterior distribution
            
        Returns:
            Estimated success probability
        """
        if task_id not in self.posterior_distributions:
            if self.task_graph and task_id in self.task_graph.tasks:
                # Return the task's default success_probability
                return self.task_graph.tasks[task_id].success_probability
            return 0.5  # Default fallback
        
        if use_posterior:
            return self.posterior_distributions[task_id].mean
        return self.prior_distributions[task_id].mean
    
    def sample_task_probability(self, task_id: UUID, use_posterior: bool = True) -> float:
        """
        Sample a success probability for a task.
        
        Args:
            task_id: ID of the task
            use_posterior: Whether to use the posterior distribution
            
        Returns:
            Sampled success probability
        """
        if task_id not in self.posterior_distributions:
            if self.task_graph and task_id in self.task_graph.tasks:
                # Sample around the task's default success_probability
                mean = self.task_graph.tasks[task_id].success_probability
                return max(0.01, min(0.99, random.gauss(mean, 0.1)))
            return random.random()  # Default fallback
        
        if use_posterior:
            return self.posterior_distributions[task_id].sample()
        return self.prior_distributions[task_id].sample()
    
    def estimate_sequence_probability(
        self,
        task_sequence: List[UUID],
        sample_count: int = 1000,
        use_posterior: bool = True
    ) -> Tuple[float, float]:
        """
        Estimate the success probability for a sequence of tasks.
        
        Args:
            task_sequence: Sequence of task IDs
            sample_count: Number of samples to use for estimation
            use_posterior: Whether to use posterior distributions
            
        Returns:
            Tuple of (mean probability, standard deviation)
        """
        sequence_probabilities = []
        
        for _ in range(sample_count):
            sequence_prob = 1.0
            
            for task_id in task_sequence:
                task_prob = self.sample_task_probability(task_id, use_posterior)
                sequence_prob *= task_prob
            
            sequence_probabilities.append(sequence_prob)
        
        mean_prob = sum(sequence_probabilities) / sample_count
        std_dev = (sum((p - mean_prob) ** 2 for p in sequence_probabilities) / sample_count) ** 0.5
        
        return mean_prob, std_dev
    
    def save_model(self, file_path: Optional[Path] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            file_path: Path to save the model
            
        Returns:
            Path to the saved file
        """
        if file_path is None:
            simulations_dir = get_workspace_subdirectory('planning/simulations')
            file_path = simulations_dir / f"probability_model_{uuid4()}.json"
        
        # Convert model to dictionary
        model_data = {
            "prior_distributions": {
                str(task_id): prior.to_dict()
                for task_id, prior in self.prior_distributions.items()
            },
            "posterior_distributions": {
                str(task_id): posterior.to_dict()
                for task_id, posterior in self.posterior_distributions.items()
            },
            "observations": {
                str(task_id): list(obs)
                for task_id, obs in self.observations.items()
            }
        }
        
        # Save to disk
        return save_json(model_data, file_path)
    
    @classmethod
    def load_model(cls, file_path: Path, task_graph: Optional[TaskGraph] = None) -> 'TaskProbabilityModel':
        """
        Load a model from disk.
        
        Args:
            file_path: Path to the model file
            task_graph: Optional task graph
            
        Returns:
            Loaded model
        """
        model_data = load_json(file_path)
        
        if not model_data:
            logger.warning(f"Failed to load model from {file_path}")
            return cls(task_graph)
        
        model = cls(task_graph)
        
        # Load prior distributions
        model.prior_distributions = {
            UUID(task_id): PriorDistribution.from_dict(prior_data)
            for task_id, prior_data in model_data["prior_distributions"].items()
        }
        
        # Load posterior distributions
        model.posterior_distributions = {
            UUID(task_id): PriorDistribution.from_dict(posterior_data)
            for task_id, posterior_data in model_data["posterior_distributions"].items()
        }
        
        # Load observations
        model.observations = {
            UUID(task_id): tuple(obs)
            for task_id, obs in model_data["observations"].items()
        }
        
        return model


def calculate_task_success_probability(task: Task) -> float:
    """
    Calculate the success probability for a task based on its properties.
    
    Args:
        task: Task to calculate probability for
        
    Returns:
        Estimated success probability
    """
    # Start with the base success probability
    probability = task.success_probability
    
    # Adjust based on risk level
    risk_adjustment = {
        RiskLevel.LOW: 0.1,
        RiskLevel.MEDIUM: 0.0,
        RiskLevel.HIGH: -0.1,
        RiskLevel.VERY_HIGH: -0.2
    }
    
    probability += risk_adjustment.get(task.risk_level, 0.0)
    
    # Clamp to valid range
    probability = max(0.01, min(0.99, probability))
    
    return probability


def calculate_sequence_success_probability(
    task_graph: TaskGraph,
    task_sequence: List[UUID]
) -> float:
    """
    Calculate the success probability for a sequence of tasks.
    
    Args:
        task_graph: Graph of tasks
        task_sequence: Sequence of task IDs
        
    Returns:
        Sequence success probability
    """
    # Simple model: multiply individual probabilities
    probability = 1.0
    
    for task_id in task_sequence:
        if task_id not in task_graph.tasks:
            continue
        
        task = task_graph.tasks[task_id]
        task_probability = calculate_task_success_probability(task)
        probability *= task_probability
    
    return probability


def estimate_completion_time(
    task_graph: TaskGraph,
    task_sequence: List[UUID],
    uncertainty: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Estimate the completion time for a sequence of tasks.
    
    Args:
        task_graph: Graph of tasks
        task_sequence: Sequence of task IDs
        uncertainty: Whether to include uncertainty in the estimate
        
    Returns:
        Estimated completion time, or (mean, std_dev) if uncertainty is True
    """
    if not uncertainty:
        # Simple sum of estimated hours
        total_time = 0.0
        
        for task_id in task_sequence:
            if task_id in task_graph.tasks:
                total_time += task_graph.tasks[task_id].estimated_hours
        
        return total_time
    
    # With uncertainty: use Monte Carlo simulation
    samples = []
    sample_count = 1000
    
    for _ in range(sample_count):
        total_time = 0.0
        
        for task_id in task_sequence:
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                
                # Sample from a triangular distribution
                # Mode is the estimated hours, min is 50%, max is 150%
                min_time = task.estimated_hours * 0.5
                max_time = task.estimated_hours * 1.5
                mode_time = task.estimated_hours
                
                sampled_time = random.triangular(min_time, max_time, mode_time)
                total_time += sampled_time
        
        samples.append(total_time)
    
    # Calculate mean and standard deviation
    mean_time = sum(samples) / sample_count
    std_dev = (sum((s - mean_time) ** 2 for s in samples) / sample_count) ** 0.5
    
    return mean_time, std_dev


def calculate_risk_score(
    task_graph: TaskGraph,
    task_sequence: List[UUID]
) -> float:
    """
    Calculate a risk score for a sequence of tasks.
    
    Args:
        task_graph: Graph of tasks
        task_sequence: Sequence of task IDs
        
    Returns:
        Risk score (0.0 to 1.0)
    """
    risk_scores = {
        RiskLevel.LOW: 0.25,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.HIGH: 0.75,
        RiskLevel.VERY_HIGH: 1.0
    }
    
    total_risk = 0.0
    count = 0
    
    for task_id in task_sequence:
        if task_id in task_graph.tasks:
            task = task_graph.tasks[task_id]
            total_risk += risk_scores.get(task.risk_level, 0.5)
            count += 1
    
    return total_risk / count if count > 0 else 0.0