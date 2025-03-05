"""
Development component for the Neural Child's mind.

This module contains the implementation of the development component that handles
developmental stages and progression for the simulated mind.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import random

from utils.config import DevelopmentStageConfig, DEVELOPMENT_STAGES

class DevelopmentComponent:
    """Development component that handles developmental stages and progression."""
    
    def __init__(
        self,
        initial_age_months: float = 0.0,
        development_speed: float = 1.0,
        development_stages: List[DevelopmentStageConfig] = DEVELOPMENT_STAGES,
        name: str = "development_component"
    ):
        """Initialize the development component.
        
        Args:
            initial_age_months: Initial age in months
            development_speed: Speed of development (1.0 = real-time, 10.0 = 10x faster)
            development_stages: List of development stage configurations
            name: Name of the component
        """
        self.name = name
        self.age_months = initial_age_months
        self.development_speed = development_speed
        self.development_stages = development_stages
        
        # Current developmental stage
        self.current_stage = self._determine_stage(self.age_months)
        
        # Developmental milestones
        self.milestones_achieved: Dict[str, List[str]] = {
            "language": [],
            "emotional": [],
            "cognitive": [],
            "social": []
        }
        
        # Developmental metrics
        self.developmental_metrics: Dict[str, Dict[str, float]] = {
            "language": {
                "receptive_language": 0.0,
                "expressive_language": 0.0
            },
            "emotional": {
                "basic_emotions": 0.1,
                "emotional_regulation": 0.0,
                "emotional_complexity": 0.0
            },
            "cognitive": {
                "attention": 0.1,
                "memory": 0.1,
                "problem_solving": 0.0,
                "abstract_thinking": 0.0
            },
            "social": {
                "attachment": 0.1,
                "social_awareness": 0.0,
                "empathy": 0.0,
                "theory_of_mind": 0.0
            }
        }
        
        # Development history
        self.development_history: List[Dict[str, Any]] = []
        
        # Timing
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
    def update(self, mind_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the development component based on elapsed time and mind state.
        
        Args:
            mind_state: Current state of the Neural Child's mind
            
        Returns:
            Dictionary of development updates
        """
        # Calculate elapsed time
        current_time = time.time()
        elapsed_seconds = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Convert elapsed time to months based on development speed
        elapsed_months = (elapsed_seconds / (30 * 24 * 60 * 60)) * self.development_speed
        
        # Update age
        self.age_months += elapsed_months
        
        # Determine current stage
        previous_stage = self.current_stage
        self.current_stage = self._determine_stage(self.age_months)
        
        # Check if stage has changed
        stage_changed = previous_stage != self.current_stage
        
        # Update developmental metrics based on mind state
        self._update_developmental_metrics(mind_state)
        
        # Check for new milestones
        new_milestones = self._check_milestones()
        
        # Record development history
        self._record_development_history()
        
        # Prepare updates
        updates = {
            "age_months": self.age_months,
            "developmental_stage": self.current_stage,
            "stage_changed": stage_changed,
            "developmental_metrics": self.developmental_metrics,
            "new_milestones": new_milestones
        }
        
        return updates
    
    def _determine_stage(self, age_months: float) -> str:
        """Determine the developmental stage based on age.
        
        Args:
            age_months: Age in months
            
        Returns:
            String representing the developmental stage
        """
        for stage in self.development_stages:
            if stage.min_age_months <= age_months <= stage.max_age_months:
                return stage.name
        
        # If age is beyond the last stage, return the last stage
        return self.development_stages[-1].name
    
    def _update_developmental_metrics(self, mind_state: Dict[str, Any]):
        """Update developmental metrics based on mind state.
        
        Args:
            mind_state: Current state of the Neural Child's mind
        """
        # Extract metrics from mind state
        if "developmental_metrics" in mind_state:
            for category in self.developmental_metrics:
                if category in mind_state["developmental_metrics"]:
                    for metric in self.developmental_metrics[category]:
                        if metric in mind_state["developmental_metrics"][category]:
                            # Update metric with a weighted average (80% current, 20% new)
                            self.developmental_metrics[category][metric] = (
                                0.8 * self.developmental_metrics[category][metric] +
                                0.2 * mind_state["developmental_metrics"][category][metric]
                            )
        
        # Add some random variation to simulate natural development
        for category in self.developmental_metrics:
            for metric in self.developmental_metrics[category]:
                # Small random adjustment
                adjustment = random.uniform(-0.01, 0.01)
                self.developmental_metrics[category][metric] = max(
                    0.0, min(1.0, self.developmental_metrics[category][metric] + adjustment)
                )
                
                # Age-based progression
                age_factor = min(1.0, self.age_months / 360.0)  # Max out at 30 years
                self.developmental_metrics[category][metric] = max(
                    self.developmental_metrics[category][metric],
                    age_factor * 0.5  # Age alone can bring metrics up to 0.5
                )
    
    def _check_milestones(self) -> Dict[str, List[str]]:
        """Check for new developmental milestones.
        
        Returns:
            Dictionary of new milestones achieved
        """
        new_milestones: Dict[str, List[str]] = {
            "language": [],
            "emotional": [],
            "cognitive": [],
            "social": []
        }
        
        # Define milestones for each category and metric
        milestones = {
            "language": {
                "receptive_language": [
                    {"threshold": 0.2, "description": "Recognizes familiar voices"},
                    {"threshold": 0.4, "description": "Understands simple instructions"},
                    {"threshold": 0.6, "description": "Comprehends complex sentences"},
                    {"threshold": 0.8, "description": "Understands abstract language"}
                ],
                "expressive_language": [
                    {"threshold": 0.2, "description": "Babbling with intonation"},
                    {"threshold": 0.4, "description": "Uses single words"},
                    {"threshold": 0.6, "description": "Forms simple sentences"},
                    {"threshold": 0.8, "description": "Engages in complex conversations"}
                ]
            },
            "emotional": {
                "basic_emotions": [
                    {"threshold": 0.3, "description": "Expresses basic emotions"},
                    {"threshold": 0.6, "description": "Distinguishes between emotions"},
                    {"threshold": 0.9, "description": "Experiences complex emotions"}
                ],
                "emotional_regulation": [
                    {"threshold": 0.3, "description": "Beginning to self-soothe"},
                    {"threshold": 0.6, "description": "Can delay gratification"},
                    {"threshold": 0.9, "description": "Regulates emotions in difficult situations"}
                ],
                "emotional_complexity": [
                    {"threshold": 0.3, "description": "Experiences mixed emotions"},
                    {"threshold": 0.6, "description": "Understands emotional nuance"},
                    {"threshold": 0.9, "description": "Processes complex emotional scenarios"}
                ]
            },
            "cognitive": {
                "attention": [
                    {"threshold": 0.3, "description": "Focuses on interesting stimuli"},
                    {"threshold": 0.6, "description": "Sustains attention on tasks"},
                    {"threshold": 0.9, "description": "Manages divided attention effectively"}
                ],
                "memory": [
                    {"threshold": 0.3, "description": "Forms simple memories"},
                    {"threshold": 0.6, "description": "Recalls past experiences in detail"},
                    {"threshold": 0.9, "description": "Demonstrates excellent memory retrieval"}
                ],
                "problem_solving": [
                    {"threshold": 0.3, "description": "Solves simple problems"},
                    {"threshold": 0.6, "description": "Uses tools to achieve goals"},
                    {"threshold": 0.9, "description": "Approaches problems strategically"}
                ],
                "abstract_thinking": [
                    {"threshold": 0.3, "description": "Understands simple symbols"},
                    {"threshold": 0.6, "description": "Grasps metaphorical concepts"},
                    {"threshold": 0.9, "description": "Engages in complex abstract reasoning"}
                ]
            },
            "social": {
                "attachment": [
                    {"threshold": 0.3, "description": "Forms basic attachment"},
                    {"threshold": 0.6, "description": "Shows secure attachment behaviors"},
                    {"threshold": 0.9, "description": "Maintains healthy attachment patterns"}
                ],
                "social_awareness": [
                    {"threshold": 0.3, "description": "Recognizes social cues"},
                    {"threshold": 0.6, "description": "Understands social norms"},
                    {"threshold": 0.9, "description": "Navigates complex social situations"}
                ],
                "empathy": [
                    {"threshold": 0.3, "description": "Shows basic empathy"},
                    {"threshold": 0.6, "description": "Demonstrates concern for others"},
                    {"threshold": 0.9, "description": "Exhibits deep empathic understanding"}
                ],
                "theory_of_mind": [
                    {"threshold": 0.3, "description": "Recognizes others have different perspectives"},
                    {"threshold": 0.6, "description": "Understands false beliefs"},
                    {"threshold": 0.9, "description": "Comprehends complex mental states of others"}
                ]
            }
        }
        
        # Check each milestone
        for category in milestones:
            for metric in milestones[category]:
                current_value = self.developmental_metrics[category][metric]
                
                for milestone in milestones[category][metric]:
                    threshold = milestone["threshold"]
                    description = milestone["description"]
                    
                    # Check if milestone is achieved and not already recorded
                    if current_value >= threshold and description not in self.milestones_achieved[category]:
                        self.milestones_achieved[category].append(description)
                        new_milestones[category].append(description)
        
        return new_milestones
    
    def _record_development_history(self):
        """Record the current developmental state in history."""
        history_entry = {
            "timestamp": time.time(),
            "age_months": self.age_months,
            "developmental_stage": self.current_stage,
            "developmental_metrics": {
                category: dict(metrics) for category, metrics in self.developmental_metrics.items()
            },
            "milestones_achieved": {
                category: list(milestones) for category, milestones in self.milestones_achieved.items()
            }
        }
        
        self.development_history.append(history_entry)
        
        # Limit history size
        if len(self.development_history) > 1000:
            self.development_history = self.development_history[-1000:]
    
    def get_current_stage_config(self) -> Optional[DevelopmentStageConfig]:
        """Get the configuration for the current developmental stage.
        
        Returns:
            DevelopmentStageConfig for the current stage, or None if not found
        """
        for stage in self.development_stages:
            if stage.name == self.current_stage:
                return stage
        return None
    
    def get_development_progress(self) -> Dict[str, float]:
        """Get the overall development progress for each category.
        
        Returns:
            Dictionary of development progress (0.0 to 1.0) for each category
        """
        progress = {}
        
        for category, metrics in self.developmental_metrics.items():
            # Average the metrics for this category
            if metrics:
                progress[category] = sum(metrics.values()) / len(metrics)
            else:
                progress[category] = 0.0
        
        return progress
    
    def get_age_appropriate_expectations(self) -> Dict[str, Dict[str, float]]:
        """Get age-appropriate expectations for developmental metrics.
        
        Returns:
            Dictionary of expected values for each metric based on age
        """
        expectations = {
            "language": {},
            "emotional": {},
            "cognitive": {},
            "social": {}
        }
        
        # Get current stage config
        stage_config = self.get_current_stage_config()
        
        if stage_config:
            # Set expectations based on stage milestones
            for metric, value in stage_config.language_milestones.items():
                if metric in self.developmental_metrics["language"]:
                    expectations["language"][metric] = value
            
            for metric, value in stage_config.emotional_milestones.items():
                if metric in self.developmental_metrics["emotional"]:
                    expectations["emotional"][metric] = value
            
            for metric, value in stage_config.cognitive_milestones.items():
                if metric in self.developmental_metrics["cognitive"]:
                    expectations["cognitive"][metric] = value
            
            for metric, value in stage_config.social_milestones.items():
                if metric in self.developmental_metrics["social"]:
                    expectations["social"][metric] = value
        
        return expectations
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save state
        state = {
            "name": self.name,
            "age_months": self.age_months,
            "development_speed": self.development_speed,
            "current_stage": self.current_stage,
            "milestones_achieved": self.milestones_achieved,
            "developmental_metrics": self.developmental_metrics,
            "development_history": self.development_history,
            "start_time": self.start_time,
            "last_update_time": self.last_update_time
        }
        
        # Save state
        state_path = directory / f"{self.name}_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Load state
        state_path = directory / f"{self.name}_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
                self.name = state["name"]
                self.age_months = state["age_months"]
                self.development_speed = state["development_speed"]
                self.current_stage = state["current_stage"]
                self.milestones_achieved = state["milestones_achieved"]
                self.developmental_metrics = state["developmental_metrics"]
                self.development_history = state["development_history"]
                self.start_time = state["start_time"]
                self.last_update_time = state["last_update_time"] 