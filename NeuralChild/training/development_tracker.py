# development_tracker.py - Logs milestones and growth metrics
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator

from language.developmental_stages import LanguageDevelopmentStage
from neural_child import HumanMindDevelopmentStage
from utils.logging_utils import log_development_milestone

logger = logging.getLogger("DevelopmentTracker")

class DevelopmentMilestone(BaseModel):
    """A specific developmental milestone"""
    milestone_id: str
    name: str
    description: str
    category: str
    subcategory: Optional[str] = None
    age_range: Tuple[float, float]  # Expected age range in days (min, max)
    importance: int = Field(1, ge=1, le=3)  # 1=minor, 2=moderate, 3=major
    prerequisites: List[str] = Field(default_factory=list)
    triggers: Dict[str, Any] = Field(default_factory=dict)
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    
    @field_validator('age_range')
    @classmethod
    def validate_age_range(cls, v):
        """Ensure age_range is valid"""
        if not isinstance(v, tuple) or len(v) != 2:
            raise ValueError("age_range must be a tuple of (min, max)")
        if v[0] < 0 or v[1] < v[0]:
            raise ValueError("age_range must be (min, max) with min >= 0 and max >= min")
        return v

class DevelopmentMetric(BaseModel):
    """A tracked developmental metric"""
    name: str
    description: str
    category: str
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=datetime.now)
    age_days: float
    expected_range: Tuple[float, float]  # Expected range for this age (min, max)
    percentile: Optional[float] = None

class DevelopmentTracker:
    """Tracks and logs developmental milestones and growth metrics"""
    
    def __init__(self, data_dir: Path = Path("./data/training")):
        """Initialize development tracker
        
        Args:
            data_dir: Directory for storing development data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load milestone definitions
        self.milestones: Dict[str, DevelopmentMilestone] = self._load_milestone_definitions()
        
        # Initialize tracking data
        self.achieved_milestones: Dict[str, DevelopmentMilestone] = {}
        self.recent_milestones: List[DevelopmentMilestone] = []
        self.metrics_history: Dict[str, List[DevelopmentMetric]] = {}
        self.current_stage = HumanMindDevelopmentStage.PRENATAL
        
        # Load previous state if exists
        self._load_state()
        
        logger.info(f"Development tracker initialized with {len(self.milestones)} milestones")
    
    def _load_milestone_definitions(self) -> Dict[str, DevelopmentMilestone]:
        """Load milestone definitions from file
        
        Returns:
            Dictionary of milestone definitions
        """
        milestones = {}
        definitions_file = self.data_dir / "milestone_definitions.json"
        
        if definitions_file.exists():
            try:
                with open(definitions_file, 'r') as f:
                    raw_milestones = json.load(f)
                
                for milestone_data in raw_milestones:
                    milestone = DevelopmentMilestone(**milestone_data)
                    milestones[milestone.milestone_id] = milestone
                
            except Exception as e:
                logger.error(f"Error loading milestone definitions: {str(e)}")
                # Fall back to default milestones
                milestones = self._create_default_milestones()
        else:
            # Generate default milestones
            milestones = self._create_default_milestones()
            
            # Save default definitions
            self._save_milestone_definitions(milestones)
        
        return milestones
    
    def _create_default_milestones(self) -> Dict[str, DevelopmentMilestone]:
        """Create default milestone definitions
        
        Returns:
            Dictionary of default milestone definitions
        """
        milestones = {}
        
        # Language milestones
        language_milestones = [
            {
                "milestone_id": "lang_babbling",
                "name": "First Babbling",
                "description": "First meaningful babbling sounds",
                "category": "language",
                "subcategory": "vocalization",
                "age_range": (1.0, 5.0),
                "importance": 2,
                "prerequisites": []
            },
            {
                "milestone_id": "lang_first_word",
                "name": "First Word",
                "description": "First recognizable word",
                "category": "language",
                "subcategory": "vocabulary",
                "age_range": (5.0, 15.0),
                "importance": 3,
                "prerequisites": ["lang_babbling"]
            },
            {
                "milestone_id": "lang_vocabulary_10",
                "name": "10-Word Vocabulary",
                "description": "Vocabulary reaches 10 words",
                "category": "language",
                "subcategory": "vocabulary",
                "age_range": (10.0, 25.0),
                "importance": 2,
                "prerequisites": ["lang_first_word"]
            },
            {
                "milestone_id": "lang_two_word",
                "name": "Two-Word Phrases",
                "description": "First two-word combinations",
                "category": "language",
                "subcategory": "syntax",
                "age_range": (15.0, 30.0),
                "importance": 3,
                "prerequisites": ["lang_vocabulary_10"]
            }
        ]
        
        # Cognitive milestones
        cognitive_milestones = [
            {
                "milestone_id": "cog_object_permanence",
                "name": "Object Permanence",
                "description": "Understanding that objects continue to exist when not observed",
                "category": "cognitive",
                "subcategory": "conceptual",
                "age_range": (5.0, 20.0),
                "importance": 3,
                "prerequisites": []
            },
            {
                "milestone_id": "cog_cause_effect",
                "name": "Cause and Effect",
                "description": "Understanding basic cause and effect relationships",
                "category": "cognitive",
                "subcategory": "reasoning",
                "age_range": (10.0, 25.0),
                "importance": 2,
                "prerequisites": []
            },
            {
                "milestone_id": "cog_categorization",
                "name": "Basic Categorization",
                "description": "Ability to categorize objects by simple properties",
                "category": "cognitive",
                "subcategory": "conceptual",
                "age_range": (20.0, 40.0),
                "importance": 2,
                "prerequisites": ["cog_object_permanence"]
            }
        ]
        
        # Emotional milestones
        emotional_milestones = [
            {
                "milestone_id": "emo_attachment",
                "name": "Primary Attachment",
                "description": "Formation of strong attachment to caregiver",
                "category": "emotional",
                "subcategory": "relationships",
                "age_range": (1.0, 10.0),
                "importance": 3,
                "prerequisites": []
            },
            {
                "milestone_id": "emo_regulation",
                "name": "Basic Emotional Regulation",
                "description": "Beginning to regulate emotional responses",
                "category": "emotional",
                "subcategory": "self-regulation",
                "age_range": (15.0, 35.0),
                "importance": 2,
                "prerequisites": ["emo_attachment"]
            },
            {
                "milestone_id": "emo_empathy",
                "name": "Early Empathy",
                "description": "First signs of empathetic response",
                "category": "emotional",
                "subcategory": "social",
                "age_range": (30.0, 60.0),
                "importance": 2,
                "prerequisites": ["emo_regulation"]
            }
        ]
        
        # Combine all milestones
        for milestone_data in language_milestones + cognitive_milestones + emotional_milestones:
            milestone = DevelopmentMilestone(**milestone_data)
            milestones[milestone.milestone_id] = milestone
        
        return milestones
    
    def _save_milestone_definitions(self, milestones: Dict[str, DevelopmentMilestone]) -> None:
        """Save milestone definitions to file
        
        Args:
            milestones: Dictionary of milestone definitions
        """
        definitions_file = self.data_dir / "milestone_definitions.json"
        
        try:
            with open(definitions_file, 'w') as f:
                json.dump([m.model_dump() for m in milestones.values()], f, indent=2, default=str)
            
            logger.info(f"Saved {len(milestones)} milestone definitions to {definitions_file}")
            
        except Exception as e:
            logger.error(f"Error saving milestone definitions: {str(e)}")
    
    def _load_state(self) -> None:
        """Load previous state from file"""
        state_file = self.data_dir / "development_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Load achieved milestones
                achieved = state.get("achieved_milestones", [])
                for milestone_data in achieved:
                    milestone_id = milestone_data.get("milestone_id")
                    if milestone_id in self.milestones:
                        # Update milestone with achieved status
                        milestone = self.milestones[milestone_id]
                        milestone.achieved = True
                        milestone.achieved_at = datetime.fromisoformat(milestone_data.get("achieved_at", datetime.now().isoformat()))
                        
                        # Add to achieved milestones
                        self.achieved_milestones[milestone_id] = milestone
                
                # Load metrics history
                metrics_history = state.get("metrics_history", {})
                for category, metrics in metrics_history.items():
                    self.metrics_history[category] = []
                    for metric_data in metrics:
                        # Convert timestamp string to datetime
                        if "timestamp" in metric_data:
                            metric_data["timestamp"] = datetime.fromisoformat(metric_data["timestamp"])
                        
                        metric = DevelopmentMetric(**metric_data)
                        self.metrics_history[category].append(metric)
                
                logger.info(f"Loaded development state: {len(self.achieved_milestones)} achieved milestones")
                
            except Exception as e:
                logger.error(f"Error loading development state: {str(e)}")
    
    def save_state(self) -> None:
        """Save current state to file"""
        state_file = self.data_dir / "development_state.json"
        
        try:
            # Prepare state data
            state = {
                "achieved_milestones": [
                    {
                        **milestone.model_dump(),
                        "achieved_at": milestone.achieved_at.isoformat() if milestone.achieved_at else None
                    }
                    for milestone in self.achieved_milestones.values()
                ],
                "metrics_history": {
                    category: [metric.model_dump() for metric in metrics]
                    for category, metrics in self.metrics_history.items()
                },
                "current_stage": self.current_stage.value,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Saved development state to {state_file}")
            
        except Exception as e:
            logger.error(f"Error saving development state: {str(e)}")
    
    def check_milestones(
        self, 
        age_days: float, 
        vocabulary_size: int,
        language_stage: LanguageDevelopmentStage,
        emotional_data: Dict[str, float],
        cognitive_data: Dict[str, Any]
    ) -> List[DevelopmentMilestone]:
        """Check for achieved milestones
        
        Args:
            age_days: Current age in days
            vocabulary_size: Current vocabulary size
            language_stage: Current language development stage
            emotional_data: Emotional state data
            cognitive_data: Cognitive capabilities data
            
        Returns:
            List of newly achieved milestones
        """
        newly_achieved = []
        
        # Update developmental stage
        self.current_stage = HumanMindDevelopmentStage.get_stage_for_age(age_days)
        
        # Check each milestone
        for milestone_id, milestone in self.milestones.items():
            # Skip already achieved milestones
            if milestone_id in self.achieved_milestones:
                continue
            
            # Check if age is in range
            if age_days < milestone.age_range[0]:
                continue  # Too young
            
            # Check prerequisites
            prerequisites_met = all(
                prereq in self.achieved_milestones
                for prereq in milestone.prerequisites
            )
            
            if not prerequisites_met:
                continue  # Prerequisites not met
            
            # Check specific triggers based on category
            achieved = False
            
            if milestone.category == "language":
                achieved = self._check_language_milestone(milestone, age_days, vocabulary_size, language_stage)
            elif milestone.category == "cognitive":
                achieved = self._check_cognitive_milestone(milestone, age_days, cognitive_data)
            elif milestone.category == "emotional":
                achieved = self._check_emotional_milestone(milestone, age_days, emotional_data)
            
            # If achieved, record it
            if achieved:
                milestone.achieved = True
                milestone.achieved_at = datetime.now()
                
                self.achieved_milestones[milestone_id] = milestone
                newly_achieved.append(milestone)
                
                # Log the milestone
                log_development_milestone(
                    milestone=milestone.name,
                    category=milestone.category,
                    importance=milestone.importance,
                    details={
                        "description": milestone.description,
                        "age_days": age_days,
                        "subcategory": milestone.subcategory
                    }
                )
                
                logger.info(f"Milestone achieved: {milestone.name} [{milestone.category}]")
        
        # Add newly achieved milestones to recent list
        self.recent_milestones.extend(newly_achieved)
        # Keep only recent 10
        self.recent_milestones = self.recent_milestones[-10:]
        
        # Save state if any new milestones were achieved
        if newly_achieved:
            self.save_state()
        
        return newly_achieved
    
    def _check_language_milestone(
        self, 
        milestone: DevelopmentMilestone, 
        age_days: float,
        vocabulary_size: int,
        language_stage: LanguageDevelopmentStage
    ) -> bool:
        """Check if a language milestone has been achieved
        
        Args:
            milestone: Milestone to check
            age_days: Current age in days
            vocabulary_size: Current vocabulary size
            language_stage: Current language development stage
            
        Returns:
            True if milestone achieved, False otherwise
        """
        if milestone.milestone_id == "lang_babbling":
            # Achieved when language stage progresses beyond pre-linguistic
            return language_stage != LanguageDevelopmentStage.PRE_LINGUISTIC
        
        elif milestone.milestone_id == "lang_first_word":
            # Achieved when vocabulary has at least one word
            return vocabulary_size >= 1
        
        elif milestone.milestone_id == "lang_vocabulary_10":
            # Achieved when vocabulary reaches 10 words
            return vocabulary_size >= 10
        
        elif milestone.milestone_id == "lang_two_word":
            # Achieved when language stage progresses to telegraphic or beyond
            stage_values = list(LanguageDevelopmentStage)
            return stage_values.index(language_stage) >= stage_values.index(LanguageDevelopmentStage.TELEGRAPHIC)
        
        return False
    
    def _check_cognitive_milestone(
        self, 
        milestone: DevelopmentMilestone, 
        age_days: float,
        cognitive_data: Dict[str, Any]
    ) -> bool:
        """Check if a cognitive milestone has been achieved
        
        Args:
            milestone: Milestone to check
            age_days: Current age in days
            cognitive_data: Cognitive capabilities data
            
        Returns:
            True if milestone achieved, False otherwise
        """
        # Simple age-based check for now, could be made more sophisticated
        # For cognitive milestones, we check if age is at least midpoint of range
        midpoint = (milestone.age_range[0] + milestone.age_range[1]) / 2
        
        if age_days >= midpoint:
            # For specific milestones, check cognitive_data
            if milestone.milestone_id == "cog_object_permanence":
                return cognitive_data.get("object_permanence", 0.0) > 0.5
                
            elif milestone.milestone_id == "cog_cause_effect":
                return cognitive_data.get("cause_effect_understanding", 0.0) > 0.5
                
            elif milestone.milestone_id == "cog_categorization":
                return cognitive_data.get("categorization_ability", 0.0) > 0.4
                
            # Default to age-based achievement
            return True
            
        return False
    
    def _check_emotional_milestone(
        self, 
        milestone: DevelopmentMilestone, 
        age_days: float,
        emotional_data: Dict[str, float]
    ) -> bool:
        """Check if an emotional milestone has been achieved
        
        Args:
            milestone: Milestone to check
            age_days: Current age in days
            emotional_data: Emotional state data
            
        Returns:
            True if milestone achieved, False otherwise
        """
        # Check specific emotional milestones
        if milestone.milestone_id == "emo_attachment":
            # Check attachment based on trust level
            return emotional_data.get("trust", 0.0) > 0.6
            
        elif milestone.milestone_id == "emo_regulation":
            # Measure of emotional stability/volatility
            return emotional_data.get("emotional_stability", 0.0) > 0.4
            
        elif milestone.milestone_id == "emo_empathy":
            # Check empathy development
            return (
                emotional_data.get("empathy", 0.0) > 0.3 or
                (age_days >= milestone.age_range[1] and age_days > 40)
            )
        
        # Default to age-based achievement
        midpoint = (milestone.age_range[0] + milestone.age_range[1]) / 2
        return age_days >= midpoint
    
    def record_metric(
        self, 
        name: str,
        category: str,
        value: float,
        unit: str,
        age_days: float,
        expected_range: Optional[Tuple[float, float]] = None,
        description: Optional[str] = None
    ) -> None:
        """Record a developmental metric
        
        Args:
            name: Metric name
            category: Metric category
            value: Metric value
            unit: Metric unit
            age_days: Age in days
            expected_range: Expected range for this age (min, max)
            description: Metric description
        """
        # Use default expected range if not provided
        if expected_range is None:
            expected_range = (0.0, 1.0)
        
        # Create metric object
        metric = DevelopmentMetric(
            name=name,
            description=description or name,
            category=category,
            value=value,
            unit=unit,
            age_days=age_days,
            expected_range=expected_range
        )
        
        # Calculate percentile
        min_val, max_val = expected_range
        if max_val > min_val:
            metric.percentile = (value - min_val) / (max_val - min_val) * 100
        
        # Add to history
        if category not in self.metrics_history:
            self.metrics_history[category] = []
            
        self.metrics_history[category].append(metric)
        
        # Save state periodically
        if sum(len(metrics) for metrics in self.metrics_history.values()) % 50 == 0:
            self.save_state()
            
        logger.debug(f"Recorded metric: {name} = {value} {unit} [{category}]")
    
    def get_development_report(self) -> Dict[str, Any]:
        """Generate a comprehensive development report
        
        Returns:
            Development report data
        """
        # Count milestones by category
        milestone_counts = {}
        for milestone in self.milestones.values():
            category = milestone.category
            if category not in milestone_counts:
                milestone_counts[category] = {"total": 0, "achieved": 0}
                
            milestone_counts[category]["total"] += 1
            if milestone.milestone_id in self.achieved_milestones:
                milestone_counts[category]["achieved"] += 1
        
        # Calculate development percentages by category
        development_percentages = {}
        for category, counts in milestone_counts.items():
            if counts["total"] > 0:
                development_percentages[category] = (counts["achieved"] / counts["total"]) * 100
            else:
                development_percentages[category] = 0
        
        # Get recent metrics by category
        recent_metrics = {}
        for category, metrics in self.metrics_history.items():
            if metrics:
                # Get most recent for each unique name
                metric_by_name = {}
                for metric in metrics:
                    if metric.name not in metric_by_name or metric.timestamp > metric_by_name[metric.name].timestamp:
                        metric_by_name[metric.name] = metric
                
                recent_metrics[category] = list(metric_by_name.values())
        
        return {
            "milestone_counts": milestone_counts,
            "development_percentages": development_percentages,
            "recent_metrics": recent_metrics,
            "recent_milestones": [m.model_dump() for m in self.recent_milestones],
            "current_stage": self.current_stage.value,
            "achieved_milestone_count": len(self.achieved_milestones),
            "total_milestone_count": len(self.milestones)
        }
    
    def get_growth_chart_data(self, category: str, metric_name: str) -> Dict[str, Any]:
        """Get data for a growth chart
        
        Args:
            category: Metric category
            metric_name: Metric name
            
        Returns:
            Growth chart data
        """
        if category not in self.metrics_history:
            return {"ages": [], "values": [], "expected_mins": [], "expected_maxs": []}
        
        # Filter metrics by name
        metrics = [m for m in self.metrics_history[category] if m.name == metric_name]
        
        if not metrics:
            return {"ages": [], "values": [], "expected_mins": [], "expected_maxs": []}
        
        # Sort by age
        metrics.sort(key=lambda m: m.age_days)
        
        # Extract data
        ages = [m.age_days for m in metrics]
        values = [m.value for m in metrics]
        expected_mins = [m.expected_range[0] for m in metrics]
        expected_maxs = [m.expected_range[1] for m in metrics]
        
        return {
            "ages": ages,
            "values": values,
            "expected_mins": expected_mins,
            "expected_maxs": expected_maxs,
            "unit": metrics[0].unit if metrics else ""
        }
    
    def export_metrics_to_csv(self, output_file: Optional[Path] = None) -> Path:
        """Export metrics history to CSV
        
        Args:
            output_file: Output file path (default: metrics_history.csv in data directory)
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.data_dir / "metrics_history.csv"
        
        # Flatten metrics into a list
        metrics_list = []
        for category, metrics in self.metrics_history.items():
            for metric in metrics:
                metrics_list.append({
                    "timestamp": metric.timestamp,
                    "category": category,
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "age_days": metric.age_days,
                    "percentile": metric.percentile
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(metrics_list)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(metrics_list)} metrics to {output_file}")
        return output_file
    
    def export_milestones_to_csv(self, output_file: Optional[Path] = None) -> Path:
        """Export milestone data to CSV
        
        Args:
            output_file: Output file path (default: milestones.csv in data directory)
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.data_dir / "milestones.csv"
        
        # Prepare milestone data
        milestone_data = []
        for milestone in self.milestones.values():
            milestone_data.append({
                "milestone_id": milestone.milestone_id,
                "name": milestone.name,
                "category": milestone.category,
                "subcategory": milestone.subcategory,
                "expected_min_age": milestone.age_range[0],
                "expected_max_age": milestone.age_range[1],
                "achieved": milestone.achieved,
                "achieved_at": milestone.achieved_at,
                "importance": milestone.importance
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(milestone_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(milestone_data)} milestones to {output_file}")
        return output_file