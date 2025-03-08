from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Tuple, Optional, Set, Literal, Union
from datetime import datetime, timedelta
import uuid

class DevelopmentalStage(BaseModel):
    """
    Represents a specific developmental stage in the mind's growth
    
    Each stage has distinct capabilities and expectations for various modules
    """
    name: str
    age_range: Tuple[float, float]  # in simulated age units
    capabilities: Dict[str, float] = Field(default_factory=dict)  # module -> capability level (0.0-1.0)
    prerequisites: Dict[str, float] = Field(default_factory=dict)  # capabilities required to enter this stage
    description: str = ""
    is_active: bool = False
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Key milestones that should be achieved in this stage
    expected_milestones: List[str] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class CriticalPeriod(BaseModel):
    """
    Represents a critical/sensitive period for development of a specific capability
    
    During critical periods, certain capabilities develop more rapidly and
    with greater plasticity, but may be permanently limited if not developed
    during this window.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    capability: str  # What capability this period affects
    age_range: Tuple[float, float]  # When this period occurs
    plasticity_multiplier: float = Field(default=2.0, ge=1.0)  # How much faster development happens
    status: Literal["pending", "active", "completed", "missed"] = "pending"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)  # How critical (vs. merely sensitive)
    module_targets: List[str] = Field(default_factory=list)  # Which modules are affected
    
    # Whether missing this period results in permanent limitation
    causes_permanent_limitation: bool = False
    # How much capability is permanently limited if missed (0.0-1.0, with 1.0 meaning no limitation)
    limitation_factor: float = Field(default=0.8, ge=0.0, le=1.0)  
    
    # Specific experiences that should happen during this period
    recommended_experiences: List[str] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class Milestone(BaseModel):
    """
    A specific developmental milestone that should be achieved
    
    Milestones represent significant achievements in cognitive, emotional,
    or social development that indicate healthy progression.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    category: str  # cognitive, emotional, social, language, etc.
    typical_age: float  # when this is typically achieved
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    prerequisite_milestones: List[str] = Field(default_factory=list)  # IDs of prerequisites
    associated_stage: str  # Which developmental stage this belongs to
    
    # The specific capabilities required to achieve this milestone
    capability_requirements: Dict[str, float] = Field(default_factory=dict)
    
    # Progress towards achieving this milestone (0.0-1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Whether this milestone is considered essential for healthy development
    is_essential: bool = True
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def update_progress(self, new_progress: float) -> None:
        """Update the progress towards this milestone"""
        self.progress = min(1.0, max(0.0, new_progress))
        if self.progress >= 1.0 and not self.achieved:
            self.achieved = True
            self.achieved_at = datetime.now()

class DevelopmentalTrajectory(BaseModel):
    """
    Tracks overall developmental progression over time
    
    This model maintains the history of development and current trajectory,
    allowing for comparison to typical/expected development patterns.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=datetime.now)
    current_age: float = 0.0
    
    # Current and historical stages
    current_stage: str = "prenatal"
    stage_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Module-specific development levels
    module_development: Dict[str, float] = Field(default_factory=dict)
    
    # Milestone tracking
    achieved_milestones: List[str] = Field(default_factory=list)
    pending_milestones: List[str] = Field(default_factory=list)
    
    # Development velocity (how fast development is occurring)
    development_velocity: Dict[str, float] = Field(default_factory=dict)
    
    # How development compares to typical trajectory (-1.0 to 1.0, with 0 being typical)
    # Negative values indicate delayed development, positive values advanced development
    developmental_offsets: Dict[str, float] = Field(default_factory=dict)
    
    # Whether development appears balanced across domains
    is_balanced: bool = True
    
    # Areas of concern or special strength
    areas_of_concern: List[str] = Field(default_factory=list)
    areas_of_strength: List[str] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def add_stage_transition(self, from_stage: str, to_stage: str) -> None:
        """Record a developmental stage transition"""
        self.stage_history.append({
            "from": from_stage,
            "to": to_stage,
            "time": datetime.now(),
            "age": self.current_age
        })
        self.current_stage = to_stage

class GrowthRateParameters(BaseModel):
    """Parameters that control the rate of development"""
    base_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    # How much natural variation occurs in development rate
    variability: float = Field(default=0.2, ge=0.0, le=1.0)
    # Factors that can accelerate development
    acceleration_factors: Dict[str, float] = Field(default_factory=dict)
    # Factors that can slow development
    inhibition_factors: Dict[str, float] = Field(default_factory=dict)
    # How development rate changes with age
    age_modifiers: List[Tuple[float, float]] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class DevelopmentalEvent(BaseModel):
    """
    Represents a significant event in development
    
    These can be milestones, transitions, or critical learning experiences
    that shape the developmental trajectory.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["milestone", "stage_transition", "critical_period", "regression", "growth_spurt"] 
    description: str
    age: float
    affected_modules: List[str] = Field(default_factory=list)
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Additional event-specific data
    details: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
