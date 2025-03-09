from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum

class DevelopmentalStage(str, Enum):
    """Developmental stages of the mind model"""
    PRENATAL = "prenatal"
    INFANCY = "infancy"
    EARLY_CHILDHOOD = "early_childhood"
    MIDDLE_CHILDHOOD = "middle_childhood"
    ADOLESCENCE = "adolescence"
    ADULTHOOD = "adulthood"

class MetricCategory(str, Enum):
    """Categories for metrics collected about the system"""
    LANGUAGE = "language"
    MEMORY = "memory"
    EMOTION = "emotion"
    CONSCIOUSNESS = "consciousness"
    ATTENTION = "attention"
    EXECUTIVE = "executive"
    SOCIAL = "social"
    MOTIVATION = "motivation"
    TEMPORAL = "temporal"
    CREATIVITY = "creativity"
    IDENTITY = "identity"
    LEARNING = "learning"
    SYSTEM = "system"
    INTEGRATION = "integration"
    GENERAL = "general"

class ResearchMetrics(BaseModel):
    """Research metrics for tracking development"""
    category: Union[MetricCategory, str]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: Optional[DevelopmentalStage] = None
    session_id: Optional[str] = None

class CognitiveModuleState(BaseModel):
    """State information about a specific cognitive module"""
    module_name: str
    active: bool = True
    activation_level: float = Field(ge=0.0, le=1.0, default=0.0)
    last_update: datetime = Field(default_factory=datetime.now)
    internal_state: Dict[str, Any] = Field(default_factory=dict)
    connections: Dict[str, float] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    developmental_metrics: Dict[str, float] = Field(default_factory=dict)

class DevelopmentalMilestone(BaseModel):
    """Record of a developmental milestone achievement"""
    milestone_id: str
    module: str
    name: str
    description: str
    achieved: bool = False
    timestamp: Optional[datetime] = None
    developmental_stage: DevelopmentalStage
    difficulty: float = Field(ge=0.0, le=1.0, default=0.5)
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    prerequisites: List[str] = Field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = Field(default_factory=dict)

class DevelopmentalEvent(BaseModel):
    """Record of a significant developmental event"""
    event_id: str
    event_type: str
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    module: Optional[str] = None
    developmental_stage: Optional[DevelopmentalStage] = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    related_milestones: List[str] = Field(default_factory=list)
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class LearningAnalysis(BaseModel):
    """Analysis of a learning episode or pattern"""
    analysis_id: str
    module: str
    learning_type: str
    period_start: datetime
    period_end: datetime
    duration_seconds: float
    improvement_metrics: Dict[str, float] = Field(default_factory=dict)
    learning_rate: float = Field(ge=0.0, default=0.0)
    plateau_detected: bool = False
    efficiency: float = Field(ge=0.0, le=1.0, default=0.5)
    correlated_experiences: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class NeuralActivitySnapshot(BaseModel):
    """Snapshot of neural activity patterns in a module"""
    module: str
    timestamp: datetime = Field(default_factory=datetime.now)
    pattern_type: str
    activity_level: float = Field(ge=0.0, le=1.0)
    activation_map: Dict[str, float] = Field(default_factory=dict)
    context: Optional[str] = None
    duration_ms: float
    related_stimulus: Optional[str] = None
    
class DevelopmentalTrajectory(BaseModel):
    """Analysis of developmental trajectory over time"""
    module: str
    metric: str
    timeframe_start: datetime
    timeframe_end: datetime
    data_points: List[Dict[str, Any]]
    trend_type: str
    trend_strength: float = Field(ge=0.0, le=1.0)
    growth_rate: float
    plateaus: List[Dict[str, Any]] = Field(default_factory=list)
    milestones_achieved: List[str] = Field(default_factory=list)
    predicted_trajectory: Optional[List[Dict[str, Any]]] = None

class VisualizationRequest(BaseModel):
    """Request for visualization of specific aspect of development"""
    visualization_type: str
    module: Optional[str] = None
    metric: Optional[str] = None
    timeframe_start: Optional[datetime] = None
    timeframe_end: Optional[datetime] = None
    comparison_modules: List[str] = Field(default_factory=list)
    comparison_metrics: List[str] = Field(default_factory=list)
    aggregation_level: str = "day"
    additional_parameters: Dict[str, Any] = Field(default_factory=dict)

class SystemStateSnapshot(BaseModel):
    """Complete snapshot of the system state at a point in time"""
    snapshot_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_stage: DevelopmentalStage
    global_metrics: Dict[str, Any] = Field(default_factory=dict)
    module_states: Dict[str, CognitiveModuleState] = Field(default_factory=dict)
    active_processes: List[str] = Field(default_factory=list)
    system_load: Dict[str, float] = Field(default_factory=dict)
    recent_experiences: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
