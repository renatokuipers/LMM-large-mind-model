"""
Pydantic models for the development module of the LMM project.

This module defines the data structures for developmental stages, milestones,
critical periods, and growth rate parameters that form the foundation for
modeling cognitive development in the LMM system.
"""
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union, Any

from pydantic import BaseModel, Field, field_validator, model_validator

class DevelopmentalStage(str, Enum):
    """Enumeration of developmental stages in the LMM system."""
    PRENATAL = "prenatal"
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"

class StageRange(BaseModel):
    """Age range for a developmental stage."""
    min_age: float = Field(..., ge=0.0, description="Minimum age in age units")
    max_age: Optional[float] = Field(None, description="Maximum age in age units, None for open-ended")
    
    @field_validator('max_age')
    @classmethod
    def max_age_greater_than_min(cls, v, info):
        if v is not None and v <= info.data.get('min_age', 0):
            raise ValueError(f"max_age must be greater than min_age")
        return v

class StageDefinition(BaseModel):
    """Definition of a developmental stage with its characteristics."""
    stage: DevelopmentalStage
    range: StageRange
    description: str
    learning_rate_multiplier: float = Field(1.0, gt=0.0, description="Multiplier for learning rates during this stage")
    key_capabilities: List[str] = Field(default_factory=list)
    neural_characteristics: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "forbid"}

class MilestoneStatus(str, Enum):
    """Status of a developmental milestone."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

class MilestoneDefinition(BaseModel):
    """Definition of a developmental milestone."""
    id: str
    name: str
    description: str
    typical_stage: DevelopmentalStage
    typical_age: float = Field(..., ge=0.0)
    prerequisite_milestones: List[str] = Field(default_factory=list)
    module_dependencies: List[str] = Field(default_factory=list)
    evaluation_criteria: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(1.0, ge=0.0, le=2.0, description="Importance factor (0.0-2.0)")
    
    model_config = {"extra": "forbid"}

class MilestoneRecord(BaseModel):
    """Record of a milestone's status and achievement details."""
    milestone_id: str
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)
    progress: float = Field(0.0, ge=0.0, le=1.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_age: Optional[float] = None
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode='after')
    def validate_dates_and_progress(self) -> 'MilestoneRecord':
        """Validate dates and progress consistency."""
        if self.status == MilestoneStatus.IN_PROGRESS and self.started_at is None:
            self.started_at = datetime.now()
        
        if self.status == MilestoneStatus.COMPLETED:
            if self.progress != 1.0:
                self.progress = 1.0
            if self.completed_at is None:
                self.completed_at = datetime.now()
                
        if self.status == MilestoneStatus.NOT_STARTED and self.progress > 0.0:
            self.status = MilestoneStatus.IN_PROGRESS
        
        if self.progress == 1.0 and self.status != MilestoneStatus.COMPLETED and self.status != MilestoneStatus.SKIPPED:
            self.status = MilestoneStatus.COMPLETED
            self.completed_at = self.completed_at or datetime.now()
            
        return self

class CriticalPeriodType(str, Enum):
    """Types of critical periods in development."""
    LANGUAGE = "language"
    MOTOR = "motor"
    SENSORY = "sensory"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    MEMORY = "memory"
    ATTENTION = "attention"
    
class CriticalPeriodDefinition(BaseModel):
    """Definition of a critical period for enhanced learning."""
    id: str
    name: str
    period_type: CriticalPeriodType
    description: str
    begin_age: float = Field(..., ge=0.0)
    end_age: float = Field(..., gt=0.0)
    learning_multiplier: float = Field(2.0, gt=1.0)
    affected_modules: List[str]
    affected_capabilities: List[str]
    
    model_config = {"extra": "forbid"}
    
    @field_validator('end_age')
    @classmethod
    def end_after_begin(cls, v, info):
        if v <= info.data.get('begin_age', 0):
            raise ValueError(f"end_age must be greater than begin_age")
        return v

class GrowthRateModel(BaseModel):
    """Model for controlling growth rate in different aspects of development."""
    base_rate: float = Field(1.0, gt=0.0, description="Base growth rate units per time unit")
    stage_multipliers: Dict[DevelopmentalStage, float] = Field(default_factory=dict)
    critical_period_boost: float = Field(1.5, gt=1.0)
    practice_effect: float = Field(1.2, gt=1.0, description="Multiplier for actively used capabilities")
    plateau_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Threshold at which growth slows")
    plateau_factor: float = Field(0.5, gt=0.0, lt=1.0, description="Factor for growth slowdown at plateau")
    
    model_config = {"extra": "forbid"}
    
    @field_validator('stage_multipliers')
    @classmethod
    def validate_stage_multipliers(cls, v):
        for stage in DevelopmentalStage:
            if stage not in v:
                v[stage] = 1.0
        return v

class DevelopmentConfig(BaseModel):
    """Configuration for the development module."""
    initial_age: float = Field(0.0, ge=0.0)
    time_acceleration: float = Field(1000.0, gt=0.0, description="How much faster than real-time development occurs")
    stage_definitions: List[StageDefinition]
    milestone_definitions: List[MilestoneDefinition]
    critical_period_definitions: List[CriticalPeriodDefinition]
    growth_rate_model: GrowthRateModel = Field(default_factory=GrowthRateModel)
    enable_variability: bool = Field(True, description="Enable random variability in development")
    variability_factor: float = Field(0.2, ge=0.0, le=0.5, description="Random variability factor (0.0-0.5)")
    
    model_config = {"extra": "forbid"}
    
    @field_validator('stage_definitions')
    @classmethod
    def validate_stage_definitions(cls, v):
        """Validate that all stages are defined and ranges don't overlap."""
        stages = {stage_def.stage for stage_def in v}
        if len(stages) != len(DevelopmentalStage):
            missing = set(DevelopmentalStage) - stages
            raise ValueError(f"Missing stage definitions for: {missing}")
            
        # Sort by min_age to check for overlaps
        sorted_defs = sorted(v, key=lambda x: x.range.min_age)
        for i in range(1, len(sorted_defs)):
            curr = sorted_defs[i]
            prev = sorted_defs[i-1]
            if prev.range.max_age is None:
                raise ValueError(f"Stage {prev.stage} has no max_age but is not the final stage")
            if curr.range.min_age < prev.range.max_age:
                raise ValueError(f"Stage ranges overlap between {prev.stage} and {curr.stage}")
                
        # Ensure the last stage has no max_age
        last_stage = sorted_defs[-1]
        if last_stage.range.max_age is not None:
            raise ValueError(f"Final stage {last_stage.stage} should have no max_age")
            
        return v
