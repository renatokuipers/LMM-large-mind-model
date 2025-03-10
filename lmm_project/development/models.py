from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, List, Tuple, Optional, Set, Literal, Union, ClassVar
from datetime import datetime, timedelta
import uuid
import logging
import json
from enum import Enum, auto

logger = logging.getLogger(__name__)

class DevelopmentalDomain(str, Enum):
    """Enumeration of developmental domains for categorization"""
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    LANGUAGE = "language"
    MOTOR = "motor"
    NEURAL = "neural"
    PERCEPTUAL = "perceptual"
    ATTENTION = "attention"
    MEMORY = "memory"
    EXECUTIVE = "executive"
    CREATIVE = "creative"
    IDENTITY = "identity"

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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('capabilities', 'prerequisites')
    @classmethod
    def validate_capability_levels(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that capability levels are between 0.0 and 1.0"""
        for capability, level in v.items():
            if not isinstance(level, (int, float)):
                raise ValueError(f"Capability level for {capability} must be a number")
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"Capability level for {capability} must be between 0.0 and 1.0")
        return v
    
    @field_validator('age_range')
    @classmethod
    def validate_age_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate that age range is valid"""
        if len(v) != 2:
            raise ValueError("Age range must be a tuple of (min_age, max_age)")
        min_age, max_age = v
        if min_age < 0:
            raise ValueError("Minimum age cannot be negative")
        if max_age <= min_age:
            raise ValueError("Maximum age must be greater than minimum age")
        return v
    
    def activate(self) -> None:
        """Activate this developmental stage"""
        self.is_active = True
        self.entry_time = datetime.now()
        
    def deactivate(self) -> None:
        """Deactivate this developmental stage"""
        self.is_active = False
        self.exit_time = datetime.now()
    
    def duration(self) -> Optional[timedelta]:
        """Calculate the duration of this stage if it has been active"""
        if not self.is_active or not self.entry_time:
            return None
        end_time = self.exit_time or datetime.now()
        return end_time - self.entry_time
    
    def meets_prerequisites(self, current_capabilities: Dict[str, float]) -> bool:
        """Check if the current capabilities meet the prerequisites for this stage"""
        for capability, required_level in self.prerequisites.items():
            if capability not in current_capabilities:
                return False
            if current_capabilities[capability] < required_level:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "age_range": self.age_range,
            "capabilities": self.capabilities,
            "prerequisites": self.prerequisites,
            "description": self.description,
            "is_active": self.is_active,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "expected_milestones": self.expected_milestones
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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('age_range')
    @classmethod
    def validate_age_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate that age range is valid"""
        if len(v) != 2:
            raise ValueError("Age range must be a tuple of (min_age, max_age)")
        min_age, max_age = v
        if min_age < 0:
            raise ValueError("Minimum age cannot be negative")
        if max_age <= min_age:
            raise ValueError("Maximum age must be greater than minimum age")
        return v
    
    def is_active_at_age(self, age: float) -> bool:
        """Check if this critical period is active at the given age"""
        min_age, max_age = self.age_range
        return min_age <= age <= max_age
    
    def is_missed_at_age(self, age: float) -> bool:
        """Check if this critical period is missed at the given age"""
        _, max_age = self.age_range
        return age > max_age and self.status == "pending"
    
    def get_development_multiplier(self) -> float:
        """Get the development multiplier for this critical period"""
        if self.status != "active":
            return 1.0
        return self.plasticity_multiplier
    
    def get_limitation_factor(self) -> float:
        """Get the limitation factor if this period was missed"""
        if not self.causes_permanent_limitation or self.status != "missed":
            return 1.0
        return self.limitation_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "capability": self.capability,
            "age_range": self.age_range,
            "plasticity_multiplier": self.plasticity_multiplier,
            "status": self.status,
            "importance": self.importance,
            "module_targets": self.module_targets,
            "causes_permanent_limitation": self.causes_permanent_limitation,
            "limitation_factor": self.limitation_factor,
            "recommended_experiences": self.recommended_experiences
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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('capability_requirements')
    @classmethod
    def validate_capability_requirements(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that capability requirements are between 0.0 and 1.0"""
        for capability, level in v.items():
            if not isinstance(level, (int, float)):
                raise ValueError(f"Capability requirement for {capability} must be a number")
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"Capability requirement for {capability} must be between 0.0 and 1.0")
        return v
    
    @field_validator('typical_age')
    @classmethod
    def validate_typical_age(cls, v: float) -> float:
        """Validate that typical age is non-negative"""
        if v < 0:
            raise ValueError("Typical age cannot be negative")
        return v
    
    def update_progress(self, new_progress: float) -> None:
        """
        Update the progress towards this milestone
        
        Args:
            new_progress: The new progress value (0.0-1.0)
            
        Returns:
            None
        """
        self.progress = min(1.0, max(0.0, new_progress))
        if self.progress >= 1.0 and not self.achieved:
            self.achieved = True
            self.achieved_at = datetime.now()
            logger.info(f"Milestone achieved: {self.name}")
    
    def calculate_progress(self, current_capabilities: Dict[str, float]) -> float:
        """
        Calculate progress based on current capabilities
        
        Args:
            current_capabilities: Dictionary of current capability levels
            
        Returns:
            Progress value between 0.0 and 1.0
        """
        if not self.capability_requirements:
            return 0.0
            
        total_progress = 0.0
        for capability, required in self.capability_requirements.items():
            if capability not in current_capabilities:
                continue
                
            current = current_capabilities[capability]
            capability_progress = min(1.0, current / required) if required > 0 else 1.0
            total_progress += capability_progress
            
        # Average progress across all required capabilities
        return total_progress / len(self.capability_requirements)
    
    def can_be_achieved(self, achieved_milestone_ids: Set[str]) -> bool:
        """
        Check if this milestone can be achieved based on prerequisites
        
        Args:
            achieved_milestone_ids: Set of IDs of already achieved milestones
            
        Returns:
            True if all prerequisites are met, False otherwise
        """
        return all(prereq in achieved_milestone_ids for prereq in self.prerequisite_milestones)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "typical_age": self.typical_age,
            "achieved": self.achieved,
            "achieved_at": self.achieved_at.isoformat() if self.achieved_at else None,
            "prerequisite_milestones": self.prerequisite_milestones,
            "associated_stage": self.associated_stage,
            "capability_requirements": self.capability_requirements,
            "progress": self.progress,
            "is_essential": self.is_essential
        }

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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('current_age')
    @classmethod
    def validate_current_age(cls, v: float) -> float:
        """Validate that current age is non-negative"""
        if v < 0:
            raise ValueError("Current age cannot be negative")
        return v
    
    def add_stage_transition(self, from_stage: str, to_stage: str) -> None:
        """
        Record a developmental stage transition
        
        Args:
            from_stage: The previous developmental stage
            to_stage: The new developmental stage
            
        Returns:
            None
        """
        self.stage_history.append({
            "from": from_stage,
            "to": to_stage,
            "time": datetime.now(),
            "age": self.current_age
        })
        self.current_stage = to_stage
        logger.info(f"Stage transition: {from_stage} -> {to_stage} at age {self.current_age:.2f}")
    
    def update_age(self, delta_age: float) -> None:
        """
        Update the developmental age
        
        Args:
            delta_age: The amount to increase the age by
            
        Returns:
            None
            
        Raises:
            ValueError: If delta_age is negative
        """
        if delta_age < 0:
            raise ValueError("delta_age must be non-negative")
        self.current_age += delta_age
    
    def update_module_development(self, module: str, level: float) -> None:
        """
        Update the development level for a specific module
        
        Args:
            module: The module name
            level: The new development level (0.0-1.0)
            
        Returns:
            None
            
        Raises:
            ValueError: If level is outside the valid range
        """
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"Development level for {module} must be between 0.0 and 1.0")
        self.module_development[module] = level
    
    def update_velocity(self, module: str, velocity: float) -> None:
        """
        Update the development velocity for a specific module
        
        Args:
            module: The module name
            velocity: The development velocity
            
        Returns:
            None
        """
        self.development_velocity[module] = velocity
    
    def calculate_balance(self) -> None:
        """
        Calculate whether development is balanced across domains
        
        Updates the is_balanced property and areas_of_concern/areas_of_strength lists
        
        Returns:
            None
        """
        if not self.module_development:
            self.is_balanced = True
            return
            
        # Calculate average development level
        avg_level = sum(self.module_development.values()) / len(self.module_development)
        
        # Calculate standard deviation
        variance = sum((level - avg_level) ** 2 for level in self.module_development.values())
        std_dev = (variance / len(self.module_development)) ** 0.5
        
        # Development is considered balanced if standard deviation is less than 0.15
        self.is_balanced = std_dev < 0.15
        
        # Update areas of concern and strength
        threshold = 0.2  # Threshold for concern/strength
        self.areas_of_concern = [
            module for module, level in self.module_development.items()
            if level < avg_level - threshold
        ]
        self.areas_of_strength = [
            module for module, level in self.module_development.items()
            if level > avg_level + threshold
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "current_age": self.current_age,
            "current_stage": self.current_stage,
            "stage_history": self.stage_history,
            "module_development": self.module_development,
            "achieved_milestones": self.achieved_milestones,
            "pending_milestones": self.pending_milestones,
            "development_velocity": self.development_velocity,
            "developmental_offsets": self.developmental_offsets,
            "is_balanced": self.is_balanced,
            "areas_of_concern": self.areas_of_concern,
            "areas_of_strength": self.areas_of_strength
        }

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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('acceleration_factors', 'inhibition_factors')
    @classmethod
    def validate_factors(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that factors have valid multipliers"""
        for factor, multiplier in v.items():
            if not isinstance(multiplier, (int, float)):
                raise ValueError(f"Multiplier for {factor} must be a number")
            if multiplier <= 0:
                raise ValueError(f"Multiplier for {factor} must be positive")
        return v
    
    @field_validator('age_modifiers')
    @classmethod
    def validate_age_modifiers(cls, v: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Validate that age modifiers are valid"""
        for i, (age, multiplier) in enumerate(v):
            if not isinstance(age, (int, float)) or not isinstance(multiplier, (int, float)):
                raise ValueError(f"Age modifier at index {i} must be a tuple of (age, multiplier)")
            if age < 0:
                raise ValueError(f"Age in modifier at index {i} cannot be negative")
            if multiplier <= 0:
                raise ValueError(f"Multiplier in modifier at index {i} must be positive")
        return v
    
    def get_age_multiplier(self, age: float) -> float:
        """
        Get the multiplier for a specific age
        
        Args:
            age: The developmental age
            
        Returns:
            The age-based multiplier
        """
        if not self.age_modifiers:
            return 1.0
            
        # Sort modifiers by age
        sorted_modifiers = sorted(self.age_modifiers, key=lambda x: x[0])
        
        # Find the appropriate age bracket
        for i, (bracket_age, multiplier) in enumerate(sorted_modifiers):
            if age < bracket_age:
                if i == 0:
                    return multiplier
                
                # Interpolate between brackets
                prev_age, prev_mult = sorted_modifiers[i-1]
                weight = (age - prev_age) / (bracket_age - prev_age)
                return prev_mult + weight * (multiplier - prev_mult)
        
        # If beyond the last bracket, use the last multiplier
        return sorted_modifiers[-1][1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "base_rate": self.base_rate,
            "variability": self.variability,
            "acceleration_factors": self.acceleration_factors,
            "inhibition_factors": self.inhibition_factors,
            "age_modifiers": self.age_modifiers
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
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Validate that age is non-negative"""
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate that event type is one of the allowed values"""
        allowed_types = ["milestone", "stage_transition", "critical_period", "regression", "growth_spurt"]
        if v not in allowed_types:
            raise ValueError(f"Event type must be one of: {', '.join(allowed_types)}")
        return v
    
    def get_age_appropriate(self, typical_age: float) -> bool:
        """
        Check if this event occurred at an age-appropriate time
        
        Args:
            typical_age: The typical age for this event
            
        Returns:
            True if the event occurred within a reasonable age range
        """
        # Allow for 30% variation in either direction
        lower_bound = typical_age * 0.7
        upper_bound = typical_age * 1.3
        return lower_bound <= self.age <= upper_bound
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "age": self.age,
            "affected_modules": self.affected_modules,
            "significance": self.significance,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DevelopmentalEvent":
        """Create an event from a dictionary"""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
