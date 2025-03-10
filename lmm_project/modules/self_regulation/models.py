from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Any, Optional, Union, Literal, Set
from datetime import datetime
import uuid

class EmotionalState(BaseModel):
    """
    Represents an emotional state that needs regulation
    
    This model captures the emotional state of the mind at a point in time,
    including intensity, valence, and regulation status.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Core emotion properties
    emotion_type: str  # e.g., "anger", "fear", "joy", "sadness"
    intensity: float = Field(ge=0.0, le=1.0)  # How strong the emotion is
    valence: float = Field(ge=-1.0, le=1.0)   # Negative to positive
    arousal: float = Field(ge=0.0, le=1.0)    # Low to high activation
    
    # Regulation status
    is_regulated: bool = False          # Whether regulation has been applied
    regulation_strategy: Optional[str] = None  # What strategy was used
    regulation_success: float = Field(ge=0.0, le=1.0, default=0.0)  # How successful the regulation was
    
    # Context information
    trigger: Optional[str] = None       # What caused this emotion
    context: Dict[str, Any] = Field(default_factory=dict)  # Additional context
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "emotion_type": "anger",
                    "intensity": 0.8,
                    "valence": -0.7,
                    "arousal": 0.9,
                    "is_regulated": True,
                    "regulation_strategy": "cognitive_reappraisal",
                    "regulation_success": 0.6,
                    "trigger": "perceived_criticism"
                }
            ]
        }
    }
    
    @field_validator('emotion_type')
    @classmethod
    def validate_emotion_type(cls, v: str) -> str:
        """Validate emotion type is not empty"""
        if not v.strip():
            raise ValueError("Emotion type cannot be empty")
        return v.lower()

class ImpulseEvent(BaseModel):
    """
    Represents an impulse that needs to be controlled
    
    This model captures impulsive urges that arise and may need
    inhibition through self-regulation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Impulse characteristics
    impulse_type: str  # e.g., "approach", "avoidance", "consumption", "expression"
    strength: float = Field(ge=0.0, le=1.0)  # How strong the impulse is
    valence: float = Field(ge=-1.0, le=1.0)  # Negative to positive
    
    # Control status
    is_controlled: bool = False          # Whether the impulse was controlled
    control_strategy: Optional[str] = None  # What strategy was used
    control_success: float = Field(ge=0.0, le=1.0, default=0.0)  # How successful the control was
    
    # Context and triggers
    trigger: Optional[str] = None  # What triggered this impulse
    context: Dict[str, Any] = Field(default_factory=dict)  # Additional context
    
    # Outcome if acted upon
    was_acted_on: bool = False  # Whether the impulse led to action
    action_outcome: Optional[Dict[str, Any]] = None  # Result of the action if taken
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "impulse_type": "approach",
                    "strength": 0.7,
                    "valence": 0.5,
                    "is_controlled": True,
                    "control_strategy": "delay_gratification",
                    "control_success": 0.8,
                    "trigger": "attractive_stimulus",
                    "was_acted_on": False
                }
            ]
        }
    }
    
    @field_validator('impulse_type')
    @classmethod
    def validate_impulse_type(cls, v: str) -> str:
        """Validate impulse type is not empty"""
        if not v.strip():
            raise ValueError("Impulse type cannot be empty")
        return v.lower()

class MonitoringEvent(BaseModel):
    """
    Represents a self-monitoring observation
    
    This model captures the mind's observations about its own states,
    behaviors, and the gaps between goals and current situations.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Monitoring type
    monitoring_type: Literal["emotional", "behavioral", "cognitive", "goal_progress", "error_detection"]
    
    # What was observed
    observed_state: Dict[str, Any]  # The internal state that was monitored
    
    # Discrepancy detection
    goal_state: Optional[Dict[str, Any]] = None  # The desired state if applicable
    discrepancy_detected: bool = False  # Whether a gap was found between current and goal
    discrepancy_magnitude: float = Field(ge=0.0, le=1.0, default=0.0)  # Size of the gap
    
    # Error detection
    error_detected: bool = False  # Whether an error was detected
    error_type: Optional[str] = None  # Type of error if detected
    error_severity: float = Field(ge=0.0, le=1.0, default=0.0)  # Severity of error
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict)  # Context of the monitoring
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "monitoring_type": "goal_progress",
                    "observed_state": {
                        "current_progress": 0.3,
                        "time_elapsed": 0.5,
                        "effort_expended": 0.6
                    },
                    "goal_state": {
                        "expected_progress": 0.5
                    },
                    "discrepancy_detected": True,
                    "discrepancy_magnitude": 0.2
                }
            ]
        }
    }

class RegulationStrategy(BaseModel):
    """
    Represents a strategy for self-regulation
    
    This model defines different strategies that can be used for
    emotional regulation, impulse control, or self-correction.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # Name of the strategy
    description: str  # Description of what the strategy does
    
    # Strategy type
    strategy_type: Literal["emotional", "impulse", "behavioral", "cognitive"]
    
    # Effectiveness parameters
    effectiveness: float = Field(ge=0.0, le=1.0, default=0.5)  # General effectiveness
    complexity: float = Field(ge=0.0, le=1.0, default=0.5)  # How complex/demanding the strategy is
    min_development_level: float = Field(ge=0.0, le=1.0, default=0.0)  # Minimum development needed
    
    # Application context
    applicable_situations: Set[str] = Field(default_factory=set)  # When to use this strategy
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "name": "cognitive_reappraisal",
                    "description": "Reinterpreting a situation to change its emotional impact",
                    "strategy_type": "emotional",
                    "effectiveness": 0.8,
                    "complexity": 0.7,
                    "min_development_level": 0.6,
                    "applicable_situations": ["negative_emotions", "anxiety", "frustration"]
                }
            ]
        }
    }
    
    @field_validator('name', 'description')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Validate text fields are not empty"""
        if not v.strip():
            raise ValueError("Text fields cannot be empty")
        return v

class SelfRegulationState(BaseModel):
    """
    Represents the overall state of the self-regulation system
    
    This model tracks the current status of self-regulation capabilities,
    including emotional regulation, impulse control, and self-monitoring.
    """
    # System state
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_level: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Component states
    emotional_regulation_capacity: float = Field(ge=0.0, le=1.0, default=0.0)
    impulse_control_capacity: float = Field(ge=0.0, le=1.0, default=0.0)
    self_monitoring_capacity: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Current regulation status
    current_emotional_state: Optional[EmotionalState] = None
    current_impulse_state: Optional[ImpulseEvent] = None
    recent_monitoring_events: List[MonitoringEvent] = Field(default_factory=list)
    
    # Available strategies
    available_emotional_strategies: List[str] = Field(default_factory=list)
    available_impulse_strategies: List[str] = Field(default_factory=list)
    
    # Regulation outcomes
    regulation_success_rate: float = Field(ge=0.0, le=1.0, default=0.5)
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "developmental_level": 0.6,
                    "emotional_regulation_capacity": 0.7,
                    "impulse_control_capacity": 0.5,
                    "self_monitoring_capacity": 0.6,
                    "available_emotional_strategies": [
                        "cognitive_reappraisal", 
                        "distraction", 
                        "situation_modification"
                    ],
                    "regulation_success_rate": 0.65
                }
            ]
        }
    }
    
    def update_capacities(self, developmental_level: float) -> None:
        """Update capacities based on developmental level"""
        self.developmental_level = developmental_level
        
        # Simple model: capacities grow with development but at different rates
        self.emotional_regulation_capacity = min(1.0, developmental_level * 1.1)
        self.impulse_control_capacity = min(1.0, developmental_level * 0.9 + 0.1)
        self.self_monitoring_capacity = min(1.0, developmental_level * 1.2 - 0.1)
        
        self.timestamp = datetime.now()

class RegulationNeuralState(BaseModel):
    """
    Represents the neural state of the self-regulation system
    
    This model tracks the internal neural state of the self-regulation system,
    including network parameters and learning state.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    developmental_level: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Neural activation states
    emotional_regulation_activation: Dict[str, float] = Field(default_factory=dict)
    impulse_control_activation: Dict[str, float] = Field(default_factory=dict)
    self_monitoring_activation: Dict[str, float] = Field(default_factory=dict)
    
    # Learning parameters
    learning_rate: float = Field(ge=0.0, le=1.0, default=0.1)
    plasticity: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Network weights (simplified representation)
    network_weights: Dict[str, List[float]] = Field(default_factory=dict)
    
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "developmental_level": 0.5,
                    "emotional_regulation_activation": {
                        "reappraisal": 0.6,
                        "suppression": 0.3
                    },
                    "learning_rate": 0.05,
                    "plasticity": 0.4
                }
            ]
        }
    }
    
    def update_developmental_parameters(self, level: float) -> None:
        """Update neural parameters based on developmental level"""
        self.developmental_level = level
        
        # Learning rate decreases with development (more stable)
        self.learning_rate = max(0.01, 0.2 - (level * 0.15))
        
        # Plasticity decreases with development
        self.plasticity = max(0.1, 0.8 - (level * 0.4))
        
        self.timestamp = datetime.now()
