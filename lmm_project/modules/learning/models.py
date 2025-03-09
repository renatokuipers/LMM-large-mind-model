from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Set, Union, Literal
from datetime import datetime
import uuid

class LearningEvent(BaseModel):
    """Base model for all learning events in the system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    # Source of the learning event (experience, instruction, observation, etc.)
    source: str
    # Content being learned
    content: str
    # Type of learning event
    learning_type: str
    # Developmental level when this learning occurred (0.0 to 1.0)
    developmental_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # Confidence in the learned information (0.0 to 1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # How many times this has been reinforced
    reinforcement_count: int = Field(default=0, ge=0)
    # Last time this was reinforced
    last_reinforced: Optional[datetime] = None
    # Learning rate for this event (how quickly it's learned)
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    # Tags for categorizing the learning event
    tags: Set[str] = Field(default_factory=set)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class AssociativeLearningEvent(LearningEvent):
    """Model for associative learning events"""
    learning_type: str = "associative"
    # Stimulus that triggered the learning
    stimulus: str
    # Response or associated concept
    response: str
    # Strength of the association (0.0 to 1.0)
    association_strength: float = Field(default=0.3, ge=0.0, le=1.0)
    # Whether this is classical (stimulus-response) or operant (action-consequence) conditioning
    conditioning_type: Literal["classical", "operant"] = "classical"
    # Delay between stimulus and response (in seconds)
    temporal_delay: float = Field(default=0.0, ge=0.0)

class ReinforcementLearningEvent(LearningEvent):
    """Model for reinforcement learning events"""
    learning_type: str = "reinforcement"
    # Action that was taken
    action: str
    # Reward or punishment received
    consequence: str
    # Value of the reward/punishment (-1.0 to 1.0, negative for punishment)
    reward_value: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Delay between action and consequence (in seconds)
    delay: float = Field(default=0.0, ge=0.0)
    # Context in which the action was taken
    context: str
    # Whether this is positive reinforcement, negative reinforcement, or punishment
    reinforcement_type: Literal["positive", "negative", "punishment"] = "positive"

class ProceduralLearningEvent(LearningEvent):
    """Model for procedural learning events"""
    learning_type: str = "procedural"
    # Skill or procedure being learned
    skill: str
    # Current proficiency level (0.0 to 1.0)
    proficiency: float = Field(default=0.1, ge=0.0, le=1.0)
    # Number of practice repetitions
    practice_count: int = Field(default=1, ge=0)
    # Time spent practicing (in seconds)
    practice_time: float = Field(default=0.0, ge=0.0)
    # Whether this is explicit (conscious) or implicit (unconscious) learning
    learning_mode: Literal["explicit", "implicit"] = "explicit"
    # Steps or components of the procedure
    procedure_steps: List[str] = Field(default_factory=list)

class MetaLearningEvent(LearningEvent):
    """Model for meta-learning events"""
    learning_type: str = "meta"
    # Learning strategy being developed
    strategy: str
    # Effectiveness of the strategy (0.0 to 1.0)
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    # Contexts where this strategy works well
    applicable_contexts: List[str] = Field(default_factory=list)
    # Types of learning this strategy helps with
    target_learning_types: List[str] = Field(default_factory=list)
    # Cognitive resource cost (0.0 to 1.0, higher = more resource intensive)
    resource_cost: float = Field(default=0.5, ge=0.0, le=1.0)

class LearningStrategy(BaseModel):
    """Model for learning strategies"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    # Overall effectiveness (0.0 to 1.0)
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    # Cognitive load required (0.0 to 1.0)
    cognitive_load: float = Field(default=0.5, ge=0.0, le=1.0)
    # Minimum developmental level needed to use this strategy
    min_developmental_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # Domains this strategy works well in
    applicable_domains: List[str] = Field(default_factory=list)
    # When this strategy was learned/created
    created_at: datetime = Field(default_factory=datetime.now)
    # How many times this strategy has been used
    usage_count: int = Field(default=0, ge=0)
    # Success rate when using this strategy (0.0 to 1.0)
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
