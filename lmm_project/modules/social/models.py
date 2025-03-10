from pydantic import BaseModel, Field 
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from uuid import uuid4
import numpy as np

# Agent models for theory of mind
class MentalState(BaseModel):
    """Represents an agent's mental state for theory of mind processing"""
    agent_id: str
    beliefs: Dict[str, float] = Field(default_factory=dict)  # belief key -> confidence
    desires: Dict[str, float] = Field(default_factory=dict)  # desire -> strength
    intentions: Dict[str, float] = Field(default_factory=dict)  # intention -> commitment
    emotions: Dict[str, float] = Field(default_factory=dict)  # emotion -> intensity
    knowledge: Dict[str, bool] = Field(default_factory=dict)  # fact -> known/unknown
    attention: List[str] = Field(default_factory=list)  # objects of attention
    perspective: Dict[str, Any] = Field(default_factory=dict)  # perspective data
    last_updated: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "extra": "forbid"
    }

class AgentModel(BaseModel):
    """Model of an agent including identity and mental state"""
    agent_id: str
    name: Optional[str] = None
    mental_state: MentalState
    metadata: Dict[str, Any] = Field(default_factory=dict)
    history: List[MentalState] = Field(default_factory=list, max_items=20)
    
    model_config = {
        "extra": "forbid"
    }

# Relationship models
class RelationshipType(BaseModel):
    """Defines a type of relationship and its attributes"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    attributes: Dict[str, Tuple[float, float]] = Field(default_factory=dict)  # attribute -> (min, max)
    expectations: Dict[str, Any] = Field(default_factory=dict)  # behavior expectations
    
    model_config = {
        "extra": "forbid"
    }

class Relationship(BaseModel):
    """Represents a relationship between agents"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_ids: Tuple[str, str]  # IDs of the agents in the relationship
    type_id: str  # Type of relationship
    qualities: Dict[str, float] = Field(default_factory=dict)  # quality -> value
    history: List[Dict[str, Any]] = Field(default_factory=list)  # interaction history
    created_at: datetime = Field(default_factory=datetime.now)
    last_interaction: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "extra": "forbid"
    }

# Moral reasoning models
class EthicalPrinciple(BaseModel):
    """Represents an ethical principle used in moral reasoning"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    framework: str  # e.g., "consequentialism", "deontology", "virtue_ethics"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    
    model_config = {
        "extra": "forbid"
    }

class MoralEvaluation(BaseModel):
    """Represents a moral evaluation of an action or situation"""
    action_id: str
    principles_applied: Dict[str, float] = Field(default_factory=dict)  # principle ID -> relevance
    judgment: float = Field(ge=-1.0, le=1.0)  # -1 (morally wrong) to 1 (morally right)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    justification: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "extra": "forbid"
    }

# Social norm models
class SocialNorm(BaseModel):
    """Represents a social norm or convention"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    contexts: List[str] = Field(default_factory=list)  # contexts where this norm applies
    norm_type: str  # e.g., "etiquette", "moral", "conventional", "descriptive"
    strength: float = Field(default=0.5, ge=0.0, le=1.0)  # How strong/important the norm is
    flexibility: float = Field(default=0.5, ge=0.0, le=1.0)  # How flexible application is
    behaviors: Dict[str, float] = Field(default_factory=dict)  # expected behaviors and their importance
    
    model_config = {
        "extra": "forbid"
    }

class NormViolation(BaseModel):
    """Represents a detected violation of a social norm"""
    norm_id: str
    agent_id: str
    context: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "extra": "forbid"
    }
