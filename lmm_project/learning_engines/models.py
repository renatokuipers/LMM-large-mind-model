from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, Set
from datetime import datetime
import uuid
import numpy as np

class LearningEngine(BaseModel):
    """Base class for learning engines"""
    engine_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    engine_type: str
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class LearningEvent(BaseModel):
    """Records a learning event that occurred in the system"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    engine_id: str
    engine_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    target_module: str
    target_network_id: Optional[str] = None
    neurons_affected: List[str] = Field(default_factory=list)
    synapses_affected: List[str] = Field(default_factory=list)
    magnitude: float = Field(default=0.0, ge=0.0)  # How significant the learning was
    details: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class ReinforcementParameters(BaseModel):
    """Parameters for reinforcement learning"""
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    discount_factor: float = Field(default=0.95, ge=0.0, le=1.0)  # Gamma
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)  # Epsilon
    eligibility_trace_decay: float = Field(default=0.9, ge=0.0, le=1.0)  # Lambda
    update_method: Literal["q_learning", "sarsa", "actor_critic"] = "q_learning"
    reward_scaling: float = Field(default=1.0, gt=0.0)
    min_reward: float = Field(default=-1.0)
    max_reward: float = Field(default=1.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class HebbianParameters(BaseModel):
    """Parameters for Hebbian learning"""
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    min_weight: float = Field(default=-1.0)
    max_weight: float = Field(default=1.0)
    learning_rule: Literal["hebbian", "oja", "bcm", "stdp"] = "hebbian"
    modulation_factor: float = Field(default=1.0, ge=0.0)  # For neuromodulation
    stability_threshold: float = Field(default=0.01, ge=0.0)  # For weight stabilization
    time_window: float = Field(default=20.0, ge=0.0)  # For STDP (in ms)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class PruningParameters(BaseModel):
    """Parameters for neural pruning"""
    weight_threshold: float = Field(default=0.01, ge=0.0)  # Minimum weight to keep
    usage_threshold: float = Field(default=0.1, ge=0.0)  # Minimum usage to keep
    importance_threshold: float = Field(default=0.2, ge=0.0)  # Minimum importance to keep
    max_prune_percent: float = Field(default=0.2, ge=0.0, le=1.0)  # Max % to prune at once
    pruning_frequency: int = Field(default=1000, ge=1)  # How often to prune
    preserve_io_neurons: bool = Field(default=True)  # Preserve input/output neurons
    pruning_strategy: Literal["weight", "usage", "importance", "combined"] = "combined"
    recovery_probability: float = Field(default=0.01, ge=0.0, le=1.0)  # Chance to recover pruned connections
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class ConsolidationParameters(BaseModel):
    """Parameters for memory consolidation"""
    consolidation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # When to consolidate
    importance_factor: float = Field(default=0.7, ge=0.0, le=1.0)  # How much importance influences consolidation
    recency_factor: float = Field(default=0.3, ge=0.0, le=1.0)  # How much recency influences consolidation
    reactivation_strength: float = Field(default=0.8, ge=0.0, le=1.0)  # How strongly to reactivate memories
    consolidation_frequency: int = Field(default=100, ge=1)  # How often to consolidate
    stabilization_rate: float = Field(default=0.1, ge=0.0, le=1.0)  # Rate of stabilization
    sleep_consolidation_boost: float = Field(default=2.0, ge=1.0)  # Boost during "sleep" mode
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class SynapticTaggingParameters(BaseModel):
    """Parameters for synaptic tagging and capture"""
    tag_threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Activation threshold for tagging
    tag_duration: int = Field(default=60, ge=1)  # How long tags persist (in cycles)
    capture_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # Threshold for PRPs to capture
    prp_production_threshold: float = Field(default=0.7, ge=0.0, le=1.0)  # Threshold for producing PRPs
    tag_strength_factor: float = Field(default=0.5, ge=0.0, le=1.0)  # How tag strength affects capture
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class NeuronUsageStats(BaseModel):
    """Tracks usage statistics for a neuron"""
    neuron_id: str
    activation_count: int = Field(default=0, ge=0)
    total_activation: float = Field(default=0.0, ge=0.0)
    average_activation: float = Field(default=0.0, ge=0.0)
    last_activated: Optional[datetime] = None
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class SynapseUsageStats(BaseModel):
    """Tracks usage statistics for a synapse"""
    synapse_id: str
    transmission_count: int = Field(default=0, ge=0)
    total_transmission: float = Field(default=0.0, ge=0.0)
    average_transmission: float = Field(default=0.0, ge=0.0)
    last_used: Optional[datetime] = None
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tag_status: bool = Field(default=False)
    tag_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    tag_expiration: Optional[datetime] = None
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class LearningRuleApplication(BaseModel):
    """Records the application of a learning rule"""
    neuron_id: Optional[str] = None
    synapse_id: Optional[str] = None
    rule_type: str
    pre_value: float  # Value before rule application
    post_value: float  # Value after rule application
    delta: float  # Change in value
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
