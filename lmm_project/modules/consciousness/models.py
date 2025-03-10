from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime
import uuid

class AwarenessState(BaseModel):
    """Represents the current awareness state of the system"""
    external_awareness: float = Field(default=0.1, ge=0.0, le=1.0)
    internal_awareness: float = Field(default=0.1, ge=0.0, le=1.0)
    social_awareness: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_awareness: float = Field(default=0.1, ge=0.0, le=1.0)
    monitored_states: Dict[str, Any] = Field(default_factory=dict)
    attentional_focus: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class GlobalWorkspaceItem(BaseModel):
    """An item that has entered the global workspace"""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any]
    source_module: str
    activation_level: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)  # How quickly activation decays
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class WorkspaceState(BaseModel):
    """The state of the global workspace"""
    active_items: Dict[str, GlobalWorkspaceItem] = Field(default_factory=dict)
    capacity: int = Field(default=7, ge=1, le=20)  # Working memory capacity
    competition_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class SelfModelState(BaseModel):
    """Represents the system's model of itself"""
    identity: Dict[str, Any] = Field(default_factory=dict)
    capabilities: Dict[str, float] = Field(default_factory=dict)
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    self_evaluation: Dict[str, float] = Field(default_factory=dict)
    autobiographical_memories: List[str] = Field(default_factory=list)  # IDs of key memories
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class IntrospectionState(BaseModel):
    """Represents the system's introspective capabilities"""
    depth: float = Field(default=0.1, ge=0.0, le=1.0)  # Depth of introspection
    active_processes: Dict[str, float] = Field(default_factory=dict)  # Processes being examined
    insights: List[Dict[str, Any]] = Field(default_factory=list)  # Insights gained
    metacognitive_monitoring: Dict[str, float] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class ConsciousnessState(BaseModel):
    """Integrated state of the consciousness system"""
    awareness: AwarenessState = Field(default_factory=AwarenessState)
    global_workspace: WorkspaceState = Field(default_factory=WorkspaceState)
    self_model: SelfModelState = Field(default_factory=SelfModelState)
    introspection: IntrospectionState = Field(default_factory=IntrospectionState)
    
    # Overall consciousness level (emergent property of components)
    consciousness_level: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Development level of consciousness
    developmental_level: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Time tracking for consciousness states
    last_update: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @model_validator(mode='after')
    def update_consciousness_level(self):
        """Calculate consciousness level from components"""
        self.consciousness_level = (
            self.awareness.external_awareness * 0.2 +
            self.awareness.internal_awareness * 0.2 +
            min(1.0, len(self.global_workspace.active_items) / self.global_workspace.capacity) * 0.3 +
            self.introspection.depth * 0.3
        ) * (0.5 + 0.5 * self.developmental_level)  # Developmental scaling
        return self
