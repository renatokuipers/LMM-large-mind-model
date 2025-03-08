from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

class SalienceScore(BaseModel):
    """
    Represents the salience (noticeability/importance) of an item
    
    Salience is influenced by multiple factors including novelty,
    emotional significance, and relevance to current goals.
    """
    # Unique identifier for the item
    item_id: str
    # Overall salience score (0.0-1.0)
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    # Novelty contribution (how new/unexpected)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0) 
    # Emotional significance contribution
    emotional_significance: float = Field(default=0.0, ge=0.0, le=1.0)
    # Relevance to current goals
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    # Sensory intensity contribution
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Timestamp when this score was calculated
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @model_validator(mode='after')
    def calculate_overall_score(self):
        """Calculate overall score based on component factors"""
        # Simple weighted average of factors (weights could be learned over time)
        self.score = (
            0.3 * self.novelty +
            0.3 * self.emotional_significance +
            0.25 * self.goal_relevance +
            0.15 * self.intensity
        )
        return self

class AttentionTarget(BaseModel):
    """
    A single target of attention
    
    Represents an item that is currently receiving attention,
    with metadata about why and how much attention it's receiving.
    """
    # Unique identifier for the target
    target_id: str
    # Description of this attention target
    description: str = ""
    # Type of the target (e.g., "perception", "memory", "thought")
    target_type: str
    # Activation level (how much attention it's receiving)
    activation: float = Field(default=1.0, ge=0.0, le=1.0)
    # When this target first received attention
    entry_time: datetime = Field(default_factory=datetime.now)
    # Last time this target's activation was updated
    last_updated: datetime = Field(default_factory=datetime.now)
    # Metadata about this target
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_activation(self, amount: float) -> float:
        """Update the activation of this target"""
        self.activation = max(0.0, min(1.0, self.activation + amount))
        self.last_updated = datetime.now()
        return self.activation
    
    def decay_activation(self, decay_rate: float) -> float:
        """Apply decay to activation over time"""
        # Calculate time since last update
        time_delta = (datetime.now() - self.last_updated).total_seconds()
        # Apply decay
        decay_amount = decay_rate * time_delta
        self.activation = max(0.0, self.activation - decay_amount)
        self.last_updated = datetime.now()
        return self.activation

class AttentionFocus(BaseModel):
    """
    Current focus of attention
    
    Maintains a collection of items currently receiving attention
    and manages the limited capacity of attention.
    """
    # Mapping of target_id to activation level
    targets: Dict[str, float] = Field(default_factory=dict)
    # Maximum number of items that can receive attention simultaneously
    capacity: float = Field(default=3.0, ge=1.0, le=10.0)
    # If attention is fully engaged or free for new stimuli
    is_at_capacity: bool = Field(default=False)
    # Detailed target information
    target_details: Dict[str, AttentionTarget] = Field(default_factory=dict)
    # Current overall attention state (e.g., "focused", "divided", "diffuse")
    state: str = "focused"
    # Timestamp of last focus update
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @model_validator(mode='after')
    def update_capacity_state(self):
        """Update is_at_capacity based on number of targets"""
        self.is_at_capacity = len(self.targets) >= self.capacity
        return self
    
    def add_target(self, target: AttentionTarget) -> bool:
        """
        Add a new target to attention focus
        
        Returns:
        True if successfully added, False if at capacity
        """
        # Check if already at capacity and this isn't already a target
        if self.is_at_capacity and target.target_id not in self.targets:
            return False
            
        # Add or update target
        self.targets[target.target_id] = target.activation
        self.target_details[target.target_id] = target
        self.last_updated = datetime.now()
        
        # Update capacity state
        self.is_at_capacity = len(self.targets) >= self.capacity
        
        return True
    
    def remove_target(self, target_id: str) -> bool:
        """Remove a target from attention focus"""
        if target_id in self.targets:
            del self.targets[target_id]
            if target_id in self.target_details:
                del self.target_details[target_id]
                
            self.is_at_capacity = len(self.targets) >= self.capacity
            self.last_updated = datetime.now()
            return True
        return False
    
    def update_targets(self, activations: Dict[str, float]) -> None:
        """Update activation levels for multiple targets"""
        for target_id, activation in activations.items():
            if target_id in self.target_details:
                self.target_details[target_id].update_activation(activation - self.targets[target_id])
                self.targets[target_id] = self.target_details[target_id].activation
        
        self.last_updated = datetime.now()
    
    def decay_all(self, decay_rate: float) -> List[str]:
        """
        Apply decay to all targets and remove those below threshold
        
        Returns:
        List of removed target IDs
        """
        removal_threshold = 0.1
        removed_ids = []
        
        for target_id, target in list(self.target_details.items()):
            target.decay_activation(decay_rate)
            self.targets[target_id] = target.activation
            
            # Remove if below threshold
            if target.activation < removal_threshold:
                self.remove_target(target_id)
                removed_ids.append(target_id)
        
        self.last_updated = datetime.now()
        return removed_ids
    
    def get_dominant_target(self) -> Optional[Tuple[str, float]]:
        """Get the target with highest activation level"""
        if not self.targets:
            return None
            
        target_id = max(self.targets.items(), key=lambda x: x[1])[0]
        return (target_id, self.targets[target_id])

class AttentionParameters(BaseModel):
    """
    Parameters controlling attention behavior
    
    These parameters can be adjusted based on development level
    or situational factors.
    """
    # How quickly attention decays over time
    decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    # How strongly salience influences attention
    salience_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    # How strongly novelty contributes to salience
    novelty_bias: float = Field(default=0.6, ge=0.0, le=1.0)
    # How strongly emotional significance contributes to salience
    emotional_bias: float = Field(default=0.7, ge=0.0, le=1.0)
    # How easily attention shifts to new targets
    shift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)