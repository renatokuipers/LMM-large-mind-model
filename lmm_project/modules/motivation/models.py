from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Set, Union, Literal
from datetime import datetime
import uuid
import numpy as np

class Drive(BaseModel):
    """
    Represents a basic biological or psychological drive
    
    Drives are fundamental motivational forces that push the agent 
    toward certain behaviors. They represent internal states that 
    require regulation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    # Current intensity of this drive (0.0 = satisfied, 1.0 = maximum intensity)
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    # How quickly this drive intensity increases over time (higher = faster)
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    # Priority level of this drive (higher = more important)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    # Category of drive (physiological, safety, cognitive, etc.)
    category: str
    # What actions can satisfy this drive
    satisfying_actions: Set[str] = Field(default_factory=set)
    # Last time this drive was updated
    last_updated: datetime = Field(default_factory=datetime.now)
    # Whether this drive is active (some drives only activate at certain dev levels)
    is_active: bool = Field(default=True)
    
    def update_intensity(self, amount: float) -> None:
        """Update the drive intensity, ensuring it stays within bounds"""
        self.intensity = max(0.0, min(1.0, self.intensity + amount))
        self.last_updated = datetime.now()
    
    def decay(self, time_delta: float) -> None:
        """Increase drive intensity naturally over time"""
        decay_amount = self.decay_rate * time_delta
        self.update_intensity(decay_amount)

class Need(BaseModel):
    """
    Represents a psychological need
    
    Needs are higher-level than drives and represent psychological
    requirements for well-being. Based on theories like Maslow's hierarchy.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    # Current satisfaction level (0.0 = unsatisfied, 1.0 = fully satisfied)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    # How quickly satisfaction decays (higher = faster decay)
    decay_rate: float = Field(default=0.005, ge=0.0, le=0.1)
    # Minimum development level required for this need to be active
    min_development_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # Hierarchy level (1 = most basic, 5 = highest)
    hierarchy_level: int = Field(default=1, ge=1, le=5)
    # What other needs must be satisfied before this one becomes active
    prerequisites: List[str] = Field(default_factory=list)
    # Related drives that contribute to satisfying this need
    related_drives: List[str] = Field(default_factory=list)
    # What goals can satisfy this need
    satisfying_goals: Set[str] = Field(default_factory=set)
    # Last time this need was updated
    last_updated: datetime = Field(default_factory=datetime.now)
    # Whether this need is currently active
    is_active: bool = Field(default=True)
    
    def update_satisfaction(self, amount: float) -> None:
        """Update the satisfaction level, ensuring it stays within bounds"""
        self.satisfaction = max(0.0, min(1.0, self.satisfaction + amount))
        self.last_updated = datetime.now()
    
    def decay(self, time_delta: float) -> None:
        """Decrease satisfaction naturally over time"""
        decay_amount = self.decay_rate * time_delta
        self.update_satisfaction(-decay_amount)

class Goal(BaseModel):
    """
    Represents a goal or intention
    
    Goals are targets for action that help satisfy needs and drives.
    They can be hierarchical, with subgoals contributing to higher goals.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    # Current progress toward goal (0.0 = not started, 1.0 = completed)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    # Importance of this goal (higher = more important)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # How urgent is this goal (higher = more urgent)
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    # Specific needs this goal can satisfy
    satisfies_needs: List[str] = Field(default_factory=list)
    # Drives this goal can satisfy
    satisfies_drives: List[str] = Field(default_factory=list)
    # Parent goal, if this is a subgoal
    parent_goal_id: Optional[str] = None
    # Subgoals that contribute to this goal
    subgoals: List[str] = Field(default_factory=list)
    # Expected reward for completing this goal
    expected_reward: float = Field(default=0.5, ge=0.0, le=1.0)
    # Whether this goal is currently being pursued
    is_active: bool = Field(default=True)
    # When this goal was created
    created_at: datetime = Field(default_factory=datetime.now)
    # When this goal was last updated
    last_updated: datetime = Field(default_factory=datetime.now)
    # Deadline for goal completion, if any
    deadline: Optional[datetime] = None
    # Whether this goal has been achieved
    is_achieved: bool = Field(default=False)
    # Goal type (approach or avoidance)
    goal_type: Literal["approach", "avoidance"] = "approach"
    
    def update_progress(self, amount: float) -> None:
        """Update progress toward goal"""
        self.progress = max(0.0, min(1.0, self.progress + amount))
        self.last_updated = datetime.now()
        if self.progress >= 1.0:
            self.is_achieved = True
    
    def abandon(self) -> None:
        """Abandon this goal"""
        self.is_active = False
        self.last_updated = datetime.now()

class RewardEvent(BaseModel):
    """
    Represents a reward or reinforcement event
    
    Reward events are used to reinforce behaviors and learn
    from experience. They can be positive (rewards) or negative
    (punishments).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Type of reward (intrinsic or extrinsic)
    reward_type: Literal["intrinsic", "extrinsic"] = "intrinsic"
    # Magnitude of reward (positive = reward, negative = punishment)
    magnitude: float = Field(default=0.5, ge=-1.0, le=1.0)
    # What action or goal triggered this reward
    source: str
    # Specific context where the reward occurred
    context: str
    # Timestamp when this reward was received
    timestamp: datetime = Field(default_factory=datetime.now)
    # What drives or needs were affected by this reward
    affected_drives: List[str] = Field(default_factory=list)
    affected_needs: List[str] = Field(default_factory=list)
    # Whether this reward has been processed for learning
    is_processed: bool = Field(default=False)
    # Additional metadata about this reward
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('magnitude')
    @classmethod
    def validate_magnitude(cls, v: float) -> float:
        """Ensure magnitude is within bounds"""
        return max(-1.0, min(1.0, v))

class MotivationalState(BaseModel):
    """
    Represents the current motivational state of the system
    
    This includes the current state of all drives, needs, and goals,
    as well as overall motivational metrics.
    """
    # Current drives
    drives: Dict[str, Drive] = Field(default_factory=dict)
    # Current needs
    needs: Dict[str, Need] = Field(default_factory=dict)
    # Current goals
    goals: Dict[str, Goal] = Field(default_factory=dict)
    # Recent reward events
    recent_rewards: List[RewardEvent] = Field(default_factory=list)
    # Overall motivation level
    motivation_level: float = Field(default=0.5, ge=0.0, le=1.0)
    # Current development level
    development_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # Dominant current motivation
    dominant_motivation: Optional[str] = None
    # Timestamp when this state was created
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def update_motivation_level(self) -> None:
        """Update the overall motivation level based on drives and needs"""
        if not self.drives and not self.needs:
            return
            
        # Calculate motivation from active drives
        drive_motivation = 0.0
        active_drives = [d for d in self.drives.values() if d.is_active]
        if active_drives:
            drive_motivation = sum(d.intensity * d.priority for d in active_drives) / sum(d.priority for d in active_drives)
            
        # Calculate motivation from needs
        need_motivation = 0.0
        active_needs = [n for n in self.needs.values() if n.is_active]
        if active_needs:
            # Invert satisfaction since lower satisfaction means higher motivation
            need_motivation = sum((1.0 - n.satisfaction) * (6 - n.hierarchy_level) 
                                  for n in active_needs) / sum(6 - n.hierarchy_level 
                                                             for n in active_needs)
        
        # Combine drive and need motivation (weight shifts with development)
        drive_weight = max(0.2, 1.0 - self.development_level * 0.8)
        need_weight = 1.0 - drive_weight
        
        if active_drives and active_needs:
            self.motivation_level = (drive_motivation * drive_weight + 
                                     need_motivation * need_weight)
        elif active_drives:
            self.motivation_level = drive_motivation
        elif active_needs:
            self.motivation_level = need_motivation
        
        # Update dominant motivation
        self._update_dominant_motivation()
    
    def _update_dominant_motivation(self) -> None:
        """Determine the dominant current motivation"""
        # Check drives first
        max_drive = None
        max_drive_value = -1.0
        
        for drive_id, drive in self.drives.items():
            if drive.is_active:
                drive_value = drive.intensity * drive.priority
                if drive_value > max_drive_value:
                    max_drive_value = drive_value
                    max_drive = drive_id
        
        # Check needs
        max_need = None
        max_need_value = -1.0
        
        for need_id, need in self.needs.items():
            if need.is_active:
                need_value = (1.0 - need.satisfaction) * (6 - need.hierarchy_level)
                if need_value > max_need_value:
                    max_need_value = need_value
                    max_need = need_id
        
        # Compare max drive and max need
        if max_drive and max_need:
            if max_drive_value > max_need_value:
                self.dominant_motivation = f"drive:{max_drive}"
            else:
                self.dominant_motivation = f"need:{max_need}"
        elif max_drive:
            self.dominant_motivation = f"drive:{max_drive}"
        elif max_need:
            self.dominant_motivation = f"need:{max_need}"
        else:
            self.dominant_motivation = None

class MotivationNeuralState(BaseModel):
    """Neural state information for the motivation system"""
    # Input embedding from current state
    input_embedding: Optional[List[float]] = None
    # Output activation patterns
    output_activations: Dict[str, List[float]] = Field(default_factory=dict)
    # Current learning rate
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    # Hebbian connection strengths
    connection_strengths: Dict[str, List[float]] = Field(default_factory=dict)
    # Developmental level of neural substrate
    development_level: float = Field(default=0.0, ge=0.0, le=1.0)
    # Last update timestamp
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_activation(self, activation_type: str, activations: List[float]) -> None:
        """Store activation pattern"""
        self.output_activations[activation_type] = activations
        self.last_updated = datetime.now()
    
    def update_connection(self, connection_type: str, strengths: List[float]) -> None:
        """Update connection strengths"""
        self.connection_strengths[connection_type] = strengths
        self.last_updated = datetime.now()
