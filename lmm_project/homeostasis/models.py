from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Literal
from datetime import datetime
from enum import Enum, auto

class HomeostaticNeedType(str, Enum):
    """Types of homeostatic needs the system must regulate"""
    ENERGY = "energy"                  # Overall system energy
    AROUSAL = "arousal"                # Activation/stimulation level
    COGNITIVE_LOAD = "cognitive_load"  # Processing resource utilization
    SOCIAL = "social"                  # Need for interaction
    NOVELTY = "novelty"                # Need for new experiences
    REST = "rest"                      # Need for processing downtime
    COHERENCE = "coherence"            # Need for internal consistency

class NeedState(BaseModel):
    """State of a specific homeostatic need"""
    current_value: float = Field(0.0, description="Current value (0.0-1.0)")
    setpoint: float = Field(0.5, description="Optimal value (0.0-1.0)")
    min_threshold: float = Field(0.2, description="Lower threshold before compensatory action")
    max_threshold: float = Field(0.8, description="Upper threshold before compensatory action")
    last_updated: datetime = Field(default_factory=datetime.now)
    decay_rate: float = Field(0.05, description="How quickly this need changes without intervention")
    importance: float = Field(1.0, description="Weight/priority of this need (0.0-1.0)")
    
    @model_validator(mode='after')
    def validate_thresholds(self) -> 'NeedState':
        """Ensure thresholds are in the correct order"""
        if self.min_threshold >= self.setpoint:
            self.min_threshold = self.setpoint - 0.1
        if self.max_threshold <= self.setpoint:
            self.max_threshold = self.setpoint + 0.1
        return self
    
    @property
    def deviation(self) -> float:
        """Calculate deviation from setpoint"""
        return abs(self.current_value - self.setpoint)
    
    @property 
    def is_deficient(self) -> bool:
        """Check if need is below minimum threshold"""
        return self.current_value < self.min_threshold
    
    @property
    def is_excessive(self) -> bool:
        """Check if need is above maximum threshold"""
        return self.current_value > self.max_threshold
    
    @property
    def is_balanced(self) -> bool:
        """Check if need is within thresholds"""
        return not (self.is_deficient or self.is_excessive)
    
    @property
    def urgency(self) -> float:
        """Calculate urgency of addressing this need (0.0-1.0)"""
        if self.is_balanced:
            return 0.0
        deviation = self.deviation
        threshold_distance = (
            self.setpoint - self.min_threshold if self.is_deficient 
            else self.max_threshold - self.setpoint
        )
        return min(1.0, (deviation / threshold_distance) * self.importance)

class HomeostaticSystem(BaseModel):
    """Maintains internal balance and stability"""
    needs: Dict[HomeostaticNeedType, NeedState] = Field(default_factory=dict)
    developmental_adaptation: Dict[str, float] = Field(
        default_factory=dict, 
        description="How needs adapt with development"
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="History of significant homeostatic events"
    )
    
    def initialize_needs(self) -> None:
        """Initialize all homeostatic needs with default values"""
        for need_type in HomeostaticNeedType:
            if need_type not in self.needs:
                self.needs[need_type] = NeedState()
    
    def update_need(
        self, 
        need_type: HomeostaticNeedType, 
        value_change: float,
        reason: Optional[str] = None
    ) -> NeedState:
        """Update a specific need with the given change"""
        if need_type not in self.needs:
            self.needs[need_type] = NeedState()
            
        need = self.needs[need_type]
        old_value = need.current_value
        need.current_value = max(0.0, min(1.0, need.current_value + value_change))
        need.last_updated = datetime.now()
        
        # Record significant changes to history
        if abs(value_change) > 0.1 or need.is_deficient or need.is_excessive:
            self.history.append({
                "need_type": need_type,
                "timestamp": need.last_updated,
                "old_value": old_value,
                "new_value": need.current_value,
                "change": value_change,
                "reason": reason
            })
            
        return need
    
    def get_imbalanced_needs(self) -> Dict[HomeostaticNeedType, NeedState]:
        """Get all needs that are outside their threshold ranges"""
        return {
            need_type: need for need_type, need in self.needs.items() 
            if not need.is_balanced
        }
    
    def get_most_urgent_need(self) -> Optional[Tuple[HomeostaticNeedType, NeedState]]:
        """Get the most urgent need requiring attention"""
        imbalanced = self.get_imbalanced_needs()
        if not imbalanced:
            return None
            
        return max(
            [(need_type, need) for need_type, need in imbalanced.items()],
            key=lambda x: x[1].urgency
        )
    
    def adapt_to_development(self, development_level: float) -> None:
        """Adapt homeostatic setpoints based on developmental stage"""
        # Adjust energy needs (higher in childhood, lower in adulthood)
        energy_setpoint = 0.7 - (development_level * 0.2)
        self.needs[HomeostaticNeedType.ENERGY].setpoint = energy_setpoint
        
        # Adjust arousal (high for infants, lower for adults)
        arousal_setpoint = 0.8 - (development_level * 0.3)
        self.needs[HomeostaticNeedType.AROUSAL].setpoint = arousal_setpoint
        
        # Adjust cognitive load (increases with development)
        cognitive_load_setpoint = 0.3 + (development_level * 0.3)
        self.needs[HomeostaticNeedType.COGNITIVE_LOAD].setpoint = cognitive_load_setpoint
        
        # Adjust social needs (higher in childhood/adolescence)
        social_curve = 0.4 + (0.3 * (1 - abs(development_level - 0.5) * 2))
        self.needs[HomeostaticNeedType.SOCIAL].setpoint = social_curve
        
        # Record adaptation
        self.developmental_adaptation = {
            "development_level": development_level,
            "energy_setpoint": energy_setpoint,
            "arousal_setpoint": arousal_setpoint,
            "cognitive_load_setpoint": cognitive_load_setpoint,
            "social_setpoint": social_curve
        }
    
    def overall_homeostatic_balance(self) -> float:
        """Calculate overall homeostatic balance (0.0-1.0)"""
        if not self.needs:
            return 1.0
            
        total_deviation = sum(need.deviation for need in self.needs.values())
        max_possible_deviation = len(self.needs)
        
        return 1.0 - (total_deviation / max_possible_deviation)

class HomeostaticResponse(BaseModel):
    """Represents a response to a homeostatic imbalance"""
    need_type: HomeostaticNeedType
    response_type: str
    intensity: float
    description: str
    expected_effect: Dict[HomeostaticNeedType, float]
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = 0
