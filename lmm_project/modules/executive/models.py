from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid
import json

class PlanStep(BaseModel):
    """
    Represents a single step in a plan
    
    Each step has an action, expected outcome, and status
    """
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this step")
    action: str = Field(..., description="Action to be performed")
    description: str = Field(..., description="Description of this step")
    expected_outcome: str = Field(..., description="Expected result of this action")
    prerequisites: List[str] = Field(default_factory=list, description="IDs of steps that must be completed first")
    status: str = Field("pending", description="Current status of this step (pending, in_progress, completed, failed)")
    estimated_difficulty: float = Field(0.5, ge=0.0, le=1.0, description="Estimated difficulty of this step")
    completion_percentage: float = Field(0.0, ge=0.0, le=1.0, description="Percentage of completion")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result

class Plan(BaseModel):
    """
    Represents a complete plan with multiple steps
    
    Plans have goals, steps, and overall status information
    """
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this plan")
    goal: str = Field(..., description="The goal this plan aims to achieve")
    description: str = Field(..., description="Description of this plan")
    steps: List[PlanStep] = Field(default_factory=list, description="Steps in this plan")
    current_step_index: Optional[int] = Field(None, description="Index of the current step being executed")
    status: str = Field("created", description="Current status (created, in_progress, completed, failed, abandoned)")
    success_likelihood: float = Field(0.5, ge=0.0, le=1.0, description="Estimated likelihood of success")
    completion_percentage: float = Field(0.0, ge=0.0, le=1.0, description="Overall completion percentage")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        result['steps'] = [step.dict() for step in self.steps]
        return result

class Decision(BaseModel):
    """
    Represents a decision with options and evaluations
    
    Decisions include options, their evaluations, and the selected choice
    """
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this decision")
    context: str = Field(..., description="Context in which the decision is being made")
    options: Dict[str, Dict[str, Any]] = Field(..., description="Available options with their attributes")
    criteria: Dict[str, float] = Field(..., description="Decision criteria and their weights")
    evaluations: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Evaluations of each option on each criterion")
    option_scores: Dict[str, float] = Field(default_factory=dict, description="Overall score for each option")
    selected_option: Optional[str] = Field(None, description="The option that was selected")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in the decision")
    decision_time: float = Field(0.0, ge=0.0, description="Time taken to make the decision (seconds)")
    created_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['created_at'] = self.created_at.isoformat()
        return result

class InhibitionEvent(BaseModel):
    """
    Represents an inhibition event
    
    Records when and how inhibitory control was applied
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this event")
    inhibition_type: str = Field(..., description="Type of inhibition (response, distraction, etc.)")
    trigger: str = Field(..., description="What triggered the need for inhibition")
    strength: float = Field(..., ge=0.0, le=1.0, description="Strength of inhibition applied")
    success: bool = Field(..., description="Whether inhibition was successful")
    resource_cost: float = Field(..., ge=0.0, description="Resource cost of this inhibition")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class InhibitionState(BaseModel):
    """
    Represents the current state of the inhibition system
    
    Includes resource levels and recent inhibition events
    """
    available_resources: float = Field(1.0, ge=0.0, le=1.0, description="Currently available inhibitory resources")
    recovery_rate: float = Field(0.1, ge=0.0, le=1.0, description="Rate at which resources recover")
    recent_events: List[InhibitionEvent] = Field(default_factory=list, description="Recent inhibition events")
    threshold_adjustments: Dict[str, float] = Field(default_factory=dict, description="Context-specific threshold adjustments")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        result['recent_events'] = [event.dict() for event in self.recent_events]
        return result

class WorkingMemoryItem(BaseModel):
    """
    Represents an item in working memory
    
    Items have content, activation level, and metadata
    """
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this item")
    content: Any = Field(..., description="The content being held in working memory")
    content_type: str = Field(..., description="Type of content (visual, verbal, spatial, etc.)")
    activation: float = Field(1.0, ge=0.0, le=1.0, description="Current activation level")
    creation_time: datetime = Field(default_factory=datetime.now, description="When this item was created")
    last_access: datetime = Field(default_factory=datetime.now, description="When this item was last accessed")
    access_count: int = Field(0, ge=0, description="How many times this item has been accessed")
    tags: List[str] = Field(default_factory=list, description="Tags or categories for this item")
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['creation_time'] = self.creation_time.isoformat()
        result['last_access'] = self.last_access.isoformat()
        
        # Handle serialization of content based on type
        if isinstance(self.content, (dict, list)):
            result['content'] = self.content
        else:
            try:
                result['content'] = str(self.content)
            except:
                result['content'] = "Unserializable content"
                
        return result

class WorkingMemoryState(BaseModel):
    """
    Represents the current state of working memory
    
    Includes current contents and capacity information
    """
    items: Dict[str, WorkingMemoryItem] = Field(default_factory=dict, description="Items currently in working memory")
    capacity: int = Field(3, ge=1, description="Maximum number of items that can be held")
    capacity_utilization: float = Field(0.0, ge=0.0, le=1.0, description="Current utilization of capacity")
    focus_of_attention: Optional[str] = Field(None, description="ID of the item currently in focus")
    last_operation: Optional[str] = Field(None, description="Last operation performed")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        result['items'] = {k: v.dict() for k, v in self.items.items()}
        return result

class ExecutiveParameters(BaseModel):
    """
    Parameters controlling the executive system's behavior
    
    These adjust based on developmental level
    """
    planning_depth: int = Field(1, ge=1, description="How many steps ahead to plan")
    decision_time_allocation: float = Field(0.5, ge=0.0, le=1.0, description="How much time to allocate to decisions")
    inhibition_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Threshold for applying inhibition")
    working_memory_decay_rate: float = Field(0.1, ge=0.0, le=1.0, description="Rate of activation decay in working memory")
    cognitive_flexibility: float = Field(0.5, ge=0.0, le=1.0, description="Ability to switch between tasks or strategies")
    resource_allocation_strategy: str = Field("balanced", description="How to allocate limited resources")
    
class ExecutiveNeuralState(BaseModel):
    """
    State information for executive neural networks
    
    Tracks the state of neural networks for executive functions
    """
    planning_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of planning network")
    decision_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of decision network")
    inhibition_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of inhibition network")
    working_memory_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of working memory network")
    
    # Track recent activations for each neural component
    recent_planning_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the planning network"
    )
    recent_decision_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the decision network"
    )
    recent_inhibition_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the inhibition network"
    )
    recent_working_memory_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the working memory network"
    )
    
    # Network performance metrics
    planning_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of planning network")
    decision_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of decision network")
    inhibition_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of inhibition network")
    working_memory_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of working memory network")
    
    # Last update timestamp
    last_updated: datetime = Field(default_factory=datetime.now, description="When neural state was last updated")
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        return result
    
    def update_accuracy(self, component: str, accuracy: float) -> None:
        """
        Update the accuracy for a specific neural component
        
        Args:
            component: The component to update ('planning', 'decision', 'inhibition', 'working_memory')
            accuracy: The new accuracy value (0.0 to 1.0)
        """
        if component == 'planning':
            self.planning_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'decision':
            self.decision_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'inhibition':
            self.inhibition_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'working_memory':
            self.working_memory_accuracy = max(0.0, min(1.0, accuracy))
        
        self.last_updated = datetime.now()
    
    def add_activation(self, component: str, activation: Dict[str, Any]) -> None:
        """
        Add a recent activation for a neural component
        
        Args:
            component: The component that was activated
            activation: Dictionary with activation details
        """
        activation_with_timestamp = {
            **activation,
            "timestamp": datetime.now().isoformat()
        }
        
        if component == 'planning':
            self.recent_planning_activations.append(activation_with_timestamp)
            if len(self.recent_planning_activations) > 10:  # Keep last 10
                self.recent_planning_activations = self.recent_planning_activations[-10:]
        elif component == 'decision':
            self.recent_decision_activations.append(activation_with_timestamp)
            if len(self.recent_decision_activations) > 10:
                self.recent_decision_activations = self.recent_decision_activations[-10:]
        elif component == 'inhibition':
            self.recent_inhibition_activations.append(activation_with_timestamp)
            if len(self.recent_inhibition_activations) > 10:
                self.recent_inhibition_activations = self.recent_inhibition_activations[-10:]
        elif component == 'working_memory':
            self.recent_working_memory_activations.append(activation_with_timestamp)
            if len(self.recent_working_memory_activations) > 10:
                self.recent_working_memory_activations = self.recent_working_memory_activations[-10:]
            
        self.last_updated = datetime.now()

class ExecutiveSystemState(BaseModel):
    """
    Complete state of the executive system
    
    Combines state information from all executive functions
    """
    planning_state: Dict[str, Any] = Field(default_factory=dict, description="State of planning system")
    decision_state: Dict[str, Any] = Field(default_factory=dict, description="State of decision making system")
    inhibition_state: InhibitionState = Field(default_factory=InhibitionState, description="State of inhibition system")
    working_memory_state: WorkingMemoryState = Field(default_factory=WorkingMemoryState, description="State of working memory system")
    parameters: ExecutiveParameters = Field(default_factory=ExecutiveParameters, description="Executive system parameters")
    neural_state: ExecutiveNeuralState = Field(default_factory=ExecutiveNeuralState, description="Neural network states")
    
    # Active plans and decisions
    active_plans: Dict[str, Plan] = Field(default_factory=dict, description="Currently active plans")
    recent_decisions: List[Decision] = Field(default_factory=list, description="Recent decisions made")
    
    # System metadata
    module_id: str = Field(..., description="Module identifier")
    developmental_level: float = Field(0.0, ge=0.0, le=1.0, description="Overall developmental level")
    last_updated: datetime = Field(default_factory=datetime.now, description="When system state was last updated")
    
    def dict(self, *args, **kwargs):
        """Convert datetime to ISO format for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        result['inhibition_state'] = self.inhibition_state.dict()
        result['working_memory_state'] = self.working_memory_state.dict()
        result['neural_state'] = self.neural_state.dict()
        result['active_plans'] = {k: v.dict() for k, v in self.active_plans.items()}
        result['recent_decisions'] = [decision.dict() for decision in self.recent_decisions]
        return result
    
    def add_plan(self, plan: Plan, max_active_plans: int = 5) -> None:
        """
        Add a plan to active plans
        
        Args:
            plan: The plan to add
            max_active_plans: Maximum number of active plans to keep
        """
        self.active_plans[plan.plan_id] = plan
        
        # Limit number of active plans
        if len(self.active_plans) > max_active_plans:
            oldest_key = min(self.active_plans.keys(), key=lambda k: self.active_plans[k].created_at)
            del self.active_plans[oldest_key]
        
        self.last_updated = datetime.now()
    
    def add_decision(self, decision: Decision, max_decisions: int = 20) -> None:
        """
        Add a decision to recent decisions
        
        Args:
            decision: The decision to add
            max_decisions: Maximum number of recent decisions to keep
        """
        self.recent_decisions.append(decision)
        
        # Limit number of recent decisions
        if len(self.recent_decisions) > max_decisions:
            self.recent_decisions = self.recent_decisions[-max_decisions:]
            
        self.last_updated = datetime.now()
