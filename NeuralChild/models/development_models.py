from typing import Dict, List, Optional, Union, Literal, Any
from enum import Enum, auto
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
import numpy as np
from datetime import datetime, timedelta

class DevelopmentalStage(str, Enum):
    """Developmental stages of child cognitive growth"""
    NEWBORN = "newborn"                    # 0-1 month
    EARLY_INFANCY = "early_infancy"        # 1-4 months
    MIDDLE_INFANCY = "middle_infancy"      # 4-8 months
    LATE_INFANCY = "late_infancy"          # 8-12 months
    EARLY_TODDLER = "early_toddler"        # 12-18 months
    LATE_TODDLER = "late_toddler"          # 18-24 months
    EARLY_PRESCHOOL = "early_preschool"    # 2-3 years
    LATE_PRESCHOOL = "late_preschool"      # 3-5 years
    EARLY_CHILDHOOD = "early_childhood"    # 5-7 years
    MIDDLE_CHILDHOOD = "middle_childhood"  # 7-9 years

class DevelopmentalDomain(str, Enum):
    """Different domains of development"""
    COGNITIVE = "cognitive"       # General thinking abilities
    LANGUAGE = "language"         # Language acquisition
    EMOTIONAL = "emotional"       # Emotional development
    SOCIAL = "social"             # Social understanding
    MEMORY = "memory"             # Memory capabilities
    SELF_AWARENESS = "self_awareness"  # Sense of self
    MORAL = "moral"               # Moral reasoning

class MilestoneStatus(str, Enum):
    """Status of developmental milestones"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"

class DevelopmentalMilestone(BaseModel):
    """A specific developmental milestone to be achieved"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    domain: DevelopmentalDomain
    typical_stage: DevelopmentalStage
    typical_age_min: int = Field(..., description="Minimum age in months for typical achievement")
    typical_age_max: int = Field(..., description="Maximum age in months for typical achievement")
    prerequisites: List[UUID] = Field(default_factory=list, description="Prerequisite milestones")
    status: MilestoneStatus = Field(MilestoneStatus.NOT_STARTED)
    progress: float = Field(0.0, ge=0.0, le=1.0)
    achieved_at: Optional[datetime] = None
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance for future development")
    
    def update_progress(self, amount: float) -> bool:
        """Update progress toward this milestone"""
        if self.status == MilestoneStatus.ACHIEVED:
            return False
            
        if self.status == MilestoneStatus.NOT_STARTED:
            self.status = MilestoneStatus.IN_PROGRESS
            
        self.progress = min(1.0, self.progress + amount)
        
        if self.progress >= 1.0:
            self.status = MilestoneStatus.ACHIEVED
            self.achieved_at = datetime.now()
            return True
            
        return False

class LanguageMilestone(DevelopmentalMilestone):
    """Specialized milestone for language development"""
    domain: Literal[DevelopmentalDomain.LANGUAGE] = DevelopmentalDomain.LANGUAGE
    vocabulary_threshold: Optional[int] = None
    grammar_complexity: Optional[float] = None
    utterance_length: Optional[int] = None

class CognitiveMilestone(DevelopmentalMilestone):
    """Specialized milestone for cognitive development"""
    domain: Literal[DevelopmentalDomain.COGNITIVE] = DevelopmentalDomain.COGNITIVE
    abstraction_level: Optional[float] = None
    reasoning_complexity: Optional[float] = None
    problem_solving_capacity: Optional[float] = None

class EmotionalMilestone(DevelopmentalMilestone):
    """Specialized milestone for emotional development"""
    domain: Literal[DevelopmentalDomain.EMOTIONAL] = DevelopmentalDomain.EMOTIONAL
    regulation_capacity: Optional[float] = None
    emotional_complexity: Optional[float] = None
    emotion_recognition: Optional[float] = None

class SocialMilestone(DevelopmentalMilestone):
    """Specialized milestone for social development"""
    domain: Literal[DevelopmentalDomain.SOCIAL] = DevelopmentalDomain.SOCIAL
    social_awareness: Optional[float] = None
    perspective_taking: Optional[float] = None
    cooperation_level: Optional[float] = None

class MemoryMilestone(DevelopmentalMilestone):
    """Specialized milestone for memory development"""
    domain: Literal[DevelopmentalDomain.MEMORY] = DevelopmentalDomain.MEMORY
    memory_span: Optional[int] = None
    retrieval_capacity: Optional[float] = None
    encoding_efficiency: Optional[float] = None

class SelfAwarenessMilestone(DevelopmentalMilestone):
    """Specialized milestone for self-awareness development"""
    domain: Literal[DevelopmentalDomain.SELF_AWARENESS] = DevelopmentalDomain.SELF_AWARENESS
    self_recognition: Optional[float] = None
    self_concept_complexity: Optional[float] = None
    autobiographical_capacity: Optional[float] = None

class MoralMilestone(DevelopmentalMilestone):
    """Specialized milestone for moral development"""
    domain: Literal[DevelopmentalDomain.MORAL] = DevelopmentalDomain.MORAL
    moral_reasoning_level: Optional[float] = None
    empathy_level: Optional[float] = None
    fairness_understanding: Optional[float] = None

class DevelopmentalConfig(BaseModel):
    """Configuration for the developmental system"""
    start_date: datetime = Field(default_factory=datetime.now)
    time_acceleration: float = Field(720.0, description="Factor by which development is accelerated")
    development_variability: float = Field(0.2, description="Random variability in development rate")
    sensitive_period_factor: float = Field(1.5, description="Learning boost during sensitive periods")
    environmental_richness: float = Field(0.8, description="Quality of environmental stimulation")
    caregiver_interaction_quality: float = Field(0.9, description="Quality of caregiver interactions")

class DevelopmentalTracker(BaseModel):
    """System for tracking developmental progress"""
    milestones: Dict[UUID, DevelopmentalMilestone] = Field(default_factory=dict)
    config: DevelopmentalConfig = Field(default_factory=DevelopmentalConfig)
    current_stage: DevelopmentalStage = Field(DevelopmentalStage.NEWBORN)
    simulated_age_months: float = Field(0.0, ge=0.0, description="Current simulated age in months")
    stage_progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress in current stage")
    domain_development: Dict[DevelopmentalDomain, float] = Field(
        default_factory=lambda: {domain: 0.0 for domain in DevelopmentalDomain}
    )
    last_update_time: datetime = Field(default_factory=datetime.now)
    
    def add_milestone(self, milestone: DevelopmentalMilestone) -> UUID:
        """Add a milestone to track"""
        self.milestones[milestone.id] = milestone
        return milestone.id
    
    def update_milestone_progress(self, milestone_id: UUID, amount: float) -> bool:
        """Update progress on a specific milestone"""
        if milestone_id in self.milestones:
            achieved = self.milestones[milestone_id].update_progress(amount)
            
            # If achieved, update domain development
            if achieved:
                domain = self.milestones[milestone_id].domain
                importance = self.milestones[milestone_id].importance
                self.domain_development[domain] += importance * 0.1
                self._check_stage_progression()
                
            return achieved
        return False
    
    def update_simulated_time(self, real_seconds_elapsed: float) -> None:
        """Update the simulated age based on elapsed real time"""
        # Convert real seconds to simulated time
        simulated_seconds = real_seconds_elapsed * self.config.time_acceleration
        simulated_months_passed = simulated_seconds / (30 * 24 * 60 * 60)  # Approx 30 days per month
        
        # Update the simulated age
        self.simulated_age_months += simulated_months_passed
        self.last_update_time = datetime.now()
        
        # Check if we should progress to next stage
        self._check_stage_progression()
    
    def get_active_milestones(self) -> List[DevelopmentalMilestone]:
        """Get milestones that should be currently active based on age"""
        active_milestones = []
        
        for milestone in self.milestones.values():
            if (milestone.status != MilestoneStatus.ACHIEVED and 
                self.simulated_age_months >= milestone.typical_age_min and
                self.simulated_age_months <= milestone.typical_age_max * 1.5):
                
                # Check prerequisites
                prerequisites_met = True
                for prereq_id in milestone.prerequisites:
                    if (prereq_id in self.milestones and 
                        self.milestones[prereq_id].status != MilestoneStatus.ACHIEVED):
                        prerequisites_met = False
                        break
                
                if prerequisites_met:
                    active_milestones.append(milestone)
        
        return active_milestones
    
    def _check_stage_progression(self) -> None:
        """Check if child should progress to the next developmental stage"""
        # Map stages to typical age ranges (in months)
        stage_age_ranges = {
            DevelopmentalStage.NEWBORN: (0, 1),
            DevelopmentalStage.EARLY_INFANCY: (1, 4),
            DevelopmentalStage.MIDDLE_INFANCY: (4, 8),
            DevelopmentalStage.LATE_INFANCY: (8, 12),
            DevelopmentalStage.EARLY_TODDLER: (12, 18),
            DevelopmentalStage.LATE_TODDLER: (18, 24),
            DevelopmentalStage.EARLY_PRESCHOOL: (24, 36),
            DevelopmentalStage.LATE_PRESCHOOL: (36, 60),
            DevelopmentalStage.EARLY_CHILDHOOD: (60, 84),
            DevelopmentalStage.MIDDLE_CHILDHOOD: (84, 108)
        }
        
        # Get the age range for the current stage
        current_min, current_max = stage_age_ranges[self.current_stage]
        
        # Calculate progress within the current stage
        stage_duration = current_max - current_min
        stage_progress = (self.simulated_age_months - current_min) / stage_duration
        self.stage_progress = max(0.0, min(1.0, stage_progress))
        
        # Check if we should move to the next stage
        if self.simulated_age_months >= current_max:
            # Find the next stage
            stages = list(DevelopmentalStage)
            current_index = stages.index(self.current_stage)
            
            if current_index < len(stages) - 1:
                self.current_stage = stages[current_index + 1]
                self.stage_progress = 0.0

class LanguageDevelopmentMetrics(BaseModel):
    """Metrics for tracking language development"""
    vocabulary_size: int = Field(0, ge=0)
    active_vocabulary: int = Field(0, ge=0)  # Words actively used
    passive_vocabulary: int = Field(0, ge=0)  # Words understood
    grammar_complexity: float = Field(0.0, ge=0.0, le=1.0)
    mean_utterance_length: float = Field(0.0, ge=0.0)
    syntax_rules_mastered: int = Field(0, ge=0)
    communication_effectiveness: float = Field(0.0, ge=0.0, le=1.0)

class CognitiveDevelopmentMetrics(BaseModel):
    """Metrics for tracking cognitive development"""
    object_permanence: float = Field(0.0, ge=0.0, le=1.0)
    cause_effect_understanding: float = Field(0.0, ge=0.0, le=1.0)
    symbolic_thinking: float = Field(0.0, ge=0.0, le=1.0)
    problem_solving_ability: float = Field(0.0, ge=0.0, le=1.0)
    attention_span_seconds: float = Field(0.0, ge=0.0)
    classification_ability: float = Field(0.0, ge=0.0, le=1.0)
    abstraction_level: float = Field(0.0, ge=0.0, le=1.0)

class EmotionalDevelopmentMetrics(BaseModel):
    """Metrics for tracking emotional development"""
    emotions_recognized: int = Field(0, ge=0)
    emotional_regulation: float = Field(0.0, ge=0.0, le=1.0)
    emotional_complexity: float = Field(0.0, ge=0.0, le=1.0)
    emotional_expression_clarity: float = Field(0.0, ge=0.0, le=1.0)
    empathy_level: float = Field(0.0, ge=0.0, le=1.0)
    emotional_self_awareness: float = Field(0.0, ge=0.0, le=1.0)

class SocialDevelopmentMetrics(BaseModel):
    """Metrics for tracking social development"""
    attachment_security: float = Field(0.0, ge=0.0, le=1.0)
    social_referencing: float = Field(0.0, ge=0.0, le=1.0)
    joint_attention: float = Field(0.0, ge=0.0, le=1.0)
    theory_of_mind: float = Field(0.0, ge=0.0, le=1.0)
    cooperation_ability: float = Field(0.0, ge=0.0, le=1.0)
    social_rules_understood: int = Field(0, ge=0)

class MemoryDevelopmentMetrics(BaseModel):
    """Metrics for tracking memory development"""
    working_memory_capacity: int = Field(0, ge=0)
    episodic_memory_detail: float = Field(0.0, ge=0.0, le=1.0)
    memory_duration_days: float = Field(0.0, ge=0.0)
    autobiographical_memory_quality: float = Field(0.0, ge=0.0, le=1.0)
    encoding_efficiency: float = Field(0.0, ge=0.0, le=1.0)
    retrieval_accuracy: float = Field(0.0, ge=0.0, le=1.0)

class SelfAwarenessMetrics(BaseModel):
    """Metrics for tracking self-awareness development"""
    mirror_self_recognition: float = Field(0.0, ge=0.0, le=1.0)
    self_other_differentiation: float = Field(0.0, ge=0.0, le=1.0)
    self_concept_complexity: float = Field(0.0, ge=0.0, le=1.0)
    personal_pronoun_usage: float = Field(0.0, ge=0.0, le=1.0)
    autobiographical_understanding: float = Field(0.0, ge=0.0, le=1.0)
    future_self_projection: float = Field(0.0, ge=0.0, le=1.0)

class DevelopmentalMetrics(BaseModel):
    """Combined metrics for all developmental domains"""
    language: LanguageDevelopmentMetrics = Field(default_factory=LanguageDevelopmentMetrics)
    cognitive: CognitiveDevelopmentMetrics = Field(default_factory=CognitiveDevelopmentMetrics)
    emotional: EmotionalDevelopmentMetrics = Field(default_factory=EmotionalDevelopmentMetrics)
    social: SocialDevelopmentMetrics = Field(default_factory=SocialDevelopmentMetrics)
    memory: MemoryDevelopmentMetrics = Field(default_factory=MemoryDevelopmentMetrics)
    self_awareness: SelfAwarenessMetrics = Field(default_factory=SelfAwarenessMetrics)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def update_metric(self, domain: str, metric: str, value: Union[float, int]) -> bool:
        """Update a specific developmental metric"""
        if hasattr(self, domain):
            domain_metrics = getattr(self, domain)
            if hasattr(domain_metrics, metric):
                setattr(domain_metrics, metric, value)
                self.last_updated = datetime.now()
                return True
        return False