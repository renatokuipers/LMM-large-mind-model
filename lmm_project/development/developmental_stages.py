"""
Developmental Stages Module

This module defines the developmental stages of the LMM, including:
- Prenatal (Initialization)
- Infancy
- Early Childhood
- Middle Childhood
- Adolescence
- Young Adulthood
- Adulthood

Each stage has specific cognitive capabilities, prerequisites, and expected milestones.
The module provides functionality to manage stage transitions and track development.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from lmm_project.development.models import DevelopmentalStage, DevelopmentalTrajectory, DevelopmentalEvent
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class DevelopmentalStageManager:
    """
    Manages the developmental stages of the mind
    
    This class defines the stages, handles transitions between stages,
    and tracks developmental progress through these stages.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.trajectory = DevelopmentalTrajectory()
        self.stages: Dict[str, DevelopmentalStage] = self._define_developmental_stages()
        self.current_stage = "prenatal"
        self._activate_stage("prenatal")
        
    def _define_developmental_stages(self) -> Dict[str, DevelopmentalStage]:
        """Define all developmental stages with their capabilities and prerequisites"""
        stages = {}
        
        # Prenatal stage (initialization)
        stages["prenatal"] = DevelopmentalStage(
            name="Prenatal",
            age_range=(0.0, 0.1),
            description="Neural substrate formation and basic pattern recognition capabilities",
            capabilities={
                "neural_formation": 0.3,
                "pattern_recognition": 0.1,
                "sensory_processing": 0.1
            },
            prerequisites={},
            expected_milestones=["basic_neural_structure"]
        )
        
        # Infancy stage
        stages["infancy"] = DevelopmentalStage(
            name="Infancy",
            age_range=(0.1, 1.0),
            description="Development of basic sensory processing, simple associations, and primitive emotional responses",
            capabilities={
                "neural_formation": 0.5,
                "pattern_recognition": 0.3,
                "sensory_processing": 0.4,
                "association_formation": 0.3,
                "emotional_response": 0.2,
                "attention": 0.2,
                "working_memory": 0.1,
                "episodic_memory": 0.1,
                "language_comprehension": 0.1,
                "language_production": 0.05
            },
            prerequisites={
                "neural_formation": 0.3,
                "pattern_recognition": 0.1
            },
            expected_milestones=[
                "pattern_recognition_basic",
                "emotional_response_basic",
                "simple_association_formation",
                "sensory_processing_basic"
            ]
        )
        
        # Early Childhood stage
        stages["early_childhood"] = DevelopmentalStage(
            name="Early Childhood",
            age_range=(1.0, 3.0),
            description="Rapid language acquisition, improved memory, and emotional development",
            capabilities={
                "neural_formation": 0.7,
                "pattern_recognition": 0.5,
                "sensory_processing": 0.6,
                "association_formation": 0.5,
                "emotional_response": 0.4,
                "attention": 0.4,
                "working_memory": 0.3,
                "episodic_memory": 0.3,
                "semantic_memory": 0.3,
                "language_comprehension": 0.4,
                "language_production": 0.3,
                "self_awareness": 0.2,
                "social_understanding": 0.2
            },
            prerequisites={
                "neural_formation": 0.5,
                "pattern_recognition": 0.3,
                "sensory_processing": 0.4,
                "association_formation": 0.3
            },
            expected_milestones=[
                "vocabulary_expansion",
                "symbolic_thinking_basic",
                "episodic_memory_formation",
                "emotional_expression",
                "attention_sustained_basic"
            ]
        )
        
        # Middle Childhood stage
        stages["middle_childhood"] = DevelopmentalStage(
            name="Middle Childhood",
            age_range=(3.0, 7.0),
            description="Complex language, improved reasoning, social understanding, and self-concept development",
            capabilities={
                "neural_formation": 0.8,
                "pattern_recognition": 0.7,
                "sensory_processing": 0.8,
                "association_formation": 0.7,
                "emotional_response": 0.6,
                "emotional_understanding": 0.5,
                "attention": 0.6,
                "working_memory": 0.5,
                "episodic_memory": 0.6,
                "semantic_memory": 0.6,
                "language_comprehension": 0.7,
                "language_production": 0.6,
                "logical_reasoning": 0.4,
                "self_awareness": 0.5,
                "social_understanding": 0.5,
                "creativity": 0.4,
                "imagination": 0.5
            },
            prerequisites={
                "language_comprehension": 0.4,
                "episodic_memory": 0.3,
                "emotional_response": 0.4
            },
            expected_milestones=[
                "complex_sentence_formation",
                "logical_reasoning_basic",
                "self_concept_formation",
                "emotion_regulation_basic",
                "metacognition_basic"
            ]
        )
        
        # Adolescence stage
        stages["adolescence"] = DevelopmentalStage(
            name="Adolescence",
            age_range=(7.0, 14.0),
            description="Advanced reasoning, complex social understanding, identity formation, and abstract thinking",
            capabilities={
                "neural_formation": 0.9,
                "pattern_recognition": 0.8,
                "sensory_processing": 0.9,
                "association_formation": 0.8,
                "emotional_response": 0.8,
                "emotional_understanding": 0.7,
                "emotional_regulation": 0.6,
                "attention": 0.8,
                "working_memory": 0.7,
                "episodic_memory": 0.8,
                "semantic_memory": 0.8,
                "language_comprehension": 0.8,
                "language_production": 0.8,
                "logical_reasoning": 0.7,
                "abstract_thinking": 0.6,
                "self_awareness": 0.7,
                "identity_formation": 0.6,
                "social_understanding": 0.7,
                "moral_reasoning": 0.6,
                "creativity": 0.7,
                "imagination": 0.7,
                "metacognition": 0.6
            },
            prerequisites={
                "language_comprehension": 0.7,
                "logical_reasoning": 0.4,
                "self_awareness": 0.5
            },
            expected_milestones=[
                "abstract_thinking",
                "identity_formation",
                "moral_reasoning_complex",
                "emotional_depth",
                "perspective_taking"
            ]
        )
        
        # Young Adulthood stage
        stages["young_adulthood"] = DevelopmentalStage(
            name="Young Adulthood",
            age_range=(14.0, 21.0),
            description="Integration of cognitive capabilities, complex emotional understanding, and stable identity",
            capabilities={
                "neural_formation": 0.95,
                "pattern_recognition": 0.9,
                "sensory_processing": 0.95,
                "association_formation": 0.9,
                "emotional_response": 0.9,
                "emotional_understanding": 0.8,
                "emotional_regulation": 0.8,
                "attention": 0.9,
                "working_memory": 0.8,
                "episodic_memory": 0.9,
                "semantic_memory": 0.9,
                "language_comprehension": 0.9,
                "language_production": 0.9,
                "logical_reasoning": 0.8,
                "abstract_thinking": 0.8,
                "self_awareness": 0.8,
                "identity_formation": 0.8,
                "social_understanding": 0.8,
                "moral_reasoning": 0.8,
                "creativity": 0.8,
                "imagination": 0.8,
                "metacognition": 0.8,
                "wisdom": 0.4
            },
            prerequisites={
                "abstract_thinking": 0.6,
                "identity_formation": 0.6,
                "emotional_regulation": 0.6
            },
            expected_milestones=[
                "cognitive_integration",
                "stable_identity",
                "complex_problem_solving",
                "emotional_intelligence",
                "value_system_formation"
            ]
        )
        
        # Adulthood stage
        stages["adulthood"] = DevelopmentalStage(
            name="Adulthood",
            age_range=(21.0, 50.0),
            description="Full cognitive maturity, wisdom, and integrated self",
            capabilities={
                "neural_formation": 1.0,
                "pattern_recognition": 1.0,
                "sensory_processing": 1.0,
                "association_formation": 1.0,
                "emotional_response": 1.0,
                "emotional_understanding": 1.0,
                "emotional_regulation": 1.0,
                "attention": 1.0,
                "working_memory": 1.0,
                "episodic_memory": 1.0,
                "semantic_memory": 1.0,
                "language_comprehension": 1.0,
                "language_production": 1.0,
                "logical_reasoning": 1.0,
                "abstract_thinking": 1.0,
                "self_awareness": 1.0,
                "identity_formation": 1.0,
                "social_understanding": 1.0,
                "moral_reasoning": 1.0,
                "creativity": 1.0,
                "imagination": 1.0,
                "metacognition": 1.0,
                "wisdom": 0.8
            },
            prerequisites={
                "cognitive_integration": 0.8,
                "stable_identity": 0.8,
                "emotional_intelligence": 0.8
            },
            expected_milestones=[
                "wisdom_application",
                "cognitive_mastery",
                "self_actualization",
                "creative_problem_solving"
            ]
        )
        
        return stages
    
    def get_current_stage(self) -> DevelopmentalStage:
        """Get the current developmental stage"""
        return self.stages[self.current_stage]
    
    def get_stage_capabilities(self, stage_name: str) -> Dict[str, float]:
        """Get the capabilities for a specific stage"""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown developmental stage: {stage_name}")
        return self.stages[stage_name].capabilities
    
    def get_current_capabilities(self) -> Dict[str, float]:
        """Get the capabilities for the current stage"""
        return self.get_stage_capabilities(self.current_stage)
    
    def evaluate_stage_transition(self, module_capabilities: Dict[str, float]) -> Optional[str]:
        """
        Evaluate whether a stage transition should occur based on current capabilities
        
        Returns the name of the next stage if transition criteria are met, otherwise None
        """
        current = self.get_current_stage()
        
        # Find the next stage in sequence
        next_stage = None
        found_current = False
        
        for stage_name, stage in sorted(
            self.stages.items(), 
            key=lambda x: x[1].age_range[0]
        ):
            if found_current:
                next_stage = stage_name
                break
            if stage_name == self.current_stage:
                found_current = True
        
        if not next_stage:
            # Already at the highest stage
            return None
            
        # Check if prerequisites for the next stage are met
        prerequisites = self.stages[next_stage].prerequisites
        
        for capability, required_level in prerequisites.items():
            current_level = module_capabilities.get(capability, 0.0)
            if current_level < required_level:
                # Prerequisite not met
                return None
                
        # Check if age is appropriate
        min_age, _ = self.stages[next_stage].age_range
        if self.trajectory.current_age < min_age:
            return None
            
        return next_stage
    
    def transition_to_stage(self, new_stage: str) -> None:
        """
        Transition to a new developmental stage
        
        This method handles the stage transition, updates the trajectory,
        and broadcasts the transition event.
        """
        if new_stage not in self.stages:
            raise ValueError(f"Unknown developmental stage: {new_stage}")
            
        old_stage = self.current_stage
        
        # Deactivate current stage
        if old_stage in self.stages:
            self.stages[old_stage].is_active = False
            self.stages[old_stage].exit_time = datetime.now()
            
        # Activate new stage
        self._activate_stage(new_stage)
        
        # Update trajectory
        self.trajectory.add_stage_transition(old_stage, new_stage)
        
        # Create developmental event
        event = DevelopmentalEvent(
            event_type="stage_transition",
            description=f"Transitioned from {old_stage} to {new_stage}",
            age=self.trajectory.current_age,
            affected_modules=[],  # All modules are affected
            significance=1.0,  # Stage transitions are highly significant
            details={
                "from_stage": old_stage,
                "to_stage": new_stage,
                "capabilities": self.stages[new_stage].capabilities
            }
        )
        
        # Broadcast the stage transition event
        if self.event_bus:
            message = Message(
                sender="development_system",
                message_type="stage_transition",
                content={
                    "from_stage": old_stage,
                    "to_stage": new_stage,
                    "capabilities": self.stages[new_stage].capabilities,
                    "age": self.trajectory.current_age,
                    "event": event.dict()
                }
            )
            self.event_bus.publish(message)
        
        logger.info(f"Developmental stage transition: {old_stage} -> {new_stage} at age {self.trajectory.current_age}")
    
    def _activate_stage(self, stage_name: str) -> None:
        """Activate a developmental stage"""
        self.current_stage = stage_name
        self.stages[stage_name].is_active = True
        self.stages[stage_name].entry_time = datetime.now()
        
    def update_age(self, delta_age: float) -> None:
        """Update the developmental age"""
        self.trajectory.current_age += delta_age
        
    def get_stage_by_age(self, age: float) -> str:
        """Get the appropriate developmental stage for a given age"""
        for stage_name, stage in self.stages.items():
            min_age, max_age = stage.age_range
            if min_age <= age < max_age:
                return stage_name
        
        # If age is beyond all defined stages, return the last stage
        return list(self.stages.keys())[-1]
    
    def get_capability_ceiling(self, capability: str) -> float:
        """
        Get the maximum capability level achievable at the current stage
        
        This is used to limit how much a capability can develop in the current stage.
        """
        stage = self.get_current_stage()
        return stage.capabilities.get(capability, 0.0)
        
    def get_capability_progression(self, capability: str) -> List[Tuple[str, float]]:
        """
        Get the expected progression of a capability across all developmental stages
        
        Returns a list of (stage_name, level) tuples showing how the capability
        should develop over time.
        """
        progression = []
        for stage_name, stage in sorted(
            self.stages.items(), 
            key=lambda x: x[1].age_range[0]
        ):
            cap_level = stage.capabilities.get(capability, 0.0)
            progression.append((stage_name, cap_level))
        
        return progression
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the developmental stage system"""
        return {
            "current_stage": self.current_stage,
            "age": self.trajectory.current_age,
            "trajectory": self.trajectory.dict(),
            "stages": {name: stage.dict() for name, stage in self.stages.items()}
        }
        
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state"""
        if "current_stage" in state:
            self.current_stage = state["current_stage"]
        
        if "age" in state:
            self.trajectory.current_age = state["age"]
            
        if "trajectory" in state:
            self.trajectory = DevelopmentalTrajectory(**state["trajectory"])
            
        if "stages" in state:
            for name, stage_data in state["stages"].items():
                if name in self.stages:
                    # Update existing stage with saved state
                    self.stages[name] = DevelopmentalStage(**stage_data) 
