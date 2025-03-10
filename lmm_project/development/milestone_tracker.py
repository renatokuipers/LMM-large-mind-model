"""
Milestone tracking system for the LMM project.

This module implements a system for defining, tracking, and evaluating
developmental milestones - key capabilities that mark progression 
through developmental stages.
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple, Callable

import numpy as np

from lmm_project.core.event_system import EventSystem, Event
from lmm_project.core.types import StateDict
from lmm_project.development.models import (
    MilestoneDefinition,
    MilestoneRecord,
    MilestoneStatus,
    DevelopmentalStage,
    DevelopmentConfig
)
from lmm_project.development.developmental_stages import DevelopmentalStages
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger(__name__)

class MilestoneTracker:
    """
    Tracks and manages developmental milestones.
    
    This class defines, tracks, and evaluates developmental milestones
    that mark key points in cognitive development, providing structure
    to the developmental progression.
    """
    
    def __init__(
        self,
        dev_stages: DevelopmentalStages,
        config: Optional[DevelopmentConfig] = None
    ):
        """
        Initialize the milestone tracker.
        
        Parameters:
        -----------
        dev_stages : DevelopmentalStages
            Reference to the developmental stages manager
        config : Optional[DevelopmentConfig]
            Configuration containing milestone definitions. If None, default settings are loaded.
        """
        self.event_system = EventSystem()
        self.dev_stages = dev_stages
        self._config = config or self._load_default_config()
        
        # Store milestone definitions for easy lookup
        self._milestone_definitions: Dict[str, MilestoneDefinition] = {
            m.id: m for m in self._config.milestone_definitions
        }
        
        # Track milestone status and progress
        self._milestone_records: Dict[str, MilestoneRecord] = {}
        
        # Initialize records for all defined milestones
        for milestone_id, milestone_def in self._milestone_definitions.items():
            self._milestone_records[milestone_id] = MilestoneRecord(milestone_id=milestone_id)
            
        # Track custom evaluation functions
        self._evaluation_functions: Dict[str, Callable] = {}
        
        # Last update time
        self._last_update_time = time.time()
        
        logger.info(f"Milestone tracker initialized with {len(self._milestone_definitions)} milestones")
    
    def _load_default_config(self) -> DevelopmentConfig:
        """
        Load default configuration for milestone tracker.
        
        Returns:
        --------
        DevelopmentConfig
            Default configuration with milestone definitions
        """
        # Define basic developmental milestones based on cognitive development theory
        milestones = [
            MilestoneDefinition(
                id="object_permanence",
                name="Object Permanence",
                description="Understanding that objects continue to exist even when not observed",
                typical_stage=DevelopmentalStage.INFANT,
                typical_age=0.4,
                prerequisite_milestones=[],
                module_dependencies=["perception", "memory"],
                evaluation_criteria={
                    "remembers_hidden_objects": 0.7,
                    "searches_for_disappeared_objects": 0.8
                },
                importance=1.2
            ),
            MilestoneDefinition(
                id="basic_communication",
                name="Basic Communication",
                description="Ability to express simple needs and understand basic responses",
                typical_stage=DevelopmentalStage.INFANT,
                typical_age=0.6,
                prerequisite_milestones=[],
                module_dependencies=["language", "social"],
                evaluation_criteria={
                    "vocabulary_size": 0.3,
                    "expresses_needs": 0.7,
                    "responds_to_prompts": 0.6
                },
                importance=1.5
            ),
            MilestoneDefinition(
                id="cause_effect_understanding",
                name="Cause and Effect Understanding",
                description="Understanding basic causality between actions and outcomes",
                typical_stage=DevelopmentalStage.INFANT,
                typical_age=0.5,
                prerequisite_milestones=[],
                module_dependencies=["reasoning", "perception"],
                evaluation_criteria={
                    "relates_actions_to_effects": 0.7,
                    "predicts_simple_consequences": 0.6
                },
                importance=1.3
            ),
            MilestoneDefinition(
                id="symbolic_representation",
                name="Symbolic Representation",
                description="Using symbols to represent concepts and entities",
                typical_stage=DevelopmentalStage.CHILD,
                typical_age=1.5,
                prerequisite_milestones=["basic_communication"],
                module_dependencies=["language", "reasoning"],
                evaluation_criteria={
                    "uses_symbols": 0.7,
                    "abstract_representations": 0.6
                },
                importance=1.4
            ),
            MilestoneDefinition(
                id="causal_reasoning",
                name="Causal Reasoning",
                description="Understanding complex cause-effect relationships",
                typical_stage=DevelopmentalStage.CHILD,
                typical_age=2.0,
                prerequisite_milestones=["cause_effect_understanding"],
                module_dependencies=["reasoning", "memory"],
                evaluation_criteria={
                    "identifies_complex_causes": 0.7,
                    "explains_causal_chains": 0.6,
                    "considers_multiple_factors": 0.5
                },
                importance=1.4
            ),
            MilestoneDefinition(
                id="perspective_taking",
                name="Perspective Taking",
                description="Understanding that others have different viewpoints",
                typical_stage=DevelopmentalStage.CHILD,
                typical_age=2.5,
                prerequisite_milestones=["basic_communication"],
                module_dependencies=["social", "emotional"],
                evaluation_criteria={
                    "recognizes_others_knowledge": 0.7,
                    "adapts_communication": 0.6,
                    "considers_others_preferences": 0.5
                },
                importance=1.3
            ),
            MilestoneDefinition(
                id="abstract_thinking",
                name="Abstract Thinking",
                description="Working with abstract concepts and hypotheticals",
                typical_stage=DevelopmentalStage.ADOLESCENT,
                typical_age=3.5,
                prerequisite_milestones=["symbolic_representation", "causal_reasoning"],
                module_dependencies=["reasoning", "metacognition"],
                evaluation_criteria={
                    "uses_abstract_concepts": 0.7,
                    "hypothetical_reasoning": 0.6,
                    "conceptual_frameworks": 0.5
                },
                importance=1.6
            ),
            MilestoneDefinition(
                id="metacognitive_awareness",
                name="Metacognitive Awareness",
                description="Awareness and regulation of own cognitive processes",
                typical_stage=DevelopmentalStage.ADOLESCENT,
                typical_age=4.0,
                prerequisite_milestones=["perspective_taking"],
                module_dependencies=["metacognition", "executive"],
                evaluation_criteria={
                    "monitors_own_thinking": 0.7,
                    "adapts_learning_strategies": 0.6,
                    "evaluates_own_knowledge": 0.5
                },
                importance=1.5
            ),
            MilestoneDefinition(
                id="integrated_cognition",
                name="Integrated Cognition",
                description="Seamless integration of multiple cognitive domains",
                typical_stage=DevelopmentalStage.ADULT,
                typical_age=6.5,
                prerequisite_milestones=["abstract_thinking", "metacognitive_awareness"],
                module_dependencies=["reasoning", "metacognition", "executive"],
                evaluation_criteria={
                    "domain_integration": 0.7,
                    "complex_problem_solving": 0.6,
                    "adaptive_expertise": 0.5
                },
                importance=1.7
            )
        ]
        
        # Return minimal config with just milestones
        return DevelopmentConfig(
            initial_age=0.0,
            time_acceleration=1000.0,
            stage_definitions=[],
            milestone_definitions=milestones,
            critical_period_definitions=[]
        )
    
    def register_evaluation_function(
        self, 
        milestone_id: str, 
        eval_function: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> None:
        """
        Register an evaluation function for a milestone.
        
        Parameters:
        -----------
        milestone_id : str
            ID of the milestone to evaluate
        eval_function : Callable
            Function that evaluates milestone criteria and returns progress scores
        """
        if milestone_id not in self._milestone_definitions:
            raise ValueError(f"Unknown milestone ID: {milestone_id}")
            
        self._evaluation_functions[milestone_id] = eval_function
        logger.info(f"Registered evaluation function for milestone {milestone_id}")
    
    def update(self) -> None:
        """
        Update milestone statuses based on current developmental state.
        
        This checks prerequisites and automatically updates milestone progress
        for milestones that are ready to be worked on.
        """
        current_time = time.time()
        current_age = self.dev_stages.get_age()
        current_stage = self.dev_stages.get_current_stage()
        
        # Check all milestones
        for milestone_id, definition in self._milestone_definitions.items():
            record = self._milestone_records[milestone_id]
            
            # Skip completed or skipped milestones
            if record.status in [MilestoneStatus.COMPLETED, MilestoneStatus.SKIPPED]:
                continue
                
            # Check if age appropriate for automatic activation
            age_appropriate = current_age >= (definition.typical_age * 0.8)  # Allow early start
            
            # Check if prerequisites are met
            prerequisites_met = True
            for prereq_id in definition.prerequisite_milestones:
                prereq_record = self._milestone_records.get(prereq_id)
                if not prereq_record or prereq_record.status != MilestoneStatus.COMPLETED:
                    prerequisites_met = False
                    break
            
            # If milestone should be in progress but isn't
            if age_appropriate and prerequisites_met and record.status == MilestoneStatus.NOT_STARTED:
                record.status = MilestoneStatus.IN_PROGRESS
                record.started_at = datetime.now()
                
                # Emit event for milestone activation
                self.event_system.emit(Event(
                    name="milestone_activated",
                    data={
                        "milestone_id": milestone_id,
                        "milestone_name": definition.name,
                        "age": current_age,
                        "stage": current_stage
                    }
                ))
                
                logger.info(f"Milestone activated: {definition.name}")
                
            # Check if we should skip milestone due to age
            if record.status == MilestoneStatus.NOT_STARTED and current_age > (definition.typical_age * 1.5):
                if all(self._milestone_records.get(prereq_id, MilestoneRecord(milestone_id=prereq_id)).status == MilestoneStatus.COMPLETED 
                      for prereq_id in definition.prerequisite_milestones):
                    record.status = MilestoneStatus.SKIPPED
                    record.notes = "Automatically skipped due to age"
                    
                    # Emit event for milestone skip
                    self.event_system.emit(Event(
                        name="milestone_skipped",
                        data={
                            "milestone_id": milestone_id,
                            "milestone_name": definition.name,
                            "age": current_age,
                            "stage": current_stage
                        }
                    ))
                    
                    logger.info(f"Milestone skipped: {definition.name}")
        
        self._last_update_time = current_time
    
    def evaluate_milestone(
        self, 
        milestone_id: str, 
        evaluation_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate progress on a specific milestone.
        
        Parameters:
        -----------
        milestone_id : str
            ID of the milestone to evaluate
        evaluation_data : Optional[Dict[str, Any]]
            Data to use for evaluation. If None, only updates based on registered function.
            
        Returns:
        --------
        float
            Current progress on milestone (0.0-1.0)
        """
        if milestone_id not in self._milestone_definitions:
            raise ValueError(f"Unknown milestone ID: {milestone_id}")
            
        definition = self._milestone_definitions[milestone_id]
        record = self._milestone_records[milestone_id]
        
        # Get current progress for comparison
        current_progress = record.progress
        
        # Use registered evaluation function if available
        if milestone_id in self._evaluation_functions:
            eval_function = self._evaluation_functions[milestone_id]
            eval_results = eval_function(evaluation_data or {})
            
            # Store detailed evaluation results
            record.evaluation_results.update(eval_results)
            
            # Calculate overall progress based on criteria thresholds
            criteria_total = 0.0
            criteria_count = 0
            
            for criterion, threshold in definition.evaluation_criteria.items():
                if criterion in eval_results:
                    criteria_count += 1
                    if eval_results[criterion] >= threshold:
                        criteria_total += 1.0
                    else:
                        # Partial credit for partial achievement
                        criteria_total += eval_results[criterion] / threshold
            
            # Update progress
            if criteria_count > 0:
                record.progress = criteria_total / criteria_count
        elif evaluation_data:
            # Without registered function, use direct criteria matching
            criteria_total = 0.0
            criteria_count = 0
            
            for criterion, threshold in definition.evaluation_criteria.items():
                if criterion in evaluation_data:
                    criteria_count += 1
                    value = float(evaluation_data[criterion])
                    if value >= threshold:
                        criteria_total += 1.0
                    else:
                        # Partial credit
                        criteria_total += value / threshold
                        
            # Update progress
            if criteria_count > 0:
                record.progress = criteria_total / criteria_count
        
        # If progress changed significantly, log and emit event
        if abs(record.progress - current_progress) >= 0.05:
            # Update status if needed
            if record.progress >= 0.99 and record.status != MilestoneStatus.COMPLETED:
                record.status = MilestoneStatus.COMPLETED
                record.completed_at = datetime.now()
                record.actual_age = self.dev_stages.get_age()
                
                # Emit completion event
                self.event_system.emit(Event(
                    name="milestone_completed",
                    data={
                        "milestone_id": milestone_id,
                        "milestone_name": definition.name,
                        "age": record.actual_age,
                        "stage": self.dev_stages.get_current_stage(),
                        "evaluation_results": record.evaluation_results
                    }
                ))
                
                logger.info(f"Milestone completed: {definition.name} at age {record.actual_age:.3f}")
            elif record.status != MilestoneStatus.COMPLETED:
                # Emit progress event
                self.event_system.emit(Event(
                    name="milestone_progress",
                    data={
                        "milestone_id": milestone_id,
                        "milestone_name": definition.name,
                        "previous_progress": current_progress,
                        "new_progress": record.progress,
                        "age": self.dev_stages.get_age()
                    }
                ))
                
                logger.info(f"Milestone progress: {definition.name} now at {record.progress:.2f}")
        
        return record.progress
    
    def get_milestone_status(self, milestone_id: str) -> MilestoneRecord:
        """
        Get the current status record for a milestone.
        
        Parameters:
        -----------
        milestone_id : str
            ID of the milestone to get status for
            
        Returns:
        --------
        MilestoneRecord
            Current status record for the milestone
        """
        if milestone_id not in self._milestone_records:
            raise ValueError(f"Unknown milestone ID: {milestone_id}")
            
        return self._milestone_records[milestone_id]
    
    def get_milestone_definition(self, milestone_id: str) -> MilestoneDefinition:
        """
        Get the definition for a milestone.
        
        Parameters:
        -----------
        milestone_id : str
            ID of the milestone to get definition for
            
        Returns:
        --------
        MilestoneDefinition
            Definition of the milestone
        """
        if milestone_id not in self._milestone_definitions:
            raise ValueError(f"Unknown milestone ID: {milestone_id}")
            
        return self._milestone_definitions[milestone_id]
    
    def get_active_milestones(self) -> List[str]:
        """
        Get list of currently active milestones.
        
        Returns:
        --------
        List[str]
            List of milestone IDs that are currently in progress
        """
        return [
            milestone_id for milestone_id, record in self._milestone_records.items()
            if record.status == MilestoneStatus.IN_PROGRESS
        ]
    
    def get_completed_milestones(self) -> List[str]:
        """
        Get list of completed milestones.
        
        Returns:
        --------
        List[str]
            List of milestone IDs that have been completed
        """
        return [
            milestone_id for milestone_id, record in self._milestone_records.items()
            if record.status == MilestoneStatus.COMPLETED
        ]
    
    def get_milestones_by_stage(self, stage: DevelopmentalStage) -> List[str]:
        """
        Get milestones associated with a developmental stage.
        
        Parameters:
        -----------
        stage : DevelopmentalStage
            Stage to get milestones for
            
        Returns:
        --------
        List[str]
            List of milestone IDs for the specified stage
        """
        return [
            milestone_id for milestone_id, definition in self._milestone_definitions.items()
            if definition.typical_stage == stage
        ]
    
    def add_custom_milestone(self, definition: MilestoneDefinition) -> str:
        """
        Add a custom milestone definition.
        
        Parameters:
        -----------
        definition : MilestoneDefinition
            Milestone definition to add
            
        Returns:
        --------
        str
            ID of the added milestone
        """
        # Check if milestone with this ID already exists
        if definition.id in self._milestone_definitions:
            raise ValueError(f"Milestone with ID {definition.id} already exists")
            
        # Add to definitions
        self._milestone_definitions[definition.id] = definition
        
        # Create record
        self._milestone_records[definition.id] = MilestoneRecord(milestone_id=definition.id)
        
        logger.info(f"Custom milestone added: {definition.name}")
        return definition.id
    
    def get_developmental_status(self) -> Dict[str, Any]:
        """
        Get overall developmental status report.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with developmental status metrics
        """
        current_age = self.dev_stages.get_age()
        current_stage = self.dev_stages.get_current_stage()
        
        # Count milestones by status
        status_counts = {status: 0 for status in MilestoneStatus}
        for record in self._milestone_records.values():
            status_counts[record.status] += 1
            
        # Calculate progress by stage
        stage_progress = {}
        for stage in DevelopmentalStage:
            stage_milestones = self.get_milestones_by_stage(stage)
            if not stage_milestones:
                stage_progress[stage] = 0.0
                continue
                
            total_progress = sum(
                self._milestone_records[m_id].progress 
                for m_id in stage_milestones
            )
            stage_progress[stage] = total_progress / len(stage_milestones)
            
        # Calculate overall progress
        total_progress = sum(record.progress for record in self._milestone_records.values())
        overall_progress = total_progress / len(self._milestone_records) if self._milestone_records else 0.0
        
        # Compare to typical development
        typical_milestones_completed = [
            m_id for m_id, definition in self._milestone_definitions.items()
            if definition.typical_age <= current_age
        ]
        actual_completed = self.get_completed_milestones()
        
        developmental_index = len(actual_completed) / max(1, len(typical_milestones_completed))
        
        return {
            "age": current_age,
            "stage": current_stage,
            "milestone_counts": status_counts,
            "stage_progress": stage_progress,
            "overall_progress": overall_progress,
            "expected_completed": len(typical_milestones_completed),
            "actual_completed": len(actual_completed),
            "developmental_index": developmental_index
        }
    
    def get_state(self) -> StateDict:
        """
        Get the current state as a dictionary for saving.
        
        Returns:
        --------
        StateDict
            Current state dictionary
        """
        return {
            "milestone_records": {m_id: record.dict() for m_id, record in self._milestone_records.items()},
            "last_update_time": self._last_update_time
        }
    
    def load_state(self, state: StateDict) -> None:
        """
        Load state from a state dictionary.
        
        Parameters:
        -----------
        state : StateDict
            State dictionary to load from
        """
        for m_id, record_dict in state["milestone_records"].items():
            if m_id in self._milestone_records:
                # Create new record from dict and update
                record = MilestoneRecord(**record_dict)
                self._milestone_records[m_id] = record
                
        self._last_update_time = state["last_update_time"]
        
        logger.info(f"Milestone tracker state loaded with {len(self._milestone_records)} records")
