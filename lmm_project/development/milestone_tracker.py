"""
Milestone Tracker Module

This module defines and tracks developmental milestones in the cognitive system.
Milestones represent significant achievements in the mind's development across
different domains (cognitive, linguistic, social, emotional, etc.).

The milestone system is used to:
1. Track developmental progress
2. Provide goals for the development process
3. Generate meaningful events for the mind's growth
4. Assess whether development is proceeding normally
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime
import uuid
import threading
import traceback
import json
import os
from pathlib import Path

from lmm_project.development.models import Milestone, DevelopmentalEvent, DevelopmentalDomain
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.exceptions import DevelopmentError, InitializationError

logger = logging.getLogger(__name__)

class MilestoneTracker:
    """
    Tracks developmental milestones across different cognitive domains
    
    Milestones represent significant achievements that indicate
    healthy developmental progression. This class manages the definition,
    monitoring, and achievement of these milestones.
    
    Features:
    - Thread-safe milestone tracking
    - Efficient milestone evaluation
    - Dependency tracking between milestones
    - Progress monitoring for pending milestones
    - Custom milestone registration
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the milestone tracker
        
        Args:
            event_bus: Optional event bus for publishing milestone events
            
        Raises:
            InitializationError: If initialization fails
        """
        try:
            self.event_bus = event_bus
            self._lock = threading.RLock()
            self.milestones: Dict[str, Milestone] = {}
            self.achieved_milestones: Set[str] = set()
            self.pending_milestones: Set[str] = set()
            
            # Cache for frequently accessed data
            self._cache = {
                "achieved_milestones_list": [],
                "pending_milestones_list": [],
                "milestone_by_category": {},
                "milestone_by_stage": {},
                "last_cache_update": datetime.now()
            }
            self._cache_ttl = 5.0  # Cache time-to-live in seconds
            
            # Dependency graph for milestones
            self.dependency_graph: Dict[str, Set[str]] = {}  # milestone_id -> set of dependent milestone_ids
            self.reverse_dependency_graph: Dict[str, Set[str]] = {}  # milestone_id -> set of prerequisite milestone_ids
            
            # Initialize with predefined milestones
            self._define_milestones()
            
            # Build dependency graphs
            self._build_dependency_graphs()
            
            logger.info("Milestone tracker initialized with %d milestones", len(self.milestones))
            
        except Exception as e:
            error_msg = f"Failed to initialize milestone tracker: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise InitializationError(error_msg, component="MilestoneTracker", 
                                     details={"original_error": str(e)})
    
    def _define_milestones(self) -> None:
        """
        Define the developmental milestones
        
        This method creates all the predefined milestones for the system.
        Each milestone is categorized by domain and associated with a 
        developmental stage.
        
        Raises:
            InitializationError: If milestone definition fails
        """
        try:
            # Create the milestones for each developmental domain
            
            # ==================== Prenatal / Neural Formation ====================
            basic_neural_structure = Milestone(
                name="basic_neural_structure",
                description="Formation of basic neural substrate with connection capabilities",
                category=DevelopmentalDomain.NEURAL,
                typical_age=0.05,
                associated_stage="prenatal",
                capability_requirements={"neural_formation": 0.3},
                is_essential=True
            )
            self._add_milestone(basic_neural_structure)
            
            # ==================== Perception ====================
            pattern_recognition_basic = Milestone(
                name="pattern_recognition_basic",
                description="Basic ability to recognize simple patterns in input data",
                category="perception",
                typical_age=0.2,
                associated_stage="infancy",
                capability_requirements={"pattern_recognition": 0.3},
                is_essential=True
            )
            self.milestones[pattern_recognition_basic.id] = pattern_recognition_basic
            self.pending_milestones.add(pattern_recognition_basic.id)
            
            sensory_processing_basic = Milestone(
                name="sensory_processing_basic",
                description="Basic processing of sensory input information",
                category="perception",
                typical_age=0.3,
                associated_stage="infancy",
                capability_requirements={"sensory_processing": 0.3},
                is_essential=True
            )
            self.milestones[sensory_processing_basic.id] = sensory_processing_basic
            self.pending_milestones.add(sensory_processing_basic.id)
            
            attention_sustained_basic = Milestone(
                name="attention_sustained_basic",
                description="Ability to sustain attention on relevant stimuli",
                category="attention",
                typical_age=1.0,
                associated_stage="early_childhood",
                capability_requirements={"attention": 0.4},
                prerequisite_milestones=[pattern_recognition_basic.id],
                is_essential=True
            )
            self.milestones[attention_sustained_basic.id] = attention_sustained_basic
            self.pending_milestones.add(attention_sustained_basic.id)
            
            # ==================== Emotion ====================
            emotional_response_basic = Milestone(
                name="emotional_response_basic",
                description="Development of basic emotional responses to stimuli",
                category="emotion",
                typical_age=0.5,
                associated_stage="infancy",
                capability_requirements={"emotional_response": 0.2},
                is_essential=True
            )
            self.milestones[emotional_response_basic.id] = emotional_response_basic
            self.pending_milestones.add(emotional_response_basic.id)
            
            emotional_expression = Milestone(
                name="emotional_expression",
                description="Ability to express basic emotional states",
                category="emotion",
                typical_age=1.5,
                associated_stage="early_childhood",
                capability_requirements={"emotional_response": 0.4},
                prerequisite_milestones=[emotional_response_basic.id],
                is_essential=True
            )
            self.milestones[emotional_expression.id] = emotional_expression
            self.pending_milestones.add(emotional_expression.id)
            
            emotion_regulation_basic = Milestone(
                name="emotion_regulation_basic",
                description="Basic ability to regulate emotional responses",
                category="emotion",
                typical_age=4.0,
                associated_stage="middle_childhood",
                capability_requirements={"emotional_regulation": 0.4},
                prerequisite_milestones=[emotional_expression.id],
                is_essential=True
            )
            self.milestones[emotion_regulation_basic.id] = emotion_regulation_basic
            self.pending_milestones.add(emotion_regulation_basic.id)
            
            emotional_depth = Milestone(
                name="emotional_depth",
                description="Development of complex, nuanced emotional experiences",
                category="emotion",
                typical_age=10.0,
                associated_stage="adolescence",
                capability_requirements={"emotional_understanding": 0.7},
                prerequisite_milestones=[emotion_regulation_basic.id],
                is_essential=False
            )
            self.milestones[emotional_depth.id] = emotional_depth
            self.pending_milestones.add(emotional_depth.id)
            
            emotional_intelligence = Milestone(
                name="emotional_intelligence",
                description="Integration of emotional awareness, regulation, and social understanding",
                category="emotion",
                typical_age=16.0,
                associated_stage="young_adulthood",
                capability_requirements={
                    "emotional_understanding": 0.8,
                    "emotional_regulation": 0.8,
                    "social_understanding": 0.7
                },
                prerequisite_milestones=[emotional_depth.id],
                is_essential=False
            )
            self.milestones[emotional_intelligence.id] = emotional_intelligence
            self.pending_milestones.add(emotional_intelligence.id)
            
            # ==================== Memory ====================
            simple_association_formation = Milestone(
                name="simple_association_formation",
                description="Ability to form basic associations between related inputs",
                category="memory",
                typical_age=0.4,
                associated_stage="infancy",
                capability_requirements={"association_formation": 0.3},
                is_essential=True
            )
            self.milestones[simple_association_formation.id] = simple_association_formation
            self.pending_milestones.add(simple_association_formation.id)
            
            episodic_memory_formation = Milestone(
                name="episodic_memory_formation",
                description="Formation of basic episodic memories of experiences",
                category="memory",
                typical_age=1.2,
                associated_stage="early_childhood",
                capability_requirements={"episodic_memory": 0.3},
                prerequisite_milestones=[simple_association_formation.id],
                is_essential=True
            )
            self.milestones[episodic_memory_formation.id] = episodic_memory_formation
            self.pending_milestones.add(episodic_memory_formation.id)
            
            # ==================== Language ====================
            vocabulary_expansion = Milestone(
                name="vocabulary_expansion",
                description="Rapid growth in vocabulary and word association",
                category="language",
                typical_age=2.0,
                associated_stage="early_childhood",
                capability_requirements={
                    "language_comprehension": 0.3,
                    "language_production": 0.2
                },
                is_essential=True
            )
            self.milestones[vocabulary_expansion.id] = vocabulary_expansion
            self.pending_milestones.add(vocabulary_expansion.id)
            
            complex_sentence_formation = Milestone(
                name="complex_sentence_formation",
                description="Ability to form grammatically complex sentences",
                category="language",
                typical_age=4.0,
                associated_stage="middle_childhood",
                capability_requirements={
                    "language_comprehension": 0.6,
                    "language_production": 0.5
                },
                prerequisite_milestones=[vocabulary_expansion.id],
                is_essential=True
            )
            self.milestones[complex_sentence_formation.id] = complex_sentence_formation
            self.pending_milestones.add(complex_sentence_formation.id)
            
            # ==================== Cognition ====================
            symbolic_thinking_basic = Milestone(
                name="symbolic_thinking_basic",
                description="Basic ability to use symbols to represent concepts",
                category="cognition",
                typical_age=2.0,
                associated_stage="early_childhood",
                capability_requirements={"abstract_thinking": 0.2},
                is_essential=True
            )
            self.milestones[symbolic_thinking_basic.id] = symbolic_thinking_basic
            self.pending_milestones.add(symbolic_thinking_basic.id)
            
            logical_reasoning_basic = Milestone(
                name="logical_reasoning_basic",
                description="Basic logical reasoning capabilities",
                category="cognition",
                typical_age=5.0,
                associated_stage="middle_childhood",
                capability_requirements={"logical_reasoning": 0.4},
                prerequisite_milestones=[symbolic_thinking_basic.id],
                is_essential=True
            )
            self.milestones[logical_reasoning_basic.id] = logical_reasoning_basic
            self.pending_milestones.add(logical_reasoning_basic.id)
            
            metacognition_basic = Milestone(
                name="metacognition_basic",
                description="Basic awareness of own cognitive processes",
                category="cognition",
                typical_age=6.0,
                associated_stage="middle_childhood",
                capability_requirements={
                    "metacognition": 0.3,
                    "self_awareness": 0.5
                },
                is_essential=False
            )
            self.milestones[metacognition_basic.id] = metacognition_basic
            self.pending_milestones.add(metacognition_basic.id)
            
            abstract_thinking = Milestone(
                name="abstract_thinking",
                description="Ability to think about abstract concepts and hypotheticals",
                category="cognition",
                typical_age=9.0,
                associated_stage="adolescence",
                capability_requirements={"abstract_thinking": 0.6},
                prerequisite_milestones=[logical_reasoning_basic.id],
                is_essential=True
            )
            self.milestones[abstract_thinking.id] = abstract_thinking
            self.pending_milestones.add(abstract_thinking.id)
            
            complex_problem_solving = Milestone(
                name="complex_problem_solving",
                description="Ability to solve complex, multi-step problems",
                category="cognition",
                typical_age=15.0,
                associated_stage="young_adulthood",
                capability_requirements={
                    "logical_reasoning": 0.7,
                    "abstract_thinking": 0.7,
                    "working_memory": 0.7
                },
                prerequisite_milestones=[abstract_thinking.id],
                is_essential=False
            )
            self.milestones[complex_problem_solving.id] = complex_problem_solving
            self.pending_milestones.add(complex_problem_solving.id)
            
            cognitive_integration = Milestone(
                name="cognitive_integration",
                description="Integration of multiple cognitive processes for complex thinking",
                category="cognition",
                typical_age=17.0,
                associated_stage="young_adulthood",
                capability_requirements={
                    "logical_reasoning": 0.8,
                    "abstract_thinking": 0.8,
                    "metacognition": 0.7
                },
                prerequisite_milestones=[complex_problem_solving.id],
                is_essential=False
            )
            self.milestones[cognitive_integration.id] = cognitive_integration
            self.pending_milestones.add(cognitive_integration.id)
            
            cognitive_mastery = Milestone(
                name="cognitive_mastery",
                description="Mastery of cognitive faculties and deep understanding",
                category="cognition",
                typical_age=25.0,
                associated_stage="adulthood",
                capability_requirements={
                    "logical_reasoning": 0.9,
                    "abstract_thinking": 0.9,
                    "metacognition": 0.9
                },
                prerequisite_milestones=[cognitive_integration.id],
                is_essential=False
            )
            self.milestones[cognitive_mastery.id] = cognitive_mastery
            self.pending_milestones.add(cognitive_mastery.id)
            
            # ==================== Self/Identity ====================
            self_concept_formation = Milestone(
                name="self_concept_formation",
                description="Development of a basic self-concept",
                category="identity",
                typical_age=4.0,
                associated_stage="middle_childhood",
                capability_requirements={"self_awareness": 0.5},
                is_essential=True
            )
            self.milestones[self_concept_formation.id] = self_concept_formation
            self.pending_milestones.add(self_concept_formation.id)
            
            identity_formation = Milestone(
                name="identity_formation",
                description="Development of a coherent sense of identity",
                category="identity",
                typical_age=12.0,
                associated_stage="adolescence",
                capability_requirements={"identity_formation": 0.6},
                prerequisite_milestones=[self_concept_formation.id],
                is_essential=True
            )
            self.milestones[identity_formation.id] = identity_formation
            self.pending_milestones.add(identity_formation.id)
            
            stable_identity = Milestone(
                name="stable_identity",
                description="Achievement of a stable, integrated identity",
                category="identity",
                typical_age=18.0,
                associated_stage="young_adulthood",
                capability_requirements={"identity_formation": 0.8},
                prerequisite_milestones=[identity_formation.id],
                is_essential=False
            )
            self.milestones[stable_identity.id] = stable_identity
            self.pending_milestones.add(stable_identity.id)
            
            self_actualization = Milestone(
                name="self_actualization",
                description="Integration of self-knowledge, values, and purpose",
                category="identity",
                typical_age=30.0,
                associated_stage="adulthood",
                capability_requirements={
                    "identity_formation": 0.9,
                    "self_awareness": 0.9,
                    "wisdom": 0.7
                },
                prerequisite_milestones=[stable_identity.id],
                is_essential=False
            )
            self.milestones[self_actualization.id] = self_actualization
            self.pending_milestones.add(self_actualization.id)
            
            # ==================== Social ====================
            perspective_taking = Milestone(
                name="perspective_taking",
                description="Ability to understand others' perspectives",
                category="social",
                typical_age=8.0,
                associated_stage="adolescence",
                capability_requirements={"social_understanding": 0.6},
                is_essential=True
            )
            self.milestones[perspective_taking.id] = perspective_taking
            self.pending_milestones.add(perspective_taking.id)
            
            moral_reasoning_complex = Milestone(
                name="moral_reasoning_complex",
                description="Development of complex moral reasoning capabilities",
                category="social",
                typical_age=10.0,
                associated_stage="adolescence",
                capability_requirements={"moral_reasoning": 0.6},
                prerequisite_milestones=[perspective_taking.id],
                is_essential=False
            )
            self.milestones[moral_reasoning_complex.id] = moral_reasoning_complex
            self.pending_milestones.add(moral_reasoning_complex.id)
            
            value_system_formation = Milestone(
                name="value_system_formation",
                description="Development of a coherent personal value system",
                category="social",
                typical_age=16.0,
                associated_stage="young_adulthood",
                capability_requirements={
                    "moral_reasoning": 0.7,
                    "identity_formation": 0.7
                },
                prerequisite_milestones=[moral_reasoning_complex.id],
                is_essential=False
            )
            self.milestones[value_system_formation.id] = value_system_formation
            self.pending_milestones.add(value_system_formation.id)
            
            # ==================== Creativity ====================
            creative_problem_solving = Milestone(
                name="creative_problem_solving",
                description="Ability to solve problems through creative approaches",
                category="creativity",
                typical_age=22.0,
                associated_stage="adulthood",
                capability_requirements={
                    "creativity": 0.8,
                    "imagination": 0.8,
                    "logical_reasoning": 0.7
                },
                is_essential=False
            )
            self.milestones[creative_problem_solving.id] = creative_problem_solving
            self.pending_milestones.add(creative_problem_solving.id)
            
            # ==================== Wisdom ====================
            wisdom_application = Milestone(
                name="wisdom_application",
                description="Application of accumulated knowledge and experience with wisdom",
                category="wisdom",
                typical_age=35.0,
                associated_stage="adulthood",
                capability_requirements={"wisdom": 0.7},
                is_essential=False
            )
            self.milestones[wisdom_application.id] = wisdom_application
            self.pending_milestones.add(wisdom_application.id)
            
        except Exception as e:
            error_msg = f"Failed to define milestones: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise InitializationError(error_msg, component="MilestoneTracker", 
                                     details={"original_error": str(e)})
    
    def _add_milestone(self, milestone: Milestone) -> None:
        """
        Add a milestone to the tracker
        
        Args:
            milestone: The milestone to add
            
        Raises:
            ValueError: If a milestone with the same ID already exists
        """
        with self._lock:
            if milestone.id in self.milestones:
                raise ValueError(f"Milestone with ID {milestone.id} already exists")
                
            self.milestones[milestone.id] = milestone
            self.pending_milestones.add(milestone.id)
            
            # Invalidate cache
            self._invalidate_cache()
    
    def _build_dependency_graphs(self) -> None:
        """
        Build the dependency graphs for milestones
        
        This creates two graphs:
        1. dependency_graph: Maps milestones to those that depend on them
        2. reverse_dependency_graph: Maps milestones to their prerequisites
        """
        with self._lock:
            self.dependency_graph = {}
            self.reverse_dependency_graph = {}
            
            # Initialize empty sets for all milestones
            for milestone_id in self.milestones:
                self.dependency_graph[milestone_id] = set()
                self.reverse_dependency_graph[milestone_id] = set()
            
            # Build the graphs
            for milestone_id, milestone in self.milestones.items():
                # Add prerequisites to reverse dependency graph
                for prereq_id in milestone.prerequisite_milestones:
                    if prereq_id in self.reverse_dependency_graph:
                        self.reverse_dependency_graph[milestone_id].add(prereq_id)
                    
                    # Add this milestone as dependent to each prerequisite
                    if prereq_id in self.dependency_graph:
                        self.dependency_graph[prereq_id].add(milestone_id)
            
            logger.debug("Built milestone dependency graphs")
    
    def _invalidate_cache(self) -> None:
        """Invalidate all cached data"""
        self._cache["last_cache_update"] = datetime.min
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        time_diff = (datetime.now() - self._cache["last_cache_update"]).total_seconds()
        return time_diff < self._cache_ttl
    
    def evaluate_milestones(self, capabilities: Dict[str, float]) -> List[DevelopmentalEvent]:
        """
        Evaluate which milestones have been achieved based on current capabilities
        
        Args:
            capabilities: Dictionary mapping capability names to their current levels (0.0-1.0)
            
        Returns:
            List of DevelopmentalEvents for any newly achieved milestones
            
        Raises:
            ValueError: If capabilities dictionary is invalid
            DevelopmentError: If milestone evaluation fails
        """
        if not capabilities:
            raise ValueError("Capabilities dictionary cannot be empty")
            
        try:
            with self._lock:
                events = []
                newly_achieved = []
                
                # Get list of achievable milestones (prerequisites already met)
                achievable_milestones = self._get_achievable_milestones()
                
                # Check each achievable pending milestone
                for milestone_id in achievable_milestones:
                    if milestone_id not in self.pending_milestones:
                        continue
                        
                    milestone = self.milestones[milestone_id]
                    
                    # Calculate progress based on capabilities
                    progress = milestone.calculate_progress(capabilities)
                    milestone.update_progress(progress)
                    
                    # Check if milestone is now achieved
                    if milestone.achieved and milestone_id not in self.achieved_milestones:
                        # Mark as achieved
                        self.achieved_milestones.add(milestone_id)
                        self.pending_milestones.remove(milestone_id)
                        newly_achieved.append(milestone_id)
                        
                        # Create event
                        event = DevelopmentalEvent(
                            event_type="milestone",
                            description=f"Milestone achieved: {milestone.name}",
                            age=0.0,  # Will be set by caller
                            affected_modules=[],  # Will be determined based on capability requirements
                            significance=0.7 if milestone.is_essential else 0.5,
                            details={
                                "milestone_id": milestone_id,
                                "milestone_name": milestone.name,
                                "category": milestone.category,
                                "is_essential": milestone.is_essential,
                                "typical_age": milestone.typical_age
                            }
                        )
                        
                        # Determine affected modules based on capability requirements
                        affected_modules = set()
                        for capability in milestone.capability_requirements:
                            # Extract module name from capability (e.g., "language_comprehension" -> "language")
                            if "_" in capability:
                                module = capability.split("_")[0]
                                affected_modules.add(module)
                        
                        event.affected_modules = list(affected_modules)
                        events.append(event)
                        
                        # Publish event if event bus is available
                        if self.event_bus:
                            self.event_bus.publish(Message(
                                sender="milestone_tracker",
                                message_type="milestone_achieved",
                                content={
                                    "milestone_id": milestone_id,
                                    "milestone_name": milestone.name,
                                    "category": milestone.category,
                                    "is_essential": milestone.is_essential
                                }
                            ))
                            
                        logger.info("Milestone achieved: %s", milestone.name)
                
                # Invalidate cache if any milestones were achieved
                if newly_achieved:
                    self._invalidate_cache()
                    
                return events
                
        except Exception as e:
            error_msg = f"Failed to evaluate milestones: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"original_error": str(e)})
    
    def _get_achievable_milestones(self) -> Set[str]:
        """
        Get the set of milestone IDs that are achievable (prerequisites met)
        
        Returns:
            Set of milestone IDs that can be achieved
        """
        achievable = set()
        
        for milestone_id in self.pending_milestones:
            milestone = self.milestones[milestone_id]
            
            # Check if all prerequisites are met
            prereqs_met = True
            for prereq_id in milestone.prerequisite_milestones:
                if prereq_id not in self.achieved_milestones:
                    prereqs_met = False
                    break
                    
            if prereqs_met:
                achievable.add(milestone_id)
                
        return achievable
    
    def get_milestone_by_name(self, name: str) -> Optional[Milestone]:
        """
        Get a milestone by its name
        
        Args:
            name: The name of the milestone to find
            
        Returns:
            The milestone if found, None otherwise
        """
        if not name:
            return None
            
        with self._lock:
            for milestone in self.milestones.values():
                if milestone.name == name:
                    return milestone
        return None
        
    def get_milestones_by_category(self, category: str) -> List[Milestone]:
        """
        Get all milestones for a specific category
        
        Args:
            category: The category to filter by
            
        Returns:
            List of milestones in the specified category
        """
        if not category:
            return []
            
        with self._lock:
            # Check if we can use cached data
            if "milestone_by_category" in self._cache and self._is_cache_valid():
                if category in self._cache["milestone_by_category"]:
                    return self._cache["milestone_by_category"][category]
                    
            # Build the result
            result = [m for m in self.milestones.values() if m.category == category]
            
            # Cache the result
            if "milestone_by_category" not in self._cache:
                self._cache["milestone_by_category"] = {}
            self._cache["milestone_by_category"][category] = result
            
            return result
        
    def get_milestones_by_stage(self, stage: str) -> List[Milestone]:
        """
        Get all milestones for a specific developmental stage
        
        Args:
            stage: The developmental stage to filter by
            
        Returns:
            List of milestones in the specified stage
        """
        if not stage:
            return []
            
        with self._lock:
            # Check if we can use cached data
            if "milestone_by_stage" in self._cache and self._is_cache_valid():
                if stage in self._cache["milestone_by_stage"]:
                    return self._cache["milestone_by_stage"][stage]
                    
            # Build the result
            result = [m for m in self.milestones.values() if m.associated_stage == stage]
            
            # Cache the result
            if "milestone_by_stage" not in self._cache:
                self._cache["milestone_by_stage"] = {}
            self._cache["milestone_by_stage"][stage] = result
            
            return result
        
    def get_essential_milestones(self) -> List[Milestone]:
        """
        Get all essential milestones
        
        Returns:
            List of essential milestones
        """
        with self._lock:
            # Check if we can use cached data
            if "essential_milestones" in self._cache and self._is_cache_valid():
                return self._cache["essential_milestones"]
                
            # Build the result
            result = [m for m in self.milestones.values() if m.is_essential]
            
            # Cache the result
            self._cache["essential_milestones"] = result
            
            return result
        
    def get_achieved_milestones(self) -> List[Milestone]:
        """
        Get all achieved milestones
        
        Returns:
            List of achieved milestones
        """
        with self._lock:
            # Check if we can use cached data
            if "achieved_milestones_list" in self._cache and self._is_cache_valid():
                return self._cache["achieved_milestones_list"]
                
            # Build the result
            result = [self.milestones[m_id] for m_id in self.achieved_milestones]
            
            # Cache the result
            self._cache["achieved_milestones_list"] = result
            
            return result
        
    def get_pending_milestones(self) -> List[Milestone]:
        """
        Get all pending milestones
        
        Returns:
            List of pending milestones
        """
        with self._lock:
            # Check if we can use cached data
            if "pending_milestones_list" in self._cache and self._is_cache_valid():
                return self._cache["pending_milestones_list"]
                
            # Build the result
            result = [self.milestones[m_id] for m_id in self.pending_milestones]
            
            # Cache the result
            self._cache["pending_milestones_list"] = result
            
            return result
        
    def get_milestone_dependency_tree(self) -> Dict[str, List[str]]:
        """
        Get the dependency tree of milestones
        
        Returns:
            Dictionary mapping milestone IDs to lists of IDs that depend on them
        """
        with self._lock:
            # We can use the precomputed dependency graph
            return {milestone_id: list(dependents) for milestone_id, dependents in self.dependency_graph.items()}
    
    def register_custom_milestone(
        self,
        name: str,
        description: str,
        category: str,
        typical_age: float,
        associated_stage: str,
        capability_requirements: Dict[str, float],
        prerequisite_milestones: List[str] = None,
        is_essential: bool = False
    ) -> str:
        """
        Register a custom milestone
        
        Args:
            name: The name of the milestone
            description: A description of the milestone
            category: The category of the milestone
            typical_age: The typical age at which this milestone is achieved
            associated_stage: The developmental stage this milestone is associated with
            capability_requirements: Dictionary mapping capabilities to required levels
            prerequisite_milestones: List of milestone IDs that must be achieved first
            is_essential: Whether this milestone is essential for healthy development
            
        Returns:
            The ID of the created milestone
            
        Raises:
            ValueError: If parameters are invalid
            DevelopmentError: If milestone registration fails
        """
        if not name:
            raise ValueError("Milestone name cannot be empty")
            
        if not category:
            raise ValueError("Milestone category cannot be empty")
            
        if typical_age < 0:
            raise ValueError("Typical age cannot be negative")
            
        if not associated_stage:
            raise ValueError("Associated stage cannot be empty")
            
        if not capability_requirements:
            raise ValueError("Capability requirements cannot be empty")
            
        try:
            with self._lock:
                # Check if a milestone with this name already exists
                existing = self.get_milestone_by_name(name)
                if existing:
                    raise ValueError(f"Milestone with name '{name}' already exists")
                    
                # Create the milestone
                milestone = Milestone(
                    name=name,
                    description=description,
                    category=category,
                    typical_age=typical_age,
                    associated_stage=associated_stage,
                    capability_requirements=capability_requirements,
                    prerequisite_milestones=prerequisite_milestones or [],
                    is_essential=is_essential
                )
                
                # Add to tracker
                self._add_milestone(milestone)
                
                # Update dependency graphs
                self._build_dependency_graphs()
                
                logger.info("Registered custom milestone: %s in category %s", name, category)
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(Message(
                        sender="milestone_tracker",
                        message_type="milestone_registered",
                        content={
                            "milestone_id": milestone.id,
                            "milestone_name": name,
                            "category": category,
                            "is_essential": is_essential
                        }
                    ))
                
                return milestone.id
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
                
            error_msg = f"Failed to register custom milestone: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"original_error": str(e)})
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the milestone tracker
        
        Returns:
            Dictionary with the complete state for persistence
            
        Raises:
            DevelopmentError: If state retrieval fails
        """
        try:
            with self._lock:
                return {
                    "milestones": {m_id: milestone.to_dict() for m_id, milestone in self.milestones.items()},
                    "achieved_milestones": list(self.achieved_milestones),
                    "pending_milestones": list(self.pending_milestones)
                }
        except Exception as e:
            error_msg = f"Failed to retrieve milestone tracker state: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"original_error": str(e)})
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state dictionary to load
            
        Raises:
            ValueError: If state dictionary is invalid
            DevelopmentError: If state loading fails
        """
        if not state:
            raise ValueError("State dictionary cannot be empty")
            
        try:
            with self._lock:
                if "milestones" in state:
                    # Clear existing milestones
                    self.milestones.clear()
                    self.achieved_milestones.clear()
                    self.pending_milestones.clear()
                    
                    # Load milestones
                    for milestone_id, milestone_data in state["milestones"].items():
                        self.milestones[milestone_id] = Milestone(**milestone_data)
                        
                    # Load achievement status
                    if "achieved_milestones" in state:
                        self.achieved_milestones = set(state["achieved_milestones"])
                        
                    if "pending_milestones" in state:
                        self.pending_milestones = set(state["pending_milestones"])
                    else:
                        # If pending milestones not specified, assume all non-achieved are pending
                        self.pending_milestones = set(self.milestones.keys()) - self.achieved_milestones
                    
                    # Rebuild dependency graphs
                    self._build_dependency_graphs()
                    
                    # Invalidate cache
                    self._invalidate_cache()
                    
                    logger.info("Milestone tracker state loaded with %d milestones (%d achieved, %d pending)",
                               len(self.milestones), len(self.achieved_milestones), len(self.pending_milestones))
                    
        except Exception as e:
            error_msg = f"Failed to load milestone tracker state: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"original_error": str(e)})
    
    def save_state_to_file(self, filepath: str) -> None:
        """
        Save the current state to a file
        
        Args:
            filepath: Path to save the state file
            
        Raises:
            IOError: If file cannot be written
            DevelopmentError: If state saving fails
        """
        try:
            state = self.get_state()
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info(f"Milestone tracker state saved to {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to save milestone tracker state to file: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"filepath": filepath, "original_error": str(e)})
    
    def load_state_from_file(self, filepath: str) -> None:
        """
        Load state from a file
        
        Args:
            filepath: Path to the state file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If file cannot be read
            ValueError: If file contains invalid state
            DevelopmentError: If state loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.load_state(state)
            logger.info(f"Milestone tracker state loaded from {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to load milestone tracker state from file: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise DevelopmentError(error_msg, details={"filepath": filepath, "original_error": str(e)}) 
