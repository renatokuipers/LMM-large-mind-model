"""
Component Integration

This module provides integration mechanisms that enable sophisticated interactions
between the psychological components of the NeuralChild system. It allows components
to influence each other and share information in a more nuanced way.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

from .data_types import (
    DevelopmentalStage, Emotion, EmotionType, 
    Memory, ComponentState
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentIntegration:
    """
    Facilitates integration and information sharing between psychological components.
    
    This class provides methods for:
    1. Cross-component influence calculations
    2. Information sharing between components
    3. Developmental synchronization
    4. Emergent behavior tracking
    """
    
    def __init__(self):
        """Initialize the component integration system."""
        # Integration metrics
        self.integration_level = 0.1  # Starts minimal, increases with development
        
        # Component state tracking
        self.component_states: Dict[str, Dict[str, Any]] = {}
        
        # Cross-component influence matrices (how components affect each other)
        self._initialize_influence_matrices()
        
        # Interaction history
        self.integration_history: List[Dict[str, Any]] = []
        self.max_history_size = 50
        
        logger.info("Component integration system initialized")
    
    def _initialize_influence_matrices(self):
        """Initialize the matrices that define how components influence each other."""
        # Base influence matrix - defines how components affect each other at maximum integration
        self.influence_matrix = {
            # How memory affects other components
            "memory": {
                "language": 0.7,  # Memory strongly affects language (vocabulary recall, etc.)
                "emotional": 0.6,  # Memory affects emotional responses (past experiences)
                "consciousness": 0.8,  # Memory is crucial for self-concept
                "social": 0.6,  # Memory affects social interactions (remembering relationships)
                "cognitive": 0.9,  # Memory is essential for cognition
            },
            # How language affects other components
            "language": {
                "memory": 0.6,  # Language provides structure for memories
                "emotional": 0.5,  # Language helps label and process emotions
                "consciousness": 0.7,  # Language is important for self-reflection
                "social": 0.8,  # Language is crucial for social interaction
                "cognitive": 0.7,  # Language structures thought
            },
            # How emotional state affects other components
            "emotional": {
                "memory": 0.6,  # Emotional state affects memory formation and recall
                "language": 0.5,  # Emotions influence language use
                "consciousness": 0.6,  # Emotions impact self-awareness
                "social": 0.7,  # Emotions strongly affect social interactions
                "cognitive": 0.8,  # Emotions significantly impact thinking
            },
            # How consciousness affects other components
            "consciousness": {
                "memory": 0.5,  # Self-awareness affects memory organization
                "language": 0.6,  # Self-awareness influences communication
                "emotional": 0.7,  # Self-awareness aids emotional regulation
                "social": 0.8,  # Theory of mind crucial for social understanding
                "cognitive": 0.7,  # Metacognition improves reasoning
            },
            # How social awareness affects other components
            "social": {
                "memory": 0.5,  # Social context affects memory formation
                "language": 0.7,  # Social needs drive language development
                "emotional": 0.8,  # Social interactions trigger emotions
                "consciousness": 0.7,  # Social feedback shapes self-concept
                "cognitive": 0.6,  # Social learning affects cognition
            },
            # How cognitive abilities affect other components
            "cognitive": {
                "memory": 0.7,  # Cognition helps organize memory
                "language": 0.6,  # Reasoning affects language complexity
                "emotional": 0.6,  # Cognition helps with emotional regulation
                "consciousness": 0.8,  # Metacognition enhances self-awareness
                "social": 0.6,  # Problem-solving helps social interaction
            }
        }
        
        # Developmental scaling factors - how influence changes by developmental stage
        self.developmental_scaling = {
            DevelopmentalStage.INFANCY: 0.2,          # Limited integration in infancy
            DevelopmentalStage.EARLY_CHILDHOOD: 0.4,  # Beginning integration
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.6, # Moderate integration
            DevelopmentalStage.ADOLESCENCE: 0.8,      # Strong integration
            DevelopmentalStage.EARLY_ADULTHOOD: 1.0,  # Full integration
        }
    
    def register_component_state(self, component_id: str, component_type: str, initial_state: Dict[str, Any] = None):
        """
        Register a component's state with the integration system.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (memory, language, etc.)
            initial_state: Initial state metrics for the component (optional for backward compatibility)
        """
        if initial_state is None:
            initial_state = {}  # Default to empty dict for backward compatibility
            
        self.component_states[component_id] = {
            "type": component_type,
            "last_updated": datetime.now(),
            "metrics": initial_state
        }
        logger.info(f"Registered component state for {component_id} ({component_type})")
    
    def update_component_state(self, component_id: str, updated_state: Dict[str, Any]):
        """
        Update a component's state in the integration system.
        
        Args:
            component_id: Unique identifier for the component
            updated_state: Updated state metrics for the component
        """
        if component_id in self.component_states:
            self.component_states[component_id]["metrics"] = updated_state
            self.component_states[component_id]["last_updated"] = datetime.now()
    
    def calculate_cross_influences(
        self, 
        developmental_stage: DevelopmentalStage,
        active_component_id: str = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate how components should influence each other based on current states.
        
        Args:
            developmental_stage: Current developmental stage
            active_component_id: ID of the component that is currently active (optional)
            
        Returns:
            Dict mapping component IDs to their influences on other components
        """
        # Get the developmental scaling factor
        scaling = self.developmental_scaling[developmental_stage]
        
        # Calculate base influences
        influences = {}
        
        for source_id, source_data in self.component_states.items():
            source_type = source_data["type"]
            
            # Skip if this component type doesn't have defined influences
            if source_type not in self.influence_matrix:
                continue
            
            influences[source_id] = {}
            
            for target_id, target_data in self.component_states.items():
                # Skip self-influence
                if source_id == target_id:
                    continue
                
                target_type = target_data["type"]
                
                # Skip if this influence isn't defined
                if target_type not in self.influence_matrix[source_type]:
                    continue
                
                # Calculate base influence
                base_influence = self.influence_matrix[source_type][target_type]
                
                # Apply developmental scaling
                scaled_influence = base_influence * scaling
                
                # Boost influence of the active component
                if active_component_id and source_id == active_component_id:
                    scaled_influence *= 1.5
                
                influences[source_id][target_id] = scaled_influence
        
        return influences
    
    def apply_cross_component_effects(
        self, 
        developmental_stage: DevelopmentalStage,
        active_component_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate and apply cross-component effects based on current states.
        
        Args:
            developmental_stage: Current developmental stage
            active_component_id: ID of the component that triggered the effects
            
        Returns:
            Dictionary of effects applied to each component
        """
        effects = {}
        active_component_type = None
        
        # Find the component type of the active component
        for component_id, data in self.component_states.items():
            if component_id == active_component_id:
                active_component_type = data["type"]
                break
                
        if not active_component_type:
            return effects
            
        # Calculate effects for each component
        for target_id, target_data in self.component_states.items():
            # Skip the active component - it doesn't affect itself
            if target_id == active_component_id:
                continue
                
            target_type = target_data["type"]
            target_effects = {}
            
            # Calculate influence from each other component
            for source_id, source_data in self.component_states.items():
                # Skip self-influence
                if source_id == target_id:
                    continue
                    
                source_type = source_data["type"]
                influence = self.get_component_influence(
                    source_component=source_type,
                    target_component=target_type,
                    developmental_stage=developmental_stage
                )
                
                if influence > 0:
                    target_effects[source_id] = influence
            
            if target_effects:
                effects[target_id] = target_effects
        
        # Record a single integration event for this cross-component effect calculation
        if effects:
            self._record_integration_event(
                source_id=active_component_id,
                source_type=active_component_type,
                target_id="multiple",
                target_type="multiple",
                influence=1.0,  # Using a placeholder value
                developmental_stage=developmental_stage
            )
        
        return effects
    
    def get_component_influence(
        self, 
        source_component: str, 
        target_component: str, 
        developmental_stage: DevelopmentalStage
    ) -> float:
        """
        Get the influence value of one component on another.
        
        Args:
            source_component: The component exerting influence
            target_component: The component being influenced
            developmental_stage: Current developmental stage
            
        Returns:
            float: Influence value
        """
        # Get base influence from matrix
        if source_component not in self.influence_matrix or target_component not in self.influence_matrix[source_component]:
            return 0.0
            
        base_influence = self.influence_matrix[source_component][target_component]
        
        # Apply developmental scaling
        scaling_factor = self.developmental_scaling.get(developmental_stage.value, 1.0)
        
        return base_influence * scaling_factor
    
    def _record_integration_event(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        influence: float,
        developmental_stage: DevelopmentalStage
    ):
        """
        Record an integration event for history tracking.
        
        Args:
            source_id: ID of the source component
            source_type: Type of the source component
            target_id: ID of the target component
            target_type: Type of the target component
            influence: Calculated influence factor
            developmental_stage: Current developmental stage
        """
        event = {
            "timestamp": datetime.now(),
            "source_id": source_id,
            "source_type": source_type,
            "target_id": target_id,
            "target_type": target_type,
            "influence": influence,
            "developmental_stage": developmental_stage.value
        }
        
        self.integration_history.append(event)
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the current integration state.
        
        Returns:
            Dict of integration metrics
        """
        # Count integration events by type
        integration_counts = {}
        for event in self.integration_history:
            key = f"{event['source_type']}->{event['target_type']}"
            if key not in integration_counts:
                integration_counts[key] = 0
            integration_counts[key] += 1
        
        # Calculate average influence by type
        avg_influences = {}
        for event in self.integration_history:
            key = f"{event['source_type']}->{event['target_type']}"
            if key not in avg_influences:
                avg_influences[key] = []
            avg_influences[key].append(event['influence'])
        
        # Convert to averages
        for key, values in avg_influences.items():
            avg_influences[key] = sum(values) / len(values) if values else 0
        
        # Most active components
        component_activity = {}
        for event in self.integration_history:
            source = event['source_id']
            if source not in component_activity:
                component_activity[source] = 0
            component_activity[source] += 1
        
        # Sort by activity
        most_active = sorted(component_activity.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "integration_level": self.integration_level,
            "integration_counts": integration_counts,
            "average_influences": avg_influences,
            "most_active_components": most_active[:3] if most_active else [],
            "total_integration_events": len(self.integration_history)
        }
    
    def synchronize_development(self, developmental_stage: DevelopmentalStage):
        """
        Synchronize development across components to ensure consistency.
        
        Args:
            developmental_stage: Current developmental stage
        """
        # Update integration level based on developmental stage
        self.integration_level = self.developmental_scaling[developmental_stage]
        
        logger.info(f"Component integration synchronized to level {self.integration_level:.2f} for {developmental_stage.value} stage") 