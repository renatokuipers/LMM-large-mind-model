"""
Emotional Component

This module implements the emotional development and regulation capabilities
of the child's mind. It models how emotions are formed, regulated, and expressed
at different developmental stages.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
import inspect

import numpy as np

from ..utils.data_types import (
    Emotion, EmotionType, DevelopmentalStage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalComponent:
    """
    The EmotionalComponent handles emotional development and regulation.
    
    It models how emotions develop from simple to complex, how they are
    regulated, and how they influence behavior and cognition.
    """
    
    def __init__(self):
        """Initialize the emotional component."""
        # Core emotional abilities
        self.emotional_awareness = 0.1
        self.emotional_understanding = 0.1
        
        # Test compatibility attributes
        self.emotional_regulation = 0.1
        self.emotional_recognition = 0.2
        self.emotional_expression = 0.3
        
        # Emotional development tracking
        self.emotion_vocabulary = ["happy", "sad", "angry", "scared"]
        
        # Emotional state tracking
        self.baseline_emotions = {
            EmotionType.JOY: 0.5,  # Default moderate joy
            EmotionType.SADNESS: 0.2,  # Low baseline sadness
            EmotionType.FEAR: 0.2,  # Low baseline fear
            EmotionType.ANGER: 0.1,  # Low baseline anger
            EmotionType.SURPRISE: 0.3,  # Moderate surprise (everything is new)
            EmotionType.TRUST: 0.4,  # Moderate trust
        }
        
        # Complex emotions only develop later
        self.complex_emotions = {
            EmotionType.PRIDE: 0.0,
            EmotionType.SHAME: 0.0,
            EmotionType.GUILT: 0.0,
            EmotionType.ENVY: 0.0,
            EmotionType.JEALOUSY: 0.0,
            EmotionType.LOVE: 0.1,  # Some capacity for attachment from birth
        }
        
        # Emotional memory (recent emotional experiences)
        self.emotional_memory: List[Tuple[EmotionType, float, str]] = []  # (type, intensity, cause)
        self.memory_capacity = 10  # Number of emotional memories to retain
        
        # Temperament (innate emotional tendencies)
        self.temperament = {
            "reactivity": random.uniform(0.3, 0.7),  # How strongly emotions are felt
            "regulation": random.uniform(0.3, 0.7),  # Natural regulation ability
            "sociability": random.uniform(0.3, 0.7),  # Tendency toward social emotions
            "negative_bias": random.uniform(0.3, 0.7),  # Tendency toward negative emotions
        }
        
        logger.info("Emotional component initialized with temperament: " + 
                   ", ".join(f"{k}: {v:.2f}" for k, v in self.temperament.items()))
    
    def process_emotional_input(
        self, 
        input_emotions: List[Emotion], 
        developmental_stage: DevelopmentalStage,
        context: str = ""
    ) -> List[Emotion]:
        """
        Process emotions from input and generate the child's emotional response.
        
        Args:
            input_emotions: List of input emotions
            developmental_stage: Current developmental stage of the child
            context: Optional context text for the interaction
            
        Returns:
            List of child's resultant emotions
        """
        # Update emotional development metrics based on stage
        self._update_development_metrics(developmental_stage)
        
        # Processed emotions to return
        resulting_emotions = []
        
        # Check for integration influences if available (NEW)
        memory_influence = 0.0
        cognitive_influence = 0.0
        social_influence = 0.0
        consciousness_influence = 0.0
        
        try:
            # If the child has an integration system, get influences
            from ..core.child import Child
            child = None
            
            # Try to get the parent Child object
            for frame in inspect.stack():
                frame_self = frame[0].f_locals.get('self', None)
                if isinstance(frame_self, Child):
                    child = frame_self
                    break
            
            if child and hasattr(child, "integration"):
                # Get influences of other components on emotional processing
                memory_influence = child.integration.get_component_influence("memory", "emotional", developmental_stage)
                cognitive_influence = child.integration.get_component_influence("cognitive", "emotional", developmental_stage)
                social_influence = child.integration.get_component_influence("social", "emotional", developmental_stage)
                consciousness_influence = child.integration.get_component_influence("consciousness", "emotional", developmental_stage)
        except (ImportError, AttributeError):
            # Integration not available, continue without it
            pass
        
        # Process each input emotion
        for input_emotion in input_emotions:
            # Check if this emotion is developmentally available
            if not self._is_emotion_available(input_emotion.type, developmental_stage):
                # If emotion isn't available at this stage, convert to a basic emotion
                basic_emotion_mapping = {
                    EmotionType.PRIDE: EmotionType.JOY,
                    EmotionType.SHAME: EmotionType.SADNESS,
                    EmotionType.GUILT: EmotionType.SADNESS,
                    EmotionType.ENVY: EmotionType.ANGER,
                    EmotionType.JEALOUSY: EmotionType.ANGER,
                    EmotionType.LOVE: EmotionType.JOY,
                }
                emotion_type = basic_emotion_mapping.get(input_emotion.type, EmotionType.SURPRISE)
                
                # Reduce intensity for converted emotions
                intensity = input_emotion.intensity * 0.7
            else:
                emotion_type = input_emotion.type
                intensity = input_emotion.intensity
            
            # Apply emotional contagion based on developmental stage
            contagion_factor = self._get_contagion_factor(developmental_stage)
            
            # Modify contagion based on integration influences
            if social_influence > 0:
                # Social understanding increases emotional contagion
                contagion_factor *= (1 + social_influence * 0.5)
            
            if consciousness_influence > 0:
                # Self-awareness modifies emotional contagion - makes it more selective
                contagion_factor *= (1 - consciousness_influence * 0.3 + consciousness_influence * 0.5)
            
            # Apply contagion
            contagion_intensity = intensity * contagion_factor
            
            # Apply emotion regulation based on developmental stage
            regulated_intensity = self._regulate_emotion(emotion_type, contagion_intensity, developmental_stage)
            
            # Modify regulation based on integration influences
            if cognitive_influence > 0:
                # Cognitive abilities improve emotional regulation
                regulated_intensity *= (1 - cognitive_influence * 0.3)
            
            if memory_influence > 0:
                # Memories can either amplify or reduce emotions based on past experiences
                if random.random() < 0.5:  # Simplified representation of memory impact
                    regulated_intensity *= (1 + memory_influence * 0.2)
                else:
                    regulated_intensity *= (1 - memory_influence * 0.2)
            
            # Create child's emotional response
            cause = input_emotion.cause or f"Response to external {emotion_type.value}"
            
            # Only include emotions with meaningful intensity
            if regulated_intensity > 0.1:
                resulting_emotions.append(Emotion(
                    type=emotion_type,
                    intensity=regulated_intensity,
                    cause=cause
                ))
                
                # Store in emotional memory
                self._add_to_emotional_memory(emotion_type, regulated_intensity, cause)
        
        # If no emotions were processed, add a neutral default emotion
        if not resulting_emotions:
            resulting_emotions.append(Emotion(
                type=EmotionType.SURPRISE,
                intensity=0.3,
                cause="Default response to neutral input"
            ))
        
        # Sort by intensity (highest first)
        resulting_emotions.sort(key=lambda e: e.intensity, reverse=True)
        
        # Log emotional response
        logger.info(f"Emotional response: {', '.join([f'{e.type.value}({e.intensity:.2f})' for e in resulting_emotions])}")
        
        return resulting_emotions
    
    def _get_contagion_factor(self, stage: DevelopmentalStage) -> float:
        """
        Get the emotional contagion factor based on developmental stage.
        
        Younger children are more susceptible to emotional contagion.
        
        Args:
            stage: Current developmental stage
            
        Returns:
            Contagion factor (0-1)
        """
        if stage == DevelopmentalStage.INFANCY:
            return 0.9  # Very high contagion
        elif stage == DevelopmentalStage.EARLY_CHILDHOOD:
            return 0.7
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            return 0.5
        elif stage == DevelopmentalStage.ADOLESCENCE:
            return 0.3
        else:  # EARLY_ADULTHOOD
            return 0.2  # Low contagion
    
    def _is_emotion_available(self, emotion_type: EmotionType, stage: DevelopmentalStage) -> bool:
        """
        Check if an emotion type is available at the current developmental stage.
        
        Args:
            emotion_type: The emotion type to check
            stage: Current developmental stage
            
        Returns:
            True if the emotion is available, False otherwise
        """
        # Basic emotions are always available
        if emotion_type in [
            EmotionType.JOY, EmotionType.SADNESS, EmotionType.FEAR,
            EmotionType.ANGER, EmotionType.SURPRISE, EmotionType.TRUST
        ]:
            return True
        
        # Complex emotions develop gradually
        if emotion_type == EmotionType.LOVE:
            # Love/attachment develops early
            return stage >= DevelopmentalStage.EARLY_CHILDHOOD
        elif emotion_type in [EmotionType.PRIDE, EmotionType.SHAME]:
            # Pride and shame develop in middle childhood
            return stage >= DevelopmentalStage.MIDDLE_CHILDHOOD
        elif emotion_type in [EmotionType.GUILT, EmotionType.ENVY, EmotionType.JEALOUSY]:
            # More complex social emotions develop in adolescence
            return stage >= DevelopmentalStage.ADOLESCENCE
        
        return False
    
    def _regulate_emotion(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        stage: DevelopmentalStage
    ) -> float:
        """
        Apply emotional regulation to an emotion.
        
        Args:
            emotion_type: The type of emotion
            intensity: The initial intensity
            stage: Current developmental stage
            
        Returns:
            Regulated intensity
        """
        # Base regulation ability improves with development
        base_regulation = self.emotional_regulation_ability
        
        # Innate temperament affects regulation
        temperament_factor = self.temperament["regulation"]
        
        # Developmental stage affects regulation
        stage_factor = 0.0
        if stage == DevelopmentalStage.EARLY_CHILDHOOD:
            stage_factor = 0.1
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            stage_factor = 0.3
        elif stage == DevelopmentalStage.ADOLESCENCE:
            stage_factor = 0.5
        elif stage == DevelopmentalStage.EARLY_ADULTHOOD:
            stage_factor = 0.7
        
        # Calculate regulation strength
        regulation_strength = base_regulation * temperament_factor * (1 + stage_factor)
        
        # Apply regulation (reduce intensity)
        regulated_intensity = intensity * (1 - regulation_strength)
        
        # Ensure within bounds
        return max(0.0, min(1.0, regulated_intensity))
    
    def _add_to_emotional_memory(self, emotion_type: EmotionType, intensity: float, cause: str):
        """
        Add an emotional experience to memory.
        
        Args:
            emotion_type: The type of emotion
            intensity: The intensity of the emotion
            cause: The cause of the emotion
        """
        self.emotional_memory.append((emotion_type, intensity, cause))
        
        # Limit memory size
        if len(self.emotional_memory) > self.memory_capacity:
            self.emotional_memory.pop(0)  # Remove oldest memory
    
    def _update_development_metrics(self, stage: DevelopmentalStage):
        """
        Update emotional development metrics based on stage and experiences.
        
        Args:
            stage: Current developmental stage
        """
        # Emotional regulation improves with development
        max_regulation = 0.3  # Infancy
        if stage == DevelopmentalStage.EARLY_CHILDHOOD:
            max_regulation = 0.5
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            max_regulation = 0.7
        elif stage == DevelopmentalStage.ADOLESCENCE:
            max_regulation = 0.9
        elif stage == DevelopmentalStage.EARLY_ADULTHOOD:
            max_regulation = 1.0
        
        # Gradually improve regulation (slower than max to create developmental challenge)
        self.emotional_regulation_ability = min(
            max_regulation,
            self.emotional_regulation_ability + 0.01
        )
        
        # Emotional awareness improves with development
        max_awareness = 0.3  # Infancy
        if stage == DevelopmentalStage.EARLY_CHILDHOOD:
            max_awareness = 0.5
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            max_awareness = 0.8
        elif stage == DevelopmentalStage.ADOLESCENCE:
            max_awareness = 0.9
        elif stage == DevelopmentalStage.EARLY_ADULTHOOD:
            max_awareness = 1.0
        
        self.emotional_awareness = min(
            max_awareness,
            self.emotional_awareness + 0.01
        )
        
        # Emotional complexity improves with development
        max_complexity = 0.2  # Infancy
        if stage == DevelopmentalStage.EARLY_CHILDHOOD:
            max_complexity = 0.4
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            max_complexity = 0.6
        elif stage == DevelopmentalStage.ADOLESCENCE:
            max_complexity = 0.8
        elif stage == DevelopmentalStage.EARLY_ADULTHOOD:
            max_complexity = 1.0
        
        self.emotional_complexity = min(
            max_complexity,
            self.emotional_complexity + 0.01
        )
        
        # Update complex emotions based on complexity
        if stage >= DevelopmentalStage.EARLY_CHILDHOOD:
            self.complex_emotions[EmotionType.LOVE] = min(0.8, self.emotional_complexity * 0.8)
        
        if stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
            self.complex_emotions[EmotionType.PRIDE] = min(0.7, self.emotional_complexity * 0.6)
            self.complex_emotions[EmotionType.SHAME] = min(0.7, self.emotional_complexity * 0.6)
        
        if stage >= DevelopmentalStage.ADOLESCENCE:
            self.complex_emotions[EmotionType.GUILT] = min(0.6, self.emotional_complexity * 0.5)
            self.complex_emotions[EmotionType.ENVY] = min(0.6, self.emotional_complexity * 0.5)
            self.complex_emotions[EmotionType.JEALOUSY] = min(0.6, self.emotional_complexity * 0.5)
    
    def get_development_metrics(self) -> Dict[str, float]:
        """
        Get the current emotional development metrics.
        
        Returns:
            Dictionary of development metrics
        """
        return {
            "emotional_regulation": self.emotional_regulation_ability,
            "emotional_awareness": self.emotional_awareness,
            "emotional_complexity": self.emotional_complexity,
            "complex_emotions_developed": sum(1 for _, v in self.complex_emotions.items() if v > 0.3)
        }
    
    def get_dominant_emotion(self) -> Optional[Tuple[EmotionType, float]]:
        """
        Get the dominant emotion in the current emotional memory.
        
        Returns:
            Tuple of (emotion_type, intensity) or None if no emotions
        """
        if not self.emotional_memory:
            return None
        
        # Count occurrences of each emotion type
        emotion_counts = {}
        for emotion_type, intensity, _ in self.emotional_memory:
            if emotion_type not in emotion_counts:
                emotion_counts[emotion_type] = {"count": 0, "total_intensity": 0}
            
            emotion_counts[emotion_type]["count"] += 1
            emotion_counts[emotion_type]["total_intensity"] += intensity
        
        # Find the dominant emotion (highest average intensity)
        dominant_emotion = None
        max_avg_intensity = 0
        
        for emotion_type, data in emotion_counts.items():
            avg_intensity = data["total_intensity"] / data["count"]
            if avg_intensity > max_avg_intensity:
                max_avg_intensity = avg_intensity
                dominant_emotion = emotion_type
        
        if dominant_emotion:
            return (dominant_emotion, max_avg_intensity)
        
        return None
    
    def update_emotional_development(self, developmental_stage: DevelopmentalStage, substage: str = None) -> Dict[str, float]:
        """
        Update emotional development based on developmental stage.
        Test compatibility method.
        
        Args:
            developmental_stage: The current developmental stage
            substage: The developmental substage (optional)
            
        Returns:
            Dict of updated emotional metrics
        """
        # Call process_developmental_changes to handle the actual update
        self.process_developmental_changes(developmental_stage, substage)
        
        # Return current metrics for tests
        return {
            "emotional_regulation": self.emotional_regulation_ability,
            "emotional_awareness": self.emotional_awareness,
            "emotional_complexity": self.emotional_complexity
        }
        
    def get_emotional_regulation(self) -> float:
        """Return the current emotional regulation ability for test compatibility."""
        return self.emotional_regulation_ability 