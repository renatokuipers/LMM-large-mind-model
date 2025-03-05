"""
Emotion component for the NeuralChild project.

This module defines the EmotionComponent class that handles the emotional
state of the neural child, modeling how emotions emerge, develop, and
interact with other psychological functions.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import random
import math
from enum import Enum
from pydantic import BaseModel, Field

from .base import NeuralComponent, ConnectionType
from ..config import DevelopmentalStage


class EmotionDimension(str, Enum):
    """Primary dimensions of emotion."""
    VALENCE = "valence"  # Positive to negative
    AROUSAL = "arousal"  # Calm to excited
    DOMINANCE = "dominance"  # Submissive to dominant


class EmotionCategory(str, Enum):
    """Discrete emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    # More complex emotions emerge in later developmental stages
    LOVE = "love"
    GUILT = "guilt"
    SHAME = "shame"
    PRIDE = "pride"
    JEALOUSY = "jealousy"
    GRATITUDE = "gratitude"
    EMPATHY = "empathy"
    AWE = "awe"
    CONTEMPT = "contempt"
    CONTENTMENT = "contentment"


class EmotionalState(BaseModel):
    """Representation of an emotional state."""
    # Dimensional representation
    dimensions: Dict[EmotionDimension, float] = Field(
        default_factory=lambda: {
            EmotionDimension.VALENCE: 0.0,
            EmotionDimension.AROUSAL: 0.0,
            EmotionDimension.DOMINANCE: 0.0
        }
    )
    
    # Categorical representation
    categories: Dict[EmotionCategory, float] = Field(default_factory=dict)
    
    # Primary active emotion
    primary_emotion: Optional[EmotionCategory] = None
    
    # Emotional stability (develops over time)
    stability: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Emotional regulation capacity (develops over time)
    regulation: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Emotional diversity (range of emotions experienced)
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)


class EmotionTrigger(BaseModel):
    """Representation of something that can trigger an emotion."""
    source: str  # Where the trigger came from
    valence: float  # Inherent valence of the trigger (-1.0 to 1.0)
    arousal: float  # Inherent arousal of the trigger (0.0 to 1.0)
    dominance: float  # Inherent dominance of the trigger (-1.0 to 1.0)
    associated_categories: List[EmotionCategory]  # Associated emotion categories
    intensity: float  # Intensity of the trigger (0.0 to 1.0)


class EmotionComponent(NeuralComponent):
    """
    Component that handles the emotional state of the neural child.
    
    This component models how emotions emerge, develop, and interact with
    other psychological functions.
    """
    
    def __init__(
        self,
        development_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
        component_id: Optional[str] = None
    ):
        """
        Initialize the emotion component.
        
        Args:
            development_stage: Current developmental stage
            component_id: Optional ID (generated if not provided)
        """
        super().__init__(
            name="Emotion",
            activation_threshold=0.2,  # Emotions have a lower threshold
            activation_decay_rate=0.05,
            learning_rate=0.08,  # Emotions learn quickly
            development_stage=development_stage,
            component_id=component_id
        )
        
        # Current emotional state
        self.state = EmotionalState()
        
        # Emotional memory (triggers that have caused emotions)
        self.emotional_memory: List[Tuple[EmotionTrigger, EmotionCategory]] = []
        
        # Emotion regulation strategies (develop over time)
        self.regulation_strategies: Set[str] = set()
        
        # Available emotions at each developmental stage
        self.available_emotions_by_stage = {
            DevelopmentalStage.PRENATAL: [
                # Even prenatally, basic emotion precursors exist
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.FEAR
            ],
            DevelopmentalStage.INFANCY: [
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.FEAR,
                EmotionCategory.SURPRISE,
                EmotionCategory.DISGUST,
                EmotionCategory.TRUST
            ],
            DevelopmentalStage.EARLY_CHILDHOOD: [
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.FEAR,
                EmotionCategory.SURPRISE,
                EmotionCategory.DISGUST,
                EmotionCategory.TRUST,
                EmotionCategory.ANTICIPATION,
                EmotionCategory.LOVE,
                EmotionCategory.GUILT,
                EmotionCategory.SHAME
            ],
            DevelopmentalStage.MIDDLE_CHILDHOOD: [
                # All basic emotions
                EmotionCategory.JOY,
                EmotionCategory.SADNESS,
                EmotionCategory.ANGER,
                EmotionCategory.FEAR,
                EmotionCategory.SURPRISE,
                EmotionCategory.DISGUST,
                EmotionCategory.TRUST,
                EmotionCategory.ANTICIPATION,
                # More complex emotions
                EmotionCategory.LOVE,
                EmotionCategory.GUILT,
                EmotionCategory.SHAME,
                EmotionCategory.PRIDE,
                EmotionCategory.JEALOUSY,
                EmotionCategory.GRATITUDE
            ],
            DevelopmentalStage.ADOLESCENCE: [
                # All emotions available
                *list(EmotionCategory)
            ],
            DevelopmentalStage.EARLY_ADULTHOOD: [
                # All emotions available
                *list(EmotionCategory)
            ],
            DevelopmentalStage.MID_ADULTHOOD: [
                # All emotions available
                *list(EmotionCategory)
            ]
        }
        
        # Initialize available emotions based on current stage
        self._update_available_emotions()
        
        # Default emotional state regulation parameters
        self.metadata.update({
            "regulation_threshold": 0.7,  # Threshold for regulation to kick in
            "regulation_strength": 0.3,  # How strongly regulation affects emotions
            "baseline_decay_rate": 0.1,  # How quickly emotions return to baseline
            "contagion_factor": 0.5,  # How strongly emotions are affected by others
            "emotion_blending_factor": 0.3  # How much emotions blend together
        })
    
    def _update_available_emotions(self) -> None:
        """Update available emotions based on developmental stage."""
        self.available_emotions = set(
            self.available_emotions_by_stage.get(
                self.development_stage, 
                [EmotionCategory.JOY, EmotionCategory.SADNESS, EmotionCategory.FEAR]
            )
        )
    
    def _on_stage_transition(
        self, 
        old_stage: DevelopmentalStage, 
        new_stage: DevelopmentalStage
    ) -> None:
        """
        Handle developmental stage transitions.
        
        Args:
            old_stage: Previous developmental stage
            new_stage: New developmental stage
        """
        # Call parent method
        super()._on_stage_transition(old_stage, new_stage)
        
        # Update available emotions
        self._update_available_emotions()
        
        # Update emotional stability based on stage
        stability_by_stage = {
            DevelopmentalStage.PRENATAL: 0.1,
            DevelopmentalStage.INFANCY: 0.2,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.3,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.5,
            DevelopmentalStage.ADOLESCENCE: 0.6,  # Dips in adolescence
            DevelopmentalStage.EARLY_ADULTHOOD: 0.8,
            DevelopmentalStage.MID_ADULTHOOD: 0.9
        }
        self.state.stability = stability_by_stage.get(new_stage, 0.5)
        
        # Update emotional regulation based on stage
        regulation_by_stage = {
            DevelopmentalStage.PRENATAL: 0.0,
            DevelopmentalStage.INFANCY: 0.1,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.3,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.5,
            DevelopmentalStage.ADOLESCENCE: 0.6,
            DevelopmentalStage.EARLY_ADULTHOOD: 0.8,
            DevelopmentalStage.MID_ADULTHOOD: 0.9
        }
        self.state.regulation = regulation_by_stage.get(new_stage, 0.5)
        
        # Add regulation strategies based on stage
        if new_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            self.regulation_strategies.add("seeking comfort")
        elif new_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            self.regulation_strategies.add("seeking comfort")
            self.regulation_strategies.add("distraction")
            self.regulation_strategies.add("simple verbalization")
        elif new_stage == DevelopmentalStage.ADOLESCENCE:
            self.regulation_strategies.add("seeking comfort")
            self.regulation_strategies.add("distraction")
            self.regulation_strategies.add("verbalization")
            self.regulation_strategies.add("cognitive reappraisal")
        elif new_stage in [DevelopmentalStage.EARLY_ADULTHOOD, DevelopmentalStage.MID_ADULTHOOD]:
            self.regulation_strategies.add("seeking comfort")
            self.regulation_strategies.add("distraction")
            self.regulation_strategies.add("verbalization")
            self.regulation_strategies.add("cognitive reappraisal")
            self.regulation_strategies.add("mindfulness")
            self.regulation_strategies.add("problem-solving")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process emotional inputs and update emotional state.
        
        Args:
            inputs: Dictionary containing:
                - 'triggers': List of EmotionTrigger objects
                - 'context': Optional contextual information
                - 'perception_input': Optional input from perception component
                - 'language_input': Optional input from language component
                - 'social_input': Optional input from social component
                
        Returns:
            Dictionary containing:
                - 'emotional_state': Current emotional state
                - 'primary_emotion': Primary active emotion
                - 'emotional_response': Generated emotional response
                - 'regulation_applied': Whether emotion regulation was applied
        """
        triggers = inputs.get('triggers', [])
        context = inputs.get('context', {})
        
        # Increase activation based on triggers
        if triggers:
            activation_increase = min(1.0, sum(t.intensity for t in triggers) / len(triggers))
            self.activate(activation_increase)
        
        # If activated above threshold, process emotions
        if self.activation >= self.activation_threshold:
            # Process each trigger
            for trigger in triggers:
                self._process_trigger(trigger)
            
            # Update emotional state
            self._update_emotional_state(context)
            
            # Apply emotion regulation if developed enough
            regulation_applied = False
            if self.state.regulation > 0.2:  # Minimum regulation capacity
                regulation_applied = self._apply_regulation()
            
            # Generate emotional response
            emotional_response = self._generate_response()
            
            # Update metrics
            self._update_metrics()
            
            # Store result
            result = {
                'emotional_state': self.state,
                'primary_emotion': self.state.primary_emotion,
                'emotional_response': emotional_response,
                'regulation_applied': regulation_applied,
                'activation': self.activation
            }
            
            # Add trigger to emotional memory
            if self.state.primary_emotion and triggers:
                # Just store the strongest trigger
                strongest_trigger = max(triggers, key=lambda t: t.intensity)
                self.emotional_memory.append((strongest_trigger, self.state.primary_emotion))
                # Keep memory at a reasonable size
                if len(self.emotional_memory) > 100:
                    self.emotional_memory.pop(0)
            
            return result
        else:
            # Not activated enough, return current state
            return {
                'emotional_state': self.state,
                'primary_emotion': self.state.primary_emotion,
                'emotional_response': None,
                'regulation_applied': False,
                'activation': self.activation
            }
    
    def _process_trigger(self, trigger: EmotionTrigger) -> None:
        """
        Process an emotion trigger and update dimensional emotions.
        
        Args:
            trigger: The emotion trigger to process
        """
        # Update dimensional emotions
        self.state.dimensions[EmotionDimension.VALENCE] = self._blend_emotion(
            self.state.dimensions[EmotionDimension.VALENCE],
            trigger.valence,
            factor=trigger.intensity
        )
        
        self.state.dimensions[EmotionDimension.AROUSAL] = self._blend_emotion(
            self.state.dimensions[EmotionDimension.AROUSAL],
            trigger.arousal,
            factor=trigger.intensity
        )
        
        self.state.dimensions[EmotionDimension.DOMINANCE] = self._blend_emotion(
            self.state.dimensions[EmotionDimension.DOMINANCE],
            trigger.dominance,
            factor=trigger.intensity
        )
        
        # Update categorical emotions based on associated categories
        for category in trigger.associated_categories:
            # Skip emotions not available at current developmental stage
            if category not in self.available_emotions:
                continue
                
            # Get current intensity for this category
            current_intensity = self.state.categories.get(category, 0.0)
            
            # Calculate new intensity
            new_intensity = self._blend_emotion(
                current_intensity,
                trigger.intensity,
                factor=trigger.intensity
            )
            
            # Update category
            self.state.categories[category] = new_intensity
    
    def _blend_emotion(self, current: float, new_value: float, factor: float = 0.5) -> float:
        """
        Blend a current emotional value with a new one.
        
        Args:
            current: Current emotional value
            new_value: New emotional value
            factor: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended emotional value
        """
        # Apply factor
        adjusted_factor = factor * self.metadata["emotion_blending_factor"]
        
        # Apply stability as a moderating influence
        stability_effect = self.state.stability * (1.0 - adjusted_factor)
        
        # Calculate blended value
        blended = current * (1.0 - adjusted_factor + stability_effect) + new_value * adjusted_factor
        
        # Ensure value is within range
        return max(-1.0, min(1.0, blended))
    
    def _update_emotional_state(self, context: Dict[str, Any]) -> None:
        """
        Update the overall emotional state.
        
        Args:
            context: Contextual information
        """
        # Determine primary emotion from categories
        if self.state.categories:
            # Get the strongest emotion
            self.state.primary_emotion = max(
                self.state.categories.items(),
                key=lambda x: x[1]
            )[0]
        else:
            self.state.primary_emotion = None
        
        # Apply emotional decay (return to baseline)
        baseline_decay = self.metadata["baseline_decay_rate"] * (1.0 + self.state.stability)
        
        # Decay dimensional emotions
        for dimension in EmotionDimension:
            current = self.state.dimensions[dimension]
            # Decay toward zero
            self.state.dimensions[dimension] = current * (1.0 - baseline_decay)
        
        # Decay categorical emotions
        for category in list(self.state.categories.keys()):
            current = self.state.categories[category]
            # Decay the emotion
            self.state.categories[category] = current * (1.0 - baseline_decay)
            
            # Remove if too weak
            if self.state.categories[category] < 0.1:
                del self.state.categories[category]
        
        # Update emotional diversity based on number of emotions experienced
        num_emotions = len(self.state.categories)
        max_emotions = len(self.available_emotions)
        self.state.diversity = min(1.0, num_emotions / max(1, max_emotions))
    
    def _apply_regulation(self) -> bool:
        """
        Apply emotional regulation if needed.
        
        Returns:
            Whether regulation was applied
        """
        # Don't regulate if no primary emotion
        if not self.state.primary_emotion:
            return False
        
        # Check if regulation is needed (if primary emotion is too intense)
        primary_intensity = self.state.categories.get(self.state.primary_emotion, 0.0)
        
        if primary_intensity > self.metadata["regulation_threshold"]:
            # Calculate regulation strength based on capacity
            regulation_strength = self.metadata["regulation_strength"] * self.state.regulation
            
            # Apply regulation to primary emotion
            self.state.categories[self.state.primary_emotion] *= (1.0 - regulation_strength)
            
            # Also dampen arousal
            self.state.dimensions[EmotionDimension.AROUSAL] *= (1.0 - regulation_strength)
            
            return True
            
        return False
    
    def _generate_response(self) -> Dict[str, Any]:
        """
        Generate an emotional response based on current state.
        
        Returns:
            Response object with expression, intensity, and description
        """
        # Simple responses based on primary emotion
        if not self.state.primary_emotion:
            return {
                "expression": "neutral",
                "intensity": 0.0,
                "description": "No significant emotional response"
            }
        
        # Get intensity of primary emotion
        intensity = self.state.categories.get(self.state.primary_emotion, 0.0)
        
        # Generate expression based on emotion and developmental stage
        expression_mapping = {
            EmotionCategory.JOY: ["smile", "laugh", "excitement"],
            EmotionCategory.SADNESS: ["frown", "cry", "withdraw"],
            EmotionCategory.ANGER: ["frown", "yell", "tantrum"],
            EmotionCategory.FEAR: ["widened eyes", "freeze", "seek comfort"],
            EmotionCategory.SURPRISE: ["widened eyes", "gasp", "startle"],
            EmotionCategory.DISGUST: ["wrinkled nose", "recoil", "spit"],
            EmotionCategory.TRUST: ["approach", "relax", "open posture"],
            EmotionCategory.ANTICIPATION: ["lean forward", "excitement", "alertness"],
            EmotionCategory.LOVE: ["smile", "approach", "hugging motion"],
            EmotionCategory.GUILT: ["lowered gaze", "hunched posture", "apology"],
            EmotionCategory.SHAME: ["hide face", "avoid eye contact", "withdraw"],
            EmotionCategory.PRIDE: ["upright posture", "smile", "display"],
            EmotionCategory.JEALOUSY: ["frown", "watching", "possessiveness"],
            EmotionCategory.GRATITUDE: ["smile", "approach", "thanking"],
            EmotionCategory.EMPATHY: ["mirrored expression", "concerned look", "comforting"],
            EmotionCategory.AWE: ["widened eyes", "open mouth", "stillness"],
            EmotionCategory.CONTEMPT: ["sneer", "dismissive gesture", "turning away"],
            EmotionCategory.CONTENTMENT: ["relaxed posture", "gentle smile", "calm breathing"]
        }
        
        # Select expression based on developmental stage
        expressions = expression_mapping.get(self.state.primary_emotion, ["neutral expression"])
        
        if self.development_stage in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
            # Simplest expression
            expression = expressions[0] if expressions else "neutral expression"
        elif self.development_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Basic expressions
            expression = random.choice(expressions[:2]) if len(expressions) >= 2 else expressions[0]
        else:
            # Full range of expressions
            expression = random.choice(expressions)
        
        # Generate description based on emotion
        description_templates = {
            EmotionCategory.JOY: [
                "Feeling happy",
                "Experiencing joy",
                "Feeling pleased"
            ],
            EmotionCategory.SADNESS: [
                "Feeling sad",
                "Experiencing sadness",
                "Feeling down"
            ],
            # Add templates for other emotions...
        }
        
        # Get template for primary emotion or use generic one
        templates = description_templates.get(
            self.state.primary_emotion, 
            [f"Experiencing {self.state.primary_emotion.value}"]
        )
        
        # Select template
        description = random.choice(templates)
        
        # Add intensity qualifier based on intensity
        if intensity > 0.8:
            intensity_qualifier = "intensely"
        elif intensity > 0.5:
            intensity_qualifier = "moderately"
        elif intensity > 0.2:
            intensity_qualifier = "slightly"
        else:
            intensity_qualifier = "faintly"
        
        # Combine description with intensity (if beyond early development)
        if self.development_stage not in [DevelopmentalStage.PRENATAL, DevelopmentalStage.INFANCY]:
            description = f"{description} {intensity_qualifier}"
        
        return {
            "expression": expression,
            "intensity": intensity,
            "description": description
        }
    
    def _update_metrics(self) -> None:
        """Update emotional development metrics."""
        # Update confidence based on alignment of dimensional and categorical emotions
        # A well-developed emotional system has alignment between these
        
        # Check if primary emotion's typical dimensions match current dimensions
        if self.state.primary_emotion:
            success_rate = 0.7  # Base rate
            
            # Simple mappings of emotions to expected dimensional values
            dimension_mappings = {
                EmotionCategory.JOY: {EmotionDimension.VALENCE: 0.8},
                EmotionCategory.SADNESS: {EmotionDimension.VALENCE: -0.8},
                EmotionCategory.ANGER: {EmotionDimension.VALENCE: -0.6, EmotionDimension.AROUSAL: 0.8},
                EmotionCategory.FEAR: {EmotionDimension.VALENCE: -0.7, EmotionDimension.AROUSAL: 0.7},
                # Add mappings for other emotions...
            }
            
            # Get expected dimensions for primary emotion
            expected_dimensions = dimension_mappings.get(self.state.primary_emotion, {})
            
            # Calculate alignment
            if expected_dimensions:
                alignment_scores = []
                for dimension, expected_value in expected_dimensions.items():
                    actual_value = self.state.dimensions.get(dimension, 0.0)
                    # Calculate how close the actual value is to expected (0.0-1.0)
                    alignment = 1.0 - min(1.0, abs(actual_value - expected_value))
                    alignment_scores.append(alignment)
                
                # Average alignment across dimensions
                if alignment_scores:
                    avg_alignment = sum(alignment_scores) / len(alignment_scores)
                    success_rate = (success_rate + avg_alignment) / 2
            
            # Update confidence
            self.update_confidence(success_rate)
    
    def create_emotion_trigger(self, 
        source: str,
        input_text: str,
        intensity: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionTrigger:
        """
        Create an emotion trigger from input text.
        
        This is a simplified version that maps certain words to emotions.
        In a real system, this would use more sophisticated NLP.
        
        Args:
            source: Source of the trigger
            input_text: Text input that might trigger emotions
            intensity: Base intensity of the trigger
            context: Optional contextual information
            
        Returns:
            An EmotionTrigger object
        """
        # Default dimensional values
        valence = 0.0
        arousal = 0.5
        dominance = 0.0
        
        # Default categories
        categories = []
        
        # Simple keyword mapping (would be much more sophisticated in a real system)
        joy_words = ["happy", "joy", "smile", "laugh", "love", "good", "wonderful", "play", "fun"]
        sad_words = ["sad", "cry", "unhappy", "bad", "hurt", "pain", "alone", "miss"]
        fear_words = ["scared", "afraid", "fear", "danger", "dark", "monster", "loud"]
        anger_words = ["angry", "mad", "hate", "upset", "yell", "break", "stop", "no"]
        
        # Check for emotional words
        lower_text = input_text.lower()
        
        # Count matches for each category
        joy_count = sum(word in lower_text for word in joy_words)
        sad_count = sum(word in lower_text for word in sad_words)
        fear_count = sum(word in lower_text for word in fear_words)
        anger_count = sum(word in lower_text for word in anger_words)
        
        # Determine primary emotions based on word counts
        if joy_count > 0:
            categories.append(EmotionCategory.JOY)
            valence += 0.7 * min(1.0, joy_count * 0.3)
            arousal += 0.3 * min(1.0, joy_count * 0.3)
            dominance += 0.3 * min(1.0, joy_count * 0.3)
        
        if sad_count > 0:
            categories.append(EmotionCategory.SADNESS)
            valence -= 0.7 * min(1.0, sad_count * 0.3)
            arousal -= 0.2 * min(1.0, sad_count * 0.3)
            dominance -= 0.4 * min(1.0, sad_count * 0.3)
        
        if fear_count > 0:
            categories.append(EmotionCategory.FEAR)
            valence -= 0.6 * min(1.0, fear_count * 0.3)
            arousal += 0.6 * min(1.0, fear_count * 0.3)
            dominance -= 0.7 * min(1.0, fear_count * 0.3)
        
        if anger_count > 0:
            categories.append(EmotionCategory.ANGER)
            valence -= 0.6 * min(1.0, anger_count * 0.3)
            arousal += 0.7 * min(1.0, anger_count * 0.3)
            dominance += 0.4 * min(1.0, anger_count * 0.3)
        
        # No emotional content detected
        if not categories:
            categories = [EmotionCategory.NEUTRAL if hasattr(EmotionCategory, "NEUTRAL") else EmotionCategory.JOY]
            # Keep default dimensional values
        
        # Account for context
        if context:
            # If there's a tone specified in context, adjust dimensions
            if "tone" in context:
                tone = context["tone"].lower()
                if "angry" in tone or "stern" in tone:
                    valence -= 0.2
                    arousal += 0.2
                elif "happy" in tone or "playful" in tone:
                    valence += 0.2
                    arousal += 0.1
                elif "sad" in tone or "disappointed" in tone:
                    valence -= 0.2
                    arousal -= 0.1
            
            # If there's a relationship specified in context, adjust dimensions
            if "relationship" in context:
                relationship = context["relationship"].lower()
                if "mother" in relationship or "caregiver" in relationship:
                    # Strengthen emotional impact from mother figure
                    intensity *= 1.5
                    
        # Make sure dimensions are in valid range
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))
        
        # Create the trigger
        return EmotionTrigger(
            source=source,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            associated_categories=categories,
            intensity=intensity
        )
    
    def get_current_emotion(self) -> Dict[str, Any]:
        """
        Get the current emotional state in a simplified format.
        
        Returns:
            Dictionary with primary emotion and intensity
        """
        emotion = self.state.primary_emotion.value if self.state.primary_emotion else "neutral"
        intensity = 0.0
        if self.state.primary_emotion:
            intensity = self.state.categories.get(self.state.primary_emotion, 0.0)
        
        return {
            "emotion": emotion,
            "intensity": intensity,
            "dimensions": {
                "valence": self.state.dimensions[EmotionDimension.VALENCE],
                "arousal": self.state.dimensions[EmotionDimension.AROUSAL],
                "dominance": self.state.dimensions[EmotionDimension.DOMINANCE]
            }
        }