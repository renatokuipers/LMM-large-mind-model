import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from collections import defaultdict, deque

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType

from .models import (
    SalienceFeature,
    SalienceAssessment,
    AttentionEvent,
    AttentionConfig
)

# Initialize logger
logger = get_module_logger("modules.attention.salience_detector")

class SalienceDetector:
    """
    Evaluates the salience (noteworthiness) of inputs and patterns.
    Determines what deserves attention based on various factors like
    novelty, intensity, emotional relevance, and importance to goals.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[AttentionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the salience detector.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the detector
            developmental_age: Current developmental age of the mind
        """
        self._config = config or AttentionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Recent salience assessments for reference
        self._recent_assessments = deque(maxlen=50)
        
        # Novelty tracking for different input types
        self._novelty_trackers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Feature importance weights (adjusted by learning)
        self._feature_weights: Dict[str, float] = {
            # Basic features (even very young minds can detect)
            "intensity": 1.0,
            "contrast": 0.9,
            "suddenness": 0.8,
            "novelty": self._config.novelty_weight,
            
            # Emotional features (develop at intermediate ages)
            "emotional_valence": self._config.emotional_salience_weight * 0.8,
            "emotional_arousal": self._config.emotional_salience_weight,
            
            # Complex features (develop at later ages)
            "goal_relevance": 0.3,
            "social_significance": 0.3,
            "pattern_strength": 0.4,
            "contextual_anomaly": 0.5,
            "predictive_error": 0.4
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        # Apply developmental adjustments
        self._apply_developmental_adjustments()
        
        logger.info(f"Salience detector initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("sensory_input_processed", self._handle_sensory_input)
        self._event_bus.subscribe("pattern_recognized", self._handle_pattern_recognition)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        self._event_bus.subscribe("emotional_response", self._handle_emotional_response)
    
    def _handle_sensory_input(self, event: Dict[str, Any]) -> None:
        """
        Handle processed sensory input events.
        
        Args:
            event: The event containing processed input data
        """
        try:
            # Extract input data
            input_data = event.get("processed_input")
            if not input_data:
                logger.warning("Received sensory input event with no input data")
                return
            
            # Extract relevant information
            input_id = input_data.get("id")
            modality = input_data.get("modality")
            features = input_data.get("features", [])
            context_vector = input_data.get("context_vector", [])
            
            if not (input_id and modality):
                logger.warning("Sensory input missing required fields")
                return
            
            # Convert to feature dictionary for salience assessment
            feature_dict = {}
            if features:
                for feature in features:
                    feature_name = feature.get("name")
                    feature_value = feature.get("value")
                    if feature_name and feature_value is not None:
                        feature_dict[feature_name] = feature_value
            
            # Add modality as a feature
            feature_dict["modality"] = modality
            
            # Get contextual data based on modality
            context_data = {"modality": modality}
            if context_vector:
                context_data["context_vector"] = context_vector
            
            # Assess salience
            assessment = self.assess_salience(
                input_id=UUID(input_id) if isinstance(input_id, str) else input_id,
                input_type="sensory_input",
                features=feature_dict,
                context_data=context_data
            )
            
            # Track feature frequency for novelty calculation
            if len(feature_dict) > 0:
                feature_hash = self._create_feature_hash(feature_dict)
                self._novelty_trackers["sensory"].append(feature_hash)
                
        except Exception as e:
            logger.error(f"Error handling sensory input: {e}")
    
    def _handle_pattern_recognition(self, event: Dict[str, Any]) -> None:
        """
        Handle pattern recognition events.
        
        Args:
            event: The event containing pattern data
        """
        try:
            # Extract pattern data
            pattern_data = event.get("pattern")
            if not pattern_data:
                logger.warning("Received pattern event with no pattern data")
                return
            
            # Extract relevant information
            pattern_id = pattern_data.get("id")
            pattern_type = pattern_data.get("pattern_type")
            confidence = pattern_data.get("confidence", 0.5)
            features = pattern_data.get("features", [])
            vector = pattern_data.get("vector_representation", [])
            context_data = pattern_data.get("context_data", {})
            
            if not (pattern_id and pattern_type):
                logger.warning("Pattern missing required fields")
                return
            
            # Convert to feature dictionary for salience assessment
            feature_dict = {
                "pattern_type": pattern_type,
                "confidence": confidence
            }
            
            # Add pattern features
            if features:
                for feature in features:
                    feature_name = feature.get("name")
                    feature_value = feature.get("value")
                    if feature_name and feature_value is not None:
                        feature_dict[f"pattern_{feature_name}"] = feature_value
            
            # Get contextual data
            context = {
                "pattern_type": pattern_type,
                **context_data
            }
            if vector:
                context["vector"] = vector
            
            # Assess salience
            assessment = self.assess_salience(
                input_id=UUID(pattern_id) if isinstance(pattern_id, str) else pattern_id,
                input_type="pattern",
                features=feature_dict,
                context_data=context
            )
            
            # Track feature frequency for novelty calculation
            if len(feature_dict) > 0:
                feature_hash = self._create_feature_hash(feature_dict)
                tracker_key = f"pattern_{pattern_type}"
                self._novelty_trackers[tracker_key].append(feature_hash)
                
        except Exception as e:
            logger.error(f"Error handling pattern recognition: {e}")
    
    def _handle_emotional_response(self, event: Dict[str, Any]) -> None:
        """
        Handle emotional response events.
        
        Args:
            event: The event containing emotional response data
        """
        try:
            # Extract emotional data
            emotional_data = event.get("emotional_response")
            if not emotional_data:
                return  # Silently ignore - not all emotional events need processing
            
            # We don't need to do a full assessment here - just track emotional response
            # for future salience assessments
            
            # However, if the emotional response is very strong, we could create a
            # special salience assessment for it
            valence = emotional_data.get("valence", 0.0)
            arousal = emotional_data.get("arousal", 0.0)
            
            # Only process strong emotional responses
            if abs(valence) > 0.7 or arousal > 0.7:
                source_id = emotional_data.get("source_id")
                if source_id:
                    feature_dict = {
                        "emotional_valence": valence,
                        "emotional_arousal": arousal,
                        "emotion_type": emotional_data.get("emotion_type", "unknown")
                    }
                    
                    # Assess salience of the emotional response itself
                    assessment = self.assess_salience(
                        input_id=UUID(source_id) if isinstance(source_id, str) else source_id,
                        input_type="emotional_response",
                        features=feature_dict,
                        context_data=emotional_data
                    )
        except Exception as e:
            logger.error(f"Error handling emotional response: {e}")
    
    def _handle_age_update(self, event: Dict[str, Any]) -> None:
        """
        Handle development age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.get("new_age")
        if new_age is not None:
            self.update_developmental_age(new_age)
    
    def assess_salience(
        self,
        input_id: UUID,
        input_type: str,
        features: Dict[str, float],
        context_data: Optional[Dict[str, Any]] = None
    ) -> SalienceAssessment:
        """
        Assess the salience of an input based on its features and context.
        
        Args:
            input_id: ID of the input
            input_type: Type of the input (e.g., 'sensory_input', 'pattern')
            features: Dictionary of features that may affect salience
            context_data: Optional context that may affect salience assessment
            
        Returns:
            Salience assessment for the input
        """
        # Copy features to avoid modifying the original
        features_copy = features.copy()
        
        # Calculate feature contributions to salience
        contributing_features = []
        salience_score = 0.0
        total_weight = 0.0
        
        # First, calculate novelty if we have enough history
        novelty = self._calculate_novelty(features_copy, input_type)
        if novelty is not None:
            features_copy["novelty"] = novelty
        
        # Process each feature for salience contribution
        for feature_name, feature_value in features_copy.items():
            # Skip non-numeric features
            if not isinstance(feature_value, (int, float)):
                continue
                
            # Get the weight for this feature
            weight = self._feature_weights.get(feature_name, 0.5)
            
            # Apply developmental adjustments
            if self._developmental_age < 0.3 and feature_name in [
                "goal_relevance", "social_significance", "contextual_anomaly",
                "predictive_error", "pattern_strength"
            ]:
                # Young minds focus on simpler features
                weight *= self._developmental_age * 3
            
            # Normalize value to [0,1] if outside that range
            normalized_value = feature_value
            if isinstance(feature_value, (int, float)):
                if feature_name == "emotional_valence":
                    # For valence, we care about the magnitude (positive or negative)
                    normalized_value = abs(feature_value)
                else:
                    # For other features, normalize to [0,1]
                    normalized_value = max(0.0, min(1.0, feature_value))
            
            # Calculate contribution
            contribution = normalized_value * weight
            
            # Add to salience score
            salience_score += contribution
            total_weight += weight
            
            # Add to contributing features
            if weight > 0.01:  # Only include features with non-negligible weight
                contributing_features.append(
                    SalienceFeature(
                        name=feature_name,
                        value=normalized_value,
                        weight=weight
                    )
                )
        
        # Normalize salience score
        if total_weight > 0:
            salience_score /= total_weight
        
        # Apply context influence
        context_influence = 0.0
        if context_data:
            context_influence = self._calculate_context_influence(
                features_copy, context_data, input_type
            )
            
            # Apply context influence to salience score
            salience_score = salience_score * (1.0 + context_influence * self._config.context_influence_strength)
            
            # Ensure salience score is within valid range
            salience_score = max(0.0, min(1.0, salience_score))
        
        # Create assessment
        assessment = SalienceAssessment(
            input_id=input_id,
            input_type=input_type,
            salience_score=salience_score,
            contributing_features=contributing_features,
            context_influence=context_influence
        )
        
        # Add to recent assessments
        self._recent_assessments.append(assessment)
        
        # Publish assessment event
        self._publish_assessment_event(assessment)
        
        logger.debug(f"Assessed salience for {input_type} {input_id}: {salience_score:.3f}")
        return assessment
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the salience detector.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        
        # Apply developmental adjustments
        self._apply_developmental_adjustments()
        
        logger.info(f"Salience detector age updated to {new_age}")
    
    def _apply_developmental_adjustments(self) -> None:
        """Apply developmental adjustments to feature weights."""
        # Adjust feature weights based on developmental age
        
        # Complex features develop later
        complex_feature_factor = min(1.0, self._developmental_age * 2.5)
        self._feature_weights.update({
            "goal_relevance": 0.3 * complex_feature_factor,
            "social_significance": 0.3 * complex_feature_factor,
            "contextual_anomaly": 0.5 * complex_feature_factor,
            "predictive_error": 0.4 * complex_feature_factor
        })
        
        # Emotional features develop at intermediate ages
        emotional_feature_factor = min(1.0, self._developmental_age * 1.5 + 0.2)
        self._feature_weights.update({
            "emotional_valence": self._config.emotional_salience_weight * 0.8 * emotional_feature_factor,
            "emotional_arousal": self._config.emotional_salience_weight * emotional_feature_factor
        })
        
        # Novelty is important at all ages but gradually decreases in relative importance
        novelty_factor = max(0.5, 1.0 - self._developmental_age * 0.3)
        self._feature_weights["novelty"] = self._config.novelty_weight * novelty_factor
        
        logger.debug(f"Applied developmental adjustments for age {self._developmental_age}")
    
    def _calculate_novelty(
        self, 
        features: Dict[str, Any],
        input_type: str
    ) -> Optional[float]:
        """
        Calculate the novelty of an input based on its features.
        
        Args:
            features: Features of the input
            input_type: Type of the input
            
        Returns:
            Novelty score between 0 and 1, or None if not enough history
        """
        # Get the tracker for this input type
        tracker_key = input_type.split('_')[0]  # 'sensory_input' -> 'sensory'
        if input_type.startswith("pattern"):
            pattern_type = features.get("pattern_type", "unknown")
            tracker_key = f"pattern_{pattern_type}"
        
        tracker = self._novelty_trackers.get(tracker_key)
        
        # If we don't have enough history, return None
        if not tracker or len(tracker) < 5:
            return None
        
        # Create a feature hash for comparison
        feature_hash = self._create_feature_hash(features)
        
        # Check how often we've seen similar features
        similar_count = 0
        total_checks = min(len(tracker), 20)  # Limit history search
        
        for past_hash in list(tracker)[-total_checks:]:
            similarity = self._calculate_hash_similarity(feature_hash, past_hash)
            if similarity > 0.8:  # Similar feature set
                similar_count += 1
        
        # Calculate novelty score
        if total_checks > 0:
            familiarity = similar_count / total_checks
            novelty = 1.0 - familiarity
            return novelty
        
        return None
    
    def _calculate_context_influence(
        self,
        features: Dict[str, Any],
        context_data: Dict[str, Any],
        input_type: str
    ) -> float:
        """
        Calculate how context influences salience.
        
        Args:
            features: Features of the input
            context_data: Contextual information
            input_type: Type of the input
            
        Returns:
            Context influence score between -1 and 1
        """
        # By default, context has minimal influence
        influence = 0.0
        
        # Expected patterns have reduced salience
        if "expected" in context_data and context_data["expected"] is True:
            influence -= 0.3
        
        # Relevant to current goals have increased salience
        if "goal_relevant" in context_data and context_data["goal_relevant"] is True:
            influence += 0.4
        
        # Relevant to current emotional state have increased salience
        if "emotional_relevance" in context_data and isinstance(context_data["emotional_relevance"], (int, float)):
            influence += context_data["emotional_relevance"] * 0.4
        
        # Patterns building on recently recognized patterns have increased salience
        if input_type == "pattern" and "builds_on_recent" in context_data and context_data["builds_on_recent"] is True:
            influence += 0.3
        
        # Sensory inputs that contradict predictions have increased salience
        if input_type == "sensory_input" and "prediction_error" in context_data:
            pred_error = context_data["prediction_error"]
            if isinstance(pred_error, (int, float)):
                influence += min(0.5, pred_error * 0.6)
        
        # Limit influence to range [-1, 1]
        return max(-1.0, min(1.0, influence))
    
    def _create_feature_hash(self, features: Dict[str, Any]) -> str:
        """
        Create a simple hash for a feature set to compare similarity.
        
        Args:
            features: Features to hash
            
        Returns:
            A string hash representing the features
        """
        # Extract numeric features and their approximate values
        hash_components = []
        
        for name, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                # Discretize to reduce sensitivity to small changes
                discrete_value = round(value * 10) / 10
                hash_components.append(f"{name}:{discrete_value}")
            elif isinstance(value, str):
                hash_components.append(f"{name}:{value}")
        
        return "|".join(hash_components)
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate the similarity between two feature hashes.
        
        Args:
            hash1: First feature hash
            hash2: Second feature hash
            
        Returns:
            Similarity score between 0 and 1
        """
        # Split hashes into components
        components1 = set(hash1.split("|"))
        components2 = set(hash2.split("|"))
        
        # If either is empty, return 0 similarity
        if not components1 or not components2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(components1.intersection(components2))
        union = len(components1.union(components2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _publish_assessment_event(self, assessment: SalienceAssessment) -> None:
        """
        Publish an event for a salience assessment.
        
        Args:
            assessment: The salience assessment
        """
        event = AttentionEvent(
            event_type="salience_assessment",
            payload={
                "assessment": assessment.dict(),
                "developmental_age": self._developmental_age
            }
        )
        
        self._event_bus.publish("salience_assessment_created", event.dict())
