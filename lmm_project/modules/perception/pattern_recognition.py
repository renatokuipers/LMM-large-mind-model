import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import re
from pydantic import ValidationError

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from lmm_project.utils.vector_store import VectorStore, get_vector_store
from lmm_project.core.event_bus import EventBus, Event

from .models import (
    SensoryModality, 
    ProcessedInput, 
    PatternType,
    RecognizedPattern,
    SensoryFeature,
    PerceptionEvent,
    PerceptionConfig
)

# Initialize logger
logger = get_module_logger("modules.perception.pattern_recognition")

class PatternRecognizer:
    """
    Recognizes patterns in processed sensory inputs. Detects recurring
    features, temporal sequences, and semantic relationships.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config: Optional[PerceptionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the pattern recognizer.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the recognizer
            developmental_age: Current developmental age of the mind
        """
        self._config = config or PerceptionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Vector store for semantic pattern storage
        self._vector_store = get_vector_store(
            dimension=self._config.vector_dimension,
            index_type="Flat",
            use_gpu=None  # Let the vector store decide based on config
        )
        
        # Known patterns by type
        self._patterns: Dict[PatternType, Set[str]] = defaultdict(set)
        
        # Feature history for tracking occurrences of features
        self._feature_history: Dict[str, Counter] = defaultdict(Counter)
        
        # Feature sequence history for temporal patterns
        self._feature_sequences: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)
        
        # Recently recognized patterns
        self._recent_patterns: List[RecognizedPattern] = []
        
        # Pattern recognition strategies
        self._pattern_recognizers = {
            PatternType.SEMANTIC: self._recognize_semantic_patterns,
            PatternType.ASSOCIATIVE: self._recognize_associative_patterns,
            PatternType.TEMPORAL: self._recognize_temporal_patterns,
            PatternType.CATEGORICAL: self._recognize_categorical_patterns,
            PatternType.EMOTIONAL: self._recognize_emotional_patterns
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Pattern recognizer initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register for relevant events"""
        self._event_bus.subscribe("sensory_input_processed", self._handle_processed_input)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
    
    def _handle_processed_input(self, event: Event) -> None:
        """
        Handle a processed input event.
        
        Args:
            event: The event containing the processed input
        """
        try:
            # Extract processed input from event
            processed_input_data = event.data.get("processed_input")
            if not processed_input_data:
                logger.warning("Received event with no processed input data")
                return
            
            # Create processed input model
            processed_input = ProcessedInput(**processed_input_data)
            
            # Recognize patterns in the input
            patterns = self.recognize_patterns(processed_input)
            
            # For each recognized pattern, publish an event
            for pattern in patterns:
                self._publish_pattern_event(pattern)
                
        except ValidationError as e:
            logger.error(f"Invalid processed input data: {e}")
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
    
    def _handle_age_update(self, event: Event) -> None:
        """
        Handle a developmental age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.data.get("age")
        if new_age is not None and isinstance(new_age, (int, float)):
            self._developmental_age = float(new_age)
            logger.debug(f"Updated developmental age to {self._developmental_age}")
    
    def recognize_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize patterns in a processed input.
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized patterns
        """
        # Skip if we're not yet at threshold for pattern recognition
        if self._developmental_age < 0.1:
            logger.debug("Developmental age too low for pattern recognition")
            return []
            
        # Update feature history based on this input
        self._update_feature_history(processed_input)
        
        # Initialize recognized patterns list
        recognized_patterns = []
        
        # Apply each pattern recognition strategy that's appropriate for the age
        for pattern_type, recognizer in self._pattern_recognizers.items():
            # Skip if the pattern type is too advanced for current age
            if not self._is_pattern_type_enabled(pattern_type):
                continue
                
            # Recognize patterns of this type
            patterns = recognizer(processed_input)
            recognized_patterns.extend(patterns)
            
        # Save recent patterns
        self._recent_patterns.extend(recognized_patterns)
        if len(self._recent_patterns) > 50:  # Limit to last 50 patterns
            self._recent_patterns = self._recent_patterns[-50:]
            
        return recognized_patterns
    
    def _is_pattern_type_enabled(self, pattern_type: PatternType) -> bool:
        """
        Check if a pattern type is enabled at the current developmental age.
        
        Args:
            pattern_type: The pattern type to check
            
        Returns:
            True if the pattern type is enabled, False otherwise
        """
        # Age thresholds for different pattern types
        age_thresholds = {
            PatternType.SEMANTIC: 0.1,     # Basic semantic from very early on
            PatternType.ASSOCIATIVE: 0.2,  # Simple associations
            PatternType.TEMPORAL: 0.3,     # Temporal sequences come a bit later
            PatternType.CATEGORICAL: 0.5,  # Categories emerge later
            PatternType.EMOTIONAL: 0.7,    # Emotional patterns need more development
            PatternType.CAUSAL: 1.0,       # Causal patterns are advanced
            PatternType.SPATIAL: 1.5       # Spatial patterns are complex
        }
        
        # Get threshold for this pattern type
        threshold = age_thresholds.get(pattern_type, 0.0)
        
        # Check if we're at or above the threshold
        return self._developmental_age >= threshold
    
    def _update_feature_history(self, processed_input: ProcessedInput) -> None:
        """
        Update feature history with features from the processed input.
        
        Args:
            processed_input: The processed input to update history with
        """
        # Extract feature names and modalities
        for feature in processed_input.features:
            feature_key = f"{feature.modality.value}:{feature.name}"
            self._feature_history[feature_key][feature.value] += 1
            
            # Add to sequence history for temporal patterns
            self._feature_sequences[feature_key].append(
                (str(feature.value), processed_input.timestamp)
            )
            
            # Trim sequence history if it's getting too long
            if len(self._feature_sequences[feature_key]) > 50:
                self._feature_sequences[feature_key] = self._feature_sequences[feature_key][-50:]
    
    def _recognize_semantic_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize semantic patterns in a processed input.
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized semantic patterns
        """
        patterns = []
        
        # Skip if we don't have context vector
        if not processed_input.context_vector:
            return patterns
            
        # Search for semantically similar inputs in vector store
        try:
            # Convert to numpy array for vector search
            query_vector = np.array(processed_input.context_vector, dtype=np.float32)
            
            # Search for similar vectors
            results = self._vector_store.search(
                query_vector, 
                k=3,  # Get top 3 similar vectors
                metadata_filter=None
            )
            
            if results and results[0]:  # If we have results
                ids, distances, metadata_list = results
                
                # Create pattern for each match above threshold
                threshold = self._config.pattern_recognition_threshold
                
                for i, (_, distance) in enumerate(zip(ids, distances)):
                    # Convert distance to similarity (1 = identical, 0 = completely different)
                    similarity = 1.0 - min(1.0, distance)
                    
                    if similarity >= threshold:
                        # Create pattern
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.SEMANTIC,
                            pattern_key=f"semantic:{processed_input.modality.value}",
                            confidence=similarity,
                            salience=processed_input.salience,
                            input_ids=[processed_input.id],
                            vector_representation=processed_input.context_vector
                        )
                        
                        # Add metadata from similar item if available
                        if metadata_list and i < len(metadata_list) and metadata_list[i]:
                            pattern.context_data.update(metadata_list[i])
                            
                        patterns.append(pattern)
                        
                        # Add to known patterns
                        self._patterns[PatternType.SEMANTIC].add(pattern.pattern_key)
            
            # Store this vector for future pattern recognition
            metadata = {
                "input_id": str(processed_input.id),
                "modality": processed_input.modality.value,
                "timestamp": processed_input.timestamp.isoformat()
            }
            self._vector_store.add_vectors([query_vector], [metadata])
            
        except Exception as e:
            logger.error(f"Error recognizing semantic patterns: {e}")
            
        return patterns
    
    def _recognize_associative_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize associative patterns (co-occurrence) in features.
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized associative patterns
        """
        patterns = []
        
        # Need at least 2 features to detect associations
        if len(processed_input.features) < 2:
            return patterns
            
        # Look for co-occurring features
        for i, feature1 in enumerate(processed_input.features):
            for feature2 in processed_input.features[i+1:]:
                # Skip if same feature
                if feature1.name == feature2.name and feature1.modality == feature2.modality:
                    continue
                    
                # Create a key for this feature pair
                key1 = f"{feature1.modality.value}:{feature1.name}"
                key2 = f"{feature2.modality.value}:{feature2.name}"
                pattern_key = f"association:{key1}+{key2}"
                
                # Check if we've seen this association before
                is_known = pattern_key in self._patterns[PatternType.ASSOCIATIVE]
                
                # Create pattern with medium confidence (will increase with repeated exposure)
                confidence = 0.7 if is_known else 0.5
                
                pattern = RecognizedPattern(
                    pattern_type=PatternType.ASSOCIATIVE,
                    pattern_key=pattern_key,
                    confidence=confidence,
                    salience=max(feature1.value, feature2.value),
                    input_ids=[processed_input.id],
                    features=[feature1, feature2]
                )
                
                patterns.append(pattern)
                
                # Add to known patterns
                self._patterns[PatternType.ASSOCIATIVE].add(pattern_key)
                
        return patterns
    
    def _recognize_temporal_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize temporal patterns (sequences over time).
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized temporal patterns
        """
        patterns = []
        
        # Need enough developmental age for temporal patterns
        if self._developmental_age < 0.3:
            return patterns
            
        # Need features to detect temporal patterns
        if not processed_input.features:
            return patterns
            
        # For each feature, check for recurring sequences
        for feature in processed_input.features:
            feature_key = f"{feature.modality.value}:{feature.name}"
            
            # Skip if we don't have enough history
            if feature_key not in self._feature_sequences or len(self._feature_sequences[feature_key]) < 3:
                continue
                
            # Get recent sequence for this feature
            sequence = self._feature_sequences[feature_key]
            values = [item[0] for item in sequence[-3:]]  # Last 3 values
            
            # Check if we have a repeated subsequence
            if len(values) >= 3:
                pattern_key = f"temporal:{feature_key}:{'+'.join(values)}"
                
                # Check if we've seen this sequence before
                is_known = pattern_key in self._patterns[PatternType.TEMPORAL]
                confidence = 0.7 if is_known else 0.5
                
                pattern = RecognizedPattern(
                    pattern_type=PatternType.TEMPORAL,
                    pattern_key=pattern_key,
                    confidence=confidence,
                    salience=processed_input.salience,
                    input_ids=[processed_input.id],
                    features=[feature]
                )
                
                patterns.append(pattern)
                
                # Add to known patterns
                self._patterns[PatternType.TEMPORAL].add(pattern_key)
                
        return patterns
    
    def _recognize_categorical_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize categorical patterns (grouping of similar items).
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized categorical patterns
        """
        patterns = []
        
        # Need enough developmental age for category recognition
        if self._developmental_age < 0.5:
            return patterns
            
        # Need features to detect categories
        if not processed_input.features:
            return patterns
            
        # For each feature, check if it falls into a known category
        for feature in processed_input.features:
            feature_key = f"{feature.modality.value}:{feature.name}"
            
            # Skip if we don't have enough history for this feature
            if feature_key not in self._feature_history or len(self._feature_history[feature_key]) < 2:
                continue
                
            # Get the range of values we've seen for this feature
            values = list(self._feature_history[feature_key].keys())
            if all(isinstance(v, (int, float)) for v in values):
                # For numerical features, check where this value falls in the range
                try:
                    num_values = [float(v) for v in values]
                    min_val = min(num_values)
                    max_val = max(num_values)
                    
                    # Skip if no range
                    if min_val == max_val:
                        continue
                        
                    # Determine category (low, medium, high)
                    range_size = max_val - min_val
                    relative_pos = (float(feature.value) - min_val) / range_size
                    
                    if relative_pos < 0.33:
                        category = "low"
                    elif relative_pos < 0.66:
                        category = "medium"
                    else:
                        category = "high"
                        
                    pattern_key = f"category:{feature_key}:{category}"
                    
                    # Create pattern
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.CATEGORICAL,
                        pattern_key=pattern_key,
                        confidence=0.6,
                        salience=processed_input.salience,
                        input_ids=[processed_input.id],
                        features=[feature],
                        context_data={"category": category}
                    )
                    
                    patterns.append(pattern)
                    
                    # Add to known patterns
                    self._patterns[PatternType.CATEGORICAL].add(pattern_key)
                    
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                    
        return patterns
    
    def _recognize_emotional_patterns(self, processed_input: ProcessedInput) -> List[RecognizedPattern]:
        """
        Recognize emotional patterns in processed input.
        
        Args:
            processed_input: The processed input to recognize patterns in
            
        Returns:
            List of recognized emotional patterns
        """
        patterns = []
        
        # Need enough developmental age for emotional patterns
        if self._developmental_age < 0.7 or not self._config.enable_emotional_processing:
            return patterns
            
        # Check for emotional features
        emotional_features = [
            f for f in processed_input.features 
            if f.modality == SensoryModality.EMOTIONAL
            or f.name == "sentiment"
        ]
        
        if not emotional_features:
            return patterns
            
        # Recognize patterns in emotional features
        for feature in emotional_features:
            intensity = abs(feature.value)
            
            if intensity > 0.3:  # Only recognize significant emotions
                # Determine the emotional valence (positive/negative)
                valence = "positive" if feature.value > 0 else "negative"
                
                pattern_key = f"emotional:{feature.name}:{valence}"
                
                # Create pattern
                pattern = RecognizedPattern(
                    pattern_type=PatternType.EMOTIONAL,
                    pattern_key=pattern_key,
                    confidence=intensity,
                    salience=intensity * processed_input.salience,
                    input_ids=[processed_input.id],
                    features=[feature],
                    context_data={"valence": valence, "intensity": intensity}
                )
                
                patterns.append(pattern)
                
                # Add to known patterns
                self._patterns[PatternType.EMOTIONAL].add(pattern_key)
                
        return patterns
    
    def _publish_pattern_event(self, pattern: RecognizedPattern) -> None:
        """
        Publish an event for a recognized pattern.
        
        Args:
            pattern: The recognized pattern
        """
        # Create event payload
        payload = {
            "pattern": pattern.model_dump(),
            "developmental_age": self._developmental_age,
            "is_novel": pattern.pattern_key not in self._patterns[pattern.pattern_type]
        }
        
        # Create and publish event
        event = Event(
            type="pattern_recognized",
            source="perception.pattern_recognition",
            data=payload
        )
        self._event_bus.publish(event)
    
    def get_recent_patterns(self, count: int = 10) -> List[RecognizedPattern]:
        """
        Get recently recognized patterns.
        
        Args:
            count: Maximum number of patterns to return
            
        Returns:
            List of recently recognized patterns
        """
        return self._recent_patterns[-count:]
    
    def get_known_patterns(self, pattern_type: Optional[PatternType] = None) -> Set[str]:
        """
        Get known pattern keys of a specific type or all types.
        
        Args:
            pattern_type: The pattern type to get, or None for all types
            
        Returns:
            Set of known pattern keys
        """
        if pattern_type:
            return self._patterns[pattern_type].copy()
        else:
            # Combine all pattern types
            all_patterns = set()
            for patterns in self._patterns.values():
                all_patterns.update(patterns)
            return all_patterns
