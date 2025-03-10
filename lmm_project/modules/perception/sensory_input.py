import logging
from collections import deque
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import re
import numpy as np
from pydantic import ValidationError

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from lmm_project.utils.vector_store import get_embeddings
from lmm_project.core.message import Message, MessageType, Recipient
from lmm_project.core.event_bus import EventBus, Event

from .models import (
    SensoryModality, 
    SalienceLevel,
    SensoryInput, 
    ProcessedInput, 
    SensoryFeature,
    PerceptionEvent,
    PerceptionConfig
)

# Initialize logger
logger = get_module_logger("modules.perception.sensory_input")

class SensoryInputProcessor:
    """
    Processes incoming sensory input data, manages the sensory buffer,
    and extracts features from raw sensory inputs.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config: Optional[PerceptionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the sensory input processor.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the processor
            developmental_age: Current developmental age of the mind
        """
        self._config = config or PerceptionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Input buffer for each modality
        self._input_buffers: Dict[SensoryModality, deque] = {
            modality: deque(maxlen=self._config.buffer_capacity)
            for modality in self._config.default_modalities
        }
        
        # Active inputs currently being processed
        self._active_inputs: Dict[UUID, SensoryInput] = {}
        
        # Feature extraction strategies for each modality
        self._feature_extractors = {
            SensoryModality.TEXT: self._extract_text_features,
            SensoryModality.AUDIO: self._extract_audio_features,
            SensoryModality.VISUAL: self._extract_visual_features,
            SensoryModality.EMOTIONAL: self._extract_emotional_features,
            SensoryModality.ABSTRACT: self._extract_abstract_features,
            SensoryModality.INTERNAL: self._extract_internal_features
        }
        
        # Register for relevant event types
        self._register_event_handlers()
        
        logger.info(f"Sensory input processor initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("input_received", self._handle_input_event)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        self._event_bus.subscribe("sensory_config_updated", self._handle_config_update)
    
    def _handle_input_event(self, event: Event) -> None:
        """
        Handle an incoming input event.
        
        Args:
            event: The event containing input data
        """
        try:
            # Extract input data from event
            input_data = event.data.get("input")
            if not input_data:
                logger.warning("Received input event with no input data")
                return
            
            # Create sensory input model
            sensory_input = self._create_sensory_input(input_data)
            
            # Process the input
            self.process_input(sensory_input)
            
        except ValidationError as e:
            logger.error(f"Invalid input data: {e}")
        except Exception as e:
            logger.error(f"Error processing input event: {e}")
    
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
    
    def _handle_config_update(self, event: Event) -> None:
        """
        Handle a configuration update event.
        
        Args:
            event: The event containing the new configuration
        """
        try:
            config_data = event.data.get("config")
            if config_data:
                self._config = PerceptionConfig(**config_data)
                logger.info("Updated sensory input configuration")
                
                # Update buffer sizes if needed
                for modality in self._input_buffers:
                    if self._input_buffers[modality].maxlen != self._config.buffer_capacity:
                        # Create new buffer with updated capacity
                        new_buffer = deque(self._input_buffers[modality], 
                                          maxlen=self._config.buffer_capacity)
                        self._input_buffers[modality] = new_buffer
        except ValidationError as e:
            logger.error(f"Invalid configuration data: {e}")
    
    def _create_sensory_input(self, data: Dict[str, Any]) -> SensoryInput:
        """
        Create a SensoryInput model from raw data.
        
        Args:
            data: Raw input data
            
        Returns:
            A validated SensoryInput object
        """
        # Determine modality from data or default to TEXT
        modality = data.get("modality", SensoryModality.TEXT)
        if isinstance(modality, str):
            try:
                modality = SensoryModality(modality)
            except ValueError:
                logger.warning(f"Unknown modality '{modality}', defaulting to TEXT")
                modality = SensoryModality.TEXT
        
        # Create the sensory input
        return SensoryInput(
            modality=modality,
            content=data.get("content", ""),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
            salience=data.get("salience", SalienceLevel.MEDIUM)
        )
    
    def process_input(self, sensory_input: SensoryInput) -> Optional[ProcessedInput]:
        """
        Process a sensory input, extract features, and add to buffer.
        
        Args:
            sensory_input: The raw sensory input to process
            
        Returns:
            Processed input with extracted features, or None if below threshold
        """
        # Check if input meets salience threshold
        if sensory_input.salience < self._config.base_salience_threshold:
            logger.debug(f"Input {sensory_input.id} below salience threshold, ignoring")
            return None
        
        # Extract features based on modality
        features = self._extract_features(sensory_input)
        
        # Get context vector for the input
        context_vector = self._generate_context_vector(sensory_input)
        
        # Create processed input
        processed_input = ProcessedInput(
            id=uuid4(),
            raw_input_id=sensory_input.id,
            modality=sensory_input.modality,
            features=features,
            context_vector=context_vector,
            salience=sensory_input.salience
        )
        
        # Add to buffer for this modality
        self._add_to_buffer(sensory_input)
        
        # Add to active inputs
        self._active_inputs[sensory_input.id] = sensory_input
        
        # Publish processed input event
        self._publish_processed_input_event(processed_input)
        
        return processed_input
    
    def _extract_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """
        Extract features from a sensory input based on its modality.
        
        Args:
            sensory_input: The input to extract features from
            
        Returns:
            List of extracted features
        """
        # Get the appropriate extractor for this modality
        extractor = self._feature_extractors.get(sensory_input.modality)
        if not extractor:
            logger.warning(f"No feature extractor for modality {sensory_input.modality}")
            return []
        
        # Extract features
        return extractor(sensory_input)
    
    def _generate_context_vector(self, sensory_input: SensoryInput) -> List[float]:
        """
        Generate a context vector for the input.
        
        Args:
            sensory_input: The input to generate a context vector for
            
        Returns:
            Context vector as a list of floats
        """
        # For text modality, use embeddings
        if sensory_input.modality == SensoryModality.TEXT and isinstance(sensory_input.content, str):
            try:
                # Generate embedding for the text
                embedding = get_embeddings(sensory_input.content)
                if isinstance(embedding, list) and embedding:
                    return embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        # Fallback to empty vector
        return []
    
    def _add_to_buffer(self, sensory_input: SensoryInput) -> None:
        """
        Add input to the appropriate modality buffer.
        
        Args:
            sensory_input: The input to add to the buffer
        """
        # Ensure buffer exists for this modality
        if sensory_input.modality not in self._input_buffers:
            self._input_buffers[sensory_input.modality] = deque(
                maxlen=self._config.buffer_capacity
            )
        
        # Add to buffer
        self._input_buffers[sensory_input.modality].append(sensory_input)
    
    def _publish_processed_input_event(self, processed_input: ProcessedInput) -> None:
        """
        Publish an event with the processed input.
        
        Args:
            processed_input: The processed input to publish
        """
        # Create event payload
        payload = {
            "processed_input": processed_input.model_dump(),
            "developmental_age": self._developmental_age
        }
        
        # Create and publish event
        event = Event(
            type="sensory_input_processed",
            source="perception.sensory_input",
            data=payload
        )
        self._event_bus.publish(event)
    
    def get_recent_inputs(self, modality: SensoryModality, count: int = 5) -> List[SensoryInput]:
        """
        Get recent inputs for a specific modality.
        
        Args:
            modality: The sensory modality to get inputs for
            count: Maximum number of inputs to return
            
        Returns:
            List of recent inputs for the specified modality
        """
        if modality not in self._input_buffers:
            return []
        
        # Convert deque to list and reverse to get most recent first
        buffer = list(self._input_buffers[modality])
        buffer.reverse()
        
        # Return up to count inputs
        return buffer[:min(count, len(buffer))]
    
    def get_active_inputs(self) -> Dict[UUID, SensoryInput]:
        """
        Get all active inputs.
        
        Returns:
            Dictionary of active inputs by ID
        """
        return self._active_inputs.copy()
    
    def clear_buffer(self, modality: Optional[SensoryModality] = None) -> None:
        """
        Clear the input buffer for a specific modality or all modalities.
        
        Args:
            modality: The modality to clear, or None to clear all
        """
        if modality:
            if modality in self._input_buffers:
                self._input_buffers[modality].clear()
        else:
            for buffer in self._input_buffers.values():
                buffer.clear()
    
    # Feature extraction methods for different modalities
    
    def _extract_text_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from text input"""
        features = []
        
        if not isinstance(sensory_input.content, str):
            return features
            
        text = sensory_input.content
        
        # Extract basic text features
        features.append(SensoryFeature(
            name="length",
            value=len(text),
            modality=SensoryModality.TEXT
        ))
        
        # Word count
        words = re.findall(r'\w+', text.lower())
        features.append(SensoryFeature(
            name="word_count",
            value=len(words),
            modality=SensoryModality.TEXT
        ))
        
        # Extract more sophisticated features based on developmental age
        if self._developmental_age >= 0.5:
            # Simple sentiment analysis
            positive_words = set(['good', 'happy', 'nice', 'love', 'like', 'joy'])
            negative_words = set(['bad', 'sad', 'angry', 'hate', 'dislike', 'fear'])
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            sentiment = (pos_count - neg_count) / max(1, len(words))
            features.append(SensoryFeature(
                name="sentiment",
                value=sentiment,
                modality=SensoryModality.TEXT
            ))
        
        if self._developmental_age >= 1.0:
            # Question detection
            is_question = 1.0 if '?' in text else 0.0
            features.append(SensoryFeature(
                name="is_question",
                value=is_question,
                modality=SensoryModality.TEXT
            ))
            
            # Complexity estimation (simple proxy via avg word length)
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            features.append(SensoryFeature(
                name="complexity",
                value=min(1.0, avg_word_length / 10.0),
                modality=SensoryModality.TEXT
            ))
        
        return features
    
    def _extract_audio_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from audio input"""
        # Basic implementation - would be expanded with actual audio processing
        return []
    
    def _extract_visual_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from visual input"""
        # Placeholder for future visual processing
        return []
    
    def _extract_emotional_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from emotional input"""
        features = []
        
        if not isinstance(sensory_input.content, dict):
            return features
            
        # Process emotional content data
        emotional_data = sensory_input.content
        
        for emotion, intensity in emotional_data.items():
            if isinstance(intensity, (int, float)):
                features.append(SensoryFeature(
                    name=f"emotion_{emotion}",
                    value=float(intensity),
                    modality=SensoryModality.EMOTIONAL
                ))
        
        return features
    
    def _extract_abstract_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from abstract concept input"""
        # Placeholder for abstract concept processing
        return []
    
    def _extract_internal_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """Extract features from internal state input"""
        # Internal state processing
        return []
