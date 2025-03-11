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
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message, MessageType

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
    
    def _handle_input_event(self, event: Dict[str, Any]) -> None:
        """
        Handle an incoming input event.
        
        Args:
            event: The event containing input data
        """
        try:
            # Extract input data from event
            input_data = event.get("input")
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
    
    def _handle_age_update(self, event: Dict[str, Any]) -> None:
        """
        Handle a developmental age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.get("age")
        if new_age is not None and isinstance(new_age, (int, float)):
            self._developmental_age = float(new_age)
            logger.debug(f"Updated developmental age to {self._developmental_age}")
    
    def _handle_config_update(self, event: Dict[str, Any]) -> None:
        """
        Handle a configuration update event.
        
        Args:
            event: The event containing the new configuration
        """
        try:
            config_data = event.get("config")
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
        features = []
        
        # For audio modality, we expect content to be a path to an audio file
        if not isinstance(sensory_input.content, str):
            logger.warning("Audio input content should be a path to an audio file")
            return features
            
        audio_path = sensory_input.content
        
        try:
            # Import audio extraction module
            from lmm_project.utils.audio_extraction import extract_features_from_file, get_available_feature_types
            
            # Get available feature types based on developmental age
            available_feature_types = get_available_feature_types(self._developmental_age)
            
            # Extract features from the audio file
            audio_features = extract_features_from_file(
                file_path=audio_path,
                developmental_age=self._developmental_age,
                feature_types=available_feature_types
            )
            
            # Convert extracted features to SensoryFeature objects
            for name, value in audio_features.items():
                # Skip non-numeric features or metadata
                if not isinstance(value, (int, float)) or name in ["file_path", "file_size", "error"]:
                    continue
                    
                features.append(SensoryFeature(
                    name=name,
                    value=float(value),
                    modality=SensoryModality.AUDIO
                ))
                
            # Log success
            logger.debug(f"Extracted {len(features)} audio features from {audio_path}")
            
        except ImportError:
            logger.warning("Audio extraction utilities not available")
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            
        return features
    
    def _extract_visual_features(self, sensory_input: SensoryInput) -> List[SensoryFeature]:
        """
        Extract abstract visual features from textual descriptions using NLP techniques.
        
        This creates dynamic abstract representations of visual concepts without requiring
        actual image inputs, suitable for developmental learning of visual concepts.
        """
        features = []
        
        # For visual modality, we expect content to be a textual description
        if not isinstance(sensory_input.content, str):
            logger.warning("Visual input content should be a textual description")
            return features
            
        description = sensory_input.content

        try:
            # Basic developmental level - simple text analysis
            if self._developmental_age >= 0.0:
                # Basic text statistics
                word_count = len(description.split())
                features.append(SensoryFeature(
                    name="visual_description_length",
                    value=min(1.0, word_count / 50.0),  # Normalize
                    modality=SensoryModality.VISUAL
                ))
                
                # Import basic NLP tools
                import re
                from collections import Counter
                
                # Brightness perception (using advanced pattern matching)
                brightness_pattern = r'\b(bright|dark|dim|light|glow|shining|shadowy|shadow)\b'
                brightness_matches = re.findall(brightness_pattern, description.lower())
                
                if brightness_matches:
                    # Map terms to brightness values
                    brightness_values = {
                        "bright": 0.9, "light": 0.8, "glow": 0.7, "shining": 1.0,
                        "dim": 0.3, "shadowy": 0.2, "shadow": 0.2, "dark": 0.1
                    }
                    
                    # Calculate weighted average brightness
                    total_weight = 0
                    brightness_sum = 0
                    
                    for term in brightness_matches:
                        brightness_sum += brightness_values.get(term, 0.5)
                        total_weight += 1
                    
                    avg_brightness = brightness_sum / max(1, total_weight)
                    features.append(SensoryFeature(
                        name="brightness",
                        value=avg_brightness,
                        modality=SensoryModality.VISUAL
                    ))
                else:
                    # Default neutral brightness
                    features.append(SensoryFeature(
                        name="brightness",
                        value=0.5,
                        modality=SensoryModality.VISUAL
                    ))
            
            # More advanced NLP at later stages
            if self._developmental_age >= 0.5:
                try:
                    # Import NLTK for more advanced text analysis
                    import nltk
                    from nltk.tokenize import word_tokenize
                    from nltk.tag import pos_tag
                    
                    # Ensure NLTK resources are available
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                        
                    try:
                        nltk.data.find('taggers/averaged_perceptron_tagger')
                    except LookupError:
                        nltk.download('averaged_perceptron_tagger', quiet=True)
                    
                    # Extract parts of speech
                    tokens = word_tokenize(description)
                    tagged = pos_tag(tokens)
                    
                    # Count visual properties (adjectives often describe visual attributes)
                    pos_counts = Counter(tag for _, tag in tagged)
                    adjective_ratio = pos_counts.get('JJ', 0) / max(1, len(tagged))
                    
                    features.append(SensoryFeature(
                        name="visual_detail_level",
                        value=min(1.0, adjective_ratio * 3),  # Scale up for better range
                        modality=SensoryModality.VISUAL
                    ))
                    
                    # Extract color information from adjectives
                    common_colors = {
                        "red", "green", "blue", "yellow", "black", "white", "gray", "grey",
                        "purple", "pink", "orange", "brown", "cyan", "magenta", "violet",
                        "indigo", "teal", "lime", "maroon", "navy", "olive", "silver", "gold"
                    }
                    
                    colors_found = []
                    for word, tag in tagged:
                        if tag.startswith('JJ') and word.lower() in common_colors:
                            colors_found.append(word.lower())
                    
                    # Create features for detected colors
                    for color in set(colors_found):
                        features.append(SensoryFeature(
                            name=f"color_{color}",
                            value=1.0,
                            modality=SensoryModality.VISUAL
                        ))
                    
                    # Color diversity score
                    features.append(SensoryFeature(
                        name="color_diversity",
                        value=min(1.0, len(set(colors_found)) / 5),  # Normalize to 0-1
                        modality=SensoryModality.VISUAL
                    ))
                    
                    # Extract nouns for object detection
                    nouns = [word.lower() for word, tag in tagged if tag.startswith('NN')]
                    
                    # Create scene complexity score based on noun diversity
                    features.append(SensoryFeature(
                        name="object_diversity",
                        value=min(1.0, len(set(nouns)) / 10),  # Normalize to 0-1
                        modality=SensoryModality.VISUAL
                    ))
                except ImportError:
                    logger.warning("NLTK not available for advanced text analysis")
                except Exception as e:
                    logger.error(f"Error in NLTK processing: {e}")
            
            # More sophisticated NLP at even later developmental stages
            if self._developmental_age >= 1.0:
                try:
                    # Try to use spaCy for advanced NLP 
                    import spacy
                    
                    # Try to load English model - only proceed if available
                    try:
                        nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        # Fallback to small model that might be installed
                        try:
                            nlp = spacy.load("en")
                        except OSError:
                            # No spaCy model available
                            raise ImportError("No spaCy model available")
                    
                    # Process the text with spaCy
                    doc = nlp(description)
                    
                    # Extract entities (objects, locations, etc.)
                    entity_types = Counter(ent.label_ for ent in doc.ents)
                    
                    # Map common visual entity types to features
                    visual_entity_types = {
                        "PERSON": "person", "PRODUCT": "object", "WORK_OF_ART": "artwork",
                        "LOC": "location", "GPE": "place", "ORG": "building",
                        "FAC": "facility", "EVENT": "event", "NORP": "group"
                    }
                    
                    for ent_type, count in entity_types.items():
                        if ent_type in visual_entity_types:
                            features.append(SensoryFeature(
                                name=f"entity_{visual_entity_types[ent_type]}",
                                value=min(1.0, count / 3),  # Normalize
                                modality=SensoryModality.VISUAL
                            ))
                    
                    # Spatial relationships through dependency parsing
                    spatial_preps = {"above", "below", "under", "over", "beside", "next", 
                                   "behind", "in front", "inside", "outside", "between"}
                    
                    spatial_relations = []
                    for token in doc:
                        if token.text.lower() in spatial_preps or token.lemma_.lower() in spatial_preps:
                            # Get the connected words in the relation
                            connected = [child.text for child in token.children]
                            if connected and token.head.text:
                                relation = f"{token.lemma_}:{token.head.text}:{','.join(connected)}"
                                spatial_relations.append(relation)
                    
                    # Create spatial relationship features
                    for i, relation in enumerate(spatial_relations[:3]):  # Limit to 3
                        features.append(SensoryFeature(
                            name=f"spatial_relation_{i}",
                            value=1.0,
                            modality=SensoryModality.VISUAL
                        ))
                    
                    # Overall spatial complexity
                    features.append(SensoryFeature(
                        name="spatial_complexity",
                        value=min(1.0, len(spatial_relations) / 5),  # Normalize
                        modality=SensoryModality.VISUAL
                    ))
                    
                    # Analyze verbs for motion/action detection in the scene
                    motion_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB" and 
                                  not token.is_stop and token.lemma_ not in ("be", "have")]
                    
                    features.append(SensoryFeature(
                        name="scene_dynamism",
                        value=min(1.0, len(motion_verbs) / 3),  # Normalize
                        modality=SensoryModality.VISUAL
                    ))
                    
                except ImportError:
                    logger.warning("spaCy not available for advanced NLP processing")
                except Exception as e:
                    logger.error(f"Error in spaCy processing: {e}")
            
            # Most advanced - embedding-based visual semantics
            if self._developmental_age >= 1.5:
                try:
                    # Use embeddings for semantic visual properties
                    from lmm_project.utils.vector_store import get_embeddings
                    
                    # Generate embedding from description
                    embedding = get_embeddings(description)
                    
                    if isinstance(embedding, list) and len(embedding) >= 100:
                        # Use embedding similarity to predict visual properties
                        
                        # Normalize embedding to have values between 0-1
                        embedding_array = np.array(embedding)
                        embedding_min = embedding_array.min()
                        embedding_max = embedding_array.max()
                        normalized_embedding = (embedding_array - embedding_min) / (embedding_max - embedding_min + 1e-10)
                        
                        # Use specific regions of embedding for different properties
                        visual_properties = {
                            "texture_smoothness": np.mean(normalized_embedding[10:15]),
                            "color_warmth": np.mean(normalized_embedding[20:25]),
                            "visual_complexity": np.mean(normalized_embedding[30:35]),
                            "shape_roundness": np.mean(normalized_embedding[40:45]),
                            "lighting_contrast": np.mean(normalized_embedding[50:55]),
                            "object_size": np.mean(normalized_embedding[60:65]),
                            "scene_depth": np.mean(normalized_embedding[70:75]),
                            "visual_harmony": np.mean(normalized_embedding[80:85])
                        }
                        
                        for name, value in visual_properties.items():
                            features.append(SensoryFeature(
                                name=name,
                                value=float(value),  # Ensure it's a Python float
                                modality=SensoryModality.VISUAL
                            ))
                
                except ImportError:
                    logger.warning("Vector embeddings not available for visual semantics")
                except Exception as e:
                    logger.error(f"Error in embedding-based visual processing: {e}")
                
                # Use sentiment analysis for scene mood
                try:
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    
                    try:
                        nltk.data.find('sentiment/vader_lexicon')
                    except LookupError:
                        nltk.download('vader_lexicon', quiet=True)
                    
                    sia = SentimentIntensityAnalyzer()
                    sentiment = sia.polarity_scores(description)
                    
                    features.append(SensoryFeature(
                        name="visual_mood_positivity",
                        value=sentiment['pos'],
                        modality=SensoryModality.VISUAL
                    ))
                    
                    features.append(SensoryFeature(
                        name="visual_mood_negativity",
                        value=sentiment['neg'],
                        modality=SensoryModality.VISUAL
                    ))
                    
                    features.append(SensoryFeature(
                        name="visual_mood_neutrality",
                        value=sentiment['neu'],
                        modality=SensoryModality.VISUAL
                    ))
                    
                except ImportError:
                    logger.warning("NLTK sentiment analysis not available")
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    
        except Exception as e:
            logger.error(f"Error extracting visual features from text: {e}")
            
        return features
    
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
