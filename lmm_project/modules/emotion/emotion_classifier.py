import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import math
from collections import defaultdict, Counter
import re
import random

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.emotion.models import EmotionState

# Initialize logger
logger = logging.getLogger(__name__)

class EmotionClassifier(BaseModule):
    """
    Classifier for mapping dimensional emotions to categorical emotions
    
    This system develops from basic positive/negative distinction
    to nuanced recognition of complex emotional states.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic pleasure/displeasure distinction",
        0.2: "Primary emotion recognition",
        0.4: "Secondary emotion classification",
        0.6: "Mixed emotion recognition",
        0.8: "Complex emotional state understanding",
        1.0: "Nuanced emotion classification"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the emotion classifier
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="emotion_classifier",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Define emotion prototypes in valence-arousal space
        # Values based on psychological research on emotion dimensions
        self.emotion_prototypes = {
            # Primary emotions
            "joy": (0.8, 0.6),         # High valence, moderate-high arousal
            "sadness": (-0.7, 0.2),    # Low valence, low arousal
            "anger": (-0.6, 0.8),      # Low valence, high arousal
            "fear": (-0.7, 0.7),       # Low valence, high arousal
            
            # Secondary emotions
            "surprise": (0.1, 0.8),    # Neutral-positive valence, high arousal
            "disgust": (-0.6, 0.5),    # Low valence, moderate arousal
            "anticipation": (0.4, 0.6), # Positive valence, moderate arousal
            "trust": (0.6, 0.3),       # High valence, low-moderate arousal
            
            # Complex emotions (only available at higher development levels)
            "shame": (-0.4, 0.3),      # Negative valence, low-moderate arousal
            "guilt": (-0.5, 0.4),      # Negative valence, moderate arousal
            "pride": (0.7, 0.5),       # Positive valence, moderate arousal
            "love": (0.9, 0.4),        # Very positive valence, moderate arousal
            "jealousy": (-0.3, 0.6),   # Negative valence, moderate-high arousal
            "awe": (0.5, 0.8),         # Positive valence, high arousal
            "contentment": (0.7, 0.1),  # Positive valence, very low arousal
            "boredom": (-0.2, 0.1),    # Slight negative valence, very low arousal
            
            # Neutral state
            "neutral": (0.0, 0.2)      # Neutral valence, low arousal
        }
        
        # Emotion lexicons for textual emotion detection
        self.emotion_lexicons = {
            "joy": {
                "happy", "joy", "delighted", "pleased", "glad", "cheerful",
                "content", "satisfied", "merry", "jovial", "blissful",
                "ecstatic", "elated", "thrilled", "overjoyed", "exuberant"
            },
            "sadness": {
                "sad", "unhappy", "depressed", "miserable", "gloomy", "melancholy",
                "sorrowful", "downhearted", "downcast", "blue", "dejected",
                "heartbroken", "grief", "distressed", "woeful", "despondent"
            },
            "anger": {
                "angry", "mad", "furious", "enraged", "irate", "irritated",
                "annoyed", "vexed", "indignant", "outraged", "offended",
                "heated", "fuming", "infuriated", "livid", "seething"
            },
            "fear": {
                "afraid", "scared", "frightened", "terrified", "fearful", "anxious",
                "worried", "nervous", "panicked", "alarmed", "horrified",
                "startled", "suspicious", "uneasy", "wary", "dread"
            },
            "surprise": {
                "surprised", "astonished", "amazed", "astounded", "shocked", "stunned",
                "startled", "dumbfounded", "bewildered", "awestruck", "wonderstruck",
                "flabbergasted", "thunderstruck", "dazed", "speechless", "agog"
            },
            "disgust": {
                "disgusted", "revolted", "repulsed", "nauseated", "sickened", "appalled",
                "repelled", "offended", "abhorrent", "loathsome", "detestable",
                "distasteful", "repugnant", "vile", "gross", "creepy"
            },
            "anticipation": {
                "anticipate", "expect", "await", "look forward", "hope", "excited",
                "eager", "enthusiastic", "keen", "prepared", "ready",
                "watchful", "vigilant", "alert", "attentive", "mindful"
            },
            "trust": {
                "trust", "confident", "secure", "faithful", "reliable", "dependable",
                "trustworthy", "honest", "loyal", "sincere", "devoted",
                "authentic", "genuine", "believing", "convinced", "assured"
            },
            "neutral": {
                "neutral", "okay", "fine", "alright", "balanced", "moderate",
                "neither", "indifferent", "impartial", "uninvolved", "dispassionate"
            }
        }
        
        # Classification parameters
        self.params = {
            # Size of influence sphere for each emotion in VA space (how wide the emotion region is)
            "emotion_radii": {
                "neutral": 0.3,  # Neutral has a wider region
                "default": 0.2   # Default for other emotions
            },
            
            # Weight for dimensional vs lexical classification
            "dimensional_weight": 0.7,
            "lexical_weight": 0.3,
            
            # Threshold for emotion detection
            "emotion_threshold": 0.1,
            
            # Whether mixed emotions are allowed
            "allow_mixed_emotions": False,
            
            # Maximum number of simultaneous emotions
            "max_emotions": 1
        }
        
        # Adjust parameters based on development level
        self._adjust_parameters_for_development()
        
        # History of recent classifications
        self.classification_history = []
        
        logger.info(f"Emotion classifier initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust classification parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very basic classification - only pleasure/displeasure
            self.params.update({
                "dimensional_weight": 0.9,   # Mostly dimensional at early stages
                "lexical_weight": 0.1,
                "emotion_threshold": 0.3,    # Higher threshold (less sensitive)
                "allow_mixed_emotions": False,
                "max_emotions": 1
            })
        elif self.development_level < 0.4:
            # Primary emotion recognition
            self.params.update({
                "dimensional_weight": 0.8,
                "lexical_weight": 0.2,
                "emotion_threshold": 0.25,
                "allow_mixed_emotions": False,
                "max_emotions": 1
            })
        elif self.development_level < 0.6:
            # Secondary emotion recognition
            self.params.update({
                "dimensional_weight": 0.7,
                "lexical_weight": 0.3,
                "emotion_threshold": 0.2,
                "allow_mixed_emotions": True,  # Begin allowing mixed emotions
                "max_emotions": 2             # Up to 2 emotions
            })
        elif self.development_level < 0.8:
            # Mixed emotion recognition
            self.params.update({
                "dimensional_weight": 0.6,
                "lexical_weight": 0.4,
                "emotion_threshold": 0.15,
                "allow_mixed_emotions": True,
                "max_emotions": 3             # Up to 3 emotions
            })
        else:
            # Complex emotion recognition
            self.params.update({
                "dimensional_weight": 0.5,
                "lexical_weight": 0.5,
                "emotion_threshold": 0.1,
                "allow_mixed_emotions": True,
                "max_emotions": 4             # Up to 4 emotions
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to classify emotions
        
        Args:
            input_data: Input data to process
                Required keys: 'valence', 'arousal'
                Optional keys: 'text', 'context'
                
        Returns:
            Dictionary with emotion classification results
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract valence and arousal
        valence = input_data.get("valence", 0.0)
        arousal = input_data.get("arousal", 0.0)
        
        # Extract text (if available)
        text = ""
        if "text" in input_data:
            text = input_data["text"]
        elif "content" in input_data:
            content = input_data["content"]
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict) and "text" in content:
                text = content["text"]
        
        # Classification approach based on development level
        if self.development_level < 0.2:
            # Very basic classification at early stages - only positive/negative
            result = self._basic_classification(valence, arousal, text)
        else:
            # More sophisticated classification at higher levels
            result = self._dimensional_classification(valence, arousal, text)
            
        # Add process ID and development info
        result["process_id"] = process_id
        result["development_level"] = self.development_level
        
        # Add to history
        self.classification_history.append({
            "valence": valence,
            "arousal": arousal,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.classification_history) > 50:
            self.classification_history = self.classification_history[-50:]
        
        return result
    
    def _basic_classification(self, valence: float, arousal: float, text: str) -> Dict[str, Any]:
        """
        Basic emotion classification for early development stages
        
        Args:
            valence: Pleasure-displeasure value (-1 to 1)
            arousal: Activation level (0 to 1)
            text: Optional text to analyze
            
        Returns:
            Dictionary with classification results
        """
        # At the most basic level, only distinguish pleasure/displeasure
        if valence > 0.2:
            # Positive emotion (pleasure)
            dominant_emotion = "joy"
            emotion_intensities = {
                "joy": min(1.0, valence + 0.2),
                "neutral": max(0.0, 1.0 - (valence + 0.2))
            }
        elif valence < -0.2:
            # Negative emotion (displeasure)
            # At early stages, don't differentiate between different negative emotions
            if arousal > 0.5:
                # High arousal negative is classified as anger
                dominant_emotion = "anger"
                emotion_intensities = {
                    "anger": min(1.0, abs(valence) + 0.2),
                    "neutral": max(0.0, 1.0 - (abs(valence) + 0.2))
                }
            else:
                # Low arousal negative is classified as sadness
                dominant_emotion = "sadness"
                emotion_intensities = {
                    "sadness": min(1.0, abs(valence) + 0.2),
                    "neutral": max(0.0, 1.0 - (abs(valence) + 0.2))
                }
        else:
            # Neutral emotional state
            dominant_emotion = "neutral"
            emotion_intensities = {
                "neutral": 0.8,
                "joy" if valence >= 0 else "sadness": 0.2
            }
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensities": emotion_intensities,
            "classification_method": "basic",
            "confidence": 0.5 + (0.2 * self.development_level)
        }
    
    def _dimensional_classification(self, valence: float, arousal: float, text: str) -> Dict[str, Any]:
        """
        Dimensional emotion classification for higher development stages
        
        Args:
            valence: Pleasure-displeasure value (-1 to 1)
            arousal: Activation level (0 to 1)
            text: Optional text to analyze
            
        Returns:
            Dictionary with classification results
        """
        # Determine which emotions to consider based on development level
        available_emotions = self._get_available_emotions()
        
        # Classify based on dimensional coordinates
        dimensional_result = self._classify_by_dimensions(valence, arousal, available_emotions)
        
        # If we have text, also classify based on lexical content
        lexical_result = None
        if text:
            lexical_result = self._classify_by_text(text, available_emotions)
            
        # Combine results if we have both
        if lexical_result and self.development_level >= 0.2:
            combined_result = self._combine_classifications(
                dimensional_result, 
                lexical_result,
                self.params["dimensional_weight"],
                self.params["lexical_weight"]
            )
            
            combined_result["classification_method"] = "dimensional+lexical"
            return combined_result
        else:
            # Just use dimensional result
            dimensional_result["classification_method"] = "dimensional"
            return dimensional_result
    
    def _get_available_emotions(self) -> List[str]:
        """Get emotions available at current development level"""
        if self.development_level < 0.2:
            # Very basic emotional distinctions
            return ["joy", "sadness", "anger", "neutral"]
            
        elif self.development_level < 0.4:
            # Primary emotions
            return ["joy", "sadness", "anger", "fear", "neutral"]
            
        elif self.development_level < 0.6:
            # Add secondary emotions
            return ["joy", "sadness", "anger", "fear", 
                   "surprise", "disgust", "anticipation", "trust", 
                   "neutral"]
                   
        elif self.development_level < 0.8:
            # Add some complex emotions
            return ["joy", "sadness", "anger", "fear", 
                   "surprise", "disgust", "anticipation", "trust",
                   "shame", "guilt", "love", "neutral"]
                   
        else:
            # Full range of emotions
            return list(self.emotion_prototypes.keys())
    
    def _classify_by_dimensions(
        self, 
        valence: float, 
        arousal: float, 
        available_emotions: List[str]
    ) -> Dict[str, Any]:
        """
        Classify emotions based on location in dimensional space
        
        Args:
            valence: Pleasure-displeasure value (-1 to 1)
            arousal: Activation level (0 to 1)
            available_emotions: List of emotions to consider
            
        Returns:
            Dictionary with classification results
        """
        # Calculate distance to each emotion prototype
        distances = {}
        for emotion in available_emotions:
            if emotion in self.emotion_prototypes:
                prototype_valence, prototype_arousal = self.emotion_prototypes[emotion]
                # Euclidean distance in VA space
                distance = math.sqrt(
                    (valence - prototype_valence) ** 2 + 
                    (arousal - prototype_arousal) ** 2
                )
                distances[emotion] = distance
        
        # Convert distances to intensities (closer = higher intensity)
        # Use Gaussian activation function
        intensities = {}
        for emotion, distance in distances.items():
            # Get radius for this emotion (how wide the region is)
            radius = self.params["emotion_radii"].get(
                emotion, self.params["emotion_radii"]["default"]
            )
            
            # Calculate intensity using Gaussian function
            # exp(-distance²/radius²) gives 1.0 at center, decreasing with distance
            intensity = math.exp(-(distance ** 2) / (radius ** 2))
            
            # Apply threshold
            if intensity >= self.params["emotion_threshold"]:
                intensities[emotion] = intensity
                
        # If no emotions pass threshold, use neutral
        if not intensities:
            intensities["neutral"] = 1.0
            
        # Normalize intensities to sum to 1.0
        total_intensity = sum(intensities.values())
        if total_intensity > 0:
            intensities = {e: i / total_intensity for e, i in intensities.items()}
            
        # Limit to max number of emotions if mixed emotions aren't allowed
        if not self.params["allow_mixed_emotions"] or len(intensities) > self.params["max_emotions"]:
            # Keep only the strongest emotions up to max_emotions
            top_emotions = sorted(intensities.items(), key=lambda x: x[1], reverse=True)
            top_emotions = top_emotions[:self.params["max_emotions"]]
            
            # Create new intensities dict with only top emotions
            intensities = {e: i for e, i in top_emotions}
            
            # Re-normalize
            total_intensity = sum(intensities.values())
            intensities = {e: i / total_intensity for e, i in intensities.items()}
            
        # Determine dominant emotion (highest intensity)
        dominant_emotion = max(intensities.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensities": intensities,
            "confidence": 0.6 + (0.2 * self.development_level)
        }
    
    def _classify_by_text(
        self, 
        text: str, 
        available_emotions: List[str]
    ) -> Dict[str, Any]:
        """
        Classify emotions based on textual content
        
        Args:
            text: Text to analyze
            available_emotions: List of emotions to consider
            
        Returns:
            Dictionary with classification results
        """
        # Simple lexical approach - look for emotion words
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Count emotion words for each available emotion
        emotion_counts = {}
        for emotion in available_emotions:
            if emotion in self.emotion_lexicons:
                # Count words in this emotion's lexicon
                emotion_words = [token for token in tokens if token in self.emotion_lexicons[emotion]]
                if emotion_words:
                    emotion_counts[emotion] = len(emotion_words)
                    
        # If no emotions detected, use neutral
        if not emotion_counts:
            return {
                "dominant_emotion": "neutral",
                "emotion_intensities": {"neutral": 1.0},
                "confidence": 0.3
            }
            
        # Convert counts to intensities
        total_count = sum(emotion_counts.values())
        intensities = {e: c / total_count for e, c in emotion_counts.items()}
        
        # Apply threshold and limit number of emotions
        intensities = {e: i for e, i in intensities.items() 
                     if i >= self.params["emotion_threshold"]}
                     
        # If no emotions pass threshold, use neutral
        if not intensities:
            intensities["neutral"] = 1.0
            
        # Limit to max number of emotions
        if len(intensities) > self.params["max_emotions"]:
            # Keep only the strongest emotions
            top_emotions = sorted(intensities.items(), key=lambda x: x[1], reverse=True)
            top_emotions = top_emotions[:self.params["max_emotions"]]
            intensities = {e: i for e, i in top_emotions}
            
        # Re-normalize
        total_intensity = sum(intensities.values())
        intensities = {e: i / total_intensity for e, i in intensities.items()}
        
        # Determine dominant emotion (highest intensity)
        dominant_emotion = max(intensities.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensities": intensities,
            "confidence": 0.4 + (0.1 * self.development_level)
        }
    
    def _combine_classifications(
        self,
        dimensional_result: Dict[str, Any],
        lexical_result: Dict[str, Any],
        dimensional_weight: float,
        lexical_weight: float
    ) -> Dict[str, Any]:
        """
        Combine dimensional and lexical classifications
        
        Args:
            dimensional_result: Results from dimensional classification
            lexical_result: Results from lexical classification
            dimensional_weight: Weight for dimensional results
            lexical_weight: Weight for lexical results
            
        Returns:
            Dictionary with combined classification
        """
        # Normalize weights
        total_weight = dimensional_weight + lexical_weight
        dim_weight_norm = dimensional_weight / total_weight
        lex_weight_norm = lexical_weight / total_weight
        
        # Combine emotion intensities
        combined_intensities = {}
        
        # Add dimensional emotions
        for emotion, intensity in dimensional_result["emotion_intensities"].items():
            combined_intensities[emotion] = intensity * dim_weight_norm
            
        # Add lexical emotions
        for emotion, intensity in lexical_result["emotion_intensities"].items():
            if emotion in combined_intensities:
                combined_intensities[emotion] += intensity * lex_weight_norm
            else:
                combined_intensities[emotion] = intensity * lex_weight_norm
                
        # Apply threshold
        combined_intensities = {e: i for e, i in combined_intensities.items() 
                              if i >= self.params["emotion_threshold"]}
                              
        # If empty, use neutral
        if not combined_intensities:
            combined_intensities["neutral"] = 1.0
            
        # Limit to max emotions
        if len(combined_intensities) > self.params["max_emotions"]:
            top_emotions = sorted(combined_intensities.items(), key=lambda x: x[1], reverse=True)
            top_emotions = top_emotions[:self.params["max_emotions"]]
            combined_intensities = {e: i for e, i in top_emotions}
            
        # Re-normalize
        total_intensity = sum(combined_intensities.values())
        combined_intensities = {e: i / total_intensity for e, i in combined_intensities.items()}
        
        # Determine dominant emotion
        dominant_emotion = max(combined_intensities.items(), key=lambda x: x[1])[0]
        
        # Calculate combined confidence
        confidence = (
            dimensional_result["confidence"] * dim_weight_norm +
            lexical_result["confidence"] * lex_weight_norm
        )
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensities": combined_intensities,
            "confidence": confidence
        }
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New developmental level
        """
        # Update base module development
        new_level = super().update_development(amount)
        
        # Adjust parameters for new development level
        self._adjust_parameters_for_development()
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state
        
        Returns:
            Dictionary with current state
        """
        base_state = super().get_state()
        
        # Add classifier-specific state
        classifier_state = {
            "params": self.params,
            "available_emotions": self._get_available_emotions(),
            "history_length": len(self.classification_history),
            "last_classification": self.classification_history[-1] if self.classification_history else None
        }
        
        # Combine states
        combined_state = {**base_state, **classifier_state}
        
        return combined_state 
