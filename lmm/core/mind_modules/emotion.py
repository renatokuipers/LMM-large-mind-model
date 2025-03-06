"""
Emotion module for the Large Mind Model (LMM).

This module handles the emotional state and processes for the LMM,
including emotion recognition, expression, and regulation.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import random
import math
import re
import os
import json
import numpy as np
from collections import defaultdict
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.mind_modules.base import MindModule
from lmm.core.development.stages import DevelopmentalStage

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/sentiwordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('sentiwordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

logger = get_logger("lmm.mind_modules.emotion")

# NRC Emotion Lexicon - A compact version embedded directly
# Full lexicon has over 14,000 words, this is a smaller subset of common emotional words
# Format: {word: {"joy": score, "sadness": score, ...}}
EMOTION_LEXICON = {
    # Joy related
    "happy": {"joy": 0.9, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.1, "trust": 0.5, "anticipation": 0.3, "disgust": 0.0},
    "joy": {"joy": 1.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.2, "trust": 0.3, "anticipation": 0.4, "disgust": 0.0},
    "delight": {"joy": 0.8, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.3, "trust": 0.4, "anticipation": 0.5, "disgust": 0.0},
    "pleased": {"joy": 0.7, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.1, "trust": 0.5, "anticipation": 0.2, "disgust": 0.0},
    "love": {"joy": 0.8, "sadness": 0.1, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.9, "anticipation": 0.4, "disgust": 0.0},
    "grateful": {"joy": 0.6, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.8, "anticipation": 0.2, "disgust": 0.0},
    
    # Sadness related
    "sad": {"joy": 0.0, "sadness": 0.9, "anger": 0.1, "fear": 0.1, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.1},
    "unhappy": {"joy": 0.0, "sadness": 0.8, "anger": 0.2, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.1},
    "grief": {"joy": 0.0, "sadness": 1.0, "anger": 0.1, "fear": 0.2, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    "disappointed": {"joy": 0.0, "sadness": 0.7, "anger": 0.2, "fear": 0.0, "surprise": 0.2, "trust": 0.0, "anticipation": 0.0, "disgust": 0.1},
    "alone": {"joy": 0.0, "sadness": 0.6, "anger": 0.0, "fear": 0.3, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    
    # Anger related
    "angry": {"joy": 0.0, "sadness": 0.1, "anger": 0.9, "fear": 0.1, "surprise": 0.1, "trust": 0.0, "anticipation": 0.0, "disgust": 0.4},
    "mad": {"joy": 0.0, "sadness": 0.0, "anger": 0.8, "fear": 0.0, "surprise": 0.1, "trust": 0.0, "anticipation": 0.0, "disgust": 0.3},
    "furious": {"joy": 0.0, "sadness": 0.0, "anger": 1.0, "fear": 0.1, "surprise": 0.2, "trust": 0.0, "anticipation": 0.0, "disgust": 0.4},
    "annoyed": {"joy": 0.0, "sadness": 0.1, "anger": 0.6, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.3},
    "hate": {"joy": 0.0, "sadness": 0.2, "anger": 0.8, "fear": 0.1, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.7},
    
    # Fear related
    "afraid": {"joy": 0.0, "sadness": 0.2, "anger": 0.0, "fear": 0.9, "surprise": 0.2, "trust": 0.0, "anticipation": 0.1, "disgust": 0.0},
    "scared": {"joy": 0.0, "sadness": 0.1, "anger": 0.0, "fear": 0.9, "surprise": 0.3, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    "anxious": {"joy": 0.0, "sadness": 0.2, "anger": 0.1, "fear": 0.8, "surprise": 0.0, "trust": 0.0, "anticipation": 0.4, "disgust": 0.0},
    "worried": {"joy": 0.0, "sadness": 0.3, "anger": 0.0, "fear": 0.7, "surprise": 0.0, "trust": 0.0, "anticipation": 0.5, "disgust": 0.0},
    "terrified": {"joy": 0.0, "sadness": 0.1, "anger": 0.0, "fear": 1.0, "surprise": 0.3, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    
    # Surprise related
    "surprised": {"joy": 0.3, "sadness": 0.0, "anger": 0.0, "fear": 0.1, "surprise": 0.9, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    "amazed": {"joy": 0.5, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.8, "trust": 0.1, "anticipation": 0.0, "disgust": 0.0},
    "astonished": {"joy": 0.2, "sadness": 0.0, "anger": 0.0, "fear": 0.1, "surprise": 1.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    "shocked": {"joy": 0.0, "sadness": 0.1, "anger": 0.1, "fear": 0.3, "surprise": 0.9, "trust": 0.0, "anticipation": 0.0, "disgust": 0.1},
    "unexpected": {"joy": 0.1, "sadness": 0.0, "anger": 0.0, "fear": 0.2, "surprise": 0.7, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0},
    
    # Trust related
    "trust": {"joy": 0.3, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 1.0, "anticipation": 0.2, "disgust": 0.0},
    "believe": {"joy": 0.1, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.8, "anticipation": 0.1, "disgust": 0.0},
    "reliable": {"joy": 0.2, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.9, "anticipation": 0.0, "disgust": 0.0},
    "faithful": {"joy": 0.3, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.9, "anticipation": 0.0, "disgust": 0.0},
    "honest": {"joy": 0.3, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.9, "anticipation": 0.0, "disgust": 0.0},
    
    # Anticipation related
    "expect": {"joy": 0.1, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.2, "anticipation": 0.8, "disgust": 0.0},
    "anticipate": {"joy": 0.2, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 1.0, "disgust": 0.0},
    "await": {"joy": 0.1, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.8, "disgust": 0.0},
    "excited": {"joy": 0.7, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.2, "trust": 0.0, "anticipation": 0.8, "disgust": 0.0},
    "hope": {"joy": 0.5, "sadness": 0.1, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "trust": 0.4, "anticipation": 0.9, "disgust": 0.0},
    
    # Disgust related
    "disgust": {"joy": 0.0, "sadness": 0.2, "anger": 0.4, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 1.0},
    "gross": {"joy": 0.0, "sadness": 0.0, "anger": 0.1, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.8},
    "revolting": {"joy": 0.0, "sadness": 0.0, "anger": 0.3, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.9},
    "yuck": {"joy": 0.0, "sadness": 0.0, "anger": 0.1, "fear": 0.0, "surprise": 0.1, "trust": 0.0, "anticipation": 0.0, "disgust": 0.8},
    "nasty": {"joy": 0.0, "sadness": 0.0, "anger": 0.3, "fear": 0.0, "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.7}
}

# Emotion intensifiers
INTENSIFIERS = {
    "very": 0.3,
    "extremely": 0.5,
    "really": 0.3,
    "so": 0.2,
    "incredibly": 0.4,
    "absolutely": 0.4,
    "totally": 0.3,
    "completely": 0.3,
    "utterly": 0.4,
    "deeply": 0.35
}

# Negation words
NEGATION_WORDS = ["not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely", "nobody", "nothing", "nowhere", "cannot", "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"]

class EmotionModule(MindModule):
    """
    Handles emotional processes for the LMM.
    
    This module manages the emotional state of the LMM, including
    recognition of emotions in text, emotional reactions, and
    developmental changes in emotional capacity.
    """
    
    def __init__(self):
        """Initialize the Emotion Module."""
        super().__init__("Emotion")
        
        # Initialize emotional state
        self.current_state = {
            "joy": 0.2,
            "sadness": 0.1,
            "anger": 0.05,
            "fear": 0.05,
            "surprise": 0.1,
            "trust": 0.3,
            "anticipation": 0.2
        }
        
        # Emotional development parameters
        self.emotional_capacity = 0.3  # Starts low, increases with development
        self.emotional_reactivity = 0.5  # How strongly emotions change based on input
        self.emotional_regulation = 0.2  # Ability to regulate emotions
        
        # Emotional memory
        self.emotional_memory = []
        
        logger.info("Initialized Emotion Module with baseline emotional state")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process emotion-related operations.
        
        Args:
            input_data: Dictionary containing operation and parameters
                - operation: One of 'update', 'get_state', 'analyze'
                - input: Input text to analyze (for update or analyze)
                - response: Response text (for update)
                - developmental_stage: Current developmental stage
                
        Returns:
            Dictionary with operation results
        """
        operation = input_data.get("operation", "get_state")
        stage = input_data.get("developmental_stage", DevelopmentalStage.PRENATAL.value)
        
        # Update developmental parameters based on stage
        self._update_developmental_parameters(stage)
        
        if operation == "update":
            return self._update_emotional_state(input_data)
        elif operation == "get_state":
            return self._get_emotional_state()
        elif operation == "analyze":
            return self._analyze_emotional_content(input_data)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def _update_emotional_state(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the emotional state based on input.
        
        Args:
            input_data: Dictionary with input parameters
                - input: Input text
                - response: Response text
                
        Returns:
            Dictionary with operation results
        """
        input_text = input_data.get("input", "")
        response_text = input_data.get("response", "")
        
        # Analyze emotional content in the input
        input_emotions = self._detect_emotions(input_text)
        
        # Calculate emotional reactions
        reactions = {}
        for emotion, intensity in input_emotions.items():
            # Emotional reaction is influenced by current state and reactivity
            reaction = intensity * self.emotional_reactivity
            reactions[emotion] = reaction
        
        # Update emotional state
        new_state = dict(self.current_state)
        for emotion, reaction in reactions.items():
            if emotion in new_state:
                new_state[emotion] += reaction
            else:
                new_state[emotion] = reaction
        
        # Apply emotional regulation
        for emotion in new_state:
            # Regulate extreme emotions
            if new_state[emotion] > 0.7:
                regulation = (new_state[emotion] - 0.7) * self.emotional_regulation
                new_state[emotion] -= regulation
            
            # Ensure values are in valid range
            new_state[emotion] = max(0.0, min(1.0, new_state[emotion]))
        
        # Normalize state to ensure sum is reasonable
        total = sum(new_state.values())
        if total > 1.5:  # Allow some emotions to co-exist
            factor = 1.5 / total
            new_state = {k: v * factor for k, v in new_state.items()}
        
        # Update current state
        self.current_state = new_state
        
        # Record emotional memory
        self.emotional_memory.append({
            "timestamp": datetime.now().isoformat(),
            "trigger": input_text[:100] if len(input_text) > 100 else input_text,
            "state": dict(self.current_state),
            "developmental_stage": input_data.get("developmental_stage")
        })
        
        # Limit memory size
        if len(self.emotional_memory) > 50:
            self.emotional_memory = self.emotional_memory[-50:]
        
        return {
            "success": True,
            "operation": "update",
            "state": dict(self.current_state)
        }
    
    def _get_emotional_state(self) -> Dict[str, Any]:
        """
        Get the current emotional state.
        
        Returns:
            Dictionary with emotional state
        """
        # Calculate primary emotion
        primary_emotion = max(self.current_state.items(), key=lambda x: x[1])
        
        return {
            "success": True,
            "operation": "get_state",
            "state": dict(self.current_state),
            "primary_emotion": primary_emotion[0],
            "primary_intensity": primary_emotion[1],
            "emotional_capacity": self.emotional_capacity,
            "emotional_regulation": self.emotional_regulation
        }
    
    def _analyze_emotional_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotional content in text.
        
        Args:
            input_data: Dictionary with input parameters
                - input: Text to analyze
                
        Returns:
            Dictionary with analysis results
        """
        text = input_data.get("input", "")
        emotions = self._detect_emotions(text)
        
        return {
            "success": True,
            "operation": "analyze",
            "emotions": emotions,
            "valence": self._calculate_valence(emotions),
            "arousal": self._calculate_arousal(emotions)
        }
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text using advanced NLP techniques and emotional lexicons.
        
        This comprehensive emotion detection system uses:
        1. Lexicon-based emotion analysis with word-level emotion scores
        2. Contextual analysis including negation handling and intensifiers
        3. Sentence-level emotional flow modeling
        4. Syntactic and semantic pattern recognition for emotional expressions
        5. Development-appropriate emotional comprehension
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion intensities for eight primary emotions
        """
        # Early return for empty text
        if not text.strip():
            return {
                "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
                "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0
            }
            
        # Get config for development stage
        config = get_config()
        stage = config.development.current_stage
        
        # Initialize emotion intensities
        emotions = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0
        }
        
        # Basic sentiment analysis (provides foundation for emotional understanding)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1.0 to 1.0
        subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0
        
        # Split text into sentences for context awareness
        sentences = sent_tokenize(text)
        sentence_emotions = []
        
        # Process each sentence to maintain local context
        for sentence in sentences:
            sentence_emotion_scores = self._process_sentence_emotions(sentence, stage)
            sentence_emotions.append(sentence_emotion_scores)
        
        # Aggregate sentence-level emotions with recency bias (more recent sentences have stronger effect)
        if sentence_emotions:
            recency_weights = np.linspace(0.7, 1.0, len(sentence_emotions))
            for i, sent_emo in enumerate(sentence_emotions):
                weight = recency_weights[i]
                for emotion, score in sent_emo.items():
                    emotions[emotion] += score * weight
        
        # Apply development-appropriate emotional complexity scaling
        emotions = self._apply_developmental_emotion_scaling(emotions, stage)
        
        # Normalize emotion scores to a 0.0-1.0 scale
        total = sum(emotions.values())
        if total > 0:
            normalization_factor = max(emotions.values())
            if normalization_factor > 0:
                emotions = {k: min(v / normalization_factor, 1.0) for k, v in emotions.items()}
        
        # Apply base sentiment polarity to guide emotion distribution
        if polarity > 0.2:
            # Positive sentiment boosts positive emotions
            emotions["joy"] = min(emotions["joy"] * 1.25, 1.0)
            emotions["trust"] = min(emotions["trust"] * 1.15, 1.0)
        elif polarity < -0.2:
            # Negative sentiment boosts negative emotions
            emotions["sadness"] = min(emotions["sadness"] * 1.15, 1.0)
            emotions["anger"] = min(emotions["anger"] * 1.15, 1.0)
            emotions["fear"] = min(emotions["fear"] * 1.15, 1.0)
            emotions["disgust"] = min(emotions["disgust"] * 1.15, 1.0)
        
        # Ensure emotion intensities stay within bounds
        emotions = {k: max(0.0, min(v, 1.0)) for k, v in emotions.items()}
        
        return emotions
    
    def _process_sentence_emotions(self, sentence: str, stage: str) -> Dict[str, float]:
        """Process emotions in a single sentence with context awareness."""
        # Initialize emotion scores for this sentence
        emotion_scores = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "surprise": 0.0, "trust": 0.0, "anticipation": 0.0, "disgust": 0.0
        }
        
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        
        # Part-of-speech tagging for better word sense disambiguation
        pos_tags = nltk.pos_tag(tokens)
        
        # Track negation and intensifiers
        negation_active = False
        current_intensifier = 0.0
        
        # Process each token in context
        for i, (token, pos) in enumerate(pos_tags):
            # Check for negation
            if token in NEGATION_WORDS:
                negation_active = True
                continue
                
            # Check for intensifiers
            if token in INTENSIFIERS:
                current_intensifier = INTENSIFIERS[token]
                continue
            
            # Lemmatize token for better lexicon matching
            if pos.startswith('J'):
                lemma = lemmatizer.lemmatize(token, wordnet.ADJ)
            elif pos.startswith('V'):
                lemma = lemmatizer.lemmatize(token, wordnet.VERB)
            elif pos.startswith('R'):
                lemma = lemmatizer.lemmatize(token, wordnet.ADV)
            else:
                lemma = lemmatizer.lemmatize(token, wordnet.NOUN)
                
            # Apply emotion scoring based on lexicon
            if lemma in EMOTION_LEXICON:
                # Get base emotion scores for this word
                word_emotions = EMOTION_LEXICON[lemma]
                
                # Apply negation if active
                if negation_active:
                    # Negate emotions (invert the scores)
                    for emotion, score in word_emotions.items():
                        reversed_score = 1.0 - score if score > 0.5 else 0.0
                        emotion_scores[emotion] += reversed_score
                    negation_active = False  # Reset negation after application
                else:
                    # Apply normal emotion scores with any intensifiers
                    for emotion, score in word_emotions.items():
                        emotion_scores[emotion] += score * (1.0 + current_intensifier)
                    current_intensifier = 0.0  # Reset intensifier after application
        
        # Pattern-based emotional expressions
        emotion_patterns = {
            # Joy patterns
            r"(?:^|\s)(?:i\s+(?:am|feel|am\s+feeling)\s+happy|makes\s+me\s+happy|happy\s+to)": {"joy": 0.8},
            r"(?:^|\s)(?:yay|woohoo|hurray|congratulations|congrats)(?:$|\s|!)": {"joy": 0.9, "surprise": 0.3},
            
            # Sadness patterns
            r"(?:^|\s)(?:i\s+(?:am|feel|am\s+feeling)\s+sad|makes\s+me\s+sad|sorry\s+to\s+hear)": {"sadness": 0.8},
            r"(?:^|\s)(?:alas|sigh|oh\s+no|too\s+bad)(?:$|\s|!)": {"sadness": 0.7},
            
            # Anger patterns
            r"(?:^|\s)(?:i\s+(?:am|feel|am\s+feeling)\s+angry|makes\s+me\s+angry|how\s+dare)": {"anger": 0.8},
            r"(?:^|\s)(?:grr|argh|damn|what\s+the\s+hell)(?:$|\s|!)": {"anger": 0.7, "disgust": 0.3},
            
            # Fear patterns
            r"(?:^|\s)(?:i\s+(?:am|feel|am\s+feeling)\s+scared|makes\s+me\s+afraid|terrified\s+of)": {"fear": 0.8},
            r"(?:^|\s)(?:oh\s+my\s+god|omg|holy\s+cow|yikes)(?:$|\s|!)": {"fear": 0.5, "surprise": 0.5},
            
            # Surprise patterns
            r"(?:^|\s)(?:i\s+(?:am|feel|am\s+feeling)\s+surprised|surprising|unexpected)": {"surprise": 0.8},
            r"(?:^|\s)(?:wow|whoa|oh\s+my|really|seriously)(?:$|\s|\?)": {"surprise": 0.7},
            
            # Multiple exclamation or question marks indicate emotional intensity
            r"!{2,}": {"intensity": 0.3},
            r"\?{2,}": {"surprise": 0.3, "intensity": 0.2}
        }
        
        # Apply pattern matching
        for pattern, pattern_emotions in emotion_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                for emotion, score in pattern_emotions.items():
                    if emotion == "intensity":
                        # Apply intensity boost to existing emotions
                        for e in emotion_scores:
                            emotion_scores[e] *= (1.0 + score)
                    else:
                        emotion_scores[emotion] += score
        
        # Normalize scores
        return {k: min(v, 1.0) for k, v in emotion_scores.items()}
    
    def _apply_developmental_emotion_scaling(self, emotions: Dict[str, float], stage: str) -> Dict[str, float]:
        """Apply developmental stage-appropriate scaling to emotions."""
        # Define stage-based emotion comprehension capabilities
        stage_capabilities = {
            DevelopmentalStage.PRENATAL.value: {
                # Very limited emotional range in prenatal stage
                "primary_only": True,  # Only basic emotions
                "complexity": 0.2,     # Limited complexity
                "primary_emotions": ["joy", "sadness"],  # Only these are understood
                "secondary_emotions": []
            },
            DevelopmentalStage.INFANCY.value: {
                # Basic emotions in infancy
                "primary_only": True,
                "complexity": 0.4,
                "primary_emotions": ["joy", "sadness", "fear", "surprise"],
                "secondary_emotions": []
            },
            DevelopmentalStage.CHILDHOOD.value: {
                # Expanded but still developing emotional range
                "primary_only": False,
                "complexity": 0.6,
                "primary_emotions": ["joy", "sadness", "anger", "fear", "surprise"],
                "secondary_emotions": ["trust", "anticipation"]
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                # Nearly complete emotional understanding
                "primary_only": False,
                "complexity": 0.8,
                "primary_emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
                "secondary_emotions": ["trust", "anticipation"]
            },
            DevelopmentalStage.ADULTHOOD.value: {
                # Full emotional capabilities
                "primary_only": False,
                "complexity": 1.0,
                "primary_emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
                "secondary_emotions": ["trust", "anticipation"]
            }
        }
        
        # Get capabilities for current stage
        capabilities = stage_capabilities.get(stage, stage_capabilities[DevelopmentalStage.CHILDHOOD.value])
        
        # Apply developmental filtering
        scaled_emotions = {}
        for emotion, score in emotions.items():
            # Primary emotions are felt more strongly
            if emotion in capabilities["primary_emotions"]:
                scaled_emotions[emotion] = score * capabilities["complexity"]
            # Secondary emotions develop later and are felt less strongly in early stages
            elif emotion in capabilities["secondary_emotions"]:
                scaled_emotions[emotion] = score * capabilities["complexity"] * 0.8
            # Emotions not yet understood are dampened
            else:
                scaled_emotions[emotion] = score * 0.1  # Minimal recognition
        
        return scaled_emotions
    
    def _calculate_valence(self, emotions: Dict[str, float]) -> float:
        """
        Calculate emotional valence (positive vs. negative).
        
        Args:
            emotions: Dictionary with emotion intensities
            
        Returns:
            Valence score (-1.0 to 1.0)
        """
        positive = emotions.get("joy", 0) + emotions.get("trust", 0) + emotions.get("anticipation", 0)
        negative = emotions.get("sadness", 0) + emotions.get("anger", 0) + emotions.get("fear", 0)
        
        # Surprise can be positive or negative
        surprise = emotions.get("surprise", 0)
        
        # Calculate valence (normalize to -1 to 1)
        total = positive + negative + surprise
        if total > 0:
            valence = (positive - negative) / total
        else:
            valence = 0.0
        
        return valence
    
    def _calculate_arousal(self, emotions: Dict[str, float]) -> float:
        """
        Calculate emotional arousal (intensity).
        
        Args:
            emotions: Dictionary with emotion intensities
            
        Returns:
            Arousal score (0.0 to 1.0)
        """
        # High arousal emotions
        high_arousal = emotions.get("joy", 0) + emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("surprise", 0)
        
        # Low arousal emotions
        low_arousal = emotions.get("sadness", 0) + emotions.get("trust", 0)
        
        # Medium arousal emotions
        medium_arousal = emotions.get("anticipation", 0)
        
        # Calculate arousal (normalize to 0 to 1)
        total = sum(emotions.values())
        if total > 0:
            arousal = (high_arousal + (medium_arousal * 0.5)) / total
        else:
            arousal = 0.0
        
        return arousal
    
    def _update_developmental_parameters(self, stage: str) -> None:
        """
        Update emotional parameters based on developmental stage.
        
        Args:
            stage: Current developmental stage
        """
        # Define emotional development by stage
        stage_params = {
            DevelopmentalStage.PRENATAL.value: {
                "emotional_capacity": 0.1,
                "emotional_reactivity": 0.8,
                "emotional_regulation": 0.1
            },
            DevelopmentalStage.INFANCY.value: {
                "emotional_capacity": 0.3,
                "emotional_reactivity": 0.7,
                "emotional_regulation": 0.2
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "emotional_capacity": 0.5,
                "emotional_reactivity": 0.6,
                "emotional_regulation": 0.4
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "emotional_capacity": 0.7,
                "emotional_reactivity": 0.5,
                "emotional_regulation": 0.6
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "emotional_capacity": 0.8,
                "emotional_reactivity": 0.6,
                "emotional_regulation": 0.7
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "emotional_capacity": 0.9,
                "emotional_reactivity": 0.5,
                "emotional_regulation": 0.9
            }
        }
        
        # Get parameters for current stage
        params = stage_params.get(stage, stage_params[DevelopmentalStage.PRENATAL.value])
        
        # Update parameters
        self.emotional_capacity = params["emotional_capacity"]
        self.emotional_reactivity = params["emotional_reactivity"]
        self.emotional_regulation = params["emotional_regulation"]
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get the current status of the emotion module.
        
        Returns:
            Dictionary with module status
        """
        # Get the base status
        status = super().get_module_status()
        
        # Add emotion-specific status
        primary_emotion = max(self.current_state.items(), key=lambda x: x[1])
        
        status.update({
            "current_state": dict(self.current_state),
            "primary_emotion": primary_emotion[0],
            "primary_intensity": primary_emotion[1],
            "emotional_capacity": self.emotional_capacity,
            "emotional_reactivity": self.emotional_reactivity,
            "emotional_regulation": self.emotional_regulation
        })
        
        return status 