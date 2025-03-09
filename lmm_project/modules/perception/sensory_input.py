"""
Sensory Input Processing Module

This module is responsible for processing raw text input into a form
suitable for further processing by the perception system. It serves
as the primary sensory interface for the LMM, converting text into
meaningful sensory representations.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
import logging
from datetime import datetime
from collections import deque, Counter
import string
import torch

# Add tokenization libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class SensoryInputProcessor(BaseModule):
    """
    Processes raw text input into a form suitable for perception
    
    This module is the first stage of perception, converting raw text
    into feature vectors and preliminary sensory representations.
    """
    # Development stages for sensory processing
    development_milestones = {
        0.0: "Basic text detection",
        0.2: "Simple tokenization",
        0.4: "Feature extraction",
        0.6: "Multi-level analysis",
        0.8: "Context sensitivity",
        1.0: "Advanced sensory processing"
    }
    
    def __init__(
        self, 
        module_id: str,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        """
        Initialize the sensory input processor
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
        """
        super().__init__(
            module_id=module_id,
            module_type="sensory_processor",
            event_bus=event_bus,
            **kwargs
        )
        
        # Initialize sensory memory
        self.recent_inputs = deque(maxlen=20)
        self.input_history = []
        
        # Initialize frequency tracking
        self.token_frequencies = Counter()
        self.bigram_frequencies = Counter()
        self.character_frequencies = Counter()
        
        # Processing parameters that develop over time
        self.max_tokens = 100  # Maximum tokens to process
        self.token_threshold = 0.1  # Threshold for token significance
        self.similarity_threshold = 0.7  # Threshold for similar inputs
        
        # Get stopwords for filtering
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set(['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it'])
            
        # Try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Subscribe to raw text input events
        if self.event_bus:
            self.subscribe_to_message("raw_text_input")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw input text into sensory representations
        
        Args:
            input_data: Dictionary containing input data with text
            
        Returns:
            Dictionary with processed sensory data
        """
        # Validate input
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        text = input_data.get("text", "")
        
        if not text:
            logger.warning(f"Sensory processor received empty text input for process {process_id}")
            return {
                "process_id": process_id,
                "status": "error",
                "error": "Empty input text"
            }
            
        # Log the incoming input
        logger.debug(f"Processing sensory input: '{text[:50]}...' (process {process_id})")
        
        # Create response structure
        timestamp = datetime.now()
        result = {
            "process_id": process_id,
            "timestamp": timestamp.isoformat(),
            "text": text,
            "development_level": self.development_level,
            "module_id": self.module_id
        }
        
        # Extract features based on development level
        features = self._extract_features(text)
        
        # Add features to result
        result.update(features)
        
        # Store in recent inputs
        self.recent_inputs.append({
            "text": text,
            "process_id": process_id,
            "timestamp": timestamp,
            "features": features
        })
        
        # Track occurrence frequencies
        self._update_frequencies(text)
        
        # Publish processed result if we have an event bus
        if self.event_bus:
            self.publish_message(
                "sensory_processed",
                {"result": result, "process_id": process_id}
            )
            
        return result
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text based on developmental level
        
        At lower developmental levels, only basic features are extracted.
        As development progresses, more sophisticated features are extracted.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Start with empty containers
        basic_features = {}
        intermediate_features = {}
        advanced_features = {}
        
        # Level 0: Basic text properties (always extracted)
        basic_features["length"] = len(text)
        basic_features["has_letters"] = bool(re.search(r'[a-zA-Z]', text))
        basic_features["has_numbers"] = bool(re.search(r'\d', text))
        basic_features["has_special_chars"] = bool(re.search(r'[^\w\s]', text))
        
        # Count frequencies
        letter_count = sum(c.isalpha() for c in text)
        number_count = sum(c.isdigit() for c in text)
        special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
        
        basic_features["letter_count"] = letter_count
        basic_features["number_count"] = number_count
        basic_features["special_char_count"] = special_char_count
        basic_features["whitespace_count"] = text.count(' ') + text.count('\n') + text.count('\t')
        
        # Calculate character distribution
        if text:
            basic_features["letter_ratio"] = letter_count / len(text)
            basic_features["number_ratio"] = number_count / len(text)
            basic_features["special_ratio"] = special_char_count / len(text)
        else:
            basic_features["letter_ratio"] = 0
            basic_features["number_ratio"] = 0
            basic_features["special_ratio"] = 0
            
        # Add some additional basic features even at level 0
        # This provides better information for pattern recognition
        if text:
            # Simple token count (just splitting by whitespace)
            simple_tokens = text.split()
            basic_features["simple_token_count"] = len(simple_tokens)
            
            # Check for question or exclamation
            basic_features["has_question_mark"] = '?' in text
            basic_features["has_exclamation_mark"] = '!' in text
            
            # Add some character n-gram counts
            char_bigrams = [text[i:i+2] for i in range(len(text)-1)]
            char_bigram_counts = Counter(char_bigrams)
            basic_features["common_char_bigrams"] = dict(char_bigram_counts.most_common(5))
            
            # Create a unique signature for the text
            basic_features["text_signature"] = hash(text) % 10000
        
        # Level 1: Token-based features (dev level >= 0.2)
        # But we'll extract some token features even at level 0 
        # to support better pattern recognition
        
        # Tokenize text using our helper method
        tokens = self._tokenize_text(text)
        
        # Extract token features
        intermediate_features["token_count"] = len(tokens)
        intermediate_features["unique_token_count"] = len(set(tokens))
        
        # Calculate token statistics
        if tokens:
            intermediate_features["avg_token_length"] = np.mean([len(t) for t in tokens])
            intermediate_features["max_token_length"] = max(len(t) for t in tokens)
            intermediate_features["min_token_length"] = min(len(t) for t in tokens)
            intermediate_features["has_long_tokens"] = any(len(t) > 8 for t in tokens)
            
            # Get token frequency distribution
            freq_dist = FreqDist(tokens)
            most_common = freq_dist.most_common(5)
            intermediate_features["most_common_tokens"] = most_common
            intermediate_features["token_diversity"] = len(set(tokens)) / len(tokens) if tokens else 0
            
            # Add basic n-gram features even at low levels
            if len(tokens) >= 2:
                # Generate bigrams
                token_bigrams = list(ngrams(tokens, 2))
                intermediate_features["bigram_count"] = len(token_bigrams)
                
                # Get most common bigrams
                if token_bigrams:
                    bigram_freq = FreqDist(token_bigrams)
                    intermediate_features["common_bigrams"] = bigram_freq.most_common(3)
        else:
            intermediate_features["avg_token_length"] = 0
            intermediate_features["max_token_length"] = 0
            intermediate_features["min_token_length"] = 0
            intermediate_features["has_long_tokens"] = False
            intermediate_features["most_common_tokens"] = []
            intermediate_features["token_diversity"] = 0
            intermediate_features["bigram_count"] = 0
            intermediate_features["common_bigrams"] = []
                
        # Level 2: Linguistic features (dev level >= 0.4)
        if self.development_level >= 0.4:
            # Extract sentences
            sentences = sent_tokenize(text)
            
            # Sentence features
            advanced_features["sentence_count"] = len(sentences)
            
            if sentences:
                advanced_features["avg_sentence_length"] = np.mean([len(s) for s in sentences])
                advanced_features["max_sentence_length"] = max(len(s) for s in sentences)
                
                # Words per sentence
                words_per_sentence = [len(self._tokenize_text(s)) for s in sentences]
                advanced_features["avg_words_per_sentence"] = np.mean(words_per_sentence) if words_per_sentence else 0
            else:
                advanced_features["avg_sentence_length"] = 0
                advanced_features["max_sentence_length"] = 0
                advanced_features["avg_words_per_sentence"] = 0
                
            # Check for question marks and exclamation points
            advanced_features["question_mark_count"] = text.count('?')
            advanced_features["exclamation_mark_count"] = text.count('!')
            advanced_features["is_question"] = text.strip().endswith('?')
            advanced_features["is_exclamation"] = text.strip().endswith('!')
                
        # Level 3: Context-sensitive features (dev level >= 0.6)
        if self.development_level >= 0.6:
            # Check similarity to recent inputs
            if self.recent_inputs:
                similarities = []
                for recent in self.recent_inputs:
                    if recent.get("text") != text:  # Don't compare to self
                        similarity = self._simple_similarity(text, recent.get("text", ""))
                        similarities.append(similarity)
                
                if similarities:
                    advanced_features["max_similarity_to_recent"] = max(similarities)
                    advanced_features["avg_similarity_to_recent"] = np.mean(similarities)
                    advanced_features["is_similar_to_recent"] = max(similarities) > self.similarity_threshold
                else:
                    advanced_features["max_similarity_to_recent"] = 0
                    advanced_features["avg_similarity_to_recent"] = 0
                    advanced_features["is_similar_to_recent"] = False
            
            # Word frequencies compared to historical frequencies
            tokens = self._tokenize_text(text)
            if tokens:
                # Check for unusual words (not in frequent tokens)
                unusual_tokens = [t for t in tokens if self.token_frequencies[t] < 3 and len(t) > 3]
                advanced_features["unusual_token_count"] = len(unusual_tokens)
                advanced_features["unusual_tokens"] = unusual_tokens[:5]  # Limit to 5
                
                # Calculate frequency novelty (how different from typical frequency)
                if self.token_frequencies:
                    token_freqs = {t: self.token_frequencies[t] for t in tokens}
                    if token_freqs:
                        avg_freq = np.mean(list(token_freqs.values()))
                        advanced_features["frequency_novelty"] = 1.0 - min(avg_freq / 10, 1.0)
                    else:
                        advanced_features["frequency_novelty"] = 1.0
                else:
                    advanced_features["frequency_novelty"] = 0.5  # Default mid value
            
        # Level 4: Advanced analysis (dev level >= 0.8)
        if self.development_level >= 0.8:
            # More sophisticated linguistic analysis would go here
            # This could include sentiment analysis, topic detection, etc.
            
            # For now, we'll add some simple additional features
            tokens = self._tokenize_text(text)
            
            # Check for specific linguistic features
            linguistic_features = {}
            
            # Question words
            question_words = {"what", "who", "when", "where", "why", "how"}
            linguistic_features["has_question_words"] = any(t.lower() in question_words for t in tokens)
            
            # Check for imperative sentences (commands)
            # Simple approach: starts with verb
            if tokens and tokens[0].lower() in {"do", "go", "be", "try", "make", "take", "get", "come", "give", "find", "look", "run", "turn", "put", "bring"}:
                linguistic_features["likely_imperative"] = True
            else:
                linguistic_features["likely_imperative"] = False
                
            # Check for named entities (very simple approach)
            # Look for capitalized words not at the start of sentences
            sentence_starts = {s.split()[0] if s.split() else "" for s in sent_tokenize(text)}
            capitalized_non_starters = [t for t in tokens if t[0].isupper() and t not in sentence_starts]
            linguistic_features["potential_named_entities"] = capitalized_non_starters[:5]
            
            # Add to advanced features
            advanced_features["linguistic_features"] = linguistic_features
        
        # Assemble features based on development level
        features["basic_features"] = basic_features
        
        if self.development_level >= 0.2:
            features["features"] = intermediate_features
            
        if self.development_level >= 0.4:
            features["linguistic_features"] = advanced_features
        
        return features
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words, handling various cases based on development level
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Basic tokenization - always use NLTK for better tokenization
        tokens = word_tokenize(text.lower())
        
        # Filter based on development level
        if self.development_level < 0.3:
            # Basic level - keep most tokens but remove punctuation-only tokens
            filtered_tokens = []
            for token in tokens:
                # Keep tokens that have at least one alphanumeric character
                if any(c.isalnum() for c in token):
                    filtered_tokens.append(token)
            return filtered_tokens[:self.max_tokens]
            
        elif self.development_level < 0.6:
            # Intermediate: filter out stopwords and very short tokens
            filtered_tokens = [t for t in tokens if (t not in self.stopwords or t in {'?', '!'}) and len(t) > 1]
            return filtered_tokens[:self.max_tokens]
            
        else:
            # Advanced: keep more structure and context
            # Just remove punctuation-only tokens except for meaningful punctuation
            important_punct = {'?', '!', '.'}
            filtered_tokens = [t for t in tokens if any(c.isalnum() for c in t) or t in important_punct]
            return filtered_tokens[:self.max_tokens]
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # Convert to sets of tokens
        tokens1 = set(self._tokenize_text(text1))
        tokens2 = set(self._tokenize_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _update_frequencies(self, text: str):
        """
        Update the frequency counters with new text
        
        Args:
            text: Text to process for frequency updates
        """
        if not text:
            return
            
        # Update character frequencies
        self.character_frequencies.update(text)
        
        # Update token frequencies
        tokens = self._tokenize_text(text)
        self.token_frequencies.update(tokens)
        
        # Update bigram frequencies if enough tokens
        if len(tokens) >= 2:
            bigrams = ngrams(tokens, 2)
            self.bigram_frequencies.update(bigrams)
    
    def _handle_message(self, message: Message):
        """
        Handle incoming messages from the event bus
        
        Args:
            message: The message to handle
        """
        if message.message_type == "raw_text_input" and message.content:
            # Process the raw text input
            text = message.content.get("text", "")
            if text:
                process_id = message.content.get("process_id", str(uuid.uuid4()))
                self.process_input({
                    "text": text,
                    "process_id": process_id,
                    "source": message.content.get("source", "unknown"),
                    "metadata": message.content.get("metadata", {})
                })
    
    def update_development(self, amount: float) -> float:
        """
        Update the module's developmental level
        
        As development progresses, the sensory processing becomes more sophisticated
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update processing parameters based on development
        self.max_tokens = int(100 + new_level * 400)  # 100 to 500 tokens - higher minimum for better feature extraction
        self.token_threshold = max(0.03, 0.15 - new_level * 0.12)  # Lower threshold with development
        self.similarity_threshold = max(0.4, 0.7 - new_level * 0.3)  # More nuanced similarity detection
        
        # Log development progress
        logger.info(f"Sensory processor {self.module_id} developmental level updated to {new_level:.2f}")
        logger.debug(f"Updated parameters: max_tokens={self.max_tokens}, token_threshold={self.token_threshold:.2f}")
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing module state
        """
        base_state = super().get_state()
        
        # Add sensory processor specific state
        state = {
            **base_state,
            "recent_input_count": len(self.recent_inputs),
            "token_frequency_count": len(self.token_frequencies),
            "processing_parameters": {
                "max_tokens": self.max_tokens,
                "token_threshold": self.token_threshold,
                "similarity_threshold": self.similarity_threshold
            }
        }
        
        return state
    
    def get_recent_inputs(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent inputs
        
        Args:
            count: Maximum number of inputs to return
            
        Returns:
            List of recent inputs
        """
        # Return the most recent inputs, limited by count
        return list(self.recent_inputs)[-count:]
    
    def get_token_frequencies(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get the most frequent tokens
        
        Args:
            top_n: Number of top tokens to return
            
        Returns:
            List of (token, frequency) tuples
        """
        return self.token_frequencies.most_common(top_n)
        
    def clear_history(self):
        """Clear the input history and frequency counters"""
        self.recent_inputs.clear()
        self.token_frequencies.clear()
        self.bigram_frequencies.clear()
        self.character_frequencies.clear() 
