"""
Pattern Recognition Module

This module is responsible for identifying patterns in sensory input.
It progressively develops the ability to recognize increasingly complex
patterns, from simple feature detection to sophisticated pattern extraction.
"""

import logging
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, Counter
from datetime import datetime
import re
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import LLM client for embeddings
from lmm_project.utils.llm_client import LLMClient

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.perception.neural_net import PerceptionNetwork, TemporalPatternNetwork

logger = logging.getLogger(__name__)

class PatternRecognizer(BaseModule):
    """
    Recognizes patterns in sensory input
    
    This module develops from simple feature detection to complex pattern
    recognition, supporting the perception system's ability to identify
    meaningful structures in input.
    """
    # Development stages for pattern recognition
    development_milestones = {
        0.0: "Basic feature detection",
        0.2: "Simple pattern matching",
        0.4: "Pattern abstraction",
        0.6: "Multi-feature patterns",
        0.8: "Context-sensitive patterns",
        1.0: "Advanced pattern recognition"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        """
        Initialize the pattern recognizer
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
        """
        super().__init__(
            module_id=module_id,
            module_type="pattern_recognizer",
            event_bus=event_bus,
            **kwargs
        )
        
        # Initialize neural network
        self.neural_net = PerceptionNetwork(
            input_dim=64,
            hidden_dim=128,
            pattern_dim=32,
            developmental_level=self.development_level
        )
        
        # Initialize temporal pattern network
        self.temporal_net = TemporalPatternNetwork(
            input_dim=32,
            hidden_dim=64,
            sequence_length=5,
            developmental_level=self.development_level
        )
        
        # Pattern memory
        self.known_patterns = {}
        self.pattern_frequency = Counter()
        self.recent_inputs = deque(maxlen=10)
        self.temporal_sequence = deque(maxlen=5)
        
        # Initialize LLM client for embeddings
        self.llm_client = LLMClient()
        self.use_embeddings = True  # Flag to control whether to use embeddings
        
        # Pattern recognition parameters - making these more sensitive
        self.token_sensitivity = 0.8  # Increased from 0.6
        self.ngram_sensitivity = 0.6  # Increased from 0.4
        self.semantic_sensitivity = 0.4  # Increased from 0.3
        self.novelty_threshold = 0.5  # Reduced from 0.6
        self.pattern_activation_threshold = 0.1  # Reduced from 0.2
        
        # For TF-IDF based pattern recognition
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            max_df=0.9,
            ngram_range=(1, 3),
            max_features=100
        )
        self.document_vectors = []
        self.documents = []
        
        # Try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_net.to_device(self.device)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensory input to recognize patterns
        
        Args:
            input_data: Dictionary with sensory processing results
            
        Returns:
            Dictionary with recognized patterns
        """
        # Debug logging to see what's in the input data
        logger.info(f"Pattern recognizer received input with keys: {list(input_data.keys())}")
        
        # Extract process ID or generate new one
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract text from input data
        text = input_data.get("text", "")
        logger.info(f"Pattern recognizer extracted text: '{text[:30]}...' (length: {len(text)})")
        
        if not text:
            logger.warning(f"Pattern recognizer received empty text input for process {process_id}")
            return {
                "process_id": process_id,
                "patterns": [],
                "status": "error",
                "error": "Empty input text"
            }
            
        # Log the input processing
        logger.debug(f"Processing input for patterns: '{text[:50]}...' (process {process_id})")
        
        # Extract relevant features from sensory data
        features = self._extract_relevant_features(input_data)
        
        # Add to recent inputs
        self.recent_inputs.append({
            "text": text,
            "features": features,
            "timestamp": datetime.now()
        })
        
        # Recognize patterns
        patterns = self._recognize_patterns(features)
        
        # Add patterns to temporal sequence for sequence learning
        if patterns and self.development_level >= 0.4:
            # Create a feature vector from the patterns
            pattern_feature = self._create_pattern_feature_vector(patterns)
            
            # Add to temporal sequence
            self.temporal_sequence.append(pattern_feature)
            
            # If we have enough sequence data, process the temporal pattern
            if len(self.temporal_sequence) >= 3 and self.development_level >= 0.6:
                temporal_results = self._process_temporal_sequence()
                # Add temporal pattern to results if found
                if temporal_results and "temporal_pattern" in temporal_results:
                    patterns.append(self._create_pattern(
                        "temporal", 
                        confidence=0.7, 
                        attributes={
                            "sequence_length": len(self.temporal_sequence),
                            "temporal_features": temporal_results.get("temporal_features", {})
                        }
                    ))
        
        # Update the neural network's developmental level if needed
        if abs(self.neural_net.developmental_level - self.development_level) > 0.05:
            self.neural_net.update_developmental_level(self.development_level)
            self.temporal_net.update_developmental_level(self.development_level)
        
        # Create the output result
        result = {
            "process_id": process_id,
            "timestamp": datetime.now().isoformat(),
            "patterns": patterns,
            "pattern_count": len(patterns),
            "development_level": self.development_level,
            "novelty_average": np.mean([p.get("novelty", 0.5) for p in patterns]) if patterns else 0.5
        }
        
        return result
    
    def _extract_relevant_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features relevant for pattern recognition from sensory input
        
        Args:
            input_data: Raw sensory input data
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Get text from input
        text = input_data.get("text", "")
        if not text:
            logger.warning("Empty text received in _extract_relevant_features")
            return features
            
        # Make sure to include text in the features
        features["text"] = text
        
        # Basic text statistics
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["avg_word_length"] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        # Add features from sensory processing if available
        if "basic_features" in input_data:
            features.update(input_data["basic_features"])
            
        if "features" in input_data:
            features.update(input_data["features"])
            
        if "linguistic_features" in input_data:
            features.update(input_data["linguistic_features"])
            
        # Create tensors for neural processing
        features["text_vector"] = self._text_to_vector(text)
            
        return features
    
    def _recognize_patterns(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recognize patterns in the provided features
        
        The pattern recognition approach changes with developmental level:
        - Early stages (0.0-0.2): Simple token pattern detection
        - Intermediate stages (0.2-0.6): N-gram patterns and basic semantics
        - Advanced stages (0.6-1.0): Semantic patterns, syntactic patterns, and context
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            List of recognized patterns
        """
        # Add debug logging
        logger.info(f"Pattern recognition started with development level: {self.development_level:.2f}")
        logger.info(f"Pattern recognition parameters: token={self.token_sensitivity:.2f}, ngram={self.ngram_sensitivity:.2f}, semantic={self.semantic_sensitivity:.2f}, threshold={self.pattern_activation_threshold:.2f}")
        
        patterns = []
        
        # Get text for pattern recognition
        text = features.get("text", "")
        if not text:
            logger.warning("Empty text received for pattern recognition")
            return patterns
            
        # Vector representation for neural processing
        text_vector = features.get("text_vector")
        
        # Always detect basic token and n-gram patterns regardless of developmental level
        # This ensures we always have at least some patterns at any level
        token_patterns = self._detect_token_patterns(text)
        if token_patterns:
            logger.info(f"Detected {len(token_patterns)} token patterns")
            patterns.extend(token_patterns)
        else:
            logger.warning("No token patterns detected")
            
        ngram_patterns = self._detect_ngram_patterns(text)
        if ngram_patterns:
            logger.info(f"Detected {len(ngram_patterns)} n-gram patterns")
            patterns.extend(ngram_patterns)
        else:
            logger.warning("No n-gram patterns detected")
        
        # Adapt additional pattern recognition to developmental level
        if self.development_level >= 0.2:
            # Basic neural processing if we have a text vector
            if text_vector is not None:
                neural_patterns = self._detect_neural_patterns(text_vector)
                if neural_patterns:
                    logger.info(f"Detected {len(neural_patterns)} neural patterns")
                    patterns.extend(neural_patterns)
                else:
                    logger.warning("No neural patterns detected")
                
        if self.development_level >= 0.4:
            # Add semantic patterns
            semantic_patterns = self._detect_semantic_patterns(text, features)
            if semantic_patterns:
                logger.info(f"Detected {len(semantic_patterns)} semantic patterns")
                patterns.extend(semantic_patterns)
            else:
                logger.warning("No semantic patterns detected")
            
        if self.development_level >= 0.6:
            # Add syntactic patterns
            syntactic_patterns = self._detect_syntactic_patterns(text)
            if syntactic_patterns:
                logger.info(f"Detected {len(syntactic_patterns)} syntactic patterns")
                patterns.extend(syntactic_patterns)
            else:
                logger.warning("No syntactic patterns detected")
            
        if self.development_level >= 0.8:
            # Add contextual patterns that incorporate prior inputs
            if len(self.recent_inputs) > 1:
                contextual_patterns = self._detect_contextual_patterns(text, features)
                if contextual_patterns:
                    logger.info(f"Detected {len(contextual_patterns)} contextual patterns")
                    patterns.extend(contextual_patterns)
                else:
                    logger.warning("No contextual patterns detected")
        
        # Update pattern frequencies
        for pattern in patterns:
            pattern_id = pattern.get("pattern_id", "")
            if pattern_id:
                self.pattern_frequency[pattern_id] += 1
                pattern["frequency"] = self.pattern_frequency[pattern_id]
                
        # If no patterns detected, create a basic fallback pattern
        if not patterns and text:
            logger.warning(f"No patterns detected, creating fallback pattern for: {text[:30]}...")
            patterns.append(self._create_pattern(
                "unknown",
                confidence=0.5,
                attributes={
                    "text": text[:50] + ("..." if len(text) > 50 else ""),
                    "is_fallback": True
                }
            ))
        
        logger.info(f"Pattern recognition completed with {len(patterns)} total patterns detected")
        return patterns
    
    def _detect_token_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect patterns at the token level"""
        if not text:
            logger.warning("Empty text passed to token pattern detection")
            return []
            
        # Debug info
        logger.info(f"Token pattern detection for text: '{text[:30]}...'")
            
        patterns = []
        tokens = word_tokenize(text.lower())
        
        logger.info(f"Tokenized into {len(tokens)} tokens: {tokens[:5]}...")
        
        # Always create a token pattern if tokens exist
        if tokens:
            token_pattern = self._create_pattern(
                "token",
                confidence=0.9,
                attributes={
                    "tokens": tokens[:5],  # First 5 tokens
                    "token_count": len(tokens)
                }
            )
            patterns.append(token_pattern)
            logger.info("Created basic token pattern")
        
        # Check for repeated tokens
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if count > 1 and len(token) > 1:  # Avoid single characters
                repeated_token_pattern = self._create_pattern(
                    "token",
                    confidence=min(0.5 + count * 0.1, 0.9),
                    attributes={
                        "token": token,
                        "count": count,
                        "frequency": count / len(tokens) if tokens else 0
                    }
                )
                patterns.append(repeated_token_pattern)
                logger.info(f"Created repeated token pattern for '{token}' (count: {count})")
        
        # Check for unusual tokens (numbers, symbols, etc.)
        for token in tokens:
            # Numbers
            if token.isdigit() or (token.replace('.', '', 1).isdigit() and '.' in token):
                number_pattern = self._create_pattern(
                    "token",
                    confidence=0.8,
                    attributes={
                        "token": token,
                        "token_type": "number"
                    }
                )
                patterns.append(number_pattern)
                logger.info(f"Created number token pattern for '{token}'")
            
            # Special symbols
            elif any(c in token for c in "!@#$%^&*()_+-=[]{}|;':\",./<>?"):
                symbol_pattern = self._create_pattern(
                    "token",
                    confidence=0.7,
                    attributes={
                        "token": token,
                        "token_type": "symbol"
                    }
                )
                patterns.append(symbol_pattern)
                logger.info(f"Created symbol token pattern for '{token}'")
        
        logger.info(f"Token pattern detection completed with {len(patterns)} patterns")
        return patterns
    
    def _detect_ngram_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect n-gram patterns in text"""
        if not text:
            return []
            
        patterns = []
        
        # Tokenize text
        tokens = word_tokenize(text.lower())
        if len(tokens) < 2:
            return patterns
        
        # Always create a basic n-gram pattern if possible
        if len(tokens) >= 2:
            patterns.append(self._create_pattern(
                "n_gram",
                confidence=0.8,
                attributes={
                    "bigrams": [" ".join(bg) for bg in list(ngrams(tokens, 2))[:3]],  # First 3 bigrams
                    "count": len(list(ngrams(tokens, 2)))
                }
            ))
        
        # Generate n-grams
        bigrams = list(ngrams(tokens, 2))
        if len(tokens) >= 3:
            trigrams = list(ngrams(tokens, 3))
        else:
            trigrams = []
            
        # Check for repeated bigrams
        bigram_counts = Counter(bigrams)
        for bigram, count in bigram_counts.items():
            if count > 1:
                patterns.append(self._create_pattern(
                    "n_gram", 
                    confidence=min(0.6 + count * 0.1, 0.9),
                    attributes={
                        "n_gram": " ".join(bigram),
                        "n": 2,
                        "count": count
                    }
                ))
                
        # Check for repeated trigrams
        trigram_counts = Counter(trigrams)
        for trigram, count in trigram_counts.items():
            if count > 1:
                patterns.append(self._create_pattern(
                    "n_gram", 
                    confidence=min(0.7 + count * 0.1, 0.95),
                    attributes={
                        "n_gram": " ".join(trigram),
                        "n": 3,
                        "count": count
                    }
                ))
        
        return patterns
    
    def _detect_neural_patterns(self, text_vector: torch.Tensor) -> List[Dict[str, Any]]:
        """Use neural network to detect patterns in text vector"""
        patterns = []
        
        # Process through neural network
        with torch.no_grad():
            # Move to appropriate device
            text_vector = text_vector.to(self.device)
            
            # Get neural output
            output = self.neural_net.forward(text_vector)
            
            # Detect patterns
            neural_patterns, updated_known_patterns = self.neural_net.detect_patterns(
                text_vector, 
                self.known_patterns,
                activation_threshold=self.pattern_activation_threshold
            )
            
            # Update known patterns
            self.known_patterns = updated_known_patterns
            
            # Convert to pattern objects
            for np in neural_patterns:
                pattern_id = np.get("pattern_id", "")
                is_new = np.get("is_new", False)
                
                patterns.append(self._create_pattern(
                    "neural", 
                    confidence=np.get("activation", 0.5),
                    attributes={
                        "neural_id": pattern_id,
                        "novelty_score": np.get("novelty", 0.5),
                        "is_new_pattern": is_new
                    }
                ))
        
        return patterns
    
    def _detect_semantic_patterns(self, text: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect semantic patterns in text"""
        if not text or len(text.split()) < 3:
            return []
            
        patterns = []
        
        # Add basic semantic pattern for any text with sufficient length
        if len(text) > 10:
            patterns.append(self._create_pattern(
                "semantic",
                confidence=0.7,
                attributes={
                    "text_length": len(text),
                    "complexity": "low" if len(text) < 50 else "medium" if len(text) < 100 else "high"
                }
            ))
        
        # Add to document collection for TF-IDF
        if text not in self.documents:
            self.documents.append(text)
            
            # Rebuild vectorizer if we have enough documents
            if len(self.documents) > 1:
                try:
                    self.document_vectors = self.vectorizer.fit_transform(self.documents)
                except:
                    # If vectorizer fails, reset and continue
                    self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 3))
                    if len(self.documents) > 0:
                        self.document_vectors = self.vectorizer.fit_transform(self.documents)
                        
        # Check for common topics/themes
        # Basic approach using TF-IDF to extract important terms
        if len(self.documents) > 0 and hasattr(self.vectorizer, 'vocabulary_'):
            try:
                # Get the current document vector
                current_vector = self.vectorizer.transform([text])
                
                # Get the most significant terms
                feature_names = self.vectorizer.get_feature_names_out()
                
                # Find non-zero elements in the current vector
                nonzero_indices = current_vector.nonzero()[1]
                
                # Get the most important features
                important_features = [(feature_names[idx], current_vector[0, idx]) 
                                    for idx in nonzero_indices]
                
                # Sort by importance
                important_features.sort(key=lambda x: x[1], reverse=True)
                
                # Create patterns for the most important terms
                for term, importance in important_features[:5]:  # Top 5 terms
                    if importance > 0.1:  # Threshold
                        patterns.append(self._create_pattern(
                            "semantic", 
                            confidence=min(0.5 + float(importance), 0.9),
                            attributes={
                                "term": term,
                                "importance": float(importance),
                                "frequency": text.lower().count(term)
                            }
                        ))
            except Exception as e:
                logger.warning(f"Error in semantic pattern detection: {str(e)}")
                
        return patterns
    
    def _detect_syntactic_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect syntactic patterns in text"""
        patterns = []
        
        # Detect questions
        if '?' in text:
            patterns.append(self._create_pattern(
                "syntactic", 
                confidence=0.9,
                attributes={
                    "pattern_type": "question",
                    "count": text.count('?')
                }
            ))
            
        # Detect exclamations
        if '!' in text:
            patterns.append(self._create_pattern(
                "syntactic", 
                confidence=0.9,
                attributes={
                    "pattern_type": "exclamation",
                    "count": text.count('!')
                }
            ))
            
        # Detect common sentence structures
        if self.development_level >= 0.7:
            # Check for conditional statements
            if re.search(r'\bif\b.*\bthen\b', text, re.IGNORECASE) or 'if' in text.lower():
                patterns.append(self._create_pattern(
                    "syntactic", 
                    confidence=0.8,
                    attributes={
                        "pattern_type": "conditional"
                    }
                ))
                
            # Check for comparisons
            if re.search(r'\bmore\b.*\bthan\b|\bless\b.*\bthan\b|\bbetter\b.*\bthan\b', text, re.IGNORECASE):
                patterns.append(self._create_pattern(
                    "syntactic", 
                    confidence=0.8,
                    attributes={
                        "pattern_type": "comparison"
                    }
                ))
                
            # Check for negations
            if re.search(r'\bnot\b|\bno\b|\bnever\b', text, re.IGNORECASE):
                patterns.append(self._create_pattern(
                    "syntactic", 
                    confidence=0.8,
                    attributes={
                        "pattern_type": "negation"
                    }
                ))
        
        return patterns
    
    def _detect_contextual_patterns(self, text: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns that depend on context from recent inputs"""
        patterns = []
        
        if len(self.recent_inputs) < 2:
            return patterns
            
        # Get previous input
        prev_input = self.recent_inputs[-2]
        prev_text = prev_input.get("text", "")
        
        # Check for repetition patterns
        if text == prev_text:
            patterns.append(self._create_pattern(
                "contextual", 
                confidence=0.9,
                attributes={
                    "pattern_type": "repetition",
                    "repeated_text": text[:50] + ("..." if len(text) > 50 else "")
                }
            ))
            
        # Check for similarity
        elif prev_text and self._simple_similarity(text, prev_text) > 0.7:
            patterns.append(self._create_pattern(
                "contextual", 
                confidence=0.8,
                attributes={
                    "pattern_type": "similar_to_previous",
                    "similarity": self._simple_similarity(text, prev_text)
                }
            ))
            
        # Check for question-answer patterns
        if prev_text and prev_text.strip().endswith('?') and not text.strip().endswith('?'):
            patterns.append(self._create_pattern(
                "contextual", 
                confidence=0.9,
                attributes={
                    "pattern_type": "answer_to_question",
                    "question": prev_text[:50] + ("..." if len(prev_text) > 50 else "")
                }
            ))
            
        return patterns
    
    def _process_temporal_sequence(self) -> Dict[str, Any]:
        """Process the temporal sequence of patterns"""
        if len(self.temporal_sequence) < 3:
            return {}
            
        # Convert sequence to tensor
        sequence_tensor = torch.stack(list(self.temporal_sequence), dim=0).unsqueeze(0)
        
        # Process through temporal network
        with torch.no_grad():
            sequence_tensor = sequence_tensor.to(self.device)
            result = self.temporal_net.forward(sequence_tensor)
            
        # Extract results
        return {
            "temporal_pattern": result["temporal_pattern"].cpu().numpy().tolist(),
            "next_prediction": result["next_prediction"].cpu().numpy().tolist(),
            "temporal_features": {
                "sequence_length": len(self.temporal_sequence),
                "has_pattern": result["temporal_pattern"].norm().item() > 0.5
            }
        }
        
    def _text_to_vector(self, text: str) -> torch.Tensor:
        """
        Convert text to a vector representation for neural processing
        
        If LLM embeddings are available, uses them for a more sophisticated
        representation. Otherwise falls back to a simple feature-based encoding.
        """
        # Use LLM client for embeddings if available
        if self.use_embeddings:
            try:
                # Get embeddings from LLM client
                embedding = self.llm_client.get_embedding(text)
                
                # Check if embedding dimensions match what we need
                if isinstance(embedding, list) and len(embedding) > 0:
                    # Check if embedding dimensions match what we need
                    if len(embedding) > 64:
                        # Truncate if too large
                        embedding = embedding[:64]
                    elif len(embedding) < 64:
                        # Pad if too small
                        embedding = embedding + [0.0] * (64 - len(embedding))
                    
                    # Convert to tensor
                    return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                else:
                    logger.warning(f"Received invalid embedding format: {type(embedding)}")
                    # Continue to fallback
            except Exception as e:
                logger.warning(f"Failed to get embeddings from LLM client: {e}")
                logger.warning("Falling back to simple vector encoding")
        
        # Fallback: More robust bag-of-words approach
        vector = np.zeros(64)  # Match input_dim of neural net
        
        # Tokenize text
        tokens = word_tokenize(text.lower()) if text else []
        
        # Add token-based features
        for i, token in enumerate(tokens[:20]):  # Use up to 20 tokens
            # Use a hash function to distribute tokens across the vector
            idx = hash(token) % 32
            vector[idx] += 1.0  # Count occurrences
            
            # Add character-level information
            for char in token[:5]:  # First 5 chars of each token
                vector[(hash(char) % 16) + 32] += 0.1  # Use second half of vector for chars
            
        # Add basic text statistics
        if text:
            vector[48] = min(len(text) / 1000, 1.0)  # Text length (normalized)
            vector[49] = min(len(tokens) / 100, 1.0)  # Word count (normalized)
            vector[50] = min(text.count('.') / 10, 1.0)  # Sentence count approx
            vector[51] = min(text.count('?') / 5, 1.0)  # Question mark count
            vector[52] = min(text.count('!') / 5, 1.0)  # Exclamation count
            vector[53] = min(sum(1 for c in text if c.isupper()) / 20, 1.0)  # Uppercase count
            vector[54] = min(sum(1 for c in text if c.isdigit()) / 20, 1.0)  # Digit count
            
            # Add n-gram presence
            if len(tokens) >= 2:
                vector[55] = 1.0  # Has bigrams
            if len(tokens) >= 3:
                vector[56] = 1.0  # Has trigrams
                
            # Average word length
            vector[57] = min(np.mean([len(w) for w in tokens]) / 10 if tokens else 0, 1.0)
            
            # Set the remaining elements to ensure the vector is non-zero
            vector[58:64] = np.random.rand(6) * 0.1  # Small random values
            
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    
    def _create_pattern_feature_vector(self, patterns: List[Dict[str, Any]]) -> torch.Tensor:
        """Create a feature vector from patterns for temporal sequence processing"""
        vector = np.zeros(32)  # Match input_dim of temporal net
        
        # Count pattern types
        pattern_types = Counter([p.get("pattern_type", "") for p in patterns])
        
        # Set vector values based on pattern information
        for i, (pattern_type, count) in enumerate(pattern_types.items()):
            idx = hash(pattern_type) % 16  # Use hash to distribute the pattern types
            vector[idx] += count / 5  # Scale the count
            
        # Add pattern count
        vector[16] = len(patterns) / 10
        
        # Add average confidence
        vector[17] = np.mean([p.get("confidence", 0.5) for p in patterns]) if patterns else 0.5
        
        # Add average novelty
        vector[18] = np.mean([p.get("attributes", {}).get("novelty_score", 0.5) 
                           for p in patterns]) if patterns else 0.5
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate a simple similarity score between two texts"""
        if not text1 or not text2:
            return 0.0
            
        # Convert to lowercase and tokenize
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
        
    def _create_pattern(self, pattern_type: str, confidence: float = 1.0, 
                       attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a pattern object with the provided attributes
        
        Args:
            pattern_type: Type of pattern
            confidence: Confidence level
            attributes: Additional pattern attributes
            
        Returns:
            Dictionary representing the pattern
        """
        # Generate a unique ID for the pattern
        pattern_id = f"{pattern_type}_{uuid.uuid4().hex[:8]}"
        
        # Create pattern object
        pattern = {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
            "developmental_level": self.development_level
        }
        
        # Add additional information based on pattern type
        if pattern_type == "token":
            pattern["novelty"] = 0.3  # Token patterns are common
        elif pattern_type == "n_gram":
            pattern["novelty"] = 0.5  # N-gram patterns have medium novelty
        elif pattern_type == "semantic":
            pattern["novelty"] = 0.7  # Semantic patterns are more novel
        elif pattern_type == "neural":
            pattern["novelty"] = attributes.get("novelty_score", 0.5) if attributes else 0.5
        elif pattern_type == "temporal":
            pattern["novelty"] = 0.8  # Temporal patterns are quite novel
        else:
            pattern["novelty"] = 0.5  # Default novelty
            
        return pattern
        
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        # Handle specific message types as needed
        pass
        
    def update_development(self, amount: float) -> float:
        """
        Update the module's developmental level
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update neural network development level
        self.neural_net.update_developmental_level(new_level)
        self.temporal_net.update_developmental_level(new_level)
        
        # Adjust pattern recognition parameters based on development
        self.token_sensitivity = 0.7 + new_level * 0.2  # Starts higher and increases less
        self.ngram_sensitivity = 0.5 + new_level * 0.3  # Starts higher and increases less
        self.semantic_sensitivity = 0.4 + new_level * 0.4  # Starts higher
        
        # Refine thresholds as development progresses
        self.pattern_activation_threshold = max(0.05, 0.2 - new_level * 0.15)  # Lower threshold
        
        # Log development progress
        logger.info(f"Pattern recognizer {self.module_id} developmental level updated to {new_level:.2f}")
        
        return new_level
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing module state
        """
        base_state = super().get_state()
        
        # Add pattern recognizer specific state
        state = {
            **base_state,
            "known_pattern_count": len(self.known_patterns),
            "neural_net_state": self.neural_net.get_state(),
            "temporal_net_state": self.temporal_net.get_state(),
            "recent_input_count": len(self.recent_inputs),
            "token_sensitivity": self.token_sensitivity,
            "ngram_sensitivity": self.ngram_sensitivity,
            "semantic_sensitivity": self.semantic_sensitivity,
            "pattern_activation_threshold": self.pattern_activation_threshold,
            "use_embeddings": self.use_embeddings
        }
        
        return state
        
    def get_recent_patterns(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recently detected patterns
        
        Args:
            count: Maximum number of patterns to return
            
        Returns:
            List of recent patterns
        """
        recent_patterns = []
        
        # Extract patterns from recent inputs
        for input_data in reversed(list(self.recent_inputs)):
            if "patterns" in input_data:
                recent_patterns.extend(input_data["patterns"])
                
            if len(recent_patterns) >= count:
                break
                
        return recent_patterns[:count] 
