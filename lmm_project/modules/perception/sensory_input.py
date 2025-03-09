"""
Sensory Input Processing Module

This module is responsible for processing raw text input into a form
suitable for further processing by the perception system. It serves
as the primary sensory interface for the LMM, converting text into
meaningful sensory representations.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import uuid
import logging
from datetime import datetime
from pydantic import ValidationError

# Add tokenization libraries
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer, WordPunctTokenizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.util import ngrams

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.perception.models import SensoryInput, PerceptionResult, Pattern
from lmm_project.utils.vector_store import get_embedding

logger = logging.getLogger(__name__)

class SensoryInputProcessor(BaseModule):
    """
    Processes raw text input into a form suitable for perception
    
    This module is the first stage of perception, converting raw text
    into feature vectors and preliminary sensory representations.
    """
    # Configuration
    embedding_dimension: int = 768  # Updated to match actual embedding size
    max_input_history: int = 20
    tokenization_level: str = "basic"  # Options: primitive, basic, advanced, sophisticated
    feature_cache: Dict[str, np.ndarray] = {}
    
    # Tokenizers for different developmental levels
    tokenizers: Dict[str, Any] = {}
    
    # Input history
    input_history: List[SensoryInput] = []
    
    # Device for tensor operations
    device = None
    
    # Dimension reduction layer (for compatibility with pattern recognition)
    dim_reduction = None
    
    def __init__(
        self, 
        module_id: str,
        event_bus: Optional[EventBus] = None,
        developmental_level: float = 0.0,
        **kwargs
    ):
        """Initialize the sensory input processor"""
        super().__init__(
            module_id=module_id,
            module_type="sensory_input_processor",
            event_bus=event_bus,
            development_level=developmental_level,
            **kwargs
        )
        
        # Initialize device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SensoryInputProcessor using device: {self.device}")
        
        # Initialize dimension reduction layer from 768 to 384
        # This makes our embeddings compatible with pattern recognition
        self.dim_reduction = torch.nn.Linear(768, 384).to(self.device)
        # Initialize with simple weights that preserve some information
        with torch.no_grad():
            # Initialize to average pairs of dimensions
            for i in range(384):
                self.dim_reduction.weight[i, i*2] = 0.7
                self.dim_reduction.weight[i, i*2+1] = 0.7
        
        # Initialize tokenizers
        self._initialize_tokenizers()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("raw_text_input", self._handle_raw_text)
            
        # Set development-appropriate tokenization
        self._adjust_tokenization_for_development()
    
    def _initialize_tokenizers(self):
        """Initialize the different tokenizers for various developmental levels"""
        # Primitive tokenizer (character-level)
        self.tokenizers["primitive"] = lambda text: list(text)
        
        # Basic tokenizer (simple space and punctuation splitting)
        self.tokenizers["basic"] = RegexpTokenizer(r'\w+|[^\w\s]').tokenize
        
        # Advanced tokenizer (NLTK's word_tokenize with better handling of contractions)
        self.tokenizers["advanced"] = word_tokenize
        
        # Sophisticated tokenizer (handles social media text, emoticons, etc.)
        tweet_tokenizer = TweetTokenizer(preserve_case=False)
        self.tokenizers["sophisticated"] = tweet_tokenizer.tokenize
            
    def _adjust_tokenization_for_development(self):
        """Adjust tokenization approach based on developmental level"""
        if self.development_level < 0.2:
            # Very basic tokenization for earliest stages (character level)
            self.tokenization_level = "primitive"
        elif self.development_level < 0.5:
            # Simple tokenization (word level without linguistic knowledge)
            self.tokenization_level = "basic"
        elif self.development_level < 0.8:
            # More advanced tokenization with linguistic knowledge
            self.tokenization_level = "advanced"
        else:
            # Sophisticated tokenization for advanced development
            self.tokenization_level = "sophisticated"
            
        logger.debug(f"Sensory tokenization set to: {self.tokenization_level}")
        
    def update_development(self, amount: float) -> float:
        """Update the module's developmental level"""
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Check if we need to update tokenization approach
        if abs(self.development_level - prev_level) > 0.1:
            self._adjust_tokenization_for_development()
            
        return self.development_level
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text based on current developmental level
        
        The tokenization approach evolves with development:
        - Primitive: Character-level tokens (early stage)
        - Basic: Simple word splitting
        - Advanced: More sophisticated word and sub-word tokenization
        - Sophisticated: Advanced text handling including emoticons, slang, etc.
        """
        if not text:
            return []
            
        # Use the appropriate tokenizer based on developmental level
        tokenizer = self.tokenizers.get(self.tokenization_level, self.tokenizers["basic"])
        tokens = tokenizer(text.lower())
        
        # Add n-grams for more advanced developmental levels
        if self.tokenization_level in ["advanced", "sophisticated"] and len(tokens) > 1:
            # Add bigrams
            bigrams_list = list(ngrams(tokens, 2))
            bigram_tokens = [f"{t[0]}_{t[1]}" for t in bigrams_list]
            
            # For sophisticated level, also add trigrams if we have enough tokens
            if self.tokenization_level == "sophisticated" and len(tokens) > 2:
                trigrams_list = list(ngrams(tokens, 3))
                trigram_tokens = [f"{t[0]}_{t[1]}_{t[2]}" for t in trigrams_list]
                tokens = tokens + bigram_tokens + trigram_tokens
            else:
                tokens = tokens + bigram_tokens
                
        # For primitive and basic levels, supplement with character-level insights
        if self.tokenization_level in ["primitive", "basic"] and self.development_level > 0.1:
            # Add some character-level tokens for longer words
            for word in tokens[:]:  # Iterate over a copy to avoid modifying during iteration
                if len(word) > 4:
                    # Add first and last characters as tokens
                    tokens.append(word[0])  # First character
                    tokens.append(word[-1])  # Last character
                    
        return tokens
    
    def _extract_linguistic_features(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract linguistic features based on developmental level
        
        As development progresses, more sophisticated features are extracted
        """
        features = {}
        
        # Basic features available at all levels
        features["text_length"] = len(text)
        features["token_count"] = len(tokens)
        features["avg_token_length"] = sum(len(t) for t in tokens) / max(1, len(tokens))
        
        # More advanced features for higher developmental levels
        if self.development_level >= 0.3:
            # Add punctuation features
            features["punctuation_ratio"] = sum(1 for c in text if c in '.,;:!?') / max(1, len(text))
            features["question_mark_present"] = 1.0 if '?' in text else 0.0
            features["exclamation_mark_present"] = 1.0 if '!' in text else 0.0
            
        # Even more features for advanced development
        if self.development_level >= 0.6:
            # Add capitalization features
            features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(1, len(text))
            features["starts_uppercase"] = 1.0 if text and text[0].isupper() else 0.0
            
            # Add token diversity
            unique_tokens = set(tokens)
            features["token_diversity"] = len(unique_tokens) / max(1, len(tokens))
            
        # Sophisticated features for highly developed perception
        if self.development_level >= 0.8:
            # Add character n-gram features
            char_bigrams = [''.join(bg) for bg in ngrams(text.lower(), 2)] if len(text) > 1 else []
            features["char_bigram_count"] = len(char_bigrams)
            
            # Add repetition detection
            if tokens:
                repeats = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
                features["repetition_ratio"] = repeats / (len(tokens) - 1) if len(tokens) > 1 else 0
                
        return features
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text
        
        Uses vector_store utility to get embeddings for text.
        Implements caching for efficiency.
        """
        # Check cache first
        cache_key = f"{text[:50]}_{len(text)}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
            
        try:
            # Get embedding from vector store utility
            # In early development stages, add noise to the embedding
            embedding = get_embedding(text)
            
            # Log embedding shape for debugging
            if not hasattr(self, '_logged_embedding_shape'):
                logger.info(f"Text embedding shape: {embedding.shape}")
                self._logged_embedding_shape = True
            
            # Add developmental noise (less precise in early stages)
            if self.development_level < 0.5:
                noise_level = 0.2 * (1 - self.development_level * 2)
                noise = np.random.normal(0, noise_level, size=embedding.shape)
                embedding = embedding + noise
                # Re-normalize
                embedding = embedding / np.linalg.norm(embedding)
                
            # Cache the result
            self.feature_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {e}")
            # Return a random embedding as fallback
            return np.random.randn(self.embedding_dimension)
    
    def _reduce_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reduce embedding dimension from 768 to 384 for compatibility with pattern recognition
        """
        # Convert to tensor, move to device, and process
        with torch.no_grad():
            tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            reduced = self.dim_reduction(tensor).cpu().numpy()
            return reduced
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text input
        
        Returns a dictionary of features including:
        - tokens: List of tokens from the text
        - embedding: Vector representation of the text
        - length: Length of the input
        - linguistic_features: Extracted linguistic features
        """
        if not text:
            # Create a zero embedding and its reduced version
            empty_embedding = np.zeros(self.embedding_dimension)
            empty_reduced = np.zeros(384)  # The reduced dimension is 384
            return {
                "tokens": [],
                "embedding": empty_embedding,
                "reduced_embedding": empty_reduced,  # Add reduced embedding for empty input
                "length": 0,
                "linguistic_features": {}
            }
        
        # Get tokens
        tokens = self._tokenize_text(text)
        
        # Get embedding
        embedding = self._get_text_embedding(text)
        
        # Get reduced embedding for pattern recognition
        reduced_embedding = self._reduce_embedding_dimension(embedding)
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text, tokens)
        
        return {
            "tokens": tokens,
            "embedding": embedding,
            "reduced_embedding": reduced_embedding,  # Add reduced embedding
            "length": len(text),
            "linguistic_features": linguistic_features
        }
    
    def _record_input(self, sensory_input: SensoryInput):
        """Record input in history, maintaining max size"""
        self.input_history.append(sensory_input)
        
        # Maintain maximum history size
        if len(self.input_history) > self.max_input_history:
            self.input_history = self.input_history[-self.max_input_history:]
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw input data into sensory representation
        
        Args:
            input_data: Dictionary containing the raw input data
                Must include 'text' key with the input text
                
        Returns:
            Dictionary with the processed sensory information
        """
        # Extract text from input
        if "text" not in input_data:
            logger.warning("No text found in input data")
            return {"error": "No text input provided", "status": "error"}
        
        text = input_data["text"]
        source = input_data.get("source", "mother")
        context = input_data.get("context", {})
        
        # Create unique ID for this input
        input_id = f"input_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create SensoryInput model
            sensory_input = SensoryInput(
                input_id=input_id,
                text=text,
                source=source,
                context=context,
                metadata=input_data.get("metadata", {})
            )
            
            # Record this input
            self._record_input(sensory_input)
            
            # Extract features
            features = self._extract_features(text)
            
            # Verify that reduced_embedding exists in features
            if "reduced_embedding" not in features:
                logger.error(f"Missing reduced_embedding in features: {list(features.keys())}")
                return {
                    "status": "error",
                    "error": "Missing reduced_embedding in features",
                    "available_keys": list(features.keys())
                }
            
            # Create embedding tensor for neural network processing using the REDUCED embedding
            # Ensure tensor is on the correct device
            try:
                embedding_tensor = torch.tensor(features["reduced_embedding"], dtype=torch.float32).to(self.device).unsqueeze(0)
            except Exception as e:
                logger.error(f"Error creating embedding tensor: {e}")
                return {
                    "status": "error",
                    "error": f"Failed to create embedding tensor: {str(e)}",
                    "reduced_embedding_type": str(type(features["reduced_embedding"])),
                    "reduced_embedding_shape": str(features["reduced_embedding"].shape if hasattr(features["reduced_embedding"], 'shape') else "unknown")
                }
            
            # Create the result
            result = {
                "status": "success",
                "input_id": input_id,
                "features": features,
                "embedding_tensor": embedding_tensor,
                "sensory_input": sensory_input.model_dump(),
                "developmental_level": self.development_level,
                "tokenization_level": self.tokenization_level,
                "device": str(self.device),
                "embedding_dimensions": {
                    "original": len(features["embedding"]),
                    "reduced": len(features["reduced_embedding"])
                }
            }
            
            # Publish the processed sensory input event
            if self.event_bus:
                self.publish_message(
                    message_type="sensory_input_processed",
                    content=result
                )
            
            return result
            
        except ValidationError as e:
            logger.error(f"Invalid sensory input: {e}")
            return {"error": f"Invalid input data: {str(e)}", "status": "error"}
        except Exception as e:
            logger.error(f"Error processing sensory input: {e}")
            return {"error": f"Failed to process input: {str(e)}", "status": "error"}
            
    def _handle_raw_text(self, message: Message):
        """Handle raw text input event"""
        if not message.content or "text" not in message.content:
            return
            
        # Process the input
        self.process_input(message.content)
        
    def get_recent_inputs(self, count: int = 5) -> List[SensoryInput]:
        """Get the most recent inputs"""
        return self.input_history[-count:] if self.input_history else []
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        state.update({
            "tokenization_level": self.tokenization_level,
            "input_history_count": len(self.input_history),
            "feature_cache_size": len(self.feature_cache),
            "supported_tokenizers": list(self.tokenizers.keys()),
            "device": str(self.device),
            "embedding_dimension": self.embedding_dimension,
            "reduced_dimension": 384
        })
        return state 
