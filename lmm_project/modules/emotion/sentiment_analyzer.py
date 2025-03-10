import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import re
import numpy as np
from collections import deque, Counter

# Use TextBlob for basic sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Basic sentiment analysis will be used.")

# Use NLTK for more advanced NLP if available
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    # Download necessary NLTK resources if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Advanced NLP features will be limited.")

# PyTorch for neural sentiment analysis
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Neural sentiment analysis will be disabled.")

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.emotion.models import SentimentAnalysis, EmotionNeuralState
from lmm_project.utils.llm_client import LLMClient, Message as LLMMessage

# Initialize logger
logger = logging.getLogger(__name__)

class SentimentNN(nn.Module):
    """Simple neural network for sentiment analysis"""
    def __init__(self, vocab_size=5000, embedding_dim=50, hidden_dim=100, output_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # Simple averaging of embeddings
        embedded = embedded.mean(dim=1)
        x = F.relu(self.fc1(embedded))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class SentimentAnalyzer(BaseModule):
    """
    Sentiment analyzer for detecting emotional tone in text
    
    This module develops from basic positive/negative detection to
    sophisticated emotional tone analysis with contextual understanding.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic sentiment detection",
        0.2: "Positive/negative classification",
        0.4: "Multi-class emotion detection",
        0.6: "Contextual sentiment analysis",
        0.8: "Emotion intensity detection",
        1.0: "Sophisticated sentiment understanding"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the sentiment analyzer
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="sentiment_analyzer",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize emotion lexicons
        self._initialize_emotion_lexicons()
        
        # Parameters that vary with development
        self.params = {
            "basic_weight": 0.8,      # Weight for basic sentiment analysis
            "advanced_weight": 0.2,    # Weight for advanced analysis
            "context_weight": 0.1,     # Weight for contextual factors
            "intensity_threshold": 0.3, # Threshold for emotion detection
            "development_factor": development_level
        }
        
        # Neural state for tracking activations and development
        self.neural_state = EmotionNeuralState()
        self.neural_state.sentiment_development = development_level
        
        # Adjust parameters based on development level
        self._adjust_parameters_for_development()
        
        # History of recent analyses
        self.analysis_history = deque(maxlen=50)
        
        # Initialize neural sentiment analyzer if PyTorch is available
        self.neural_analyzer = None
        self.vocab = {}
        
        if TORCH_AVAILABLE and development_level >= 0.4:
            self._initialize_neural_analyzer()
            
        # Try to initialize LLM client for advanced analysis
        self.llm_client = None
        if development_level >= 0.6:
            try:
                self.llm_client = LLMClient()
                logger.info("LLM client initialized for advanced sentiment analysis")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
        
        logger.info(f"Sentiment analyzer initialized at development level {development_level:.2f}")
        
    def _initialize_emotion_lexicons(self):
        """Initialize lexicons for emotion detection"""
        # Basic positive and negative word lists
        self.positive_words = {
            "good", "great", "excellent", "wonderful", "amazing", "fantastic",
            "terrific", "outstanding", "superb", "brilliant", "awesome",
            "happy", "joy", "delighted", "pleased", "glad", "satisfied",
            "love", "adore", "like", "enjoy", "appreciate", "admire",
            "beautiful", "lovely", "pleasant", "nice", "perfect"
        }
        
        self.negative_words = {
            "bad", "terrible", "horrible", "awful", "dreadful", "poor",
            "sad", "unhappy", "depressed", "miserable", "gloomy", "disappointed",
            "angry", "mad", "furious", "upset", "annoyed", "irritated",
            "hate", "dislike", "despise", "detest", "loathe", "abhor",
            "ugly", "unpleasant", "nasty", "disgusting", "offensive"
        }
        
        # Emotion-specific lexicons
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
            "trust": {
                "trust", "confident", "secure", "faithful", "reliable", "dependable",
                "trustworthy", "honest", "loyal", "sincere", "devoted",
                "authentic", "genuine", "believing", "convinced", "assured"
            },
            "anticipation": {
                "anticipate", "expect", "await", "look forward", "hope", "excited",
                "eager", "enthusiastic", "keen", "prepared", "ready",
                "watchful", "vigilant", "alert", "attentive", "mindful"
            }
        }
        
        # Intensity modifiers
        self.intensifiers = {
            "very", "extremely", "incredibly", "exceptionally", "tremendously",
            "absolutely", "completely", "totally", "utterly", "highly", 
            "deeply", "profoundly", "intensely", "remarkably", "seriously"
        }
        
        self.diminishers = {
            "slightly", "somewhat", "a bit", "a little", "fairly",
            "rather", "kind of", "sort of", "moderately", "relatively",
            "barely", "hardly", "scarcely", "faintly", "mildly"
        }
        
    def _initialize_neural_analyzer(self):
        """Initialize neural sentiment analyzer"""
        if not TORCH_AVAILABLE:
            return
            
        # Very simple vocabulary for demo purposes
        # In a real implementation, this would be trained on a corpus
        words = list(self.positive_words | self.negative_words)
        for emotion_words in self.emotion_lexicons.values():
            words.extend(emotion_words)
            
        # Add common words
        common_words = [
            "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "why", "who", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should", "can", "could",
            "may", "might", "must", "to", "for", "of", "in", "on", "at", "by", "with"
        ]
        words.extend(common_words)
        
        # Create vocabulary
        words = list(set(words))[:5000]  # Limit vocabulary size
        self.vocab = {word: i for i, word in enumerate(words)}
        
        # Create neural network
        self.neural_analyzer = SentimentNN(
            vocab_size=len(self.vocab) + 1,  # +1 for unknown words
            embedding_dim=50,
            hidden_dim=100,
            output_dim=3  # Positive, Negative, Neutral
        )
        
        # Try to use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.neural_analyzer.to(self.device)
        logger.info(f"Neural sentiment analyzer initialized with {len(self.vocab)} words vocabulary")
        
    def _adjust_parameters_for_development(self):
        """Adjust parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very basic processing - simple positive/negative
            self.params.update({
                "basic_weight": 1.0,
                "advanced_weight": 0.0,
                "context_weight": 0.0,
                "intensity_threshold": 0.4,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.4:
            # Developing multi-class classification
            self.params.update({
                "basic_weight": 0.8,
                "advanced_weight": 0.2,
                "context_weight": 0.1,
                "intensity_threshold": 0.35,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.6:
            # Developing contextual understanding
            self.params.update({
                "basic_weight": 0.6,
                "advanced_weight": 0.4,
                "context_weight": 0.2,
                "intensity_threshold": 0.3,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.8:
            # Developing intensity detection
            self.params.update({
                "basic_weight": 0.4,
                "advanced_weight": 0.6,
                "context_weight": 0.3,
                "intensity_threshold": 0.25,
                "development_factor": self.development_level
            })
        else:
            # Sophisticated analysis
            self.params.update({
                "basic_weight": 0.2,
                "advanced_weight": 0.8,
                "context_weight": 0.4,
                "intensity_threshold": 0.2,
                "development_factor": self.development_level
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to analyze sentiment
        
        Args:
            input_data: Input data to process
                Required keys: 'text' or 'content'
                Optional keys: 'context'
                
        Returns:
            Dictionary with sentiment analysis results
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract text from input
        text = ""
        if "text" in input_data:
            text = input_data["text"]
        elif "content" in input_data:
            content = input_data["content"]
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict) and "text" in content:
                text = content["text"]
            
        if not text:
            return {
                "status": "error",
                "message": "No text provided for sentiment analysis",
                "process_id": process_id
            }
        
        # Process text to analyze sentiment
        context = input_data.get("context", {})
        analysis = self._analyze_sentiment(text, context)
        
        # Create SentimentAnalysis object
        sentiment_analysis = SentimentAnalysis(
            text=text,
            positive_score=analysis["positive_score"],
            negative_score=analysis["negative_score"],
            neutral_score=analysis["neutral_score"],
            compound_score=analysis["compound_score"],
            detected_emotions=analysis["detected_emotions"],
            highlighted_phrases=analysis["highlighted_phrases"],
            process_id=process_id,
            confidence=analysis["confidence"]
        )
        
        # Add to history
        self.analysis_history.append(sentiment_analysis)
        
        # Return analysis results
        result = {
            "status": "success",
            "analysis": sentiment_analysis.dict(),
            "process_id": process_id,
            "development_level": self.development_level
        }
        
        # Publish result if we have event bus
        if self.event_bus:
            self.publish_message(
                "sentiment_analysis_result",
                {
                    "analysis": sentiment_analysis.dict(),
                    "process_id": process_id
                }
            )
        
        return result
    
    def _analyze_sentiment(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment in text
        
        Args:
            text: Text to analyze
            context: Contextual information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Methods to use based on development level
        methods = []
        
        # Always use lexical approach
        methods.append(self._lexical_sentiment_analysis)
        
        # Add more sophisticated methods with development
        if TEXTBLOB_AVAILABLE and self.development_level >= 0.2:
            methods.append(self._textblob_sentiment_analysis)
            
        if TORCH_AVAILABLE and self.neural_analyzer and self.development_level >= 0.4:
            methods.append(self._neural_sentiment_analysis)
            
        if self.llm_client and self.development_level >= 0.6:
            methods.append(self._llm_sentiment_analysis)
            
        # Process with each method and combine results
        results = []
        method_names = []
        
        for method in methods:
            try:
                result = method(text, context)
                results.append(result)
                method_names.append(method.__name__.replace("_sentiment_analysis", ""))
            except Exception as e:
                logger.error(f"Error in sentiment analysis method {method.__name__}: {str(e)}")
        
        # Initialize combined results
        if not results:
            # Fallback to basic neutral sentiment
            return {
                "positive_score": 0.33,
                "negative_score": 0.33,
                "neutral_score": 0.34,
                "compound_score": 0.0,
                "detected_emotions": {},
                "highlighted_phrases": [],
                "confidence": 0.1,
                "method": "fallback"
            }
            
        # Calculate weights based on development and method sophistication
        weights = []
        if len(results) == 1:
            weights = [1.0]
        elif len(results) == 2:
            weights = [self.params["basic_weight"], self.params["advanced_weight"]]
        elif len(results) == 3:
            weights = [0.2, 0.3, 0.5]
        elif len(results) == 4:
            weights = [0.1, 0.2, 0.3, 0.4]
        else:
            weights = [1.0 / len(results)] * len(results)
            
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combine results
        combined = {
            "positive_score": sum(w * r["positive_score"] for w, r in zip(weights, results)),
            "negative_score": sum(w * r["negative_score"] for w, r in zip(weights, results)),
            "neutral_score": sum(w * r["neutral_score"] for w, r in zip(weights, results)),
            "compound_score": sum(w * r["compound_score"] for w, r in zip(weights, results)),
            "detected_emotions": self._combine_emotions([r["detected_emotions"] for r in results], weights),
            "highlighted_phrases": self._combine_phrases([r.get("highlighted_phrases", []) for r in results]),
            "confidence": min(0.2 + self.development_level, 
                             sum(w * r.get("confidence", 0.5) for w, r in zip(weights, results))),
            "method": "+".join(method_names)
        }
        
        return combined
    
    def _combine_emotions(self, emotion_dicts: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
        """
        Combine emotion dictionaries from multiple methods
        
        Args:
            emotion_dicts: List of emotion dictionaries
            weights: Weights for each dictionary
            
        Returns:
            Combined emotion dictionary
        """
        # Initialize combined dictionary
        combined = {}
        
        # Combine all emotions
        for emotion_dict, weight in zip(emotion_dicts, weights):
            for emotion, score in emotion_dict.items():
                if emotion in combined:
                    combined[emotion] += score * weight
                else:
                    combined[emotion] = score * weight
        
        # Filter low-confidence emotions
        threshold = self.params["intensity_threshold"]
        combined = {k: v for k, v in combined.items() if v >= threshold}
        
        return combined
    
    def _combine_phrases(self, phrase_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Combine highlighted phrases from multiple methods
        
        Args:
            phrase_lists: List of phrase lists
            
        Returns:
            Combined phrase list
        """
        # Flatten and deduplicate phrases
        all_phrases = []
        seen_phrases = set()
        
        for phrase_list in phrase_lists:
            for phrase in phrase_list:
                text = phrase.get("text", "")
                if text and text not in seen_phrases:
                    all_phrases.append(phrase)
                    seen_phrases.add(text)
        
        # Sort by score and limit to top 10
        all_phrases.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
        return all_phrases[:10]
    
    def _lexical_sentiment_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic lexical sentiment analysis
        
        Args:
            text: Text to analyze
            context: Contextual information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Tokenize text
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Count positive and negative words
        pos_words = [token for token in tokens if token in self.positive_words]
        neg_words = [token for token in tokens if token in self.negative_words]
        
        pos_count = len(pos_words)
        neg_count = len(neg_words)
        total_words = len(tokens)
        
        # Calculate scores
        if total_words == 0:
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
            compound_score = 0.0
        else:
            positive_score = pos_count / total_words
            negative_score = neg_count / total_words
            neutral_score = 1.0 - positive_score - negative_score
            
            # Ensure neutral score is not negative
            neutral_score = max(0.0, neutral_score)
            
            # Normalize scores
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
                
            # Calculate compound score (-1 to 1)
            if pos_count == 0 and neg_count == 0:
                compound_score = 0.0
            else:
                compound_score = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Detect emotions
        detected_emotions = {}
        for emotion, word_set in self.emotion_lexicons.items():
            emotion_words = [token for token in tokens if token in word_set]
            if emotion_words:
                detected_emotions[emotion] = len(emotion_words) / total_words if total_words > 0 else 0.0
        
        # Highlight key phrases
        highlighted_phrases = []
        
        # For simple implementation, just highlight individual emotional words
        emotional_words = pos_words + neg_words
        for word in emotional_words[:5]:
            score = 1.0 if word in pos_words else -1.0
            highlighted_phrases.append({
                "text": word,
                "score": score,
                "index": text.lower().find(word)
            })
        
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "compound_score": compound_score,
            "detected_emotions": detected_emotions,
            "highlighted_phrases": highlighted_phrases,
            "confidence": 0.3,
            "method": "lexical"
        }
    
    def _textblob_sentiment_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        TextBlob-based sentiment analysis
        
        Args:
            text: Text to analyze
            context: Contextual information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not TEXTBLOB_AVAILABLE:
            raise ImportError("TextBlob is not available")
            
        # Analyze with TextBlob
        blob = TextBlob(text)
        
        # TextBlob polarity is -1 to 1
        polarity = blob.sentiment.polarity
        
        # TextBlob subjectivity is 0 to 1
        subjectivity = blob.sentiment.subjectivity
        
        # Convert to our format
        if polarity > 0:
            positive_score = 0.5 + (polarity / 2)
            negative_score = 0.0
            neutral_score = 0.5 - (polarity / 2)
        elif polarity < 0:
            positive_score = 0.0
            negative_score = 0.5 + (abs(polarity) / 2)
            neutral_score = 0.5 - (abs(polarity) / 2)
        else:
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
            
        # Adjust by subjectivity
        if subjectivity < 0.5:
            # More objective (factual) text should be more neutral
            factor = 1.0 - subjectivity
            neutral_score = neutral_score * (1 - factor) + factor
            positive_score *= (1 - factor)
            negative_score *= (1 - factor)
            
        # Normalize scores
        total = positive_score + negative_score + neutral_score
        if total > 0:
            positive_score /= total
            negative_score /= total
            neutral_score /= total
        
        # Detect emotions (TextBlob doesn't do this directly)
        # Use our lexical approach for emotion detection
        tokens = re.findall(r'\b\w+\b', text.lower())
        total_words = len(tokens)
        
        detected_emotions = {}
        for emotion, word_set in self.emotion_lexicons.items():
            emotion_words = [token for token in tokens if token in word_set]
            if emotion_words and total_words > 0:
                detected_emotions[emotion] = len(emotion_words) / total_words
        
        # Highlight key sentences by polarity
        highlighted_phrases = []
        
        # TextBlob can analyze sentiment by sentence
        for i, sentence in enumerate(blob.sentences):
            if abs(sentence.sentiment.polarity) > 0.3:
                highlighted_phrases.append({
                    "text": str(sentence),
                    "score": sentence.sentiment.polarity,
                    "index": text.find(str(sentence))
                })
        
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "compound_score": polarity,
            "detected_emotions": detected_emotions,
            "highlighted_phrases": highlighted_phrases,
            "confidence": 0.5,
            "method": "textblob"
        }
    
    def _neural_sentiment_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neural network-based sentiment analysis
        
        Args:
            text: Text to analyze
            context: Contextual information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not TORCH_AVAILABLE or not self.neural_analyzer:
            raise ImportError("PyTorch is not available or neural analyzer not initialized")
            
        # Tokenize text
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Convert to indices
        unknown_idx = len(self.vocab)
        indices = [self.vocab.get(token, unknown_idx) for token in tokens]
        
        # Handle empty input
        if not indices:
            indices = [unknown_idx]
            
        # Convert to tensor
        tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.neural_analyzer(tensor).squeeze(0)
            
        # Extract scores
        positive_score = predictions[0].item()
        negative_score = predictions[1].item()
        neutral_score = predictions[2].item()
        
        # Calculate compound score
        compound_score = positive_score - negative_score
        
        # Record activation for tracking purposes
        if hasattr(self, "neural_state") and self.neural_state is not None:
            self.neural_state.add_activation('sentiment', {
                'inputs': len(tokens),
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score,
                'compound': compound_score
            })
        
        # Detect emotions
        # In a real implementation, this would be done by the neural network
        # Here we'll use a hybrid approach combining neural sentiment with lexical emotion detection
        detected_emotions = {}
        
        # Use our lexical approach for emotion detection
        for emotion, word_set in self.emotion_lexicons.items():
            emotion_words = [token for token in tokens if token in word_set]
            if emotion_words:
                # Weight the emotion by the sentiment scores
                emotion_score = len(emotion_words) / max(1, len(tokens))
                
                # Adjust score based on sentiment alignment
                if emotion in ["joy", "trust", "anticipation"]:
                    emotion_score *= (0.5 + 0.5 * positive_score)
                elif emotion in ["sadness", "anger", "fear", "disgust"]:
                    emotion_score *= (0.5 + 0.5 * negative_score)
                
                detected_emotions[emotion] = emotion_score
        
        # Highlight key phrases
        highlighted_phrases = []
        
        # For simple implementation, split into sentences and analyze each
        sentences = text.split('. ')
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Analyze sentiment of sentence
            sentence_tokens = re.findall(r'\b\w+\b', sentence.lower())
            
            # Skip very short sentences
            if len(sentence_tokens) < 3:
                continue
                
            # Count emotional words in sentence
            pos_count = sum(1 for token in sentence_tokens if token in self.positive_words)
            neg_count = sum(1 for token in sentence_tokens if token in self.negative_words)
            
            # Calculate sentence sentiment
            if pos_count > neg_count:
                sentence_score = 0.5 + (pos_count / (2 * len(sentence_tokens)))
            elif neg_count > pos_count:
                sentence_score = -0.5 - (neg_count / (2 * len(sentence_tokens)))
            else:
                sentence_score = 0.0
                
            # Only include sentences with clear sentiment
            if abs(sentence_score) > 0.2:
                highlighted_phrases.append({
                    "text": sentence,
                    "score": sentence_score,
                    "index": text.find(sentence)
                })
        
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": neutral_score,
            "compound_score": compound_score,
            "detected_emotions": detected_emotions,
            "highlighted_phrases": highlighted_phrases,
            "confidence": 0.6,
            "method": "neural"
        }
    
    def _llm_sentiment_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM-based sentiment analysis for advanced processing
        
        Args:
            text: Text to analyze
            context: Contextual information
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.llm_client:
            raise ImportError("LLM client is not available")
            
        try:
            # Only use for substantial text
            if len(text) < 20:
                raise ValueError("Text too short for LLM analysis")
                
            # Prepare prompt for structured output
            messages = [
                LLMMessage(role="system", content="""
                You are a sentiment analysis system. Analyze the emotional tone of the provided text and return a JSON object with the following structure:
                {
                    "positive_score": float (0-1),
                    "negative_score": float (0-1),
                    "neutral_score": float (0-1),
                    "compound_score": float (-1 to 1),
                    "detected_emotions": {
                        "emotion1": float (0-1),
                        "emotion2": float (0-1),
                        ...
                    },
                    "highlighted_phrases": [
                        {"text": "phrase 1", "score": float (-1 to 1)},
                        {"text": "phrase 2", "score": float (-1 to 1)},
                        ...
                    ]
                }
                
                Primary emotions to detect: joy, sadness, anger, fear, surprise, disgust, trust, anticipation.
                Ensure all scores sum to 1.0. Only include emotions with significant presence.
                """),
                LLMMessage(role="user", content=f"Analyze the emotional tone and sentiment of this text: \"{text}\"")
            ]
            
            # Get response from LLM
            response = self.llm_client.chat_completion(messages)
            
            # Extract JSON from response
            import json
            import re
            
            # First try to parse the whole response as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            
            # Extract values from result
            positive_score = result.get("positive_score", 0.33)
            negative_score = result.get("negative_score", 0.33)
            neutral_score = result.get("neutral_score", 0.34)
            compound_score = result.get("compound_score", 0.0)
            detected_emotions = result.get("detected_emotions", {})
            highlighted_phrases = result.get("highlighted_phrases", [])
            
            # Ensure all values are in correct ranges
            positive_score = max(0.0, min(1.0, positive_score))
            negative_score = max(0.0, min(1.0, negative_score))
            neutral_score = max(0.0, min(1.0, neutral_score))
            compound_score = max(-1.0, min(1.0, compound_score))
            
            # Normalize scores
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            # Ensure detected emotions are in range
            detected_emotions = {
                k: max(0.0, min(1.0, v)) 
                for k, v in detected_emotions.items()
            }
            
            # Process highlighted phrases
            for phrase in highlighted_phrases:
                if "score" in phrase:
                    phrase["score"] = max(-1.0, min(1.0, phrase["score"]))
                if "text" in phrase and "index" not in phrase:
                    phrase["index"] = text.find(phrase["text"])
            
            return {
                "positive_score": positive_score,
                "negative_score": negative_score,
                "neutral_score": neutral_score,
                "compound_score": compound_score,
                "detected_emotions": detected_emotions,
                "highlighted_phrases": highlighted_phrases,
                "confidence": 0.8,
                "method": "llm"
            }
                
        except Exception as e:
            logger.warning(f"LLM-based sentiment analysis failed: {str(e)}")
            # Fallback to textblob analysis
            if TEXTBLOB_AVAILABLE:
                return self._textblob_sentiment_analysis(text, context)
            else:
                return self._lexical_sentiment_analysis(text, context)
    
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
        
        # Update neural state development level
        if hasattr(self, "neural_state"):
            self.neural_state.sentiment_development = new_level
            self.neural_state.last_updated = datetime.now()
        
        # Adjust parameters for new development level
        self._adjust_parameters_for_development()
        
        # Initialize neural analyzer if development is high enough
        if new_level >= 0.4 and not self.neural_analyzer and TORCH_AVAILABLE:
            self._initialize_neural_analyzer()
            
        # Initialize LLM client if development is high enough
        if new_level >= 0.6 and not self.llm_client:
            try:
                self.llm_client = LLMClient()
                logger.info("LLM client initialized for advanced sentiment analysis")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state
        
        Returns:
            Dictionary with current state
        """
        base_state = super().get_state()
        
        # Add sentiment-specific state
        sentiment_state = {
            "params": self.params,
            "analysis_history_length": len(self.analysis_history),
            "last_analysis": self.analysis_history[-1].dict() if self.analysis_history else None,
            "available_methods": {
                "lexical": True,
                "textblob": TEXTBLOB_AVAILABLE,
                "neural": TORCH_AVAILABLE and self.neural_analyzer is not None,
                "llm": self.llm_client is not None
            }
        }
        
        # Combine states
        combined_state = {**base_state, **sentiment_state}
        
        return combined_state 
