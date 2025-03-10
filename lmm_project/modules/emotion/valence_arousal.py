"""
Valence-Arousal Emotional Processing System

This component processes inputs to determine their emotional valence 
(positive to negative) and arousal (level of activation/intensity).
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import re

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient, Message as LLMMessage
from lmm_project.modules.emotion.models import EmotionNeuralState

# Initialize logger
logger = logging.getLogger(__name__)

class ValenceArousalNetwork(nn.Module):
    """
    Neural network for extracting valence and arousal from input features
    
    This network gets increasingly sophisticated with development
    """
    def __init__(self, input_dim=10, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        valence = torch.tanh(self.valence_head(x))  # -1 to 1
        arousal = torch.sigmoid(self.arousal_head(x))  # 0 to 1
        return valence, arousal

class ValenceArousalSystem(BaseModule):
    """
    System for processing emotional valence and arousal
    
    This system develops from basic pleasure/pain distinction
    to nuanced emotional dimension processing.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic pleasure/pain distinction",
        0.2: "Intensity differentiation",
        0.4: "Context-sensitive valence",
        0.6: "Nuanced arousal sensitivity",
        0.8: "Complex emotional dimensionality",
        1.0: "Sophisticated valence-arousal processing"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the valence-arousal system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="valence_arousal",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize lexical resources for emotion detection
        self._initialize_lexical_resources()
        
        # Create neural network for valence-arousal processing
        self.input_dim = 10
        self.hidden_dim = 20
        self.network = ValenceArousalNetwork(self.input_dim, self.hidden_dim)
        
        # Try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Neural state for tracking activations and development
        self.neural_state = EmotionNeuralState()
        self.neural_state.encoder_development = development_level
        
        # Valence-Arousal history
        self.history = deque(maxlen=100)
        
        # Initialize parameters
        self.params = {
            "default_valence": 0.0,     # Neutral by default
            "default_arousal": 0.2,     # Low arousal by default
            "valence_sensitivity": 0.6, # How sensitive to valence cues
            "arousal_sensitivity": 0.5, # How sensitive to arousal cues
            "context_weight": 0.3,      # How much context affects VA
            "development_factor": development_level,  # Scales with development
        }
        
        # Adjust parameters based on development level
        self._adjust_params_for_development()
        
        # Try to initialize LLM client if needed for advanced processing
        try:
            self.llm_client = LLMClient()
            self.has_llm = True
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}")
            self.has_llm = False
            
        logger.info(f"Valence-Arousal system initialized, development level: {development_level:.2f}")
        
    def _initialize_lexical_resources(self):
        """Initialize lexical resources for emotion detection"""
        # Valence word lists (positive and negative words)
        self.positive_words = {
            "joy", "happy", "glad", "delight", "pleasure", "content", 
            "satisfied", "bliss", "ecstatic", "good", "wonderful", 
            "great", "excellent", "amazing", "fantastic", "terrific",
            "lovely", "beautiful", "nice", "pleasant", "enjoyable"
        }
        
        self.negative_words = {
            "sad", "unhappy", "miserable", "depressed", "gloomy", "somber",
            "melancholy", "sorrow", "grief", "despair", "distress",
            "anger", "angry", "furious", "enraged", "mad", "irritated",
            "fear", "afraid", "scared", "terrified", "anxious", "worried",
            "hate", "dislike", "disgust", "awful", "terrible", "horrible",
            "bad", "unpleasant", "hurt", "painful", "suffering"
        }
        
        # Arousal word lists (high and low activation)
        self.high_arousal_words = {
            "excited", "thrilled", "ecstatic", "energetic", "alert",
            "active", "aroused", "stimulated", "agitated", "frantic",
            "tense", "stressed", "nervous", "restless", "hyper",
            "enraged", "furious", "terrified", "shocked", "overwhelmed",
            "exhilarated", "vibrant", "intense", "passionate", "eager"
        }
        
        self.low_arousal_words = {
            "calm", "relaxed", "serene", "peaceful", "tranquil",
            "quiet", "still", "idle", "passive", "inactive", 
            "tired", "sleepy", "drowsy", "lethargic", "sluggish",
            "dull", "bored", "uninterested", "apathetic", "indifferent",
            "mellow", "soothing", "gentle", "mild", "subtle"
        }
        
        # Intensifiers and diminishers
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
        
    def _adjust_params_for_development(self):
        """Adjust parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very basic processing - simple pleasure/pain
            self.params.update({
                "valence_sensitivity": 0.5,
                "arousal_sensitivity": 0.3,
                "context_weight": 0.1,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.4:
            # Developing basic VA sensitivity
            self.params.update({
                "valence_sensitivity": 0.6,
                "arousal_sensitivity": 0.4,
                "context_weight": 0.2,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.6:
            # Developing context sensitivity
            self.params.update({
                "valence_sensitivity": 0.7,
                "arousal_sensitivity": 0.5,
                "context_weight": 0.3,
                "development_factor": self.development_level
            })
        elif self.development_level < 0.8:
            # Developing nuanced processing
            self.params.update({
                "valence_sensitivity": 0.8,
                "arousal_sensitivity": 0.7,
                "context_weight": 0.4,
                "development_factor": self.development_level
            })
        else:
            # Sophisticated processing
            self.params.update({
                "valence_sensitivity": 0.9,
                "arousal_sensitivity": 0.8,
                "context_weight": 0.5,
                "development_factor": self.development_level
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to extract valence and arousal
        
        Args:
            input_data: Input data to process
                Required keys: at least one of 'content' or 'valence'/'arousal'
                Optional keys: 'source', 'context'
                
        Returns:
            Dictionary with valence and arousal results
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Direct valence/arousal values take precedence if provided
        if "valence" in input_data and "arousal" in input_data:
            valence = max(-1.0, min(1.0, input_data["valence"]))
            arousal = max(0.0, min(1.0, input_data["arousal"]))
            
            # Add to history
            self.history.append({
                "valence": valence,
                "arousal": arousal,
                "source": input_data.get("source", "direct"),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "valence": valence,
                "arousal": arousal,
                "method": "direct",
                "process_id": process_id,
                "development_level": self.development_level
            }
        
        # Otherwise, extract from content
        content = input_data.get("content", {})
        text = ""
        
        # Extract text from content
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict) and "text" in content:
            text = content["text"]
        
        if not text:
            # No content to process
            return {
                "valence": self.params["default_valence"],
                "arousal": self.params["default_arousal"],
                "method": "default",
                "process_id": process_id,
                "development_level": self.development_level
            }
        
        # Process text to extract VA
        result = self._process_text(text, input_data.get("context", {}))
        result["process_id"] = process_id
        result["development_level"] = self.development_level
        
        # Add to history
        self.history.append({
            "valence": result["valence"],
            "arousal": result["arousal"],
            "source": input_data.get("source", "text"),
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def _process_text(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text to extract valence and arousal
        
        Args:
            text: Text to process
            context: Contextual information
            
        Returns:
            Dictionary with valence and arousal results
        """
        # Methods to use based on development level
        methods = []
        
        # Always use lexical approach
        methods.append(self._lexical_va_extraction)
        
        # Add more sophisticated methods with development
        if self.development_level >= 0.3:
            methods.append(self._pattern_va_extraction)
            
        if self.development_level >= 0.6 and self.has_llm:
            methods.append(self._llm_va_extraction)
            
        # Process with each method and combine results
        results = []
        method_names = []
        
        for method in methods:
            try:
                result = method(text, context)
                results.append((result["valence"], result["arousal"]))
                method_names.append(result["method"])
            except Exception as e:
                logger.error(f"Error in VA extraction method {method.__name__}: {str(e)}")
        
        # Calculate weighted average of results
        # More sophisticated methods have higher weights as development increases
        if not results:
            return {
                "valence": self.params["default_valence"],
                "arousal": self.params["default_arousal"],
                "method": "default",
                "confidence": 0.1
            }
        
        # Apply weights based on development and method sophistication
        if len(results) == 1:
            weights = [1.0]
        elif len(results) == 2:
            weights = [0.6, 0.4] if self.development_level < 0.5 else [0.3, 0.7]
        elif len(results) == 3:
            weights = [0.4, 0.3, 0.3] if self.development_level < 0.7 else [0.2, 0.3, 0.5]
        else:
            weights = [1.0 / len(results)] * len(results)
            
        weighted_valence = sum(w * v for w, (v, _) in zip(weights, results))
        weighted_arousal = sum(w * a for w, (_, a) in zip(weights, results))
        
        return {
            "valence": weighted_valence,
            "arousal": weighted_arousal,
            "method": "+".join(method_names),
            "confidence": min(0.3 + self.development_level, 0.9)
        }
    
    def _lexical_va_extraction(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract valence and arousal using lexical approach
        
        Args:
            text: Text to process
            context: Contextual information
            
        Returns:
            Dictionary with valence and arousal results
        """
        # Basic lexical approach - count positive and negative words
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Count valence words
        pos_count = sum(1 for token in tokens if token in self.positive_words)
        neg_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Count arousal words
        high_arousal_count = sum(1 for token in tokens if token in self.high_arousal_words)
        low_arousal_count = sum(1 for token in tokens if token in self.low_arousal_words)
        
        # Count intensifiers and diminishers
        intensifier_count = sum(1 for token in tokens if token in self.intensifiers)
        diminisher_count = sum(1 for token in tokens if token in self.diminishers)
        
        # Calculate valence (-1 to 1)
        if pos_count == 0 and neg_count == 0:
            valence = 0.0  # Neutral
        else:
            valence = (pos_count - neg_count) / (pos_count + neg_count)
            
        # Modify valence based on intensifiers/diminishers
        if valence > 0:
            valence_modifier = 0.2 * intensifier_count - 0.1 * diminisher_count
            valence = min(1.0, valence + valence_modifier * self.params["valence_sensitivity"])
        elif valence < 0:
            valence_modifier = 0.2 * intensifier_count - 0.1 * diminisher_count
            valence = max(-1.0, valence - valence_modifier * self.params["valence_sensitivity"])
            
        # Calculate arousal (0 to 1)
        if high_arousal_count == 0 and low_arousal_count == 0:
            arousal = 0.5  # Moderate
        else:
            arousal = (high_arousal_count) / (high_arousal_count + low_arousal_count + 0.001)
            
        # Modify arousal based on intensifiers/diminishers
        arousal_modifier = 0.2 * intensifier_count - 0.1 * diminisher_count
        arousal = min(1.0, max(0.0, arousal + arousal_modifier * self.params["arousal_sensitivity"]))
        
        # Adjust based on development level - lower development gives more extreme values
        if self.development_level < 0.3:
            # Exaggerate responses at low development
            valence = 0.6 * valence + 0.4 * np.sign(valence) * abs(valence) ** 0.5
            arousal = 0.6 * arousal + 0.4 * (0.2 + 0.8 * arousal ** 0.5)
        
        return {
            "valence": valence,
            "arousal": arousal,
            "method": "lexical",
            "confidence": 0.3 + 0.2 * self.development_level
        }
    
    def _pattern_va_extraction(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract valence and arousal using pattern-based approach
        
        Args:
            text: Text to process
            context: Contextual information
            
        Returns:
            Dictionary with valence and arousal results
        """
        # Extract features for neural network processing
        features = self._extract_text_features(text)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Process through network
        with torch.no_grad():
            valence_tensor, arousal_tensor = self.network(features_tensor)
            
        # Convert to scalar values
        valence = valence_tensor.item()
        arousal = arousal_tensor.item()
        
        # Record activation for tracking
        if hasattr(self, "neural_state"):
            self.neural_state.add_activation('encoder', {
                'features': features,
                'valence': valence,
                'arousal': arousal
            })
        
        return {
            "valence": valence,
            "arousal": arousal,
            "method": "pattern",
            "confidence": 0.4 + 0.3 * self.development_level
        }
    
    def _extract_text_features(self, text: str) -> List[float]:
        """
        Extract features from text for neural processing
        
        Args:
            text: Text to extract features from
            
        Returns:
            List of feature values
        """
        # Count total words
        tokens = re.findall(r'\b\w+\b', text.lower())
        word_count = len(tokens)
        
        # Calculate feature values
        features = [
            # 1. Positive word ratio
            sum(1 for token in tokens if token in self.positive_words) / max(1, word_count),
            
            # 2. Negative word ratio
            sum(1 for token in tokens if token in self.negative_words) / max(1, word_count),
            
            # 3. High arousal word ratio
            sum(1 for token in tokens if token in self.high_arousal_words) / max(1, word_count),
            
            # 4. Low arousal word ratio
            sum(1 for token in tokens if token in self.low_arousal_words) / max(1, word_count),
            
            # 5. Intensifier ratio
            sum(1 for token in tokens if token in self.intensifiers) / max(1, word_count),
            
            # 6. Diminisher ratio
            sum(1 for token in tokens if token in self.diminishers) / max(1, word_count),
            
            # 7. Exclamation mark count
            text.count('!') / max(1, len(text) / 50),
            
            # 8. Question mark count
            text.count('?') / max(1, len(text) / 50),
            
            # 9. Capitalization ratio
            sum(1 for c in text if c.isupper()) / max(1, len([c for c in text if c.isalpha()])),
            
            # 10. Average word length
            sum(len(token) for token in tokens) / max(1, word_count)
        ]
        
        return features
    
    def _llm_va_extraction(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract valence and arousal using LLM
        
        This sophisticated method is only used at higher development levels
        
        Args:
            text: Text to process
            context: Contextual information
            
        Returns:
            Dictionary with valence and arousal results
        """
        try:
            # Only use for non-trivial text
            if len(text) < 10 or not self.has_llm:
                raise ValueError("Text too short or LLM unavailable")
                
            # Prepare prompt
            messages = [
                LLMMessage(role="system", content="""
                You are an emotion analysis system focused on extracting valence and arousal from text.
                - Valence ranges from -1.0 (very negative) to 1.0 (very positive), with 0 being neutral.
                - Arousal ranges from 0.0 (calm/inactive) to 1.0 (excited/agitated).
                Respond ONLY with a JSON object containing valence and arousal values.
                """),
                LLMMessage(role="user", content=f"Analyze the emotional dimensions of this text: \"{text}\"")
            ]
            
            # Get response from LLM (with timeout)
            response = self.llm_client.chat_completion(messages)
            
            # Extract values from response
            # Expecting format like {"valence": 0.5, "arousal": 0.7}
            import json
            
            # First try to parse the whole response as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            
            # Validate and extract values
            valence = result.get("valence", 0.0)
            arousal = result.get("arousal", 0.5)
            
            # Ensure values are in the correct range
            valence = max(-1.0, min(1.0, valence))
            arousal = max(0.0, min(1.0, arousal))
            
            return {
                "valence": valence,
                "arousal": arousal,
                "method": "llm",
                "confidence": 0.6 + 0.3 * self.development_level
            }
            
        except Exception as e:
            logger.warning(f"LLM-based VA extraction failed: {str(e)}")
            # Fallback to pattern-based approach
            return self._pattern_va_extraction(text, context)
    
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
        self._adjust_params_for_development()
        
        # Update neural state
        if hasattr(self, "neural_state"):
            self.neural_state.encoder_development = new_level
            self.neural_state.last_updated = datetime.now()
        
        # More sophisticated neural network as development progresses
        if new_level > 0.5 and self.hidden_dim < 40:
            # Increase network complexity
            self.hidden_dim = 40
            self.network = ValenceArousalNetwork(self.input_dim, self.hidden_dim)
            self.network.to(self.device)
            logger.info(f"Upgraded VA network complexity at development level {new_level:.2f}")
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state
        
        Returns:
            Dictionary with current state
        """
        base_state = super().get_state()
        
        # Add VA-specific state
        va_state = {
            "params": self.params,
            "history_length": len(self.history),
            "last_values": list(self.history)[-1] if self.history else None
        }
        
        # Combine states
        combined_state = {**base_state, **va_state}
        
        return combined_state