"""
Emotion Module

This module is responsible for processing, generating, and regulating
emotional responses. It serves as the affective core of the Mind, enabling
emotional experiences, sentiment evaluation, and emotional regulation.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.emotion.valence_arousal import ValenceArousalSystem
from lmm_project.modules.emotion.emotion_classifier import EmotionClassifier
from lmm_project.modules.emotion.sentiment_analyzer import SentimentAnalyzer
from lmm_project.modules.emotion.regulation import EmotionRegulator
from lmm_project.modules.emotion.models import EmotionState, EmotionalResponse, SentimentAnalysis

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "emotion",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "EmotionSystem":
    """
    Factory function to create and return an emotion module
    
    This function initializes and returns a complete emotion system with
    valence-arousal tracking, emotion classification, sentiment analysis,
    and emotion regulation capabilities.
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication
        development_level: Initial developmental level for the system
        
    Returns:
        Initialized EmotionSystem
    """
    return EmotionSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class EmotionSystem(BaseModule):
    """
    Emotion system responsible for affective processing
    
    The emotion system develops from basic pleasure/displeasure responses
    to sophisticated emotional understanding, regulation, and expression.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic affect reactions",
        0.2: "Primary emotions",
        0.4: "Secondary emotions",
        0.6: "Emotional self-awareness",
        0.8: "Complex emotional understanding",
        1.0: "Sophisticated emotional intelligence"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the emotion system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="emotion_system",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Emotional state tracking
        self.current_state = EmotionState(
            valence=0.0,      # Neutral valence initially
            arousal=0.1,      # Low arousal initially
            dominant_emotion="neutral",
            emotion_intensities={
                "neutral": 1.0,
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0,
                "anticipation": 0.0,
                "trust": 0.0
            },
            timestamp=datetime.now()
        )
        
        # Emotional memory - recent emotional states
        self.emotion_history = deque(maxlen=50)
        self.emotion_history.append(self.current_state)
        
        # Create emotional subsystems
        self.valence_arousal_system = ValenceArousalSystem(
            module_id=f"{module_id}_va",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.emotion_classifier = EmotionClassifier(
            module_id=f"{module_id}_classifier",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.sentiment_analyzer = SentimentAnalyzer(
            module_id=f"{module_id}_sentiment",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.emotion_regulator = EmotionRegulator(
            module_id=f"{module_id}_regulation",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Emotional parameters - adjust based on development
        self.emotional_params = {
            # How quickly emotions change
            "emotional_inertia": 0.8,
            
            # How strongly stimulus affects emotion
            "stimulus_sensitivity": 0.6,
            
            # How quickly emotions decay over time
            "emotion_decay_rate": 0.05,
            
            # Baseline emotional state to return to
            "baseline_valence": 0.1,
            "baseline_arousal": 0.2,
            
            # Emotional regulation strength
            "regulation_capacity": 0.2
        }
        
        # Adjust parameters based on development level
        self._adjust_parameters_for_development()
        
        # Try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("perception_result")
            self.subscribe_to_message("attention_focus")
            self.subscribe_to_message("memory_retrieval")
            self.subscribe_to_message("emotion_query")
            self.subscribe_to_message("emotion_regulation")
        
        logger.info(f"Emotion system initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust emotional parameters based on developmental level"""
        if self.development_level < 0.2:
            # Early development - basic emotional reactions
            self.emotional_params.update({
                "emotional_inertia": 0.4,         # Emotions change quickly
                "stimulus_sensitivity": 0.8,      # Strong reactions to stimuli
                "emotion_decay_rate": 0.1,        # Quick return to baseline
                "baseline_valence": 0.2,          # Slightly positive baseline
                "baseline_arousal": 0.3,          # Moderate arousal baseline
                "regulation_capacity": 0.1        # Very limited regulation
            })
        elif self.development_level < 0.4:
            # Developing primary emotions
            self.emotional_params.update({
                "emotional_inertia": 0.5,
                "stimulus_sensitivity": 0.7,
                "emotion_decay_rate": 0.08,
                "baseline_valence": 0.15,
                "baseline_arousal": 0.25,
                "regulation_capacity": 0.2
            })
        elif self.development_level < 0.6:
            # Developing secondary emotions
            self.emotional_params.update({
                "emotional_inertia": 0.6,
                "stimulus_sensitivity": 0.6,
                "emotion_decay_rate": 0.06,
                "baseline_valence": 0.1,
                "baseline_arousal": 0.2,
                "regulation_capacity": 0.4
            })
        elif self.development_level < 0.8:
            # Developing emotional self-awareness
            self.emotional_params.update({
                "emotional_inertia": 0.7,
                "stimulus_sensitivity": 0.5,
                "emotion_decay_rate": 0.04,
                "baseline_valence": 0.05,
                "baseline_arousal": 0.15,
                "regulation_capacity": 0.6
            })
        else:
            # Advanced emotional intelligence
            self.emotional_params.update({
                "emotional_inertia": 0.8,
                "stimulus_sensitivity": 0.4,
                "emotion_decay_rate": 0.02,
                "baseline_valence": 0.0,
                "baseline_arousal": 0.1,
                "regulation_capacity": 0.8
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to generate emotional responses
        
        Args:
            input_data: Data to process for emotional response
                Required keys: 'content'
                Optional keys: 'valence', 'arousal', 'source', 'context'
                
        Returns:
            Dictionary with emotional response
        """
        # Generate ID for this emotion process
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Determine operation type
        operation = input_data.get("operation", "generate")
        
        # Route to appropriate handler
        if operation == "generate":
            return self._handle_generate_emotion(input_data)
        elif operation == "analyze":
            return self._handle_analyze_sentiment(input_data)
        elif operation == "regulate":
            return self._handle_regulate_emotion(input_data)
        elif operation == "query":
            return self._handle_emotion_query(input_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _handle_generate_emotion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emotion generation operation"""
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        content = input_data.get("content", {})
        text = content.get("text", "")
        context = input_data.get("context", {})
        
        # Process through valence-arousal system
        va_result = self.valence_arousal_system.process_input(input_data)
        
        # Get valence and arousal
        valence = va_result.get("valence", 0.0)
        arousal = va_result.get("arousal", 0.0)
        
        # If direct values were provided, use those
        if "valence" in input_data:
            valence = input_data["valence"]
        if "arousal" in input_data:
            arousal = input_data["arousal"]
        
        # Classify the emotion based on valence and arousal
        classification_input = {
            "valence": valence,
            "arousal": arousal,
            "text": text,
            "context": context
        }
        classification_result = self.emotion_classifier.process_input(classification_input)
        
        # Get emotional classification
        emotion_intensities = classification_result.get("emotion_intensities", 
                                                     {"neutral": 1.0})
        dominant_emotion = classification_result.get("dominant_emotion", "neutral")
        
        # Update current emotional state with inertia
        inertia = self.emotional_params["emotional_inertia"]
        
        new_valence = (inertia * self.current_state.valence + 
                      (1 - inertia) * valence)
        new_arousal = (inertia * self.current_state.arousal + 
                      (1 - inertia) * arousal)
        
        # Create new emotional state
        new_state = EmotionState(
            valence=new_valence,
            arousal=new_arousal,
            dominant_emotion=dominant_emotion,
            emotion_intensities=emotion_intensities,
            timestamp=datetime.now()
        )
        
        # Apply regulation if development allows
        if self.development_level >= 0.2:
            regulation_input = {
                "current_state": new_state,
                "context": context,
                "regulation_capacity": self.emotional_params["regulation_capacity"]
            }
            regulation_result = self.emotion_regulator.process_input(regulation_input)
            
            # Get regulated state
            if "regulated_state" in regulation_result:
                new_state = regulation_result["regulated_state"]
        
        # Update the current state
        self.current_state = new_state
        self.emotion_history.append(new_state)
        
        # Create emotional response
        response = EmotionalResponse(
            valence=new_state.valence,
            arousal=new_state.arousal,
            dominant_emotion=new_state.dominant_emotion,
            emotion_intensities=new_state.emotion_intensities,
            regulated=self.development_level >= 0.2,
            stimulus=text,
            process_id=process_id,
            timestamp=datetime.now()
        )
        
        # Prepare result dictionary
        result = {
            "status": "success",
            "process_id": process_id,
            "response": response.dict(),
            "development_level": self.development_level
        }
        
        # Publish emotional state update
        if self.event_bus:
            self.publish_message(
                "emotion_state",
                {
                    "state": new_state.dict(),
                    "process_id": process_id
                }
            )
        
        return result
    
    def _handle_analyze_sentiment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sentiment analysis operation"""
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Process through sentiment analyzer
        sentiment_result = self.sentiment_analyzer.process_input(input_data)
        
        return {
            "status": "success",
            "process_id": process_id,
            "analysis": sentiment_result,
            "development_level": self.development_level
        }
    
    def _handle_regulate_emotion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emotion regulation operation"""
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Only handle if sufficiently developed
        if self.development_level < 0.2:
            return {
                "status": "undeveloped",
                "message": "Emotion regulation not yet developed",
                "process_id": process_id,
                "development_level": self.development_level
            }
        
        # Extract regulation parameters
        current_state = input_data.get("current_state")
        if current_state is None:
            # Use current emotional state if none provided
            current_state = self.current_state
            
        # Process through emotion regulator
        regulation_input = {
            "current_state": current_state,
            "process_id": process_id,
            "regulation_capacity": self.emotional_params["regulation_capacity"]
        }
        
        # Add target valence and arousal if provided
        if "target_valence" in input_data:
            regulation_input["target_valence"] = input_data["target_valence"]
            
        if "target_arousal" in input_data:
            regulation_input["target_arousal"] = input_data["target_arousal"]
            
        if "regulation_strategy" in input_data:
            regulation_input["regulation_strategy"] = input_data["regulation_strategy"]
            
        # Process the regulation request
        regulation_result = self.emotion_regulator.process_input(regulation_input)
        
        # Update current state if regulation was applied and using system's emotional state
        if current_state == self.current_state and "regulated_state" in regulation_result:
            self.current_state = regulation_result["regulated_state"]
            self.emotion_history.append(self.current_state)
            
            # Publish updated state
            if self.event_bus:
                self.publish_message(
                    "emotion_state",
                    {
                        "state": self.current_state.dict(),
                        "process_id": process_id,
                        "regulated": True
                    }
                )
        
        # Return the regulation result directly without extra nesting
        return regulation_result
    
    def _handle_emotion_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query about current emotional state"""
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        return {
            "status": "success",
            "process_id": process_id,
            "current_state": self.current_state.dict(),
            "development_level": self.development_level,
            "emotional_capacity": self._get_emotional_capacity()
        }
    
    def _get_emotional_capacity(self) -> Dict[str, Any]:
        """Get information about current emotional capabilities"""
        if self.development_level < 0.2:
            return {
                "available_emotions": ["pleasure", "displeasure"],
                "regulation_capacity": self.emotional_params["regulation_capacity"],
                "emotional_complexity": "basic",
                "self_awareness": "none"
            }
        elif self.development_level < 0.4:
            return {
                "available_emotions": ["joy", "sadness", "anger", "fear"],
                "regulation_capacity": self.emotional_params["regulation_capacity"],
                "emotional_complexity": "primary",
                "self_awareness": "minimal"
            }
        elif self.development_level < 0.6:
            return {
                "available_emotions": [
                    "joy", "sadness", "anger", "fear", 
                    "surprise", "disgust", "anticipation", "trust"
                ],
                "regulation_capacity": self.emotional_params["regulation_capacity"],
                "emotional_complexity": "secondary",
                "self_awareness": "developing"
            }
        elif self.development_level < 0.8:
            return {
                "available_emotions": [
                    "joy", "sadness", "anger", "fear", 
                    "surprise", "disgust", "anticipation", "trust",
                    "shame", "guilt", "pride", "love", "jealousy"
                ],
                "regulation_capacity": self.emotional_params["regulation_capacity"],
                "emotional_complexity": "complex",
                "self_awareness": "substantial"
            }
        else:
            return {
                "available_emotions": [
                    "joy", "sadness", "anger", "fear", 
                    "surprise", "disgust", "anticipation", "trust",
                    "shame", "guilt", "pride", "love", "jealousy",
                    "gratitude", "awe", "contentment", "interest",
                    "contempt", "embarrassment", "longing"
                ],
                "regulation_capacity": self.emotional_params["regulation_capacity"],
                "emotional_complexity": "nuanced",
                "self_awareness": "sophisticated"
            }
    
    def _handle_message(self, message: Message):
        """Handle messages from the event bus"""
        if message.message_type == "perception_result":
            self._handle_perception_message(message)
        elif message.message_type == "attention_focus":
            self._handle_attention_message(message)
        elif message.message_type == "memory_retrieval":
            self._handle_memory_message(message)
        elif message.message_type == "emotion_query":
            self._handle_query_message(message)
        elif message.message_type == "emotion_regulation":
            self._handle_regulation_message(message)
    
    def _handle_perception_message(self, message: Message):
        """Process perception results to generate emotional responses"""
        content = message.content
        if "result" not in content:
            return
            
        result = content["result"]
        
        # Process text for emotional content
        if "text" in result:
            input_data = {
                "content": {"text": result["text"]},
                "process_id": content.get("process_id", str(uuid.uuid4())),
                "source": "perception"
            }
            emotion_result = self._handle_generate_emotion(input_data)
    
    def _handle_attention_message(self, message: Message):
        """Process attention focus to modulate emotional responses"""
        content = message.content
        if "focus" not in content:
            return
            
        focus = content["focus"]
        
        # Attention amplifies emotional response to focused content
        if "content" in focus:
            # Extract text if present
            text = ""
            if isinstance(focus["content"], dict) and "text" in focus["content"]:
                text = focus["content"]["text"]
            elif isinstance(focus["content"], str):
                text = focus["content"]
                
            if text:
                # Amplify emotional response to attended content
                input_data = {
                    "content": {"text": text},
                    "process_id": content.get("process_id", str(uuid.uuid4())),
                    "source": "attention",
                    # Boost sensitivity for attended content
                    "sensitivity_boost": 0.3
                }
                self._handle_generate_emotion(input_data)
    
    def _handle_memory_message(self, message: Message):
        """Process memory retrievals to generate emotional responses"""
        content = message.content
        if "memory" not in content:
            return
            
        memory = content["memory"]
        
        # Extract emotional aspects from memory
        if "emotional_valence" in memory:
            # Direct emotional content in memory
            input_data = {
                "valence": memory.get("emotional_valence", 0.0),
                "arousal": memory.get("emotional_arousal", 0.3),
                "process_id": content.get("process_id", str(uuid.uuid4())),
                "source": "memory",
                # Memories have reduced emotional impact
                "intensity": 0.7
            }
            self._handle_generate_emotion(input_data)
        elif "content" in memory:
            # Process content for emotional aspects
            text = ""
            if isinstance(memory["content"], dict) and "text" in memory["content"]:
                text = memory["content"]["text"]
            elif isinstance(memory["content"], str):
                text = memory["content"]
                
            if text:
                input_data = {
                    "content": {"text": text},
                    "process_id": content.get("process_id", str(uuid.uuid4())),
                    "source": "memory",
                    # Memories have reduced emotional impact
                    "intensity": 0.7
                }
                self._handle_generate_emotion(input_data)
    
    def _handle_query_message(self, message: Message):
        """Handle queries about emotional state"""
        content = message.content
        query_type = content.get("query_type", "current_state")
        
        response_data = None
        if query_type == "current_state":
            response_data = self._handle_emotion_query(content)
        elif query_type == "emotional_capacity":
            response_data = {
                "status": "success",
                "emotional_capacity": self._get_emotional_capacity(),
                "development_level": self.development_level
            }
        elif query_type == "emotion_history":
            count = content.get("count", 5)
            history = list(self.emotion_history)[-count:]
            response_data = {
                "status": "success",
                "history": [state.dict() for state in history],
                "count": len(history)
            }
            
        # Publish response if we have event bus
        if response_data and self.event_bus:
            self.publish_message(
                "emotion_query_response",
                {
                    "query_id": content.get("query_id", ""),
                    "response": response_data
                }
            )
    
    def _handle_regulation_message(self, message: Message):
        """Handle emotion regulation requests"""
        content = message.content
        
        # Only process if sufficiently developed
        if self.development_level < 0.3:
            # Not yet developed enough for regulation
            if self.event_bus:
                self.publish_message(
                    "emotion_regulation_response",
                    {
                        "regulation_id": content.get("regulation_id", ""),
                        "status": "undeveloped",
                        "message": "Emotion regulation not yet developed"
                    }
                )
            return
        
        # Process regulation request
        regulation_result = self._handle_regulate_emotion(content)
        
        # Publish response
        if self.event_bus:
            self.publish_message(
                "emotion_regulation_response",
                {
                    "regulation_id": content.get("regulation_id", ""),
                    "response": regulation_result
                }
            )
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the emotion system
        
        Args:
            amount: Amount to increase development (0.0 to 1.0)
            
        Returns:
            New developmental level
        """
        # Update base module development
        new_level = super().update_development(amount)
        
        # Update submodules development
        self.valence_arousal_system.update_development(amount)
        self.emotion_classifier.update_development(amount)
        self.sentiment_analyzer.update_development(amount)
        self.emotion_regulator.update_development(amount)
        
        # Adjust parameters for new development level
        self._adjust_parameters_for_development()
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the emotion system
        
        Returns:
            Dictionary with current state
        """
        base_state = super().get_state()
        
        # Add emotion-specific state
        emotion_state = {
            "current_emotion": self.current_state.dict(),
            "emotion_history_length": len(self.emotion_history),
            "emotional_params": self.emotional_params,
            "emotional_capacity": self._get_emotional_capacity()
        }
        
        # Combine states
        combined_state = {**base_state, **emotion_state}
        
        return combined_state
