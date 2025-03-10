"""
Emotional Regulation Module

This module implements emotional regulation capabilities, which develop
from basic emotion detection in early stages to sophisticated regulation
strategies in later developmental stages.
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import os
import json
from pathlib import Path

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.self_regulation.models import EmotionalState, RegulationStrategy
from lmm_project.modules.self_regulation.neural_net import EmotionalRegulationNetwork

class EmotionalRegulation(BaseModule):
    """
    Handles emotional regulation
    
    This module implements mechanisms for monitoring and regulating emotions,
    developing from basic awareness to sophisticated cognitive strategies.
    """
    
    # Developmental milestones for emotional regulation
    development_milestones = {
        0.1: "Basic emotion detection",
        0.2: "Primitive regulation through external support",
        0.3: "Simple distraction-based regulation",
        0.4: "Basic suppression strategies",
        0.5: "Beginning of cognitive reappraisal",
        0.6: "Situation selection and modification",
        0.7: "Attention deployment strategies",
        0.8: "Advanced cognitive reappraisal",
        0.9: "Integrated regulation strategies",
        1.0: "Adaptive, flexible regulation"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the emotional regulation module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id,
            module_type="emotional_regulation",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.self_regulation.emotional_regulation.{module_id}")
        
        # Initialize neural network
        self.network = EmotionalRegulationNetwork(
            developmental_level=development_level
        )
        
        # Available regulation strategies
        self.strategies = {
            0: RegulationStrategy(
                name="distraction",
                description="Directing attention away from emotional stimuli",
                strategy_type="emotional",
                effectiveness=0.5,
                complexity=0.3,
                min_development_level=0.2,
                applicable_situations={"negative_emotions", "boredom", "anxiety"}
            ),
            1: RegulationStrategy(
                name="reappraisal",
                description="Reinterpreting a situation to change its emotional impact",
                strategy_type="emotional",
                effectiveness=0.8,
                complexity=0.7,
                min_development_level=0.6,
                applicable_situations={"negative_emotions", "anxiety", "anger"}
            ),
            2: RegulationStrategy(
                name="suppression",
                description="Inhibiting emotional expression",
                strategy_type="emotional",
                effectiveness=0.4,
                complexity=0.5,
                min_development_level=0.4,
                applicable_situations={"intense_emotions", "social_situations"}
            ),
            3: RegulationStrategy(
                name="situation_modification",
                description="Changing aspects of the situation to alter emotions",
                strategy_type="emotional",
                effectiveness=0.7,
                complexity=0.6,
                min_development_level=0.5,
                applicable_situations={"conflict", "stress", "interpersonal"}
            ),
            4: RegulationStrategy(
                name="attention_deployment",
                description="Focusing attention on specific aspects of a situation",
                strategy_type="emotional",
                effectiveness=0.6,
                complexity=0.4,
                min_development_level=0.3,
                applicable_situations={"complex_situations", "mixed_emotions"}
            ),
            5: RegulationStrategy(
                name="response_modulation",
                description="Modifying physiological or behavioral responses",
                strategy_type="emotional",
                effectiveness=0.5,
                complexity=0.5,
                min_development_level=0.4,
                applicable_situations={"physical_reactions", "stress"}
            ),
            6: RegulationStrategy(
                name="cognitive_change",
                description="Changing thoughts about a situation",
                strategy_type="emotional",
                effectiveness=0.7,
                complexity=0.8,
                min_development_level=0.7,
                applicable_situations={"complex_emotions", "depression", "anxiety"}
            ),
            7: RegulationStrategy(
                name="acceptance",
                description="Accepting emotions without trying to change them",
                strategy_type="emotional",
                effectiveness=0.6,
                complexity=0.6,
                min_development_level=0.5,
                applicable_situations={"grief", "unchangeable_situations"}
            )
        }
        
        # Recent emotional states
        self.recent_emotions: List[EmotionalState] = []
        self.max_emotion_history = 20
        
        # Regulation history
        self.regulation_attempts = 0
        self.successful_regulations = 0
        
        # Last emotion that was regulated
        self.current_emotion: Optional[EmotionalState] = None
        
        # Subscribe to relevant messages
        if event_bus:
            self.subscribe_to_message("emotional_state")
            self.subscribe_to_message("regulation_request")
            self.subscribe_to_message("emotion_query")
        
        self.logger.info(f"Emotional regulation module initialized at development level {development_level:.2f}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to regulate emotions
        
        Args:
            input_data: Dictionary with input data
                Required keys depend on input type:
                - "type": Type of input ("emotion", "regulation_request", "query")
                - For emotion input: "emotion_type", "intensity", "valence", "arousal"
                - For regulation request: "emotion_id"
                - For query: "query_type"
            
        Returns:
            Dictionary with process results
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input")
        
        result = {
            "success": False,
            "message": f"Unknown input type: {input_type}"
        }
        
        # Process different input types
        if input_type == "emotion":
            result = self._process_emotion(input_data)
            
        elif input_type == "regulation_request":
            result = self._process_regulation_request(input_data)
            
        elif input_type == "query":
            result = self._process_query(input_data)
            
        return result
    
    def _process_emotion(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an emotional state"""
        # Create an emotional state from the data
        try:
            # Extract required fields
            emotion_type = emotion_data.get("emotion_type")
            if not emotion_type:
                return {"success": False, "message": "Missing emotion_type"}
                
            intensity = emotion_data.get("intensity", 0.5)
            valence = emotion_data.get("valence", 0.0)
            arousal = emotion_data.get("arousal", 0.5)
            
            # Create the emotional state
            emotion = EmotionalState(
                emotion_type=emotion_type,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                trigger=emotion_data.get("trigger"),
                context=emotion_data.get("context", {})
            )
            
            # Add to recent emotions
            self.recent_emotions.append(emotion)
            if len(self.recent_emotions) > self.max_emotion_history:
                self.recent_emotions = self.recent_emotions[-self.max_emotion_history:]
                
            # Set as current emotion
            self.current_emotion = emotion
            
            # Automatically regulate if intensity is high enough
            should_regulate = intensity > 0.7 or abs(valence) > 0.7
            
            if should_regulate and self.development_level >= 0.3:
                regulation_result = self._regulate_emotion(emotion)
                return {
                    "success": True,
                    "emotion_id": emotion.id,
                    "emotion": emotion.dict(),
                    "was_regulated": True,
                    "regulation_result": regulation_result
                }
            else:
                return {
                    "success": True,
                    "emotion_id": emotion.id,
                    "emotion": emotion.dict(),
                    "was_regulated": False
                }
                
        except Exception as e:
            self.logger.error(f"Error processing emotion: {e}")
            return {"success": False, "message": f"Error processing emotion: {str(e)}"}
    
    def _process_regulation_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request to regulate an emotion"""
        emotion_id = request_data.get("emotion_id")
        
        # Find the emotion
        target_emotion = None
        for emotion in self.recent_emotions:
            if emotion.id == emotion_id:
                target_emotion = emotion
                break
                
        if not target_emotion:
            return {
                "success": False,
                "message": f"Emotion with ID {emotion_id} not found"
            }
            
        # Regulate the emotion
        regulation_result = self._regulate_emotion(target_emotion)
        
        return {
            "success": True,
            "emotion_id": emotion_id,
            "regulation_result": regulation_result
        }
    
    def _process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query about emotional regulation"""
        query_type = query_data.get("query_type", "current")
        
        if query_type == "current":
            if self.current_emotion:
                return {
                    "success": True,
                    "current_emotion": self.current_emotion.dict()
                }
            else:
                return {
                    "success": True,
                    "current_emotion": None
                }
                
        elif query_type == "recent":
            limit = query_data.get("limit", 5)
            recent = self.recent_emotions[-limit:] if self.recent_emotions else []
            return {
                "success": True,
                "recent_emotions": [e.dict() for e in recent]
            }
            
        elif query_type == "strategies":
            # Return available strategies based on development
            available = []
            for idx, strategy in self.strategies.items():
                if strategy.min_development_level <= self.development_level:
                    available.append(strategy.dict())
                    
            return {
                "success": True,
                "available_strategies": available
            }
            
        elif query_type == "stats":
            # Return regulation statistics
            success_rate = 0.0
            if self.regulation_attempts > 0:
                success_rate = self.successful_regulations / self.regulation_attempts
                
            return {
                "success": True,
                "regulation_attempts": self.regulation_attempts,
                "successful_regulations": self.successful_regulations,
                "success_rate": success_rate,
                "development_level": self.development_level
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown query type: {query_type}"
            }
    
    def _regulate_emotion(self, emotion: EmotionalState) -> Dict[str, Any]:
        """
        Apply regulation to an emotion
        
        Args:
            emotion: The emotional state to regulate
            
        Returns:
            Dictionary with regulation results
        """
        # Increment regulation attempt counter
        self.regulation_attempts += 1
        
        # Convert emotion to format expected by neural network
        emotion_data = {
            "emotion_type": emotion.emotion_type,
            "intensity": emotion.intensity,
            "valence": emotion.valence,
            "arousal": emotion.arousal
        }
        
        # Get regulation strategy from neural network
        try:
            # Prepare emotion vector for neural network
            emotion_vector = self._create_emotion_vector(emotion_data)
            
            # Select regulation strategy
            strategy_idx, effectiveness = self.network.select_strategy(emotion_vector)
            strategy = self.strategies.get(strategy_idx)
            
            if not strategy:
                return {
                    "success": False,
                    "message": f"Invalid strategy index: {strategy_idx}"
                }
                
            # Check if the developmental level allows this strategy
            if strategy.min_development_level > self.development_level:
                # Fall back to a simpler strategy if this one is too advanced
                for idx, fallback in self.strategies.items():
                    if fallback.min_development_level <= self.development_level:
                        strategy = fallback
                        # Recalculate effectiveness based on fallback strategy
                        effectiveness = max(0.1, 0.5 - (self.development_level - fallback.min_development_level) * 0.5)
                        break
            
            # Calculate regulation effect on emotion
            # Intensity reduction depends on strategy effectiveness and emotion intensity
            intensity_reduction = effectiveness * 0.7
            regulated_intensity = max(0.1, emotion.intensity - intensity_reduction)
            
            # Calculate regulation success as percentage of intensity reduction
            if emotion.intensity > 0:
                regulation_success = (emotion.intensity - regulated_intensity) / emotion.intensity
            else:
                regulation_success = 0.0
                
            # Update emotion with regulation info
            emotion.is_regulated = True
            emotion.regulation_strategy = strategy.name
            emotion.regulation_success = regulation_success
            
            # Update success counter if regulation was effective
            if regulation_success > 0.3:
                self.successful_regulations += 1
                
            # Return regulation results
            result = {
                "success": True,
                "strategy": strategy.dict(),
                "original_intensity": emotion.intensity,
                "regulated_intensity": regulated_intensity,
                "regulation_success": regulation_success,
                "effectiveness": effectiveness
            }
            
            # Publish regulation event if significant effect
            if self.event_bus and regulation_success > 0.2:
                self.publish_message("emotion_regulated", {
                    "emotion": emotion.dict(),
                    "regulation_result": result
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in emotion regulation: {e}")
            return {
                "success": False,
                "message": f"Regulation error: {str(e)}"
            }
    
    def _create_emotion_vector(self, emotion_data: Dict[str, Any]) -> np.ndarray:
        """
        Create a vector representation of an emotion
        
        Args:
            emotion_data: Dictionary with emotion data
            
        Returns:
            Numpy array representation of the emotion
        """
        # Extract emotion attributes
        emotion_type = emotion_data.get("emotion_type", "neutral")
        intensity = emotion_data.get("intensity", 0.5)
        valence = emotion_data.get("valence", 0.0)
        arousal = emotion_data.get("arousal", 0.5)
        
        # One-hot encode emotion type (simplified)
        emotion_types = ["anger", "fear", "joy", "sadness", "disgust", "surprise", "neutral"]
        emotion_idx = emotion_types.index(emotion_type) if emotion_type in emotion_types else 6
        emotion_onehot = [0] * len(emotion_types)
        emotion_onehot[emotion_idx] = 1
        
        # Combine features
        features = emotion_onehot + [intensity, valence, arousal, self.development_level]
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "emotional_state":
            # Process incoming emotional state
            result = self.process_input({
                "type": "emotion",
                **content
            })
            
            # Send response if requested
            if message.reply_to and self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="emotional_state_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "regulation_request":
            # Process regulation request
            result = self.process_input({
                "type": "regulation_request",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="regulation_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "emotion_query":
            # Process query
            result = self.process_input({
                "type": "query",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="emotion_query_response",
                    content=result,
                    reply_to=message.id
                ))
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        super().update_development(amount)
        
        # Update neural network development
        self.network.update_developmental_level(self.development_level)
        
        # Check for developmental milestones
        self._check_development_milestones(old_level)
        
        self.logger.info(f"Updated emotional regulation development to {self.development_level:.2f}")
        return self.development_level
    
    def _check_development_milestones(self, previous_level: float) -> None:
        """
        Check if any developmental milestones have been reached
        
        Args:
            previous_level: The previous development level
        """
        # Check each milestone to see if we've crossed the threshold
        for level, description in self.development_milestones.items():
            # If we've crossed a milestone threshold
            if previous_level < level <= self.development_level:
                self.logger.info(f"Emotional regulation milestone reached at {level:.1f}: {description}")
                
                # Adjust regulation strategies based on the new milestone
                if level == 0.1:
                    self.logger.info("Now capable of basic emotion detection")
                elif level == 0.2:
                    self.logger.info("Now capable of seeking external support for regulation")
                elif level == 0.3:
                    self.logger.info("Now capable of using simple distraction techniques")
                elif level == 0.4:
                    self.logger.info("Now capable of basic emotion suppression")
                elif level == 0.5:
                    self.logger.info("Now capable of beginning cognitive reappraisal")
                elif level == 0.6:
                    self.logger.info("Now capable of situation selection and modification")
                elif level == 0.7:
                    self.logger.info("Now capable of attention deployment strategies")
                elif level == 0.8:
                    self.logger.info("Now capable of advanced cognitive reappraisal")
                elif level == 0.9:
                    self.logger.info("Now capable of integrated regulation strategies")
                elif level == 1.0:
                    self.logger.info("Now capable of adaptive, flexible emotional regulation")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with the module state
        """
        base_state = super().get_state()
        
        # Add emotional regulation specific state
        regulation_state = {
            "current_emotion": self.current_emotion.dict() if self.current_emotion else None,
            "recent_emotions_count": len(self.recent_emotions),
            "regulation_attempts": self.regulation_attempts,
            "successful_regulations": self.successful_regulations,
            "available_strategies": [
                s.name for s in self.strategies.values() 
                if s.min_development_level <= self.development_level
            ]
        }
        
        return {**base_state, **regulation_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save basic module state
        state_path = os.path.join(module_dir, "module_state.json")
        with open(state_path, 'w') as f:
            # Get state with serializable emotion data
            state = self.get_state()
            state["recent_emotions"] = [e.dict() for e in self.recent_emotions[-10:]] # Save last 10
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Saved emotional regulation state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update base state
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Update emotional regulation specific state
            self.regulation_attempts = state.get("regulation_attempts", 0)
            self.successful_regulations = state.get("successful_regulations", 0)
            
            # Recreate current emotion
            from lmm_project.modules.self_regulation.models import EmotionalState
            current_emotion_data = state.get("current_emotion")
            if current_emotion_data:
                self.current_emotion = EmotionalState(**current_emotion_data)
                
            # Recreate recent emotions
            self.recent_emotions = []
            for emotion_data in state.get("recent_emotions", []):
                try:
                    self.recent_emotions.append(EmotionalState(**emotion_data))
                except Exception as e:
                    self.logger.warning(f"Could not recreate emotion: {e}")
            
            # Update neural network
            self.network.update_developmental_level(self.development_level)
            
            self.logger.info(f"Loaded emotional regulation state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False
