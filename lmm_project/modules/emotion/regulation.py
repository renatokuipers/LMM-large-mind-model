"""
Emotion Regulation

This component is responsible for modulating emotional responses,
providing mechanisms to adjust emotional intensity and expression
based on context and developmental capabilities.
"""

import logging
import uuid
import time
import math
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from collections import defaultdict

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.emotion.models import EmotionState, EmotionRegulationRequest, EmotionRegulationResult, EmotionNeuralState

# Initialize logger
logger = logging.getLogger(__name__)

class EmotionRegulator(BaseModule):
    """
    Regulates emotional states through various strategies
    
    This module develops from minimal regulation capability to
    sophisticated emotional control strategies based on context.
    """
    # Development milestones
    development_milestones = {
        0.0: "Minimal regulation capability",
        0.2: "Basic emotional suppression",
        0.4: "Attentional deployment strategies",
        0.6: "Cognitive reappraisal",
        0.8: "Context-sensitive regulation",
        1.0: "Sophisticated emotional regulation"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the emotion regulator
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="emotion_regulator",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize regulation strategies
        self._initialize_regulation_strategies()
        
        # Neural state for tracking activations and development
        self.neural_state = EmotionNeuralState()
        self.neural_state.regulation_development = development_level
        
        # Regulation parameters
        self.params = {
            # How much emotional intensity can be regulated (0-1)
            "regulation_capacity": 0.1,
            
            # How long the regulation effect lasts
            "regulation_duration": 60,  # seconds
            
            # How much cognitive resources regulation requires
            "cognitive_cost": 0.2,
            
            # Default regulation target
            "default_valence_target": 0.1,  # slightly positive
            "default_arousal_target": 0.3,  # moderate arousal
            
            # Whether regulation has side effects
            "side_effects_enabled": False
        }
        
        # Adjust parameters based on development level
        self._adjust_parameters_for_development()
        
        # History of regulation attempts
        self.regulation_history = []
        
        # Active regulation effects
        self.active_regulations = {}
        
        logger.info(f"Emotion regulator initialized at development level {development_level:.2f}")
    
    def _initialize_regulation_strategies(self):
        """Initialize available emotion regulation strategies"""
        # Define strategies based on psychological research
        self.strategies = {
            # Suppression - hide emotional expression
            # Available at early development (0.2+)
            "suppression": {
                "min_development": 0.2,
                "effectiveness": 0.4,  # moderately effective
                "side_effects": {
                    "cognitive_cost": 0.4,  # high cognitive cost
                    "rebound": 0.3      # causes some rebound effect
                },
                "description": "Masking or hiding emotional expression"
            },
            
            # Distraction - redirect attention away from emotion stimulus
            # Available at intermediate development (0.3+)
            "distraction": {
                "min_development": 0.3,
                "effectiveness": 0.5,
                "side_effects": {
                    "cognitive_cost": 0.2,
                    "rebound": 0.1
                },
                "description": "Redirecting attention away from emotional stimuli"
            },
            
            # Reappraisal - reconsider the meaning of the stimulus
            # Available at higher development (0.6+)
            "reappraisal": {
                "min_development": 0.6,
                "effectiveness": 0.7,
                "side_effects": {
                    "cognitive_cost": 0.3,
                    "rebound": 0.0
                },
                "description": "Reinterpreting the meaning of emotional stimuli"
            },
            
            # Acceptance - allow emotions to exist without judgment
            # Available at higher development (0.7+)
            "acceptance": {
                "min_development": 0.7,
                "effectiveness": 0.6,
                "side_effects": {
                    "cognitive_cost": 0.1,
                    "rebound": 0.0
                },
                "description": "Accepting emotions without judgment"
            },
            
            # Problem-solving - address the cause of the emotion
            # Available at higher development (0.8+)
            "problem_solving": {
                "min_development": 0.8,
                "effectiveness": 0.8,
                "side_effects": {
                    "cognitive_cost": 0.4,
                    "rebound": 0.0
                },
                "description": "Addressing the causes of emotional reactions"
            }
        }
    
    def _adjust_parameters_for_development(self):
        """Adjust regulation parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very limited regulation at early stages
            self.params.update({
                "regulation_capacity": 0.1,
                "cognitive_cost": 0.5,  # High cost for limited effect
                "side_effects_enabled": True,  # More side effects at low development
                "regulation_duration": 30  # Short duration
            })
        elif self.development_level < 0.4:
            # Basic regulation capabilities
            self.params.update({
                "regulation_capacity": 0.2,
                "cognitive_cost": 0.4,
                "side_effects_enabled": True,
                "regulation_duration": 60
            })
        elif self.development_level < 0.6:
            # Improved regulation
            self.params.update({
                "regulation_capacity": 0.4,
                "cognitive_cost": 0.3,
                "side_effects_enabled": True,
                "regulation_duration": 120
            })
        elif self.development_level < 0.8:
            # Advanced regulation
            self.params.update({
                "regulation_capacity": 0.6,
                "cognitive_cost": 0.2,
                "side_effects_enabled": False,
                "regulation_duration": 300
            })
        else:
            # Sophisticated regulation
            self.params.update({
                "regulation_capacity": 0.8,
                "cognitive_cost": 0.1,
                "side_effects_enabled": False,
                "regulation_duration": 600
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to regulate emotions
        
        Args:
            input_data: Data to process for regulation
                Required keys: 'current_state'
                Optional keys: 'target_valence', 'target_arousal', 'target_emotion',
                               'regulation_strategy', 'context'
                
        Returns:
            Dictionary with regulation results
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract the current emotional state
        if "current_state" not in input_data:
            return {
                "status": "error",
                "message": "No current emotional state provided",
                "process_id": process_id
            }
            
        current_state = input_data["current_state"]
        
        # Get regulation capacity (how much we can regulate)
        # This may be overridden by input data
        regulation_capacity = input_data.get(
            "regulation_capacity", 
            self.params["regulation_capacity"]
        )
        
        # Determine regulation approach
        target_valence = input_data.get("target_valence")
        target_arousal = input_data.get("target_arousal")
        target_emotion = input_data.get("target_emotion")
        specified_strategy = input_data.get("regulation_strategy")
        context = input_data.get("context", {})
        
        # If no specific targets provided, use defaults
        if target_valence is None and target_arousal is None and target_emotion is None:
            target_valence = self.params["default_valence_target"]
            target_arousal = self.params["default_arousal_target"]
        
        # Select regulation strategy if not specified
        strategy = specified_strategy or self._select_strategy(
            current_state, target_valence, target_arousal, target_emotion, context
        )
        
        # Apply regulation strategy
        regulation_result = self._apply_regulation(
            current_state, 
            strategy, 
            target_valence, 
            target_arousal, 
            target_emotion,
            regulation_capacity,
            context
        )
        
        # Add to history
        self.regulation_history.append({
            "original_state": current_state,
            "regulated_state": regulation_result["regulated_state"],
            "strategy": strategy,
            "success_level": regulation_result["success_level"],
            "timestamp": datetime.now().isoformat(),
            "process_id": process_id
        })
        
        # Limit history size
        if len(self.regulation_history) > 50:
            self.regulation_history = self.regulation_history[-50:]
        
        # Create result
        result = {
            "status": "success",
            "regulation_result": regulation_result,
            "process_id": process_id,
            "development_level": self.development_level
        }
        
        # Publish result if we have event bus
        if self.event_bus:
            self.publish_message(
                "emotion_regulation_result",
                {
                    "result": regulation_result,
                    "process_id": process_id
                }
            )
        
        return result
    
    def _select_strategy(
        self, 
        current_state: Any, 
        target_valence: Optional[float], 
        target_arousal: Optional[float],
        target_emotion: Optional[str],
        context: Dict[str, Any]
    ) -> str:
        """
        Select appropriate regulation strategy
        
        Args:
            current_state: Current emotional state
            target_valence: Target valence (if any)
            target_arousal: Target arousal (if any)
            target_emotion: Target emotion (if any)
            context: Contextual information
            
        Returns:
            Selected strategy name
        """
        # Get available strategies based on development level
        available_strategies = [
            name for name, info in self.strategies.items()
            if self.development_level >= info["min_development"]
        ]
        
        # At early development, just use whatever's available
        if self.development_level < 0.4 or not available_strategies:
            return available_strategies[0] if available_strategies else "suppression"
        
        # At higher development, use more sophisticated selection
        if self.development_level >= 0.7:
            # Consider context and specific regulation goals
            
            # If we need to increase positive valence
            if target_valence is not None and target_valence > current_state.valence:
                if "reappraisal" in available_strategies:
                    return "reappraisal"
                elif "problem_solving" in available_strategies:
                    return "problem_solving"
            
            # If we need to decrease negative valence
            elif target_valence is not None and target_valence < current_state.valence:
                if current_state.valence < -0.5 and "acceptance" in available_strategies:
                    return "acceptance"
                elif "distraction" in available_strategies:
                    return "distraction"
            
            # If we need to decrease arousal
            if target_arousal is not None and target_arousal < current_state.arousal:
                if "acceptance" in available_strategies:
                    return "acceptance"
                elif "distraction" in available_strategies:
                    return "distraction"
            
            # If we need to increase arousal
            elif target_arousal is not None and target_arousal > current_state.arousal:
                if "problem_solving" in available_strategies:
                    return "problem_solving"
                
            # Default to most effective available strategy
            effectiveness_sorted = sorted(
                [(s, self.strategies[s]["effectiveness"]) for s in available_strategies],
                key=lambda x: x[1],
                reverse=True
            )
            return effectiveness_sorted[0][0]
        
        # Middle development - use simpler selection
        else:
            # Just select randomly from available, weighted by effectiveness
            weights = [self.strategies[s]["effectiveness"] for s in available_strategies]
            total = sum(weights)
            normalized_weights = [w / total for w in weights]
            
            # Random selection with weights
            return random.choices(available_strategies, weights=normalized_weights, k=1)[0]
    
    def _apply_regulation(
        self,
        current_state: Any,
        strategy: str,
        target_valence: Optional[float],
        target_arousal: Optional[float],
        target_emotion: Optional[str],
        regulation_capacity: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply regulation strategy to emotional state
        
        Args:
            current_state: Current emotional state
            strategy: Selected regulation strategy
            target_valence: Target valence (if any)
            target_arousal: Target arousal (if any) 
            target_emotion: Target emotion (if any)
            regulation_capacity: Regulation strength
            context: Contextual information
            
        Returns:
            Dictionary with regulation results
        """
        # Get strategy info
        strategy_info = self.strategies.get(strategy, self.strategies["suppression"])
        
        # Effectiveness is based on strategy and development
        effectiveness = strategy_info["effectiveness"] * (0.5 + 0.5 * self.development_level)
        
        # Calculate total regulation strength
        regulation_strength = regulation_capacity * effectiveness
        
        # Create new state values
        new_valence = current_state.valence
        new_arousal = current_state.arousal
        new_dominant_emotion = current_state.dominant_emotion
        new_emotion_intensities = dict(current_state.emotion_intensities)
        
        # Regulate valence if target provided
        if target_valence is not None:
            # Move current valence toward target
            valence_diff = target_valence - current_state.valence
            valence_change = valence_diff * regulation_strength
            new_valence = current_state.valence + valence_change
            
        # Regulate arousal if target provided
        if target_arousal is not None:
            # Move current arousal toward target
            arousal_diff = target_arousal - current_state.arousal
            arousal_change = arousal_diff * regulation_strength
            new_arousal = current_state.arousal + arousal_change
            
        # Ensure values are in valid ranges
        new_valence = max(-1.0, min(1.0, new_valence))
        new_arousal = max(0.0, min(1.0, new_arousal))
        
        # Regulate specific emotion if target provided
        if target_emotion is not None and target_emotion in new_emotion_intensities:
            # Increase target emotion intensity
            current_intensity = new_emotion_intensities[target_emotion]
            new_intensity = current_intensity + (1.0 - current_intensity) * regulation_strength
            new_emotion_intensities[target_emotion] = new_intensity
            
            # Decrease other emotions proportionally
            intensity_increase = new_intensity - current_intensity
            other_emotions = [e for e in new_emotion_intensities if e != target_emotion]
            
            if other_emotions:
                for emotion in other_emotions:
                    new_emotion_intensities[emotion] = max(
                        0.0, 
                        new_emotion_intensities[emotion] - (intensity_increase / len(other_emotions))
                    )
                
                # Normalize intensities to sum to 1.0
                total_intensity = sum(new_emotion_intensities.values())
                new_emotion_intensities = {
                    e: i / total_intensity for e, i in new_emotion_intensities.items()
                }
                
            # Update dominant emotion if target is now strongest
            if target_emotion != new_dominant_emotion and new_emotion_intensities[target_emotion] > new_emotion_intensities.get(new_dominant_emotion, 0):
                new_dominant_emotion = target_emotion
        
        # Create new emotional state
        regulated_state = EmotionState(
            valence=new_valence,
            arousal=new_arousal,
            dominant_emotion=new_dominant_emotion,
            emotion_intensities=new_emotion_intensities,
            timestamp=datetime.now()
        )
        
        # Calculate regulation success
        if target_valence is not None and target_arousal is not None:
            # Calculate Euclidean distance in VA space from target
            original_distance = math.sqrt(
                (current_state.valence - target_valence) ** 2 +
                (current_state.arousal - target_arousal) ** 2
            )
            
            new_distance = math.sqrt(
                (regulated_state.valence - target_valence) ** 2 +
                (regulated_state.arousal - target_arousal) ** 2
            )
            
            # Success is proportional to distance reduction
            if original_distance > 0:
                success_level = min(1.0, max(0.0, (original_distance - new_distance) / original_distance))
            else:
                success_level = 1.0  # Already at target
                
        elif target_valence is not None:
            # Only valence matters
            original_distance = abs(current_state.valence - target_valence)
            new_distance = abs(regulated_state.valence - target_valence)
            
            if original_distance > 0:
                success_level = min(1.0, max(0.0, (original_distance - new_distance) / original_distance))
            else:
                success_level = 1.0
                
        elif target_arousal is not None:
            # Only arousal matters
            original_distance = abs(current_state.arousal - target_arousal)
            new_distance = abs(regulated_state.arousal - target_arousal)
            
            if original_distance > 0:
                success_level = min(1.0, max(0.0, (original_distance - new_distance) / original_distance))
            else:
                success_level = 1.0
                
        elif target_emotion is not None:
            # Emotion category matters
            original_intensity = current_state.emotion_intensities.get(target_emotion, 0.0)
            new_intensity = regulated_state.emotion_intensities.get(target_emotion, 0.0)
            
            success_level = min(1.0, max(0.0, (new_intensity - original_intensity)))
            
        else:
            # No specific target, success is based on regulation strength
            success_level = regulation_strength
        
        # Record activation for tracking
        if hasattr(self, "neural_state"):
            self.neural_state.add_activation('regulation', {
                'strategy': strategy,
                'effectiveness': effectiveness,
                'regulation_strength': regulation_strength,
                'original_valence': current_state.valence,
                'original_arousal': current_state.arousal,
                'new_valence': regulated_state.valence,
                'new_arousal': regulated_state.arousal,
                'success_level': success_level
            })
            
        # Create regulation result
        result = {
            "original_state": current_state,
            "regulated_state": regulated_state,
            "regulation_strategy": strategy,
            "success_level": success_level,
            "process_id": str(uuid.uuid4())
        }
        
        return result
    
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
        
        # Update neural state
        if hasattr(self, "neural_state"):
            self.neural_state.regulation_development = new_level
            self.neural_state.last_updated = datetime.now()
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state
        
        Returns:
            Dictionary with current state
        """
        base_state = super().get_state()
        
        # Add regulator-specific state
        regulator_state = {
            "params": self.params,
            "available_strategies": [
                name for name, info in self.strategies.items()
                if self.development_level >= info["min_development"]
            ],
            "history_length": len(self.regulation_history),
            "last_regulation": self.regulation_history[-1] if self.regulation_history else None
        }
        
        # Combine states
        combined_state = {**base_state, **regulator_state}
        
        return combined_state 
