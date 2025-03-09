import numpy as np
import torch
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import logging
import os

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.learning.models import AssociativeLearningEvent

logger = logging.getLogger(__name__)

class AssociativeLearning(BaseModule):
    """
    Learns relationships between stimuli and events
    
    This module detects correlations, forms associative links,
    strengthens connections through experience, and applies
    associations to predict outcomes.
    """
    
    # Development milestones for associative learning
    development_milestones = {
        0.0: "Simple stimulus-response associations",
        0.2: "Multiple associations per stimulus",
        0.4: "Temporal sequence learning",
        0.6: "Context-dependent associations",
        0.8: "Advanced statistical correlation detection",
        1.0: "Abstract relational learning"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the associative learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id, 
            module_type="associative_learning", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Stimulus-response associations (stimulus -> list of responses)
        self.associations = {}
        
        # Association strengths (stimulus-response pair -> strength)
        self.association_strengths = {}
        
        # Temporal sequence tracking
        self.recent_stimuli = []
        self.max_sequence_length = 3  # Will increase with development
        
        # Co-occurrence matrix for statistical learning
        self.co_occurrence = {}
        
        # Adjust capabilities based on developmental level
        self._adjust_for_development()
        
        # Subscribe to perception events to create associations
        if self.event_bus:
            self.subscribe_to_message("perception_input", self._handle_perception_input)
            self.subscribe_to_message("learning_reinforce", self._handle_reinforcement)
    
    def _adjust_for_development(self):
        """Adjust capabilities based on current developmental level"""
        # Sequence length increases with development
        self.max_sequence_length = max(2, int(3 + (self.development_level * 7)))
        
        # Activation threshold decreases with development (becomes more sensitive)
        self.activation_threshold = max(0.3, 0.7 - (self.development_level * 0.4))
        
        # Statistical learning complexity increases with development
        self.statistical_complexity = self.development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to learn associations
        
        Args:
            input_data: Dictionary containing stimuli and events for association
            
        Returns:
            Dictionary with the learned associations and predictions
        """
        operation = input_data.get("operation", "learn")
        
        if operation == "learn":
            return self._learn_association(input_data)
        elif operation == "predict":
            return self._predict_from_stimulus(input_data)
        elif operation == "reinforce":
            return self._reinforce_association(input_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "module_id": self.module_id
            }
    
    def _learn_association(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a new association between stimulus and response"""
        stimulus = input_data.get("stimulus")
        response = input_data.get("response")
        
        if not stimulus or not response:
            return {"status": "error", "message": "Missing stimulus or response"}
        
        # Calculate initial association strength based on development
        base_strength = 0.3 + (self.development_level * 0.2)
        strength = input_data.get("strength", base_strength)
        
        # Create the association
        if stimulus not in self.associations:
            self.associations[stimulus] = []
        
        # Add if not already present
        if response not in self.associations[stimulus]:
            self.associations[stimulus].append(response)
        
        # Set or update association strength
        pair_key = f"{stimulus}|{response}"
        self.association_strengths[pair_key] = strength
        
        # Update co-occurrence matrix for statistical learning
        if self.development_level >= 0.3:  # Only with sufficient development
            self._update_co_occurrence(stimulus, response)
        
        # Create learning event
        event = AssociativeLearningEvent(
            source=input_data.get("source", "experience"),
            content=f"Association between '{stimulus}' and '{response}'",
            stimulus=stimulus,
            response=response,
            association_strength=strength,
            conditioning_type=input_data.get("conditioning_type", "classical"),
            temporal_delay=input_data.get("delay", 0.0),
            developmental_level=self.development_level
        )
        
        # Add to recent stimuli for sequence learning
        self._update_recent_stimuli(stimulus)
        
        return {
            "status": "success",
            "association_id": pair_key,
            "stimulus": stimulus,
            "response": response,
            "strength": strength,
            "learning_event_id": event.id
        }
    
    def _predict_from_stimulus(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict response based on stimulus"""
        stimulus = input_data.get("stimulus")
        
        if not stimulus:
            return {"status": "error", "message": "Missing stimulus"}
        
        if stimulus not in self.associations:
            return {
                "status": "not_found",
                "message": f"No associations found for stimulus: {stimulus}"
            }
        
        # Get all responses and their strengths
        responses = self.associations[stimulus]
        prediction_threshold = input_data.get("threshold", self.activation_threshold)
        
        # Calculate response probabilities based on association strengths
        predictions = []
        for response in responses:
            pair_key = f"{stimulus}|{response}"
            strength = self.association_strengths.get(pair_key, 0.0)
            
            if strength >= prediction_threshold:
                predictions.append({
                    "response": response,
                    "confidence": strength,
                    "association_id": pair_key
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Update recent stimuli for sequence learning
        self._update_recent_stimuli(stimulus)
        
        # Add sequence prediction if we have enough development
        if self.development_level >= 0.4 and len(self.recent_stimuli) >= 2:
            sequence_predictions = self._predict_from_sequence()
            if sequence_predictions:
                return {
                    "status": "success",
                    "predictions": predictions,
                    "sequence_predictions": sequence_predictions,
                    "stimulus": stimulus
                }
        
        return {
            "status": "success" if predictions else "no_predictions",
            "predictions": predictions,
            "stimulus": stimulus
        }
    
    def _reinforce_association(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reinforce or weaken an existing association"""
        stimulus = input_data.get("stimulus")
        response = input_data.get("response")
        
        if not stimulus or not response:
            return {"status": "error", "message": "Missing stimulus or response"}
        
        pair_key = f"{stimulus}|{response}"
        if pair_key not in self.association_strengths:
            return {
                "status": "not_found",
                "message": f"Association not found: {stimulus}->{response}"
            }
        
        # Get reinforcement amount (positive = strengthen, negative = weaken)
        amount = input_data.get("amount", 0.1)
        
        # Update strength
        current_strength = self.association_strengths[pair_key]
        new_strength = max(0.0, min(1.0, current_strength + amount))
        self.association_strengths[pair_key] = new_strength
        
        # If strength drops to zero, remove the association
        if new_strength <= 0.0:
            if response in self.associations[stimulus]:
                self.associations[stimulus].remove(response)
            if not self.associations[stimulus]:
                del self.associations[stimulus]
            del self.association_strengths[pair_key]
        
        return {
            "status": "success",
            "association_id": pair_key,
            "previous_strength": current_strength,
            "new_strength": new_strength,
            "change": amount
        }
    
    def _update_recent_stimuli(self, stimulus: str):
        """Update the list of recent stimuli for sequence learning"""
        self.recent_stimuli.append(stimulus)
        if len(self.recent_stimuli) > self.max_sequence_length:
            self.recent_stimuli.pop(0)
    
    def _predict_from_sequence(self) -> List[Dict[str, Any]]:
        """Predict next stimulus based on recent sequence"""
        if len(self.recent_stimuli) < 2:
            return []
        
        # Create sequence key
        sequence = "|".join(self.recent_stimuli)
        
        # Check if this sequence exists in associations
        if sequence not in self.associations:
            return []
        
        # Return predictions
        predictions = []
        for response in self.associations[sequence]:
            pair_key = f"{sequence}|{response}"
            strength = self.association_strengths.get(pair_key, 0.0)
            
            if strength >= self.activation_threshold:
                predictions.append({
                    "response": response,
                    "confidence": strength,
                    "sequence": self.recent_stimuli.copy(),
                    "association_id": pair_key
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions
    
    def _update_co_occurrence(self, stimulus: str, response: str):
        """Update co-occurrence matrix for statistical learning"""
        if stimulus not in self.co_occurrence:
            self.co_occurrence[stimulus] = {}
        
        if response not in self.co_occurrence[stimulus]:
            self.co_occurrence[stimulus][response] = 0
            
        self.co_occurrence[stimulus][response] += 1
    
    def _handle_perception_input(self, message):
        """Handle perception input events for automatic association learning"""
        if not message.content:
            return
            
        # Extract perception data
        perception_data = message.content
        
        # Only process if we have both current and previous perceptions
        if "current" in perception_data and "previous" in perception_data:
            stimulus = perception_data["previous"].get("pattern", "")
            response = perception_data["current"].get("pattern", "")
            
            if stimulus and response:
                # Automatically learn the association
                self._learn_association({
                    "stimulus": stimulus,
                    "response": response,
                    "source": "perception",
                    "delay": perception_data.get("time_delta", 0.0)
                })
    
    def _handle_reinforcement(self, message):
        """Handle reinforcement events"""
        if not message.content:
            return
            
        reinforcement_data = message.content
        
        if "stimulus" in reinforcement_data and "response" in reinforcement_data:
            self._reinforce_association(reinforcement_data)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # If development level changed significantly, adjust capabilities
        if abs(new_level - previous_level) >= 0.05:
            self._adjust_for_development()
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        base_state = super().get_state()
        
        # Add associative learning specific state
        module_state = {
            "association_count": sum(len(responses) for responses in self.associations.values()),
            "unique_stimuli": len(self.associations),
            "sequence_capacity": self.max_sequence_length,
            "activation_threshold": self.activation_threshold,
            "statistical_complexity": self.statistical_complexity
        }
        
        base_state.update(module_state)
        return base_state
