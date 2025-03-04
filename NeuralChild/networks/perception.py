# perception.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import numpy as np
from collections import deque

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("PerceptionNetwork")

class PerceptualInput:
    """Represents a perceptual input that has been processed"""
    def __init__(
        self,
        source: str,
        content: str,
        modality: str = "verbal",
        salience: float = 0.5,
        timestamp: Optional[datetime] = None
    ):
        self.source = source
        self.content = content
        self.modality = modality  # verbal, visual, tactile, etc.
        self.salience = salience  # How attention-grabbing this input is
        self.timestamp = timestamp or datetime.now()
        self.processed = False
        self.extracted_features = {}  # Features extracted during processing

class PerceptionNetwork(BaseNetwork):
    """
    Interpreting input data network using CNN-like processing
    
    The perception network processes raw sensory information into meaningful
    percepts that can be further processed by other networks. It acts as the
    interface between external stimuli and internal representations.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.1,  # Low threshold for perception
        name: str = "Perception"
    ):
        """Initialize the perception network"""
        super().__init__(
            network_type=NetworkType.PERCEPTION,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Perception parameters
        self.sensory_acuity = 0.6  # How clearly sensory information is received
        self.pattern_recognition = 0.4  # Ability to recognize patterns
        self.perception_biases = {
            "novelty": 0.7,       # Bias toward novel stimuli
            "emotional": 0.6,     # Bias toward emotional content
            "social": 0.8         # Bias toward social stimuli (especially mother)
        }
        
        # Recent percepts storage
        self.recent_percepts = deque(maxlen=50)
        self.current_percepts = []
        
        # Feature extractors - improve with development
        self.feature_extractors = {
            "verbal": self._extract_verbal_features,
            "visual": self._extract_visual_features,
            "emotional": self._extract_emotional_features
        }
        
        # Perception history for training
        self.perception_history = {}
        
        logger.info(f"Initialized perception network with acuity {self.sensory_acuity:.2f}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process incoming sensory information into meaningful percepts"""
        # Reset current percepts
        self.current_percepts = []
        
        if not self.input_buffer:
            return {"network_activation": 0.0, "percepts": []}
        
        # Collect raw inputs
        raw_inputs = []
        contextual_data = {}
        
        for input_item in self.input_buffer:
            data = input_item["data"]
            source = input_item.get("source", "unknown")
            
            # Handle verbal inputs (like mother's speech)
            if "verbal" in data:
                content = data["verbal"]
                if isinstance(content, dict) and "text" in content:
                    content = content["text"]
                raw_inputs.append(PerceptualInput(
                    source=source,
                    content=content,
                    modality="verbal",
                    salience=0.7
                ))
            
            # Handle direct perceptual inputs
            if "percepts" in data:
                for percept in data["percepts"]:
                    if isinstance(percept, str):
                        raw_inputs.append(PerceptualInput(
                            source=source,
                            content=percept,
                            modality="direct",
                            salience=0.8
                        ))
                    elif isinstance(percept, dict):
                        raw_inputs.append(PerceptualInput(
                            source=source,
                            content=percept.get("content", ""),
                            modality=percept.get("modality", "direct"),
                            salience=percept.get("salience", 0.8)
                        ))
            
            # Handle visual inputs
            if "visual" in data:
                content = data["visual"]
                if isinstance(content, dict):
                    content = str(content)
                raw_inputs.append(PerceptualInput(
                    source=source,
                    content=content,
                    modality="visual",
                    salience=0.6
                ))
            
            # Collect emotional context
            if "emotional_state" in data:
                contextual_data["emotional_state"] = data["emotional_state"]
            
            # Collect attention state
            if "attention_focus" in data:
                contextual_data["attention_focus"] = data["attention_focus"]
        
        # Process each raw input
        processed_percepts = self._process_raw_inputs(raw_inputs, contextual_data)
        
        # Apply perceptual biases
        biased_percepts = self._apply_perceptual_biases(processed_percepts, contextual_data)
        
        # Update current percepts and history
        self.current_percepts = biased_percepts
        for percept in biased_percepts:
            self.recent_percepts.append(percept)
            
            # Update perception history
            content = percept.content
            if content not in self.perception_history:
                self.perception_history[content] = {
                    "first_seen": datetime.now(),
                    "count": 0,
                    "sources": set()
                }
            self.perception_history[content]["count"] += 1
            self.perception_history[content]["sources"].add(percept.source)
        
        # Calculate activation level based on percept salience and quantity
        if biased_percepts:
            total_salience = sum(p.salience for p in biased_percepts)
            avg_salience = total_salience / len(biased_percepts)
            # Activation increases with both quantity and quality of percepts
            activation = min(1.0, avg_salience * 0.7 + (len(biased_percepts) / 10) * 0.3)
        else:
            activation = 0.0
        
        # Extract percept strings for output
        percept_strings = [p.content for p in biased_percepts]
        
        # Clear input buffer
        self.input_buffer = []
        
        return {
            "network_activation": activation,
            "percepts": percept_strings,
            "modalities": {p.modality for p in biased_percepts}
        }
    
    def _process_raw_inputs(self, raw_inputs: List[PerceptualInput], context: Dict[str, Any]) -> List[PerceptualInput]:
        """Process raw inputs into meaningful percepts"""
        processed = []
        
        for input_item in raw_inputs:
            # Apply basic sensory acuity - chance to miss details based on development
            if random.random() > self.sensory_acuity:
                # Slightly alter or simplify the input if acuity is low
                if len(input_item.content) > 5:
                    simplified = input_item.content[:5] + "..."
                    input_item.content = simplified
            
            # Extract features based on modality
            if input_item.modality in self.feature_extractors:
                features = self.feature_extractors[input_item.modality](input_item.content, context)
                input_item.extracted_features = features
            
            # Mark as processed
            input_item.processed = True
            
            # For verbal inputs, break into multiple percepts (words/phrases)
            if input_item.modality == "verbal" and len(input_item.content.split()) > 1:
                # Split into individual word percepts
                for word in input_item.content.split():
                    if len(word) > 1:  # Skip very short words
                        word_percept = PerceptualInput(
                            source=input_item.source,
                            content=word.lower().strip(".,!?;:\"'()"),
                            modality="verbal_word",
                            salience=input_item.salience * 0.8  # Slightly lower salience for individual words
                        )
                        word_percept.processed = True
                        word_percept.extracted_features = {
                            "part_of": input_item.content,
                            "is_word": True
                        }
                        processed.append(word_percept)
                
                # Also keep the full phrase
                processed.append(input_item)
            else:
                processed.append(input_item)
        
        return processed
    
    def _apply_perceptual_biases(self, percepts: List[PerceptualInput], context: Dict[str, Any]) -> List[PerceptualInput]:
        """Apply perceptual biases to prioritize certain percepts"""
        # Get current attention focus if available
        attention_focus = context.get("attention_focus", [])
        
        # Apply biases to adjust salience
        for percept in percepts:
            # Novelty bias - increase salience for new percepts
            if percept.content in self.perception_history:
                # Familiarity reduces salience
                history = self.perception_history[percept.content]
                novelty_factor = max(0.2, 1.0 - (history["count"] / 20))
                percept.salience *= (1.0 - self.perception_biases["novelty"] * (1.0 - novelty_factor))
            else:
                # New percepts get salience boost
                percept.salience = min(1.0, percept.salience * (1.0 + self.perception_biases["novelty"] * 0.3))
            
            # Social bias - mother-related percepts get higher salience
            if percept.source == "mother" or "mother" in percept.content.lower():
                percept.salience = min(1.0, percept.salience * (1.0 + self.perception_biases["social"] * 0.4))
            
            # Emotional bias - emotionally relevant percepts get higher salience
            emotional_state = context.get("emotional_state", {})
            if emotional_state:
                for emotion, intensity in emotional_state.items():
                    if emotion.lower() in percept.content.lower():
                        percept.salience = min(1.0, percept.salience * (1.0 + intensity * self.perception_biases["emotional"]))
            
            # Attention bias - attended items get higher salience
            if percept.content in attention_focus:
                percept.salience = min(1.0, percept.salience * 1.5)
        
        # Sort by salience (highest first)
        sorted_percepts = sorted(percepts, key=lambda p: p.salience, reverse=True)
        
        # Limit to most salient percepts (attentional capacity)
        capacity = max(3, int(10 * self.state.training_progress))  # Capacity grows with development
        return sorted_percepts[:capacity]
    
    def _extract_verbal_features(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from verbal input"""
        words = content.split()
        features = {
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / max(1, len(words)),
            "contains_question": "?" in content,
            "keywords": [w.lower() for w in words if len(w) > 3][:5]  # Identify potential keywords
        }
        return features
    
    def _extract_visual_features(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from visual input"""
        # In a real system, this would process visual data
        # For this simplified version, we just extract basic text features
        features = {
            "description_length": len(content),
            "contains_colors": any(color in content.lower() for color in ["red", "blue", "green", "yellow", "black", "white"]),
            "contains_shapes": any(shape in content.lower() for shape in ["square", "circle", "triangle", "round"])
        }
        return features
    
    def _extract_emotional_features(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotional features from input"""
        emotional_words = {
            "joy": ["happy", "joy", "smile", "laugh", "excited", "fun"],
            "sadness": ["sad", "cry", "unhappy", "tears", "upset"],
            "anger": ["angry", "mad", "frustrated", "upset"],
            "fear": ["afraid", "scared", "fear", "frightened"],
            "surprise": ["surprised", "wow", "unexpected", "amazing"],
            "disgust": ["yuck", "gross", "disgusting", "ew"],
            "trust": ["trust", "safe", "secure", "good"],
            "anticipation": ["waiting", "soon", "expect", "anticipate"]
        }
        
        features = {"emotional_content": {}}
        
        for emotion, words in emotional_words.items():
            count = sum(1 for word in words if word in content.lower())
            if count > 0:
                features["emotional_content"][emotion] = min(1.0, count * 0.3)
        
        return features
    
    def update_development(self, age_days: float, vocabulary_size: int) -> None:
        """Update developmental parameters based on age and vocabulary"""
        # Sensory acuity improves with age
        self.sensory_acuity = min(0.95, 0.6 + (age_days / 250))
        
        # Pattern recognition improves with age and vocabulary
        vocab_factor = min(0.4, vocabulary_size / 1000)
        age_factor = min(0.4, age_days / 300)
        self.pattern_recognition = min(0.95, 0.4 + age_factor + vocab_factor)
        
        # Perception biases change with development
        # Novelty bias decreases with age (less distracted by novelty)
        self.perception_biases["novelty"] = max(0.3, 0.7 - (age_days / 400))
        
        # Social bias remains high but shifts slightly
        self.perception_biases["social"] = 0.8
        
        # Emotional bias becomes more balanced
        self.perception_biases["emotional"] = max(0.3, 0.6 - (age_days / 600))
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Extract current percepts
        percept_strings = [p.content for p in self.current_percepts]
        
        # Calculate perception metrics
        modality_counts = {}
        for p in self.current_percepts:
            if p.modality not in modality_counts:
                modality_counts[p.modality] = 0
            modality_counts[p.modality] += 1
        
        # Extract frequently perceived items
        frequent_percepts = []
        for content, info in sorted(
            self.perception_history.items(), 
            key=lambda x: x[1]["count"], 
            reverse=True
        )[:5]:
            frequent_percepts.append({
                "content": content,
                "count": info["count"],
                "sources": list(info["sources"])
            })
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "percepts": percept_strings,
            "sensory_acuity": self.sensory_acuity,
            "pattern_recognition": self.pattern_recognition,
            "modality_distribution": modality_counts,
            "frequent_percepts": frequent_percepts
        }
    
    def get_current_percepts(self) -> List[str]:
        """Get list of current percept strings"""
        return [p.content for p in self.current_percepts]
    
    def has_perceived(self, content: str) -> bool:
        """Check if something has been perceived before"""
        return content in self.perception_history