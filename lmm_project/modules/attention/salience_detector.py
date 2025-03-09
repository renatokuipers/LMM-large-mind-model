from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.attention.models import SalienceScore, AttentionParameters

class SalienceDetector(BaseModule):
    """
    Detects salient aspects of inputs.
    
    The salience detector evaluates inputs to determine what aspects
    are important, novel, emotionally significant, or otherwise worthy
    of attention. It's the first stage of the attention process, identifying
    candidates for focus.
    """
    # Parameters controlling salience detection
    parameters: AttentionParameters = Field(default_factory=AttentionParameters)
    # History of previously seen inputs (for novelty detection)
    input_history: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Maximum history size
    max_history_size: int = Field(default=1000)
    # Last calculated salience scores
    last_salience_scores: Dict[str, SalienceScore] = Field(default_factory=dict)
    # Emotion state influence
    emotion_state: Dict[str, float] = Field(default_factory=dict)
    # Current goals influence
    current_goals: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, **data):
        """Initialize salience detector module"""
        super().__init__(
            module_id=module_id,
            module_type="salience_detector",
            event_bus=event_bus,
            **data
        )
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("perception_input", self._handle_perception_input)
            self.subscribe_to_message("emotion_update", self._handle_emotion_update)
            self.subscribe_to_message("goal_update", self._handle_goal_update)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to detect salience
        
        Parameters:
        input_data: Dictionary containing operation data
            - operation: The operation to perform
            - inputs: Dictionary of input items to evaluate
            - Additional parameters specific to the operation
            
        Returns:
        Operation result with salience scores
        """
        operation = input_data.get("operation", "")
        
        if operation == "detect_salience":
            inputs = input_data.get("inputs", {})
            context = input_data.get("context", {})
            return self.detect_salience(inputs, context)
            
        elif operation == "get_last_scores":
            return {
                "status": "success",
                "salience_scores": {
                    k: v.model_dump() for k, v in self.last_salience_scores.items()
                }
            }
            
        elif operation == "check_novelty":
            item_id = input_data.get("item_id", "")
            item_data = input_data.get("item_data", {})
            
            if item_id and item_data:
                novelty = self.calculate_novelty(item_id, item_data)
                return {
                    "status": "success",
                    "item_id": item_id,
                    "novelty": novelty
                }
            else:
                return {"status": "error", "message": "Missing item ID or data"}
                
        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update module's developmental level
        
        As the salience detector develops:
        - Novelty detection becomes more sophisticated
        - Emotional significance assessment improves
        - Goal relevance evaluation becomes more accurate
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update parameters based on development level change
        delta = self.development_level - prev_level
        
        # Decrease novelty bias (less distracted by mere novelty)
        novelty_decrease = delta * 0.1
        self.parameters.novelty_bias = max(0.3, self.parameters.novelty_bias - novelty_decrease)
        
        # Increase emotional bias (better emotional understanding)
        emotional_increase = delta * 0.1
        self.parameters.emotional_bias = min(0.9, self.parameters.emotional_bias + emotional_increase)
        
        # Increase our history capacity (better memory for novelty detection)
        self.max_history_size = int(1000 + 4000 * self.development_level)
        
        return self.development_level
    
    def detect_salience(self, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect salience in the provided inputs
        
        Parameters:
        inputs: Dictionary of input items to evaluate for salience
        context: Optional context information for salience calculation
        
        Returns:
        Operation result with salience scores
        """
        if not inputs:
            return {"status": "error", "message": "No inputs provided"}
            
        # Reset salience scores
        self.last_salience_scores = {}
        
        # Process each input item
        salience_scores = {}
        
        for item_id, item_data in inputs.items():
            # Calculate component factors
            novelty = self.calculate_novelty(item_id, item_data)
            emotional_significance = self.calculate_emotional_significance(item_data)
            goal_relevance = self.calculate_goal_relevance(item_data)
            intensity = self.calculate_intensity(item_data)
            
            # Create salience score
            score = SalienceScore(
                item_id=item_id,
                novelty=novelty,
                emotional_significance=emotional_significance,
                goal_relevance=goal_relevance,
                intensity=intensity
            )
            
            # Store in results
            salience_scores[item_id] = score.score
            self.last_salience_scores[item_id] = score
            
            # Add to input history for future novelty detection
            self._update_input_history(item_id, item_data)
        
        # Publish salience detection event
        self.publish_message("salience_detected", {
            "salience_scores": salience_scores,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "salience_scores": salience_scores,
            "detailed_scores": {
                k: v.model_dump() for k, v in self.last_salience_scores.items()
            }
        }
    
    def calculate_novelty(self, item_id: str, item_data: Any) -> float:
        """
        Calculate novelty factor for an item
        
        The novelty detection depends on the developmental level:
        - Early development: simple has-been-seen-before check
        - Later development: more sophisticated feature-based comparison
        
        Parameters:
        item_id: ID of the item
        item_data: Data for the item
        
        Returns:
        Novelty score (0.0-1.0)
        """
        # Check if item has been seen before
        if item_id in self.input_history:
            # Item has been seen, calculate how novel its current state is
            prev_data = self.input_history[item_id]
            
            # Simple comparison for primitive development
            if self.development_level < 0.3:
                # Very basic novelty detection - just seen/not seen
                return 0.2  # Low novelty since we've seen it before
                
            elif self.development_level < 0.6:
                # More nuanced - check if key attributes have changed
                if isinstance(item_data, dict) and isinstance(prev_data.get("data"), dict):
                    # Compare dictionaries
                    changes = 0
                    total_keys = 0
                    
                    for key, value in item_data.items():
                        total_keys += 1
                        if key not in prev_data["data"] or prev_data["data"][key] != value:
                            changes += 1
                    
                    if total_keys > 0:
                        return min(1.0, changes / total_keys)
                    
                return 0.3  # Moderate novelty
                
            else:
                # Advanced novelty detection 
                # This would ideally use embeddings/feature vectors for comparison
                # For now, we'll use a simplified approach
                
                # Frequency-based novelty (less frequently seen = more novel)
                freq = prev_data.get("frequency", 1)
                recency = (datetime.now() - prev_data.get("last_seen", datetime.now())).total_seconds()
                
                # Normalize recency (higher = more recent = less novel)
                recency_factor = min(1.0, recency / (3600 * 24))  # Normalize to 1 day
                recency_novelty = 1.0 - recency_factor
                
                # Frequency-based novelty (higher frequency = less novel)
                freq_novelty = 1.0 / (1.0 + np.log1p(freq))
                
                # Combine factors
                return 0.6 * recency_novelty + 0.4 * freq_novelty
        else:
            # Item has never been seen before - maximum novelty
            return 1.0
    
    def calculate_emotional_significance(self, item_data: Any) -> float:
        """
        Calculate emotional significance of an item
        
        Parameters:
        item_data: Data for the item
        
        Returns:
        Emotional significance score (0.0-1.0)
        """
        # Early development - very basic emotional significance detection
        if self.development_level < 0.3:
            # Check for basic emotional content if item is a string
            if isinstance(item_data, str):
                # Very primitive emotion detection
                positive_words = {"happy", "good", "nice", "love", "like"}
                negative_words = {"sad", "bad", "angry", "fear", "hate"}
                
                item_text = item_data.lower()
                pos_count = sum(1 for word in positive_words if word in item_text)
                neg_count = sum(1 for word in negative_words if word in item_text)
                
                # Simple emotional significance based on emotion word count
                emotion_count = pos_count + neg_count
                if emotion_count > 0:
                    return min(1.0, 0.3 + 0.1 * emotion_count)
            
            # Default low emotional significance
            return 0.1
            
        elif self.development_level < 0.6:
            # More nuanced emotional significance
            # Use current emotion state to influence significance
            if self.emotion_state:
                # Check if item content matches current emotional state
                if isinstance(item_data, dict):
                    # If item has emotion data
                    if "emotion" in item_data:
                        item_emotion = item_data["emotion"]
                        # Check how closely item emotion matches current emotion
                        if isinstance(item_emotion, str) and item_emotion in self.emotion_state:
                            return min(1.0, 0.4 + 0.6 * self.emotion_state[item_emotion])
                        elif isinstance(item_emotion, dict):
                            # Calculate overlap between item emotions and current emotions
                            match_score = 0.0
                            for emotion, intensity in item_emotion.items():
                                if emotion in self.emotion_state:
                                    match_score += intensity * self.emotion_state[emotion]
                            return min(1.0, 0.4 + match_score)
                
                # Moderate default emotional significance
                return 0.3
            
            return 0.2
            
        else:
            # Advanced emotional significance
            # This would ideally use an emotion classifier/sentiment analyzer
            # For now, we'll use a simplified approach based on current emotions
            
            # Calculate emotional congruence with current state
            if isinstance(item_data, dict) and "emotional_valence" in item_data:
                # Item has explicit emotional valence
                item_valence = item_data["emotional_valence"]
                
                # Calculate how strongly this valence aligns with current emotions
                valence_match = 0.0
                for emotion, intensity in self.emotion_state.items():
                    # Simple valence matching
                    if emotion in {"joy", "trust", "anticipation"} and item_valence > 0:
                        valence_match += intensity * item_valence
                    elif emotion in {"sadness", "fear", "anger", "disgust"} and item_valence < 0:
                        valence_match += intensity * abs(item_valence)
                
                return min(1.0, 0.3 + 0.7 * valence_match)
            
            # Default moderate emotional significance
            return 0.4
    
    def calculate_goal_relevance(self, item_data: Any) -> float:
        """
        Calculate relevance of an item to current goals
        
        Parameters:
        item_data: Data for the item
        
        Returns:
        Goal relevance score (0.0-1.0)
        """
        # No goals means no goal relevance
        if not self.current_goals:
            return 0.0
            
        # Early development - very basic goal relevance
        if self.development_level < 0.3:
            # Primitive development can't really assess goal relevance
            return 0.1
            
        elif self.development_level < 0.6:
            # Simple keyword matching for goal relevance
            if isinstance(item_data, str) or (isinstance(item_data, dict) and "content" in item_data):
                content = item_data if isinstance(item_data, str) else item_data["content"]
                
                # Extract keywords from goals
                goal_keywords = set()
                for goal in self.current_goals:
                    goal_desc = goal.get("description", "")
                    keywords = goal.get("keywords", [])
                    
                    # Add explicit keywords
                    goal_keywords.update(keywords)
                    
                    # Add words from goal description
                    if isinstance(goal_desc, str):
                        words = goal_desc.lower().split()
                        # Filter out common words
                        content_words = [w for w in words if len(w) > 3]
                        goal_keywords.update(content_words)
                
                # Check for keyword matches in content
                if isinstance(content, str):
                    content_lower = content.lower()
                    matches = sum(1 for kw in goal_keywords if kw.lower() in content_lower)
                    
                    if matches > 0:
                        return min(1.0, 0.3 + 0.1 * matches)
            
            # Moderate default goal relevance
            return 0.2
            
        else:
            # Advanced goal relevance
            # This would ideally use semantic similarity between item and goals
            # For now, use a simplified approach
            
            max_relevance = 0.0
            
            for goal in self.current_goals:
                # Get goal importance
                importance = goal.get("importance", 0.5)
                
                # Calculate relevance based on goal type and item data
                relevance = 0.0
                
                if isinstance(item_data, dict):
                    # Check for direct goal references
                    if "goal_id" in item_data and item_data["goal_id"] == goal.get("id"):
                        relevance = 0.8  # High relevance for direct goal references
                    
                    # Check for context matches
                    elif "context" in item_data and "context" in goal:
                        if item_data["context"] == goal["context"]:
                            relevance = 0.6  # Good relevance for context matches
                    
                    # Check for concept matches
                    elif "concepts" in item_data and "concepts" in goal:
                        item_concepts = set(item_data["concepts"])
                        goal_concepts = set(goal["concepts"])
                        
                        overlap = item_concepts.intersection(goal_concepts)
                        if overlap:
                            relevance = 0.4 + 0.4 * (len(overlap) / len(goal_concepts))
                
                # Apply importance weighting
                weighted_relevance = relevance * importance
                max_relevance = max(max_relevance, weighted_relevance)
            
            return max_relevance
    
    def calculate_intensity(self, item_data: Any) -> float:
        """
        Calculate sensory intensity of an item
        
        Parameters:
        item_data: Data for the item
        
        Returns:
        Intensity score (0.0-1.0)
        """
        # Check for explicit intensity value
        if isinstance(item_data, dict) and "intensity" in item_data:
            return min(1.0, max(0.0, float(item_data["intensity"])))
            
        # Check for volume/size indicators
        if isinstance(item_data, dict):
            # Check size attribute
            if "size" in item_data:
                size = item_data["size"]
                if isinstance(size, (int, float)):
                    return min(1.0, size / 10.0)  # Normalize to 0-1 range
                elif isinstance(size, str):
                    size_map = {"tiny": 0.1, "small": 0.3, "medium": 0.5, "large": 0.7, "huge": 0.9}
                    return size_map.get(size.lower(), 0.5)
            
            # Check volume attribute
            if "volume" in item_data:
                volume = item_data["volume"]
                if isinstance(volume, (int, float)):
                    return min(1.0, volume / 10.0)  # Normalize to 0-1 range
                elif isinstance(volume, str):
                    volume_map = {"silent": 0.1, "quiet": 0.3, "normal": 0.5, "loud": 0.8, "deafening": 1.0}
                    return volume_map.get(volume.lower(), 0.5)
            
            # Check brightness attribute
            if "brightness" in item_data:
                brightness = item_data["brightness"]
                if isinstance(brightness, (int, float)):
                    return min(1.0, brightness / 10.0)  # Normalize to 0-1 range
        
        # Default moderate intensity
        return 0.5
    
    def _update_input_history(self, item_id: str, item_data: Any) -> None:
        """
        Update input history for novelty detection
        
        Parameters:
        item_id: ID of the item
        item_data: Data for the item
        """
        # If item exists, update it
        if item_id in self.input_history:
            prev_data = self.input_history[item_id]
            
            # Update frequency
            freq = prev_data.get("frequency", 1)
            prev_data["frequency"] = freq + 1
            
            # Update last seen time
            prev_data["last_seen"] = datetime.now()
            
            # Update data (store latest version)
            prev_data["data"] = item_data
            
        else:
            # New item
            self.input_history[item_id] = {
                "data": item_data,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "frequency": 1
            }
        
        # Trim history if needed
        if len(self.input_history) > self.max_history_size:
            # Remove least recently seen items
            sorted_items = sorted(
                self.input_history.items(),
                key=lambda x: x[1].get("last_seen", datetime.min)
            )
            
            # Remove oldest items to get back to 90% of max size
            items_to_remove = len(self.input_history) - int(self.max_history_size * 0.9)
            
            for i in range(items_to_remove):
                if i < len(sorted_items):
                    del self.input_history[sorted_items[i][0]]
    
    # Event handlers
    
    def _handle_perception_input(self, message: Message) -> None:
        """
        Handle perception input events
        
        Automatically calculates salience for new perception inputs.
        """
        content = message.content
        perception_data = content.get("perception_data", {})
        
        if perception_data:
            # Detect salience in the perception data
            result = self.detect_salience(perception_data)
            
            # If successful, add salience info to the message
            if result["status"] == "success":
                perception_data["salience"] = result["salience_scores"]
    
    def _handle_emotion_update(self, message: Message) -> None:
        """
        Handle emotion update events
        
        Updates internal emotion state for emotional significance calculation.
        """
        content = message.content
        emotions = content.get("emotions", {})
        
        if emotions:
            # Update our emotion state
            self.emotion_state = emotions.copy()
    
    def _handle_goal_update(self, message: Message) -> None:
        """
        Handle goal update events
        
        Updates current goals for goal relevance calculation.
        """
        content = message.content
        goals = content.get("goals", [])
        
        if goals:
            # Update our current goals
            self.current_goals = goals.copy()