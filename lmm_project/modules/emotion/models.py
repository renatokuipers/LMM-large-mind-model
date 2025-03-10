from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time

class EmotionState(BaseModel):
    """
    Represents the current emotional state
    
    This includes dimensional values (valence, arousal) and categorical emotions
    """
    valence: float = Field(..., ge=-1.0, le=1.0, description="Pleasure-displeasure dimension")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Activation level")
    dominant_emotion: str = Field(..., description="The primary emotion being experienced")
    emotion_intensities: Dict[str, float] = Field(..., description="Intensity of each emotion")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this state was recorded")
    
    @validator('emotion_intensities')
    def check_intensities(cls, v):
        """Ensure all intensities are between 0 and 1"""
        for emotion, intensity in v.items():
            if not 0 <= intensity <= 1:
                raise ValueError(f"Intensity of {emotion} must be between 0 and 1")
        return v
    
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class EmotionalResponse(BaseModel):
    """
    An emotional response to a specific stimulus
    
    This captures how the emotional system responds to particular input
    """
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    dominant_emotion: str
    emotion_intensities: Dict[str, float]
    regulated: bool = Field(False, description="Whether this response has been regulated")
    regulation_strategy: Optional[str] = Field(None, description="Strategy used for regulation")
    stimulus: Optional[str] = Field(None, description="What triggered this response")
    process_id: str = Field(..., description="ID of the process that generated this response")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('emotion_intensities')
    def check_intensities(cls, v):
        """Ensure all intensities are between 0 and 1"""
        for emotion, intensity in v.items():
            if not 0 <= intensity <= 1:
                raise ValueError(f"Intensity of {emotion} must be between 0 and 1")
        return v
    
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class SentimentAnalysis(BaseModel):
    """
    Analysis of sentiment in text
    
    This captures various aspects of emotional tone in language
    """
    text: str = Field(..., description="The text being analyzed")
    positive_score: float = Field(..., ge=0.0, le=1.0, description="Degree of positive sentiment")
    negative_score: float = Field(..., ge=0.0, le=1.0, description="Degree of negative sentiment")
    neutral_score: float = Field(..., ge=0.0, le=1.0, description="Degree of neutral sentiment")
    compound_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment score")
    detected_emotions: Dict[str, float] = Field(default_factory=dict, description="Detected emotions and intensities")
    highlighted_phrases: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Emotionally salient phrases with scores"
    )
    process_id: str = Field(..., description="ID of the process that generated this analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class EmotionRegulationRequest(BaseModel):
    """
    Request to regulate an emotional state
    
    This specifies how emotions should be modified
    """
    current_state: EmotionState
    target_valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    target_arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_emotion: Optional[str] = Field(None)
    regulation_strategy: Optional[str] = Field(None)
    context: Dict[str, Any] = Field(default_factory=dict)
    process_id: str = Field(..., description="ID of the regulation process")
    
    class Config:
        arbitrary_types_allowed = True

class EmotionRegulationResult(BaseModel):
    """
    Result of an emotion regulation attempt
    
    This captures how emotions were modified through regulation
    """
    original_state: EmotionState
    regulated_state: EmotionState
    regulation_strategy: str
    success_level: float = Field(..., ge=0.0, le=1.0)
    process_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['timestamp'] = self.timestamp.isoformat()
        result['original_state'] = self.original_state.dict()
        result['regulated_state'] = self.regulated_state.dict()
        return result

class EmotionalParameters(BaseModel):
    """
    Parameters that control emotional processing
    
    These parameters are adjusted based on developmental level
    """
    emotional_inertia: float = Field(..., ge=0.0, le=1.0, description="Resistance to emotional change")
    stimulus_sensitivity: float = Field(..., ge=0.0, le=1.0, description="Sensitivity to emotional stimuli")
    emotion_decay_rate: float = Field(..., ge=0.0, le=1.0, description="How quickly emotions return to baseline")
    baseline_valence: float = Field(..., ge=-1.0, le=1.0, description="Default valence state")
    baseline_arousal: float = Field(..., ge=0.0, le=1.0, description="Default arousal state")
    regulation_capacity: float = Field(..., ge=0.0, le=1.0, description="Ability to regulate emotions")

class EmotionNeuralState(BaseModel):
    """
    State information for the emotion neural networks
    
    This tracks the state of neural networks used for emotional processing,
    including developmental levels and recent activations.
    """
    encoder_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of emotion encoder")
    classifier_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of emotion classifier")
    sentiment_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of sentiment analyzer")
    regulation_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of emotion regulator")
    
    # Track recent activations for each neural component
    recent_encoder_activations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recent activations of the emotion encoder"
    )
    recent_classifier_activations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recent activations of the emotion classifier"
    )
    recent_sentiment_activations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recent activations of the sentiment analyzer"
    )
    recent_regulation_activations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recent activations of the emotion regulator"
    )
    
    # Network performance metrics
    encoder_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of emotion encoder")
    classifier_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of emotion classifier")
    sentiment_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of sentiment analyzer")
    regulation_success_rate: float = Field(0.5, ge=0.0, le=1.0, description="Success rate of emotion regulation")
    
    # Last update timestamp
    last_updated: datetime = Field(default_factory=datetime.now, description="When neural state was last updated")
    
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        return result
    
    def update_accuracy(self, component: str, accuracy: float) -> None:
        """
        Update the accuracy for a specific neural component
        
        Args:
            component: The component to update ('encoder', 'classifier', 'sentiment', 'regulation')
            accuracy: The new accuracy value (0.0 to 1.0)
        """
        if component == 'encoder':
            self.encoder_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'classifier':
            self.classifier_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'sentiment':
            self.sentiment_accuracy = max(0.0, min(1.0, accuracy))
        elif component == 'regulation':
            self.regulation_success_rate = max(0.0, min(1.0, accuracy))
        
        self.last_updated = datetime.now()
    
    def add_activation(self, component: str, activation: Dict[str, Any]) -> None:
        """
        Add a recent activation for a neural component
        
        Args:
            component: The component that was activated
            activation: Dictionary with activation details
        """
        activation_with_timestamp = {
            **activation,
            "timestamp": datetime.now().isoformat()
        }
        
        if component == 'encoder':
            self.recent_encoder_activations.append(activation_with_timestamp)
            if len(self.recent_encoder_activations) > 10:  # Keep last 10
                self.recent_encoder_activations = self.recent_encoder_activations[-10:]
        elif component == 'classifier':
            self.recent_classifier_activations.append(activation_with_timestamp)
            if len(self.recent_classifier_activations) > 10:
                self.recent_classifier_activations = self.recent_classifier_activations[-10:]
        elif component == 'sentiment':
            self.recent_sentiment_activations.append(activation_with_timestamp)
            if len(self.recent_sentiment_activations) > 10:
                self.recent_sentiment_activations = self.recent_sentiment_activations[-10:]
        elif component == 'regulation':
            self.recent_regulation_activations.append(activation_with_timestamp)
            if len(self.recent_regulation_activations) > 10:
                self.recent_regulation_activations = self.recent_regulation_activations[-10:]
            
        self.last_updated = datetime.now()

class EmotionSystemState(BaseModel):
    """
    Complete state of the emotion system
    
    This combines the current emotional state, parameters, and neural state
    """
    current_state: EmotionState
    parameters: EmotionalParameters
    neural_state: EmotionNeuralState = Field(default_factory=EmotionNeuralState)
    
    # History of emotional states
    emotion_history: List[EmotionState] = Field(default_factory=list, description="Recent emotion states")
    
    # History of emotional responses
    response_history: List[EmotionalResponse] = Field(default_factory=list, description="Recent emotional responses")
    
    # History of regulation attempts
    regulation_history: List[EmotionRegulationResult] = Field(default_factory=list, description="Recent regulation attempts")
    
    # System metadata
    module_id: str
    developmental_level: float = Field(0.0, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    def dict(self, *args, **kwargs):
        """Convert datetime to timestamp for serialization"""
        result = super().dict(*args, **kwargs)
        result['last_updated'] = self.last_updated.isoformat()
        
        # Convert emotional states to dictionaries
        result['current_state'] = self.current_state.dict()
        result['emotion_history'] = [state.dict() for state in self.emotion_history]
        result['response_history'] = [response.dict() for response in self.response_history]
        result['regulation_history'] = [regulation.dict() for regulation in self.regulation_history]
        
        return result
    
    def add_emotion_state(self, state: EmotionState, max_history: int = 20) -> None:
        """
        Add an emotion state to history
        
        Args:
            state: The emotion state to add
            max_history: Maximum number of states to keep
        """
        self.emotion_history.append(state)
        if len(self.emotion_history) > max_history:
            self.emotion_history = self.emotion_history[-max_history:]
        self.last_updated = datetime.now()
    
    def add_emotional_response(self, response: EmotionalResponse, max_history: int = 20) -> None:
        """
        Add an emotional response to history
        
        Args:
            response: The emotional response to add
            max_history: Maximum number of responses to keep
        """
        self.response_history.append(response)
        if len(self.response_history) > max_history:
            self.response_history = self.response_history[-max_history:]
        self.last_updated = datetime.now()
    
    def add_regulation_result(self, result: EmotionRegulationResult, max_history: int = 20) -> None:
        """
        Add a regulation result to history
        
        Args:
            result: The regulation result to add
            max_history: Maximum number of results to keep
        """
        self.regulation_history.append(result)
        if len(self.regulation_history) > max_history:
            self.regulation_history = self.regulation_history[-max_history:]
        self.last_updated = datetime.now()
