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
