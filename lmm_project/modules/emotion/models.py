from pydantic import BaseModel, Field 
from typing import List, Dict, Any, Optional 
 
class Emotion(BaseModel): 
    """Representation of an emotional state""" 
    valence: float = Field(..., ge=-1.0, le=1.0) 
    arousal: float = Field(..., ge=0.0, le=1.0) 
    type: str 
    confidence: float = Field(..., ge=0.0, le=1.0) 
