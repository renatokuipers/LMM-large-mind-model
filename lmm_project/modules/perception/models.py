from pydantic import BaseModel, Field 
from typing import List, Dict, Any 
 
class SensoryPattern(BaseModel): 
    """Basic pattern that can be recognized""" 
    id: str 
    pattern_data: List[float] 
    recognition_threshold: float = Field(default=0.7) 
