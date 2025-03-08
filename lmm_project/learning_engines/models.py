from pydantic import BaseModel, Field 
from typing import Dict, List, Any 
 
class LearningEngine(BaseModel): 
    """Base class for learning engines""" 
    engine_type: str 
    learning_rate: float = Field(default=0.01) 
