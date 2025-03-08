from pydantic import BaseModel, Field 
from typing import Dict, List, Any 
 
class AttentionFocus(BaseModel): 
    """Current focus of attention""" 
    targets: Dict[str, float] = Field(default_factory=dict) 
    capacity: float = Field(default=3.0) 
