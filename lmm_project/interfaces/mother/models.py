from pydantic import BaseModel, Field 
from typing import Dict, Any, List 
 
class MotherPersonality(BaseModel): 
    """Mother's personality configuration""" 
    traits: Dict[str, float] = Field(default_factory=dict) 
    teaching_style: str = Field(default="balanced") 
