from pydantic import BaseModel, Field 
from typing import Dict, Any, List 
 
class ResearchMetrics(BaseModel): 
    """Research metrics for tracking development""" 
    category: str 
    metrics: Dict[str, Any] = Field(default_factory=dict) 
