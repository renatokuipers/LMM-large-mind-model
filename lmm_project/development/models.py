from pydantic import BaseModel, Field 
from typing import Dict, Any, List, Tuple, Optional 
 
class DevelopmentalStage(BaseModel): 
    name: str 
    age_range: Tuple[float, float] 
    capabilities: Dict[str, float] = Field(default_factory=dict) 
