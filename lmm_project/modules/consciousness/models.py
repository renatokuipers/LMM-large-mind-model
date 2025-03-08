from pydantic import BaseModel, Field 
from typing import Dict, Any, List 
 
class ConsciousnessState(BaseModel): 
    awareness_level: float = Field(default=0.1, ge=0.0, le=1.0) 
    global_workspace: Dict[str, Any] = Field(default_factory=dict) 
