from typing import Dict, List, Tuple 
from pydantic import BaseModel, Field 
 
class Synapse(BaseModel): 
    source_id: str 
    target_id: str 
    weight: float = Field(default=0.1) 
    plasticity: float = Field(default=0.05) 
