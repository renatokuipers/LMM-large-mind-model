from typing import List, Dict, Any 
from pydantic import BaseModel, Field 
 
class Neuron(BaseModel): 
    id: str 
    activation: float = Field(default=0.0) 
    threshold: float = Field(default=0.5) 
    connections: Dict[str, float] = Field(default_factory=dict) 
