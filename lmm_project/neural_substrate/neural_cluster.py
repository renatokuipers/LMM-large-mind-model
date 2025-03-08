from typing import Dict, List, Optional 
from pydantic import BaseModel, Field 
 
class NeuralCluster(BaseModel): 
    id: str 
    neurons: Dict[str, 'Neuron'] = Field(default_factory=dict) 
    cluster_type: str 
