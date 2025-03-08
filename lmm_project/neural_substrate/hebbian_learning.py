from typing import Dict, Tuple 
from pydantic import BaseModel, Field 
 
class HebbianLearning(BaseModel): 
    learning_rate: float = Field(default=0.01) 
 
    def update_connection(self, connection_weight: float, pre_activation: float, post_activation: float) -
        """Apply Hebbian learning rule: neurons that fire together, wire together""" 
        delta = self.learning_rate * pre_activation * post_activation 
        return connection_weight + delta 
