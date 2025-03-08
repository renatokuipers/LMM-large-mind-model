from typing import Dict, Optional, Any 
from pydantic import BaseModel, Field 
 
class StateManager(BaseModel): 
    global_state: Dict[str, Any] = Field(default_factory=dict) 
 
    def get_state(self, key: str, default: Any = None) -
        return self.global_state.get(key, default) 
 
    def set_state(self, key: str, value: Any) -
        self.global_state[key] = value 
