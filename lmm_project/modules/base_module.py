from abc import ABC, abstractmethod 
from typing import Dict, List, Any, Optional 
from pydantic import BaseModel, Field 
 
class BaseModule(BaseModel, ABC): 
    """Abstract base class for all cognitive modules""" 
    module_id: str 
    module_type: str 
    is_active: bool = True 
    development_level: float = Field(default=0.0, ge=0.0, le=1.0) 
 
    class Config: 
        arbitrary_types_allowed = True 
 
    @abstractmethod 
    def process_input(self, input_data: Dict[str, Any]) -, Any]: 
        """Process input data and return results""" 
        pass 
 
    @abstractmethod 
    def update_development(self, amount: float) -
        """Update module's developmental level""" 
        pass 
