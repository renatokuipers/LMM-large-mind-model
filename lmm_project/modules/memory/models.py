from pydantic import BaseModel, Field 
from typing import List, Dict, Any, Optional 
from datetime import datetime 
 
class Memory(BaseModel): 
    """A single memory unit in the system""" 
    id: str 
    content: str 
    timestamp: datetime = Field(default_factory=datetime.now) 
    importance: float = Field(default=0.5, ge=0.0, le=1.0) 
    embedding: Optional[List[float]] = None 
