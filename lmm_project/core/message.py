from pydantic import BaseModel, Field 
from datetime import datetime 
import uuid 
 
class Message(BaseModel): 
    id: str = Field(default_factory=lambda: str(uuid.uuid4())) 
    source_module: str 
    target_module: Optional[str] = None 
    message_type: str 
    content: Dict[str, Any] = Field(default_factory=dict) 
    timestamp: datetime = Field(default_factory=datetime.now) 
