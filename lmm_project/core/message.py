from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class Message(BaseModel):
    """
    Message object for inter-module communication
    
    This class represents messages sent between cognitive modules
    via the event bus. Each message has a type, content, and sender.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    message_type: str
    content: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __repr__(self) -> str:
        """String representation of the message"""
        return f"Message(id={self.id[:8]}, type={self.message_type}, sender={self.sender})"
    
    @property
    def age(self) -> float:
        """Get the age of the message in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "sender": self.sender,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
