from typing import Any, Dict, Optional, List, Set, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import uuid4
import json

class Message(BaseModel):
    """
    Message object for inter-module communication
    
    This class represents messages sent between cognitive modules
    via the event bus. Each message has a type, content, and sender.
    
    Messages can include:
    - Unique identifier
    - Sender module identification
    - Message type for routing
    - Content payload (must be serializable)
    - Timestamp for temporal ordering
    - Priority for processing order
    - Metadata for additional context
    - References to related messages
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    message_type: str
    content: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = Field(default=0, ge=0, le=10)  # 0-10 priority scale
    metadata: Dict[str, Any] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)  # IDs of related messages
    tags: Set[str] = Field(default_factory=set)  # Tags for categorization
    
    @field_validator('content')
    @classmethod
    def validate_content_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that content is JSON serializable"""
        try:
            json.dumps(v)
            return v
        except (TypeError, OverflowError):
            # Convert non-serializable content to strings
            return {k: str(val) if not _is_json_serializable(val) else val for k, val in v.items()}
    
    def __repr__(self) -> str:
        """String representation of the message"""
        return f"Message(id={self.id[:8]}, type={self.message_type}, sender={self.sender}, priority={self.priority})"
    
    @property
    def age(self) -> float:
        """Get the age of the message in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def add_reference(self, message_id: str) -> None:
        """Add a reference to another message"""
        if message_id not in self.references:
            self.references.append(message_id)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the message"""
        self.tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if the message has a specific tag"""
        return tag in self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "sender": self.sender,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "metadata": self.metadata,
            "references": self.references,
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        # Convert timestamp string to datetime if needed
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Convert tags list to set if needed
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])
            
        return cls(**data)
    
    def is_response_to(self, other_message: Union[str, "Message"]) -> bool:
        """Check if this message is a response to another message"""
        if isinstance(other_message, Message):
            return other_message.id in self.references
        return other_message in self.references
    
    def with_content(self, content: Dict[str, Any]) -> "Message":
        """Create a new message with updated content"""
        new_data = self.model_dump()
        new_data["content"] = content
        return Message(**new_data)
    
    def with_priority(self, priority: int) -> "Message":
        """Create a new message with updated priority"""
        new_data = self.model_dump()
        new_data["priority"] = max(0, min(10, priority))  # Ensure 0-10 range
        return Message(**new_data)

def _is_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
