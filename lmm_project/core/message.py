"""
Message definitions for the LMM system.
Messages are used for communication between modules through the event bus.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from .types import MessageType, ModuleType, MessageID, ModuleID, Timestamp, generate_id, current_timestamp
from .exceptions import MessageError


class Content(BaseModel):
    """Base class for all message content types."""
    content_type: str = Field(..., description="Type of content")
    data: Any = Field(..., description="Content data")

    model_config = {"extra": "forbid"}


class TextContent(Content):
    """Text content for messages."""
    content_type: str = "text"
    data: str

    @field_validator('data')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise MessageError(error_msg="Text content must be a non-empty string")
        return v


class VectorContent(Content):
    """Vector content for messages (embeddings, neural activations)."""
    content_type: str = "vector"
    data: List[float]
    dimensions: int

    @field_validator('data')
    @classmethod
    def validate_vector(cls, v: List[float]) -> List[float]:
        if not v or not isinstance(v, list):
            raise MessageError(error_msg="Vector content must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in v):
            raise MessageError(error_msg="Vector content must contain only numeric values")
        return v


class ImageContent(Content):
    """Image content for messages."""
    content_type: str = "image"
    data: bytes
    format: str = "png"
    width: Optional[int] = None
    height: Optional[int] = None


class AudioContent(Content):
    """Audio content for messages."""
    content_type: str = "audio"
    data: bytes
    format: str = "wav"
    duration_ms: Optional[int] = None
    sample_rate: Optional[int] = None


class StructuredContent(Content):
    """Structured data content for messages."""
    content_type: str = "structured"
    data: Dict[str, Any]
    schema: Optional[str] = None


class Message(BaseModel):
    """Base message class for all system messages."""
    id: MessageID = Field(default_factory=generate_id, description="Unique message identifier")
    sender: ModuleID = Field(..., description="ID of the sending module")
    sender_type: ModuleType = Field(..., description="Type of the sending module")
    message_type: MessageType = Field(..., description="Type of message")
    content: Content = Field(..., description="Message content")
    timestamp: Timestamp = Field(default_factory=current_timestamp, description="Message creation timestamp")
    priority: int = Field(default=1, ge=0, le=10, description="Message priority (0-10)")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds")
    in_response_to: Optional[MessageID] = Field(default=None, description="ID of message this is responding to")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")

    model_config = {"extra": "forbid"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl is None:
            return False
        age = current_timestamp() - self.timestamp
        return age > self.ttl
