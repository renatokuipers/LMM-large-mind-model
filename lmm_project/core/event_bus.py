from typing import Dict, List, Callable, Any
from collections import defaultdict
from pydantic import BaseModel, Field

from .message import Message
from .exceptions import EventBusError

class EventBus(BaseModel):
    """
    Event bus for inter-module communication.
    
    The event bus allows modules to publish messages and subscribe to message types.
    This facilitates decoupled communication between cognitive modules.
    """
    subscribers: Dict[str, List[Callable]] = Field(default_factory=lambda: defaultdict(list))
    message_history: List[Message] = Field(default_factory=list)
    max_history_size: int = Field(default=1000)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def publish(self, message: Message) -> None:
        """
        Publish a message to the event bus
        
        Parameters:
        message: The message to publish
        """
        try:
            # Add to history
            self.message_history.append(message)
            
            # Trim history if needed
            if len(self.message_history) > self.max_history_size:
                self.message_history = self.message_history[-self.max_history_size:]
            
            # Notify subscribers
            message_type = message.message_type
            
            # Notify specific message type subscribers
            for callback in self.subscribers.get(message_type, []):
                callback(message)
            
            # Notify "all" subscribers
            for callback in self.subscribers.get("all", []):
                callback(message)
                
        except Exception as e:
            raise EventBusError(f"Error publishing message: {str(e)}")
    
    def subscribe(self, message_type: str, callback: Callable[[Message], None]) -> None:
        """
        Subscribe to a specific message type
        
        Parameters:
        message_type: The type of message to subscribe to ("all" for all messages)
        callback: Function to call when a message of the specified type is published
        """
        try:
            self.subscribers[message_type].append(callback)
        except Exception as e:
            raise EventBusError(f"Error subscribing to {message_type}: {str(e)}")
            
    def unsubscribe(self, message_type: str, callback: Callable[[Message], None]) -> None:
        """
        Unsubscribe from a specific message type
        
        Parameters:
        message_type: The type of message to unsubscribe from
        callback: The callback function to remove
        """
        try:
            if message_type in self.subscribers and callback in self.subscribers[message_type]:
                self.subscribers[message_type].remove(callback)
        except Exception as e:
            raise EventBusError(f"Error unsubscribing from {message_type}: {str(e)}")
            
    def get_recent_messages(self, message_type: str = None, limit: int = 10) -> List[Message]:
        """
        Get recent messages, optionally filtered by type
        
        Parameters:
        message_type: Optional message type to filter by
        limit: Maximum number of messages to return
        
        Returns:
        List of recent messages
        """
        if message_type:
            filtered = [m for m in self.message_history if m.message_type == message_type]
            return filtered[-limit:]
        
        return self.message_history[-limit:]
