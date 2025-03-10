from typing import Dict, List, Callable, Any, Optional, Set
from collections import defaultdict
from pydantic import BaseModel, Field
import threading
import logging
from datetime import datetime

from .message import Message
from .exceptions import EventBusError

logger = logging.getLogger(__name__)

class EventBus(BaseModel):
    """
    Event bus for inter-module communication.
    
    The event bus allows modules to publish messages and subscribe to message types.
    This facilitates decoupled communication between cognitive modules.
    
    Features:
    - Thread-safe message publishing and subscription management
    - Message history with configurable size
    - Filtered message retrieval
    - Priority-based message handling
    - Performance optimizations for high-frequency messages
    """
    subscribers: Dict[str, List[Callable]] = Field(default_factory=lambda: defaultdict(list))
    message_history: List[Message] = Field(default_factory=list)
    max_history_size: int = Field(default=1000)
    _lock: threading.RLock = Field(default_factory=threading.RLock, exclude=True)
    _high_frequency_types: Set[str] = Field(default_factory=set, exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def publish(self, message: Message) -> None:
        """
        Publish a message to the event bus
        
        Args:
            message: The message to publish
        
        Raises:
            EventBusError: If there's an error publishing the message
        """
        try:
            with self._lock:
                # Add to history (skip for high-frequency messages if needed)
                if message.message_type not in self._high_frequency_types:
                    self.message_history.append(message)
                    
                    # Trim history if needed
                    if len(self.message_history) > self.max_history_size:
                        self.message_history = self.message_history[-self.max_history_size:]
                
                # Get subscribers (make a copy to avoid modification during iteration)
                message_type = message.message_type
                type_subscribers = list(self.subscribers.get(message_type, []))
                all_subscribers = list(self.subscribers.get("all", []))
            
            # Notify subscribers outside the lock to prevent deadlocks
            # Notify specific message type subscribers
            for callback in type_subscribers:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for {message_type}: {str(e)}")
            
            # Notify "all" subscribers
            for callback in all_subscribers:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in 'all' subscriber callback: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error publishing message: {str(e)}")
            raise EventBusError(f"Error publishing message: {str(e)}")
    
    def subscribe(self, message_type: str, callback: Callable[[Message], None]) -> None:
        """
        Subscribe to a specific message type
        
        Args:
            message_type: The type of message to subscribe to ("all" for all messages)
            callback: Function to call when a message of the specified type is published
        
        Raises:
            EventBusError: If there's an error subscribing to the message type
        """
        try:
            with self._lock:
                if callback not in self.subscribers[message_type]:
                    self.subscribers[message_type].append(callback)
        except Exception as e:
            logger.error(f"Error subscribing to {message_type}: {str(e)}")
            raise EventBusError(f"Error subscribing to {message_type}: {str(e)}")
            
    def unsubscribe(self, message_type: str, callback: Callable[[Message], None]) -> None:
        """
        Unsubscribe from a specific message type
        
        Args:
            message_type: The type of message to unsubscribe from
            callback: The callback function to remove
        
        Raises:
            EventBusError: If there's an error unsubscribing from the message type
        """
        try:
            with self._lock:
                if message_type in self.subscribers and callback in self.subscribers[message_type]:
                    self.subscribers[message_type].remove(callback)
        except Exception as e:
            logger.error(f"Error unsubscribing from {message_type}: {str(e)}")
            raise EventBusError(f"Error unsubscribing from {message_type}: {str(e)}")
    
    def register_high_frequency_type(self, message_type: str) -> None:
        """
        Register a message type as high-frequency to optimize performance
        
        High-frequency messages bypass history storage to improve performance.
        
        Args:
            message_type: The message type to register as high-frequency
        """
        with self._lock:
            self._high_frequency_types.add(message_type)
            
    def unregister_high_frequency_type(self, message_type: str) -> None:
        """
        Unregister a message type as high-frequency
        
        Args:
            message_type: The message type to unregister as high-frequency
        """
        with self._lock:
            self._high_frequency_types.discard(message_type)
            
    def get_recent_messages(self, message_type: Optional[str] = None, limit: int = 10, 
                           since_timestamp: Optional[datetime] = None) -> List[Message]:
        """
        Get recent messages, optionally filtered by type and timestamp
        
        Args:
            message_type: Optional message type to filter by
            limit: Maximum number of messages to return
            since_timestamp: Only return messages after this timestamp
        
        Returns:
            List of recent messages matching the criteria
        """
        with self._lock:
            if message_type:
                filtered = [m for m in self.message_history if m.message_type == message_type]
            else:
                filtered = self.message_history.copy()
                
            if since_timestamp:
                filtered = [m for m in filtered if m.timestamp > since_timestamp]
                
            return filtered[-limit:]
    
    def clear_history(self) -> None:
        """Clear the message history"""
        with self._lock:
            self.message_history.clear()
            
    def get_subscriber_count(self, message_type: Optional[str] = None) -> int:
        """
        Get the number of subscribers for a message type or all subscribers
        
        Args:
            message_type: Optional message type to get subscriber count for
        
        Returns:
            Number of subscribers
        """
        with self._lock:
            if message_type:
                return len(self.subscribers.get(message_type, []))
            else:
                return sum(len(subscribers) for subscribers in self.subscribers.values())
