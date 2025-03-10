"""
Event bus implementation for message passing between cognitive modules.
"""
import threading
import queue
import logging
import time
from typing import Dict, List, Callable, Optional, Set, Any, Tuple
import uuid
from collections import deque

from .types import MessageType, ModuleID, MessageID
from .message import Message
from .exceptions import EventBusError


# Type alias for message handler functions
MessageHandler = Callable[[Message], None]


class EventBus:
    """
    Central event bus for cognitive module communication.
    Implements a publisher-subscriber pattern for message passing.
    """
    def __init__(self, max_history_size: int = 1000, message_queue_size: int = 10000):
        self.subscribers: Dict[MessageType, Dict[ModuleID, MessageHandler]] = {}
        self.global_subscribers: Dict[ModuleID, MessageHandler] = {}
        self.message_history: deque = deque(maxlen=max_history_size)
        self.message_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=message_queue_size)
        self.running: bool = False
        self.processing_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start message processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="EventBusProcessor"
        )
        self.processing_thread.start()
        self.logger.info("Event bus started")

    def stop(self) -> None:
        """Stop message processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        self.logger.info("Event bus stopped")

    def subscribe(self, 
                 module_id: ModuleID, 
                 handler: MessageHandler,
                 message_types: Optional[List[MessageType]] = None) -> None:
        """
        Subscribe a module to specific message types.
        
        Args:
            module_id: Unique ID of the subscribing module
            handler: Callback function to handle messages
            message_types: List of message types to subscribe to, or None for all types
        """
        with self.lock:
            if message_types is None or len(message_types) == 0:
                # Subscribe to all message types (global subscriber)
                self.global_subscribers[module_id] = handler
                self.logger.debug(f"Module {module_id} subscribed to all message types")
                return
            
            # Subscribe to specific message types
            for msg_type in message_types:
                if msg_type not in self.subscribers:
                    self.subscribers[msg_type] = {}
                self.subscribers[msg_type][module_id] = handler
            
            self.logger.debug(f"Module {module_id} subscribed to {len(message_types)} message types")

    def unsubscribe(self, 
                   module_id: ModuleID, 
                   message_types: Optional[List[MessageType]] = None) -> None:
        """
        Unsubscribe a module from message types.
        
        Args:
            module_id: Unique ID of the module to unsubscribe
            message_types: List of message types to unsubscribe from, or None for all
        """
        with self.lock:
            # Remove from global subscribers if applicable
            if message_types is None or len(message_types) == 0:
                if module_id in self.global_subscribers:
                    del self.global_subscribers[module_id]
                
                # Also remove from all specific message types
                for msg_type in list(self.subscribers.keys()):
                    if module_id in self.subscribers[msg_type]:
                        del self.subscribers[msg_type][module_id]
                
                self.logger.debug(f"Module {module_id} unsubscribed from all message types")
                return
            
            # Unsubscribe from specific message types
            for msg_type in message_types:
                if msg_type in self.subscribers and module_id in self.subscribers[msg_type]:
                    del self.subscribers[msg_type][module_id]
            
            self.logger.debug(f"Module {module_id} unsubscribed from specific message types")

    def publish(self, message: Message) -> None:
        """
        Publish a message to the event bus.
        
        Args:
            message: The message to publish
        """
        if not self.running:
            raise EventBusError("Cannot publish message: Event bus is not running")
        
        # Check if message has already expired
        if message.is_expired():
            self.logger.debug(f"Skipping expired message: {message.id}")
            return
        
        try:
            # Priority queue item: (priority, timestamp, message)
            # Lower number = higher priority, timestamp is used as a tiebreaker
            priority_item = (-message.priority, message.timestamp, message)
            self.message_queue.put(priority_item, block=False)
            self.logger.debug(f"Message queued: {message.id} (Type: {message.message_type.name})")
        except queue.Full:
            raise EventBusError("Message queue is full")

    def _process_messages(self) -> None:
        """
        Process messages from the queue and dispatch to subscribers.
        This runs in a separate thread.
        """
        while self.running:
            try:
                # Get message from queue, block for a short time to allow for clean shutdown
                try:
                    priority, _, message = self.message_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip expired messages
                if message.is_expired():
                    self.logger.debug(f"Skipping expired message during processing: {message.id}")
                    self.message_queue.task_done()
                    continue
                
                # Add to history
                self.message_history.append(message)
                
                # Dispatch message to subscribers
                self._dispatch_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                # Continue processing other messages
    
    def _dispatch_message(self, message: Message) -> None:
        """
        Dispatch a message to all subscribed handlers.
        
        Args:
            message: The message to dispatch
        """
        dispatched = False
        
        # Dispatch to specific subscribers for this message type
        if message.message_type in self.subscribers:
            with self.lock:
                subscribers = list(self.subscribers[message.message_type].items())
            
            for module_id, handler in subscribers:
                # Skip sending the message back to its sender
                if module_id == message.sender:
                    continue
                    
                try:
                    handler(message)
                    dispatched = True
                except Exception as e:
                    self.logger.error(f"Error in message handler for module {module_id}: {str(e)}")
        
        # Dispatch to global subscribers
        with self.lock:
            global_subscribers = list(self.global_subscribers.items())
        
        for module_id, handler in global_subscribers:
            # Skip sending the message back to its sender
            if module_id == message.sender:
                continue
                
            try:
                handler(message)
                dispatched = True
            except Exception as e:
                self.logger.error(f"Error in global message handler for module {module_id}: {str(e)}")
        
        if not dispatched:
            self.logger.debug(f"No subscribers for message {message.id} (Type: {message.message_type.name})")

    def get_message_history(self, 
                           limit: Optional[int] = None, 
                           message_type: Optional[MessageType] = None) -> List[Message]:
        """
        Get message history, optionally filtered by type.
        
        Args:
            limit: Maximum number of messages to return
            message_type: Filter by message type
        
        Returns:
            List of messages from history
        """
        result = []
        
        for message in reversed(self.message_history):
            if message_type is not None and message.message_type != message_type:
                continue
            
            result.append(message)
            
            if limit is not None and len(result) >= limit:
                break
                
        return result

    def get_queue_size(self) -> int:
        """Get current size of the message queue."""
        return self.message_queue.qsize()

    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()
        self.logger.debug("Message history cleared")


# Singleton instance
_event_bus_instance = None
_event_bus_lock = threading.RLock()

def get_event_bus(max_history_size: int = 1000, message_queue_size: int = 10000) -> EventBus:
    """
    Get the global EventBus instance (singleton).
    
    Args:
        max_history_size: Maximum number of messages to keep in history
        message_queue_size: Maximum size of the message queue
        
    Returns:
        Global EventBus instance
    """
    global _event_bus_instance
    
    with _event_bus_lock:
        if _event_bus_instance is None:
            _event_bus_instance = EventBus(
                max_history_size=max_history_size,
                message_queue_size=message_queue_size
            )
            
    return _event_bus_instance
