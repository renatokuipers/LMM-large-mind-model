"""
Base module implementation for cognitive modules
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

# Use TYPE_CHECKING to avoid runtime circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lmm_project.core.event_bus import EventBus
    from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

class BaseModule:
    """
    Base class for all cognitive modules
    
    This class defines the standard interface that all modules must implement
    and provides common functionality for state management, development tracking,
    and event communication.
    """
    
    def __init__(
        self, 
        module_id: str,
        module_type: str,
        event_bus: Optional['EventBus'] = None,
        development_level: float = 0.0,
        **kwargs
    ):
        """
        Initialize the module
        
        Args:
            module_id: Unique identifier for this module instance
            module_type: Type of module (e.g., "perception", "attention")
            event_bus: Event bus for publishing and subscribing to events
            development_level: Initial developmental level (0.0 to 1.0)
        """
        self.module_id = module_id
        self.module_type = module_type
        self.event_bus = event_bus
        self.development_level = development_level
        self.creation_time = datetime.now()
        self.last_update_time = self.creation_time
        
        self._subscriptions = []
        
        logger.debug(f"Initialized {module_type} module with ID {module_id}")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results
        
        This is the main entry point for module processing. Each module
        must implement this method to handle its specific cognitive function.
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing processing results
        """
        # Base implementation does nothing
        return {
            "status": "not_implemented",
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level
        }
        
    def update_development(self, amount: float) -> float:
        """
        Update the module's developmental level
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New developmental level
        """
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        self.last_update_time = datetime.now()
        
        # Log significant developmental changes
        if int(self.development_level * 10) > int(prev_level * 10):
            logger.info(f"Module {self.module_id} ({self.module_type}) development increased to {self.development_level:.2f}")
            
        return self.development_level
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing module state
        """
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "creation_time": self.creation_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat()
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Base implementation - to be overridden by modules
        return ""
        
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        # Base implementation - to be overridden by modules
        return False
        
    def subscribe_to_message(self, message_type: str, callback=None):
        """
        Subscribe to a specific message type
        
        Args:
            message_type: Type of message to subscribe to
            callback: Function to call when message is received
                     If None, will use self._handle_message
        """
        if not self.event_bus:
            logger.warning(f"Module {self.module_id} tried to subscribe without event bus")
            return
            
        if callback is None:
            callback = self._handle_message
            
        subscription_id = self.event_bus.subscribe(message_type, callback)
        self._subscriptions.append(subscription_id)
        
    def publish_message(self, message_type: str, content: Dict[str, Any] = None):
        """
        Publish a message to the event bus
        
        Args:
            message_type: Type of message to publish
            content: Message content
        """
        if not self.event_bus:
            logger.warning(f"Module {self.module_id} tried to publish without event bus")
            return
            
        # Import here to avoid circular imports
        from lmm_project.core.message import Message
        
        message = Message(
            sender=self.module_id,
            message_type=message_type,
            content=content or {},
            timestamp=time.time()
        )
        
        self.event_bus.publish(message)
    
    def _handle_message(self, message: 'Message'):
        """
        Default message handler - can be overridden by subclasses
        
        Args:
            message: The message to handle
        """
        # Base implementation does nothing
        pass
