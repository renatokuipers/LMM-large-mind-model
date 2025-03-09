"""
Base module implementation for cognitive modules
"""

import logging
import uuid
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
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
    
    # Developmental milestones for tracking progress
    # Override this in subclasses with specific milestones
    development_milestones = {
        0.0: "Initialization",
        0.2: "Basic functionality",
        0.4: "Intermediate capabilities",
        0.6: "Advanced processing",
        0.8: "Complex integration",
        1.0: "Fully developed"
    }
    
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
        
        # Development tracking
        self.development_history = [(self.creation_time, development_level)]
        
        # Event subscription tracking
        self._subscriptions = []
        
        # Module state
        self._enabled = True
        
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
        self.development_level = min(1.0, max(0.0, self.development_level + amount))
        now = datetime.now()
        self.last_update_time = now
        
        # Track development history
        self.development_history.append((now, self.development_level))
        
        # Log significant developmental changes and milestones
        if int(self.development_level * 10) > int(prev_level * 10):
            logger.info(f"Module {self.module_id} ({self.module_type}) development increased to {self.development_level:.2f}")
            
            # Check for milestones
            for threshold, description in sorted(self.development_milestones.items()):
                if prev_level < threshold <= self.development_level:
                    logger.info(f"Development milestone: {self.module_id} reached {description}")
                    
                    # Broadcast milestone reached event if we have an event bus
                    if self.event_bus:
                        self.publish_message(
                            "development_milestone", 
                            {
                                "module_id": self.module_id,
                                "module_type": self.module_type,
                                "milestone": description,
                                "level": threshold,
                                "timestamp": now.isoformat()
                            }
                        )
                    break
            
        return self.development_level
    
    def set_development_level(self, level: float) -> None:
        """
        Set the development level directly
        
        Useful for initialization or synchronizing components
        
        Args:
            level: Development level to set (0.0 to 1.0)
        """
        level = min(1.0, max(0.0, level))
        if level != self.development_level:
            now = datetime.now()
            self.development_level = level
            self.last_update_time = now
            self.development_history.append((now, level))
        
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
            "enabled": self._enabled,
            "creation_time": self.creation_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "subscription_count": len(self._subscriptions)
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(state_dir, exist_ok=True)
            
            # Get state and prepare for serialization
            state = self.get_state()
            
            # Convert datetime objects to strings for serialization
            state['development_history'] = [
                (dt.isoformat(), level) for dt, level in self.development_history
            ]
            
            # Create filename based on module ID and type
            filename = f"{self.module_type}_{self.module_id.replace('/', '_')}.json"
            filepath = os.path.join(state_dir, filename)
            
            # Write state to file
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved state for module {self.module_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save state for module {self.module_id}: {str(e)}")
            return ""
        
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(state_path):
                logger.error(f"State file {state_path} does not exist")
                return False
                
            # Read state from file
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update basic properties
            self.development_level = state.get('development_level', self.development_level)
            self._enabled = state.get('enabled', self._enabled)
            
            # Parse development history if present
            if 'development_history' in state:
                try:
                    self.development_history = [
                        (datetime.fromisoformat(dt), level) 
                        for dt, level in state['development_history']
                    ]
                except Exception as e:
                    logger.warning(f"Failed to parse development history: {str(e)}")
            
            # Update timestamps
            if 'last_update_time' in state:
                try:
                    self.last_update_time = datetime.fromisoformat(state['last_update_time'])
                except Exception:
                    self.last_update_time = datetime.now()
                    
            logger.info(f"Loaded state for module {self.module_id} from {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state for module {self.module_id}: {str(e)}")
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
        
    def enable(self):
        """Enable this module for processing"""
        self._enabled = True
        
    def disable(self):
        """Disable this module from processing"""
        self._enabled = False
        
    def is_enabled(self) -> bool:
        """Check if this module is enabled"""
        return self._enabled
        
    def get_development_progress(self) -> Dict[str, Any]:
        """
        Get detailed development progress information
        
        Returns:
            Dictionary with development progress details
        """
        # Find current milestone
        current_milestone = None
        next_milestone = None
        milestone_progress = 0.0
        
        sorted_milestones = sorted(self.development_milestones.items())
        
        for i, (threshold, description) in enumerate(sorted_milestones):
            if threshold <= self.development_level:
                current_milestone = (threshold, description)
                # Check if there's a next milestone
                if i + 1 < len(sorted_milestones):
                    next_milestone = sorted_milestones[i + 1]
                    next_threshold = next_milestone[0]
                    # Calculate progress to next milestone
                    milestone_range = next_threshold - threshold
                    if milestone_range > 0:
                        milestone_progress = (self.development_level - threshold) / milestone_range
                
        return {
            "development_level": self.development_level,
            "current_milestone": current_milestone[1] if current_milestone else None,
            "current_milestone_threshold": current_milestone[0] if current_milestone else None,
            "next_milestone": next_milestone[1] if next_milestone else None,
            "next_milestone_threshold": next_milestone[0] if next_milestone else None,
            "progress_to_next_milestone": milestone_progress,
            "fully_developed": self.development_level >= 1.0,
            "development_time": (datetime.now() - self.creation_time).total_seconds(),
            "development_history_length": len(self.development_history)
        }
