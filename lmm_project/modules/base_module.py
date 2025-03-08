from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, ClassVar, Set
from pydantic import BaseModel, Field, model_validator
import uuid

from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.exceptions import ModuleProcessingError

class BaseModule(BaseModel, ABC):
    """Abstract base class for all cognitive modules"""
    module_id: str
    module_type: str
    is_active: bool = True
    development_level: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_level: float = Field(default=0.0, ge=0.0, le=1.0)
    dependencies: Set[str] = Field(default_factory=set)
    created_at: Any = Field(default_factory=lambda: __import__('datetime').datetime.now())
    event_bus: Optional[EventBus] = None
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    # Class variable to track module initialization order across all modules
    initialization_order: ClassVar[List[str]] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        # Track initialization order
        self.__class__.initialization_order.append(self.module_id)
        
    @model_validator(mode='after')
    def validate_module(self):
        """Validate module after initialization"""
        if not self.module_id:
            self.module_id = f"{self.module_type}_{uuid.uuid4().hex[:8]}"
        return self
    
    @abstractmethod
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results
        
        Parameters:
        input_data: Dictionary containing input data
        
        Returns:
        Dictionary containing processed results
        """
        pass
    
    @abstractmethod
    def update_development(self, amount: float) -> float:
        """
        Update module's developmental level
        
        Parameters:
        amount: Amount to increase development level
        
        Returns:
        New development level
        """
        # Default implementation
        self.development_level = min(1.0, self.development_level + amount)
        return self.development_level
    
    def publish_message(self, message_type: str, content: Dict[str, Any]) -> None:
        """
        Publish a message to the event bus
        
        Parameters:
        message_type: Type of message
        content: Message content
        """
        if not self.event_bus:
            return
            
        message = Message(
            sender=self.module_id,
            message_type=message_type,
            content=content
        )
        
        self.event_bus.publish(message)
    
    def subscribe_to_message(self, message_type: str, callback: Any) -> None:
        """
        Subscribe to a message type on the event bus
        
        Parameters:
        message_type: Type of message to subscribe to
        callback: Function to call when a message of this type is received
        """
        if not self.event_bus:
            return
            
        self.event_bus.subscribe(message_type, callback)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
        Dictionary containing module state
        """
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "is_active": self.is_active,
            "development_level": self.development_level,
            "activation_level": self.activation_level
        }
    
    def update_activation(self, amount: float) -> float:
        """
        Update the module's activation level
        
        Parameters:
        amount: Amount to adjust activation level (positive or negative)
        
        Returns:
        New activation level
        """
        self.activation_level = max(0.0, min(1.0, self.activation_level + amount))
        return self.activation_level
