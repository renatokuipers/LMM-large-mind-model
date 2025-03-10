"""
Base Module for the LMM Cognitive Architecture.

This module defines the BaseModule class that all cognitive modules must inherit from.
It provides standardized interfaces and functionality for module registration,
event handling, developmental progression, and state management.
"""
import abc
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from lmm_project.core.event_bus import get_event_bus
from lmm_project.core.message import Content, Message, TextContent
from lmm_project.core.state_manager import get_state_manager
from lmm_project.core.types import (
    Age, DevelopmentalStage, MessageID, MessageType,
    ModuleID, ModuleType, StateDict, Timestamp
)
from lmm_project.utils.logging_utils import get_module_logger


class ModuleConfig(BaseModel):
    """Configuration settings for cognitive modules."""
    enabled: bool = Field(default=True, description="Whether the module is enabled")
    development_rate: float = Field(default=1.0, description="Rate of development relative to global rate")
    initial_development_level: float = Field(default=0.0, description="Initial development level (0.0-1.0)")
    learning_rate: float = Field(default=0.01, description="Base learning rate")
    update_frequency: float = Field(default=1.0, description="Updates per second")
    critical_period_sensitivity: float = Field(default=1.0, description="Sensitivity to critical periods")


class BaseModule(abc.ABC):
    """
    Base class for all cognitive modules in the LMM system.
    
    Provides standardized interfaces for:
    - Event subscription and publishing
    - Developmental progression
    - State management
    - Neural processing
    
    All cognitive modules must inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(
        self,
        module_id: ModuleID,
        module_type: ModuleType,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the base module.
        
        Parameters:
        module_id: Unique identifier for this module instance
        module_type: Type of cognitive module
        config: Module-specific configuration
        device: Torch device to use for neural operations
        """
        self.module_id = module_id
        self.module_type = module_type
        self.logger = get_module_logger(f"{module_type.name.lower()}.{module_id}")
        self.event_bus = get_event_bus()
        self.state_manager = get_state_manager()
        
        # Set up configuration
        default_config = ModuleConfig().model_dump()
        self.config = {**default_config, **(config or {})}
        
        # Initialize development tracking
        self._development_level = self.config["initial_development_level"]
        self._last_update_time = time.time()
        self._creation_timestamp = time.time()
        self._developmental_stage = DevelopmentalStage.PRENATAL
        
        # Track subscribed message types for cleanup
        self._subscribed_message_types: Set[MessageType] = set()
        
        # Set up device for neural operations
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize state
        self._initialize_state()
        
        # Log initialization
        self.logger.info(
            f"Initialized {self.module_type.name} module '{self.module_id}' at "
            f"development level {self._development_level:.2f}"
        )
    
    def _initialize_state(self) -> None:
        """Initialize the module's state in the state manager."""
        initial_state = {
            "development_level": self._development_level,
            "creation_timestamp": self._creation_timestamp,
            "last_update_time": self._last_update_time,
            "config": self.config,
            "is_active": True,
            "message_count": {msg_type.name: 0 for msg_type in MessageType},
            "model_state": self._get_model_state(),
            "metrics": {}
        }
        self.state_manager.register_module(self.module_id, initial_state)
    
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get the state of any neural models in this module.
        Override in subclasses that have neural models.
        """
        return {}
    
    def subscribe(self, message_types: List[MessageType]) -> None:
        """
        Subscribe to specified message types.
        
        Parameters:
        message_types: List of message types to subscribe to
        """
        self.event_bus.subscribe(self.module_id, self._handle_message, message_types)
        self._subscribed_message_types.update(message_types)
        self.logger.debug(f"Subscribed to message types: {[mt.name for mt in message_types]}")
    
    def unsubscribe(self, message_types: Optional[List[MessageType]] = None) -> None:
        """
        Unsubscribe from specified message types or all if None.
        
        Parameters:
        message_types: List of message types to unsubscribe from, or None for all
        """
        if message_types is None:
            message_types = list(self._subscribed_message_types)
            
        self.event_bus.unsubscribe(self.module_id, message_types)
        self._subscribed_message_types.difference_update(message_types)
        self.logger.debug(f"Unsubscribed from message types: {[mt.name for mt in message_types]}")
    
    def publish(self, message_type: MessageType, content: Content, priority: int = 1,
               in_response_to: Optional[MessageID] = None) -> None:
        """
        Publish a message to the event bus.
        
        Parameters:
        message_type: Type of the message
        content: Message content
        priority: Message priority (0-10)
        in_response_to: Optional ID of message this is responding to
        """
        message = Message(
            sender=self.module_id,
            sender_type=self.module_type,
            message_type=message_type,
            content=content,
            priority=priority,
            in_response_to=in_response_to
        )
        self.event_bus.publish(message)
        
        # Update message count in state
        self._update_message_count(message_type)
    
    def _update_message_count(self, message_type: MessageType) -> None:
        """Update the count of messages sent by type."""
        state_path = f"modules.{self.module_id}.message_count.{message_type.name}"
        current_count = self.state_manager.get_state_value(state_path) or 0
        self.state_manager.update_state(state_path, current_count + 1)
    
    def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages from the event bus.
        
        Parameters:
        message: The message to process
        """
        self.logger.debug(f"Received message of type {message.message_type.name} from {message.sender}")
        self.process_message(message)
    
    def update_development(self, global_age: Age, developmental_stage: DevelopmentalStage,
                          time_delta: float) -> None:
        """
        Update the module's developmental level based on global age and stage.
        
        Parameters:
        global_age: Current age of the overall mind
        developmental_stage: Current developmental stage of the mind
        time_delta: Time elapsed since last update in seconds
        """
        # Store previous developmental stage for transition detection
        previous_stage = self._developmental_stage
        self._developmental_stage = developmental_stage
        
        # Calculate development increase based on time and config
        development_rate = self.config["development_rate"]
        
        # Apply any critical period effects
        if self._is_in_critical_period(global_age, developmental_stage):
            development_rate *= self.config["critical_period_sensitivity"]
        
        # Calculate development increment
        increment = development_rate * time_delta * 0.001  # Scale to reasonable values
        
        # Cap development level at 1.0
        self._development_level = min(1.0, self._development_level + increment)
        
        # Update state
        self._last_update_time = time.time()
        self.state_manager.update_state(
            f"modules.{self.module_id}.development_level", 
            self._development_level
        )
        self.state_manager.update_state(
            f"modules.{self.module_id}.last_update_time", 
            self._last_update_time
        )
        
        # Handle developmental stage transitions
        if previous_stage != developmental_stage:
            self._on_developmental_stage_change(previous_stage, developmental_stage)
            self.logger.info(
                f"Module '{self.module_id}' transitioned from {previous_stage.name} to {developmental_stage.name} "
                f"stage at development level {self._development_level:.2f}"
            )
    
    def _is_in_critical_period(self, global_age: Age, developmental_stage: DevelopmentalStage) -> bool:
        """
        Determine if the module is currently in a critical period.
        Override in subclasses to implement module-specific critical periods.
        
        Parameters:
        global_age: Current age of the overall mind
        developmental_stage: Current developmental stage of the mind
        
        Returns:
        True if the module is in a critical period, False otherwise
        """
        return False
    
    def _on_developmental_stage_change(self, previous_stage: DevelopmentalStage, 
                                      new_stage: DevelopmentalStage) -> None:
        """
        Handle transitions between developmental stages.
        Override in subclasses to implement stage-specific behaviors.
        
        Parameters:
        previous_stage: Stage the module was in before
        new_stage: New developmental stage
        """
        pass
    
    def get_development_level(self) -> float:
        """
        Get the current development level of this module.
        
        Returns:
        Development level between 0.0 and 1.0
        """
        return self._development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of this module.
        
        Returns:
        Dictionary containing the module's state
        """
        return self.state_manager.get_state_value(f"modules.{self.module_id}")
    
    def update_state_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics in the module's state.
        
        Parameters:
        metrics: Dictionary of metrics to update
        """
        self.state_manager.update_state(f"modules.{self.module_id}.metrics", metrics)
    
    def cleanup(self) -> None:
        """Clean up resources when the module is being unregistered."""
        self.unsubscribe()
        self.logger.info(f"Module '{self.module_id}' cleaned up")
    
    @abc.abstractmethod
    def process_message(self, message: Message) -> None:
        """
        Process an incoming message from the event bus.
        Must be implemented by subclasses.
        
        Parameters:
        message: The message to process
        """
        pass
    
    @abc.abstractmethod
    def process_input(self, input_data: Any) -> Any:
        """
        Process input data specific to this module.
        Must be implemented by subclasses.
        
        Parameters:
        input_data: Module-specific input data
        
        Returns:
        Processing results
        """
        pass
