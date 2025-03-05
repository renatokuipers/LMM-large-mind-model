"""
Base component module for the NeuralChild project.

This module defines the BaseComponent class that serves as the foundation
for all psychological components in the system.
"""

import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from abc import ABC, abstractmethod
import json
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

from ..config import DevelopmentalStage


class ConnectionType(str, Enum):
    """Types of connections between components."""
    EXCITATORY = "excitatory"  # Positive influence
    INHIBITORY = "inhibitory"  # Negative influence
    MODULATORY = "modulatory"  # Modifies other connections
    FEEDFORWARD = "feedforward"  # Forward signal propagation
    FEEDBACK = "feedback"  # Signal return/feedback
    LATERAL = "lateral"  # Same-level connections
    CONTEXT = "context"  # Context-providing connections


class Connection(BaseModel):
    """Represents a connection between components."""
    target_id: str
    connection_type: ConnectionType
    strength: float = Field(default=0.0, ge=-1.0, le=1.0)
    plasticity: float = Field(default=0.5, ge=0.0, le=1.0)  # How quickly connection can change
    created_at_stage: DevelopmentalStage
    
    class Config:
        arbitrary_types_allowed = True


class ComponentState(BaseModel):
    """Represents the state of a component that can be serialized/deserialized."""
    id: str
    name: str
    activation: float
    last_activation_values: List[float]
    activation_threshold: float
    connections: List[Connection]
    development_stage: DevelopmentalStage
    activation_decay_rate: float
    learning_rate: float
    experience_count: int
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NeuralComponent(ABC):
    """
    Base class for all psychological components.
    
    This serves as the foundation for specialized neural components that
    model different aspects of psychological functions.
    """
    
    def __init__(
        self,
        name: str,
        activation_threshold: float = 0.3,
        activation_decay_rate: float = 0.1,
        learning_rate: float = 0.05,
        development_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
        component_id: Optional[str] = None
    ):
        """
        Initialize the component with basic properties.
        
        Args:
            name: Human-readable name of the component
            activation_threshold: Minimum activation required to fire
            activation_decay_rate: Rate at which activation decays
            learning_rate: Rate at which the component learns
            development_stage: Current developmental stage
            component_id: Optional ID (generated if not provided)
        """
        self.id = component_id or str(uuid.uuid4())
        self.name = name
        self.activation = 0.0  # Current activation level (0.0 to 1.0)
        self.last_activation_values = [0.0] * 10  # Store last 10 activation values
        self.activation_threshold = activation_threshold
        self.connections: List[Connection] = []
        self.development_stage = development_stage
        self.activation_decay_rate = activation_decay_rate
        self.learning_rate = learning_rate
        self.experience_count = 0  # Count of experiences processed
        self.confidence = 0.0  # Confidence in outputs (0.0 to 1.0)
        self.metadata: Dict[str, Any] = {}  # For component-specific attributes
    
    def initialize(self) -> None:
        """Initialize component state. Called at the start of a simulation."""
        self.activation = 0.0
        self.last_activation_values = [0.0] * 10
        # Any other initialization goes here
    
    def activate(self, stimulus: float) -> float:
        """
        Activate the component with a stimulus.
        
        Args:
            stimulus: Activation stimulus value (0.0 to 1.0)
            
        Returns:
            The new activation value
        """
        # Update activation based on stimulus
        new_activation = min(1.0, max(0.0, self.activation + stimulus))
        
        # Apply threshold logic
        if new_activation >= self.activation_threshold:
            # Store activation history
            self.last_activation_values.pop(0)
            self.last_activation_values.append(new_activation)
            self.activation = new_activation
            return new_activation
        else:
            # Apply decay
            self.activation = max(0.0, self.activation - self.activation_decay_rate)
            return self.activation
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and produce outputs.
        
        This is the main method for component-specific processing logic.
        Must be implemented by subclasses.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        pass
    
    def get_average_activation(self) -> float:
        """Get the average activation over recent history."""
        return sum(self.last_activation_values) / len(self.last_activation_values)
    
    def add_connection(
        self,
        target_id: str,
        connection_type: ConnectionType,
        initial_strength: float = 0.1,
        plasticity: float = 0.5
    ) -> None:
        """
        Add a connection to another component.
        
        Args:
            target_id: ID of the target component
            connection_type: Type of connection
            initial_strength: Initial connection strength (-1.0 to 1.0)
            plasticity: How quickly the connection can change (0.0 to 1.0)
        """
        connection = Connection(
            target_id=target_id,
            connection_type=connection_type,
            strength=initial_strength,
            plasticity=plasticity,
            created_at_stage=self.development_stage
        )
        self.connections.append(connection)
    
    def remove_connection(self, target_id: str) -> bool:
        """
        Remove a connection to another component.
        
        Args:
            target_id: ID of the target component
            
        Returns:
            True if connection was removed, False if not found
        """
        initial_count = len(self.connections)
        self.connections = [c for c in self.connections if c.target_id != target_id]
        return len(self.connections) < initial_count
    
    def get_connection(self, target_id: str) -> Optional[Connection]:
        """
        Get a connection to a specific component.
        
        Args:
            target_id: ID of the target component
            
        Returns:
            Connection object if found, None otherwise
        """
        for connection in self.connections:
            if connection.target_id == target_id:
                return connection
        return None
    
    def update_connection_strength(
        self,
        target_id: str,
        strength_delta: float
    ) -> bool:
        """
        Update the strength of a connection.
        
        Args:
            target_id: ID of the target component
            strength_delta: Change in connection strength
            
        Returns:
            True if connection was updated, False if not found
        """
        connection = self.get_connection(target_id)
        if connection:
            # Apply plasticity factor to the strength change
            adjusted_delta = strength_delta * connection.plasticity
            
            # Update strength within bounds
            connection.strength = min(1.0, max(-1.0, connection.strength + adjusted_delta))
            return True
        return False
    
    def connect_to(
        self,
        target_component: 'NeuralComponent',
        connection_type: ConnectionType,
        initial_strength: float = 0.1,
        plasticity: float = 0.5
    ) -> None:
        """
        Connect this component to another component.
        
        Args:
            target_component: Target component to connect to
            connection_type: Type of connection
            initial_strength: Initial connection strength (-1.0 to 1.0)
            plasticity: How quickly the connection can change (0.0 to 1.0)
        """
        self.add_connection(target_component.id, connection_type, initial_strength, plasticity)
    
    def update_connections(
        self,
        co_activated_components: Set[str],
        non_activated_components: Set[str]
    ) -> None:
        """
        Update connection strengths based on Hebbian learning.
        
        "Neurons that fire together, wire together"
        
        Args:
            co_activated_components: Set of IDs of components that activated together
            non_activated_components: Set of IDs of components that didn't activate
        """
        # Strengthen connections to co-activated components
        for target_id in co_activated_components:
            connection = self.get_connection(target_id)
            if connection:
                self.update_connection_strength(target_id, self.learning_rate)
            else:
                # Create new connection if in developmental stage where this is possible
                if self.development_stage != DevelopmentalStage.PRENATAL:
                    self.add_connection(
                        target_id,
                        ConnectionType.EXCITATORY,
                        initial_strength=self.learning_rate,
                        plasticity=0.5
                    )
        
        # Weaken connections to non-activated components
        for target_id in non_activated_components:
            connection = self.get_connection(target_id)
            if connection:
                self.update_connection_strength(target_id, -self.learning_rate * 0.5)
    
    def set_development_stage(self, new_stage: DevelopmentalStage) -> None:
        """
        Set the developmental stage of this component.
        This is an alias for update_stage for backward compatibility.
        
        Args:
            new_stage: New developmental stage
        """
        self.update_stage(new_stage)
    
    def update_stage(self, new_stage: DevelopmentalStage) -> None:
        """
        Update the developmental stage of this component.
        
        Args:
            new_stage: New developmental stage
        """
        old_stage = self.development_stage
        self.development_stage = new_stage
        
        # Perform stage-specific updates
        self._on_stage_transition(old_stage, new_stage)
    
    def _on_stage_transition(
        self,
        old_stage: DevelopmentalStage,
        new_stage: DevelopmentalStage
    ) -> None:
        """
        Called when transitioning between developmental stages.
        
        This can be overridden by subclasses to implement stage-specific behavior.
        
        Args:
            old_stage: Previous developmental stage
            new_stage: New developmental stage
        """
        # Default implementation adjusts learning rate based on stage
        stage_factors = {
            DevelopmentalStage.PRENATAL: 0.01,
            DevelopmentalStage.INFANCY: 0.05,
            DevelopmentalStage.EARLY_CHILDHOOD: 0.08,
            DevelopmentalStage.MIDDLE_CHILDHOOD: 0.06,
            DevelopmentalStage.ADOLESCENCE: 0.04,
            DevelopmentalStage.EARLY_ADULTHOOD: 0.02,
            DevelopmentalStage.MID_ADULTHOOD: 0.01
        }
        
        # Update learning rate based on developmental stage
        base_learning_rate = self.learning_rate
        self.learning_rate = base_learning_rate * stage_factors.get(new_stage, 1.0)
    
    def update_confidence(self, success_rate: float) -> None:
        """
        Update component confidence based on success rate.
        
        Args:
            success_rate: Rate of successful processing (0.0 to 1.0)
        """
        # Increase experience count
        self.experience_count += 1
        
        # Update confidence using exponential moving average
        alpha = min(0.1, 10.0 / self.experience_count)  # Dynamic learning rate
        self.confidence = (1 - alpha) * self.confidence + alpha * success_rate
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.
        
        Returns:
            Dictionary representation of the component
        """
        state = ComponentState(
            id=self.id,
            name=self.name,
            activation=self.activation,
            last_activation_values=self.last_activation_values,
            activation_threshold=self.activation_threshold,
            connections=self.connections,
            development_stage=self.development_stage,
            activation_decay_rate=self.activation_decay_rate,
            learning_rate=self.learning_rate,
            experience_count=self.experience_count,
            confidence=self.confidence,
            metadata=self.metadata
        )
        
        return state.dict()
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize data and update component state.
        
        Args:
            data: Dictionary representation of component state
        """
        state = ComponentState(**data)
        
        self.id = state.id
        self.name = state.name
        self.activation = state.activation
        self.last_activation_values = state.last_activation_values
        self.activation_threshold = state.activation_threshold
        self.connections = state.connections
        self.development_stage = state.development_stage
        self.activation_decay_rate = state.activation_decay_rate
        self.learning_rate = state.learning_rate
        self.experience_count = state.experience_count
        self.confidence = state.confidence
        self.metadata = state.metadata
    
    def __str__(self) -> str:
        """String representation of the component."""
        return f"{self.name} (ID: {self.id}, Stage: {self.development_stage.value}, Activation: {self.activation:.2f})"