# Core module for the Large Mind Model (LMM) project
"""
Core module for the Large Mind Model (LMM) project.

This module provides the foundational architecture components including:
- Mind: Central coordinator for all cognitive modules
- EventBus: Communication system for inter-module messaging
- Message: Structured message format for module communication
- StateManager: System for tracking and persisting cognitive state
- Exception classes: Specialized error types for the LMM system
"""

from lmm_project.core.mind import Mind
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.core.state_manager import StateManager
from lmm_project.core.types import (
    DevelopmentalStage, ModuleType, ActivationType, LearningType,
    ConnectionType, EmotionType, MemoryType, RelationshipType,
    HomeostaticSignalType, Activation, ModuleStateDict,
    HomeostasisStateDict, SystemStateDict
)
from lmm_project.core.exceptions import (
    LMMBaseException, LMMError, ModuleInitializationError, ModuleProcessingError,
    EventBusError, StateManagerError, NeuralSubstrateError,
    MotherLLMError, DevelopmentError, StorageError, VisualizationError,
    ConfigurationError, InitializationError, ValidationError,
    CommunicationError, ResourceNotFoundError, PerformanceError, SecurityError
)

__all__ = [
    'Mind',
    'EventBus',
    'Message',
    'StateManager',
    # Types
    'DevelopmentalStage',
    'ModuleType',
    'ActivationType',
    'LearningType',
    'ConnectionType',
    'EmotionType',
    'MemoryType',
    'RelationshipType',
    'HomeostaticSignalType',
    'Activation',
    'ModuleStateDict',
    'HomeostasisStateDict',
    'SystemStateDict',
    # Exceptions
    'LMMBaseException',
    'LMMError',
    'ModuleInitializationError',
    'ModuleProcessingError',
    'EventBusError',
    'StateManagerError',
    'NeuralSubstrateError',
    'MotherLLMError',
    'DevelopmentError',
    'StorageError',
    'VisualizationError',
    'ConfigurationError',
    'InitializationError',
    'ValidationError',
    'CommunicationError',
    'ResourceNotFoundError',
    'PerformanceError',
    'SecurityError'
] 
