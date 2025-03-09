"""
Custom exception classes for the LMM project.
"""

class LMMBaseException(Exception):
    """Base exception class for all LMM exceptions"""
    pass

class StorageError(LMMBaseException):
    """Exception raised when there is an error with storage operations"""
    pass

class ConfigurationError(LMMBaseException):
    """Exception raised when there is an error with configuration"""
    pass

class InitializationError(LMMBaseException):
    """Exception raised when there is an error during initialization"""
    pass

class ValidationError(LMMBaseException):
    """Exception raised when there is an error with data validation"""
    pass

class CommunicationError(LMMBaseException):
    """Exception raised when there is an error with communication between modules"""
    pass

class DevelopmentError(LMMBaseException):
    """Exception raised when there is an error related to developmental progression"""
    pass

class ResourceNotFoundError(LMMBaseException):
    """Exception raised when a required resource is not found"""
    pass

class LMMError(Exception):
    """Base exception for all LMM errors"""
    pass

class ModuleInitializationError(LMMError):
    """Raised when a module fails to initialize"""
    pass

class ModuleProcessingError(LMMError):
    """Raised when a module fails to process input"""
    pass

class EventBusError(LMMError):
    """Raised when there's an error in the event bus"""
    pass

class StateManagerError(LMMError):
    """Raised when there's an error in the state manager"""
    pass

class NeuralSubstrateError(LMMError):
    """Raised when there's an error in the neural substrate"""
    pass

class MotherLLMError(LMMError):
    """Raised when there's an error in the Mother LLM interface"""
    pass

class VisualizationError(LMMError):
    """Raised when there's an error in visualization components"""
    pass
