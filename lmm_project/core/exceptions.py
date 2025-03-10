"""
Exception classes for the LMM system.
"""
from typing import Optional, Any


class LMMException(Exception):
    """Base exception class for all LMM-specific exceptions."""
    def __init__(self, message: str = "An error occurred in the LMM system"):
        super().__init__(message)
        self.message = message


class ModuleError(LMMException):
    """Exception raised when there's an error in a cognitive module."""
    def __init__(self, module_name: str, message: str = "Error in cognitive module"):
        super().__init__(f"{message}: {module_name}")
        self.module_name = module_name


class EventBusError(LMMException):
    """Exception raised when there's an error in the event bus system."""
    def __init__(self, message: str = "Error in event bus"):
        super().__init__(message)


class MessageError(LMMException):
    """Exception raised when there's an error with a message."""
    def __init__(self, message_id: Optional[str] = None, error_msg: str = "Invalid message"):
        msg = f"{error_msg}"
        if message_id:
            msg = f"{error_msg} (ID: {message_id})"
        super().__init__(msg)
        self.message_id = message_id


class StateError(LMMException):
    """Exception raised when there's an error with the system state."""
    def __init__(self, message: str = "Error in system state"):
        super().__init__(message)


class DevelopmentError(LMMException):
    """Exception raised when there's an error in developmental processes."""
    def __init__(self, message: str = "Error in developmental process"):
        super().__init__(message)


class StorageError(LMMException):
    """Exception raised when there's an error with storage operations."""
    def __init__(self, message: str = "Error in storage operation"):
        super().__init__(message)


class NeuralError(LMMException):
    """Exception raised when there's an error in the neural substrate."""
    def __init__(self, message: str = "Error in neural substrate"):
        super().__init__(message)


class ConfigurationError(LMMException):
    """Exception raised when there's an error in the system configuration."""
    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message)


class ResourceUnavailableError(LMMException):
    """Exception raised when a required resource is unavailable."""
    def __init__(self, resource_name: str, message: Optional[str] = None):
        msg = message or f"Required resource unavailable: {resource_name}"
        super().__init__(msg)
        self.resource_name = resource_name


class InterfaceError(LMMException):
    """Exception raised when there's an error in an external interface."""
    def __init__(self, interface_name: str, message: str = "Error in interface"):
        super().__init__(f"{message}: {interface_name}")
        self.interface_name = interface_name
