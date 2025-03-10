"""
Custom exception classes for the LMM project.

This module provides a comprehensive set of exception classes for handling
various error conditions in the Large Mind Model system. The exceptions are
organized in a hierarchy to allow for specific error handling while maintaining
the ability to catch broader categories of errors.
"""

from typing import Optional, Dict, Any, List
import traceback
from datetime import datetime

class LMMBaseException(Exception):
    """
    Base exception class for all LMM exceptions
    
    Attributes:
        message: Error message
        details: Additional error details
        timestamp: When the error occurred
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }
    
    def __str__(self) -> str:
        """String representation of the exception"""
        if self.details:
            return f"{self.__class__.__name__}: {self.message} - {self.details}"
        return f"{self.__class__.__name__}: {self.message}"

class StorageError(LMMBaseException):
    """Exception raised when there is an error with storage operations"""
    pass

class ConfigurationError(LMMBaseException):
    """
    Exception raised when there is an error with configuration
    
    This can include missing configuration files, invalid configuration values,
    or configuration conflicts.
    """
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 invalid_value: Optional[Any] = None, expected_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
        if expected_type:
            details["expected_type"] = expected_type
        super().__init__(message, details)

class InitializationError(LMMBaseException):
    """
    Exception raised when there is an error during initialization
    
    This can include errors initializing modules, components, or the system as a whole.
    """
    def __init__(self, message: str, component: Optional[str] = None, 
                 dependency_errors: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if component:
            details["component"] = component
        if dependency_errors:
            details["dependency_errors"] = dependency_errors
        super().__init__(message, details)

class ValidationError(LMMBaseException):
    """
    Exception raised when there is an error with data validation
    
    This can include invalid input data, schema violations, or type errors.
    """
    def __init__(self, message: str, field: Optional[str] = None,
                 invalid_value: Optional[Any] = None, expected_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if field:
            details["field"] = field
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
        if expected_type:
            details["expected_type"] = expected_type
        super().__init__(message, details)

class CommunicationError(LMMBaseException):
    """
    Exception raised when there is an error with communication between modules
    
    This can include message delivery failures, timeout errors, or protocol violations.
    """
    def __init__(self, message: str, sender: Optional[str] = None,
                 receiver: Optional[str] = None, message_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if sender:
            details["sender"] = sender
        if receiver:
            details["receiver"] = receiver
        if message_type:
            details["message_type"] = message_type
        super().__init__(message, details)

class DevelopmentError(LMMBaseException):
    """
    Exception raised when there is an error related to developmental progression
    
    This can include invalid development levels, stage transition errors, or
    developmental inconsistencies.
    """
    def __init__(self, message: str, current_level: Optional[float] = None,
                 current_stage: Optional[str] = None, target_level: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if current_level is not None:
            details["current_level"] = current_level
        if current_stage:
            details["current_stage"] = current_stage
        if target_level is not None:
            details["target_level"] = target_level
        super().__init__(message, details)

class ResourceNotFoundError(LMMBaseException):
    """
    Exception raised when a required resource is not found
    
    This can include missing files, unavailable services, or non-existent resources.
    """
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None, resource_path: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        if resource_path:
            details["resource_path"] = resource_path
        super().__init__(message, details)

class LMMError(Exception):
    """
    Base exception for all LMM errors
    
    This is a legacy exception class maintained for backward compatibility.
    New code should use LMMBaseException and its subclasses.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(message)
    
    def __str__(self) -> str:
        """String representation of the exception"""
        if self.details:
            return f"{self.__class__.__name__}: {self.message} - {self.details}"
        return f"{self.__class__.__name__}: {self.message}"

class ModuleInitializationError(LMMError):
    """
    Raised when a module fails to initialize
    
    This can include errors creating module instances, loading module data,
    or establishing module connections.
    """
    def __init__(self, message: str, module_type: Optional[str] = None,
                 module_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if module_type:
            details["module_type"] = module_type
        if module_id:
            details["module_id"] = module_id
        super().__init__(message, details)

class ModuleProcessingError(LMMError):
    """
    Raised when a module fails to process input
    
    This can include errors during input processing, output generation,
    or internal module operations.
    """
    def __init__(self, message: str, module_type: Optional[str] = None,
                 module_id: Optional[str] = None, input_data: Optional[Dict[str, Any]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if module_type:
            details["module_type"] = module_type
        if module_id:
            details["module_id"] = module_id
        if input_data:
            details["input_data"] = str(input_data)
        super().__init__(message, details)

class EventBusError(LMMError):
    """
    Raised when there's an error in the event bus
    
    This can include message publishing errors, subscription errors,
    or event routing failures.
    """
    def __init__(self, message: str, message_type: Optional[str] = None,
                 sender: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if message_type:
            details["message_type"] = message_type
        if sender:
            details["sender"] = sender
        super().__init__(message, details)

class StateManagerError(LMMError):
    """
    Raised when there's an error in the state manager
    
    This can include state update errors, state persistence failures,
    or state retrieval issues.
    """
    def __init__(self, message: str, state_key: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if state_key:
            details["state_key"] = state_key
        if operation:
            details["operation"] = operation
        super().__init__(message, details)

class NeuralSubstrateError(LMMError):
    """
    Raised when there's an error in the neural substrate
    
    This can include neural network errors, learning failures,
    or activation issues.
    """
    def __init__(self, message: str, network_type: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if network_type:
            details["network_type"] = network_type
        if operation:
            details["operation"] = operation
        super().__init__(message, details)

class MotherLLMError(LMMError):
    """
    Raised when there's an error in the Mother LLM interface
    
    This can include communication errors with the LLM API, response parsing issues,
    or teaching strategy failures.
    """
    def __init__(self, message: str, api_endpoint: Optional[str] = None,
                 response_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if api_endpoint:
            details["api_endpoint"] = api_endpoint
        if response_code is not None:
            details["response_code"] = response_code
        super().__init__(message, details)

class VisualizationError(LMMError):
    """
    Raised when there's an error in visualization components
    
    This can include rendering errors, data formatting issues,
    or visualization generation failures.
    """
    def __init__(self, message: str, visualization_type: Optional[str] = None,
                 data_size: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if visualization_type:
            details["visualization_type"] = visualization_type
        if data_size is not None:
            details["data_size"] = data_size
        super().__init__(message, details)

class PerformanceError(LMMError):
    """
    Raised when there's a performance-related error
    
    This can include timeout errors, resource exhaustion issues,
    or performance degradation problems.
    """
    def __init__(self, message: str, component: Optional[str] = None,
                 threshold: Optional[float] = None, actual_value: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if component:
            details["component"] = component
        if threshold is not None:
            details["threshold"] = threshold
        if actual_value is not None:
            details["actual_value"] = actual_value
        super().__init__(message, details)

class SecurityError(LMMError):
    """
    Raised when there's a security-related error
    
    This can include authentication failures, authorization issues,
    or data protection violations.
    """
    def __init__(self, message: str, security_domain: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if security_domain:
            details["security_domain"] = security_domain
        if operation:
            details["operation"] = operation
        super().__init__(message, details)
