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

class DevelopmentError(LMMError):
    """Raised when there's an error in the developmental process"""
    pass

class StorageError(LMMError):
    """Raised when there's an error in storage operations"""
    pass

class VisualizationError(LMMError):
    """Raised when there's an error in visualization components"""
    pass
