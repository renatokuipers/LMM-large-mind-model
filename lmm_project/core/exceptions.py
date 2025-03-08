class LMMException(Exception): 
    """Base exception for LMM project""" 
    pass 
 
class ModuleInitializationError(LMMException): 
    """Raised when a module fails to initialize""" 
    pass 
