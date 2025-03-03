class CodeGeneratorError(Exception):
    """Base exception for code generator errors."""
    pass


class ValidationError(CodeGeneratorError):
    """Exception raised when generated code fails validation."""
    
    def __init__(self, message: str, errors=None):
        super().__init__(message)
        self.errors = errors or []
        
class PlanningError(CodeGeneratorError):
    """Exception raised when generated planning fails."""
    
    def __init__(self, message: str, errors=None):
        super().__init__(message)
        self.errors = errors or []


class LLMError(CodeGeneratorError):
    """Exception raised when there is an error communicating with the LLM."""
    pass


class ParseError(CodeGeneratorError):
    """Exception raised when there is an error parsing LLM output."""
    pass


class DependencyError(CodeGeneratorError):
    """Exception raised when a component has missing dependencies."""
    
    def __init__(self, component_name: str, missing_dependencies=None):
        message = f"Component {component_name} has missing dependencies: {missing_dependencies}"
        super().__init__(message)
        self.component_name = component_name
        self.missing_dependencies = missing_dependencies or []


class CodeMemoryError(CodeGeneratorError):
    """Exception raised when there is an error with the code memory system."""
    pass


class ProjectError(CodeGeneratorError):
    """Exception raised when there is an error with the project management."""
    pass