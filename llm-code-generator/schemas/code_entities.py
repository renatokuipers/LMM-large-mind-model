from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, validator


class Parameter(BaseModel):
    """Represents a function or method parameter with its type and optional default value."""
    
    name: str = Field(..., description="Parameter name")
    type_hint: Optional[str] = Field(None, description="Type hint as a string")
    default_value: Optional[str] = Field(None, description="Default value as a string if available")
    is_required: bool = Field(True, description="Whether this parameter is required")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Parameter name cannot be empty")
        return v
    
    def __str__(self) -> str:
        """String representation of the parameter for signature display."""
        result = self.name
        if self.type_hint:
            result += f": {self.type_hint}"
        if self.default_value is not None:
            result += f" = {self.default_value}"
        return result


class ReturnType(BaseModel):
    """Represents a function's return type information."""
    
    type_hint: str = Field(..., description="Return type as a string")
    description: Optional[str] = Field(None, description="Description of the return value")


class DocstringInfo(BaseModel):
    """Structured representation of a function/method/class docstring."""
    
    summary: str = Field("", description="Brief summary of the component")
    description: Optional[str] = Field(None, description="Detailed description")
    param_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Parameter descriptions keyed by parameter name"
    )
    return_description: Optional[str] = Field(None, description="Description of the return value")
    examples: List[str] = Field(default_factory=list, description="Usage examples")
    
    def is_empty(self) -> bool:
        """Check if this docstring has any meaningful content."""
        return (
            not self.summary and 
            not self.description and 
            not self.param_descriptions and
            not self.return_description and
            not self.examples
        )


class FunctionSignature(BaseModel):
    """Represents a function signature with all its metadata."""
    
    name: str = Field(..., description="Function name")
    parameters: List[Parameter] = Field(default_factory=list, description="Function parameters")
    return_type: Optional[ReturnType] = Field(None, description="Function return type")
    docstring: DocstringInfo = Field(default_factory=DocstringInfo, description="Function docstring")
    module_path: str = Field(..., description="Module path where this function is defined")
    is_async: bool = Field(False, description="Whether this is an async function")
    decorators: List[str] = Field(default_factory=list, description="Function decorators")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Function name cannot be empty")
        return v
    
    def get_signature_str(self) -> str:
        """Generate a string representation of the function signature."""
        prefix = "async " if self.is_async else ""
        params_str = ", ".join(str(param) for param in self.parameters)
        return_str = f" -> {self.return_type.type_hint}" if self.return_type else ""
        
        return f"{prefix}def {self.name}({params_str}){return_str}"
    
    def is_compatible_with(self, other: 'FunctionSignature') -> bool:
        """Check if this function signature is compatible with another one (for memory validation)."""
        # Check name and basic properties
        if self.name != other.name or self.is_async != other.is_async:
            return False
            
        # Basic parameter count check (excluding those with defaults)
        self_required = [p for p in self.parameters if p.is_required]
        other_required = [p for p in other.parameters if p.is_required]
        
        if len(self_required) != len(other_required):
            return False
            
        # TODO: Implement more sophisticated compatibility checking
        # - Type compatibility (accounting for subtyping)
        # - Return type compatibility
        
        return True


class AttributeSignature(BaseModel):
    """Represents a class attribute with its type information."""
    
    name: str = Field(..., description="Attribute name")
    type_hint: Optional[str] = Field(None, description="Type hint as string")
    default_value: Optional[str] = Field(None, description="Default value as string if available")
    is_property: bool = Field(False, description="Whether this is a @property")
    docstring: Optional[str] = Field(None, description="Attribute docstring")


class ClassSignature(BaseModel):
    """Represents a class signature with all its methods and attributes."""
    
    name: str = Field(..., description="Class name")
    base_classes: List[str] = Field(default_factory=list, description="Base classes")
    methods: List[FunctionSignature] = Field(default_factory=list, description="Class methods")
    attributes: List[AttributeSignature] = Field(default_factory=list, description="Class attributes")
    docstring: DocstringInfo = Field(default_factory=DocstringInfo, description="Class docstring")
    module_path: str = Field(..., description="Module path where this class is defined")
    decorators: List[str] = Field(default_factory=list, description="Class decorators")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Class name cannot be empty")
        return v
    
    def get_signature_str(self) -> str:
        """Generate a string representation of the class signature."""
        base_str = ", ".join(self.base_classes) if self.base_classes else "object"
        return f"class {self.name}({base_str})"
    
    def get_method(self, name: str) -> Optional[FunctionSignature]:
        """Get a method by name if it exists."""
        for method in self.methods:
            if method.name == name:
                return method
        return None


class ModuleInfo(BaseModel):
    """Information about a Python module."""
    
    path: str = Field(..., description="Path to the module")
    classes: List[ClassSignature] = Field(default_factory=list, description="Classes in the module")
    functions: List[FunctionSignature] = Field(default_factory=list, description="Functions in the module")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    docstring: Optional[str] = Field(None, description="Module docstring")


class CodeComponent(BaseModel):
    """Base model for various code components that can be incrementally generated."""
    
    component_type: Literal["function", "method", "class", "module"] = Field(
        ..., description="Type of code component"
    )
    name: str = Field(..., description="Component name")
    module_path: str = Field(..., description="Module path where this component is defined")
    signature: Union[FunctionSignature, ClassSignature] = Field(
        ..., description="Component signature"
    )
    implementation: Optional[str] = Field(None, description="Full implementation code")
    dependencies: List[str] = Field(
        default_factory=list, 
        description="List of components this component depends on"
    )
    
    @validator('signature')
    def validate_signature_type(cls, v, values):
        component_type = values.get('component_type')
        if component_type in ("function", "method") and not isinstance(v, FunctionSignature):
            raise ValueError(f"Expected FunctionSignature for {component_type}, got {type(v)}")
        if component_type == "class" and not isinstance(v, ClassSignature):
            raise ValueError(f"Expected ClassSignature for class, got {type(v)}")
        return v