from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, validator


class LLMMessage(BaseModel):
    """Represents a message in an LLM conversation."""
    
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class LLMRequest(BaseModel):
    """Request to the LLM API."""
    
    model: str = Field(..., description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    temperature: float = Field(0.2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    response_format: Optional[Dict[str, str]] = Field(None, description="Format for the response")
    stream: bool = Field(False, description="Whether to stream the response")

    @validator('messages')
    def validate_messages(cls, v):
        # Ensure messages have valid role and content
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError(f"Message must be a dictionary, got {type(msg)}")
            
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Message must have 'role' and 'content' fields")
            
            if msg['role'] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")
        
        return v


class LLMResponse(BaseModel):
    """Response from the LLM API."""
    
    content: str = Field(..., description="Generated content")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")


class CodeGenerationRequest(BaseModel):
    """Request for generating a code component."""
    
    component_type: Literal["function", "method", "class", "module"] = Field(
        ..., description="Type of code component to generate"
    )
    name: str = Field(..., description="Name of the component")
    module_path: str = Field(..., description="Module path where the component will be defined")
    description: str = Field(..., description="Detailed description of the component")
    requirements: str = Field(..., description="Specific requirements for the component")
    additional_context: Optional[str] = Field(None, description="Additional context information")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Component name cannot be empty")
        return v


class CodeGenerationResponse(BaseModel):
    """Response containing generated code."""
    
    code: str = Field(..., description="Generated code")
    explanation: str = Field("", description="Explanation of the code")
    imports: List[str] = Field(default_factory=list, description="Required import statements")
    dependencies: List[str] = Field(default_factory=list, description="Component dependencies")


class ArchitectureGenerationRequest(BaseModel):
    """Request for generating system architecture."""
    
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(..., description="Description of the project")
    requirements: List[str] = Field(..., description="List of project requirements")
    constraints: Optional[List[str]] = Field(None, description="List of project constraints")
    
    @validator('project_name')
    def validate_project_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Project name cannot be empty")
        return v


class ComponentDefinition(BaseModel):
    """Definition of a code component in the architecture."""
    
    name: str = Field(..., description="Component name")
    type: Literal["class", "function", "module"] = Field(..., description="Component type")
    module_path: str = Field(..., description="Module path")
    description: str = Field(..., description="Component description")
    responsibilities: List[str] = Field(..., description="Component responsibilities")
    dependencies: List[str] = Field(default_factory=list, description="Component dependencies")


class ArchitectureGenerationResponse(BaseModel):
    """Response containing generated architecture."""
    
    project_name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    components: List[ComponentDefinition] = Field(..., description="Components in the architecture")
    data_models: List[Dict[str, Any]] = Field(default_factory=list, description="Data models")
    api_endpoints: Optional[List[Dict[str, Any]]] = Field(None, description="API endpoints if relevant")