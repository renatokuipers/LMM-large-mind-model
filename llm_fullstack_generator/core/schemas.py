# core/schemas.py
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re

class Parameter(BaseModel):
    name: str
    type_hint: str
    default_value: Optional[str] = None
    description: Optional[str] = None
    
    @validator('name')
    def validate_parameter_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"Invalid parameter name: {v}")
        return v

class FunctionSignature(BaseModel):
    name: str
    parameters: List[Parameter] = []
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    file_path: str
    
    @validator('name')
    def validate_function_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"Invalid function name: {v}")
        return v

class ClassSignature(BaseModel):
    name: str
    methods: List[FunctionSignature] = []
    attributes: List[Parameter] = []
    base_classes: List[str] = []
    docstring: Optional[str] = None
    file_path: str
    
    @validator('name')
    def validate_class_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"Invalid class name: {v}")
        return v

class FileStructure(BaseModel):
    file_path: str
    functions: List[FunctionSignature] = []
    classes: List[ClassSignature] = []
    imports: List[str] = []
    
class Task(BaseModel):
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Short descriptive title")
    description: str = Field(..., description="Detailed task description")
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks this task depends on")
    output_files: List[str] = Field(default_factory=list, description="Files this task will create or modify")
    
    @validator('id')
    def validate_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"Invalid task ID: {v}")
        return v
    
class Epic(BaseModel):
    id: str = Field(..., description="Unique epic identifier")
    title: str = Field(..., description="Epic title")
    description: str = Field(..., description="Epic description")
    tasks: List[Task] = Field(default_factory=list)
    
    @validator('id')
    def validate_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"Invalid epic ID: {v}")
        return v

class ProjectConfig(BaseModel):
    language: Literal["python", "nodejs"] = "python" 
    framework: Optional[str] = None
    database: Optional[str] = None
    include_frontend: bool = False
    frontend_framework: Optional[str] = None