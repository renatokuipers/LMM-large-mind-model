from typing import Dict, List, Optional, Any, Tuple, Union, Set
import logging
import re
import json
from pydantic import BaseModel, Field, validator

from schemas.code_entities import CodeComponent, ClassSignature, FunctionSignature
from schemas.llm_io import CodeGenerationResponse
from generators.base import BaseGenerator, GeneratorContext
from core.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)


class APIEndpointSpec(BaseModel):
    """Specification for an API endpoint."""
    
    path: str = Field(..., description="Endpoint path")
    method: str = Field(..., description="HTTP method")
    summary: str = Field(..., description="Endpoint summary")
    description: Optional[str] = Field(None, description="Endpoint description")
    request_model: Optional[str] = Field(None, description="Request model name")
    response_model: Optional[str] = Field(None, description="Response model name")
    status_code: int = Field(200, description="Success status code")
    tags: List[str] = Field(default_factory=list, description="Endpoint tags")
    dependencies: List[str] = Field(default_factory=list, description="Endpoint dependencies")
    requires_auth: bool = Field(False, description="Whether the endpoint requires authentication")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")


class APIRouterSpec(BaseModel):
    """Specification for an API router."""
    
    prefix: str = Field(..., description="Router prefix")
    tags: List[str] = Field(default_factory=list, description="Router tags")
    dependencies: List[str] = Field(default_factory=list, description="Router dependencies")
    endpoints: List[APIEndpointSpec] = Field(default_factory=list, description="Router endpoints")


class APILayerGeneratorContext(BaseModel):
    """Specialized context for API layer generation.
    
    Extends the standard GeneratorContext with API-specific
    information such as endpoints, routers, and request/response models.
    """
    
    framework: str = Field("fastapi", description="API framework to use")
    endpoints: List[APIEndpointSpec] = Field(
        default_factory=list, 
        description="API endpoints to generate"
    )
    router_prefix: Optional[str] = Field(None, description="Router prefix")
    tags: List[str] = Field(default_factory=list, description="API tags")
    request_models: List[str] = Field(
        default_factory=list, 
        description="Request model references"
    )
    response_models: List[str] = Field(
        default_factory=list, 
        description="Response model references"
    )
    service_dependencies: List[str] = Field(
        default_factory=list, 
        description="Service dependencies"
    )
    use_case_dependencies: List[str] = Field(
        default_factory=list, 
        description="Use case dependencies"
    )
    
    def to_generator_context(self) -> GeneratorContext:
        """Convert to a standard generator context.
        
        Returns:
            Standard generator context
        """
        # Process endpoints to add to requirements
        endpoint_requirements = []
        for endpoint in self.endpoints:
            endpoint_req = f"{endpoint.method} {endpoint.path}: {endpoint.summary}"
            if endpoint.request_model:
                endpoint_req += f" (Request: {endpoint.request_model})"
            if endpoint.response_model:
                endpoint_req += f" (Response: {endpoint.response_model})"
            endpoint_requirements.append(endpoint_req)
        
        # Process dependencies
        dependencies = []
        for svc in self.service_dependencies:
            dependencies.append(svc)
        for uc in self.use_case_dependencies:
            dependencies.append(uc)
        
        # Combine all requirements
        all_requirements = []
        
        if self.framework:
            all_requirements.append(f"Use {self.framework} framework")
        
        if self.router_prefix:
            all_requirements.append(f"Create router with prefix '{self.router_prefix}'")
        
        if self.tags:
            all_requirements.append(f"Include tags: {', '.join(self.tags)}")
        
        if endpoint_requirements:
            all_requirements.append("Endpoints:")
            all_requirements.extend([f"- {req}" for req in endpoint_requirements])
        
        if self.request_models:
            all_requirements.append("Use request models:")
            all_requirements.extend([f"- {model}" for model in self.request_models])
        
        if self.response_models:
            all_requirements.append("Use response models:")
            all_requirements.extend([f"- {model}" for model in self.response_models])
        
        # Create standard generator context
        return GeneratorContext(
            component_type="router" if self.router_prefix else "endpoint",
            name=self.name,
            module_path=self.module_path,
            description=self.description,
            requirements=all_requirements,
            dependencies=dependencies,
            additional_context=self.additional_context,
            project_description=self.project_description
        )


class APILayerGenerator(BaseGenerator):
    """Generator for API layer components.
    
    This generator specializes in creating API endpoints, routers,
    and related components for RESTful interfaces.
    """
    
    async def generate(self, context: GeneratorContext) -> CodeComponent:
        """Generate an API layer component.
        
        Args:
            context: Generation context
            
        Returns:
            Generated component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating API component: {context.module_path}.{context.name}")
        
        # Add API layer specific guidance to context
        if context.additional_context:
            context.additional_context += self._get_api_layer_guidance()
        else:
            context.additional_context = self._get_api_layer_guidance()
        
        # Generate code
        try:
            response = await self._generate_with_llm(context)
            
            # Enhance generated code
            enhanced_code = self._enhance_api_code(response.code, framework="fastapi")
            response.code = enhanced_code
            
            # Create component
            component = self._create_component_from_response(context, response)
            
            # Validate component
            self._validate_component(component)
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating API component {context.name}: {str(e)}")
            raise
    
    def _get_api_layer_guidance(self) -> str:
        """Get guidance for API layer generation.
        
        Returns:
            Guidance text
        """
        return """
        ## API Layer Guidelines
        
        When implementing API endpoints, follow these guidelines:
        
        1. Use dependency injection for services and use cases
        2. Use Pydantic models for request/response validation
        3. Add proper response models with appropriate status codes
        4. Include comprehensive error handling with HTTP exception mapping
        5. Document all endpoints with descriptive summaries and parameters
        6. Group related endpoints in routers with appropriate tags
        7. Add authentication and authorization checks where needed
        8. Use appropriate HTTP methods (GET, POST, PUT, DELETE)
        9. Follow REST principles for resource naming and operations
        10. Include pagination for list endpoints
        11. Add proper content negotiation
        12. Use status codes consistently
        
        Example FastAPI Router:
        
        ```python
        router = APIRouter(
            prefix="/users",
            tags=["users"],
            dependencies=[Depends(get_current_user)]
        )
        
        @router.post(
            "/",
            response_model=UserResponse,
            status_code=201,
            summary="Create new user",
            description="Create a new user with the provided data"
        )
        async def create_user(
            user_data: UserCreate,
            user_service: UserService = Depends(get_user_service)
        ) -> UserResponse:
            """Create a new user.
            
            Args:
                user_data: User creation data
                user_service: User service dependency
                
            Returns:
                Created user data
                
            Raises:
                HTTPException: If user creation fails
            """
            try:
                user = await user_service.create_user(user_data)
                return user
            except UserAlreadyExistsError as e:
                raise HTTPException(
                    status_code=409,
                    detail=f"User already exists: {str(e)}"
                )
            except ValidationError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Validation error: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Error creating user: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error"
                )
        
        @router.get(
            "/{user_id}",
            response_model=UserResponse,
            summary="Get user by ID",
            description="Get user details by user ID"
        )
        async def get_user(
            user_id: UUID,
            user_service: UserService = Depends(get_user_service),
            current_user: User = Depends(get_current_user)
        ) -> UserResponse:
            """Get user by ID.
            
            Args:
                user_id: User ID
                user_service: User service dependency
                current_user: Current authenticated user
                
            Returns:
                User data
                
            Raises:
                HTTPException: If user is not found or access is denied
            """
            # Check permissions
            if current_user.id != user_id and not current_user.is_admin:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to access this user"
                )
            
            user = await user_service.get_user(user_id)
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail=f"User with ID {user_id} not found"
                )
                
            return user
        ```
        """
    
    def _enhance_api_code(self, code: str, framework: str = "fastapi") -> str:
        """Enhance API code in the generated code.
        
        This method post-processes the generated code to ensure
        best practices are followed in API implementation.
        
        Args:
            code: Generated code
            framework: API framework used
            
        Returns:
            Enhanced code
        """
        if framework == "fastapi":
            # Ensure proper imports
            if 'from fastapi import' in code:
                # Check for missing imports
                if 'APIRouter' not in code.split('from fastapi import')[1].split('\n')[0]:
                    code = code.replace('from fastapi import', 'from fastapi import APIRouter, ')
                
                if 'HTTPException' not in code.split('from fastapi import')[1].split('\n')[0] and 'HTTPException' in code:
                    code = code.replace('from fastapi import', 'from fastapi import HTTPException, ')
                
                if 'Depends' not in code.split('from fastapi import')[1].split('\n')[0] and 'Depends' in code:
                    code = code.replace('from fastapi import', 'from fastapi import Depends, ')
                
                if 'status' not in code and 'status_code' in code:
                    code = code.replace('from fastapi import', 'from fastapi import status, ')
            else:
                # Add imports
                imports = ['APIRouter']
                if 'HTTPException' in code:
                    imports.append('HTTPException')
                if 'Depends' in code:
                    imports.append('Depends')
                if 'status_code' in code:
                    imports.append('status')
                
                code = f'from fastapi import {", ".join(imports)}\n' + code
            
            # Check for response_model usage
            if '@router.' in code and 'response_model=' not in code:
                # Add response_model to endpoints that don't have it
                endpoint_pattern = re.compile(r'@router\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\'](?:[^)]*)\)', re.DOTALL)
                
                for match in endpoint_pattern.finditer(code):
                    http_method = match.group(1)
                    endpoint_path = match.group(2)
                    endpoint_decorator = match.group(0)
                    
                    # Skip if already has response_model
                    if 'response_model=' in endpoint_decorator:
                        continue
                    
                    # Try to infer response model from function return type
                    function_pattern = re.compile(rf'@router\.{http_method}\([^)]*\)\s*\n(?:.*?)\s*def\s+(\w+)\([^)]*\)\s*->\s*(\w+)', re.DOTALL)
                    function_match = function_pattern.search(code[match.start():])
                    
                    if function_match:
                        return_type = function_match.group(2)
                        if return_type not in ('None', 'Any', 'dict', 'list'):
                            # Add response_model
                            replacement = endpoint_decorator.replace(
                                f'@router.{http_method}(',
                                f'@router.{http_method}('
                            )
                            if ')' in replacement and 'response_model=' not in replacement:
                                replacement = replacement.replace(
                                    ')',
                                    f', response_model={return_type})'
                                )
                            code = code.replace(endpoint_decorator, replacement)
            
            # Add proper status codes if missing
            if '@router.post' in code and 'status_code=' not in code:
                code = code.replace('@router.post(', '@router.post(')
                post_endpoints = re.finditer(r'@router\.post\([^)]*\)', code)
                for match in post_endpoints:
                    if 'status_code=' not in match.group(0):
                        replacement = match.group(0).replace(')', ', status_code=201)')
                        code = code.replace(match.group(0), replacement)
            
            # Add error handling if missing
            for http_method in ('get', 'post', 'put', 'delete', 'patch'):
                method_pattern = re.compile(rf'@router\.{http_method}\([^)]*\)\s*\ndef\s+(\w+)\([^)]*\).*?:', re.DOTALL)
                for match in method_pattern.finditer(code):
                    func_name = match.group(1)
                    # Find function body
                    func_body_pattern = re.compile(rf'def\s+{func_name}\([^)]*\).*?:(.*?)(?=\n\s*@|\n\s*def|\Z)', re.DOTALL)
                    func_body_match = func_body_pattern.search(code)
                    
                    if func_body_match:
                        func_body = func_body_match.group(1)
                        if 'try:' not in func_body and 'HTTPException' not in func_body:
                            # Add try/except block
                            indent = re.match(r'(\s+)', func_body).group(1) if re.match(r'(\s+)', func_body) else '    '
                            indented_body = "\n".join([f"{indent}{line}" for line in func_body.strip().split('\n')])
                            
                            exception_handlers = f"""
{indent}try:
{indented_body}
{indent}except ValidationError as e:
{indent}    raise HTTPException(
{indent}        status_code=422,
{indent}        detail=f"Validation error: {{str(e)}}"
{indent}    )
{indent}except Exception as e:
{indent}    logger.error(f"Error in {func_name}: {{str(e)}}")
{indent}    raise HTTPException(
{indent}        status_code=500,
{indent}        detail="Internal server error"
{indent}    )
"""
                            # Replace function body
                            code = code.replace(func_body, exception_handlers)
        
            # Add logging if missing
            if 'logger' in code and 'logging.getLogger' not in code:
                if 'import logging' not in code:
                    code = 'import logging\n' + code
                
                # Add logger initialization after imports
                if 'logger = logging.getLogger' not in code:
                    # Find a good spot after imports
                    import_section_end = 0
                    for match in re.finditer(r'^(?:import|from)\s+.*$', code, re.MULTILINE):
                        import_section_end = max(import_section_end, match.end())
                    
                    if import_section_end > 0:
                        logger_init = '\n\nlogger = logging.getLogger(__name__)\n'
                        code = code[:import_section_end] + logger_init + code[import_section_end:]
        
        return code
    
    async def generate_router(self, 
                           context: GeneratorContext,
                           router_spec: APIRouterSpec,
                           dependency_signatures: List[ClassSignature]) -> CodeComponent:
        """Generate an API router with endpoints.
        
        Args:
            context: Generation context
            router_spec: Router specification
            dependency_signatures: Signatures of dependencies
            
        Returns:
            Generated router component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating API router: {context.module_path}.{context.name}")
        
        # Add router-specific guidance
        router_guidance = f"""
        ## Router Specification
        
        Generate a FastAPI router with the following specification:
        
        - Prefix: {router_spec.prefix}
        - Tags: {', '.join(router_spec.tags) if router_spec.tags else 'None'}
        
        ### Endpoints:
        
        {json.dumps([endpoint.dict() for endpoint in router_spec.endpoints], indent=2)}
        """
        
        # Add dependency signatures to context
        dependencies_info = "## Dependencies\n\nThis router should use the following dependencies:\n\n"
        
        for dep in dependency_signatures:
            dependencies_info += f"### {dep.name}\n\n"
            dependencies_info += "Methods:\n"
            
            for method in dep.methods:
                # Format method signature
                params = [str(param) for param in method.parameters[1:]]  # Skip self
                params_str = ", ".join(params)
                return_str = f" -> {method.return_type.type_hint}" if method.return_type else ""
                
                method_sig = f"- `{method.name}({params_str}){return_str}`"
                if method.docstring:
                    method_sig += f": {method.docstring.summary}"
                
                dependencies_info += method_sig + "\n"
        
        # Add to context
        if context.additional_context:
            context.additional_context += "\n\n" + router_guidance + "\n\n" + dependencies_info
        else:
            context.additional_context = router_guidance + "\n\n" + dependencies_info
        
        # Generate router
        return await self.generate(context)
    
    async def generate_endpoint(self, 
                             context: GeneratorContext,
                             endpoint_spec: APIEndpointSpec,
                             dependency_signatures: List[ClassSignature]) -> CodeComponent:
        """Generate a single API endpoint.
        
        Args:
            context: Generation context
            endpoint_spec: Endpoint specification
            dependency_signatures: Signatures of dependencies
            
        Returns:
            Generated endpoint component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating API endpoint: {context.module_path}.{context.name}")
        
        # Add endpoint-specific guidance
        endpoint_guidance = f"""
        ## Endpoint Specification
        
        Generate a FastAPI endpoint with the following specification:
        
        - Path: {endpoint_spec.path}
        - Method: {endpoint_spec.method}
        - Summary: {endpoint_spec.summary}
        - Description: {endpoint_spec.description or 'N/A'}
        - Request Model: {endpoint_spec.request_model or 'None'}
        - Response Model: {endpoint_spec.response_model or 'None'}
        - Status Code: {endpoint_spec.status_code}
        - Tags: {', '.join(endpoint_spec.tags) if endpoint_spec.tags else 'None'}
        - Requires Auth: {'Yes' if endpoint_spec.requires_auth else 'No'}
        """
        
        # Add dependency signatures to context
        dependencies_info = "## Dependencies\n\nThis endpoint should use the following dependencies:\n\n"
        
        for dep in dependency_signatures:
            dependencies_info += f"### {dep.name}\n\n"
            dependencies_info += "Methods:\n"
            
            for method in dep.methods:
                # Format method signature
                params = [str(param) for param in method.parameters[1:]]  # Skip self
                params_str = ", ".join(params)
                return_str = f" -> {method.return_type.type_hint}" if method.return_type else ""
                
                method_sig = f"- `{method.name}({params_str}){return_str}`"
                if method.docstring:
                    method_sig += f": {method.docstring.summary}"
                
                dependencies_info += method_sig + "\n"
        
        # Add to context
        if context.additional_context:
            context.additional_context += "\n\n" + endpoint_guidance + "\n\n" + dependencies_info
        else:
            context.additional_context = endpoint_guidance + "\n\n" + dependencies_info
        
        # Generate endpoint
        return await self.generate(context)
    
    async def generate_schema_models(self, 
                                   context: GeneratorContext,
                                   model_specs: List[Dict[str, Any]]) -> CodeComponent:
        """Generate Pydantic schema models for API requests and responses.
        
        Args:
            context: Generation context
            model_specs: Specifications for models to generate
            
        Returns:
            Generated schema models component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating schema models: {context.module_path}.{context.name}")
        
        # Add schema-specific guidance
        schema_guidance = """
        ## Schema Model Guidelines
        
        When implementing schema models, follow these guidelines:
        
        1. Create separate models for requests and responses
        2. Use appropriate field types with validation
        3. Add examples to model Config
        4. Include comprehensive field descriptions
        5. Create model hierarchies for shared fields
        6. Add validators for complex validation rules
        7. Use Optional[] for nullable fields
        8. Implement consistent naming patterns (e.g., UserCreate, UserResponse)
        9. Add Config class with schema_extra for examples
        10. Use Field(...) for required fields
        
        Example Schema Models:
        
        ```python
        class UserBase(BaseModel):
            """Base model with common user fields."""
            
            email: EmailStr = Field(..., description="User email address")
            full_name: str = Field(..., description="User full name")
            
            @validator('email')
            def email_must_be_valid(cls, v):
                # Email validation logic here
                return v
        
        class UserCreate(UserBase):
            """Model for user creation requests."""
            
            password: str = Field(
                ..., 
                description="User password",
                min_length=8
            )
            
            class Config:
                schema_extra = {
                    "example": {
                        "email": "user@example.com",
                        "full_name": "John Doe",
                        "password": "securepassword"
                    }
                }
        
        class UserResponse(UserBase):
            """Model for user response data."""
            
            id: UUID = Field(..., description="User ID")
            is_active: bool = Field(True, description="Whether user is active")
            created_at: datetime = Field(..., description="Account creation timestamp")
            
            class Config:
                orm_mode = True
                schema_extra = {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "email": "user@example.com",
                        "full_name": "John Doe",
                        "is_active": True,
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                }
        ```
        """
        
        # Add model specs to context
        models_info = "## Models to Generate\n\n"
        
        for i, model_spec in enumerate(model_specs):
            models_info += f"### Model {i+1}: {model_spec.get('name', f'Model{i+1}')}\n\n"
            models_info += f"Description: {model_spec.get('description', 'N/A')}\n\n"
            
            if 'fields' in model_spec:
                models_info += "Fields:\n"
                for field in model_spec['fields']:
                    field_name = field.get('name', 'unknown')
                    field_type = field.get('type', 'Any')
                    field_desc = field.get('description', '')
                    field_required = 'Required' if field.get('required', True) else 'Optional'
                    
                    models_info += f"- {field_name}: {field_type} ({field_required})"
                    if field_desc:
                        models_info += f" - {field_desc}"
                    models_info += "\n"
            
            if 'example' in model_spec:
                models_info += "Example:\n```json\n"
                models_info += json.dumps(model_spec['example'], indent=2)
                models_info += "\n```\n"
        
        # Add to context
        if context.additional_context:
            context.additional_context += "\n\n" + schema_guidance + "\n\n" + models_info
        else:
            context.additional_context = schema_guidance + "\n\n" + models_info
        
        # Generate schema models
        return await self.generate(context)