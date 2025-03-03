from typing import Dict, List, Optional, Any, Tuple
import logging
import re
from pydantic import BaseModel, Field, validator

from schemas.code_entities import CodeComponent, ClassSignature
from schemas.llm_io import CodeGenerationResponse
from generators.base import BaseGenerator, GeneratorContext
from core.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)


class DataModelGeneratorContext(BaseModel):
    """Specialized context for data model generation.
    
    Extends the standard GeneratorContext with data model-specific
    information such as fields, relationships, and validation rules.
    """
    
    fields: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Fields in the data model"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Relationships to other models"
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Validation rules for fields"
    )
    is_pydantic_model: bool = Field(
        True, 
        description="Whether this is a Pydantic model"
    )
    
    def to_generator_context(self) -> GeneratorContext:
        """Convert to a standard generator context.
        
        Returns:
            Standard generator context
        """
        # Process fields to add to requirements
        field_requirements = []
        for field in self.fields:
            field_name = field.get("name", "unknown")
            field_type = field.get("type", "Any")
            field_desc = field.get("description", "")
            field_req = f"Field '{field_name}' of type '{field_type}'{f': {field_desc}' if field_desc else ''}"
            field_requirements.append(field_req)
        
        # Process relationships
        relationship_requirements = []
        for rel in self.relationships:
            rel_type = rel.get("type", "unknown")
            rel_target = rel.get("target_model", "unknown")
            rel_desc = rel.get("description", "")
            rel_req = f"{rel_type} relationship to {rel_target}{f': {rel_desc}' if rel_desc else ''}"
            relationship_requirements.append(rel_req)
        
        # Process validation rules
        validation_requirements = []
        for rule in self.validation_rules:
            rule_field = rule.get("field", "unknown")
            rule_type = rule.get("rule_type", "unknown")
            rule_desc = rule.get("description", "")
            rule_req = f"Validate {rule_field} with {rule_type}{f': {rule_desc}' if rule_desc else ''}"
            validation_requirements.append(rule_req)
        
        # Combine all requirements
        all_requirements = []
        if field_requirements:
            all_requirements.append("Fields:")
            all_requirements.extend([f"- {req}" for req in field_requirements])
        
        if relationship_requirements:
            all_requirements.append("Relationships:")
            all_requirements.extend([f"- {req}" for req in relationship_requirements])
        
        if validation_requirements:
            all_requirements.append("Validation Rules:")
            all_requirements.extend([f"- {req}" for req in validation_requirements])
        
        if self.is_pydantic_model:
            all_requirements.append("- Implement as a Pydantic model with appropriate validators")
        
        # Create standard generator context
        return GeneratorContext(
            component_type="class",
            name=self.name,
            module_path=self.module_path,
            description=self.description,
            requirements=all_requirements,
            dependencies=self.dependencies,
            additional_context=self.additional_context,
            project_description=self.project_description
        )


class DataLayerGenerator(BaseGenerator):
    """Generator for data layer components.
    
    This generator specializes in creating data models, schemas,
    and database-related components with robust validation.
    """
    
    async def generate(self, context: GeneratorContext) -> CodeComponent:
        """Generate a data layer component.
        
        Args:
            context: Generation context
            
        Returns:
            Generated component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating data layer component: {context.module_path}.{context.name}")
        
        # Add data layer specific guidance to context
        if context.additional_context:
            context.additional_context += self._get_data_layer_guidance()
        else:
            context.additional_context = self._get_data_layer_guidance()
        
        # Generate code
        try:
            response = await self._generate_with_llm(context)
            
            # Post-process the generated code to enhance Pydantic usage
            enhanced_code = self._enhance_pydantic_models(response.code)
            response.code = enhanced_code
            
            # Create component
            component = self._create_component_from_response(context, response)
            
            # Validate component
            self._validate_component(component)
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating data layer component {context.name}: {str(e)}")
            raise
    
    def _get_data_layer_guidance(self) -> str:
        """Get guidance for data layer generation.
        
        Returns:
            Guidance text
        """
        return """
        ## Data Layer Guidelines
        
        When implementing data models, follow these guidelines:
        
        1. Use Pydantic BaseModel for all data models
        2. Include comprehensive field validation with validators
        3. Add descriptive docstrings for models and fields
        4. Use proper type annotations (str, int, bool, etc.)
        5. Implement custom validators for complex validation logic
        6. Add Config class with appropriate settings
        7. Use Field() with description and validation params
        8. Define relationships between models clearly
        9. Include examples in docstrings
        10. Handle nullable fields properly with Optional[]
        11. Use appropriate default values where needed
        
        Example Pydantic Model Structure:
        
        ```python
        class ModelName(BaseModel):
            """Model description.
            
            Detailed explanation of the model's purpose and usage.
            
            Attributes:
                field_name: Description of the field
                another_field: Description of another field
            
            Example:
                ```python
                model = ModelName(field_name="value", another_field=123)
                ```
            """
            
            field_name: str = Field(..., description="Description of the field")
            another_field: int = Field(0, description="Description of another field")
            optional_field: Optional[str] = Field(None, description="Optional field")
            
            @validator('field_name')
            def validate_field_name(cls, v):
                if not v:
                    raise ValueError("field_name cannot be empty")
                return v
            
            class Config:
                # Configuration options
                extra = "forbid"  # Prevent extra fields
                schema_extra = {
                    "example": {
                        "field_name": "example value",
                        "another_field": 123
                    }
                }
        ```
        """
    
    def _enhance_pydantic_models(self, code: str) -> str:
        """Enhance Pydantic models in the generated code.
        
        This method post-processes the generated code to ensure
        best practices are followed in Pydantic model implementation.
        
        Args:
            code: Generated code
            
        Returns:
            Enhanced code
        """
        # Look for Pydantic models (classes that inherit from BaseModel)
        model_pattern = re.compile(r'class\s+(\w+)\s*\(\s*BaseModel\s*\):')
        
        if not model_pattern.search(code):
            # No Pydantic models found
            return code
        
        # Ensure proper imports
        if 'from pydantic import' not in code:
            code = 'from pydantic import BaseModel, Field, validator\nfrom typing import Optional, List, Dict, Any\n\n' + code
        elif 'Field' not in code and 'from pydantic import' in code:
            code = code.replace('from pydantic import', 'from pydantic import Field, ')
        
        # Check if validators are used but not imported
        if '@validator' in code and 'validator' not in code.split('from pydantic import')[1].split('\n')[0]:
            code = code.replace('from pydantic import', 'from pydantic import validator, ')
        
        # Ensure Optional is imported if used
        if 'Optional[' in code and 'from typing import' not in code:
            code = 'from typing import Optional, List, Dict, Any\n' + code
        elif 'Optional[' in code and 'Optional' not in code.split('from typing import')[1].split('\n')[0]:
            code = code.replace('from typing import', 'from typing import Optional, ')
        
        # Add Config class if missing
        for model_match in model_pattern.finditer(code):
            model_name = model_match.group(1)
            # Check if Config class is missing
            if f'class Config:' not in code:
                # Find the end of the model definition
                model_end_pattern = re.compile(r'class\s+{}.*?(\n\S|\Z)'.format(model_name), re.DOTALL)
                model_end_match = model_end_pattern.search(code)
                
                if model_end_match:
                    config_class = '\n    class Config:\n        """Model configuration."""\n        extra = "forbid"  # Prevent extra fields\n        validate_assignment = True\n'
                    
                    # Insert Config class before the end of the model
                    end_pos = model_end_match.end() - len(model_end_match.group(1))
                    code = code[:end_pos] + config_class + code[end_pos:]
        
        return code


class DatabaseGenerator(DataLayerGenerator):
    """Generator for database-specific components.
    
    Extends the data layer generator with database-specific functionality
    such as SQLAlchemy models, migrations, and repositories.
    """
    
    async def generate_sqlalchemy_model(self, 
                                       context: GeneratorContext, 
                                       pydantic_model: Optional[ClassSignature] = None) -> CodeComponent:
        """Generate an SQLAlchemy model from a Pydantic model.
        
        Args:
            context: Generation context
            pydantic_model: Optional Pydantic model signature to base this on
            
        Returns:
            Generated SQLAlchemy model component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating SQLAlchemy model: {context.module_path}.{context.name}")
        
        # Add SQLAlchemy-specific guidance
        sqlalchemy_guidance = """
        ## SQLAlchemy Model Guidelines
        
        When implementing SQLAlchemy models, follow these guidelines:
        
        1. Use declarative base for model definition
        2. Map columns to appropriate SQLAlchemy types
        3. Define relationships with appropriate cascade settings
        4. Add proper table constraints (unique, index, etc.)
        5. Include __tablename__ with snake_case naming
        6. Define complete column options (nullable, default, etc.)
        7. Add __repr__ method for debugging
        8. Define foreign keys properly
        9. Consider adding helper methods for common operations
        10. Add proper docstrings for the model and methods
        
        Example SQLAlchemy Model:
        
        ```python
        class User(Base):
            """User model for authentication and profile information.
            
            Maps to the 'users' table in the database.
            """
            
            __tablename__ = "users"
            
            id = Column(Integer, primary_key=True, index=True)
            username = Column(String(50), unique=True, index=True, nullable=False)
            email = Column(String(100), unique=True, index=True, nullable=False)
            hashed_password = Column(String(100), nullable=False)
            is_active = Column(Boolean, default=True)
            created_at = Column(DateTime, default=datetime.datetime.utcnow)
            
            # Relationships
            items = relationship("Item", back_populates="owner", cascade="all, delete-orphan")
            
            def __repr__(self):
                return f"<User {self.username}>"
        ```
        """
        
        if context.additional_context:
            context.additional_context += sqlalchemy_guidance
        else:
            context.additional_context = sqlalchemy_guidance
        
        # If we have a Pydantic model, add it to the context
        if pydantic_model:
            pydantic_info = f"""
            ## Pydantic Model to Reference
            
            This SQLAlchemy model should correspond to this Pydantic model:
            
            ```python
            class {pydantic_model.name}(BaseModel):
            """
            
            # Add fields from Pydantic model
            for attr in pydantic_model.attributes:
                field_str = f"    {attr.name}: {attr.type_hint or 'Any'}"
                if attr.default_value:
                    field_str += f" = {attr.default_value}"
                pydantic_info += field_str + "\n"
                
            pydantic_info += "```"
            
            if context.additional_context:
                context.additional_context += pydantic_info
            else:
                context.additional_context = pydantic_info
        
        # Generate code
        try:
            response = await self._generate_with_llm(context)
            
            # Create component
            component = self._create_component_from_response(context, response)
            
            # Validate component
            self._validate_component(component)
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating SQLAlchemy model {context.name}: {str(e)}")
            raise
    
    async def generate_repository(self, 
                                 context: GeneratorContext, 
                                 model_signature: ClassSignature) -> CodeComponent:
        """Generate a repository for a model.
        
        Args:
            context: Generation context
            model_signature: Model signature to create repository for
            
        Returns:
            Generated repository component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        logger.info(f"Generating repository: {context.module_path}.{context.name}")
        
        # Add repository-specific guidance
        repository_guidance = """
        ## Repository Pattern Guidelines
        
        When implementing repositories, follow these guidelines:
        
        1. Create a base repository interface/abstract class
        2. Implement CRUD operations (create, read, update, delete)
        3. Use dependency injection for database session
        4. Add proper error handling for database operations
        5. Include transaction management when appropriate
        6. Add type hinting for all methods
        7. Return domain models rather than database models
        8. Add pagination support for list operations
        9. Include filtering and sorting capabilities
        10. Add proper docstrings for all methods
        
        Example Repository:
        
        ```python
        class Repository(Generic[T]):
            """Base repository interface for database operations."""
            
            @abstractmethod
            async def get(self, id: Any) -> Optional[T]:
                """Get an item by ID."""
                pass
                
            @abstractmethod
            async def list(self, skip: int = 0, limit: int = 100) -> List[T]:
                """Get a list of items with pagination."""
                pass
                
            @abstractmethod
            async def create(self, item: T) -> T:
                """Create a new item."""
                pass
                
            @abstractmethod
            async def update(self, id: Any, item: T) -> Optional[T]:
                """Update an existing item."""
                pass
                
            @abstractmethod
            async def delete(self, id: Any) -> bool:
                """Delete an item by ID."""
                pass
        ```
        """
        
        # Add model info to context
        model_info = f"""
        ## Model Information
        
        This repository should work with the following model:
        
        ```python
        class {model_signature.name}:
        """
        
        # Add fields/methods from model signature
        for attr in model_signature.attributes:
            attr_str = f"    {attr.name}: {attr.type_hint or 'Any'}"
            if attr.default_value:
                attr_str += f" = {attr.default_value}"
            model_info += attr_str + "\n"
            
        for method in model_signature.methods:
            method_str = f"    def {method.name}("
            params = [str(param) for param in method.parameters]
            method_str += ", ".join(params) + ")"
            if method.return_type:
                method_str += f" -> {method.return_type.type_hint}"
            model_info += method_str + "\n"
                
        model_info += "```"
        
        if context.additional_context:
            context.additional_context += repository_guidance + model_info
        else:
            context.additional_context = repository_guidance + model_info
        
        # Generate code
        try:
            response = await self._generate_with_llm(context)
            
            # Create component
            component = self._create_component_from_response(context, response)
            
            # Validate component
            self._validate_component(component)
            
            return component
            
        except Exception as e:
            logger.error(f"Error generating repository {context.name}: {str(e)}")
            raise