from typing import Dict, List, Optional, Union, Any
import abc
import logging
from pathlib import Path

from pydantic import BaseModel, Field, validator

from schemas.code_entities import CodeComponent
from schemas.llm_io import CodeGenerationRequest, CodeGenerationResponse
from core.code_memory import CodeMemory
from core.llm_manager import LLMManager
from core.validators import CodeValidator, ValidationResult
from core.exceptions import ValidationError, LLMError

logger = logging.getLogger(__name__)


class GeneratorContext(BaseModel):
    """Context information for code generation.
    
    This model encapsulates all contextual information needed for
    generating a specific component, providing a clean interface
    between the project manager and generators.
    """
    
    component_type: str = Field(..., description="Type of component to generate")
    name: str = Field(..., description="Name of the component")
    module_path: str = Field(..., description="Module path where this component will be defined")
    description: str = Field(..., description="Description of what the component should do")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements for this component")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies for this component")
    additional_context: Optional[str] = Field(None, description="Additional context information")
    project_description: str = Field(..., description="Overall project description")
    
    def to_prompt_text(self) -> str:
        """Convert context to prompt text format.
        
        Returns:
            Context as formatted prompt text
        """
        requirements_text = "\n".join(f"- {req}" for req in self.requirements)
        dependencies_text = "\n".join(f"- {dep}" for dep in self.dependencies) if self.dependencies else "None"
        
        return f"""
        # Component Generation Context
        
        ## Component Details
        Type: {self.component_type}
        Name: {self.name}
        Module: {self.module_path}
        
        ## Component Description
        {self.description}
        
        ## Requirements
        {requirements_text}
        
        ## Dependencies
        {dependencies_text}
        
        ## Project Context
        {self.project_description}
        
        {f"## Additional Context\n{self.additional_context}" if self.additional_context else ""}
        """


class BaseGenerator(abc.ABC):
    """Abstract base class for code generators.
    
    This class defines the common interface and functionality for all
    specialized generators, providing a consistent approach to code generation.
    """
    
    def __init__(self, 
                 llm_manager: LLMManager,
                 code_memory: CodeMemory,
                 validator: Optional[CodeValidator] = None):
        """Initialize the generator.
        
        Args:
            llm_manager: LLM manager instance
            code_memory: Code memory instance
            validator: Optional validator instance
        """
        self.llm_manager = llm_manager
        self.code_memory = code_memory
        self.validator = validator or CodeValidator(code_memory=code_memory)
    
    @abc.abstractmethod
    async def generate(self, context: GeneratorContext) -> CodeComponent:
        """Generate code for a component.
        
        Args:
            context: Generation context
            
        Returns:
            Generated code component
            
        Raises:
            LLMError: If there is an error generating the component
            ValidationError: If the generated component fails validation
        """
        pass
    
    def _enrich_context_with_signatures(self, context: GeneratorContext) -> GeneratorContext:
        """Enrich the context with relevant code signatures.
        
        This method adds signatures of dependencies to the context
        to help the LLM understand the interfaces it needs to work with.
        
        Args:
            context: Original context
            
        Returns:
            Enriched context
        """
        if not context.dependencies:
            return context
        
        # Get signatures of dependencies
        signatures = []
        for dep in context.dependencies:
            # Parse dependency string (module.component)
            parts = dep.split(".")
            
            if len(parts) < 2:
                continue
                
            module_path = ".".join(parts[:-1])
            component_name = parts[-1]
            
            # Try to get class signature first
            cls_sig = self.code_memory.get_class(component_name, module_path)
            if cls_sig:
                signatures.append(f"# Class: {dep}\n{cls_sig.get_signature_str()}")
                
                # Add method signatures
                for method in cls_sig.methods:
                    signatures.append(f"  {method.get_signature_str()}")
                    
                continue
            
            # Try to get function signature
            func_sig = self.code_memory.get_function(component_name, module_path)
            if func_sig:
                signatures.append(f"# Function: {dep}\n{func_sig.get_signature_str()}")
        
        if signatures:
            # Add signatures to additional context
            signature_text = "## Dependency Signatures\n" + "\n\n".join(signatures)
            
            if context.additional_context:
                context.additional_context += f"\n\n{signature_text}"
            else:
                context.additional_context = signature_text
        
        return context
    
    async def _generate_with_llm(self, context: GeneratorContext) -> CodeGenerationResponse:
        """Generate code using the LLM.
        
        Args:
            context: Generation context
            
        Returns:
            Code generation response
            
        Raises:
            LLMError: If there is an error generating the component
        """
        # Enrich context with dependency signatures
        enriched_context = self._enrich_context_with_signatures(context)
        
        # Create request
        request = CodeGenerationRequest(
            component_type=enriched_context.component_type,
            name=enriched_context.name,
            module_path=enriched_context.module_path,
            description=enriched_context.description,
            requirements="\n".join(enriched_context.requirements),
            additional_context=enriched_context.additional_context
        )
        
        # Generate code
        try:
            return await self.llm_manager.generate_code(request)
        except Exception as e:
            logger.error(f"Error generating component {context.name}: {str(e)}")
            raise LLMError(f"Failed to generate component {context.name}: {str(e)}")
    
    def _validate_component(self, component: CodeComponent) -> ValidationResult:
        """Validate a generated component.
        
        Args:
            component: Component to validate
            
        Returns:
            Validation result
            
        Raises:
            ValidationError: If validation fails
        """
        result = self.validator.validate_component(component)
        
        if not result.is_valid:
            errors = "\n".join(result.errors)
            raise ValidationError(
                f"Generated component {component.name} failed validation",
                errors=result.errors
            )
        
        return result
    
    def _create_component_from_response(self, 
                                       context: GeneratorContext,
                                       response: CodeGenerationResponse) -> CodeComponent:
        """Create a code component from a generation response.
        
        Args:
            context: Generation context
            response: Code generation response
            
        Returns:
            Code component
        """
        # Import dependencies from response
        dependencies = response.dependencies or []
        
        # Create component
        component = CodeComponent(
            component_type=context.component_type,
            name=context.name,
            module_path=context.module_path,
            implementation=response.code,
            dependencies=dependencies,
            # For now, we'll create a minimal signature - the validator will extract the full one
            signature=(
                ClassSignature(name=context.name, module_path=context.module_path)
                if context.component_type == "class"
                else FunctionSignature(name=context.name, module_path=context.module_path)
            )
        )
        
        return component


class TemplatedGenerator(BaseGenerator):
    """Base class for generators that use templates.
    
    This class extends the base generator with template-based generation,
    allowing for more structured and consistent code generation.
    """
    
    def __init__(self, 
                 llm_manager: LLMManager,
                 code_memory: CodeMemory,
                 validator: Optional[CodeValidator] = None,
                 templates_dir: Optional[Union[str, Path]] = None):
        """Initialize the generator.
        
        Args:
            llm_manager: LLM manager instance
            code_memory: Code memory instance
            validator: Optional validator instance
            templates_dir: Directory containing templates
        """
        super().__init__(llm_manager, code_memory, validator)
        self.templates_dir = Path(templates_dir) if templates_dir else None
    
    def _get_template(self, template_name: str) -> Optional[str]:
        """Get a template by name.
        
        Args:
            template_name: Template name
            
        Returns:
            Template content if found, None otherwise
        """
        if not self.templates_dir:
            return None
            
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            return None
            
        return template_path.read_text()
    
    def _apply_template(self, template: str, context: Dict[str, Any]) -> str:
        """Apply a template with context variables.
        
        Args:
            template: Template string
            context: Context variables
            
        Returns:
            Rendered template
        """
        # Simple string.format() based templating
        # In a real implementation, you might use Jinja2 or similar
        return template.format(**context)