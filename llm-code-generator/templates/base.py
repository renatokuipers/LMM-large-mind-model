from typing import Dict, List, Optional, Any, Union
import os
from pathlib import Path
import re
import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class TemplateVariable(BaseModel):
    """Describes a variable that can be used in a template."""
    
    name: str = Field(..., description="Variable name")
    description: str = Field(..., description="Description of the variable")
    default_value: Optional[str] = Field(None, description="Default value if not provided")
    required: bool = Field(True, description="Whether this variable is required")
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"Invalid variable name: {v}")
        return v


class Template(BaseModel):
    """Base class for templates.
    
    This class provides metadata about a template, including its
    name, description, variables, and file structure.
    """
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    variables: List[TemplateVariable] = Field(default_factory=list, description="Template variables")
    file_structure: Dict[str, str] = Field(default_factory=dict, description="Template file structure")
    
    class Config:
        arbitrary_types_allowed = True


class TemplateEngine(ABC):
    """Abstract base class for template engines.
    
    This class defines the interface for template engines that can
    render templates with variables.
    """
    
    @abstractmethod
    def render_template(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables.
        
        Args:
            template_content: Template content
            variables: Template variables
            
        Returns:
            Rendered template
        """
        pass


class SimpleTemplateEngine(TemplateEngine):
    """Simple template engine that uses Python's string.format.
    
    This template engine supports basic variable substitution using {variable}
    syntax, but does not support more advanced features like conditions or loops.
    """
    
    def render_template(self, template_content: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables.
        
        Args:
            template_content: Template content
            variables: Template variables
            
        Returns:
            Rendered template
        """
        try:
            return template_content.format(**variables)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing template variable: {e}")
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            raise ValueError(f"Error rendering template: {e}")


class TemplateManager:
    """Manages templates and provides access to them.
    
    This class is responsible for loading templates from a directory
    and rendering them with variables.
    """
    
    def __init__(self, templates_dir: Union[str, Path], engine: Optional[TemplateEngine] = None):
        """Initialize the template manager.
        
        Args:
            templates_dir: Directory containing templates
            engine: Optional template engine, defaults to SimpleTemplateEngine
        """
        self.templates_dir = Path(templates_dir)
        self.engine = engine or SimpleTemplateEngine()
        self.templates: Dict[str, Template] = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from the templates directory.
        
        This method searches for template.json files in the templates directory
        and loads them as Template objects.
        """
        logger.info(f"Loading templates from {self.templates_dir}")
        
        # Check if templates directory exists
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            return
        
        # Find all template.json files
        for template_json in self.templates_dir.glob("**/template.json"):
            try:
                # Load template metadata
                template_dir = template_json.parent
                template_name = template_dir.name
                
                # Create template object
                template = Template(
                    name=template_name,
                    description=f"Template: {template_name}",
                    file_structure={}
                )
                
                # Load file structure
                for file_path in template_dir.glob("**/*"):
                    if file_path.is_file() and file_path.name != "template.json":
                        relative_path = file_path.relative_to(template_dir)
                        template.file_structure[str(relative_path)] = str(file_path)
                
                # Add template to registry
                self.templates[template_name] = template
                logger.info(f"Loaded template: {template_name}")
                
            except Exception as e:
                logger.error(f"Error loading template from {template_json}: {e}")
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render a template with variables.
        
        Args:
            template_name: Template name
            variables: Template variables
            
        Returns:
            Dictionary mapping file paths to rendered content
            
        Raises:
            ValueError: If template not found or rendering fails
        """
        # Get template
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Validate variables
        for var in template.variables:
            if var.required and var.name not in variables:
                if var.default_value is not None:
                    variables[var.name] = var.default_value
                else:
                    raise ValueError(f"Missing required variable: {var.name}")
        
        # Render each file in the template
        rendered_files = {}
        for relative_path, file_path in template.file_structure.items():
            try:
                # Read template file
                with open(file_path, "r") as f:
                    template_content = f.read()
                
                # Render template
                rendered_content = self.engine.render_template(template_content, variables)
                
                # Add to rendered files
                rendered_files[relative_path] = rendered_content
                
            except Exception as e:
                logger.error(f"Error rendering template file {file_path}: {e}")
                raise ValueError(f"Error rendering template file {relative_path}: {e}")
        
        return rendered_files
    
    def apply_template(self, template_name: str, variables: Dict[str, Any], output_dir: Union[str, Path]) -> List[Path]:
        """Apply a template to a directory.
        
        Args:
            template_name: Template name
            variables: Template variables
            output_dir: Output directory
            
        Returns:
            List of created files
            
        Raises:
            ValueError: If template not found or application fails
        """
        # Convert output_dir to Path
        output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Render template
        rendered_files = self.render_template(template_name, variables)
        
        # Write rendered files to output directory
        created_files = []
        for relative_path, content in rendered_files.items():
            # Create output file path
            output_file = output_dir / relative_path
            
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_file, "w") as f:
                f.write(content)
            
            # Add to created files
            created_files.append(output_file)
        
        return created_files