from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import logging
from pydantic import BaseModel, Field, validator

from schemas.code_entities import (
    ClassSignature, 
    FunctionSignature, 
    ModuleInfo,
    CodeComponent
)

logger = logging.getLogger(__name__)


class CodeMemoryState(BaseModel):
    """Pydantic model representing the complete state of the code memory system.
    
    This serves as both runtime state and serializable snapshot of code signatures.
    """
    
    # Key is module path
    modules: Dict[str, ModuleInfo] = Field(default_factory=dict)
    
    # Key is fully qualified name (module.Class or module.function)
    class_index: Dict[str, ClassSignature] = Field(default_factory=dict)
    function_index: Dict[str, FunctionSignature] = Field(default_factory=dict)
    
    # Dependency graph (component name -> dependencies)
    dependencies: Dict[str, Set[str]] = Field(default_factory=dict)
    
    # Track implementation status
    implemented_components: Set[str] = Field(default_factory=set)

    # Project metadata
    project_name: str = Field("")
    project_root: Optional[str] = Field(None)
    
    @validator('dependencies', pre=True)
    def convert_sets(cls, v):
        """Convert sets to lists for serialization and back."""
        if isinstance(v, dict):
            return {k: set(v) if isinstance(v, list) else v for k, v in v.items()}
        return v
    
    class Config:
        arbitrary_types_allowed = True
    

class CodeMemory:
    """Manages code signatures and relationships for the LLM-based code generator.
    
    This class maintains a lightweight representation of the entire codebase
    being generated, focusing on signatures rather than full implementations
    to optimize context window usage.
    """
    
    def __init__(self, project_name: str, project_root: Optional[Path] = None):
        """Initialize the code memory system.
        
        Args:
            project_name: Name of the project being generated
            project_root: Root directory of the project (optional)
        """
        self.state = CodeMemoryState(
            project_name=project_name,
            project_root=str(project_root) if project_root else None
        )
    
    def clear(self):
        """Reset the code memory state."""
        project_name = self.state.project_name
        project_root = self.state.project_root
        self.state = CodeMemoryState(
            project_name=project_name,
            project_root=project_root
        )
    
    def get_module_info(self, module_path: str) -> Optional[ModuleInfo]:
        """Get information about a module by its path."""
        return self.state.modules.get(module_path)
    
    def add_module(self, module_info: ModuleInfo) -> None:
        """Add or update a module in the code memory.
        
        Args:
            module_info: Module information to add
        """
        self.state.modules[module_info.path] = module_info
        
        # Update indices for faster lookups
        for cls in module_info.classes:
            full_name = f"{module_info.path}.{cls.name}"
            self.state.class_index[full_name] = cls
            
            # Track methods in function index too
            for method in cls.methods:
                method_full_name = f"{full_name}.{method.name}"
                self.state.function_index[method_full_name] = method
        
        for func in module_info.functions:
            full_name = f"{module_info.path}.{func.name}"
            self.state.function_index[full_name] = func
    
    def add_class(self, module_path: str, class_signature: ClassSignature) -> None:
        """Add or update a class in the code memory.
        
        Args:
            module_path: Path to the module containing the class
            class_signature: Class signature to add
        """
        module = self.state.modules.get(module_path)
        if not module:
            module = ModuleInfo(path=module_path)
            self.state.modules[module_path] = module
        
        # Check if class already exists and update it
        for i, existing_class in enumerate(module.classes):
            if existing_class.name == class_signature.name:
                module.classes[i] = class_signature
                break
        else:
            # Class doesn't exist, add it
            module.classes.append(class_signature)
        
        # Update class index
        full_name = f"{module_path}.{class_signature.name}"
        self.state.class_index[full_name] = class_signature
        
        # Update function index with methods
        for method in class_signature.methods:
            method_full_name = f"{full_name}.{method.name}"
            self.state.function_index[method_full_name] = method
    
    def add_function(self, module_path: str, function_signature: FunctionSignature) -> None:
        """Add or update a function in the code memory.
        
        Args:
            module_path: Path to the module containing the function
            function_signature: Function signature to add
        """
        module = self.state.modules.get(module_path)
        if not module:
            module = ModuleInfo(path=module_path)
            self.state.modules[module_path] = module
        
        # Check if function already exists and update it
        for i, existing_func in enumerate(module.functions):
            if existing_func.name == function_signature.name:
                module.functions[i] = function_signature
                break
        else:
            # Function doesn't exist, add it
            module.functions.append(function_signature)
        
        # Update function index
        full_name = f"{module_path}.{function_signature.name}"
        self.state.function_index[full_name] = function_signature
    
    def add_dependency(self, 
                       component_name: str, 
                       dependency_name: str) -> None:
        """Register a dependency between two components.
        
        Args:
            component_name: The dependent component (e.g., "module.Class.method")
            dependency_name: The component being depended on
        """
        if component_name not in self.state.dependencies:
            self.state.dependencies[component_name] = set()
        
        self.state.dependencies[component_name].add(dependency_name)
    
    def add_component(self, component: CodeComponent) -> None:
        """Add a code component to memory and register its implementation.
        
        Args:
            component: The component to add
        """
        if component.component_type == "function":
            self.add_function(component.module_path, component.signature)
        elif component.component_type == "class":
            self.add_class(component.module_path, component.signature)
        
        # Register the component as implemented
        full_name = f"{component.module_path}.{component.name}"
        self.state.implemented_components.add(full_name)
        
        # Register dependencies
        for dep in component.dependencies:
            self.add_dependency(full_name, dep)
    
    def is_implemented(self, component_name: str) -> bool:
        """Check if a component has been implemented.
        
        Args:
            component_name: Fully qualified name of the component
        
        Returns:
            True if the component has been implemented, False otherwise
        """
        return component_name in self.state.implemented_components
    
    def get_class(self, class_name: str, module_path: Optional[str] = None) -> Optional[ClassSignature]:
        """Get a class signature by name, optionally constrained to a module.
        
        Args:
            class_name: Name of the class
            module_path: Optional module path to constrain the search
        
        Returns:
            ClassSignature if found, None otherwise
        """
        if module_path:
            full_name = f"{module_path}.{class_name}"
            return self.state.class_index.get(full_name)
        
        # Search all modules
        for full_name, cls in self.state.class_index.items():
            if cls.name == class_name:
                return cls
        
        return None
    
    def get_function(self, 
                     function_name: str, 
                     module_path: Optional[str] = None,
                     class_name: Optional[str] = None) -> Optional[FunctionSignature]:
        """Get a function signature by name, optionally constrained to module/class.
        
        Args:
            function_name: Name of the function
            module_path: Optional module path to constrain the search
            class_name: Optional class name if this is a method
            
        Returns:
            FunctionSignature if found, None otherwise
        """
        if module_path and class_name:
            full_name = f"{module_path}.{class_name}.{function_name}"
            return self.state.function_index.get(full_name)
        
        if module_path:
            full_name = f"{module_path}.{function_name}"
            return self.state.function_index.get(full_name)
        
        # Search all modules
        for full_name, func in self.state.function_index.items():
            if func.name == function_name:
                return func
        
        return None
    
    def get_dependencies(self, component_name: str) -> Set[str]:
        """Get all dependencies of a component.
        
        Args:
            component_name: Fully qualified name of the component
            
        Returns:
            Set of component names this component depends on
        """
        return self.state.dependencies.get(component_name, set())
    
    def get_dependents(self, component_name: str) -> Set[str]:
        """Get all components that depend on this component.
        
        Args:
            component_name: Fully qualified name of the component
            
        Returns:
            Set of component names that depend on this component
        """
        dependents = set()
        for dependent, dependencies in self.state.dependencies.items():
            if component_name in dependencies:
                dependents.add(dependent)
        return dependents
    
    def get_missing_dependencies(self, component_name: str) -> Set[str]:
        """Get dependencies of a component that haven't been implemented yet.
        
        Args:
            component_name: Fully qualified name of the component
            
        Returns:
            Set of component names that are dependencies but not implemented
        """
        dependencies = self.get_dependencies(component_name)
        return {dep for dep in dependencies if not self.is_implemented(dep)}
    
    def get_next_components_to_implement(self) -> List[str]:
        """Get a list of components that are ready to be implemented.
        
        A component is ready if all its dependencies have been implemented.
        
        Returns:
            List of component names that are ready to implement
        """
        result = []
        
        for component_name in self.state.dependencies:
            if not self.is_implemented(component_name):
                missing_deps = self.get_missing_dependencies(component_name)
                if not missing_deps:
                    result.append(component_name)
        
        return result
    
    def get_summary_for_context(self, 
                                max_length: int = 4000,
                                relevant_components: Optional[List[str]] = None) -> str:
        """Generate a summary of the code memory suitable for LLM context.
        
        This creates a compact representation of the code signatures to
        provide context to the LLM without consuming too much context window.
        
        Args:
            max_length: Maximum length of the summary
            relevant_components: Optional list of components to prioritize
            
        Returns:
            String summary of code signatures
        """
        # TODO: Implement smarter summarization based on relevance
        # For now, just include all signatures
        
        lines = [f"# Code Memory Summary for {self.state.project_name}"]
        
        # Add classes with their methods
        for module_path, module in self.state.modules.items():
            if not module.classes and not module.functions:
                continue
                
            lines.append(f"\n## Module: {module_path}")
            
            for cls in module.classes:
                lines.append(f"\n### Class: {cls.get_signature_str()}")
                if cls.docstring and cls.docstring.summary:
                    lines.append(f"# {cls.docstring.summary}")
                
                for method in cls.methods:
                    implemented = self.is_implemented(f"{module_path}.{cls.name}.{method.name}")
                    status = "[IMPLEMENTED]" if implemented else "[NOT IMPLEMENTED]"
                    lines.append(f"    {method.get_signature_str()} {status}")
            
            # Add standalone functions
            if module.functions:
                lines.append("\n### Functions:")
                for func in module.functions:
                    implemented = self.is_implemented(f"{module_path}.{func.name}")
                    status = "[IMPLEMENTED]" if implemented else "[NOT IMPLEMENTED]"
                    lines.append(f"{func.get_signature_str()} {status}")
        
        summary = "\n".join(lines)
        
        # Truncate if necessary
        if len(summary) > max_length:
            # TODO: Implement smarter truncation
            summary = summary[:max_length - 100] + "\n\n[Summary truncated due to length]"
        
        return summary
    
    def export_state(self) -> dict:
        """Export the current memory state as a dictionary."""
        return self.state.dict()
    
    def import_state(self, state_dict: dict) -> None:
        """Import a memory state from a dictionary."""
        self.state = CodeMemoryState.parse_obj(state_dict)