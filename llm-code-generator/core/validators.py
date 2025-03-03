from typing import Dict, List, Optional, Set, Tuple, Union
import ast
import logging
import tempfile
import subprocess
from pathlib import Path
import importlib.util
from pydantic import BaseModel, Field, validator

from schemas.code_entities import (
    ClassSignature, 
    FunctionSignature, 
    Parameter,
    ReturnType,
    CodeComponent
)
from core.code_memory import CodeMemory
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of code validation."""
    
    is_valid: bool = Field(..., description="Whether the code is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    component_name: Optional[str] = Field(None, description="Name of the component being validated")


class CodeValidator:
    """Validates generated code for correctness and consistency."""
    
    def __init__(self, code_memory: CodeMemory):
        """Initialize the validator.
        
        Args:
            code_memory: Code memory instance
        """
        self.code_memory = code_memory
    
    def _parse_code(self, code: str) -> Optional[ast.Module]:
        """Parse Python code into an AST.
        
        Args:
            code: Python code to parse
            
        Returns:
            AST if parsing succeeds, None otherwise
        """
        try:
            return ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            return None
    
    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Extract import statements from an AST.
        
        Args:
            tree: AST to extract from
            
        Returns:
            List of import statements
        """
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(name.name for name in node.names)
                imports.append(f"from {module} import {names}")
        
        return imports
    
    def _extract_function_signature(self, 
                                   func_node: ast.FunctionDef, 
                                   module_path: str) -> FunctionSignature:
        """Extract a function signature from an AST node.
        
        Args:
            func_node: AST function definition node
            module_path: Module path
            
        Returns:
            Function signature
        """
        # Extract parameters
        parameters = []
        for arg in func_node.args.args:
            # Get type hint if available
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)
            
            # Check if parameter has a default value
            is_required = True
            default_value = None
            
            # Add parameter to list
            param = Parameter(
                name=arg.arg,
                type_hint=type_hint,
                default_value=default_value,
                is_required=is_required
            )
            parameters.append(param)
        
        # Extract return type
        return_type = None
        if func_node.returns:
            return_type = ReturnType(
                type_hint=ast.unparse(func_node.returns)
            )
        
        # Extract docstring
        docstring = ast.get_docstring(func_node) or ""
        
        # Create function signature
        return FunctionSignature(
            name=func_node.name,
            parameters=parameters,
            return_type=return_type,
            module_path=module_path,
            is_async=isinstance(func_node, ast.AsyncFunctionDef),
            docstring=docstring
        )
    
    def _extract_class_signature(self, 
                                class_node: ast.ClassDef, 
                                module_path: str) -> ClassSignature:
        """Extract a class signature from an AST node.
        
        Args:
            class_node: AST class definition node
            module_path: Module path
            
        Returns:
            Class signature
        """
        # Extract base classes
        base_classes = []
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base))
        
        # Extract methods
        methods = []
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_sig = self._extract_function_signature(node, module_path)
                methods.append(method_sig)
        
        # Extract docstring
        docstring = ast.get_docstring(class_node) or ""
        
        # Create class signature
        return ClassSignature(
            name=class_node.name,
            base_classes=base_classes,
            methods=methods,
            module_path=module_path,
            docstring=docstring
        )
    
    def extract_component_signatures(self, 
                                    code: str, 
                                    module_path: str) -> List[Union[ClassSignature, FunctionSignature]]:
        """Extract component signatures from code.
        
        Args:
            code: Python code
            module_path: Module path
            
        Returns:
            List of extracted signatures
        """
        tree = self._parse_code(code)
        if not tree:
            return []
            
        signatures = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_sig = self._extract_function_signature(node, module_path)
                signatures.append(func_sig)
            elif isinstance(node, ast.ClassDef):
                class_sig = self._extract_class_signature(node, module_path)
                signatures.append(class_sig)
        
        return signatures
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """Validate code syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation result
        """
        try:
            ast.parse(code)
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[]
            )
        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Syntax error: {str(e)}"],
                warnings=[]
            )
    
    def validate_static(self, 
                       code: str, 
                       module_path: str,
                       component_name: str) -> ValidationResult:
        """Perform static validation of code.
        
        Args:
            code: Python code to validate
            module_path: Module path
            component_name: Component name
            
        Returns:
            Validation result
        """
        # Check syntax first
        syntax_result = self.validate_syntax(code)
        if not syntax_result.is_valid:
            return syntax_result
        
        errors = []
        warnings = []
        
        # Parse code to AST
        tree = self._parse_code(code)
        if not tree:
            return ValidationResult(
                is_valid=False,
                errors=["Failed to parse code"],
                warnings=[],
                component_name=component_name
            )
        
        # Validate imports
        imports = self._extract_imports(tree)
        
        # Extract signatures
        signatures = self.extract_component_signatures(code, module_path)
        
        # Validate signatures against code memory
        for sig in signatures:
            if isinstance(sig, ClassSignature):
                # Check if this class matches existing signature (if already in memory)
                existing_class = self.code_memory.get_class(sig.name, module_path)
                if existing_class:
                    # TODO: Validate compatibility with existing class
                    pass
            elif isinstance(sig, FunctionSignature):
                # Check if this function matches existing signature (if already in memory)
                existing_func = self.code_memory.get_function(sig.name, module_path)
                if existing_func:
                    if not existing_func.is_compatible_with(sig):
                        errors.append(
                            f"Function {sig.name} signature is incompatible with existing signature"
                        )
        
        # Validate naming conventions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name[0].isupper():
                warnings.append(f"Class {node.name} should follow PascalCase naming convention")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.islower():
                warnings.append(f"Function {node.name} should follow snake_case naming convention")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            component_name=component_name
        )
    
    def validate_runtime(self, 
                        code: str, 
                        module_path: str,
                        component_name: str) -> ValidationResult:
        """Perform runtime validation by executing code in a temporary environment.
        
        Args:
            code: Python code to validate
            module_path: Module path
            component_name: Component name
            
        Returns:
            Validation result
        """
        # For safety, runtime validation is optional and can be complex to implement
        # This is a simplified version that only checks if the code can be imported
        
        errors = []
        warnings = []
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(code.encode('utf-8'))
        
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location("temp_module", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            errors.append(f"Runtime error: {str(e)}")
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            component_name=component_name
        )
    
    def validate_component(self, component: CodeComponent) -> ValidationResult:
        """Validate a code component.
        
        Args:
            component: Component to validate
            
        Returns:
            Validation result
        """
        if not component.implementation:
            return ValidationResult(
                is_valid=False,
                errors=["No implementation provided"],
                warnings=[],
                component_name=f"{component.module_path}.{component.name}"
            )
        
        # Perform static validation
        static_result = self.validate_static(
            component.implementation,
            component.module_path,
            component.name
        )
        
        # Early return if static validation failed
        if not static_result.is_valid:
            return static_result
        
        # Optionally perform runtime validation
        # Uncomment to enable runtime validation
        # runtime_result = self.validate_runtime(
        #     component.implementation,
        #     component.module_path,
        #     component.name
        # )
        # 
        # if not runtime_result.is_valid:
        #     return runtime_result
        
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=static_result.warnings,
            component_name=f"{component.module_path}.{component.name}"
        )