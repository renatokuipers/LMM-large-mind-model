# modules/memory/codebase_context.py
import os
import re
from typing import Dict, List, Optional, Set
import logging
from core.schemas import FileStructure, FunctionSignature, ClassSignature, Parameter

logger = logging.getLogger(__name__)

class CodebaseContext:
    """Memory management for code signatures and imports"""
    
    def __init__(self):
        self.files: Dict[str, FileStructure] = {}
    
    def get_context_for_file(self, file_path: str) -> Dict:
        """Get relevant context for generating a file"""
        # Determine which files and functions would be relevant
        related_files = self._find_related_files(file_path)
        
        # Build context with signatures only (not implementations)
        context = {
            "file_path": file_path,
            "related_files": []
        }
        
        for related_file in related_files:
            if related_file in self.files:
                file_structure = self.files[related_file]
                context["related_files"].append({
                    "file_path": related_file,
                    "imports": file_structure.imports,
                    "functions": [
                        {
                            "name": func.name,
                            "parameters": [
                                {"name": param.name, "type_hint": param.type_hint, "default_value": param.default_value}
                                for param in func.parameters
                            ],
                            "return_type": func.return_type,
                            "docstring": func.docstring
                        }
                        for func in file_structure.functions
                    ],
                    "classes": [
                        {
                            "name": cls.name,
                            "base_classes": cls.base_classes,
                            "docstring": cls.docstring,
                            "methods": [
                                {
                                    "name": method.name,
                                    "parameters": [
                                        {"name": param.name, "type_hint": param.type_hint, "default_value": param.default_value}
                                        for param in method.parameters
                                    ],
                                    "return_type": method.return_type,
                                    "docstring": method.docstring
                                }
                                for method in cls.methods
                            ],
                            "attributes": [
                                {"name": attr.name, "type_hint": attr.type_hint, "default_value": attr.default_value}
                                for attr in cls.attributes
                            ]
                        }
                        for cls in file_structure.classes
                    ]
                })
        
        return context
    
    def _find_related_files(self, file_path: str) -> List[str]:
        """Find files that would be relevant to the current file"""
        # Simple implementation - can be enhanced with more sophisticated logic
        related = []
        directory = os.path.dirname(file_path)
        
        # Include files in the same directory
        for f in self.files:
            if os.path.dirname(f) == directory:
                related.append(f)
        
        # Include files that might be imported based on name patterns
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        for f in self.files:
            f_base = os.path.splitext(os.path.basename(f))[0]
            # If the names are similar or one is a subset of the other
            if base_name in f_base or f_base in base_name:
                related.append(f)
        
        return list(set(related))  # Remove duplicates
    
    def update_from_code(self, file_path: str, code: str) -> None:
        """Update the context from generated code"""
        logger.info(f"Updating codebase context for file: {file_path}")
        
        file_structure = self._parse_code(file_path, code)
        self.files[file_path] = file_structure
    
    def _parse_code(self, file_path: str, code: str) -> FileStructure:
        """Parse code to extract signatures"""
        # This is a simplified parser and should be enhanced for production
        
        # Extract imports
        imports = []
        import_pattern = r'^\s*(?:from\s+[\w.]+\s+)?import\s+[\w, \t.*]+$'
        for line in code.split('\n'):
            if re.match(import_pattern, line):
                imports.append(line.strip())
        
        # Extract classes with simple regex
        classes = []
        class_pattern = r'class\s+(\w+)(?:\s*\(\s*([\w\s,]+)\s*\))?:'
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            class_name = match.group(1)
            base_classes = []
            if match.group(2):
                base_classes = [c.strip() for c in match.group(2).split(',')]
            
            # Extract docstring - simplified
            class_pos = match.start()
            class_code = code[class_pos:]
            docstring = None
            docstring_match = re.search(r'"""\s*(.*?)\s*"""', class_code, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
            
            # Extract methods - very simplified
            methods = []
            method_pattern = r'def\s+(\w+)\s*\(self(?:,\s*(.*?))?\)\s*(?:->\s*([\w\[\]\'\"., ]+))?:'
            method_matches = re.finditer(method_pattern, class_code)
            
            for method_match in method_matches:
                method_name = method_match.group(1)
                parameters = []
                
                # Add self parameter
                parameters.append(Parameter(name="self", type_hint="", default_value=None))
                
                # Parse other parameters
                if method_match.group(2):
                    param_str = method_match.group(2).strip()
                    if param_str:
                        for param in param_str.split(','):
                            param = param.strip()
                            if param:
                                param_parts = param.split(':')
                                param_name = param_parts[0].strip()
                                
                                type_hint = ""
                                default_value = None
                                
                                if len(param_parts) > 1:
                                    # Handle parameter with type hint
                                    type_part = param_parts[1].strip()
                                    
                                    # Check for default value
                                    if '=' in type_part:
                                        type_hint_parts = type_part.split('=')
                                        type_hint = type_hint_parts[0].strip()
                                        default_value = type_hint_parts[1].strip()
                                    else:
                                        type_hint = type_part
                                else:
                                    # Check if parameter has default value without type hint
                                    if '=' in param_name:
                                        param_parts = param_name.split('=')
                                        param_name = param_parts[0].strip()
                                        default_value = param_parts[1].strip()
                                
                                parameters.append(Parameter(
                                    name=param_name,
                                    type_hint=type_hint,
                                    default_value=default_value
                                ))
                
                return_type = method_match.group(3) if method_match.group(3) else None
                
                # Extract method docstring - simplified
                method_pos = method_match.start()
                method_code = class_code[method_pos:]
                method_docstring = None
                method_docstring_match = re.search(r'"""\s*(.*?)\s*"""', method_code, re.DOTALL)
                if method_docstring_match:
                    method_docstring = method_docstring_match.group(1).strip()
                
                methods.append(FunctionSignature(
                    name=method_name,
                    parameters=parameters,
                    return_type=return_type,
                    docstring=method_docstring,
                    file_path=file_path
                ))
            
            # Extract class attributes - simplified
            attributes = []
            attribute_pattern = r'self\.(\w+)\s*='
            attribute_matches = re.finditer(attribute_pattern, class_code)
            for attr_match in attribute_matches:
                attr_name = attr_match.group(1)
                if attr_name not in [p.name for p in attributes]:
                    attributes.append(Parameter(
                        name=attr_name,
                        type_hint="",  # Type hints for attributes would require more sophisticated parsing
                        default_value=None
                    ))
            
            classes.append(ClassSignature(
                name=class_name,
                methods=methods,
                attributes=attributes,
                base_classes=base_classes,
                docstring=docstring,
                file_path=file_path
            ))
        
        # Extract standalone functions
        functions = []
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([\w\[\]\'\"., ]+))?:'
        func_matches = re.finditer(func_pattern, code)
        
        for match in func_matches:
            # Skip methods (already handled in classes)
            if re.search(r'def\s+' + match.group(1) + r'\s*\(self', code):
                continue
                
            func_name = match.group(1)
            parameters = []
            
            # Parse parameters
            if match.group(2):
                param_str = match.group(2).strip()
                if param_str:
                    for param in param_str.split(','):
                        param = param.strip()
                        if param:
                            param_parts = param.split(':')
                            param_name = param_parts[0].strip()
                            
                            type_hint = ""
                            default_value = None
                            
                            if len(param_parts) > 1:
                                # Handle parameter with type hint
                                type_part = param_parts[1].strip()
                                
                                # Check for default value
                                if '=' in type_part:
                                    type_hint_parts = type_part.split('=')
                                    type_hint = type_hint_parts[0].strip()
                                    default_value = type_hint_parts[1].strip()
                                else:
                                    type_hint = type_part
                            else:
                                # Check if parameter has default value without type hint
                                if '=' in param_name:
                                    param_parts = param_name.split('=')
                                    param_name = param_parts[0].strip()
                                    default_value = param_parts[1].strip()
                            
                            parameters.append(Parameter(
                                name=param_name,
                                type_hint=type_hint,
                                default_value=default_value
                            ))
            
            return_type = match.group(3) if match.group(3) else None
            
            # Extract function docstring - simplified
            func_pos = match.start()
            func_code = code[func_pos:]
            func_docstring = None
            func_docstring_match = re.search(r'"""\s*(.*?)\s*"""', func_code, re.DOTALL)
            if func_docstring_match:
                func_docstring = func_docstring_match.group(1).strip()
            
            functions.append(FunctionSignature(
                name=func_name,
                parameters=parameters,
                return_type=return_type,
                docstring=func_docstring,
                file_path=file_path
            ))
        
        return FileStructure(
            file_path=file_path,
            functions=functions,
            classes=classes,
            imports=imports
        )