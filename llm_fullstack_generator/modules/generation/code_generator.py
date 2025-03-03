# modules/generation/code_generator.py
from typing import Dict, Optional
import logging
from core.schemas import Task
from core.project_context import ProjectContext
from utils.llm_client import LLMClient, Message

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Generates code one component at a time"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_component(
        self,
        task: Task,
        project_context: ProjectContext,
        file_path: str,
        context: Dict
    ) -> str:
        """Generate code for a single component"""
        logger.info(f"Generating code for file: {file_path} (Task: {task.id})")
        
        # Determine what kind of file we're generating
        file_extension = file_path.split('.')[-1] if '.' in file_path else ''
        is_python = file_extension == 'py'
        is_js = file_extension in ['js', 'jsx', 'ts', 'tsx']
        
        # Prepare prompt context
        epic = None
        for e in project_context.epics:
            for t in e.tasks:
                if t.id == task.id:
                    epic = e
                    break
            if epic:
                break
        
        epic_description = epic.description if epic else "No epic found"
        
        # Extract related file info for context
        related_files_info = ""
        for related_file in context.get("related_files", []):
            related_files_info += f"\nFile: {related_file['file_path']}\n"
            
            # Add imports
            if related_file.get("imports"):
                related_files_info += "Imports:\n"
                for imp in related_file["imports"]:
                    related_files_info += f"- {imp}\n"
            
            # Add function signatures
            if related_file.get("functions"):
                related_files_info += "Functions:\n"
                for func in related_file["functions"]:
                    params = ", ".join([
                        f"{p['name']}: {p['type_hint']}" + (f" = {p['default_value']}" if p['default_value'] else "")
                        for p in func.get("parameters", [])
                    ])
                    ret_type = f" -> {func['return_type']}" if func.get('return_type') else ""
                    related_files_info += f"- def {func['name']}({params}){ret_type}\n"
                    if func.get('docstring'):
                        related_files_info += f"  \"{func['docstring']}\"\n"
            
            # Add class signatures
            if related_file.get("classes"):
                related_files_info += "Classes:\n"
                for cls in related_file["classes"]:
                    base_classes = f"({', '.join(cls['base_classes'])})" if cls.get('base_classes') else ""
                    related_files_info += f"- class {cls['name']}{base_classes}\n"
                    if cls.get('docstring'):
                        related_files_info += f"  \"{cls['docstring']}\"\n"
                    
                    # Add methods
                    if cls.get("methods"):
                        related_files_info += "  Methods:\n"
                        for method in cls["methods"]:
                            params = ", ".join([
                                f"{p['name']}: {p['type_hint']}" + (f" = {p['default_value']}" if p['default_value'] else "")
                                for p in method.get("parameters", [])
                            ])
                            ret_type = f" -> {method['return_type']}" if method.get('return_type') else ""
                            related_files_info += f"  - def {method['name']}({params}){ret_type}\n"
                            if method.get('docstring'):
                                related_files_info += f"    \"{method['docstring']}\"\n"
                    
                    # Add attributes
                    if cls.get("attributes"):
                        related_files_info += "  Attributes:\n"
                        for attr in cls["attributes"]:
                            type_hint = f": {attr['type_hint']}" if attr.get('type_hint') else ""
                            default = f" = {attr['default_value']}" if attr.get('default_value') else ""
                            related_files_info += f"  - {attr['name']}{type_hint}{default}\n"
        
        messages = [
            Message(role="system", content=f"""You are an expert {project_context.config.language} developer tasked with implementing a specific component of a larger project.
                   You will be given a task description and context from the project.
                   Your job is to implement ONLY the specific file requested, ensuring it integrates well with the rest of the project.
                   
                   Project Name: {project_context.name}
                   Project Description: {project_context.description}
                   
                   Epic: {epic.title if epic else "Unknown"}
                   Epic Description: {epic_description}
                   
                   Task: {task.title}
                   Task Description: {task.description}
                   
                   Write clean, well-documented code with proper type hints and docstrings.
                   Include necessary imports.
                   DO NOT include placeholder comments or TODO items - implement everything fully.
                   DO NOT generate any files other than the one requested.
                   Make sure your implementation works with the APIs described in related files."""),
            Message(role="user", content=f"""I need you to implement the file: {file_path}
                   
                   Here's the context from related files in the project:
                   {related_files_info}
                   
                   Please generate complete, production-ready code for this file.
                   Ensure all functions, classes, and methods have proper documentation.
                   Use appropriate design patterns and best practices for {project_context.config.language}.
                   The code should integrate smoothly with the existing codebase as described above.
                   """)
        ]
        
        try:
            # Generate code
            code = self.llm_client.chat_completion(messages)
            
            # Post-process code
            code = self._clean_code(code)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating code for {file_path}: {str(e)}")
            raise
    
    def _clean_code(self, code: str) -> str:
        """Clean up the generated code"""
        # Remove markdown code blocks if present
        if code.startswith("```") and code.endswith("```"):
            code = code.split("```")[1]
            
            # Check if there's a language specifier on the first line
            first_line = code.split('\n')[0].strip()
            if first_line in ['python', 'javascript', 'typescript', 'js', 'ts', 'jsx', 'tsx']:
                code = '\n'.join(code.split('\n')[1:])
        
        # Remove extra whitespace
        code = code.strip()
        
        return code