# modules/validation/code_validator.py
from typing import Dict, Optional
import logging
from core.schemas import Task
from core.project_context import ProjectContext
from utils.llm_client import LLMClient, Message

logger = logging.getLogger(__name__)

class CodeValidator:
    """Validates and improves generated code"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def validate_code(
        self,
        code: str,
        task: Task,
        file_path: str,
        project_context: ProjectContext
    ) -> str:
        """Validate and potentially improve the generated code"""
        logger.info(f"Validating code for file: {file_path}")
        
        # Determine file type
        file_extension = file_path.split('.')[-1] if '.' in file_path else ''
        is_python = file_extension == 'py'
        is_js = file_extension in ['js', 'jsx', 'ts', 'tsx']
        
        # Validation criteria based on file type
        validation_criteria = [
            "Code implements all functionality described in the task",
            "No placeholder or TODO comments",
            "Proper error handling",
            "Consistent coding style",
            "Proper docstrings and comments"
        ]
        
        if is_python:
            validation_criteria.extend([
                "Proper type hints",
                "PEP 8 compliance",
                "Imports are organized properly",
                "Uses appropriate Python idioms"
            ])
        elif is_js:
            validation_criteria.extend([
                "ES6+ syntax where appropriate",
                "Proper import/export statements",
                "Consistent variable declarations (const/let)"
            ])
        
        messages = [
            Message(role="system", content=f"""You are an expert code reviewer and fixer for {project_context.config.language} applications.
                   You will be given generated code, a task description, and validation criteria.
                   Your job is to review the code, identify any issues, and return an improved version.
                   
                   Do not completely rewrite the code unless absolutely necessary.
                   Focus on fixing issues that would prevent the code from working as expected.
                   Ensure the code is production-ready with proper error handling and documentation."""),
            Message(role="user", content=f"""Review this generated code for file: {file_path}
                   
                   Task Description: {task.description}
                   
                   Validation Criteria:
                   {chr(10).join(['- ' + criterion for criterion in validation_criteria])}
                   
                   Generated Code:
                   ```
                   {code}
                   ```
                   
                   Please identify any issues with the code based on the validation criteria.
                   Then provide an improved version of the code that fixes any issues you found.
                   If the code is already good, you can return it unchanged.
                   """)
        ]
        
        try:
            # Get validation feedback and improved code
            result = self.llm_client.chat_completion(messages)
            
            # Extract the improved code
            improved_code = self._extract_code_from_response(result)
            
            # If no code was extracted, use original
            if not improved_code:
                logger.warning(f"Could not extract improved code from validator response for {file_path}")
                return code
            
            return improved_code
            
        except Exception as e:
            logger.error(f"Error validating code for {file_path}: {str(e)}")
            # On error, return the original code
            return code
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code blocks from the validation response"""
        if "```" in response:
            # Get the last code block (in case there are multiple)
            code_blocks = response.split("```")
            for i in range(1, len(code_blocks), 2):
                if i < len(code_blocks):
                    code = code_blocks[i]
                    
                    # Check if there's a language specifier on the first line
                    first_line = code.split('\n')[0].strip()
                    if first_line in ['python', 'javascript', 'typescript', 'js', 'ts', 'jsx', 'tsx']:
                        code = '\n'.join(code.split('\n')[1:])
                    
                    # Last code block
                    if i == len(code_blocks) - 2:
                        return code.strip()
        else:
            # No code blocks found, check if the entire response is code
            lines = response.strip().split('\n')
            if lines and (
                lines[0].startswith('import ') or 
                lines[0].startswith('from ') or
                lines[0].startswith('def ') or
                lines[0].startswith('class ') or
                lines[0].startswith('const ') or
                lines[0].startswith('let ') or
                lines[0].startswith('function ')
            ):
                return response.strip()
        
        return None