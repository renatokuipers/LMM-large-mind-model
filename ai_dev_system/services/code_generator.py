import os
import re
from typing import List, Dict, Any, Optional, Tuple
import json

from models.project_model import Project, Epic, Task, CodeItem, TaskStatus
from llm_module import LLMClient, Message


class CodeGeneratorService:
    """Service for generating code incrementally using LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def generate_code_for_project(self, project: Project) -> Project:
        """Generate code for all tasks in the project"""
        for epic_idx, epic in enumerate(project.epics):
            for task_idx, task in enumerate(epic.tasks):
                updated_task = self.generate_code_for_task(project, epic, task)
                project.epics[epic_idx].tasks[task_idx] = updated_task
                
                # Update project timestamps
                project.epics[epic_idx].updated_at = updated_task.updated_at
            
            project.updated_at = project.epics[epic_idx].updated_at
                
        return project
    
    def generate_code_for_task(self, project: Project, epic: Epic, task: Task) -> Task:
        """Generate code for a specific task"""
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        
        # Generate code for each code item
        for item_idx, code_item in enumerate(task.code_items):
            if not code_item.implemented:
                # Generate file path if not specified
                if not code_item.file_path:
                    code_item.file_path = self._determine_file_path(project, epic, task, code_item)
                
                # Generate code for this item
                code, success = self._generate_code_for_item(project, epic, task, code_item)
                
                if success:
                    # Save the generated code
                    full_path = os.path.join(project.output_directory, code_item.file_path)
                    self._save_code_to_file(full_path, code, code_item)
                    
                    # Mark as implemented
                    code_item.implemented = True
                    task.code_items[item_idx] = code_item
                else:
                    # If code generation failed, mark task as failed
                    task.status = TaskStatus.FAILED
                    return task
        
        # If all code items were implemented, mark task as completed
        task.status = TaskStatus.COMPLETED
        return task
    
    def _determine_file_path(self, project: Project, epic: Epic, task: Task, code_item: CodeItem) -> str:
        """Determine the appropriate file path for a code item"""
        # Extract likely directory from the epic title
        directory = re.sub(r'[^a-zA-Z0-9]', '_', epic.title.lower())
        directory = directory.replace('__', '_').strip('_')
        
        # Extract filename from the code item name or task title
        if code_item.type.lower() == 'class':
            # For classes, use the class name
            filename = f"{code_item.name.lower()}.py"
        else:
            # For functions or methods, use something from the task title
            base_name = re.sub(r'[^a-zA-Z0-9]', '_', task.title.lower())
            base_name = base_name.replace('__', '_').strip('_')
            filename = f"{base_name}.py"
        
        # Combine into a path
        return os.path.join(directory, filename)
    
    def _generate_code_for_item(self, project: Project, epic: Epic, task: Task, 
                               code_item: CodeItem) -> Tuple[str, bool]:
        """Generate code for a specific code item"""
        # Build context from project info
        context = f"""
        Project: {project.name}
        Project Description: {project.description}
        
        Epic: {epic.title}
        Epic Description: {epic.description}
        
        Task: {task.title}
        Task Description: {task.description}
        
        You are implementing:
        Type: {code_item.type}
        Name: {code_item.name}
        Description: {code_item.description}
        Parameters: {', '.join(code_item.parameters) if code_item.parameters else 'None'}
        Result: {code_item.result}
        File Path: {code_item.file_path}
        """
        
        # Add information about related code items
        related_items = []
        for related_task in epic.tasks:
            for related_item in related_task.code_items:
                if (related_item.file_path == code_item.file_path and 
                    related_item.name != code_item.name and
                    related_item.implemented):
                    related_items.append(related_item)
        
        if related_items:
            context += "\nRelated code in the same file:"
            for item in related_items:
                context += f"\n- {item.type} {item.name}"
                if item.parameters:
                    context += f"({', '.join(item.parameters)})"
                if item.result != "None":
                    context += f" -> {item.result}"
                context += f": {item.description}"
        
        # Generate prompt based on code item type
        if code_item.type.lower() == 'class':
            code_prompt = f"""
            {context}
            
            Write a complete Python class definition for {code_item.name}.
            Use Pydantic BaseModel if appropriate.
            Include docstrings and type hints.
            Implement all necessary methods including __init__.
            Do not include example usage or main block.
            
            Only return the code, nothing else.
            """
        elif code_item.type.lower() == 'function':
            code_prompt = f"""
            {context}
            
            Write a complete Python function definition for {code_item.name}.
            Include docstrings and type hints.
            Implement the full function logic.
            Do not include example usage or main block.
            
            Only return the code, nothing else.
            """
        elif code_item.type.lower() == 'method':
            code_prompt = f"""
            {context}
            
            Write a complete Python method definition for {code_item.name}.
            This is a method of class {code_item.parent if code_item.parent else 'Unknown'}.
            Include docstrings and type hints.
            Implement the full method logic.
            Do not include the entire class, just this method.
            
            Only return the code, nothing else.
            """
        else:
            code_prompt = f"""
            {context}
            
            Write complete Python code for {code_item.name}.
            Include docstrings and type hints.
            Implement the full logic.
            Do not include example usage or main block.
            
            Only return the code, nothing else.
            """
        
        messages = [
            Message(role="system", content="You are an expert Python programmer. Write clean, efficient, and well-documented code."),
            Message(role="user", content=code_prompt)
        ]
        
        try:
            # Generate the code
            code = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
                max_tokens=2000
            )
            
            # Validate the generated code
            validation_result = self._validate_code(code)
            
            if validation_result[0]:
                return code, True
            else:
                print(f"Code validation failed: {validation_result[1]}")
                # Try to fix the code based on validation feedback
                fix_prompt = f"""
                {context}
                
                I tried to generate code for {code_item.name}, but there was an issue:
                {validation_result[1]}
                
                Here's the code that needs fixing:
                
                ```python
                {code}
                ```
                
                Please fix the issues and provide the corrected code.
                Only return the corrected code, nothing else.
                """
                
                fix_messages = [
                    Message(role="system", content="You are an expert Python programmer. Fix the issues in this code."),
                    Message(role="user", content=fix_prompt)
                ]
                
                fixed_code = self.llm_client.chat_completion(
                    fix_messages,
                    temperature=0.2,
                    max_tokens=2000
                )
                
                # Validate the fixed code
                fixed_validation = self._validate_code(fixed_code)
                return fixed_code, fixed_validation[0]
                
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return f"# Error generating code: {str(e)}", False
    
    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate the generated code for syntax errors"""
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            return True, "Code is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error during validation: {str(e)}"
    
    def _save_code_to_file(self, file_path: str, code: str, code_item: CodeItem) -> None:
        """Save generated code to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if file exists
        if os.path.exists(file_path):
            # Read existing file content
            with open(file_path, 'r') as f:
                existing_code = f.read()
                
            if code_item.type.lower() == 'method':
                # For methods, we need to insert into the existing class
                # This is a simplified approach; in practice you'd want a more robust parser
                class_regex = f"class {code_item.parent}"
                if class_regex in existing_code:
                    # Very simple method insertion - would need improvement for production
                    class_end = existing_code.find("class", existing_code.find(class_regex) + 1)
                    if class_end == -1:
                        class_end = len(existing_code)
                        
                    # Insert method before the end of the class
                    new_code = existing_code[:class_end] + "\n    " + code.replace("\n", "\n    ") + "\n" + existing_code[class_end:]
                    with open(file_path, 'w') as f:
                        f.write(new_code)
                else:
                    # Class not found, append as comment
                    with open(file_path, 'a') as f:
                        f.write(f"\n\n# TODO: Add to class {code_item.parent}:\n{code}\n")
            else:
                # For other types, simply append to the file
                with open(file_path, 'a') as f:
                    f.write(f"\n\n{code}\n")
        else:
            # Create new file with appropriate imports
            with open(file_path, 'w') as f:
                # Add standard imports
                f.write("# Generated file for project: " + os.path.basename(os.path.dirname(file_path)) + "\n")
                f.write("# Contains: " + code_item.type + " " + code_item.name + "\n\n")
                
                # Add common imports based on code content
                if "typing" in code or "List" in code or "Dict" in code or "Optional" in code:
                    f.write("from typing import List, Dict, Any, Optional, Tuple\n")
                    
                if "pydantic" in code.lower() or "BaseModel" in code:
                    f.write("from pydantic import BaseModel, Field, validator\n")
                    
                if "datetime" in code.lower():
                    f.write("from datetime import datetime\n")
                    
                if "uuid" in code.lower():
                    f.write("import uuid\n")
                    
                if "os" in code.lower() and "os." in code:
                    f.write("import os\n")
                    
                if "json" in code.lower() and "json." in code:
                    f.write("import json\n")
                    
                # Add an empty line after imports
                f.write("\n\n")
                f.write(code)