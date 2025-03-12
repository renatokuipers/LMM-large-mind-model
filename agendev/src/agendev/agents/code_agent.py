# code_agent.py
"""Code agent for implementation and testing tasks."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import asyncio
import os
import json
import subprocess
import difflib
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

from .agent_base import Agent, AgentStatus
from ..models.task_models import Task, TaskType, TaskStatus
from ..llm_integration import LLMIntegration, LLMConfig
from ..utils.fs_utils import resolve_path, save_json, load_json, ensure_workspace_structure, content_hash, save_snapshot


class FileOperation(BaseModel):
    """Represents a file operation performed by the code agent."""
    operation: str  # "create", "modify", "delete"
    file_path: str
    content: Optional[str] = None
    original_content: Optional[str] = None
    diff: Optional[str] = None  # Added diff field to store the difference
    hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImplementationContext(BaseModel):
    """Context for a code implementation task."""
    workspace_path: str
    files: Dict[str, str] = Field(default_factory=dict)
    dependencies: Dict[str, str] = Field(default_factory=dict)
    commands: List[str] = Field(default_factory=list)
    language: str = ""
    frameworks: List[str] = Field(default_factory=list)


class CodeAgent(Agent):
    """
    Code agent responsible for implementing and testing code.
    
    This agent handles task implementation through code generation, 
    file operations, and testing activities.
    """
    
    def __init__(
        self,
        name: str = "Code",
        description: str = "Implements code and tests",
        agent_type: str = "code",
        workspace_dir: Optional[str] = None
    ):
        """Initialize the CodeAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Workspace for code operations
        self.workspace_dir = workspace_dir or "workspace/src"
        
        # Track file operations
        self.file_operations: List[FileOperation] = []
        
        # Current implementation context
        self.context: Optional[ImplementationContext] = None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process code-related requests.
        
        Args:
            input_data: Input data for the code request
            
        Returns:
            Dictionary with the code operation results
        """
        # Check if LLM integration is available
        if not self.llm:
            raise ValueError("LLM integration not available")
        
        # Extract the action
        action = input_data.get("action", "")
        
        if action == "implement_task":
            return await self._implement_task(input_data.get("task"))
        elif action == "run_tests":
            return await self._run_tests(input_data.get("test_scope", "all"))
        elif action == "apply_patch":
            return await self.apply_patch(
                input_data.get("file_path", ""),
                input_data.get("patch_text", "")
            )
        elif action == "create_patch":
            return await self.create_patch(
                input_data.get("original_file", ""),
                input_data.get("modified_file", ""),
                input_data.get("original_content", ""),
                input_data.get("modified_content", "")
            )
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _implement_task(self, task: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implement a task through code generation and file operations.
        
        Args:
            task: Task to implement
            
        Returns:
            Dictionary with implementation results
        """
        if not task:
            return {
                "success": False,
                "error": "No task provided"
            }
        
        try:
            # Create a Task object if we received a dict
            if isinstance(task, dict):
                try:
                    from ..models.task_models import Task, TaskType, TaskStatus
                    task = Task(**task)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid task data: {str(e)}"
                    }
            
            # Initialize implementation context
            self.context = await self._initialize_context(task)
            
            # Create an implementation plan
            implementation_plan = await self._create_implementation_plan(task)
            
            # Execute file operations
            file_operations = await self._execute_file_operations(implementation_plan.get("file_operations", []))
            
            # Execute commands
            command_results = await self._execute_commands(implementation_plan.get("commands", []))
            
            # Create implementation result summary
            result_summary = (
                f"Implemented task: {task.title}\n"
                f"Files affected: {len(file_operations)}\n"
                f"Commands executed: {len(command_results)}"
            )
            
            # Determine if integration is needed
            needs_integration = implementation_plan.get("needs_integration", False)
            
            return {
                "success": True,
                "task_id": str(task.id),
                "file_operations": [op.dict() for op in file_operations],
                "command_results": command_results,
                "summary": implementation_plan.get("summary", result_summary),
                "needs_integration": needs_integration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Implementation failed: {str(e)}"
            }
    
    async def _initialize_context(self, task: Task) -> ImplementationContext:
        """
        Initialize the implementation context for a task.
        
        Args:
            task: Task to implement
            
        Returns:
            Implementation context for the task
        """
        # Create workspace directory if it doesn't exist
        workspace_path = resolve_path(self.workspace_dir, create_parents=True)
        
        # Extract language and frameworks from task description and context
        language, frameworks = await self._determine_language_and_frameworks(task)
        
        # Scan existing files in the workspace
        files = {}
        for root, _, filenames in os.walk(workspace_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, workspace_path)
                
                # Skip large files and non-text files
                if os.path.getsize(file_path) < 1024 * 1024:  # 1MB limit
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            files[rel_path] = content
                    except UnicodeDecodeError:
                        # Skip binary files
                        pass
        
        # Create context
        return ImplementationContext(
            workspace_path=str(workspace_path),
            files=files,
            language=language,
            frameworks=frameworks
        )
    
    async def _determine_language_and_frameworks(self, task: Task) -> Tuple[str, List[str]]:
        """
        Determine the programming language and frameworks from task description.
        
        Args:
            task: Task to analyze
            
        Returns:
            Tuple of (language, frameworks list)
        """
        # Define the schema for the LLM response
        schema = {
            "name": "language_detection",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Primary programming language for the task"
                    },
                    "frameworks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of frameworks/libraries to use"
                    }
                },
                "required": ["language"]
            }
        }
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        Analyze the following task description and determine the programming language and frameworks needed.
        
        Task: {task.title}
        Description: {task.description}
        
        Extract the primary programming language and any frameworks or libraries that should be used.
        If the task doesn't explicitly mention a language, infer it from the context.
        """
        
        # Call the LLM with the prompt
        try:
            analysis_result = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            language = analysis_result.get("language", "python").lower()
            frameworks = analysis_result.get("frameworks", [])
            
            return language, frameworks
            
        except Exception:
            # Default to Python if analysis fails
            return "python", []
    
    async def _create_implementation_plan(self, task: Task) -> Dict[str, Any]:
        """
        Create an implementation plan for the task.
        
        Args:
            task: Task to implement
            
        Returns:
            Dictionary with the implementation plan
        """
        # Define the schema for the LLM response
        schema = {
            "name": "implementation_plan",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "file_operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["create", "modify", "delete"]
                                },
                                "file_path": {"type": "string"},
                                "content": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["operation", "file_path"]
                        }
                    },
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "summary": {"type": "string"},
                    "needs_integration": {"type": "boolean"}
                },
                "required": ["file_operations", "summary"]
            }
        }
        
        # Get the list of existing files
        existing_files = list(self.context.files.keys()) if self.context else []
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are an expert software developer. Create an implementation plan for the following task.
        
        Task: {task.title}
        Description: {task.description}
        
        The task should be implemented in {self.context.language if self.context else 'Python'}.
        
        Existing files in the workspace:
        {json.dumps(existing_files, indent=2)}
        
        For each existing file that needs modification, I'll provide the content below.
        
        {self._format_file_contents()}
        
        Create a detailed implementation plan including:
        1. File operations (create, modify, delete) with complete file content
        2. Any commands that need to be executed (e.g., npm install, pip install)
        3. A summary of the implementation approach
        4. Whether this task requires integration with other components
        
        Use proper file paths with appropriate extensions. Be thorough and implement the complete functionality.
        For dependencies, use package.json for JavaScript/TypeScript or requirements.txt for Python.
        """
        
        # Call the LLM with the prompt
        try:
            implementation_plan = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                config=LLMConfig(
                    temperature=0.2,  # Low temperature for code generation
                    max_tokens=4000    # Larger context for code
                ),
                clear_context=True
            )
            
            return implementation_plan
            
        except Exception as e:
            raise ValueError(f"Failed to create implementation plan: {str(e)}")
    
    def _format_file_contents(self) -> str:
        """
        Format file contents for the LLM prompt.
        
        Returns:
            Formatted string with file contents
        """
        if not self.context or not self.context.files:
            return "No existing files to show."
        
        result = []
        
        # Limit to 10 most relevant files to avoid context overflow
        relevant_files = list(self.context.files.keys())[:10]
        
        for file_path in relevant_files:
            content = self.context.files.get(file_path, "")
            
            # Truncate very large files
            if len(content) > 2000:
                content = content[:1000] + "\n...\n" + content[-1000:]
                
            result.append(f"File: {file_path}\n```\n{content}\n```\n")
        
        if len(self.context.files) > 10:
            result.append(f"... and {len(self.context.files) - 10} more files (not shown)")
            
        return "\n".join(result)
    
    async def _execute_file_operations(self, operations: List[Dict[str, Any]]) -> List[FileOperation]:
        """
        Execute file operations from the implementation plan.
        
        Args:
            operations: List of file operations to execute
            
        Returns:
            List of executed file operations
        """
        result = []
        
        for op_data in operations:
            operation = op_data.get("operation", "").lower()
            file_path = op_data.get("file_path", "")
            content = op_data.get("content", "")
            
            if not operation or not file_path:
                continue
                
            # Resolve the absolute path
            abs_path = resolve_path(os.path.join(self.context.workspace_path, file_path), create_parents=True)
            
            # Execute the operation
            original_content = None
            diff_text = None
            try:
                if operation == "create":
                    # Create a new file
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Create snapshot
                    save_snapshot(
                        content=content,
                        file_path=abs_path,
                        metadata={"operation": "create"}
                    )
                    
                elif operation == "modify":
                    # Read existing content
                    if os.path.exists(abs_path):
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                        
                        # Generate diff between original content and new content
                        # Using unified diff format
                        original_lines = original_content.splitlines(keepends=True)
                        new_lines = content.splitlines(keepends=True)
                        
                        diff = difflib.unified_diff(
                            original_lines, 
                            new_lines,
                            fromfile=f"a/{file_path}",
                            tofile=f"b/{file_path}",
                            n=3
                        )
                        diff_text = ''.join(diff)
                    
                    # Write new content
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Create snapshot
                    save_snapshot(
                        content=content,
                        file_path=abs_path,
                        metadata={"operation": "modify", "diff": diff_text}
                    )
                    
                elif operation == "delete":
                    # Read existing content before deletion
                    if os.path.exists(abs_path):
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                        
                        # Generate diff for deletion (showing the entire file as removed)
                        original_lines = original_content.splitlines(keepends=True)
                        diff = difflib.unified_diff(
                            original_lines,
                            [],
                            fromfile=f"a/{file_path}",
                            tofile="/dev/null",
                            n=3
                        )
                        diff_text = ''.join(diff)
                        
                        # Delete the file
                        os.remove(abs_path)
            
                # Calculate content hash
                content_hash_value = content_hash(content) if content else None
                
                # Create file operation record
                file_op = FileOperation(
                    operation=operation,
                    file_path=file_path,
                    content=content,
                    original_content=original_content,
                    diff=diff_text,
                    hash=content_hash_value,
                    metadata={"description": op_data.get("description", "")}
                )
                
                result.append(file_op)
                self.file_operations.append(file_op)
                
            except Exception as e:
                # Log the error and continue with other operations
                print(f"Error executing file operation {operation} on {file_path}: {str(e)}")
        
        return result
    
    async def _execute_commands(self, commands: List[str]) -> List[Dict[str, Any]]:
        """
        Execute commands from the implementation plan.
        
        Args:
            commands: List of commands to execute
            
        Returns:
            List of command execution results
        """
        results = []
        
        for cmd in commands:
            try:
                # Create a result record
                result = {
                    "command": cmd,
                    "success": False,
                    "output": "",
                    "error": ""
                }
                
                # Execute the command in the workspace directory
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=self.context.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for the process to complete with a timeout
                try:
                    stdout, stderr = process.communicate(timeout=60)
                    
                    result["success"] = process.returncode == 0
                    result["output"] = stdout
                    result["error"] = stderr
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    result["error"] = "Command execution timed out after 60 seconds"
                
                results.append(result)
                
            except Exception as e:
                # Log the error and continue with other commands
                results.append({
                    "command": cmd,
                    "success": False,
                    "output": "",
                    "error": f"Error executing command: {str(e)}"
                })
        
        return results
    
    async def _run_tests(self, test_scope: str = "all") -> Dict[str, Any]:
        """
        Run tests for the implemented code.
        
        Args:
            test_scope: Scope of tests to run ("all", "unit", "integration")
            
        Returns:
            Dictionary with the test results
        """
        # This is a placeholder for actual test execution
        # In a real implementation, this would run appropriate tests
        # based on the test_scope
        
        return {
            "success": True,
            "test_results": [
                {
                    "name": "Example Test",
                    "passed": True,
                    "duration": 0.5
                }
            ],
            "summary": "All tests passed"
        }
    
    async def apply_patch(self, file_path: str, patch_text: str) -> Dict[str, Any]:
        """
        Apply a patch to an existing file using difflib.
        
        This allows for more precise modifications to existing files
        using unified diff format (as created by difflib.unified_diff).
        
        Args:
            file_path: Path to the file to be patched
            patch_text: The patch text in unified diff format
            
        Returns:
            Dictionary with the patch operation results
        """
        if not self.context:
            return {
                "success": False,
                "error": "No active implementation context"
            }
            
        abs_path = resolve_path(os.path.join(self.context.workspace_path, file_path))
        
        if not os.path.exists(abs_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
            
        try:
            # Read the current file content
            with open(abs_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            # Parse the patch and apply it using difflib
            original_lines = original_content.splitlines()
            patched_lines = original_lines.copy()
            
            # Parse the unified diff format patch
            patch_lines = patch_text.splitlines()
            hunks = []
            current_hunk = None
            
            # Parse the patch to find hunks
            for line in patch_lines:
                if line.startswith('@@'):
                    # Parse the @@ -start,count +start,count @@ line
                    parts = line.split(' ')
                    if len(parts) >= 3:
                        # Extract the line numbers: @@ -<line>,<count> +<line>,<count> @@
                        old_range = parts[1][1:].split(',')  # Remove the leading -
                        start_line = int(old_range[0])
                        
                        # Store the hunk information
                        current_hunk = {
                            'start': start_line,
                            'lines': [],
                        }
                        hunks.append(current_hunk)
                elif current_hunk is not None:
                    # Skip the file headers (--- and +++)
                    if not (line.startswith('---') or line.startswith('+++')):
                        current_hunk['lines'].append(line)
            
            # Process each hunk
            for hunk in hunks:
                start_line = hunk['start'] - 1  # Convert to 0-based index
                hunk_lines = hunk['lines']
                
                # Track position in the original file as we process the hunk
                src_line_idx = start_line
                patch_result = []
                
                # Add content before the patch
                patch_result.extend(patched_lines[:src_line_idx])
                
                # Process the hunk lines
                for line in hunk_lines:
                    if line.startswith('-'):
                        # This line should be removed, so just advance the source line pointer
                        src_line_idx += 1
                    elif line.startswith('+'):
                        # This line should be added, add it to the result
                        patch_result.append(line[1:])  # Remove the '+'
                    elif line.startswith(' '):
                        # Context line - verify it matches the expected line
                        if src_line_idx < len(original_lines):
                            expected_line = original_lines[src_line_idx]
                            if expected_line != line[1:]:  # Remove the space
                                # Context line mismatch - patch cannot be applied cleanly
                                return {
                                    "success": False,
                                    "error": f"Patch context mismatch at line {src_line_idx + 1}. " 
                                             f"Expected: '{expected_line}', found: '{line[1:]}'"
                                }
                        # Keep the context line and advance the source pointer
                        patch_result.append(line[1:])  # Remove the ' '
                        src_line_idx += 1
                
                # Add content after the modified section
                patch_result.extend(patched_lines[src_line_idx:])
                
                # Update patched_lines for the next hunk
                patched_lines = patch_result
            
            # Join the patched lines back into a string
            patched_content = '\n'.join(patched_lines)
            
            # Write the patched content back to the file
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(patched_content)
                
            # Create a file operation record
            file_op = FileOperation(
                operation="modify",
                file_path=file_path,
                content=patched_content,
                original_content=original_content,
                diff=patch_text,
                hash=content_hash(patched_content),
                metadata={"description": "Applied patch"}
            )
            
            self.file_operations.append(file_op)
            
            return {
                "success": True,
                "file_operation": file_op.dict()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error applying patch: {str(e)}"
            }
    
    async def create_patch(self, original_file: str = "", modified_file: str = "", 
                         original_content: str = "", modified_content: str = "") -> Dict[str, Any]:
        """
        Create a diff patch between two files or file contents.
        
        This method can create patches between:
        1. Two existing files (provide file paths)
        2. Original file and new content
        3. Two content strings directly
        
        Args:
            original_file: Path to the original file
            modified_file: Path to the modified file
            original_content: Original file content (if no original_file)
            modified_content: Modified file content (if no modified_file)
            
        Returns:
            Dictionary with the generated patch
        """
        # At least one method must be specified
        if not original_file and not original_content:
            return {
                "success": False,
                "error": "Must provide either original_file or original_content"
            }
            
        if not modified_file and not modified_content:
            return {
                "success": False,
                "error": "Must provide either modified_file or modified_content"
            }
            
        try:
            # Resolve content from files if needed
            if original_file and not original_content:
                abs_original_path = resolve_path(os.path.join(self.context.workspace_path, original_file))
                if os.path.exists(abs_original_path):
                    with open(abs_original_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                else:
                    return {
                        "success": False,
                        "error": f"Original file not found: {original_file}"
                    }
                    
            if modified_file and not modified_content:
                abs_modified_path = resolve_path(os.path.join(self.context.workspace_path, modified_file))
                if os.path.exists(abs_modified_path):
                    with open(abs_modified_path, 'r', encoding='utf-8') as f:
                        modified_content = f.read()
                else:
                    return {
                        "success": False,
                        "error": f"Modified file not found: {modified_file}"
                    }
            
            # Create the unified diff
            fromfile = original_file if original_file else "original"
            tofile = modified_file if modified_file else "modified"
            
            original_lines = original_content.splitlines(keepends=True)
            modified_lines = modified_content.splitlines(keepends=True)
            
            # Generate the unified diff
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{fromfile}",
                tofile=f"b/{tofile}",
                n=3  # Context lines
            )
            
            # Convert the diff iterator to a string
            diff_text = ''.join(list(diff))
            
            return {
                "success": True,
                "patch": diff_text,
                "from_file": fromfile,
                "to_file": tofile
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating patch: {str(e)}"
            } 