# integration_agent.py
"""
Integration agent for integrating components and managing dependencies.

This agent is responsible for integrating different components of the system,
managing dependencies, and ensuring compatibility.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import asyncio
import os
import json
import re
import difflib
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime
import subprocess
import logging

from .agent_base import Agent, AgentStatus
from ..models.task_models import Task
from ..utils.fs_utils import resolve_path, save_snapshot


class IntegrationPoint(BaseModel):
    """Represents an integration point between components."""
    source_file: str
    target_file: str
    integration_type: str  # "api", "import", "event", etc.
    source_changes: Optional[str] = None
    target_changes: Optional[str] = None
    diff: Optional[str] = None  # Added diff field to store changes in diff format
    description: str = ""


class IntegrationResult(BaseModel):
    """
    Result of an integration operation.
    """
    success: bool
    integration_points: List[IntegrationPoint] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class IntegrationAgent(Agent):
    """
    Integration agent responsible for integrating different components.
    
    This agent handles the integration of frontend with backend, 
    connecting APIs, setting up event handlers, etc.
    """
    
    def __init__(
        self,
        name: str = "Integration",
        description: str = "Integrates different components",
        agent_type: str = "integration",
        workspace_dir: Optional[str] = None
    ):
        """Initialize the IntegrationAgent."""
        super().__init__(
            name=name,
            description=description,
            agent_type=agent_type
        )
        
        # Workspace for integration operations - using private variables
        self._workspace_dir = workspace_dir or "workspace/src"
        
        # Track integration results
        self._integration_results: List[IntegrationResult] = []
    
    # Define properties for safer access
    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir
    
    @workspace_dir.setter
    def workspace_dir(self, value: str):
        self._workspace_dir = value
    
    @property
    def integration_results(self) -> List[IntegrationResult]:
        return self._integration_results
    
    @integration_results.setter
    def integration_results(self, value: List[IntegrationResult]):
        self._integration_results = value
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process integration-related requests.
        
        Args:
            input_data: Input data for the integration request
            
        Returns:
            Dictionary with the integration results
        """
        # Check if LLM integration is available
        if not self.llm:
            raise ValueError("LLM integration not available")
        
        # Extract the action
        action = input_data.get("action", "")
        
        if action == "integrate_task":
            task_data = input_data.get("task")
            implementation_result = input_data.get("implementation_result", {})
            return await self._integrate_task(task_data, implementation_result)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _integrate_task(self, task_data: Dict[str, Any], implementation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate a task with other components.
        
        Args:
            task_data: Task that was implemented
            implementation_result: Result of the implementation
            
        Returns:
            Dictionary with integration results
        """
        try:
            # Create a Task object if we received a dict
            if isinstance(task_data, dict):
                try:
                    from ..models.task_models import Task
                    task = Task(**task_data)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid task data: {str(e)}"
                    }
            else:
                task = task_data
            
            # Identify integration points
            integration_points = await self._identify_integration_points(task, implementation_result)
            
            # Execute integration
            result = await self._execute_integration(integration_points)
            
            # Add to integration results
            self.integration_results.append(result)
            
            # Create integration summary
            summary = f"Integrated task '{task.title}' with {len(result.integration_points)} integration points."
            if result.errors:
                summary += f" Encountered {len(result.errors)} errors."
            
            return {
                "success": result.success,
                "integration_points": [point.dict() for point in result.integration_points],
                "errors": result.errors,
                "summary": summary
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Integration failed: {str(e)}"
            }
    
    async def _identify_integration_points(self, task: Task, implementation_result: Dict[str, Any]) -> List[IntegrationPoint]:
        """
        Identify integration points for a task.
        
        Args:
            task: Task that was implemented
            implementation_result: Result of the implementation
            
        Returns:
            List of integration points
        """
        # Define the schema for the LLM response
        schema = {
            "name": "integration_points",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "integration_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_file": {
                                    "type": "string",
                                    "description": "Source file path"
                                },
                                "target_file": {
                                    "type": "string",
                                    "description": "Target file path"
                                },
                                "integration_type": {
                                    "type": "string",
                                    "description": "Type of integration (e.g., api, import, event)"
                                },
                                "source_changes": {
                                    "type": "string",
                                    "description": "Changes to make to the source file"
                                },
                                "target_changes": {
                                    "type": "string",
                                    "description": "Changes to make to the target file"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the integration"
                                }
                            },
                            "required": ["source_file", "target_file", "integration_type", "description"]
                        }
                    }
                },
                "required": ["integration_points"]
            }
        }
        
        # Create workspace path
        workspace_path = resolve_path(self.workspace_dir, create_parents=True)
        
        # Get file operations from implementation result
        file_operations = implementation_result.get("file_operations", [])
        
        # Get list of files in the workspace
        files = []
        for root, _, filenames in os.walk(workspace_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, workspace_path)
                files.append(rel_path)
        
        # Create a prompt for the LLM
        llm_prompt = f"""
        You are an expert software integrator. Identify integration points for the following task.
        
        Task: {task.title}
        Description: {task.description}
        
        Files modified in this task:
        {json.dumps([op.get("file_path") for op in file_operations], indent=2)}
        
        All files in the workspace:
        {json.dumps(files, indent=2)}
        
        Identify points where these files need to be integrated with other components.
        Focus on:
        - Frontend components that need to connect to backend APIs
        - Backend components that need to be exposed as APIs
        - Event handlers that need to be connected
        - Imports that need to be added
        - Configuration changes needed
        
        For each integration point, specify the source file, target file, type of integration,
        and provide the specific changes needed in both files.
        """
        
        try:
            # Call the LLM with the prompt
            response = self.llm.structured_query(
                prompt=llm_prompt,
                json_schema=schema,
                clear_context=True
            )
            
            # Extract integration points
            points_data = response.get("integration_points", [])
            
            # Create IntegrationPoint objects
            return [
                IntegrationPoint(
                    source_file=point_data.get("source_file", ""),
                    target_file=point_data.get("target_file", ""),
                    integration_type=point_data.get("integration_type", "unknown"),
                    source_changes=point_data.get("source_changes"),
                    target_changes=point_data.get("target_changes"),
                    description=point_data.get("description", ""),
                    diff=point_data.get("diff")
                )
                for point_data in points_data
            ]
            
        except Exception as e:
            print(f"Error identifying integration points: {str(e)}")
            return []
    
    async def _execute_integration(self, integration_points: List[IntegrationPoint]) -> IntegrationResult:
        """
        Execute integration based on identified points.
        
        Args:
            integration_points: List of identified integration points
            
        Returns:
            Integration result object
        """
        results = []
        errors = []
        successful_points = []
        
        for point in integration_points:
            try:
                source_path = os.path.abspath(point.source_file)
                target_path = os.path.abspath(point.target_file)
                
                # Verify files exist
                if not os.path.exists(source_path):
                    errors.append(f"Source file not found: {point.source_file}")
                    continue
                    
                if not os.path.exists(target_path):
                    errors.append(f"Target file not found: {point.target_file}")
                    continue
                    
                # Read file contents
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_content = f.read()
                    
                with open(target_path, 'r', encoding='utf-8') as f:
                    target_content = f.read()
                
                # Check for diff-based integration
                if point.diff:
                    # Apply changes using the pre-generated diff
                    try:
                        # Determine which file to apply the diff to
                        target_file_path = target_path
                        changes = {
                            "operation": "diff",
                            "patch": point.diff
                        }
                        
                        # Apply the changes
                        modified_content = self._apply_changes(target_file_path, changes)
                        
                        # Add to successful points
                        successful_points.append(point)
                        
                        results.append({
                            "source_file": point.source_file,
                            "target_file": point.target_file,
                            "integration_type": point.integration_type,
                            "description": point.description,
                            "method": "diff",
                            "success": True
                        })
                    except Exception as e:
                        errors.append(f"Error applying diff for {point.source_file} -> {point.target_file}: {str(e)}")
                        continue
                else:
                    # Apply source changes if specified
                    if point.source_changes:
                        source_content = self._apply_changes(source_path, point.source_changes)
                        
                        # Write updated content
                        with open(source_path, 'w', encoding='utf-8') as f:
                            f.write(source_content)
                            
                    # Apply target changes if specified
                    if point.target_changes:
                        target_content = self._apply_changes(target_path, point.target_changes)
                        
                        # Write updated content
                        with open(target_path, 'w', encoding='utf-8') as f:
                            f.write(target_content)
                    
                    # Add to successful points
                    successful_points.append(point)
                    
                    results.append({
                        "source_file": point.source_file,
                        "target_file": point.target_file,
                        "integration_type": point.integration_type,
                        "description": point.description,
                        "success": True
                    })
                
            except Exception as e:
                errors.append(f"Error integrating {point.source_file} -> {point.target_file}: {str(e)}")
        
        # Create and return result
        return IntegrationResult(
            success=len(errors) == 0,
            integration_points=successful_points,
            errors=errors,
            results=results
        )
    
    def _apply_changes(self, file_path: str, changes: Dict[str, Any]) -> str:
        """
        Apply specified changes to file content based on operation.
        
        Args:
            file_path: Path to the file to modify
            changes: Dictionary with change operations
            
        Returns:
            The modified content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read the original content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        operation = changes.get("operation", "").lower()
        
        if operation == "replace":
            # Replace specific text
            old_text = changes.get("old_text", "")
            new_text = changes.get("new_text", "")
            
            if old_text in content:
                content = content.replace(old_text, new_text)
            else:
                raise ValueError(f"Text to replace not found in {file_path}")
                
        elif operation == "insert":
            # Insert at a specific position
            position = changes.get("position", "")
            text = changes.get("text", "")
            
            if position == "start":
                content = text + content
            elif position == "end":
                content = content + text
            elif position == "after":
                marker = changes.get("marker", "")
                if marker in content:
                    content = content.replace(marker, marker + text, 1)
                else:
                    raise ValueError(f"Marker for insertion not found in {file_path}")
            elif position == "before":
                marker = changes.get("marker", "")
                if marker in content:
                    content = content.replace(marker, text + marker, 1)
                else:
                    raise ValueError(f"Marker for insertion not found in {file_path}")
        
        elif operation == "delete":
            # Delete specific text
            text = changes.get("text", "")
            
            if text in content:
                content = content.replace(text, "")
            else:
                raise ValueError(f"Text to delete not found in {file_path}")
                
        elif operation == "diff":
            # Apply changes using a unified diff
            patch_text = changes.get("patch", "")
            if patch_text:
                # Apply the patch using difflib
                original_lines = content.splitlines()
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
                                    raise ValueError(f"Patch context mismatch at line {src_line_idx + 1}. " 
                                                    f"Expected: '{expected_line}', found: '{line[1:]}'")
                            # Keep the context line and advance the source pointer
                            patch_result.append(line[1:])  # Remove the ' '
                            src_line_idx += 1
                    
                    # Add content after the modified section
                    patch_result.extend(patched_lines[src_line_idx:])
                    
                    # Update patched_lines for the next hunk
                    patched_lines = patch_result
                
                # Join the patched lines back into a string
                content = '\n'.join(patched_lines)
            else:
                raise ValueError("No patch provided for diff operation")
                
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return content

    def _write_file(self, file_path: str, content: str) -> bool:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve the absolute path
            abs_path = resolve_path(os.path.join(self.workspace_dir, file_path), create_parents=True)
            
            # Check if the file already exists
            if os.path.exists(abs_path):
                # If it exists, use difflib to apply changes
                return self._apply_diff_to_file(abs_path, content)
            
            # If it's a new file, write it directly
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Successfully wrote file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {str(e)}")
            return False

    def _apply_diff_to_file(self, abs_path: str, new_content: str) -> bool:
        """
        Apply changes to an existing file using difflib for better control.
        
        Args:
            abs_path: Absolute path to the file
            new_content: New content to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the original file
            with open(abs_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # If the content is identical, no need to update
            if original_content == new_content:
                self.logger.info(f"File content unchanged: {abs_path}")
                return True
            
            # Generate a unified diff
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            
            # Create a diff
            diff = difflib.unified_diff(
                original_lines, 
                new_lines,
                fromfile=f'Original: {os.path.basename(abs_path)}',
                tofile=f'Modified: {os.path.basename(abs_path)}',
                n=3  # Context lines
            )
            
            # Log the diff for debugging
            diff_text = ''.join(diff)
            self.logger.info(f"Applying changes to {abs_path}:\n{diff_text}")
            
            # Write the new content
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.logger.info(f"Successfully updated file: {abs_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error applying diff to {abs_path}: {str(e)}")
            return False 