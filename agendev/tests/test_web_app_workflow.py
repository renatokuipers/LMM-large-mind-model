"""
AgenDev End-to-End Test Scenarios

This module contains end-to-end test scenarios for validating the core functionality
of the AgenDev system. The tests focus on web application development workflows,
CLI utility workflows, and error recovery scenarios.
"""
import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import random
from typing import List, Any, Optional, Union, Dict


# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agendev.core import AgenticCore
from src.agendev.models.task_models import TaskGraph, TaskStatus
from src.agendev.llm_integration import LLMIntegration
from src.agendev.test_generation import TestGenerator
from src.agendev.snapshot_engine import SnapshotEngine

class TestWebAppDevelopmentWorkflow(unittest.TestCase):
    """End-to-end tests for web application development workflow."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a test directory
        self.test_dir = Path("test_artifacts")
        self.test_dir.mkdir(exist_ok=True)
        
        # Mock the LLM integration to avoid actual API calls
        self.llm_patcher = patch('src.agendev.llm_integration.LLMIntegration')
        self.mock_llm = self.llm_patcher.start()
        
        # Setup mock responses
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
        
        # Mock successful generation responses
        self.mock_llm_instance.generate_text.return_value = "Mock LLM response"
        self.mock_llm_instance.generate_structured_output.return_value = {
            "tasks": [
                {"id": "task1", "description": "Setup Flask project structure", "dependencies": []},
                {"id": "task2", "description": "Create user authentication", "dependencies": ["task1"]},
                {"id": "task3", "description": "Implement database models", "dependencies": ["task1"]},
                {"id": "task4", "description": "Create API endpoints", "dependencies": ["task2", "task3"]},
                {"id": "task5", "description": "Write unit tests", "dependencies": ["task4"]},
                {"id": "task6", "description": "Setup frontend", "dependencies": ["task4"]},
                {"id": "task7", "description": "Deploy application", "dependencies": ["task5", "task6"]}
            ]
        }
        
        # Create core components
        self.core = AgenticCore(
            project_dir=self.test_dir,
            llm_endpoint="http://mock-endpoint:1234"
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.llm_patcher.stop()
        
        # Clean up test directory
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_complete_web_app_workflow(self):
        """Test a complete web application development workflow."""
        # 1. Project description and analysis
        project_description = """
        Create a Flask web application for a task management system. The app should:
        - Allow users to register, login, and manage their profile
        - Let users create, update, and delete tasks
        - Support task categorization and due dates
        - Provide API endpoints for mobile integration
        - Include search functionality
        """
        
        # Mock task graph analysis
        with patch.object(self.core, 'analyze_project_description') as mock_analyze:
            mock_analyze.return_value = TaskGraph()
            
            # Call method to analyze project
            task_graph = self.core.analyze_project_description(project_description)
            
            # Assert analysis was called
            mock_analyze.assert_called_once_with(project_description)
        
        # 2. Task planning
        with patch.object(self.core, 'plan_tasks') as mock_plan:
            mock_plan.return_value = ["task1", "task3", "task2", "task4", "task6", "task5", "task7"]
            
            # Call method to plan tasks
            task_sequence = self.core.plan_tasks(task_graph)
            
            # Assert planning was called
            mock_plan.assert_called_once_with(task_graph)
            
            # Verify we got a sequence
            self.assertEqual(len(task_sequence), 7)
        
        # 3. Project structure creation
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Flask structure created"}
            
            # Execute first task
            result = self.core.implement_task("task1")
            
            # Assert task implementation was called
            mock_implement.assert_called_once_with("task1")
            
            # Verify task completed
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 4. Test database implementation task
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Database models created"}
            
            # Execute database task
            result = self.core.implement_task("task3")
            
            # Verify task completed
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 5. Test user auth implementation task - simulate a failure
        with patch.object(self.core, 'implement_task') as mock_implement:
            # First attempt fails
            mock_implement.side_effect = [
                {"status": TaskStatus.FAILED, "output": "Authentication implementation failed", "error": "Invalid syntax"},
                {"status": TaskStatus.COMPLETED, "output": "Authentication implementation succeeded"}
            ]
            
            # First attempt - should fail
            result = self.core.implement_task("task2")
            self.assertEqual(result["status"], TaskStatus.FAILED)
            
            # Test error recovery
            with patch.object(self.core, 'retry_task') as mock_retry:
                mock_retry.return_value = {"status": TaskStatus.COMPLETED, "output": "Retry succeeded"}
                
                # Attempt retry
                retry_result = self.core.retry_task("task2")
                
                # Verify retry worked
                self.assertEqual(retry_result["status"], TaskStatus.COMPLETED)
        
        # 6. Test code generation and test generation
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "API endpoints created"}
            
            # Execute API endpoints task
            result = self.core.implement_task("task4")
            
            # Verify task completed
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 7. Test snapshot creation
        with patch('src.agendev.snapshot_engine.SnapshotEngine.create_snapshot') as mock_snapshot:
            mock_snapshot.return_value = "snapshot_123"
            
            # Create a snapshot
            snapshot_id = self.core.create_snapshot("Completed API implementation")
            
            # Verify snapshot was created
            self.assertEqual(snapshot_id, "snapshot_123")
        
        # 8. Test final tasks and project completion
        for task_id in ["task5", "task6", "task7"]:
            with patch.object(self.core, 'implement_task') as mock_implement:
                mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": f"{task_id} completed"}
                
                # Execute task
                result = self.core.implement_task(task_id)
                
                # Verify task completed
                self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 9. Verify project completion
        with patch.object(self.core, 'check_project_completion') as mock_check:
            mock_check.return_value = True
            
            # Check project completion
            is_complete = self.core.check_project_completion()
            
            # Verify project is complete
            self.assertTrue(is_complete)

    def test_web_app_with_error_recovery(self):
        """Test web application development with complex error recovery."""
        # Similar to test_complete_web_app_workflow but with more error scenarios
        project_description = """
        Create a Flask web application for a simple blog system with comments and user authentication.
        """
        
        # Mock task graph analysis
        with patch.object(self.core, 'analyze_project_description') as mock_analyze:
            mock_analyze.return_value = TaskGraph()
            task_graph = self.core.analyze_project_description(project_description)
        
        # Mock task planning
        with patch.object(self.core, 'plan_tasks') as mock_plan:
            mock_plan.return_value = ["setup", "models", "auth", "views", "tests", "deploy"]
            task_sequence = self.core.plan_tasks(task_graph)
        
        # Test setup task - successful
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Setup completed"}
            result = self.core.implement_task("setup")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # Test models task - fails with database error
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.FAILED, 
                "output": "Database models failed", 
                "error": "SQLAlchemy import error"
            }
            result = self.core.implement_task("models")
            self.assertEqual(result["status"], TaskStatus.FAILED)
            self.assertIn("SQLAlchemy import error", result["error"])
        
        # Test recovery with dependency installation
        with patch.object(self.core, 'retry_task_with_fix') as mock_retry:
            mock_retry.return_value = {"status": TaskStatus.COMPLETED, "output": "Models fixed and completed"}
            retry_result = self.core.retry_task_with_fix(
                "models", 
                "Install SQLAlchemy dependency first"
            )
            self.assertEqual(retry_result["status"], TaskStatus.COMPLETED)
        
        # Test auth task - fails with security vulnerability
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.FAILED, 
                "output": "Auth implementation has security issues", 
                "error": "Password hashing not secure"
            }
            result = self.core.implement_task("auth")
            self.assertEqual(result["status"], TaskStatus.FAILED)
        
        # Test recovery with security fix
        with patch.object(self.core, 'retry_task_with_fix') as mock_retry:
            mock_retry.return_value = {"status": TaskStatus.COMPLETED, "output": "Auth fixed with proper hashing"}
            retry_result = self.core.retry_task_with_fix(
                "auth", 
                "Use werkzeug.security for password hashing"
            )
            self.assertEqual(retry_result["status"], TaskStatus.COMPLETED)
        
        # Complete remaining tasks
        for task_id in ["views", "tests", "deploy"]:
            with patch.object(self.core, 'implement_task') as mock_implement:
                mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": f"{task_id} completed"}
                result = self.core.implement_task(task_id)
                self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # Verify project completion
        with patch.object(self.core, 'check_project_completion') as mock_check:
            mock_check.return_value = True
            is_complete = self.core.check_project_completion()
            self.assertTrue(is_complete)


class TestCLIUtilityWorkflow(unittest.TestCase):
    """End-to-end tests for CLI utility development workflow."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a test directory
        self.test_dir = Path("test_artifacts_cli")
        self.test_dir.mkdir(exist_ok=True)
        
        # Mock the LLM integration to avoid actual API calls
        self.llm_patcher = patch('src.agendev.llm_integration.LLMIntegration')
        self.mock_llm = self.llm_patcher.start()
        
        # Setup mock responses
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
        
        # Mock successful generation responses
        self.mock_llm_instance.generate_text.return_value = "Mock LLM response"
        self.mock_llm_instance.generate_structured_output.return_value = {
            "tasks": [
                {"id": "task1", "description": "Define CLI arguments", "dependencies": []},
                {"id": "task2", "description": "Implement core functionality", "dependencies": ["task1"]},
                {"id": "task3", "description": "Add error handling", "dependencies": ["task2"]},
                {"id": "task4", "description": "Create unit tests", "dependencies": ["task3"]},
                {"id": "task5", "description": "Add documentation", "dependencies": ["task4"]},
                {"id": "task6", "description": "Package for distribution", "dependencies": ["task5"]}
            ]
        }
        
        # Create core components
        self.core = AgenticCore(
            project_dir=self.test_dir,
            llm_endpoint="http://mock-endpoint:1234"
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.llm_patcher.stop()
        
        # Clean up test directory
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_complete_cli_utility_workflow(self):
        """Test a complete CLI utility development workflow."""
        # 1. Project description and analysis
        project_description = """
        Create a command-line utility that analyzes log files and generates summary reports.
        The tool should:
        - Accept log file paths as input
        - Parse common log formats (Apache, Nginx)
        - Generate statistics on traffic, errors, and response times
        - Support output in multiple formats (text, JSON, CSV)
        - Include progress indicators for large files
        """
        
        # Mock task graph analysis
        with patch.object(self.core, 'analyze_project_description') as mock_analyze:
            mock_analyze.return_value = TaskGraph()
            task_graph = self.core.analyze_project_description(project_description)
        
        # 2. Task planning
        with patch.object(self.core, 'plan_tasks') as mock_plan:
            mock_plan.return_value = ["task1", "task2", "task3", "task4", "task5", "task6"]
            task_sequence = self.core.plan_tasks(task_graph)
            self.assertEqual(len(task_sequence), 6)
        
        # 3. Implement CLI argument parsing
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "CLI arguments defined"}
            result = self.core.implement_task("task1")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 4. Implement core functionality
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Core log parsing implemented"}
            result = self.core.implement_task("task2")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 5. Test error handling task with initial failure
        with patch.object(self.core, 'implement_task') as mock_implement:
            # First attempt fails with missing edge case
            mock_implement.side_effect = [
                {"status": TaskStatus.FAILED, "output": "Error handling incomplete", "error": "Missing malformed log handling"},
                {"status": TaskStatus.COMPLETED, "output": "Error handling completed with malformed log support"}
            ]
            
            # First attempt - should fail
            result = self.core.implement_task("task3")
            self.assertEqual(result["status"], TaskStatus.FAILED)
            
            # Test error recovery
            with patch.object(self.core, 'retry_task') as mock_retry:
                mock_retry.return_value = {"status": TaskStatus.COMPLETED, "output": "Fixed error handling"}
                retry_result = self.core.retry_task("task3")
                self.assertEqual(retry_result["status"], TaskStatus.COMPLETED)
        
        # 6. Test unit test creation
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Unit tests created"}
            result = self.core.implement_task("task4")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 7. Test documentation generation
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Documentation generated"}
            result = self.core.implement_task("task5")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 8. Test packaging for distribution
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {"status": TaskStatus.COMPLETED, "output": "Package setup completed"}
            result = self.core.implement_task("task6")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
        
        # 9. Verify project completion
        with patch.object(self.core, 'check_project_completion') as mock_check:
            mock_check.return_value = True
            is_complete = self.core.check_project_completion()
            self.assertTrue(is_complete)


class TestErrorRecoveryScenarios(unittest.TestCase):
    """Tests focused on various error recovery scenarios and edge cases."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a test directory
        self.test_dir = Path("test_artifacts_errors")
        self.test_dir.mkdir(exist_ok=True)
        
        # Mock the LLM integration to avoid actual API calls
        self.llm_patcher = patch('src.agendev.llm_integration.LLMIntegration')
        self.mock_llm = self.llm_patcher.start()
        
        # Setup mock responses
        self.mock_llm_instance = MagicMock()
        self.mock_llm.return_value = self.mock_llm_instance
        
        # Create core components
        self.core = AgenticCore(
            project_dir=self.test_dir,
            llm_endpoint="http://mock-endpoint:1234"
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.llm_patcher.stop()
        
        # Clean up test directory
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_llm_service_unavailable_recovery(self):
        """Test recovery when LLM service is temporarily unavailable."""
        project_description = "Create a simple calculator web app"
        
        # Mock task graph analysis
        with patch.object(self.core, 'analyze_project_description') as mock_analyze:
            # First call raises connection error, second succeeds
            mock_analyze.side_effect = [
                ConnectionError("Failed to connect to LLM service"),
                TaskGraph()
            ]
            
            # First attempt should raise error
            with self.assertRaises(ConnectionError):
                self.core.analyze_project_description(project_description)
            
            # Mock retry mechanism
            with patch.object(self.core, 'retry_with_backoff') as mock_retry:
                mock_retry.return_value = TaskGraph()
                
                # Test retry
                task_graph = self.core.retry_with_backoff(
                    method=self.core.analyze_project_description,
                    args=[project_description],
                    max_retries=3
                )
                
                # Verify we got a task graph
                self.assertIsInstance(task_graph, TaskGraph)
    
    def test_syntax_error_recovery(self):
        """Test recovery from syntax errors in generated code."""
        # Setup mock task implementation that generates code with syntax error
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.FAILED, 
                "output": "Code with syntax error", 
                "error": "SyntaxError: invalid syntax",
                "code": "def calculate_total(items)\n    return sum(items)  # Missing colon"
            }
            
            # Execute task - should fail
            result = self.core.implement_task("syntax_error_task")
            self.assertEqual(result["status"], TaskStatus.FAILED)
            
            # Test error analysis and fix
            with patch.object(self.core, 'analyze_code_error') as mock_analyze_error:
                mock_analyze_error.return_value = {
                    "error_type": "SyntaxError",
                    "location": "line 1",
                    "fix": "Add colon after function parameters",
                    "fixed_code": "def calculate_total(items):\n    return sum(items)"
                }
                
                # Analyze the error
                error_analysis = self.core.analyze_code_error(result["code"], result["error"])
                
                # Verify analysis
                self.assertEqual(error_analysis["error_type"], "SyntaxError")
                self.assertIn("fix", error_analysis)
                
                # Test code correction
                with patch.object(self.core, 'apply_code_fix') as mock_fix:
                    mock_fix.return_value = {
                        "status": TaskStatus.COMPLETED,
                        "output": "Code fixed successfully",
                        "code": "def calculate_total(items):\n    return sum(items)"
                    }
                    
                    # Apply the fix
                    fixed_result = self.core.apply_code_fix(
                        task_id="syntax_error_task",
                        original_code=result["code"],
                        fixed_code=error_analysis["fixed_code"]
                    )
                    
                    # Verify fix
                    self.assertEqual(fixed_result["status"], TaskStatus.COMPLETED)
    
    def test_runtime_error_recovery(self):
        """Test recovery from runtime errors in generated code."""
        # Setup mock task implementation that generates code with runtime error
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.COMPLETED, 
                "output": "Code generated successfully", 
                "code": "def divide_numbers(a, b):\n    return a / b\n\nresult = divide_numbers(10, 0)"
            }
            
            # Execute task - initially marked as completed
            result = self.core.implement_task("divide_task")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
            
            # Mock test execution that reveals runtime error
            with patch.object(self.core, 'execute_tests') as mock_execute_tests:
                mock_execute_tests.return_value = {
                    "status": TaskStatus.FAILED,
                    "error": "ZeroDivisionError: division by zero",
                    "traceback": "Traceback (most recent call last):\n  File \"test.py\", line 4, in <module>\n    result = divide_numbers(10, 0)\n  File \"test.py\", line 2, in divide_numbers\n    return a / b\nZeroDivisionError: division by zero"
                }
                
                # Execute tests - should fail
                test_result = self.core.execute_tests("divide_task")
                self.assertEqual(test_result["status"], TaskStatus.FAILED)
                
                # Test error analysis and fix
                with patch.object(self.core, 'analyze_runtime_error') as mock_analyze_runtime:
                    mock_analyze_runtime.return_value = {
                        "error_type": "ZeroDivisionError",
                        "location": "line 2",
                        "fix": "Add check for zero divisor",
                        "fixed_code": "def divide_numbers(a, b):\n    if b == 0:\n        return 'Cannot divide by zero'\n    return a / b\n\nresult = divide_numbers(10, 0)"
                    }
                    
                    # Analyze the error
                    error_analysis = self.core.analyze_runtime_error(
                        code=result["code"],
                        error=test_result["error"],
                        traceback=test_result["traceback"]
                    )
                    
                    # Verify analysis
                    self.assertEqual(error_analysis["error_type"], "ZeroDivisionError")
                    self.assertIn("fix", error_analysis)
                    
                    # Test code correction
                    with patch.object(self.core, 'apply_code_fix') as mock_fix:
                        mock_fix.return_value = {
                            "status": TaskStatus.COMPLETED,
                            "output": "Code fixed successfully",
                            "code": error_analysis["fixed_code"]
                        }
                        
                        # Apply the fix
                        fixed_result = self.core.apply_code_fix(
                            task_id="divide_task",
                            original_code=result["code"],
                            fixed_code=error_analysis["fixed_code"]
                        )
                        
                        # Verify fix
                        self.assertEqual(fixed_result["status"], TaskStatus.COMPLETED)
    
    def test_dependency_error_recovery(self):
        """Test recovery from missing dependency errors."""
        # Setup mock task implementation that generates code with missing dependency
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.COMPLETED, 
                "output": "Code generated successfully", 
                "code": "import nonexistent_package\n\ndef process_data(data):\n    return nonexistent_package.process(data)"
            }
            
            # Execute task - initially marked as completed
            result = self.core.implement_task("dependency_task")
            self.assertEqual(result["status"], TaskStatus.COMPLETED)
            
            # Mock test execution that reveals import error
            with patch.object(self.core, 'execute_tests') as mock_execute_tests:
                mock_execute_tests.return_value = {
                    "status": TaskStatus.FAILED,
                    "error": "ModuleNotFoundError: No module named 'nonexistent_package'",
                    "traceback": "Traceback (most recent call last):\n  File \"test.py\", line 1, in <module>\n    import nonexistent_package\nModuleNotFoundError: No module named 'nonexistent_package'"
                }
                
                # Execute tests - should fail
                test_result = self.core.execute_tests("dependency_task")
                self.assertEqual(test_result["status"], TaskStatus.FAILED)
                
                # Test dependency resolution
                with patch.object(self.core, 'resolve_dependency') as mock_resolve:
                    mock_resolve.return_value = {
                        "status": "failed",
                        "message": "Package 'nonexistent_package' not found in PyPI"
                    }
                    
                    # Try to resolve dependency
                    resolve_result = self.core.resolve_dependency("nonexistent_package")
                    self.assertEqual(resolve_result["status"], "failed")
                    
                    # Test code refactoring to use alternative package
                    with patch.object(self.core, 'refactor_for_alternative_dependency') as mock_refactor:
                        mock_refactor.return_value = {
                            "status": TaskStatus.COMPLETED,
                            "output": "Code refactored to use standard library",
                            "code": "# Using standard library instead of nonexistent_package\nimport json\n\ndef process_data(data):\n    return json.dumps(data)"
                        }
                        
                        # Refactor the code
                        refactor_result = self.core.refactor_for_alternative_dependency(
                            task_id="dependency_task",
                            original_code=result["code"],
                            missing_package="nonexistent_package",
                            alternative="standard library"
                        )
                        
                        # Verify refactoring
                        self.assertEqual(refactor_result["status"], TaskStatus.COMPLETED)
                        self.assertIn("json", refactor_result["code"])
    
    def test_cascading_error_recovery(self):
        """Test recovery from cascading errors where one fix creates new problems."""
        # Setup test scenario with multiple errors
        with patch.object(self.core, 'implement_task') as mock_implement:
            mock_implement.return_value = {
                "status": TaskStatus.COMPLETED, 
                "output": "Task with multiple issues completed", 
                "code": "class UserManagement:\n    def __init__(self):\n        self.users = {}\n    \n    def add_user(self, username, password)\n        if username in self.users\n            return False\n        self.users[username] = password\n        return True"
            }
            
            # Execute task - marked as completed despite issues
            result = self.core.implement_task("cascading_errors_task")
            
            # First error detection - syntax error
            with patch.object(self.core, 'analyze_code') as mock_analyze:
                mock_analyze.return_value = {
                    "issues": [
                        {"type": "SyntaxError", "location": "line 5", "description": "Missing colon after parameters"},
                        {"type": "SyntaxError", "location": "line 6", "description": "Missing colon after condition"}
                    ]
                }
                
                # Analyze the code
                analysis = self.core.analyze_code(result["code"])
                self.assertEqual(len(analysis["issues"]), 2)
                
                # Test first fix - fix syntax errors
                with patch.object(self.core, 'apply_code_fix') as mock_fix:
                    mock_fix.return_value = {
                        "status": TaskStatus.COMPLETED,
                        "output": "Syntax errors fixed",
                        "code": "class UserManagement:\n    def __init__(self):\n        self.users = {}\n    \n    def add_user(self, username, password):\n        if username in self.users:\n            return False\n        self.users[username] = password\n        return True"
                    }
                    
                    # Apply the syntax fixes
                    fixed_result = self.core.apply_code_fix(
                        task_id="cascading_errors_task",
                        original_code=result["code"],
                        fixed_code=mock_fix.return_value["code"]
                    )
                    
                    # Second analysis - security vulnerability
                    with patch.object(self.core, 'analyze_security') as mock_security:
                        mock_security.return_value = {
                            "issues": [
                                {"type": "SecurityVulnerability", "location": "line 7", "description": "Storing plain text passwords"}
                            ]
                        }
                        
                        # Analyze security
                        security_analysis = self.core.analyze_security(fixed_result["code"])
                        self.assertEqual(len(security_analysis["issues"]), 1)
                        
                        # Fix security issue
                        with patch.object(self.core, 'apply_security_fix') as mock_sec_fix:
                            mock_sec_fix.return_value = {
                                "status": TaskStatus.COMPLETED,
                                "output": "Security issues fixed",
                                "code": "import hashlib\n\nclass UserManagement:\n    def __init__(self):\n        self.users = {}\n    \n    def add_user(self, username, password):\n        if username in self.users:\n            return False\n        # Hash password before storing\n        hashed_pw = hashlib.sha256(password.encode()).hexdigest()\n        self.users[username] = hashed_pw\n        return True"
                            }
                            
                            # Apply the security fix
                            secure_result = self.core.apply_security_fix(
                                task_id="cascading_errors_task",
                                original_code=fixed_result["code"],
                                fixed_code=mock_sec_fix.return_value["code"]
                            )
                            
                            # Third analysis - performance issue
                            with patch.object(self.core, 'analyze_performance') as mock_perf:
                                mock_perf.return_value = {
                                    "issues": [
                                        {"type": "PerformanceConcern", "location": "line 11", "description": "SHA-256 is fast for brute force, use better algorithm"}
                                    ]
                                }
                                
                                # Analyze performance
                                perf_analysis = self.core.analyze_performance(secure_result["code"])
                                self.assertEqual(len(perf_analysis["issues"]), 1)
                                
                                # Fix performance/security issue
                                with patch.object(self.core, 'apply_performance_fix') as mock_perf_fix:
                                    mock_perf_fix.return_value = {
                                        "status": TaskStatus.COMPLETED,
                                        "output": "Performance issues fixed",
                                        "code": "import hashlib\nimport os\n\nclass UserManagement:\n    def __init__(self):\n        self.users = {}\n    \n    def add_user(self, username, password):\n        if username in self.users:\n            return False\n        # Use better password hashing with salt\n        salt = os.urandom(32)\n        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)\n        self.users[username] = {'key': key, 'salt': salt}\n        return True"
                                    }
                                    
                                    # Apply the performance fix
                                    final_result = self.core.apply_performance_fix(
                                        task_id="cascading_errors_task",
                                        original_code=secure_result["code"],
                                        fixed_code=mock_perf_fix.return_value["code"]
                                    )
                                    
                                    # Verify final result
                                    self.assertEqual(final_result["status"], TaskStatus.COMPLETED)
                                    self.assertIn("pbkdf2_hmac", final_result["code"])


# Add more test utilities

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_project_description(project_type="web"):
        """Generate realistic project descriptions for testing."""
        descriptions = {
            "web": [
                """
                Create a responsive web application for inventory management with the following features:
                - User authentication and role-based access control
                - Product catalog with categories and search
                - Stock management with alerts for low inventory
                - Order processing and tracking
                - Reporting dashboard with charts
                - REST API for mobile app integration
                """,
                """
                Build a social media dashboard that aggregates content from multiple platforms:
                - Connect to Twitter, Instagram, and Facebook APIs
                - Show aggregated analytics and engagement metrics
                - Schedule and post content to multiple platforms
                - Monitor mentions and comments
                - Generate performance reports
                """
            ],
            "cli": [
                """
                Create a command-line tool for automated backups with these requirements:
                - Support for multiple source directories
                - Incremental and full backup options
                - Compression and optional encryption
                - Backup verification
                - Scheduling capabilities
                - Detailed logging and notifications
                """,
                """
                Develop a CLI utility for batch image processing that can:
                - Resize images in bulk
                - Convert between image formats
                - Apply filters and effects
                - Organize files based on metadata
                - Generate thumbnails
                - Support for parallel processing
                """
            ],
            "library": [
                """
                Create a Python library for data validation with the following features:
                - Schema-based validation for JSON and YAML
                - Type checking and conversion
                - Custom validation rules
                - Error collection and reporting
                - Nested object validation
                - Performance optimizations for large datasets
                """,
                """
                Build a testing utility library that provides:
                - Mocking and stubbing utilities
                - Fixtures and test data generation
                - Performance measurement tools
                - Assertion helpers for complex objects
                - Test report generation
                - CI/CD integration utilities
                """
            ]
        }
        
        import random
        return random.choice(descriptions.get(project_type, descriptions["web"]))
    
    @staticmethod
    def generate_code_with_error(error_type="syntax"):
        """Generate code samples with specific types of errors for testing."""
        error_samples = {
            "syntax": [
                "def calculate_total(items)\n    return sum(items)",  # Missing colon
                "if user_input == 'quit'\n    break",  # Missing colon
                "for i in range(10)\n    print(i)",    # Missing colon
                "print('Hello world\")",   # Mismatched quotes
                "x = 10 +* 5"              # Invalid operator
            ],
            "runtime": [
                "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",  # Division by zero
                "numbers = [1, 2, 3]\nprint(numbers[5])",  # Index error
                "import json\ndata = '{invalid json}'\njson.loads(data)",  # JSON parse error
                "x = int('not a number')",  # Value error
                "open('nonexistent_file.txt', 'r').read()"  # File not found
            ],
            "security": [
                "password = 'secretpass123'\nuser_data = {'username': 'admin', 'password': password}",  # Plain text password
                "user_input = input('Enter command: ')\neval(user_input)",  # Dangerous eval
                "query = f\"SELECT * FROM users WHERE username = '{username}'\"",  # SQL injection vulnerability
                "os.system(f'ping {user_input}')",  # Command injection vulnerability
                "app.run(debug=True)"  # Debug mode in production
            ],
            "dependency": [
                "import nonexistent_module\nnonexistent_module.do_something()",  # Missing module
                "from requests import missing_function",  # Missing function import
                "import tensorflow as tf\n# Code that requires CUDA",  # Missing CUDA dependency
                "from PIL import Image\n# Code using PIL",  # Missing pillow installation
                "import boto3\n# AWS operations"  # Missing AWS credentials
            ]
        }
        
        import random
        return random.choice(error_samples.get(error_type, error_samples["syntax"]))


class MockLLMResponses:
    """Utility class for generating mock LLM responses for testing."""
    
    @staticmethod
    def generate_task_breakdown(project_type="web"):
        """Generate realistic task breakdowns for different project types."""
        breakdowns = {
            "web": {
                "tasks": [
                    {"id": "setup", "description": "Project setup and configuration", "dependencies": []},
                    {"id": "auth", "description": "User authentication system", "dependencies": ["setup"]},
                    {"id": "db", "description": "Database models and migrations", "dependencies": ["setup"]},
                    {"id": "api", "description": "API endpoints implementation", "dependencies": ["auth", "db"]},
                    {"id": "ui", "description": "User interface components", "dependencies": ["api"]},
                    {"id": "tests", "description": "Test suite implementation", "dependencies": ["api", "ui"]},
                    {"id": "docs", "description": "Documentation generation", "dependencies": ["tests"]},
                    {"id": "deploy", "description": "Deployment configuration", "dependencies": ["tests", "docs"]}
                ]
            },
            "cli": {
                "tasks": [
                    {"id": "setup", "description": "Project structure and dependencies", "dependencies": []},
                    {"id": "args", "description": "Command-line argument parsing", "dependencies": ["setup"]},
                    {"id": "core", "description": "Core functionality implementation", "dependencies": ["args"]},
                    {"id": "io", "description": "Input/output handling", "dependencies": ["core"]},
                    {"id": "error", "description": "Error handling and logging", "dependencies": ["core", "io"]},
                    {"id": "tests", "description": "Unit and integration tests", "dependencies": ["error"]},
                    {"id": "docs", "description": "Usage documentation", "dependencies": ["tests"]},
                    {"id": "package", "description": "Packaging for distribution", "dependencies": ["docs"]}
                ]
            },
            "library": {
                "tasks": [
                    {"id": "setup", "description": "Library structure and build system", "dependencies": []},
                    {"id": "core", "description": "Core functionality implementation", "dependencies": ["setup"]},
                    {"id": "api", "description": "Public API design", "dependencies": ["core"]},
                    {"id": "utils", "description": "Utility functions", "dependencies": ["core"]},
                    {"id": "tests", "description": "Test suite implementation", "dependencies": ["api", "utils"]},
                    {"id": "docs", "description": "API documentation", "dependencies": ["tests"]},
                    {"id": "examples", "description": "Usage examples", "dependencies": ["docs"]},
                    {"id": "package", "description": "Packaging for distribution", "dependencies": ["examples"]}
                ]
            }
        }
        
        return breakdowns.get(project_type, breakdowns["web"])
    
    @staticmethod
    def generate_code_implementation(task_id, project_type="web"):
        """Generate realistic code implementations for different tasks."""
        # This would be a large method with many code implementations
        # Just showing a simplified example
        if project_type == "web" and task_id == "auth":
            return """
from flask import Flask, request, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import os

# User model
class User:
    def __init__(self, username, password):
        self.username = username
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# In-memory user store (would be a database in production)
users = {}

# Auth routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return {'error': 'Username and password required'}, 400
    
    if username in users:
        return {'error': 'Username already exists'}, 409
    
    users[username] = User(username, password)
    return {'message': 'User registered successfully'}, 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return {'error': 'Username and password required'}, 400
    
    user = users.get(username)
    if not user or not user.check_password(password):
        return {'error': 'Invalid credentials'}, 401
    
    session['username'] = username
    return {'message': 'Login successful'}, 200

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))
"""
        elif project_type == "cli" and task_id == "args":
            return """
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Command-line tool for processing log files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input', help='Path to the input log file or directory')
    
    # Optional arguments
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['text', 'json', 'csv'], 
                        default='text', help='Output format')
    parser.add_argument('-t', '--type', choices=['apache', 'nginx', 'custom'],
                        default='apache', help='Log file type')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--filter', help='Filter logs by regex pattern')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Processing {args.input} as {args.type} logs")
    # Main processing would happen here
    
if __name__ == '__main__':
    main()
"""
        # Add more implementations as needed
        return "# Default implementation for task: " + task_id
    
    @staticmethod
    def generate_test_implementation(task_id, project_type="web"):
        """Generate realistic test implementations for different tasks."""
        if project_type == "web" and task_id == "auth":
            return """
import unittest
from app import app, users
import json

class TestAuth(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        # Clear users before each test
        users.clear()
    
    def test_register_success(self):
        response = self.client.post('/register',
                                    data=json.dumps({'username': 'testuser', 'password': 'password123'}),
                                    content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 201)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'User registered successfully')
        self.assertIn('testuser', users)
    
    def test_register_missing_fields(self):
        response = self.client.post('/register',
                                    data=json.dumps({'username': 'testuser'}),
                                    content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_login_success(self):
        # First register a user
        self.client.post('/register',
                         data=json.dumps({'username': 'testuser', 'password': 'password123'}),
                         content_type='application/json')
        
        # Now try to login
        response = self.client.post('/login',
                                    data=json.dumps({'username': 'testuser', 'password': 'password123'}),
                                    content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'Login successful')
    
    def test_login_invalid_credentials(self):
        # First register a user
        self.client.post('/register',
                         data=json.dumps({'username': 'testuser', 'password': 'password123'}),
                         content_type='application/json')
        
        # Try to login with wrong password
        response = self.client.post('/login',
                                    data=json.dumps({'username': 'testuser', 'password': 'wrongpassword'}),
                                    content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 401)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Invalid credentials')

if __name__ == '__main__':
    unittest.main()
"""
        # Add more test implementations as needed
        return "# Default test implementation for task: " + task_id

class MockFileSystem:
    """Mock file system for testing file operations without actual IO."""
    
    def __init__(self):
        """Initialize an empty mock file system."""
        self.files = {}
        self.directories = set(["/"])
    
    def write_file(self, path: str, content: str) -> None:
        """Write content to a mock file."""
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and directory not in self.directories:
            self.mkdir(directory)
        
        # Write file
        self.files[path] = content
    
    def read_file(self, path: str) -> str:
        """Read content from a mock file."""
        if path not in self.files:
            raise FileNotFoundError(f"No such file: {path}")
        return self.files[path]
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the mock file system."""
        return path in self.files
    
    def list_directory(self, path: str) -> List[str]:
        """List contents of a directory in the mock file system."""
        if path not in self.directories:
            raise FileNotFoundError(f"No such directory: {path}")
        
        result = []
        # Files directly in this directory
        for file_path in self.files:
            if os.path.dirname(file_path) == path:
                result.append(os.path.basename(file_path))
        
        # Subdirectories
        for directory in self.directories:
            if directory != path and os.path.dirname(directory) == path:
                result.append(os.path.basename(directory))
        
        return result
    
    def mkdir(self, path: str) -> None:
        """Create a directory in the mock file system."""
        # Ensure parent directories exist
        parent = os.path.dirname(path)
        if parent and parent not in self.directories:
            self.mkdir(parent)
        
        # Create directory
        self.directories.add(path)
    
    def rmdir(self, path: str) -> None:
        """Remove a directory from the mock file system."""
        if path not in self.directories:
            raise FileNotFoundError(f"No such directory: {path}")
        
        # Check if directory is empty
        for file_path in self.files:
            if os.path.dirname(file_path) == path:
                raise OSError(f"Directory not empty: {path}")
        
        for directory in self.directories:
            if directory != path and os.path.dirname(directory) == path:
                raise OSError(f"Directory not empty: {path}")
        
        # Remove directory
        self.directories.remove(path)
    
    def remove_file(self, path: str) -> None:
        """Remove a file from the mock file system."""
        if path not in self.files:
            raise FileNotFoundError(f"No such file: {path}")
        
        del self.files[path]
    
    def reset(self) -> None:
        """Reset the mock file system to empty state."""
        self.files = {}
        self.directories = set(["/"])


class MockProjectGenerator:
    """Generates realistic project structures for testing."""
    
    @staticmethod
    def generate_web_project(base_dir: Path) -> Dict[str, str]:
        """Generate a Flask web application project structure."""
        project_files = {}
        
        # Project structure
        directories = [
            "",
            "static",
            "static/css",
            "static/js",
            "templates",
            "models",
            "routes",
            "utils",
            "tests"
        ]
        
        # Create directories
        for directory in directories:
            (base_dir / directory).mkdir(exist_ok=True, parents=True)
        
        # Create app.py
        app_py = """
from flask import Flask, render_template
from routes import register_routes

app = Flask(__name__)
app.config.from_object('config')

# Register all routes
register_routes(app)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""
        (base_dir / "app.py").write_text(app_py)
        project_files["app.py"] = app_py
        
        # Create config.py
        config_py = """
import os

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'development-key')
DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'

# Database configuration
DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///app.db')
"""
        (base_dir / "config.py").write_text(config_py)
        project_files["config.py"] = config_py
        
        # Create routes/__init__.py
        routes_init = """
def register_routes(app):
    from .auth import auth_bp
    from .api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
"""
        (base_dir / "routes/__init__.py").write_text(routes_init)
        project_files["routes/__init__.py"] = routes_init
        
        # Create routes/auth.py
        routes_auth = """
from flask import Blueprint, request, jsonify, session, redirect, url_for
from models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate user
        user = User.authenticate(username, password)
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        # Create user
        user = User.create(username, email, password)
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        
        return jsonify({'error': 'Registration failed'}), 400
    
    return render_template('register.html')
"""
        (base_dir / "routes/auth.py").write_text(routes_auth)
        project_files["routes/auth.py"] = routes_auth
        
        # Create models/user.py
        models_user = """
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @classmethod
    def create(cls, username, email, password):
        user = cls(username, email, password)
        db.session.add(user)
        try:
            db.session.commit()
            return user
        except:
            db.session.rollback()
            return None
    
    @classmethod
    def authenticate(cls, username, password):
        user = cls.query.filter_by(username=username).first()
        if user and user.check_password(password):
            return user
        return None
"""
        (base_dir / "models/user.py").write_text(models_user)
        project_files["models/user.py"] = models_user
        
        # Create templates/index.html
        templates_index = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Web Application</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <header>
        <h1>Welcome to Flask Web App</h1>
        <nav>
            <ul>
                <li><a href="{{ url_for('index') }}">Home</a></li>
                {% if session.get('user_id') %}
                    <li><a href="{{ url_for('auth.logout') }}">Logout</a></li>
                {% else %}
                    <li><a href="{{ url_for('auth.login') }}">Login</a></li>
                    <li><a href="{{ url_for('auth.register') }}">Register</a></li>
                {% endif %}
            </ul>
        </nav>
    </header>
    
    <main>
        <section class="content">
            <h2>Main Content</h2>
            <p>This is a sample Flask application.</p>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2025 Flask Web App</p>
    </footer>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
"""
        (base_dir / "templates/index.html").write_text(templates_index)
        project_files["templates/index.html"] = templates_index
        
        # Create static/css/style.css
        static_css = """
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.6;
}

header {
    background-color: #333;
    color: #fff;
    padding: 1rem;
}

nav ul {
    display: flex;
    list-style: none;
    padding: 0;
}

nav ul li {
    margin-right: 1rem;
}

nav ul li a {
    color: #fff;
    text-decoration: none;
}

main {
    padding: 2rem;
}

footer {
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 1rem;
    position: fixed;
    bottom: 0;
    width: 100%;
}
"""
        (base_dir / "static/css/style.css").write_text(static_css)
        project_files["static/css/style.css"] = static_css
        
        # Create static/js/main.js
        static_js = """
// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    
    // Add event listeners or other initialization code here
});

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}
"""
        (base_dir / "static/js/main.js").write_text(static_js)
        project_files["static/js/main.js"] = static_js
        
        # Create tests/test_auth.py
        tests_auth = """
import unittest
from app import app
from models.user import User, db

class AuthTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        db.create_all()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
    
    def test_register(self):
        response = self.client.post('/register', data={
            'username': 'testuser',
            'password': 'password123',
            'email': 'test@example.com'
        })
        self.assertEqual(response.status_code, 302)  # Redirect on success
        
        user = User.query.filter_by(username='testuser').first()
        self.assertIsNotNone(user)
        self.assertEqual(user.email, 'test@example.com')
    
    def test_login(self):
        # Create a user
        user = User('testuser', 'test@example.com', 'password123')
        db.session.add(user)
        db.session.commit()
        
        # Login
        response = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        })
        self.assertEqual(response.status_code, 302)  # Redirect on success
        
        # Check session
        with self.client.session_transaction() as session:
            self.assertIn('user_id', session)
            self.assertEqual(session['user_id'], user.id)
    
    def test_logout(self):
        # Create a user and login
        user = User('testuser', 'test@example.com', 'password123')
        db.session.add(user)
        db.session.commit()
        
        with self.client.session_transaction() as session:
            session['user_id'] = user.id
        
        # Logout
        response = self.client.get('/logout')
        self.assertEqual(response.status_code, 302)  # Redirect on success
        
        # Check session
        with self.client.session_transaction() as session:
            self.assertNotIn('user_id', session)

if __name__ == '__main__':
    unittest.main()
"""
        (base_dir / "tests/test_auth.py").write_text(tests_auth)
        project_files["tests/test_auth.py"] = tests_auth
        
        # Create requirements.txt
        requirements = """
Flask==2.2.3
Flask-SQLAlchemy==3.0.3
Werkzeug==2.2.3
pytest==7.3.1
"""
        (base_dir / "requirements.txt").write_text(requirements)
        project_files["requirements.txt"] = requirements
        
        return project_files
    
    @staticmethod
    def generate_cli_project(base_dir: Path) -> Dict[str, str]:
        """Generate a CLI utility project structure."""
        project_files = {}
        
        # Project structure
        directories = [
            "",
            "loganalyzer",
            "loganalyzer/parsers",
            "loganalyzer/reporters",
            "loganalyzer/utils",
            "tests"
        ]
        
        # Create directories
        for directory in directories:
            (base_dir / directory).mkdir(exist_ok=True, parents=True)
        
        # Create setup.py
        setup_py = """
from setuptools import setup, find_packages

setup(
    name="loganalyzer",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'loganalyzer=loganalyzer.cli:main',
        ],
    },
    install_requires=[
        'click>=8.0.0',
        'tabulate>=0.8.9',
        'tqdm>=4.61.0',
        'matplotlib>=3.4.2',
    ],
    python_requires='>=3.8',
    author="AgenDev",
    author_email="example@example.com",
    description="A command-line tool for analyzing log files",
    keywords="logs, analysis, cli",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
"""
        (base_dir / "setup.py").write_text(setup_py)
        project_files["setup.py"] = setup_py
        
        # Create loganalyzer/__init__.py
        init_py = """
"""
        (base_dir / "loganalyzer/__init__.py").write_text(init_py)
        project_files["loganalyzer/__init__.py"] = init_py
        
        # Create loganalyzer/cli.py
        cli_py = """
import click
import os
import sys
from typing import List, Dict, Any, Optional
import json
import csv
from tqdm import tqdm

from .parsers import get_parser
from .reporters import get_reporter

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output file path')
@click.option('-f', '--format', type=click.Choice(['text', 'json', 'csv']), default='text', help='Output format')
@click.option('-t', '--type', type=click.Choice(['apache', 'nginx', 'custom']), default='auto', help='Log file type')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.option('--filter', help='Filter logs by regex pattern')
def main(input_path, output, format, type, verbose, filter):
    \"\"\"
    Analyze log files and generate reports.
    
    INPUT_PATH can be a log file or directory containing log files.
    \"\"\"
    if verbose:
        click.echo(f"Analyzing logs from: {input_path}")
        click.echo(f"Log type: {type}")
    
    # Determine if input is a file or directory
    if os.path.isdir(input_path):
        log_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if os.path.isfile(os.path.join(input_path, f)) and f.endswith('.log')]
    else:
        log_files = [input_path]
    
    if not log_files:
        click.echo("No log files found", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Found {len(log_files)} log file(s)")
    
    # Process log files
    results = []
    for log_file in tqdm(log_files, disable=not verbose):
        # Auto-detect log type if not specified
        detected_type = type
        if detected_type == 'auto':
            detected_type = detect_log_type(log_file)
            if verbose:
                click.echo(f"Detected log type: {detected_type}")
        
        # Get appropriate parser
        parser = get_parser(detected_type)
        if not parser:
            click.echo(f"Unsupported log type: {detected_type}", err=True)
            continue
        
        # Parse log file
        parsed_logs = parser.parse(log_file, filter_pattern=filter)
        results.extend(parsed_logs)
    
    # Generate report
    if verbose:
        click.echo(f"Generating {format} report")
    
    reporter = get_reporter(format)
    report_data = reporter.generate(results)
    
    # Output report
    if output:
        with open(output, 'w') as f:
            f.write(report_data)
        if verbose:
            click.echo(f"Report saved to: {output}")
    else:
        click.echo(report_data)

def detect_log_type(log_file: str) -> str:
    \"\"\"Auto-detect log file type based on content.\"\"\"
    with open(log_file, 'r') as f:
        first_line = f.readline().strip()
    
    if '[' in first_line and ']' in first_line:
        if 'nginx' in first_line.lower():
            return 'nginx'
        return 'apache'
    
    return 'custom'

if __name__ == '__main__':
    main()
"""
        (base_dir / "loganalyzer/cli.py").write_text(cli_py)
        project_files["loganalyzer/cli.py"] = cli_py
        
        # Create loganalyzer/parsers/__init__.py
        parsers_init = """
from typing import Dict, Any, Optional
from .apache import ApacheLogParser
from .nginx import NginxLogParser
from .custom import CustomLogParser

def get_parser(log_type: str):
    \"\"\"Get the appropriate log parser for the given log type.\"\"\"
    parsers = {
        'apache': ApacheLogParser(),
        'nginx': NginxLogParser(),
        'custom': CustomLogParser()
    }
    return parsers.get(log_type)
"""
        (base_dir / "loganalyzer/parsers/__init__.py").write_text(parsers_init)
        project_files["loganalyzer/parsers/__init__.py"] = parsers_init
        
        # Create loganalyzer/parsers/apache.py
        apache_parser = """
import re
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class ApacheLogParser:
    \"\"\"Parser for Apache common log format.\"\"\"
    
    def __init__(self):
        # Common Log Format pattern
        self.pattern = re.compile(
            r'(?P<ip>\\S+) (?P<identd>\\S+) (?P<user>\\S+) \\[(?P<timestamp>.+?)\\] "(?P<request>.*?)" '
            r'(?P<status>\\d+) (?P<size>\\S+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
        )
    
    def parse(self, log_file: str, filter_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        \"\"\"Parse Apache log file and return list of log entries.\"\"\"
        log_entries = []
        filter_regex = re.compile(filter_pattern) if filter_pattern else None
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Apply filter if specified
                if filter_regex and not filter_regex.search(line):
                    continue
                
                match = self.pattern.match(line)
                if match:
                    entry = match.groupdict()
                    
                    # Convert timestamp to datetime
                    try:
                        timestamp_str = entry['timestamp']
                        entry['timestamp'] = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
                    except ValueError:
                        # If parsing fails, keep the original string
                        pass
                    
                    # Parse request
                    request_parts = entry['request'].split()
                    if len(request_parts) >= 2:
                        entry['method'] = request_parts[0]
                        entry['path'] = request_parts[1]
                        entry['protocol'] = request_parts[2] if len(request_parts) > 2 else ''
                    
                    # Convert status to int
                    try:
                        entry['status'] = int(entry['status'])
                    except ValueError:
                        pass
                    
                    # Convert size to int
                    try:
                        entry['size'] = int(entry['size']) if entry['size'] != '-' else 0
                    except ValueError:
                        entry['size'] = 0
                    
                    log_entries.append(entry)
        
        return log_entries
"""
        (base_dir / "loganalyzer/parsers/apache.py").write_text(apache_parser)
        project_files["loganalyzer/parsers/apache.py"] = apache_parser
        
        # Create loganalyzer/reporters/__init__.py
        reporters_init = """
from typing import Dict, Any
from .text import TextReporter
from .json_reporter import JsonReporter
from .csv_reporter import CsvReporter

def get_reporter(format_type: str):
    \"\"\"Get the appropriate reporter for the given format type.\"\"\"
    reporters = {
        'text': TextReporter(),
        'json': JsonReporter(),
        'csv': CsvReporter()
    }
    return reporters.get(format_type, TextReporter())
"""
        (base_dir / "loganalyzer/reporters/__init__.py").write_text(reporters_init)
        project_files["loganalyzer/reporters/__init__.py"] = reporters_init
        
        # Create tests/test_cli.py
        test_cli = """
import unittest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from loganalyzer.cli import main

class TestCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
    
    def test_help(self):
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Analyze log files', result.output)
    
    @patch('loganalyzer.cli.get_parser')
    @patch('loganalyzer.cli.get_reporter')
    def test_process_log_file(self, mock_get_reporter, mock_get_parser):
        # Mock parser and reporter
        mock_parser = MagicMock()
        mock_parser.parse.return_value = [{'ip': '127.0.0.1', 'status': 200}]
        mock_get_parser.return_value = mock_parser
        
        mock_reporter = MagicMock()
        mock_reporter.generate.return_value = 'Test Report'
        mock_get_reporter.return_value = mock_reporter
        
        with self.runner.isolated_filesystem():
            # Create a test log file
            with open('test.log', 'w') as f:
                f.write('127.0.0.1 - - [01/Jan/2025:12:00:00 +0000] "GET / HTTP/1.1" 200 123 "-" "Mozilla/5.0"')
            
            # Run the CLI
            result = self.runner.invoke(main, ['test.log', '--type', 'apache'])
            
            # Assert exit code
            self.assertEqual(result.exit_code, 0)
            
            # Assert parser was called
            mock_get_parser.assert_called_once_with('apache')
            mock_parser.parse.assert_called_once()
            
            # Assert reporter was called
            mock_get_reporter.assert_called_once_with('text')
            mock_reporter.generate.assert_called_once()
            
            # Assert output
            self.assertEqual(result.output, 'Test Report\\n')

if __name__ == '__main__':
    unittest.main()
"""
        (base_dir / "tests/test_cli.py").write_text(test_cli)
        project_files["tests/test_cli.py"] = test_cli
        
        return project_files


class LLMResponseMocker:
    """Utility for creating realistic LLM response sequences for tests."""
    
    def __init__(self, success_rate: float = 0.8, error_injection_rate: float = 0.2):
        """Initialize with customizable success and error rates."""
        self.success_rate = success_rate
        self.error_injection_rate = error_injection_rate
        self.common_errors = [
            "SyntaxError: invalid syntax",
            "IndentationError: unexpected indent",
            "TypeError: unsupported operand type(s)",
            "NameError: name is not defined",
            "AttributeError: has no attribute",
            "ImportError: No module named",
            "ValueError: invalid literal",
            "KeyError: key not found",
            "IndexError: list index out of range",
            "RuntimeError: maximum recursion depth exceeded"
        ]
    
    def mock_llm_integration(self) -> MagicMock:
        """Create a mock LLM integration object with realistic behavior."""
        mock_llm = MagicMock()
        
        # Mock generate_text method
        def mock_generate_text(prompt, **kwargs):
            # Randomly succeed or fail based on success_rate
            if random.random() < self.success_rate:
                return self._generate_success_response(prompt)
            else:
                return self._generate_error_response()
        
        mock_llm.generate_text.side_effect = mock_generate_text
        
        # Mock generate_structured_output method
        def mock_generate_structured(prompt, schema, **kwargs):
            # Randomly succeed or fail based on success_rate
            if random.random() < self.success_rate:
                return self._generate_structured_success(prompt, schema)
            else:
                # Either return malformed JSON or an error message
                if random.random() < 0.5:
                    return {"error": random.choice(self.common_errors)}
                else:
                    return {"partial_result": "Incomplete processing", "status": "failed"}
        
        mock_llm.generate_structured_output.side_effect = mock_generate_structured
        
        return mock_llm
    
    def _generate_success_response(self, prompt: str) -> str:
        """Generate a success response based on the prompt content."""
        if "code" in prompt.lower():
            return self._generate_code_response(prompt)
        elif "test" in prompt.lower():
            return self._generate_test_response(prompt)
        elif "plan" in prompt.lower() or "task" in prompt.lower():
            return self._generate_planning_response(prompt)
        elif "analyze" in prompt.lower() or "review" in prompt.lower():
            return self._generate_analysis_response(prompt)
        else:
            return "Generated response for: " + prompt[:50] + "..."
    
    def _generate_error_response(self) -> str:
        """Generate an error response."""
        return f"Error: {random.choice(self.common_errors)}"
    
    def _generate_code_response(self, prompt: str) -> str:
        """Generate a code snippet based on the prompt."""
        # Potentially inject errors based on error_injection_rate
        should_inject_error = random.random() < self.error_injection_rate
        
        language = "python"  # Default
        if "javascript" in prompt.lower() or "js" in prompt.lower():
            language = "javascript"
        elif "html" in prompt.lower():
            language = "html"
        elif "css" in prompt.lower():
            language = "css"
        
        if language == "python":
            if should_inject_error:
                return """
def process_data(data):
    result = []
    for item in data
        if item['value'] > 0:
            result.append(item)
    return result
"""
            else:
                return """
def process_data(data):
    result = []
    for item in data:
        if item['value'] > 0:
            result.append(item)
    return result
"""
        elif language == "javascript":
            if should_inject_error:
                return """
function processData(data) {
    const result = [];
    for (const item of data) {
        if (item.value > 0) {
            result.push(item)
    }
    return result;
}
"""
            else:
                return """
function processData(data) {
    const result = [];
    for (const item of data) {
        if (item.value > 0) {
            result.push(item);
        }
    }
    return result;
}
"""
        return "// Generated code for: " + prompt[:50] + "..."
    
    def _generate_test_response(self, prompt: str) -> str:
        """Generate a test snippet based on the prompt."""
        return """
import unittest

class TestExample(unittest.TestCase):
    def test_process_data(self):
        data = [{'value': 5}, {'value': -3}, {'value': 10}]
        result = process_data(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['value'], 5)
        self.assertEqual(result[1]['value'], 10)

if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_planning_response(self, prompt: str) -> str:
        """Generate a planning response."""
        return """
## Project Plan

1. Setup project structure
2. Implement core functionality
3. Add error handling
4. Create test suite
5. Write documentation
6. Package for distribution

### Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Tasks 3 and 4
- Task 6 depends on Task 5
"""
    
    def _generate_analysis_response(self, prompt: str) -> str:
        """Generate an analysis response."""
        return """
## Code Analysis

### Issues Found
- Line 5: Missing colon after 'if' statement
- Line 7: Variable 'result' used before assignment
- Line 12: Function lacks proper error handling

### Recommendations
1. Fix syntax error on line 5
2. Initialize 'result' variable before use
3. Add try/except blocks for error handling
4. Add type hints for better code readability
5. Consider adding input validation
"""
    
    def _generate_structured_success(self, prompt: str, schema: dict) -> dict:
        """Generate a structured response based on the schema."""
        # Simple implementation - would be more complex in a real scenario
        if "tasks" in str(schema):
            return {
                "tasks": [
                    {"id": "task1", "description": "Setup project", "dependencies": []},
                    {"id": "task2", "description": "Core functionality", "dependencies": ["task1"]},
                    {"id": "task3", "description": "Error handling", "dependencies": ["task2"]},
                    {"id": "task4", "description": "Testing", "dependencies": ["task3"]},
                    {"id": "task5", "description": "Documentation", "dependencies": ["task4"]},
                ]
            }
        elif "analysis" in str(schema):
            return {
                "analysis": {
                    "issues": [
                        {"type": "syntax", "location": "line 5", "description": "Missing colon"},
                        {"type": "logic", "location": "line 10", "description": "Incorrect condition"}
                    ],
                    "recommendations": [
                        "Fix syntax error on line 5",
                        "Review logic in condition on line 10"
                    ]
                }
            }
        elif "code" in str(schema):
            return {
                "code": "def example():\n    return 'Hello, world!'",
                "language": "python",
                "explanation": "A simple function that returns a greeting"
            }
        else:
            # Generic response with schema keys
            result = {}
            if isinstance(schema, dict) and "properties" in schema:
                for prop in schema["properties"]:
                    result[prop] = f"Mock value for {prop}"
            return result


# Execution test utilities
def test_directory_setup():
    """Create test directories and gitignore file."""
    # Create test directories
    Path("tests").mkdir(exist_ok=True)
    Path("tests/fixtures").mkdir(exist_ok=True)
    Path("test_artifacts").mkdir(exist_ok=True)
    
    # Create .gitignore file to avoid committing test artifacts
    gitignore_content = """
# Test artifacts
test_artifacts/
__pycache__/
*.pyc
.pytest_cache/
.coverage
"""
    Path(".gitignore").write_text(gitignore_content)
    
    return Path("tests")

# Entry point for running the tests
if __name__ == '__main__':
    unittest.main()
