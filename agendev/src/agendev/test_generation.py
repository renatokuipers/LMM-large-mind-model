# test_generation.py
"""Automatic generation of unit and integration tests."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple
from pathlib import Path
import os
import re
import ast
import inspect
import subprocess
import time
import logging
import tempfile
from enum import Enum
import importlib.util
from pydantic import BaseModel, Field, model_validator

from .llm_integration import LLMIntegration, LLMConfig, Message
from .context_management import ContextManager
from .utils.fs_utils import resolve_path, save_json, load_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(str, Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    PERFORMANCE = "performance"

class CodeElement(BaseModel):
    """Represents a code element to test."""
    name: str
    element_type: str  # "function", "class", "method"
    source_file: str
    line_start: int
    line_end: int
    code: str
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

class TestCase(BaseModel):
    """Represents a test case."""
    id: str = Field(default_factory=lambda: f"test_{id(object)}")
    name: str
    description: str
    test_type: TestType
    target_element: CodeElement
    test_code: str
    created_at: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    
    @model_validator(mode='after')
    def validate_test_case(self) -> 'TestCase':
        """Ensure the test case is valid."""
        if not self.test_code or not self.target_element:
            raise ValueError("Test case must have code and target element")
        return self

class TestSuite(BaseModel):
    """Represents a test suite."""
    name: str
    description: str
    target_module: str
    test_cases: List[TestCase] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""

class TestResult(BaseModel):
    """Results of executing a test."""
    success: bool = False
    test_name: str
    execution_time: float = 0.0
    errors: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    coverage: Optional[float] = None
    raw_output: str = ""

class TestExecutionSummary(BaseModel):
    """Summary of all test executions for a test suite."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    execution_time: float = 0.0
    overall_success: bool = False
    results: List[TestResult] = Field(default_factory=list)

class CodeIssue(BaseModel):
    """Represents a detected issue in code."""
    issue_type: str
    description: str
    severity: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    source_code: Optional[str] = None
    suggested_fix: Optional[str] = None

class TestGenerator:
    """Generates tests for code."""
    
    def __init__(
        self,
        llm_integration: LLMIntegration,
        context_manager: Optional[ContextManager] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the test generator.
        
        Args:
            llm_integration: LLM integration for generating tests
            context_manager: Optional context manager for code understanding
            output_dir: Optional directory for test output
        """
        self.llm = llm_integration
        self.context_manager = context_manager
        self.output_dir = resolve_path(output_dir or "quality/tests", create_parents=True)
        
        # Set up system message for test generation
        self.llm.set_system_message(
            "You are an expert Python test engineer specialized in generating high-quality test cases."
            "Your tests should be comprehensive, follow best practices, and use pytest."
        )
    
    def extract_code_elements(self, file_path: Union[str, Path]) -> List[CodeElement]:
        """
        Extract testable code elements from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of code elements
        """
        elements = []
        file_path = resolve_path(file_path)
        
        if not file_path.exists():
            return []
            
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content)
            
            # Track imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append(f"{module}.{name.name}")
            
            # Extract functions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions
                    if node.name.startswith('_') and not node.name.startswith('__'):
                        continue
                        
                    # Get the function code
                    func_lines = content.splitlines()[node.lineno-1:node.end_lineno]
                    func_code = '\n'.join(func_lines)
                    
                    elements.append(CodeElement(
                        name=node.name,
                        element_type="function",
                        source_file=str(file_path),
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        code=func_code,
                        imports=imports
                    ))
                    
                elif isinstance(node, ast.ClassDef):
                    # Extract class and its methods
                    class_lines = content.splitlines()[node.lineno-1:node.end_lineno]
                    class_code = '\n'.join(class_lines)
                    
                    elements.append(CodeElement(
                        name=node.name,
                        element_type="class",
                        source_file=str(file_path),
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        code=class_code,
                        imports=imports
                    ))
                    
                    # Extract methods
                    for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                        # Skip private methods
                        if method.name.startswith('_') and not method.name.startswith('__'):
                            continue
                            
                        method_lines = content.splitlines()[method.lineno-1:method.end_lineno]
                        method_code = '\n'.join(method_lines)
                        
                        elements.append(CodeElement(
                            name=f"{node.name}.{method.name}",
                            element_type="method",
                            source_file=str(file_path),
                            line_start=method.lineno,
                            line_end=method.end_lineno,
                            code=method_code,
                            imports=imports,
                            dependencies=[node.name]  # The class is a dependency
                        ))
            
        except Exception as e:
            logger.error(f"Error extracting code elements from {file_path}: {e}")
            
        return elements
    
    def generate_test_case(
        self,
        element: CodeElement,
        test_type: TestType = TestType.UNIT
    ) -> TestCase:
        """
        Generate a test case for a code element.
        
        Args:
            element: Code element to test
            test_type: Type of test to generate
            
        Returns:
            Generated test case
        """
        # Prepare the prompt
        prompt = self._create_test_prompt(element, test_type)
        
        # Use LLM to generate test
        test_code = self.llm.query(prompt)
        
        # Clean up the code (remove markdown formatting if present)
        test_code = self._clean_code(test_code)
        
        # Create test case
        return TestCase(
            name=f"test_{element.name.lower().replace('.', '_')}",
            description=f"Test for {element.element_type} {element.name}",
            test_type=test_type,
            target_element=element,
            test_code=test_code
        )
    
    def _create_test_prompt(self, element: CodeElement, test_type: TestType) -> str:
        """
        Create a prompt for test generation.
        
        Args:
            element: Code element to test
            test_type: Type of test to generate
            
        Returns:
            Prompt for the LLM
        """
        # Get related context if context manager is available
        related_context = ""
        if self.context_manager:
            # Use context manager to find related code
            similar_elements = self.context_manager.find_similar_elements(element.code, top_k=3)
            if similar_elements:
                related_context = "\n\nRelated code context:\n"
                for elem, score in similar_elements:
                    if elem.content != element.code:  # Skip exact matches
                        related_context += f"\n```python\n{elem.content}\n```\n"
        
        # Build the prompt
        prompt = f"""
        Generate a comprehensive {test_type.value} test for the following Python {element.element_type}:
        
        ```python
        {element.code}
        ```
        
        File: {element.source_file}
        
        Imports in the file:
        {', '.join(element.imports)}
        
        {related_context}
        
        Requirements:
        1. Use pytest for the test framework
        2. Include detailed test cases covering edge cases
        3. Use appropriate mocking for external dependencies
        4. Include docstrings explaining the tests
        5. Follow best practices for Python testing
        
        Only return the Python test code, properly formatted and ready to use.
        Do not include explanations outside of code comments.
        """
        
        return prompt
    
    def _clean_code(self, code: str) -> str:
        """
        Clean generated code.
        
        Args:
            code: Code to clean
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = re.sub(r"^```python\n", "", code)
            code = re.sub(r"\n```$", "", code)
        elif code.startswith("```"):
            code = re.sub(r"^```\n", "", code)
            code = re.sub(r"\n```$", "", code)
            
        return code.strip()
    
    def generate_test_suite(
        self,
        module_path: Union[str, Path],
        test_types: List[TestType] = [TestType.UNIT]
    ) -> TestSuite:
        """
        Generate a test suite for a module.
        
        Args:
            module_path: Path to the module
            test_types: Types of tests to generate
            
        Returns:
            Generated test suite
        """
        module_path = resolve_path(module_path)
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
            
        # Extract code elements
        elements = self.extract_code_elements(module_path)
        
        # Create test suite
        suite = TestSuite(
            name=f"Test{module_path.stem.capitalize()}",
            description=f"Tests for {module_path.stem}",
            target_module=str(module_path),
            imports=[
                "import pytest",
                f"import {module_path.stem.replace('-', '_')}",
                "from unittest import mock"
            ]
        )
        
        # Generate test cases for each element
        for element in elements:
            for test_type in test_types:
                test_case = self.generate_test_case(element, test_type)
                suite.test_cases.append(test_case)
        
        # Generate setup and teardown code if needed
        if len(suite.test_cases) > 1:
            suite.setup_code = self._generate_setup_code(suite, elements)
            suite.teardown_code = self._generate_teardown_code(suite)
        
        return suite
    
    def _generate_setup_code(self, suite: TestSuite, elements: List[CodeElement]) -> str:
        """
        Generate setup code for a test suite.
        
        Args:
            suite: Test suite
            elements: Code elements
            
        Returns:
            Generated setup code
        """
        # Check if there are classes that might need fixtures
        has_classes = any(e.element_type == "class" for e in elements)
        
        if has_classes:
            return """
@pytest.fixture
def setup_test_environment():
    # Set up any resources needed for tests
    yield
    # Clean up resources after tests
"""
        return ""
    
    def _generate_teardown_code(self, suite: TestSuite) -> str:
        """
        Generate teardown code for a test suite.
        
        Args:
            suite: Test suite
            
        Returns:
            Generated teardown code
        """
        return ""
    
    def save_test_suite(self, suite: TestSuite) -> str:
        """
        Save a test suite to a file.
        
        Args:
            suite: Test suite to save
            
        Returns:
            Path to the saved file
        """
        # Create test file path
        module_name = Path(suite.target_module).stem
        test_file_path = self.output_dir / f"test_{module_name}.py"
        
        # Generate the test file content
        content = self._generate_test_file_content(suite)
        
        # Save the file
        os.makedirs(test_file_path.parent, exist_ok=True)
        with open(test_file_path, 'w') as f:
            f.write(content)
            
        # Save test suite metadata
        metadata_path = self.output_dir / f"test_{module_name}_metadata.json"
        save_json(suite.model_dump(), metadata_path)
        
        return str(test_file_path)
    
    def _generate_test_file_content(self, suite: TestSuite) -> str:
        """
        Generate test file content from a test suite.
        
        Args:
            suite: Test suite
            
        Returns:
            Generated file content
        """
        lines = [
            "# Auto-generated test file",
            f"# Target: {suite.target_module}",
            f"# Generated at: {__import__('datetime').datetime.now().isoformat()}",
            ""
        ]
        
        # Add imports
        for import_line in suite.imports:
            lines.append(import_line)
        lines.append("")
        
        # Add setup code
        if suite.setup_code:
            lines.append(suite.setup_code)
            lines.append("")
        
        # Add test cases
        for test_case in suite.test_cases:
            lines.append(f"# {test_case.description}")
            lines.append(test_case.test_code)
            lines.append("")
        
        # Add teardown code
        if suite.teardown_code:
            lines.append(suite.teardown_code)
        
        return "\n".join(lines)
    
    def generate_tests_for_directory(
        self,
        directory: Union[str, Path],
        test_types: List[TestType] = [TestType.UNIT],
        pattern: str = "*.py"
    ) -> List[str]:
        """
        Generate tests for all matching files in a directory.
        
        Args:
            directory: Directory to scan
            test_types: Types of tests to generate
            pattern: File pattern to match
            
        Returns:
            List of generated test file paths
        """
        directory = resolve_path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Not a valid directory: {directory}")
            
        # Find all Python files
        python_files = list(directory.glob(pattern))
        
        # Generate tests for each file
        test_files = []
        for py_file in python_files:
            # Skip test files and __init__.py
            if py_file.name.startswith("test_") or py_file.name == "__init__.py":
                continue
                
            try:
                suite = self.generate_test_suite(py_file, test_types)
                test_file = self.save_test_suite(suite)
                test_files.append(test_file)
            except Exception as e:
                logger.error(f"Error generating tests for {py_file}: {e}")
        
        return test_files

class TestExecutor:
    """Executes tests and analyzes results."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the test executor.
        
        Args:
            output_dir: Optional directory for test output
        """
        self.output_dir = resolve_path(output_dir or "quality/test_results", create_parents=True)
        self.logger = logging.getLogger(__name__)
    
    def execute_test_suite(self, test_file_path: Union[str, Path]) -> TestExecutionSummary:
        """
        Execute a test suite and process the results.
        
        Args:
            test_file_path: Path to the test file
            
        Returns:
            Test execution summary
        """
        test_file_path = resolve_path(test_file_path)
        
        if not test_file_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file_path}")
        
        # Create result directory
        results_dir = self.output_dir / test_file_path.stem
        os.makedirs(results_dir, exist_ok=True)
        
        # Execute pytest with detailed output
        start_time = time.time()
        process = subprocess.run(
            ["pytest", str(test_file_path), "-v", "--no-header", "--tb=native"],
            capture_output=True,
            text=True
        )
        execution_time = time.time() - start_time
        
        # Save raw output
        with open(results_dir / "raw_output.txt", "w") as f:
            f.write(process.stdout)
            f.write("\n\n")
            f.write(process.stderr)
        
        # Parse results
        return self._parse_pytest_output(process.stdout, process.stderr, execution_time)
    
    def _parse_pytest_output(self, stdout: str, stderr: str, execution_time: float) -> TestExecutionSummary:
        """
        Parse pytest output to extract test results.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            execution_time: Execution time in seconds
            
        Returns:
            Parsed test execution summary
        """
        lines = stdout.splitlines()
        
        # Initialize summary
        summary = TestExecutionSummary(execution_time=execution_time)
        
        # Process each line
        current_test = None
        
        for line in lines:
            # Check for test result lines
            test_match = re.match(r"(.*?)::(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)", line)
            if test_match:
                if current_test:
                    summary.results.append(current_test)
                
                module, test_name, status = test_match.groups()
                
                current_test = TestResult(
                    test_name=test_name,
                    success=status == "PASSED",
                    raw_output=line
                )
                
                # Update counts
                summary.total_tests += 1
                if status == "PASSED":
                    summary.passed_tests += 1
                elif status == "FAILED":
                    summary.failed_tests += 1
                elif status == "ERROR":
                    summary.error_tests += 1
            
            # Collect error details if we have a current test
            elif current_test and line.strip().startswith(("E ", "=")):
                if current_test.success is False:
                    # Add to failures or errors list based on the message prefix
                    if line.strip().startswith("E "):
                        current_test.errors.append(line.strip()[2:].strip())  # Remove "E " prefix
                    else:
                        current_test.failures.append(line.strip())
        
        # Add the last test
        if current_test:
            summary.results.append(current_test)
        
        # Calculate overall success
        summary.overall_success = summary.failed_tests == 0 and summary.error_tests == 0
        
        return summary

    def analyze_code_issues(self, code: str, file_name: str = "temp_code.py") -> List[CodeIssue]:
        """
        Analyze code for common issues and potential bugs.
        
        Args:
            code: Code to analyze
            file_name: Temporary file name to use
            
        Returns:
            List of detected code issues
        """
        issues = []
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # First check for syntax errors using Python's AST
            try:
                ast.parse(code)
            except SyntaxError as e:
                issues.append(CodeIssue(
                    issue_type="syntax",
                    description=str(e),
                    severity="HIGH",
                    file_path=file_name,
                    line_number=e.lineno,
                    column=e.offset,
                    source_code=e.text
                ))
            
            # Check for common code quality issues
            issues.extend(self._check_code_quality(code, file_name))
            
            # Try to run pylint if available
            try:
                # Run pylint for static analysis
                process = subprocess.run(
                    ["pylint", "--output-format=json", tmp_path],
                    capture_output=True,
                    text=True
                )
                
                # Parse pylint output if successful
                if process.returncode != 127:  # 127 means command not found
                    try:
                        import json
                        pylint_results = json.loads(process.stdout)
                        
                        for result in pylint_results:
                            if isinstance(result, dict):
                                # Convert pylint severity to our format
                                severity_map = {
                                    "error": "HIGH",
                                    "warning": "MEDIUM",
                                    "convention": "LOW",
                                    "refactor": "LOW",
                                    "info": "INFO"
                                }
                                
                                severity = severity_map.get(result.get("type", ""), "MEDIUM")
                                
                                issues.append(CodeIssue(
                                    issue_type="pylint",
                                    description=result.get("message", "Unknown issue"),
                                    severity=severity,
                                    file_path=file_name,
                                    line_number=result.get("line"),
                                    column=result.get("column"),
                                    source_code=result.get("message-id")
                                ))
                    except json.JSONDecodeError:
                        # If pylint doesn't produce valid JSON, extract issues with regex
                        for line in process.stdout.splitlines():
                            match = re.search(r"(error|warning|info|refactor|convention).*?:.*?:(.*?):", line, re.IGNORECASE)
                            if match:
                                severity, message = match.groups()
                                severity_str = "HIGH" if severity.lower() == "error" else "MEDIUM"
                                
                                issues.append(CodeIssue(
                                    issue_type="pylint",
                                    description=message.strip(),
                                    severity=severity_str,
                                    file_path=file_name,
                                    line_number=None,
                                    column=None,
                                    source_code=None
                                ))
            except Exception as e:
                logger.warning(f"Pylint check failed: {e}")
                # Continue without pylint results
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
                pass
        
        return issues
    
    def _check_code_quality(self, code: str, file_path: str) -> List[CodeIssue]:
        """
        Check code for common quality issues.
        
        Args:
            code: Code to check
            file_path: Path to the code file
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for empty code
        if not code.strip():
            issues.append(CodeIssue(
                issue_type="code_quality",
                description="Empty code implementation",
                severity="HIGH",
                file_path=file_path,
                line_number=None,
                column=None,
                source_code=None
            ))
            return issues
        
        # Check for placeholder comments or TODO markers
        todo_pattern = r"(TODO|FIXME|XXX|NOTE):"
        todos = re.findall(todo_pattern, code)
        if todos:
            issues.append(CodeIssue(
                issue_type="code_quality",
                description=f"Found {len(todos)} placeholder or TODO markers",
                severity="MEDIUM",
                file_path=file_path,
                line_number=None,
                column=None,
                source_code=None
            ))
        
        # Check for potential infinite loops
        if "while True" in code and "break" not in code:
            issues.append(CodeIssue(
                issue_type="code_quality",
                description="Potential infinite loop (while True without break)",
                severity="HIGH",
                file_path=file_path,
                line_number=None,
                column=None,
                source_code=None
            ))
        
        # Check for excessive complexity
        lines = code.split("\n")
        indentation_levels = {}
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith("#"):
                indent = len(line) - len(line.lstrip())
                indentation_levels[i] = indent // 4  # Assuming 4 spaces per indentation level
        
        # Consider excessive indentation as a code quality issue
        max_indent = max(indentation_levels.values()) if indentation_levels else 0
        if max_indent > 5:
            issues.append(CodeIssue(
                issue_type="code_quality",
                description=f"Excessive code nesting (maximum indentation level: {max_indent})",
                severity="MEDIUM",
                file_path=file_path,
                line_number=None,
                column=None,
                source_code=None
            ))
            
        # Check for missing test assertions in test files
        if "test_" in file_path:
            has_assertions = any(
                re.search(r"assert\s+|self\.assert|\.assert\w+\(|pytest\.", line) 
                for line in lines
            )
            if not has_assertions:
                issues.append(CodeIssue(
                    issue_type="test_quality",
                    description="Test file contains no assertions",
                    severity="HIGH",
                    file_path=file_path,
                    line_number=None,
                    column=None,
                    source_code=None
                ))
        
        return issues

class TestGeneratorEnhanced:
    """Enhanced version of the test generator with integrated error detection."""
    
    def __init__(
        self, 
        test_generator: TestGenerator, 
        feedback_loop_manager  # No type hint to avoid circular import
    ):
        """
        Initialize the enhanced test generator.
        
        Args:
            test_generator: The base test generator instance
            feedback_loop_manager: The feedback loop manager for error handling
        """
        self.test_generator = test_generator
        self.feedback_loop_manager = feedback_loop_manager
        self.test_executor = TestExecutor()
        self.logger = logging.getLogger(__name__)
    
    def generate_and_validate_tests(
        self, 
        module_path: Union[str, Path], 
        test_types: List[Union[TestType, str]] = [TestType.UNIT]
    ) -> Dict[str, Any]:
        """
        Generate tests for a module and validate them.
        
        Args:
            module_path: Path to the module
            test_types: Types of tests to generate
            
        Returns:
            Dictionary with test generation and validation results
        """
        module_path = resolve_path(module_path)
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        # Generate test suite
        self.logger.info(f"Generating tests for {module_path}")
        
        try:
            # Convert string test types to TestType enum if needed
            enum_test_types = []
            for test_type in test_types:
                if isinstance(test_type, str):
                    enum_test_types.append(TestType(test_type.lower()))
                else:
                    enum_test_types.append(test_type)
            
            suite = self.test_generator.generate_test_suite(module_path, enum_test_types)
            test_file_path = self.test_generator.save_test_suite(suite)
            
            # Validate the generated test code
            test_file_path = Path(test_file_path)
            with open(test_file_path, 'r') as f:
                test_code = f.read()
            
            # Check for issues in the test code
            self.logger.info(f"Analyzing test code for {test_file_path}")
            code_issues = self.test_executor.analyze_code_issues(test_code, str(test_file_path))
            
            # If there are high-severity issues, regenerate the test case
            high_severity_issues = [issue for issue in code_issues if issue.severity == "HIGH"]
            
            if high_severity_issues:
                self.logger.warning(f"Found {len(high_severity_issues)} high-severity issues in generated tests. Attempting to fix...")
                
                # Fix issues using feedback loop
                fixed_code = self._fix_test_code_issues(test_code, high_severity_issues, module_path)
                
                if fixed_code:
                    self.logger.info("Successfully fixed test code issues. Saving updated tests.")
                    # Save the fixed code
                    with open(test_file_path, 'w') as f:
                        f.write(fixed_code)
                
                # Re-analyze after fixing
                with open(test_file_path, 'r') as f:
                    test_code = f.read()
                code_issues = self.test_executor.analyze_code_issues(test_code, str(test_file_path))
            
            # Execute the tests
            self.logger.info(f"Executing tests in {test_file_path}")
            try:
                test_results = self.test_executor.execute_test_suite(test_file_path)
                
                # If tests failed and we have feedback loop enabled, try to fix
                if not test_results.overall_success:
                    self.logger.warning(f"Tests failed with {test_results.failed_tests} failures and {test_results.error_tests} errors. Attempting to fix...")
                    
                    fixed_test_code = self._fix_failing_tests(test_code, test_results, module_path)
                    
                    if fixed_test_code:
                        self.logger.info("Successfully fixed failing tests. Saving updated tests.")
                        # Save the fixed tests
                        with open(test_file_path, 'w') as f:
                            f.write(fixed_test_code)
                        
                        # Execute the fixed tests
                        self.logger.info("Re-executing fixed tests")
                        test_results = self.test_executor.execute_test_suite(test_file_path)
                
            except Exception as e:
                self.logger.error(f"Error executing tests: {e}")
                test_results = TestExecutionSummary(
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    error_tests=1,
                    overall_success=False,
                    results=[
                        TestResult(
                            test_name="test_execution",
                            success=False,
                            errors=[str(e)]
                        )
                    ]
                )
            
            # Prepare result
            result = {
                "module_path": str(module_path),
                "test_file_path": str(test_file_path),
                "test_types": [tt.value for tt in enum_test_types],
                "test_cases": len(suite.test_cases),
                "code_issues": [issue.model_dump() for issue in code_issues],
                "test_results": test_results.model_dump(),
                "success": test_results.overall_success and not any(issue.severity == "HIGH" for issue in code_issues)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating tests: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                "module_path": str(module_path),
                "test_file_path": None,
                "test_types": [tt.value if isinstance(tt, TestType) else tt for tt in test_types],
                "test_cases": 0,
                "code_issues": [],
                "test_results": None,
                "success": False,
                "error": str(e)
            }
    
    def _fix_test_code_issues(self, test_code: str, issues: List[CodeIssue], module_path: Path) -> Optional[str]:
        """
        Fix issues in test code using the feedback loop.
        
        Args:
            test_code: The test code to fix
            issues: List of detected issues
            module_path: Path to the module being tested
            
        Returns:
            Fixed test code if successful, None otherwise
        """
        try:
            # Extract module name for context
            module_name = module_path.stem
            
            # Create context for feedback loop
            context = {
                "title": f"Fix test issues for {module_name}",
                "description": f"Fix issues in tests for {module_name}",
                "type": "test_fix"
            }
            
            # Create synthetic exception with details about the issues
            issue_details = "\n".join(f"- {issue.issue_type}: {issue.description}" for issue in issues)
            error_msg = f"Test code has issues:\n{issue_details}"
            error = Exception(error_msg)
            
            # Apply feedback loop
            feedback_result = self.feedback_loop_manager.apply_feedback_loop(
                task_id=module_name,  # Use module name as task ID
                error=error,
                implementation=test_code,
                task_context=context,
                retry_count=0  # First attempt
            )
            
            return feedback_result.get("fixed_implementation")
            
        except Exception as e:
            self.logger.error(f"Error fixing test code issues: {e}")
            return None
    
    def _fix_failing_tests(self, test_code: str, test_results: TestExecutionSummary, module_path: Path) -> Optional[str]:
        """
        Fix failing tests using the feedback loop.
        
        Args:
            test_code: The test code to fix
            test_results: Results of test execution
            module_path: Path to the module being tested
            
        Returns:
            Fixed test code if successful, None otherwise
        """
        try:
            # Extract failing test details
            failing_tests = [result for result in test_results.results if not result.success]
            if not failing_tests:
                return None
                
            # Create error details
            error_details = []
            for test in failing_tests:
                error_msg = f"Test '{test.test_name}' failed:"
                for error in test.errors + test.failures:
                    error_msg += f"\n  {error}"
                error_details.append(error_msg)
                
            error_summary = "\n".join(error_details)
            
            # Create context for feedback loop
            context = {
                "title": f"Fix failing tests for {module_path.stem}",
                "description": f"Fix failing tests for {module_path.stem}",
                "type": "test_fix",
                "failing_tests": [test.test_name for test in failing_tests]
            }
            
            # Create synthetic exception with details about failures
            error = Exception(f"Tests failed:\n{error_summary}")
            
            # Apply feedback loop
            feedback_result = self.feedback_loop_manager.apply_feedback_loop(
                task_id=module_path.stem,  # Use module name as task ID
                error=error,
                implementation=test_code,
                task_context=context,
                retry_count=0  # First attempt
            )
            
            return feedback_result.get("fixed_implementation")
            
        except Exception as e:
            self.logger.error(f"Error fixing failing tests: {e}")
            return None

    def analyze_implementation_against_tests(
        self, 
        implementation_path: Union[str, Path], 
        test_file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze implementation code against tests to identify potential issues.
        
        Args:
            implementation_path: Path to the implementation code
            test_file_path: Path to the test file
            
        Returns:
            Dictionary with analysis results
        """
        implementation_path = resolve_path(implementation_path)
        test_file_path = resolve_path(test_file_path)
        
        if not implementation_path.exists():
            raise FileNotFoundError(f"Implementation file not found: {implementation_path}")
            
        if not test_file_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file_path}")
            
        try:
            # Read the files
            with open(implementation_path, 'r') as f:
                implementation_code = f.read()
                
            with open(test_file_path, 'r') as f:
                test_code = f.read()
                
            # Analyze code issues in implementation
            implementation_issues = self.test_executor.analyze_code_issues(
                implementation_code, str(implementation_path)
            )
            
            # Extract tested functions/methods from the test file
            tested_elements = self._extract_tested_elements(test_code)
            
            # Extract functions/methods from the implementation
            implementation_elements = self._extract_code_elements(implementation_code)
            
            # Identify untested elements
            untested_elements = [
                elem for elem in implementation_elements
                if elem["name"] not in tested_elements
            ]
            
            # Execute the tests
            try:
                test_results = self.test_executor.execute_test_suite(test_file_path)
            except Exception as e:
                self.logger.error(f"Error executing tests: {e}")
                test_results = TestExecutionSummary(
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    error_tests=1,
                    overall_success=False
                )
            
            # Prepare result
            result = {
                "implementation_path": str(implementation_path),
                "test_file_path": str(test_file_path),
                "implementation_issues": [issue.model_dump() for issue in implementation_issues],
                "test_coverage": {
                    "total_elements": len(implementation_elements),
                    "tested_elements": len(tested_elements),
                    "untested_elements": [elem["name"] for elem in untested_elements],
                    "coverage_percentage": len(tested_elements) / max(1, len(implementation_elements)) * 100
                },
                "test_results": test_results.model_dump(),
                "success": test_results.overall_success and not any(issue.severity == "HIGH" for issue in implementation_issues)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing implementation against tests: {e}")
            return {
                "implementation_path": str(implementation_path),
                "test_file_path": str(test_file_path),
                "implementation_issues": [],
                "test_coverage": {
                    "total_elements": 0,
                    "tested_elements": 0,
                    "untested_elements": [],
                    "coverage_percentage": 0
                },
                "test_results": None,
                "success": False,
                "error": str(e)
            }
    
    def _extract_tested_elements(self, test_code: str) -> Set[str]:
        """
        Extract elements being tested from test code.
        
        Args:
            test_code: Test code to analyze
            
        Returns:
            Set of tested element names
        """
        tested_elements = set()
        
        try:
            # Parse the code
            tree = ast.parse(test_code)
            
            # Look for test functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Extract the tested element name from the function name
                    element_name = node.name[5:]  # Remove "test_" prefix
                    
                    # If the name contains underscores, it might be a method (class_method)
                    if "_" in element_name:
                        parts = element_name.split("_")
                        # Add both the potential class name and the method name
                        tested_elements.add("_".join(parts))
                        
                        # If it looks like class_method, also add the original format (Class.method)
                        if len(parts) == 2:
                            class_name, method_name = parts
                            # Convert to camel case if it's likely a class name
                            if class_name and class_name[0].isalpha():
                                camel_class = "".join(part.capitalize() for part in class_name.split("_"))
                                tested_elements.add(f"{camel_class}.{method_name}")
                    else:
                        tested_elements.add(element_name)
                        
        except Exception as e:
            self.logger.error(f"Error extracting tested elements: {e}")
            
        return tested_elements
    
    def _extract_code_elements(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract all functions and methods from code.
        
        Args:
            code: Code to analyze
            
        Returns:
            List of code elements (functions and methods)
        """
        elements = []
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Extract functions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions
                    if node.name.startswith('_') and not node.name.startswith('__'):
                        continue
                        
                    elements.append({
                        "name": node.name,
                        "type": "function",
                        "line": node.lineno
                    })
                    
                elif isinstance(node, ast.ClassDef):
                    # Add the class
                    elements.append({
                        "name": node.name,
                        "type": "class",
                        "line": node.lineno
                    })
                    
                    # Extract methods
                    for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                        # Skip private methods
                        if method.name.startswith('_') and not method.name.startswith('__'):
                            continue
                            
                        elements.append({
                            "name": f"{node.name}.{method.name}",
                            "type": "method",
                            "line": method.lineno
                        })
                        
        except Exception as e:
            self.logger.error(f"Error extracting code elements: {e}")
            
        return elements