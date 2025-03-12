# test_generation.py
"""Automatic generation of unit and integration tests."""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Union, Any, Tuple
from pathlib import Path
import os
import re
import ast
import inspect
from enum import Enum
import importlib.util
from pydantic import BaseModel, Field, model_validator

from .llm_integration import LLMIntegration, LLMConfig, Message
from .context_management import ContextManager
from .utils.fs_utils import resolve_path, save_json, load_json

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
            print(f"Error extracting code elements from {file_path}: {e}")
            
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
                print(f"Error generating tests for {py_file}: {e}")
        
        return test_files