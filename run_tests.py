"""
Test Runner

This script provides a convenient way to run the NeuralChild test suite.
It uses pytest to run the tests and provides options to run specific test groups.
"""

import argparse
import os
import sys
import subprocess


def run_tests(args):
    """Run the specified tests using pytest."""
    # Build the pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add test path if specified
    if args.path:
        cmd.append(args.path)
    else:
        cmd.append("tests/")
    
    # Add markers if specified
    if args.marker:
        cmd.append(f"-m {args.marker}")
    
    # Add specific test file or test case if specified
    if args.test:
        cmd.append(args.test)
    
    # Add coverage reporting if requested
    if args.coverage:
        cmd.extend(["--cov=neuralchild", "--cov-report=term", "--cov-report=html"])
    
    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run the NeuralChild test suite.")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-p", "--path", help="Specify the test directory or file to run")
    parser.add_argument("-m", "--marker", help="Run tests with specific marker (e.g., 'components')")
    parser.add_argument("-t", "--test", help="Run a specific test (e.g., 'tests/unit/test_child.py::TestChild::test_init')")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate test coverage report")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main()) 