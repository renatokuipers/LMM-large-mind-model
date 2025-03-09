"""
Test script for the Perception module.

This script demonstrates how the perception module processes different 
text inputs at various developmental stages, showing how its capabilities
evolve from simple text detection to sophisticated pattern recognition.
"""

import logging
import sys
from typing import Dict, Any, List
import time
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import perception module
from lmm_project.modules.perception import get_module as get_perception_module
from lmm_project.core.event_bus import EventBus

# Helper functions for pretty printing
def print_section(title):
    """Print a section header with formatting"""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"| {title} |")
    print(f"{border}\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """Recursively print a dictionary with proper indentation"""
    if current_depth >= max_depth:
        print(" " * indent + "...")
        return
    
    for key, value in data.items():
        if key in ["text"] and isinstance(value, str) and len(value) > 100:
            # Truncate long text
            print(" " * indent + f"{key}: {value[:100]}...")
        elif isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and len(value) > 3:
                print(" " * indent + f"{key}: [{len(value)} items]")
                # Print first 3 items
                for i, item in enumerate(value[:3]):
                    print(" " * (indent + 4) + f"Item {i}:")
                    print_dict(item, indent + 8, max_depth, current_depth + 1)
                if len(value) > 3:
                    print(" " * (indent + 4) + f"... and {len(value) - 3} more items")
            else:
                print(" " * indent + f"{key}: {value}")
        else:
            print(" " * indent + f"{key}: {value}")

class PerceptionTester:
    """A class to test the perception module at different developmental levels"""
    
    def __init__(self, development_level: float = 0.0):
        """Initialize the perception tester with a specific development level"""
        self.event_bus = EventBus()
        self.perception = get_perception_module(
            module_id="perception_test",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        # Keep track of test results
        self.results = []
        
        # Log initialization
        logging.info(f"Initialized PerceptionTester at development level {development_level:.1f}")
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text input through the perception module"""
        # Generate a process ID based on timestamp
        process_id = f"test_{int(time.time())}"
        
        # Log the input
        logging.info(f"Processing: '{text[:50]}...' (id: {process_id})")
        
        # Process the input
        start_time = time.time()
        result = self.perception.process_input({
            "text": text,
            "process_id": process_id
        })
        processing_time = time.time() - start_time
        
        # Add processing time and store in results
        result["processing_time_ms"] = int(processing_time * 1000)
        self.results.append({
            "text": text,
            "result": result,
            "development_level": self.perception.development_level
        })
        
        # Log completion
        logging.info(f"Processing completed in {processing_time:.3f} seconds")
        
        return result
    
    def print_result_summary(self, result: Dict[str, Any]):
        """Print a summary of the perception result"""
        print_section("Result Summary")
        
        # Basic information
        print(f"Development Level: {result.get('development_level', 0):.2f}")
        print(f"Processing Time: {result.get('processing_time_ms', 0)} ms")
        print(f"Text: '{result.get('text', '')[:100]}...'")
        
        # Pattern summary
        patterns = result.get("patterns", [])
        print(f"\nDetected Patterns: {len(patterns)}")
        
        # Summarize pattern types
        pattern_types = {}
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "unknown")
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        print("\nPattern Types:")
        for pattern_type, count in pattern_types.items():
            print(f"  - {pattern_type}: {count}")
        
        # Show high confidence patterns
        high_confidence_patterns = [p for p in patterns if p.get("confidence", 0) > 0.7]
        if high_confidence_patterns:
            print("\nHigh Confidence Patterns:")
            for i, pattern in enumerate(high_confidence_patterns[:3]):
                print(f"  {i+1}. Type: {pattern.get('pattern_type')}, Confidence: {pattern.get('confidence', 0):.2f}")
                # Show attributes if available
                if "attributes" in pattern and pattern["attributes"]:
                    for k, v in pattern["attributes"].items():
                        if isinstance(v, str) and len(v) > 50:
                            v = v[:50] + "..."
                        print(f"     - {k}: {v}")
        
        # Show interpretation if available
        if "interpretation" in result:
            print("\nInterpretation:")
            interpretation = result["interpretation"]
            print(f"  Content Type: {interpretation.get('content_type', 'unknown')}")
            print(f"  Complexity: {interpretation.get('complexity', 'unknown')}")
            if "novelty_level" in interpretation:
                print(f"  Novelty: {interpretation.get('novelty_level', 0):.2f}")
            if "primary_pattern_type" in interpretation:
                print(f"  Primary Pattern Type: {interpretation.get('primary_pattern_type', 'unknown')}")
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Print a detailed view of the perception result"""
        print_section("Detailed Result")
        print_dict(result, max_depth=4)
    
    def print_module_state(self):
        """Print the current state of the perception module and its submodules"""
        print_section("Perception Module State")
        
        # Get module states
        state = self.perception.get_state()
        sensory_state = self.perception.sensory_processor.get_state()
        pattern_state = self.perception.pattern_recognizer.get_state()
        
        # Print summary
        print(f"Perception System:")
        print(f"  - Development Level: {state.get('development_level', 0):.2f}")
        print(f"  - Module ID: {state.get('module_id', 'unknown')}")
        print(f"  - Device: {state.get('device', 'unknown')}")
        
        print(f"\nSensory Processor:")
        print(f"  - Development Level: {sensory_state.get('development_level', 0):.2f}")
        print(f"  - Recent Input Count: {sensory_state.get('recent_input_count', 0)}")
        print(f"  - Token Frequency Count: {sensory_state.get('token_frequency_count', 0)}")
        
        print(f"\nPattern Recognizer:")
        print(f"  - Development Level: {pattern_state.get('development_level', 0):.2f}")
        print(f"  - Known Pattern Count: {pattern_state.get('known_pattern_count', 0)}")
    
    def set_development_level(self, level: float):
        """Set the development level of the perception module"""
        prev_level = self.perception.development_level
        self.perception.update_development(level - prev_level)
        logging.info(f"Updated development level from {prev_level:.2f} to {self.perception.development_level:.2f}")
    
    def save_results(self, filename: str = None):
        """Save test results to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perception_test_results_{timestamp}.json"
            
        # Create directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)
        filepath = os.path.join("test_results", filename)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            # Create a simplified version of the result for storage
            serializable_results.append({
                "text": result["text"],
                "development_level": result["development_level"],
                "timestamp": datetime.now().isoformat(),
                "pattern_count": len(result.get("result", {}).get("patterns", [])),
                "processing_time_ms": result.get("result", {}).get("processing_time_ms", 0),
            })
            
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        logging.info(f"Results saved to {filepath}")

def test_perception_at_level(level: float, inputs: List[str]):
    """Test the perception module at a specific development level"""
    print_section(f"Testing Perception at Level {level:.1f}")
    
    tester = PerceptionTester(development_level=level)
    tester.print_module_state()
    
    print_section("Processing Inputs")
    for i, text in enumerate(inputs):
        print(f"\nInput {i+1}: '{text[:50]}...'")
        result = tester.process_text(text)
        tester.print_result_summary(result)
    
    # Save results
    tester.save_results(f"perception_level_{level:.1f}.json")
    
    return tester

def test_development_progression(inputs: List[str]):
    """Test how perception capabilities evolve across development levels"""
    print_section("Testing Development Progression")
    
    # Initialize at the lowest level
    tester = PerceptionTester(development_level=0.0)
    
    # Define development stages to test
    stages = [0.0, 0.3, 0.6, 0.9]
    
    # Use the same inputs across each stage
    for stage in stages:
        # Set the development level
        tester.set_development_level(stage)
        
        print_section(f"Development Level: {stage:.1f}")
        tester.print_module_state()
        
        # Process each input
        for i, text in enumerate(inputs):
            print(f"\nInput {i+1}: '{text[:50]}...'")
            result = tester.process_text(text)
            tester.print_result_summary(result)
    
    # Save results
    tester.save_results("perception_development_progression.json")
    
    return tester

def main():
    """Main test function"""
    print_section("Perception Module Test")
    
    # Example inputs at different complexity levels
    simple_inputs = [
        "Hello world.",
        "This is a simple test.",
        "How are you today?"
    ]
    
    medium_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital city of France?",
        "I'm feeling happy today because the sun is shining!"
    ]
    
    complex_inputs = [
        "When I contemplate the wonders of the universe, I'm filled with awe at the vastness and complexity of existence.",
        "Could the fundamental nature of consciousness be an emergent property of complex neural systems, or is there something more to it?",
        "The integration of artificial intelligence into everyday life presents both unprecedented opportunities and significant ethical challenges that society must address."
    ]
    
    # Test each development stage with different inputs
    print_section("Testing Specific Development Levels")
    
    # Test basic perception (0.0)
    test_perception_at_level(0.0, simple_inputs)
    
    # Test intermediate perception (0.5)
    test_perception_at_level(0.5, medium_inputs)
    
    # Test advanced perception (0.9)
    test_perception_at_level(0.9, complex_inputs)
    
    # Test progression across development stages with the same inputs
    test_development_progression([
        "Hello, how are you today?",
        "The integration of AI into society raises important ethical questions.",
        "What is the meaning of consciousness in a digital world?"
    ])
    
    print_section("Test Complete")

if __name__ == "__main__":
    main()
