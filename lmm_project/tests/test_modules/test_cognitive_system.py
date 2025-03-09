"""
Test script for the integrated cognitive system.

This script demonstrates how the perception, attention, and memory modules
interact at different developmental stages.
"""

import logging
import sys
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import cognitive components
from lmm_project.modules.perception import get_module as get_perception_module
from lmm_project.modules.attention import get_module as get_attention_module
from lmm_project.modules.memory import get_module as get_memory_module
from lmm_project.core.event_bus import EventBus

def print_section(title):
    """Print a section divider with title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """Pretty print a dictionary with indentation and depth control"""
    if current_depth >= max_depth:
        print(" " * indent + "...")
        return
        
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(" " * indent + f"{key}: [")
            for i, item in enumerate(value[:3]):  # Show first 3 items
                print(" " * (indent + 4) + f"Item {i}:")
                print_dict(item, indent + 8, max_depth, current_depth + 1)
            if len(value) > 3:
                print(" " * (indent + 4) + f"... ({len(value) - 3} more items)")
            print(" " * indent + "]")
        else:
            # Truncate very long values
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(" " * indent + f"{key}: {value}")

class CognitiveSystem:
    """
    Simple integration of cognitive modules
    
    This class demonstrates how the different cognitive modules interact
    through the event bus.
    """
    def __init__(self, development_level: float = 0.0):
        """Initialize the cognitive system"""
        # Create shared event bus
        self.event_bus = EventBus()
        
        # Initialize modules
        self.perception = get_perception_module(
            module_id="perception",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        self.attention = get_attention_module(
            module_id="attention",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        self.memory = get_memory_module(
            module_id="memory",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        # Set development level
        self.development_level = development_level
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input through the cognitive pipeline
        
        This simulates the flow of information through the cognitive system:
        1. Perception processes the raw input
        2. Attention determines what aspects to focus on
        3. Memory stores and retrieves relevant information
        
        Args:
            text: Input text to process
            
        Returns:
            Integrated results from all modules
        """
        # Step 1: Perception processes the input
        perception_result = self.perception.process_input({"text": text})
        
        # At early development, only perception works
        if self.development_level < 0.2:
            return {
                "development_level": self.development_level,
                "perception": perception_result,
                "attention": "Not yet developed",
                "memory": "Not yet developed"
            }
            
        # Step 2: Attention processes the perception result
        # Note: this would normally happen through event messaging
        # but we're explicitly calling it here for demonstration
        attention_result = self.attention.process_input({
            "content": perception_result,
            "source": "perception",
            # Calculate intensity based on patterns
            "intensity": min(1.0, len(perception_result.get("patterns", [])) / 10),
            # Higher novelty for questions and exclamations
            "novelty": 0.8 if "?" in text or "!" in text else 0.5
        })
        
        # Step 3: Memory operations
        if self.development_level < 0.4:
            # Only working memory at early development
            memory_result = self.memory.process_input({
                "operation": "store",
                "memory_type": "working",
                "content": {
                    "text": text,
                    "perception": perception_result,
                    "attention": attention_result
                }
            })
        else:
            # At higher development, also store in episodic memory
            memory_result = self.memory.process_input({
                "operation": "store",
                "memory_type": "episodic",
                "content": {
                    "text": text,
                    "perception": perception_result,
                    "attention": attention_result,
                    "timestamp": time.time()
                }
            })
        
        # Return integrated results
        return {
            "development_level": self.development_level,
            "perception": perception_result,
            "attention": attention_result,
            "memory": memory_result
        }
        
    def set_development_level(self, level: float):
        """Set development level for all modules"""
        self.development_level = level
        self.perception.development_level = level
        self.perception._update_submodule_development()
        self.attention.development_level = level
        self.attention._adjust_parameters_for_development()
        self.memory.development_level = level
        self.memory._adjust_memory_for_development()
        
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get the current state of all cognitive modules"""
        return {
            "development_level": self.development_level,
            "perception": self.perception.get_state(),
            "attention": self.attention.get_state(),
            "memory": self.memory.get_state()
        }

def test_cognitive_system_at_level(level: float, inputs: List[str]):
    """Test the cognitive system at a specific developmental level"""
    print_section(f"Testing Cognitive System at Development Level {level:.1f}")
    
    # Create cognitive system
    system = CognitiveSystem(development_level=level)
    
    # Process each input
    for i, text in enumerate(inputs):
        print_section(f"Processing Input {i+1}: '{text}'")
        
        # Process the input
        result = system.process_text(text)
        
        # Print key results based on development level
        print("\nPerception Results:")
        patterns = result["perception"].get("patterns", [])
        if patterns:
            print(f"  Recognized {len(patterns)} patterns:")
            for pattern in patterns[:3]:  # Show first 3 patterns
                print(f"    - {pattern['pattern_type']} (confidence: {pattern['confidence']:.2f})")
            if len(patterns) > 3:
                print(f"    - ... ({len(patterns) - 3} more patterns)")
        else:
            print("  No patterns recognized")
            
        # Print interpretation if available
        if "interpretation" in result["perception"]:
            print("\nInterpretation:")
            print_dict(result["perception"]["interpretation"], 2)
        
        # Print attention results if developed
        if level >= 0.2 and result["attention"] != "Not yet developed":
            print("\nAttention Results:")
            print(f"  Captures attention: {result['attention'].get('captures_attention', False)}")
            print(f"  Salience: {result['attention'].get('salience', 0):.2f}")
            if result["attention"].get("current_focus"):
                print("  Current focus:")
                print(f"    Source: {result['attention']['current_focus'].get('source', 'unknown')}")
                print(f"    Salience: {result['attention']['current_focus'].get('salience', 0):.2f}")
        
        # Print memory results if developed
        if level >= 0.2 and result["memory"] != "Not yet developed":
            print("\nMemory Results:")
            print(f"  Operation: {result['memory'].get('operation')}")
            print(f"  Status: {result['memory'].get('status')}")
            if "item_id" in result["memory"]:
                print(f"  Item ID: {result['memory']['item_id']}")
            elif "episode_id" in result["memory"]:
                print(f"  Episode ID: {result['memory']['episode_id']}")
    
    # Print cognitive state summary
    print_section("Cognitive State Summary")
    state = system.get_cognitive_state()
    
    print(f"Development Level: {state['development_level']:.1f}")
    
    print("\nPerception:")
    print(f"  Sensory development: {state['perception'].get('capabilities', {}).get('sensory_development', 0):.1f}")
    print(f"  Pattern development: {state['perception'].get('capabilities', {}).get('pattern_development', 0):.1f}")
    
    if level >= 0.2:
        print("\nAttention:")
        print(f"  Capacity: {state['attention'].get('capacity', 1)}")
        print(f"  Active focuses: {state['attention'].get('active_focuses', 0)}")
        
        print("\nMemory:")
        print(f"  Working memory: {state['memory'].get('working_memory', {}).get('current_usage', 0)}/{state['memory'].get('working_memory', {}).get('capacity', 0)} items")
        if level >= 0.4:
            print(f"  Episodic memory: {state['memory'].get('episodic_memory', {}).get('episode_count', 0)} episodes")
        if level >= 0.6:
            print(f"  Semantic memory: {state['memory'].get('semantic_memory', {}).get('item_count', 0)} concepts")

def main():
    """Main test function"""
    # Test inputs
    test_inputs = [
        "Hello, how are you today?",
        "I'm trying to build an artificial cognitive system.",
        "This is a test of developmental cognitive architecture!",
        "Can you understand and remember complex information?",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # Test at different developmental stages
    for level in [0.0, 0.3, 0.6, 0.9]:
        test_cognitive_system_at_level(level, test_inputs)
        
    print_section("Test Complete")
    
if __name__ == "__main__":
    main() 