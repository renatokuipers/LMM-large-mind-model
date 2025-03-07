"""
Test script for thought module integration with the Large Mind Model.

This script tests the integration of the thought module with other mind modules
and verifies that cognitive processes are working correctly.
"""
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

from lmm.main import LargeMindsModel
from lmm.core.development.stages import DevelopmentalStage
from lmm.core.mind_modules.thought import ThoughtType, CognitiveProcess

def test_thought_generation():
    """Test the generation of thoughts and their integration with other modules."""
    print("Initializing Large Mind Model...")
    lmm = LargeMindsModel()
    
    # Set to a specific development stage for testing
    lmm.set_developmental_stage(DevelopmentalStage.MIDDLE_CHILDHOOD.value)
    print(f"Development stage set to: {lmm.stage_manager.get_current_stage()}")
    
    # Test message that should stimulate multiple cognitive processes
    test_message = "I wonder why the sky is blue and how rainbows form?"
    
    print(f"\nSending test message: '{test_message}'")
    response = lmm.interact(test_message)
    
    print(f"\nLMM Response: {response}")
    
    # Get module status to verify thought module integration
    modules_status = lmm.get_mind_modules_status()
    print("\nMind Modules Status:")
    for module, status in modules_status.items():
        print(f"  {module}: {status['status']}")
    
    # Test direct thought generation
    print("\nTesting direct thought generation...")
    thought_result = lmm.thought_module.process({
        "operation": "generate_thought",
        "content": "How do complex systems emerge from simple rules?",
        "context": {"topic": "complexity_theory"},
        "developmental_stage": DevelopmentalStage.MIDDLE_CHILDHOOD.value
    })
    
    print("\nGenerated thought:")
    if thought_result.get("success"):
        thought = thought_result.get("thought", {})
        print(f"  Content: {thought.get('content')}")
        print(f"  Type: {thought.get('type')}")
        print(f"  Complexity: {thought.get('complexity')}")
        print(f"  Certainty: {thought.get('certainty')}")
    else:
        print(f"  Error: {thought_result.get('error')}")
    
    # Test thought reflection
    print("\nTesting thought reflection...")
    reflection_result = lmm.thought_module.process({
        "operation": "reflect",
        "content": "I'm curious about how thoughts connect to form new ideas",
        "developmental_stage": DevelopmentalStage.MIDDLE_CHILDHOOD.value
    })
    
    print("\nReflection results:")
    if reflection_result.get("success"):
        print("  Patterns:")
        for pattern in reflection_result.get("patterns", []):
            print(f"    - {pattern}")
        
        print("  Insights:")
        for insight in reflection_result.get("insights", []):
            print(f"    - {insight}")
        
        print("  Meta-thoughts:")
        for meta in reflection_result.get("meta_thoughts", []):
            print(f"    - {meta}")
    else:
        print(f"  Error: {reflection_result.get('error')}")
    
    # Test association between thoughts
    print("\nTesting thought association...")
    # First, generate two thoughts to associate
    thought1 = lmm.thought_module.process({
        "operation": "generate_thought",
        "content": "Learning requires making connections between ideas",
        "developmental_stage": DevelopmentalStage.MIDDLE_CHILDHOOD.value
    })
    
    thought2 = lmm.thought_module.process({
        "operation": "generate_thought",
        "content": "Memory systems help organize information for retrieval",
        "developmental_stage": DevelopmentalStage.MIDDLE_CHILDHOOD.value
    })
    
    # Associate the thoughts
    if thought1.get("success") and thought2.get("success"):
        thought1_id = thought1.get("thought", {}).get("id")
        thought2_id = thought2.get("thought", {}).get("id")
        
        association_result = lmm.thought_module.process({
            "operation": "associate_thoughts",
            "thought_ids": [thought1_id, thought2_id],
            "developmental_stage": DevelopmentalStage.MIDDLE_CHILDHOOD.value
        })
        
        print(f"  Association created: {association_result.get('success')}")
        if association_result.get('success'):
            print(f"  Associations created: {association_result.get('associations_created')}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_thought_generation() 