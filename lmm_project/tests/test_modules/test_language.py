"""
Comprehensive test script for the language module.

This script tests how the language module and its components (phoneme recognition,
word learning, grammar acquisition, semantic processing, and expression generation)
function at different developmental stages.
"""

import logging
import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import random
import torch

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import language module components
from lmm_project.modules.language import LanguageSystem, get_module
from lmm_project.modules.language.phoneme_recognition import PhonemeRecognition
from lmm_project.modules.language.word_learning import WordLearning
from lmm_project.modules.language.grammar_acquisition import GrammarAcquisition
from lmm_project.modules.language.semantic_processing import SemanticProcessing
from lmm_project.modules.language.expression_generator import ExpressionGenerator
from lmm_project.modules.language.models import (
    LanguageModel, PhonemeModel, WordModel, GrammarModel, 
    SemanticModel, ExpressionModel, LanguageNeuralState
)
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.base_module import BaseModule
from lmm_project.core.message import Message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("language_test")

def print_section(title):
    """Print a section header for better readability"""
    width = 80
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """Recursively print a dictionary with proper indentation"""
    if current_depth > max_depth:
        print(" " * indent + "...")
        return
    
    if not isinstance(data, dict):
        print(" " * indent + str(data))
        return
    
    for key, value in data.items():
        if isinstance(value, dict) and value:
            print(" " * indent + str(key) + ":")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(" " * indent + str(key) + ":")
            for i, item in enumerate(value[:3]):  # Print only first 3 items
                print(" " * (indent + 2) + f"[{i}]:")
                print_dict(item, indent + 4, max_depth, current_depth + 1)
            if len(value) > 3:
                print(" " * (indent + 4) + f"... ({len(value) - 3} more items)")
        else:
            # Truncate long values
            if isinstance(value, str) and len(value) > 50:
                print(" " * indent + str(key) + ": " + value[:50] + "...")
            elif isinstance(value, list) and len(value) > 5:
                print(" " * indent + str(key) + f": [{', '.join(str(x) for x in value[:5])}... ({len(value)} items)]")
            else:
                print(" " * indent + str(key) + ": " + str(value))

class LanguageTester:
    """
    Test harness for the language module and its components.
    
    This class provides methods to test various aspects of language processing
    at different developmental levels.
    """
    
    def __init__(self, development_level: float = 0.0):
        """
        Initialize the language tester with a specific development level.
        
        Args:
            development_level: float between 0.0 and 1.0 representing the 
                              developmental stage of the language system
        """
        self.event_bus = EventBus()
        self.language_system = LanguageSystem("language", self.event_bus, development_level)
        
        # Get individual components for granular testing
        self.phoneme_recognition = PhonemeRecognition("phoneme_recognition", self.event_bus, development_level)
        self.word_learning = WordLearning("word_learning", self.event_bus, development_level)
        self.grammar_acquisition = GrammarAcquisition("grammar_acquisition", self.event_bus, development_level)
        self.semantic_processing = SemanticProcessing("semantic_processing", self.event_bus, development_level)
        self.expression_generator = ExpressionGenerator("expression_generator", self.event_bus, development_level)
        
        self.development_level = development_level
        self.results = []
        logger.info(f"Initialized language tester at development level {development_level:.2f}")
        
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input through the language system.
        
        Args:
            text: The text to process
            context: Optional context information
            
        Returns:
            Results of language processing
        """
        if context is None:
            context = {}
            
        input_data = {
            "text": text,
            "context": context,
            "operation": "comprehend",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.language_system.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "input_text": text,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def generate_expression(self, meaning: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate language expression from meaning.
        
        Args:
            meaning: The meaning to express
            context: Optional context information
            
        Returns:
            Results of expression generation
        """
        if context is None:
            context = {}
            
        input_data = {
            "meaning": meaning,
            "context": context,
            "operation": "generate_expression",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.expression_generator.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "input_meaning": meaning,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def learn_word(self, word: str, meaning: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test word learning by introducing a new word with meaning.
        
        Args:
            word: The word to learn
            meaning: The meaning of the word
            context: Optional context information
            
        Returns:
            Results of word learning
        """
        if context is None:
            context = {}
            
        input_data = {
            "word": word,
            "meaning": meaning,
            "context": context,
            "operation": "learn_word",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.word_learning.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "word": word,
            "meaning": meaning,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def analyze_grammar(self, sentence: str) -> Dict[str, Any]:
        """
        Test grammar analysis on a sentence.
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            Results of grammar analysis
        """
        input_data = {
            "text": sentence,
            "operation": "analyze_grammar",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.grammar_acquisition.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "sentence": sentence,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def understand_meaning(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test semantic understanding of text.
        
        Args:
            text: The text to understand
            context: Optional context information
            
        Returns:
            Results of semantic processing
        """
        if context is None:
            context = {}
            
        input_data = {
            "text": text,
            "context": context,
            "operation": "understand_meaning",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.semantic_processing.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "input_text": text,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def recognize_phonemes(self, audio_text_representation: str) -> Dict[str, Any]:
        """
        Test phoneme recognition from text representation of audio.
        
        Args:
            audio_text_representation: Text representing audio input
            
        Returns:
            Results of phoneme recognition
        """
        input_data = {
            "audio_text": audio_text_representation,
            "operation": "recognize_phonemes",
            "process_id": f"test_{int(time.time())}"
        }
        
        start_time = time.time()
        result = self.phoneme_recognition.process_input(input_data)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result["metadata"] = {
            "audio_text": audio_text_representation,
            "development_level": self.development_level,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store result for later analysis
        self.results.append(result)
        
        return result
    
    def test_integrated_language(self, inputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Test the integrated language system with a sequence of inputs.
        
        Args:
            inputs: List of input data dictionaries with various operations
            
        Returns:
            Dictionary of results from different operations
        """
        results = {
            "comprehend": [],
            "produce": [],
            "learn": [],
            "analyze": []
        }
        
        for input_data in inputs:
            operation = input_data.get("operation", "comprehend")
            
            if operation == "comprehend" and "text" in input_data:
                result = self.process_text(input_data["text"], input_data.get("context"))
                results["comprehend"].append(result)
                
            elif operation == "produce" and "meaning" in input_data:
                result = self.generate_expression(input_data["meaning"], input_data.get("context"))
                results["produce"].append(result)
                
            elif operation == "learn":
                if "word" in input_data and "meaning" in input_data:
                    result = self.learn_word(input_data["word"], input_data["meaning"], input_data.get("context"))
                    results["learn"].append(result)
                    
            elif operation == "analyze" and "text" in input_data:
                result = self.analyze_grammar(input_data["text"])
                results["analyze"].append(result)
        
        return results
    
    def print_result_summary(self, result: Dict[str, Any], result_type: str = "language"):
        """
        Print a summary of test results.
        
        Args:
            result: The result dictionary to summarize
            result_type: Type of result for display purposes
        """
        print_section(f"Language {result_type.title()} Result Summary")
        
        if "metadata" in result:
            print(f"Input: {result['metadata'].get('input_text', result['metadata'].get('input_meaning', 'N/A'))}")
            print(f"Development Level: {result['metadata']['development_level']:.2f}")
            print(f"Processing Time: {result['metadata']['processing_time']:.4f} seconds")
            print()
        
        # Print type-specific summaries
        if result_type == "comprehension":
            if "understanding" in result:
                print(f"Understanding Level: {result['understanding'].get('comprehension_level', 0):.2f}")
                print(f"Concepts Identified: {len(result['understanding'].get('concepts', []))}")
                
            if "phonemes" in result:
                print(f"Phonemes Recognized: {len(result['phonemes'].get('recognized', []))}")
                
            if "words" in result:
                print(f"Words Recognized: {len(result['words'].get('recognized', []))}")
                
            if "grammar" in result:
                print(f"Grammar Structures: {len(result['grammar'].get('structures', []))}")
                
        elif result_type == "expression":
            if "expression" in result:
                print(f"Expression: {result['expression'].get('text', 'N/A')}")
                print(f"Fluency: {result['expression'].get('fluency', 0):.2f}")
                print(f"Complexity: {result['expression'].get('complexity', 0):.2f}")
                
        elif result_type == "learning":
            if "learning" in result:
                print(f"Word: {result.get('word', 'N/A')}")
                print(f"Meaning: {result.get('meaning', 'N/A')}")
                print(f"Learning Success: {result['learning'].get('success', False)}")
                print(f"Confidence: {result['learning'].get('confidence', 0):.2f}")
                
        # Print general status and errors
        print(f"\nStatus: {result.get('status', 'unknown')}")
        if "errors" in result and result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error}")
        
        print("\nDetails:")
        # Filter out metadata and other large sections for summary
        summary_result = {k: v for k, v in result.items() if k not in ["metadata"]}
        print_dict(summary_result, indent=2, max_depth=2)
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Print detailed test result information"""
        print_section("Detailed Result")
        print_dict(result)
    
    def print_module_state(self):
        """Print the current state of language modules"""
        print_section("Language Module State")
        
        # Get state from each component
        states = {
            "language_system": self.language_system.get_state(),
            "phoneme_recognition": self.phoneme_recognition.get_state(),
            "word_learning": self.word_learning.get_state(),
            "grammar_acquisition": self.grammar_acquisition.get_state(),
            "semantic_processing": self.semantic_processing.get_state(),
            "expression_generator": self.expression_generator.get_state()
        }
        
        # Print development levels
        print("Development Levels:")
        print(f"  Overall: {states['language_system']['developmental_level']:.2f}")
        print(f"  Phoneme Recognition: {states['phoneme_recognition']['developmental_level']:.2f}")
        print(f"  Word Learning: {states['word_learning']['developmental_level']:.2f}")
        print(f"  Grammar Acquisition: {states['grammar_acquisition']['developmental_level']:.2f}")
        print(f"  Semantic Processing: {states['semantic_processing']['developmental_level']:.2f}")
        print(f"  Expression Generation: {states['expression_generator']['developmental_level']:.2f}")
        
        # Print vocabulary statistics
        word_count = len(states['word_learning'].get('vocabulary', {}).get('vocabulary', {}))
        print(f"\nVocabulary Size: {word_count} words")
        
        # Print grammar rule statistics
        grammar_rules = len(states['grammar_acquisition'].get('grammar', {}).get('grammatical_structures', []))
        print(f"Grammar Rules: {grammar_rules} rules")
        
        # Print concept statistics
        concept_count = len(states['semantic_processing'].get('semantics', {}).get('concept_network', {}))
        print(f"Semantic Concepts: {concept_count} concepts")
        
        # Print expression template statistics
        expression_templates = len(states['expression_generator'].get('expression', {}).get('expression_templates', []))
        print(f"Expression Templates: {expression_templates} templates")
    
    def set_development_level(self, level: float):
        """
        Set the development level for all language components.
        
        Args:
            level: New development level (0.0 to 1.0)
        """
        self.development_level = level
        self.language_system.update_development(level - self.language_system.development_level)
        self.phoneme_recognition.update_development(level - self.phoneme_recognition.development_level)
        self.word_learning.update_development(level - self.word_learning.development_level)
        self.grammar_acquisition.update_development(level - self.grammar_acquisition.development_level)
        self.semantic_processing.update_development(level - self.semantic_processing.development_level)
        self.expression_generator.update_development(level - self.expression_generator.development_level)
        
        logger.info(f"Updated development level to {level:.2f}")
        
    def save_results(self, filename: str = None):
        """
        Save test results to a file.
        
        Args:
            filename: Optional filename, or generate based on timestamp
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"language_test_results_{timestamp}.json"
        
        # Custom JSON encoder to handle non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
        
        results_data = {
            "development_level": self.development_level,
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "module_states": {
                "language_system": self.language_system.get_state(),
                "phoneme_recognition": self.phoneme_recognition.get_state(),
                "word_learning": self.word_learning.get_state(),
                "grammar_acquisition": self.grammar_acquisition.get_state(),
                "semantic_processing": self.semantic_processing.get_state(),
                "expression_generator": self.expression_generator.get_state()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, cls=CustomEncoder)
            
        logger.info(f"Saved test results to {filename}")
        return filename

def create_language_test_scenario():
    """
    Create a test scenario for language processing at different developmental stages.
    
    Returns:
        A list of test inputs for the language system
    """
    return [
        # Simple vocabulary learning
        {
            "operation": "learn",
            "word": "ball",
            "meaning": "A round object used in games and sports"
        },
        {
            "operation": "learn",
            "word": "dog",
            "meaning": "A domesticated carnivorous mammal"
        },
        {
            "operation": "learn",
            "word": "run",
            "meaning": "Move at a speed faster than walking"
        },
        {
            "operation": "learn",
            "word": "big",
            "meaning": "Of considerable size or extent"
        },
        
        # Simple comprehension
        {
            "operation": "comprehend",
            "text": "The dog is big."
        },
        {
            "operation": "comprehend",
            "text": "The ball is round."
        },
        
        # Grammar analysis
        {
            "operation": "analyze",
            "text": "The big dog runs with the ball."
        },
        {
            "operation": "analyze",
            "text": "Dogs like to play with balls."
        },
        
        # Expression generation
        {
            "operation": "produce",
            "meaning": "A dog playing with a ball"
        },
        {
            "operation": "produce",
            "meaning": "Dogs are animals that like to run"
        },
        
        # More complex learning
        {
            "operation": "learn",
            "word": "happiness",
            "meaning": "A state of well-being and contentment"
        },
        {
            "operation": "learn",
            "word": "quantum",
            "meaning": "The smallest discrete unit of a phenomenon"
        },
        
        # Complex comprehension
        {
            "operation": "comprehend",
            "text": "The quantum nature of reality challenges our intuition about how the world works."
        },
        {
            "operation": "comprehend",
            "text": "Finding happiness often involves pursuing meaningful goals and maintaining positive relationships."
        },
        
        # Complex grammar
        {
            "operation": "analyze",
            "text": "Although it had been raining for hours, the determined hikers continued their journey through the mountains."
        },
        
        # Complex expression
        {
            "operation": "produce",
            "meaning": "The relationship between quantum physics and consciousness remains a fascinating mystery"
        }
    ]

def test_language_at_level(level: float) -> LanguageTester:
    """
    Test the language module at a specific developmental level.
    
    Args:
        level: Development level to test (0.0 to 1.0)
        
    Returns:
        The language tester instance with results
    """
    print_section(f"Testing Language System at Development Level {level:.2f}")
    
    # Create language tester at specified level
    tester = LanguageTester(development_level=level)
    
    # Create test scenario
    scenario = create_language_test_scenario()
    
    # Run integrated test
    results = tester.test_integrated_language(scenario)
    
    # Print summaries of results
    if results["comprehend"]:
        tester.print_result_summary(results["comprehend"][0], "comprehension")
    
    if results["produce"]:
        tester.print_result_summary(results["produce"][0], "expression")
    
    if results["learn"]:
        tester.print_result_summary(results["learn"][0], "learning")
    
    if results["analyze"]:
        tester.print_result_summary(results["analyze"][0], "grammar")
    
    # Print module state
    tester.print_module_state()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tester.save_results(f"language_test_level_{level:.2f}_{timestamp}.json")
    
    return tester

def test_development_progression() -> LanguageTester:
    """
    Test language development progression across multiple levels.
    
    Returns:
        The language tester after progression through all levels
    """
    print_section("Testing Language Development Progression")
    
    # Define development levels to test
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Start with lowest level
    tester = LanguageTester(development_level=levels[0])
    
    # Create test scenario
    scenario = create_language_test_scenario()
    
    # Test at each level
    for level in levels:
        print_section(f"Development Level: {level:.2f}")
        
        # Set development level
        tester.set_development_level(level)
        
        # Run integrated test
        results = tester.test_integrated_language(scenario)
        
        # Print summaries
        print(f"\nLanguage capabilities at development level {level:.2f}:\n")
        
        # Print module state
        tester.print_module_state()
        
        # Show example results from each category
        examples = {
            "Simple comprehension": "The dog is big.",
            "Complex comprehension": "The quantum nature of reality challenges our intuition about how the world works.",
            "Simple expression": "A dog playing with a ball",
            "Complex expression": "The relationship between quantum physics and consciousness remains a fascinating mystery"
        }
        
        print("\nExample Results:")
        for desc, text in examples.items():
            print(f"\n{desc}:")
            if "comprehension" in desc.lower():
                result = tester.process_text(text)
                print(f"  Understanding level: {result.get('understanding', {}).get('comprehension_level', 0):.2f}")
                concepts = result.get('understanding', {}).get('concepts', [])
                print(f"  Concepts identified: {len(concepts)}")
                if concepts:
                    print(f"  Sample concepts: {', '.join(str(c) for c in concepts[:3])}")
            else:
                result = tester.generate_expression(text)
                print(f"  Expression: {result.get('expression', {}).get('text', 'N/A')}")
                print(f"  Fluency: {result.get('expression', {}).get('fluency', 0):.2f}")
                print(f"  Complexity: {result.get('expression', {}).get('complexity', 0):.2f}")
        
        # Add a pause between levels
        if level < levels[-1]:
            print("\nProgressing to next development level...\n")
            time.sleep(1)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tester.save_results(f"language_development_progression_{timestamp}.json")
    
    return tester

def main():
    """Run the language module tests"""
    print_section("Language Module Test Suite")
    
    print("1. Test at a specific development level")
    print("2. Test developmental progression")
    print("3. Run all tests")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        level_str = input("Enter development level (0.0-1.0): ")
        try:
            level = float(level_str)
            if 0.0 <= level <= 1.0:
                test_language_at_level(level)
            else:
                print("Error: Level must be between 0.0 and 1.0")
        except ValueError:
            print("Error: Invalid input. Please enter a number between 0.0 and 1.0")
    
    elif choice == "2":
        test_development_progression()
    
    elif choice == "3":
        print("\nRunning test at mid-development level (0.5)...")
        test_language_at_level(0.5)
        
        print("\nRunning developmental progression tests...")
        test_development_progression()
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main() 