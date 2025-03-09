"""
Test for the Emotion Module

This script tests the functionality of the Emotion module at different
developmental levels, examining how emotional responses, sentiment analysis,
and emotion regulation capabilities mature over time.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lmm_project.modules.emotion import get_module
from lmm_project.modules.emotion.models import EmotionState

# Helper functions
def print_section(title):
    """Print a section title with decorative formatting"""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"| {title} |")
    print(f"{border}\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """
    Recursively print a dictionary with indentation
    
    Args:
        data: Dictionary to print
        indent: Current indentation level
        max_depth: Maximum depth to recurse
        current_depth: Current recursion depth
    """
    prefix = "  " * indent
    
    # Stop recursing if we hit max depth
    if current_depth >= max_depth:
        print(f"{prefix}{data}")
        return
        
    # Print each key-value pair
    for key, value in data.items():
        if isinstance(value, dict) and current_depth < max_depth:
            print(f"{prefix}{key}:")
            print_dict(value, indent + 1, max_depth, current_depth + 1)
        elif isinstance(value, list) and value and current_depth < max_depth:
            if isinstance(value[0], dict):
                print(f"{prefix}{key}: [{len(value)} items]")
                if len(value) > 0:
                    print_dict(value[0], indent + 1, max_depth, current_depth + 1)
            else:
                if len(value) > 5:
                    print(f"{prefix}{key}: {value[:5]} ... ({len(value)} items)")
                else:
                    print(f"{prefix}{key}: {value}")
        else:
            # Truncate long strings
            if isinstance(value, str) and len(value) > 100:
                print(f"{prefix}{key}: {value[:100]}...")
            else:
                print(f"{prefix}{key}: {value}")

class EmotionTester:
    """
    Class for testing emotion module functionality
    """
    
    def __init__(self, development_level: float = 0.0):
        """
        Initialize the emotion tester
        
        Args:
            development_level: Initial developmental level
        """
        self.development_level = development_level
        
        # Initialize the emotion module
        self.emotion_module = get_module(
            module_id="test_emotion",
            event_bus=None,
            development_level=development_level
        )
        
        # History of test results
        self.test_history = []
        
        print(f"Initialized Emotion module at development level {development_level:.2f}")
    
    def generate_emotion(self, valence: float, arousal: float, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an emotional response from valence and arousal
        
        Args:
            valence: Pleasure-displeasure value (-1 to 1)
            arousal: Activation level (0 to 1)
            text: Optional text to include in processing
            
        Returns:
            Emotion processing result
        """
        input_data = {
            "operation": "generate",
            "valence": valence,
            "arousal": arousal,
            "process_id": f"test_{int(time.time())}",
        }
        
        if text:
            input_data["content"] = {"text": text}
            
        # Process input through emotion module
        result = self.emotion_module.process_input(input_data)
        
        # Add test metadata
        test_result = {
            "test_type": "generate_emotion",
            "input": {
                "valence": valence,
                "arousal": arousal,
                "text": text
            },
            "development_level": self.development_level,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.test_history.append(test_result)
        
        return result
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        input_data = {
            "operation": "analyze",
            "content": {"text": text},
            "process_id": f"test_{int(time.time())}"
        }
        
        # Process input through emotion module
        result = self.emotion_module.process_input(input_data)
        
        # Add test metadata
        test_result = {
            "test_type": "analyze_sentiment",
            "input": {
                "text": text
            },
            "development_level": self.development_level,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.test_history.append(test_result)
        
        return result
    
    def regulate_emotion(
        self, 
        target_valence: Optional[float] = None,
        target_arousal: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test emotion regulation
        
        Args:
            target_valence: Target valence value (-1 to 1)
            target_arousal: Target arousal value (0 to 1)
            strategy: Optional specific regulation strategy
            
        Returns:
            Regulation result
        """
        # Create current state for regulation
        # Using slightly negative valence and elevated arousal to simulate stress
        current_state = EmotionState(
            valence=-0.4,
            arousal=0.7,
            dominant_emotion="stress",
            emotion_intensities={
                "stress": 0.6,
                "anxiety": 0.3,
                "neutral": 0.1
            },
            timestamp=datetime.now()
        )
        
        # The issue is in how we structure the regulation input - fixing it here
        regulation_input = {
            "current_state": current_state,
            "process_id": f"test_{int(time.time())}"
        }
        
        # Add targets to regulation input with the correct parameter names
        if target_valence is not None:
            regulation_input["target_valence"] = target_valence
            
        if target_arousal is not None:
            regulation_input["target_arousal"] = target_arousal
            
        if strategy:
            regulation_input["regulation_strategy"] = strategy
        
        # Create the overall operation input
        input_data = {
            "operation": "regulate",
            **regulation_input
        }
        
        # Process regulation request
        result = self.emotion_module.process_input(input_data)
        
        # Add test metadata
        test_result = {
            "test_type": "regulate_emotion",
            "input": {
                "current_state": current_state.dict(),
                "target_valence": target_valence,
                "target_arousal": target_arousal,
                "strategy": strategy
            },
            "development_level": self.development_level,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.test_history.append(test_result)
        
        return result
    
    def query_emotion_state(self) -> Dict[str, Any]:
        """
        Query the current emotional state
        
        Returns:
            Emotion state information
        """
        input_data = {
            "operation": "query",
            "process_id": f"test_{int(time.time())}"
        }
        
        # Process query
        result = self.emotion_module.process_input(input_data)
        
        # Add test metadata
        test_result = {
            "test_type": "query_emotion",
            "development_level": self.development_level,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.test_history.append(test_result)
        
        return result
    
    def print_result_summary(self, result: Dict[str, Any], result_type: str = "emotion"):
        """
        Print a summary of the processing result
        
        Args:
            result: Result to summarize
            result_type: Type of result (emotion, sentiment, regulation)
        """
        if result_type == "emotion":
            if "response" in result:
                response = result["response"]
                print(f"Dominant Emotion: {response['dominant_emotion']}")
                print(f"Valence: {response['valence']:.2f}, Arousal: {response['arousal']:.2f}")
                print("Emotion Intensities:")
                intensities = sorted(
                    response['emotion_intensities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                for emotion, intensity in intensities[:5]:  # Show top 5
                    print(f"  - {emotion}: {intensity:.2f}")
                    
        elif result_type == "sentiment":
            if "analysis" in result:
                analysis = result["analysis"]
                if "analysis" in analysis:  # For nested analysis structure
                    analysis = analysis["analysis"]
                
                # Extract scores
                if "compound_score" in analysis:
                    print(f"Sentiment Score: {analysis['compound_score']:.2f}")
                    
                if "positive_score" in analysis and "negative_score" in analysis:
                    print(f"Positive: {analysis['positive_score']:.2f}, Negative: {analysis['negative_score']:.2f}")
                    
                if "detected_emotions" in analysis:
                    print("Detected Emotions:")
                    emotions = sorted(
                        analysis['detected_emotions'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for emotion, score in emotions[:3]:  # Show top 3
                        print(f"  - {emotion}: {score:.2f}")
                        
                if "highlighted_phrases" in analysis and analysis["highlighted_phrases"]:
                    print("Key Phrases:")
                    for phrase in analysis["highlighted_phrases"][:2]:  # Show top 2
                        text = phrase.get("text", "")
                        score = phrase.get("score", 0)
                        if text:
                            print(f"  - '{text[:40]}...' ({score:.2f})")
                
        elif result_type == "regulation":
            # Updated to handle simplified regulation result structure
            # Check if we have the necessary keys directly in the result
            if "regulation_strategy" in result or "original_state" in result:
                # Direct regulation result structure
                reg_result = result
            elif "regulation_result" in result:
                # Nested regulation result structure (for backward compatibility)
                reg_result = result["regulation_result"]
            else:
                reg_result = {}
                
            print(f"Regulation Strategy: {reg_result.get('regulation_strategy', 'unknown')}")
            print(f"Success Level: {reg_result.get('success_level', 0):.2f}")
            
            # Handle both dict and EmotionState objects
            if "original_state" in reg_result and "regulated_state" in reg_result:
                orig = reg_result["original_state"]
                reg = reg_result["regulated_state"]
                
                # Handle both dict and EmotionState objects
                if hasattr(orig, 'valence'):
                    # Original is an EmotionState object
                    orig_valence = orig.valence
                    orig_arousal = orig.arousal
                    orig_emotion = orig.dominant_emotion
                else:
                    # Original is a dict
                    orig_valence = orig.get('valence', 0)
                    orig_arousal = orig.get('arousal', 0)
                    orig_emotion = orig.get('dominant_emotion', 'unknown')
                
                if hasattr(reg, 'valence'):
                    # Regulated is an EmotionState object
                    reg_valence = reg.valence
                    reg_arousal = reg.arousal
                    reg_emotion = reg.dominant_emotion
                else:
                    # Regulated is a dict
                    reg_valence = reg.get('valence', 0)
                    reg_arousal = reg.get('arousal', 0)
                    reg_emotion = reg.get('dominant_emotion', 'unknown')
                
                print(f"Valence: {orig_valence:.2f} → {reg_valence:.2f}")
                print(f"Arousal: {orig_arousal:.2f} → {reg_arousal:.2f}")
                print(f"Emotion: {orig_emotion} → {reg_emotion}")
        
        elif result_type == "query":
            if "current_state" in result:
                state = result["current_state"]
                print(f"Current Emotion: {state['dominant_emotion']}")
                print(f"Valence: {state['valence']:.2f}, Arousal: {state['arousal']:.2f}")
                
            if "emotional_capacity" in result:
                capacity = result["emotional_capacity"]
                print(f"Emotional Capacity:")
                print(f"  Complexity: {capacity.get('emotional_complexity', 'unknown')}")
                print(f"  Regulation: {capacity.get('regulation_capacity', 0):.2f}")
                print(f"  Self-Awareness: {capacity.get('self_awareness', 'none')}")
                available = capacity.get("available_emotions", [])
                if available:
                    if len(available) > 5:
                        print(f"  Available Emotions: {len(available)} emotions")
                    else:
                        print(f"  Available Emotions: {', '.join(available)}")
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Print detailed information about a result"""
        print("\nDetailed Result:")
        print_dict(result, indent=1, max_depth=5)
    
    def print_module_state(self):
        """Print the current state of the emotion module"""
        state = self.emotion_module.get_state()
        print("\nEmotion Module State:")
        print(f"Module ID: {state.get('module_id', 'unknown')}")
        print(f"Module Type: {state.get('module_type', 'unknown')}")
        print(f"Development Level: {state.get('development_level', 0):.2f}")
        
        # Print development milestones
        if "development_level" in state:
            level = state["development_level"]
            for milestone_level, description in sorted(self.emotion_module.development_milestones.items()):
                reached = "✓" if level >= milestone_level else "✗"
                print(f"  {reached} {milestone_level:.1f}: {description}")
                
        # Print emotional capacity if available
        if hasattr(self.emotion_module, "_get_emotional_capacity"):
            capacity = self.emotion_module._get_emotional_capacity()
            print("\nEmotional Capacity:")
            print_dict(capacity, indent=1)
    
    def set_development_level(self, level: float):
        """Update the development level of the emotion module"""
        # Update development level
        self.development_level = level
        self.emotion_module.update_development(level - self.emotion_module.development_level)
        print(f"Updated development level to {level:.2f}")
        
        # Get current state after development
        return self.emotion_module.get_state()
    
    def save_results(self, filename: str = None):
        """Save test results to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_test_results_{timestamp}.json"
            
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Construct file path
        file_path = os.path.join(results_dir, filename)
        
        # Save results
        with open(file_path, 'w') as f:
            # Convert any non-serializable data
            sanitized_history = []
            for item in self.test_history:
                # Apply any necessary conversions for JSON serialization
                sanitized_item = json.loads(json.dumps(item, default=str))
                sanitized_history.append(sanitized_item)
                
            json.dump(sanitized_history, f, indent=2)
            
        print(f"Saved test results to {file_path}")
        
        return file_path

def test_emotion_generation(tester: EmotionTester):
    """
    Test emotional response generation from different
    valence-arousal combinations
    """
    print_section("Emotion Generation Test")
    tester.print_module_state()
    
    # Test various valence-arousal combinations
    test_points = [
        (0.8, 0.6, "Excited/Happy"),   # High valence, moderate-high arousal
        (-0.7, 0.2, "Sad"),            # Low valence, low arousal
        (-0.6, 0.8, "Angry/Afraid"),   # Low valence, high arousal
        (0.7, 0.2, "Content/Relaxed"), # High valence, low arousal
        (0.1, 0.8, "Surprised"),       # Neutral valence, high arousal
        (0.0, 0.2, "Neutral")          # Neutral valence, low arousal
    ]
    
    for valence, arousal, label in test_points:
        print(f"\nTesting {label} (Valence: {valence:.2f}, Arousal: {arousal:.2f})")
        result = tester.generate_emotion(valence, arousal)
        tester.print_result_summary(result, result_type="emotion")
    
    # Test with added text
    print("\nTesting emotion with text influence:")
    text = "I'm feeling really excited about this new project!"
    result = tester.generate_emotion(0.5, 0.5, text)
    tester.print_result_summary(result, result_type="emotion")
    
    return tester

def test_sentiment_analysis(tester: EmotionTester):
    """Test sentiment analysis capabilities"""
    print_section("Sentiment Analysis Test")
    
    # Test texts with different emotional content
    test_texts = [
        "I'm having a wonderful day and everything is going great!",
        "This is absolutely terrible and I'm very upset about it.",
        "I'm a bit nervous about the upcoming presentation tomorrow.",
        "The weather is quite nice today, not too hot or cold.",
        "I'm both excited and anxious about this new opportunity."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nAnalyzing Text {i+1}: '{text[:50]}...'")
        result = tester.analyze_sentiment(text)
        tester.print_result_summary(result, result_type="sentiment")
    
    return tester

def test_emotion_regulation(tester: EmotionTester):
    """Test emotion regulation capabilities"""
    print_section("Emotion Regulation Test")
    
    # Test current emotional capacity
    print("\nEmotional Regulation Capacity:")
    result = tester.query_emotion_state()
    tester.print_result_summary(result, result_type="query")
    
    # Test regulation with different targets
    print("\nRegulating toward positive valence:")
    result = tester.regulate_emotion(target_valence=0.5)
    # Debug regulation result structure
    print(f"Regulation Result Keys: {list(result.keys())}")
    if "regulation_result" in result:
        print(f"Regulation Sub-Result Keys: {list(result['regulation_result'].keys())}")
    tester.print_result_summary(result, result_type="regulation")
    
    print("\nRegulating toward lower arousal:")
    result = tester.regulate_emotion(target_arousal=0.3)
    tester.print_result_summary(result, result_type="regulation")
    
    print("\nRegulating toward both positive valence and lower arousal:")
    result = tester.regulate_emotion(target_valence=0.5, target_arousal=0.3)
    tester.print_result_summary(result, result_type="regulation")
    
    # Test available strategies if at developed level
    if tester.development_level >= 0.4:
        print("\nTesting specific regulation strategies:")
        # Access the emotion regulator's state directly to get available strategies
        regulation_state = tester.emotion_module.emotion_regulator.get_state()
        available_strategies = regulation_state.get("available_strategies", [])
        print(f"Available strategies: {available_strategies}")
        
        for strategy in available_strategies[:3]:  # Test up to 3 strategies
            print(f"\nRegulating using {strategy}:")
            result = tester.regulate_emotion(
                target_valence=0.3, 
                target_arousal=0.4,
                strategy=strategy
            )
            tester.print_result_summary(result, result_type="regulation")
    
    return tester

def test_emotion_at_level(level: float) -> EmotionTester:
    """Test emotion module at a specific development level"""
    print_section(f"Testing Emotion Module at Level {level:.1f}")
    
    # Initialize tester
    tester = EmotionTester(development_level=level)
    
    # Run the tests
    test_emotion_generation(tester)
    test_sentiment_analysis(tester)
    test_emotion_regulation(tester)
    
    # Save results
    tester.save_results(f"emotion_level_{level:.1f}.json")
    
    return tester

def test_development_progression() -> EmotionTester:
    """Test how emotion capabilities evolve across development levels"""
    print_section("Testing Emotion Development Progression")
    
    # Initialize at the lowest level
    tester = EmotionTester(development_level=0.0)
    
    # Define development stages to test
    stages = [0.0, 0.3, 0.6, 0.9]
    
    # Test each stage with all tests
    for stage in stages:
        # Set the development level
        tester.set_development_level(stage)
        
        print_section(f"Development Level: {stage:.1f}")
        tester.print_module_state()
        
        # Run standard test battery with minimal output
        print("\nTesting Emotion Generation")
        result = tester.generate_emotion(0.7, 0.6)
        tester.print_result_summary(result, result_type="emotion")
        
        print("\nTesting Sentiment Analysis")
        result = tester.analyze_sentiment("I'm really excited about this new project!")
        tester.print_result_summary(result, result_type="sentiment")
        
        print("\nTesting Emotion Regulation")
        result = tester.regulate_emotion(target_valence=0.5, target_arousal=0.4)
        tester.print_result_summary(result, result_type="regulation")
    
    # Save results
    tester.save_results("emotion_development_progression.json")
    
    return tester

def main():
    """Main test function"""
    print_section("Emotion Module Test")
    
    # Test developmental progression
    test_development_progression()
    
    # Test specific development levels in detail
    test_emotion_at_level(0.9)  # Test at a high development level
    
    print_section("Testing Complete")

if __name__ == "__main__":
    main() 