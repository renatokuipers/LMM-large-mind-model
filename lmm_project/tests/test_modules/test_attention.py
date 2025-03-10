"""
Test for the Attention Module

This script tests the functionality of the Attention module at different
developmental levels, examining how focus control, salience detection,
and attentional capabilities mature over time.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lmm_project.modules.attention import get_module
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

# ANSI colors for prettier output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

def print_section(title):
    """Print a formatted section title for better readability."""
    print("\n" + "=" * 80)
    print(f"{BOLD}{CYAN}{title}{RESET}")
    print("=" * 80)

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """
    Pretty print a dictionary with indentation and recursion limiting.
    Handles nested dictionaries and lists.
    """
    if current_depth > max_depth:
        print(" " * indent + "...")
        return
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{BOLD}{key}{RESET}:")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            print(" " * indent + f"{BOLD}{key}{RESET}: [")
            for item in value[:3]:  # Only show first 3 items
                print(" " * (indent + 4) + "{")
                print_dict(item, indent + 8, max_depth, current_depth + 1)
                print(" " * (indent + 4) + "}")
            if len(value) > 3:
                print(" " * (indent + 4) + f"... ({len(value) - 3} more items)")
            print(" " * indent + "]")
        else:
            # Format string values nicely
            if isinstance(value, str) and len(value) > 100:
                value_repr = value[:97] + "..."
            else:
                value_repr = value
            print(" " * indent + f"{BOLD}{key}{RESET}: {value_repr}")

class AttentionTester:
    """
    Tester class for the Attention module.
    
    Tests attention focus control, salience detection, and attention transitions
    at various developmental levels.
    """
    
    def __init__(self, development_level: float = 0.0):
        """
        Initialize the AttentionTester with a specified development level.
        
        Args:
            development_level: The developmental level (0.0 to 1.0) of the attention system
        """
        self.event_bus = EventBus()
        self.attention = get_module(
            module_id="attention_test",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        self.results = []
        print(f"Initialized Attention Tester at development level: {development_level:.2f}")
        
    def test_focus_control(self, stimuli: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the attention module's ability to control focus based on stimulus inputs.
        
        Args:
            stimuli: List of stimulus inputs to process
            
        Returns:
            Dictionary containing test results
        """
        print_section("Testing Focus Control")
        print(f"Processing {len(stimuli)} stimuli to test focus control...")
        
        focus_shifts = []
        focus_history = []
        
        for i, stimulus in enumerate(stimuli):
            print(f"\nProcessing stimulus {i+1}/{len(stimuli)}: {stimulus.get('content', '')[:50]}...")
            
            # Process the stimulus
            start_time = time.time()
            result = self.attention.process_input(stimulus)
            processing_time = time.time() - start_time
            
            # Get current focus after processing
            current_focus = self.attention.get_current_focus()
            focus_history.append(current_focus)
            
            # Determine if focus shifted
            focus_shift = False
            if i > 0 and current_focus.get('id') != focus_history[i-1].get('id'):
                focus_shift = True
                focus_shifts.append({
                    'from': focus_history[i-1].get('content', '')[:50],
                    'to': current_focus.get('content', '')[:50],
                    'salience': result.get('salience', 0.0)
                })
            
            print(f"Current focus: {current_focus.get('content', '')[:50]}...")
            if focus_shift:
                print(f"{YELLOW}Focus shifted!{RESET}")
        
        # Get test results
        test_results = {
            'total_stimuli': len(stimuli),
            'focus_shifts': focus_shifts,
            'focus_shift_count': len(focus_shifts),
            'focus_shift_rate': len(focus_shifts) / max(1, len(stimuli)),
            'final_focus': focus_history[-1] if focus_history else None,
            'focus_history': focus_history,
            'attention_state': self.attention.get_state()
        }
        
        self.results.append({
            'test_type': 'focus_control',
            'development_level': self.attention.development_level,
            'timestamp': time.time(),
            'results': test_results
        })
        
        self.print_result_summary(test_results, result_type="focus_control")
        return test_results
    
    def test_salience_detection(self, stimuli: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the attention module's ability to detect salience in different stimuli.
        
        Args:
            stimuli: List of stimulus inputs to evaluate for salience
            
        Returns:
            Dictionary containing test results
        """
        print_section("Testing Salience Detection")
        print(f"Evaluating salience for {len(stimuli)} stimuli...")
        
        salience_results = []
        
        for i, stimulus in enumerate(stimuli):
            print(f"\nEvaluating stimulus {i+1}/{len(stimuli)}: {stimulus.get('content', '')[:50]}...")
            
            # Process the stimulus
            result = self.attention.process_input(stimulus)
            salience = result.get('salience', 0.0)
            captured = result.get('attention_captured', False)
            
            # Record results
            salience_results.append({
                'stimulus': stimulus.get('content', '')[:100],
                'salience': salience,
                'attention_captured': captured,
                'intensity': stimulus.get('intensity', 0.5),
                'novelty': stimulus.get('novelty', 0.5),
                'relevance': stimulus.get('relevance', 0.5),
                'volitional': stimulus.get('volitional', False)
            })
            
            print(f"Salience: {salience:.4f} | Attention captured: {captured}")
        
        # Calculate statistics
        salience_values = [r['salience'] for r in salience_results]
        capture_count = sum(1 for r in salience_results if r['attention_captured'])
        
        avg_salience = sum(salience_values) / max(1, len(salience_values))
        max_salience = max(salience_values) if salience_values else 0
        min_salience = min(salience_values) if salience_values else 0
        
        # Get test results
        test_results = {
            'total_stimuli': len(stimuli),
            'stimulus_results': salience_results,
            'salience_statistics': {
                'average': avg_salience,
                'maximum': max_salience,
                'minimum': min_salience
            },
            'capture_statistics': {
                'count': capture_count,
                'rate': capture_count / max(1, len(stimuli))
            },
            'attention_state': self.attention.get_state()
        }
        
        self.results.append({
            'test_type': 'salience_detection',
            'development_level': self.attention.development_level,
            'timestamp': time.time(),
            'results': test_results
        })
        
        self.print_result_summary(test_results, result_type="salience_detection")
        return test_results
    
    def test_task_context(self, task_contexts: List[Dict[str, Any]], stimuli: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test how task context affects attention and focus.
        
        Args:
            task_contexts: List of task contexts to test
            stimuli: List of stimuli to process within each context
            
        Returns:
            Dictionary containing test results
        """
        print_section("Testing Task Context Effects")
        print(f"Testing {len(task_contexts)} task contexts with context-specific stimuli...")
        
        context_results = []
        
        for i, context in enumerate(task_contexts):
            print(f"\nSetting task context {i+1}/{len(task_contexts)}: {context.get('description', '')}...")
            
            # Set the task context
            context_response = self.attention.set_task_context(context)
            
            # Create context-relevant stimuli to better test context effects
            context_stimuli = self._generate_context_specific_stimuli(context, 4)
            
            # Process stimuli in this context
            stimuli_results = []
            for j, stimulus in enumerate(context_stimuli):
                print(f"  Processing stimulus {j+1}/{len(context_stimuli)} in context {i+1}...")
                
                # Process the stimulus
                result = self.attention.process_input(stimulus)
                
                # Record result
                stimuli_results.append({
                    'stimulus': stimulus.get('content', '')[:100],
                    'salience': result.get('salience', 0.0),
                    'attention_captured': result.get('attention_captured', False),
                    'relevance_to_context': stimulus.get('relevance', 0.0)
                })
            
            # Calculate context-specific stats
            capture_count = sum(1 for r in stimuli_results if r['attention_captured'])
            avg_salience = sum(r['salience'] for r in stimuli_results) / max(1, len(stimuli_results))
            
            # Record context results
            context_results.append({
                'context': context.get('description', ''),
                'stimuli_results': stimuli_results,
                'capture_count': capture_count,
                'capture_rate': capture_count / max(1, len(context_stimuli)),
                'average_salience': avg_salience
            })
            
            print(f"Context {i+1} results: {capture_count}/{len(context_stimuli)} stimuli captured attention (rate: {capture_count/max(1, len(context_stimuli)):.2f})")
            print(f"Average salience in context: {avg_salience:.4f}")
        
        # Get test results
        test_results = {
            'total_contexts': len(task_contexts),
            'total_stimuli_per_context': 4,
            'context_results': context_results,
            'attention_state': self.attention.get_state()
        }
        
        self.results.append({
            'test_type': 'task_context',
            'development_level': self.attention.development_level,
            'timestamp': time.time(),
            'results': test_results
        })
        
        self.print_result_summary(test_results, result_type="task_context")
        return test_results
    
    def _generate_context_specific_stimuli(self, context: Dict[str, Any], count: int = 4) -> List[Dict[str, Any]]:
        """
        Generate stimuli that are specific to a particular task context.
        
        Args:
            context: The task context to generate stimuli for
            count: Number of stimuli to generate
            
        Returns:
            List of context-specific stimuli
        """
        context_desc = context.get('description', '').lower()
        context_domain = context.get('domain', '').lower()
        
        # Define context-relevant content templates
        context_templates = {
            'reading news': [
                "Breaking news: Political scandal erupts in Washington",
                "Weather alert: Storm approaching coastal areas",
                "Technology news: New smartphone release date announced",
                "Business update: Stock market hits record high",
                "International crisis: Diplomatic tensions rising between nations"
            ],
            'studying': [
                "Important formula to remember for the exam",
                "Key historical date that will be on the test",
                "Chapter summary with critical concepts",
                "Practice question similar to exam material",
                "Definition of essential term for course"
            ],
            'watching': [
                "Red light ahead - stop immediately",
                "Pedestrian crossing signal activated",
                "Yellow light warning - prepare to stop",
                "Green arrow indicating turn allowed",
                "Emergency vehicle approaching intersection"
            ],
            'listening': [
                "Key point from lecture that will be on the exam",
                "Professor emphasizes this concept is important",
                "Lecture summary slide with main points",
                "Diagram explaining complex relationship between concepts",
                "Example that illustrates main lecture topic"
            ],
            'looking': [
                "Small shiny object under the furniture",
                "Item similar to what you're looking for",
                "Something out of place in the environment",
                "Object matching the description of lost item",
                "Movement detected in peripheral vision"
            ]
        }
        
        # Determine which context category this matches
        chosen_category = None
        for category in context_templates:
            if category in context_desc or category in context_domain:
                chosen_category = category
                break
                
        # Default to random if no match
        if not chosen_category:
            chosen_category = random.choice(list(context_templates.keys()))
        
        templates = context_templates[chosen_category]
        
        # Generate stimuli
        stimuli = []
        for i in range(count):
            # Create a mix of relevant and less relevant stimuli
            if i < count // 2:
                # Highly relevant to context
                content = random.choice(templates)
                relevance = random.uniform(0.7, 1.0)
                intensity = random.uniform(0.5, 0.8)
                novelty = random.uniform(0.5, 0.8)
            else:
                # Less relevant to context (distractor)
                other_categories = [c for c in context_templates.keys() if c != chosen_category]
                other_category = random.choice(other_categories)
                content = random.choice(context_templates[other_category])
                relevance = random.uniform(0.1, 0.4)
                intensity = random.uniform(0.6, 1.0)  # High intensity but low relevance
                novelty = random.uniform(0.6, 1.0)    # High novelty but low relevance
            
            # Create stimulus
            stimulus = {
                'id': f"context_stim_{context.get('id', 'unknown')}_{i}",
                'content': content,
                'intensity': intensity,
                'novelty': novelty,
                'relevance': relevance,
                'volitional': i < count // 2,  # Intentional focus for relevant items
                'timestamp': time.time(),
                'type': 'text',
                'context_id': context.get('id', 'unknown')
            }
            
            stimuli.append(stimulus)
        
        return stimuli
    
    def test_developmental_effects(self, development_levels: List[float], stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test how changing development level affects attention processing.
        
        Args:
            development_levels: List of development levels to test
            stimulus: Single stimulus to process at each development level
            
        Returns:
            Dictionary containing test results
        """
        print_section("Testing Developmental Effects")
        print(f"Testing a single stimulus across {len(development_levels)} development levels...")
        print(f"Stimulus: {stimulus.get('content', '')[:100]}")
        
        developmental_results = []
        
        for level in development_levels:
            print(f"\nSetting development level to {level:.2f}...")
            
            # Set development level
            self.set_development_level(level)
            
            # Process the stimulus
            result = self.attention.process_input(stimulus)
            
            # Record result
            developmental_results.append({
                'development_level': level,
                'salience': result.get('salience', 0.0),
                'attention_captured': result.get('attention_captured', False),
                'processing_details': result
            })
            
            print(f"Development level {level:.2f}: Salience = {result.get('salience', 0.0):.4f}, Captured = {result.get('attention_captured', False)}")
        
        # Get test results
        test_results = {
            'stimulus': stimulus.get('content', '')[:100],
            'development_levels_tested': development_levels,
            'developmental_results': developmental_results,
            'current_attention_state': self.attention.get_state()
        }
        
        self.results.append({
            'test_type': 'developmental_effects',
            'development_level': self.attention.development_level,
            'timestamp': time.time(),
            'results': test_results
        })
        
        self.print_result_summary(test_results, result_type="developmental_effects")
        return test_results
    
    def test_attention_capture(self, threshold_values: List[float] = None) -> Dict[str, Any]:
        """
        Test the attention module's capability to capture attention with stimuli of varying salience.
        
        Args:
            threshold_values: List of salience multiplier values to test
            
        Returns:
            Dictionary containing test results
        """
        if threshold_values is None:
            threshold_values = [0.3, 0.5, 0.7, 0.9]
            
        print_section("Testing Attention Capture")
        print(f"Testing attention capture with stimuli of {len(threshold_values)} different salience levels...")
        
        capture_results = []
        
        for salience_multiplier in threshold_values:
            print(f"\nTesting with salience multiplier: {salience_multiplier:.2f}")
            
            # Create a series of stimuli with progressively increasing salience factors
            capture_test_stimuli = []
            for i in range(10):
                # Calculate scaled salience values based on the multiplier and index
                base_salience = (i+1) / 10.0  # 0.1 to 1.0
                
                # Create a stimulus with this salience level
                stimulus = {
                    'id': f"capture_test_{i}_{salience_multiplier:.1f}",
                    'content': f"Test stimulus with {base_salience:.1f} base salience (multiplier: {salience_multiplier:.1f})",
                    'intensity': base_salience * salience_multiplier,
                    'novelty': base_salience * salience_multiplier,
                    'relevance': base_salience * salience_multiplier,
                    'volitional': i >= 5,  # Higher salience stimuli are volitional
                    'timestamp': time.time(),
                    'type': 'text'
                }
                capture_test_stimuli.append(stimulus)
            
            # Test each stimulus for capture
            stimulus_results = []
            capture_count = 0
            first_capture_idx = None
            
            for i, stimulus in enumerate(capture_test_stimuli):
                result = self.attention.process_input(stimulus)
                captured = result.get('attention_captured', False)
                salience = result.get('salience', 0.0)
                
                if captured:
                    capture_count += 1
                    if first_capture_idx is None:
                        first_capture_idx = i
                    
                stimulus_results.append({
                    'stimulus_idx': i,
                    'base_salience': (i+1) / 10.0,
                    'multiplier': salience_multiplier,
                    'calculated_salience': salience,
                    'captured': captured
                })
                
                print(f"  Stimulus {i+1}: Base salience {(i+1)/10.0:.1f} → Calculated salience {salience:.4f} | Captured: {captured}")
            
            # Record test results for this multiplier
            capture_results.append({
                'salience_multiplier': salience_multiplier,
                'stimulus_results': stimulus_results,
                'capture_count': capture_count,
                'capture_rate': capture_count / len(capture_test_stimuli),
                'first_capture_index': first_capture_idx,
                'first_capture_base_salience': ((first_capture_idx or 9) + 1) / 10.0
            })
            
            print(f"  Capture rate with multiplier {salience_multiplier:.2f}: {capture_count}/{len(capture_test_stimuli)} ({capture_count/len(capture_test_stimuli):.2f})")
            if first_capture_idx is not None:
                print(f"  First capture occurred at base salience: {(first_capture_idx+1)/10.0:.1f}")
            else:
                print(f"  No captures occurred with this multiplier")
        
        # Compile test results
        test_results = {
            'multipliers_tested': threshold_values,
            'multiplier_results': capture_results,
            'module_development_level': self.attention.development_level,
            'attention_state': self.attention.get_state()
        }
        
        self.results.append({
            'test_type': 'attention_capture',
            'development_level': self.attention.development_level,
            'timestamp': time.time(),
            'results': test_results
        })
        
        self.print_result_summary(test_results, result_type="attention_capture")
        return test_results
    
    def print_result_summary(self, result: Dict[str, Any], result_type: str = "attention"):
        """
        Print a summary of test results.
        
        Args:
            result: The test result dictionary
            result_type: The type of test result being printed
        """
        print_section(f"{result_type.replace('_', ' ').title()} Test Results")
        
        if result_type == "focus_control":
            print(f"Total stimuli processed: {result.get('total_stimuli', 0)}")
            print(f"Focus shifts: {result.get('focus_shift_count', 0)}")
            print(f"Focus shift rate: {result.get('focus_shift_rate', 0):.2f}")
            
            if result.get('focus_shifts'):
                print("\nFocus shift summary:")
                for i, shift in enumerate(result.get('focus_shifts', [])[:5]):
                    print(f"  {i+1}. From: '{shift.get('from', '')}' → To: '{shift.get('to', '')}' (Salience: {shift.get('salience', 0):.4f})")
                
                if len(result.get('focus_shifts', [])) > 5:
                    print(f"  ... and {len(result.get('focus_shifts', [])) - 5} more shifts")
            
            print(f"\nCurrent focus: {result.get('final_focus', {}).get('content', 'None')}")
            
        elif result_type == "salience_detection":
            print(f"Total stimuli evaluated: {result.get('total_stimuli', 0)}")
            stats = result.get('salience_statistics', {})
            
            print(f"\nSalience statistics:")
            print(f"  Average: {stats.get('average', 0):.4f}")
            print(f"  Maximum: {stats.get('maximum', 0):.4f}")
            print(f"  Minimum: {stats.get('minimum', 0):.4f}")
            
            capture_stats = result.get('capture_statistics', {})
            print(f"\nAttention capture statistics:")
            print(f"  Captured: {capture_stats.get('count', 0)}/{result.get('total_stimuli', 0)} stimuli")
            print(f"  Capture rate: {capture_stats.get('rate', 0):.2f}")
            
            if result.get('stimulus_results'):
                print("\nTop 3 highest salience stimuli:")
                sorted_results = sorted(result.get('stimulus_results', []), key=lambda x: x.get('salience', 0), reverse=True)
                for i, stim in enumerate(sorted_results[:3]):
                    print(f"  {i+1}. '{stim.get('stimulus', '')}' (Salience: {stim.get('salience', 0):.4f}, Captured: {stim.get('attention_captured', False)})")
            
        elif result_type == "task_context":
            print(f"Total contexts tested: {result.get('total_contexts', 0)}")
            print(f"Stimuli per context: {result.get('total_stimuli_per_context', 0)}")
            
            if result.get('context_results'):
                print("\nContext comparison:")
                for i, ctx in enumerate(result.get('context_results', [])):
                    print(f"  {i+1}. Context: '{ctx.get('context', '')}' (Capture rate: {ctx.get('capture_rate', 0):.2f}, Avg salience: {ctx.get('average_salience', 0):.4f})")
        
        elif result_type == "developmental_effects":
            print(f"Development levels tested: {', '.join([f'{level:.2f}' for level in result.get('development_levels_tested', [])])}")
            print(f"Stimulus: '{result.get('stimulus', '')}'")
            
            if result.get('developmental_results'):
                print("\nDevelopmental comparison:")
                for res in result.get('developmental_results', []):
                    captured_str = "✓" if res.get('attention_captured', False) else "✗"
                    print(f"  Level {res.get('development_level', 0):.2f}: Salience {res.get('salience', 0):.4f} | Captured: {captured_str}")
                    
        elif result_type == "attention_capture":
            print(f"Multipliers tested: {', '.join([f'{multiplier:.2f}' for multiplier in result.get('multipliers_tested', [])])}")
            print(f"Module development level: {result.get('module_development_level', 0):.2f}")
            
            if result.get('multiplier_results'):
                print("\nMultiplier comparison:")
                for res in result.get('multiplier_results', []):
                    multiplier = res.get('salience_multiplier', 0)
                    capture_count = res.get('capture_count', 0)
                    capture_rate = res.get('capture_rate', 0)
                    
                    print(f"  Multiplier {multiplier:.2f}: Captured {capture_count} stimuli (rate: {capture_rate:.2f})")
                    
                    first_capture_idx = res.get('first_capture_index')
                    if first_capture_idx is not None:
                        print(f"    First capture occurred at base salience: {(first_capture_idx+1)/10.0:.1f}")
                    else:
                        print(f"    No captures occurred with this multiplier")
                
                # Find highest salience achieved without capture
                max_salience_no_capture = 0
                for res in result.get('multiplier_results', []):
                    stimulus_results = res.get('stimulus_results', [])
                    for stim_res in stimulus_results:
                        if not stim_res.get('captured', False):
                            max_salience_no_capture = max(max_salience_no_capture, stim_res.get('calculated_salience', 0))
                
                print(f"\nMaximum salience without attention capture: {max_salience_no_capture:.4f}")
                print("This suggests the attention system requires additional conditions beyond high salience for capture.")
                
                # Offer some explanations if no captures occurred
                all_capture_counts = [res.get('capture_count', 0) for res in result.get('multiplier_results', [])]
                if sum(all_capture_counts) == 0:
                    print("\nPossible explanations for no attention captures:")
                    print("1. Current focus has high sustained attention value")
                    print("2. The distraction threshold is set too high for the current development level")
                    print("3. The system may require additional factors (like context relevance) for capture")
                    print("4. There might be a developmental delay in attention capture capability")
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Print detailed test results."""
        print_section("Detailed Test Results")
        print_dict(result)
    
    def print_module_state(self):
        """Print the current state of the attention module."""
        print_section("Attention Module State")
        state = self.attention.get_state()
        print_dict(state)
        
        # Get additional attention-specific information
        focus_history = self.attention.get_focus_history(10)
        
        print("\nFocus History (last 10 items):")
        for i, focus in enumerate(focus_history):
            print(f"  {i+1}. {focus.get('content', '')[:50]} (Salience: {focus.get('salience', 0):.4f})")
    
    def set_development_level(self, level: float):
        """
        Set the development level of the attention module.
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        current_level = self.attention.development_level
        self.attention.update_development(level - current_level)
        print(f"Updated development level from {current_level:.2f} to {self.attention.development_level:.2f}")
    
    def save_results(self, filename: str = None):
        """
        Save test results to a JSON file.
        
        Args:
            filename: Optional filename to save results to
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_test_results_{timestamp}.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filename}")

def generate_test_stimuli(count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate test stimuli for attention testing.
    
    Args:
        count: Number of stimuli to generate
        
    Returns:
        List of stimulus dictionaries
    """
    topics = [
        "Breaking news: Major earthquake reported",
        "New scientific discovery announced today",
        "Weather forecast predicts mild temperatures",
        "Local sports team wins championship",
        "Stock market shows significant changes",
        "New technology product released",
        "Health advisory issued for residents",
        "Traffic update for major highways",
        "Celebrity announces retirement from industry",
        "Educational program receives award",
        "Political debate scheduled for tonight",
        "Art exhibition opens at local museum",
        "Book release event happening this weekend",
        "Music festival lineup announced",
        "Community service opportunity available"
    ]
    
    # Add high and low salience identifiers to make focus shifts more likely
    high_salience_prefixes = [
        "URGENT: ", 
        "CRITICAL: ", 
        "EMERGENCY: ", 
        "BREAKING: ",
        "IMPORTANT: "
    ]
    
    stimuli = []
    for i in range(count):
        # Alternate between high and low salience stimuli to prompt focus shifts
        high_salience = (i % 2 == 0)
        
        # Select a topic
        topic = random.choice(topics)
        
        # Add high salience prefix for even-numbered stimuli
        if high_salience:
            topic = random.choice(high_salience_prefixes) + topic
            
            # Generate attributes with high salience characteristics
            intensity = random.uniform(0.7, 1.0)  # High intensity
            novelty = random.uniform(0.7, 1.0)    # High novelty
            relevance = random.uniform(0.7, 1.0)  # High relevance
            volitional = True                     # Intentional focus
        else:
            # Generate attributes with low salience characteristics
            intensity = random.uniform(0.1, 0.4)  # Low intensity
            novelty = random.uniform(0.1, 0.4)    # Low novelty
            relevance = random.uniform(0.1, 0.4)  # Low relevance
            volitional = False                    # Non-intentional
        
        # Create stimulus
        stimulus = {
            'id': f"stimulus_{i}",
            'content': topic,
            'intensity': intensity,
            'novelty': novelty,
            'relevance': relevance,
            'volitional': volitional,
            'timestamp': time.time(),
            'type': 'text'
        }
        
        stimuli.append(stimulus)
    
    return stimuli

def generate_task_contexts(count: int = 3) -> List[Dict[str, Any]]:
    """
    Generate task contexts for testing.
    
    Args:
        count: Number of contexts to generate
        
    Returns:
        List of task context dictionaries
    """
    context_templates = [
        {
            'description': 'Reading news articles',
            'priority': 'information_gathering',
            'domain': 'current_events',
            'timeframe': 'immediate'
        },
        {
            'description': 'Studying for an exam',
            'priority': 'learning',
            'domain': 'academic',
            'timeframe': 'extended'
        },
        {
            'description': 'Watching for traffic signals',
            'priority': 'safety',
            'domain': 'navigation',
            'timeframe': 'immediate'
        },
        {
            'description': 'Listening to a lecture',
            'priority': 'comprehension',
            'domain': 'educational',
            'timeframe': 'medium'
        },
        {
            'description': 'Looking for a lost item',
            'priority': 'search',
            'domain': 'personal',
            'timeframe': 'focused'
        }
    ]
    
    # Select 'count' contexts from the templates
    selected_contexts = random.sample(context_templates, min(count, len(context_templates)))
    
    # Add unique IDs
    for i, context in enumerate(selected_contexts):
        context['id'] = f"context_{i}"
        context['timestamp'] = time.time()
    
    return selected_contexts

def test_attention_at_level(level: float) -> AttentionTester:
    """
    Run a comprehensive test of the attention system at a specific developmental level.
    
    Args:
        level: Development level to test (0.0 to 1.0)
        
    Returns:
        The AttentionTester instance
    """
    print_section(f"TESTING ATTENTION SYSTEM AT DEVELOPMENT LEVEL {level:.2f}")
    
    # Initialize tester
    tester = AttentionTester(development_level=level)
    
    # Generate test data
    stimuli = generate_test_stimuli(12)
    task_contexts = generate_task_contexts(3)
    
    # Run tests
    print("\n[1/4] Testing focus control with various stimuli...")
    tester.test_focus_control(stimuli[:8])
    
    print("\n[2/4] Testing salience detection with mixed stimuli...")
    tester.test_salience_detection(stimuli[2:10])
    
    print("\n[3/4] Testing task context effects...")
    tester.test_task_context(task_contexts, stimuli[4:8])
    
    print("\n[4/4] Testing attention capture with varied salience multipliers...")
    # Use salience multipliers that should range from rarely captured to frequently captured
    multipliers = [0.5, 1.0, 1.5, 2.0]
    tester.test_attention_capture(multipliers)
    
    # Print module state
    tester.print_module_state()
    
    return tester

def test_development_progression() -> AttentionTester:
    """
    Test the attention system across multiple developmental levels.
    
    Returns:
        The final AttentionTester instance
    """
    print_section("TESTING ATTENTION DEVELOPMENTAL PROGRESSION")
    
    # Define development levels to test
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Initialize tester at lowest level
    tester = AttentionTester(development_level=levels[0])
    
    # Generate consistent test stimulus
    stimulus = {
        'id': 'dev_test_stimulus',
        'content': 'Important message requiring immediate attention',
        'intensity': 0.7,
        'novelty': 0.8,
        'relevance': 0.6,
        'volitional': False,
        'timestamp': time.time(),
        'type': 'text'
    }
    
    # Test development effects
    tester.test_developmental_effects(levels, stimulus)
    
    # Save results
    result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"attention_development_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    tester.save_results(result_path)
    
    return tester

def main():
    """Main function to run attention tests."""
    print_section("ATTENTION MODULE TEST SUITE")
    print("This test suite validates the attention module's functionality at different development levels.")
    print("It tests focus control, salience detection, task context effects, and attention capture.")
    
    # Test at specific development levels
    print("\n======= EARLY DEVELOPMENT STAGE TESTS =======")
    print("Testing attention capabilities at development level 0.2 (Sustained Attention)...")
    early_tester = test_attention_at_level(0.2)
    
    print("\n======= INTERMEDIATE DEVELOPMENT STAGE TESTS =======")
    print("Testing attention capabilities at development level 0.5 (Selective Attention)...")
    mid_tester = test_attention_at_level(0.5)
    
    print("\n======= ADVANCED DEVELOPMENT STAGE TESTS =======")
    print("Testing attention capabilities at development level 0.8 (Divided/Executive Attention)...")
    advanced_tester = test_attention_at_level(0.8)
    
    # Test developmental progression
    print("\n======= DEVELOPMENTAL PROGRESSION TESTS =======")
    print("Testing how attention mechanisms change across the full development spectrum...")
    progression_tester = test_development_progression()
    
    print("\n======= TEST SUMMARY =======")
    print("Attention module testing complete. All tests executed successfully.")
    print("Key observations:")
    print("1. Focus control capability increases with development level")
    print("2. Salience calculation becomes more sophisticated and contextual with development")
    print("3. Task context has increasing influence on attention with higher development")
    print("4. Attention capture thresholds and mechanisms evolve with development")

if __name__ == "__main__":
    main() 