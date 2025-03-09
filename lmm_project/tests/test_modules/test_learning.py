import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import random
from pprint import pprint

# Add the project root to the path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from lmm_project.modules.learning import get_module
from lmm_project.core.event_bus import EventBus

# ANSI colors for prettier output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
RED = "\033[31m"

def print_section(title):
    """Print a section title with decoration"""
    width = 80
    print(f"\n{BOLD}{BLUE}{'=' * width}")
    print(f"{title.center(width)}")
    print(f"{'=' * width}{RESET}\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """Recursively print dictionary contents with nice formatting"""
    if current_depth > max_depth:
        print(" " * indent + "...")
        return
    
    if not isinstance(data, dict):
        print(" " * indent + f"{YELLOW}{data}{RESET}")
        return
    
    for key, value in data.items():
        if isinstance(value, dict) and len(value) > 0:
            print(" " * indent + f"{CYAN}{key}:{RESET}")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and len(value) > 0:
            print(" " * indent + f"{CYAN}{key}:{RESET}")
            if isinstance(value[0], dict):
                for i, item in enumerate(value[:3]):  # Limit to first 3 items for brevity
                    print(" " * (indent + 2) + f"{MAGENTA}Item {i}:{RESET}")
                    print_dict(item, indent + 4, max_depth, current_depth + 1)
                if len(value) > 3:
                    print(" " * (indent + 4) + f"... ({len(value) - 3} more items)")
            else:
                print(" " * (indent + 2) + f"{YELLOW}{value[:5]}{RESET}" + 
                     (" ... " + str(len(value) - 5) + " more" if len(value) > 5 else ""))
        else:
            print(" " * indent + f"{CYAN}{key}:{RESET} {YELLOW}{value}{RESET}")

class LearningTester:
    """
    Class for testing the Learning module
    """
    
    def __init__(self, development_level: float = 0.0):
        """
        Initialize the learning tester
        
        Args:
            development_level: Initial developmental level
        """
        self.event_bus = EventBus()
        self.learning_module = get_module(
            module_id="learning_test",
            event_bus=self.event_bus,
            development_level=development_level
        )
        self.development_level = development_level
        self.test_results = []
        print(f"{GREEN}Learning system initialized at development level {development_level:.2f}{RESET}")
    
    def test_associative_learning(self, stimuli_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Test associative learning with stimulus-response pairs
        
        Args:
            stimuli_pairs: List of stimulus-response pairs to learn
            
        Returns:
            Test results
        """
        results = []
        
        # Learn associations
        for stimulus, response in stimuli_pairs:
            print(f"\nLearning association: '{stimulus}' → '{response}'")
            learn_result = self.learning_module.process_input({
                "learning_type": "associative",
                "operation": "learn",
                "stimulus": stimulus,
                "response": response,
                "source": "test"
            })
            results.append({
                "operation": "learn",
                "stimulus": stimulus,
                "response": response,
                "result": learn_result
            })
            print(f"Association strength: {learn_result.get('strength', 'N/A')}")
        
        # Test prediction from stimuli
        for stimulus, _ in stimuli_pairs:
            print(f"\nPredicting from stimulus: '{stimulus}'")
            predict_result = self.learning_module.process_input({
                "learning_type": "associative",
                "operation": "predict",
                "stimulus": stimulus
            })
            results.append({
                "operation": "predict",
                "stimulus": stimulus,
                "result": predict_result
            })
            
            # Print prediction results
            predictions = predict_result.get("predictions", [])
            if predictions:
                print(f"Predictions for '{stimulus}':")
                for pred in predictions:
                    print(f"  - '{pred['response']}' (confidence: {pred['confidence']:.2f})")
            else:
                print(f"No predictions found for '{stimulus}'")
        
        # Test reinforcement
        if stimuli_pairs:
            stimulus, response = stimuli_pairs[0]
            print(f"\nReinforcing association: '{stimulus}' → '{response}'")
            reinforce_result = self.learning_module.process_input({
                "learning_type": "associative",
                "operation": "reinforce",
                "stimulus": stimulus,
                "response": response,
                "amount": 0.2
            })
            results.append({
                "operation": "reinforce",
                "stimulus": stimulus,
                "response": response,
                "result": reinforce_result
            })
            
            previous = reinforce_result.get("previous_strength", 0)
            new_val = reinforce_result.get("new_strength", 0)
            print(f"Association strength: {previous:.2f} → {new_val:.2f}")
        
        final_result = {
            "test_type": "associative_learning",
            "developmental_level": self.development_level,
            "results": results
        }
        self.test_results.append(final_result)
        return final_result
    
    def test_reinforcement_learning(self, states_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test reinforcement learning with states, actions, and rewards
        
        Args:
            states_actions: List of state-action-reward dictionaries
            
        Returns:
            Test results
        """
        results = []
        
        # Learn from experiences
        for item in states_actions:
            state = item["state"]
            action = item["action"]
            reward = item["reward"]
            next_state = item.get("next_state")
            
            print(f"\nLearning from experience:")
            print(f"  State: '{state}', Action: '{action}', Reward: {reward}")
            if next_state:
                print(f"  Next State: '{next_state}'")
            
            learn_result = self.learning_module.process_input({
                "learning_type": "reinforcement",
                "operation": "learn",
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state
            })
            results.append({
                "operation": "learn",
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "result": learn_result
            })
            
            # Print Q-value update
            previous_q = learn_result.get("previous_q", 0)
            updated_q = learn_result.get("updated_q", 0)
            print(f"Q-value update: {previous_q:.2f} → {updated_q:.2f}")
        
        # Test action selection
        unique_states = list(set(item["state"] for item in states_actions))
        for state in unique_states:
            available_actions = [
                item["action"] for item in states_actions 
                if item["state"] == state
            ]
            
            print(f"\nSelecting action for state: '{state}'")
            select_result = self.learning_module.process_input({
                "learning_type": "reinforcement",
                "operation": "select_action",
                "state": state,
                "available_actions": available_actions
            })
            results.append({
                "operation": "select_action",
                "state": state,
                "available_actions": available_actions,
                "result": select_result
            })
            
            # Print selected action
            selected = select_result.get("selected_action", "")
            selection_type = select_result.get("selection_type", "")
            print(f"Selected action: '{selected}' (via {selection_type})")
        
        final_result = {
            "test_type": "reinforcement_learning",
            "developmental_level": self.development_level,
            "results": results
        }
        self.test_results.append(final_result)
        return final_result
    
    def test_procedural_learning(self, skills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test procedural learning with skill practice
        
        Args:
            skills: List of skill definitions with steps and practice parameters
            
        Returns:
            Test results
        """
        results = []
        
        # Learn and practice skills
        for skill_data in skills:
            skill_name = skill_data["name"]
            steps = skill_data.get("steps", [])
            practice_iterations = skill_data.get("practice_iterations", 3)
            
            # First, learn the sequence
            print(f"\nLearning new skill: '{skill_name}'")
            if steps:
                print(f"Steps: {', '.join(steps)}")
            
            learn_result = self.learning_module.process_input({
                "learning_type": "procedural",
                "operation": "learn_sequence",
                "skill": skill_name,
                "steps": steps
            })
            results.append({
                "operation": "learn_sequence",
                "skill": skill_name,
                "steps": steps,
                "result": learn_result
            })
            
            # Practice the skill multiple times
            print(f"\nPracticing skill '{skill_name}' ({practice_iterations} iterations):")
            practice_results = []
            
            for i in range(practice_iterations):
                # Randomize practice quality a bit
                quality = min(1.0, max(0.3, 0.5 + (i * 0.1) + random.uniform(-0.1, 0.1)))
                duration = random.uniform(0.5, 2.0)
                
                practice_result = self.learning_module.process_input({
                    "learning_type": "procedural",
                    "operation": "practice",
                    "skill": skill_name,
                    "quality": quality,
                    "duration": duration
                })
                
                practice_results.append(practice_result)
                
                # Show improvement
                prev = practice_result.get("previous_proficiency", 0)
                new_val = practice_result.get("new_proficiency", 0)
                print(f"  Practice {i+1}: Proficiency {prev:.2f} → {new_val:.2f} " +
                      f"(quality: {quality:.2f}, duration: {duration:.1f}min)")
            
            results.append({
                "operation": "practice",
                "skill": skill_name,
                "iterations": practice_iterations,
                "results": practice_results
            })
            
            # Test recall
            print(f"\nTesting recall of skill '{skill_name}':")
            recall_result = self.learning_module.process_input({
                "learning_type": "procedural",
                "operation": "recall_skill",
                "skill": skill_name
            })
            results.append({
                "operation": "recall_skill",
                "skill": skill_name,
                "result": recall_result
            })
            
            # Show recall results
            recall_rate = recall_result.get("recall_success_rate", 0)
            proficiency = recall_result.get("proficiency", 0)
            recalled_steps = recall_result.get("recalled_steps", [])
            missed_steps = recall_result.get("missed_steps", [])
            
            print(f"Recall success rate: {recall_rate:.2f} (proficiency: {proficiency:.2f})")
            if recalled_steps:
                print(f"Recalled steps: {', '.join(recalled_steps)}")
            if missed_steps:
                print(f"Missed steps: {', '.join(missed_steps)}")
            
            # Check automation
            print(f"\nChecking automation status for '{skill_name}':")
            automation_result = self.learning_module.process_input({
                "learning_type": "procedural",
                "operation": "check_automation",
                "skill": skill_name
            })
            results.append({
                "operation": "check_automation",
                "skill": skill_name,
                "result": automation_result
            })
            
            # Show automation status
            automated = automation_result.get("automated", False)
            cognitive_load = automation_result.get("cognitive_load", 1.0)
            print(f"Automated: {automated} (cognitive load: {cognitive_load:.2f})")
            
            if "automation_checks" in automation_result:
                checks = automation_result["automation_checks"]
                print(f"Automation checks:")
                for check, status in checks.items():
                    print(f"  - {check}: {status}")
        
        final_result = {
            "test_type": "procedural_learning",
            "developmental_level": self.development_level,
            "results": results
        }
        self.test_results.append(final_result)
        return final_result
    
    def test_meta_learning(self, domains: List[str], learning_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test meta-learning with strategy selection and evaluation
        
        Args:
            domains: List of learning domains to test
            learning_contents: List of content types and parameters
            
        Returns:
            Test results
        """
        results = []
        
        # Get available strategies
        print(f"\nQuerying available learning strategies:")
        strategies_result = self.learning_module.process_input({
            "learning_type": "meta",
            "operation": "get_strategy"
        })
        
        strategies = strategies_result.get("strategies", [])
        results.append({
            "operation": "get_strategies",
            "result": strategies_result
        })
        
        print(f"Available strategies: {len(strategies)}")
        for i, strategy in enumerate(strategies):
            print(f"  {i+1}. {strategy['name']} (effectiveness: {strategy['effectiveness']:.2f})")
        
        # Test strategy selection for different domains
        for domain in domains:
            for content in learning_contents:
                content_type = content["type"]
                cognitive_resources = content.get("cognitive_resources", 0.8)
                
                print(f"\nSelecting strategy for domain '{domain}', content type '{content_type}':")
                select_result = self.learning_module.process_input({
                    "learning_type": "meta",
                    "operation": "select_strategy",
                    "domain": domain,
                    "content_type": content_type,
                    "cognitive_resources": cognitive_resources
                })
                
                results.append({
                    "operation": "select_strategy",
                    "domain": domain,
                    "content_type": content_type,
                    "cognitive_resources": cognitive_resources,
                    "result": select_result
                })
                
                # Show selected strategy
                if "selected_strategy" in select_result:
                    strategy = select_result["selected_strategy"]
                    print(f"Selected: {strategy['name']} " +
                          f"(effectiveness: {strategy['effectiveness']:.2f}, " +
                          f"cognitive load: {strategy['cognitive_load']:.2f})")
                    print(f"Description: {strategy['description']}")
                else:
                    print(f"No suitable strategy found")
        
        # Test strategy effectiveness evaluation
        if strategies and domains:
            strategy_id = strategies[0]["id"]
            domain = domains[0]
            
            print(f"\nEvaluating strategy effectiveness:")
            success_levels = [0.3, 0.7, 0.9]
            
            for success in success_levels:
                print(f"Testing success level: {success:.2f}")
                eval_result = self.learning_module.process_input({
                    "learning_type": "meta",
                    "operation": "evaluate_outcome",
                    "strategy_id": strategy_id,
                    "domain": domain,
                    "success_level": success
                })
                
                results.append({
                    "operation": "evaluate_outcome",
                    "strategy_id": strategy_id,
                    "domain": domain,
                    "success_level": success,
                    "result": eval_result
                })
                
                # Show updated success rate
                prev = eval_result.get("previous_success_rate", 0)
                new_val = eval_result.get("updated_success_rate", 0)
                print(f"Success rate update: {prev:.2f} → {new_val:.2f}")
        
        # Test strategy creation (only at higher developmental levels)
        if self.development_level >= 0.5:
            print(f"\nCreating new learning strategy:")
            create_result = self.learning_module.process_input({
                "learning_type": "meta",
                "operation": "create_strategy",
                "name": "test_hybrid_strategy",
                "description": "A hybrid strategy combining repetition and elaboration",
                "applicable_domains": ["general", "conceptual", "language"]
            })
            
            results.append({
                "operation": "create_strategy",
                "result": create_result
            })
            
            if create_result.get("status") == "success":
                print(f"Created strategy: {create_result.get('strategy_name', '')}")
                print(f"Description: {create_result.get('description', '')}")
                print(f"Effectiveness: {create_result.get('effectiveness', 0):.2f}")
            else:
                print(f"Failed to create strategy: {create_result.get('message', '')}")
        
        final_result = {
            "test_type": "meta_learning",
            "developmental_level": self.development_level,
            "results": results
        }
        self.test_results.append(final_result)
        return final_result
    
    def test_integrated_learning(self, learning_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test integrated learning that combines multiple learning approaches
        
        Args:
            learning_tasks: List of integrated learning tasks
            
        Returns:
            Test results
        """
        results = []
        
        for task in learning_tasks:
            domain = task.get("domain", "general")
            content_type = task.get("content_type", "general")
            learning_types = task.get("learning_types", ["associative", "reinforcement"])
            
            print(f"\nTesting integrated learning for domain '{domain}', content type '{content_type}':")
            print(f"Learning types: {', '.join(learning_types)}")
            
            integrate_result = self.learning_module.process_input({
                "learning_type": "integrate",
                "domain": domain,
                "content_type": content_type,
                "learning_types": learning_types,
                "primary_type": learning_types[0] if learning_types else "associative",
                "stimulus": task.get("stimulus", "test_stimulus"),
                "response": task.get("response", "test_response"),
                "state": task.get("state", "test_state"),
                "action": task.get("action", "test_action"),
                "reward": task.get("reward", 0.5)
            })
            
            results.append({
                "operation": "integrate",
                "domain": domain,
                "content_type": content_type,
                "learning_types": learning_types,
                "result": integrate_result
            })
            
            # Show integration results
            integration_level = integrate_result.get("integration_level", 0)
            print(f"Integration level: {integration_level:.2f}")
            
            if "learning_strategy" in integrate_result and integrate_result["learning_strategy"]:
                strategy = integrate_result["learning_strategy"]
                print(f"Applied strategy: {strategy['name']}")
            
            # Show results from each learning type
            if "integrated_results" in integrate_result:
                int_results = integrate_result["integrated_results"]
                print(f"Results by learning type:")
                for l_type, l_result in int_results.items():
                    status = l_result.get("status", "unknown")
                    print(f"  - {l_type}: {status}")
        
        final_result = {
            "test_type": "integrated_learning",
            "developmental_level": self.development_level,
            "results": results
        }
        self.test_results.append(final_result)
        return final_result
    
    def print_module_state(self):
        """Print the current state of the learning module and its submodules"""
        print_section("Learning Module State")
        
        state = self.learning_module.get_state()
        print_dict(state)
    
    def set_development_level(self, level: float):
        """
        Set the developmental level of the learning system
        
        Args:
            level: New developmental level (0.0 to 1.0)
        """
        previous = self.development_level
        self.development_level = level
        
        # Update the module's development level
        self.learning_module.update_development(level - previous)
        
        print(f"{GREEN}Development level updated: {previous:.2f} → {level:.2f}{RESET}")
    
    def save_results(self, filename: str = None):
        """
        Save test results to a JSON file
        
        Args:
            filename: Name of the file to save to (default: learning_test_results.json)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_test_results_{timestamp}.json"
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "development_level": self.development_level,
            "test_results": self.test_results
        }
        
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"{GREEN}Test results saved to: {filepath}{RESET}")


def test_learning_at_level(level: float) -> LearningTester:
    """
    Run a comprehensive learning test at a specific developmental level
    
    Args:
        level: Developmental level to test at
        
    Returns:
        The configured tester instance
    """
    print_section(f"Testing Learning System at Level {level:.2f}")
    
    # Create the tester with the specified development level
    tester = LearningTester(development_level=level)
    
    # Test associative learning
    print_section("Associative Learning Test")
    stimulus_pairs = [
        ("apple", "fruit"),
        ("cat", "animal"),
        ("car", "vehicle"),
        ("happy", "emotion"),
        ("red", "color")
    ]
    tester.test_associative_learning(stimulus_pairs)
    
    # Test reinforcement learning
    print_section("Reinforcement Learning Test")
    states_actions = [
        {"state": "hungry", "action": "eat", "reward": 0.8, "next_state": "satisfied"},
        {"state": "hungry", "action": "sleep", "reward": -0.2, "next_state": "hungry"},
        {"state": "satisfied", "action": "play", "reward": 0.6, "next_state": "tired"},
        {"state": "tired", "action": "sleep", "reward": 0.9, "next_state": "rested"},
        {"state": "tired", "action": "play", "reward": -0.1, "next_state": "exhausted"},
        {"state": "rested", "action": "study", "reward": 0.7, "next_state": "knowledgeable"}
    ]
    tester.test_reinforcement_learning(states_actions)
    
    # Test procedural learning
    print_section("Procedural Learning Test")
    skills = [
        {
            "name": "make_sandwich",
            "steps": ["get bread", "add spread", "add toppings", "close sandwich"],
            "practice_iterations": 5
        },
        {
            "name": "tie_shoelaces",
            "steps": ["cross laces", "loop one lace", "wrap around", "pull through", "tighten"],
            "practice_iterations": 7
        },
        {
            "name": "simple_math",
            "steps": ["read problem", "identify operation", "apply formula", "calculate", "verify"],
            "practice_iterations": 4
        }
    ]
    tester.test_procedural_learning(skills)
    
    # Test meta-learning
    print_section("Meta-Learning Test")
    domains = ["language", "mathematics", "music", "physical", "social"]
    learning_contents = [
        {"type": "factual", "cognitive_resources": 0.9},
        {"type": "conceptual", "cognitive_resources": 0.7},
        {"type": "procedural", "cognitive_resources": 0.8}
    ]
    tester.test_meta_learning(domains, learning_contents)
    
    # Test integrated learning
    print_section("Integrated Learning Test")
    learning_tasks = [
        {
            "domain": "language",
            "content_type": "vocabulary",
            "learning_types": ["associative", "reinforcement"],
            "stimulus": "book",
            "response": "reading",
            "state": "learning_vocab",
            "action": "practice_flashcards",
            "reward": 0.7
        },
        {
            "domain": "mathematics",
            "content_type": "problem_solving",
            "learning_types": ["procedural", "reinforcement", "meta"],
            "stimulus": "equation",
            "response": "solution",
            "state": "solving_problem",
            "action": "apply_formula",
            "reward": 0.8
        }
    ]
    tester.test_integrated_learning(learning_tasks)
    
    # Print module state
    tester.print_module_state()
    
    return tester

def test_development_progression() -> LearningTester:
    """
    Test the learning system's progression through different developmental levels
    
    Returns:
        The final tester instance at the highest development level
    """
    print_section("Testing Developmental Progression")
    
    # Define test levels and create tester at lowest level
    levels = [0.0, 0.2, 0.5, 0.8, 1.0]
    tester = LearningTester(development_level=levels[0])
    
    # Basic test cases to use at each level
    stimulus_pairs = [
        ("dog", "pet"),
        ("piano", "instrument"),
        ("running", "exercise")
    ]
    
    states_actions = [
        {"state": "new_problem", "action": "analyze", "reward": 0.5, "next_state": "analyzing"},
        {"state": "analyzing", "action": "solve", "reward": 0.7, "next_state": "solved"}
    ]
    
    skills = [
        {
            "name": "test_skill",
            "steps": ["step1", "step2", "step3"],
            "practice_iterations": 3
        }
    ]
    
    domains = ["test_domain"]
    learning_contents = [{"type": "test_content", "cognitive_resources": 0.8}]
    
    # Run basic tests at each level
    for i, level in enumerate(levels):
        if i > 0:  # Skip first level as tester is already at that level
            print(f"\n{CYAN}Advancing to development level {level:.2f}{RESET}")
            tester.set_development_level(level)
            time.sleep(1)  # Small pause for readability
        
        print(f"\n{YELLOW}Testing at level {level:.2f}{RESET}")
        
        # Run simplified tests at each level
        tester.test_associative_learning(stimulus_pairs)
        tester.test_reinforcement_learning(states_actions)
        tester.test_procedural_learning(skills)
        tester.test_meta_learning(domains, learning_contents)
        
        print(f"\n{MAGENTA}Module state summary at level {level:.2f}:{RESET}")
        tester.print_module_state()
        
        print(f"\n{GREEN}Completed testing at level {level:.2f}{RESET}")
        time.sleep(1)  # Small pause for readability
    
    # Save results at the end
    tester.save_results()
    
    return tester

def main():
    """Main entry point for the learning test script"""
    print_section("Learning Module Test Suite")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "progression":
            # Test progression through developmental levels
            test_development_progression()
        elif sys.argv[1] == "level" and len(sys.argv) > 2:
            # Test at specific level
            try:
                level = float(sys.argv[2])
                test_learning_at_level(level)
            except ValueError:
                print(f"{RED}Invalid level: {sys.argv[2]}. Please provide a number between 0.0 and 1.0{RESET}")
        else:
            print(f"{YELLOW}Usage: python test_learning.py [progression | level <0.0-1.0>]{RESET}")
    else:
        # Default: test at mid and high levels
        tester_mid = test_learning_at_level(0.5)
        time.sleep(1)  # Pause for readability
        tester_high = test_learning_at_level(0.9)
        
        # Save results
        tester_high.save_results()
    
    print(f"\n{GREEN}Learning module testing completed!{RESET}")

if __name__ == "__main__":
    main() 