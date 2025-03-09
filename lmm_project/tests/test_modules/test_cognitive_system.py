"""
Comprehensive test script for the integrated cognitive system.

This script tests how all implemented modules (perception, attention, memory, 
emotion, learning) work together to process and learn from complex information.
It simulates a realistic learning scenario where the cognitive system engages
with educational content at different developmental stages.
"""

import logging
import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import random

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
from lmm_project.modules.emotion import get_module as get_emotion_module
from lmm_project.modules.learning import get_module as get_learning_module
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
            print(" " * indent + f"{CYAN}{key}:{RESET}")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(" " * indent + f"{CYAN}{key}: [{RESET}")
            for i, item in enumerate(value[:3]):  # Show first 3 items
                print(" " * (indent + 4) + f"{MAGENTA}Item {i}:{RESET}")
                print_dict(item, indent + 8, max_depth, current_depth + 1)
            if len(value) > 3:
                print(" " * (indent + 4) + f"{YELLOW}... ({len(value) - 3} more items){RESET}")
            print(" " * indent + "]")
        else:
            # Truncate very long values
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(" " * indent + f"{CYAN}{key}:{RESET} {YELLOW}{value}{RESET}")

class CognitiveSystem:
    """
    Integrated cognitive system that coordinates all cognitive modules
    
    This class demonstrates how the different cognitive modules interact
    through the event bus to process information, learn, and develop.
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
        
        self.emotion = get_emotion_module(
            module_id="emotion",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        self.learning = get_learning_module(
            module_id="learning",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        # Set development level
        self.development_level = development_level
        
        # Store system state history
        self.state_history = []
        
        logging.info(f"Cognitive system initialized at development level {development_level:.2f}")
        
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input through the cognitive pipeline
        
        This simulates the flow of information through the cognitive system:
        1. Perception processes the raw input
        2. Attention determines what aspects to focus on
        3. Emotion evaluates affective response
        4. Memory stores and retrieves relevant information
        5. Learning acquires knowledge from the experience
        
        Args:
            text: Input text to process
            context: Optional contextual information
            
        Returns:
            Integrated results from all modules
        """
        process_id = f"process_{int(time.time())}"
        logging.info(f"Processing text: '{text[:50]}...' (id: {process_id})")
        
        results = {
            "development_level": self.development_level,
            "process_id": process_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        # Step 1: Perception processes the input
        perception_start = time.time()
        perception_result = self.perception.process_input({
            "text": text,
            "process_id": process_id,
            "context": context
        })
        perception_time = time.time() - perception_start
        results["perception"] = perception_result
        results["perception_time"] = perception_time
        
        # At earliest development, only perception works
        if self.development_level < 0.1:
            logging.info("Development too low for attention/emotion/memory/learning processing")
            results.update({
                "attention": "Not yet developed",
                "emotion": "Not yet developed",
                "memory": "Not yet developed",
                "learning": "Not yet developed"
            })
            return results
            
        # Step 2: Attention processes the perception result
        attention_start = time.time()
        attention_result = self.attention.process_input({
            "content": perception_result,
            "source": "perception",
            "process_id": process_id,
            # Calculate intensity based on patterns
            "intensity": min(1.0, len(perception_result.get("patterns", [])) / 10),
            # Higher novelty for questions, exclamations, and complex patterns
            "novelty": 0.8 if ("?" in text or "!" in text) else 0.5
        })
        attention_time = time.time() - attention_start
        results["attention"] = attention_result
        results["attention_time"] = attention_time
        
        # If development is too low for emotion processing
        if self.development_level < 0.3:
            logging.info("Development too low for emotion/memory/learning processing")
            results.update({
                "emotion": "Not yet developed",
                "memory": "Not yet developed",
                "learning": "Not yet developed"
            })
            return results
        
        # Step 3: Emotional response to the input
        emotion_start = time.time()
        emotion_result = self.emotion.process_input({
            "operation": "generate",
            "content": {
                "text": text,
                "context": {
                    "attention_focus": attention_result.get("current_focus", {}),
                    "perception_patterns": perception_result.get("patterns", [])
                }
            },
            "process_id": process_id
        })
        emotion_time = time.time() - emotion_start
        results["emotion"] = emotion_result
        results["emotion_time"] = emotion_time

        # If development is too low for memory/learning processing
        if self.development_level < 0.4:
            logging.info("Development too low for memory/learning processing")
            results.update({
                "memory": "Not yet developed",
                "learning": "Not yet developed"
            })
            return results

        # Step 4: Memory operations
        memory_start = time.time()
        try:
            # Working memory at basic level
            memory_input = {
                "operation": "store",
                "memory_type": "working",
                "content": {
                    "text": text,
                    "perception": perception_result,
                    "attention": attention_result,
                    "emotion": emotion_result if isinstance(emotion_result, dict) else {}
                },
                "process_id": process_id
            }
            
            # Add to episodic memory if development is sufficient
            if self.development_level >= 0.5:
                memory_input = {
                    "operation": "store",
                    "memory_type": "episodic",
                    "content": {
                        "text": text,
                        "perception": perception_result,
                        "attention": attention_result,
                        "emotion": emotion_result if isinstance(emotion_result, dict) else {},
                        "timestamp": float(time.time())
                    },
                    "process_id": process_id
                }
                
                # Also add to semantic memory for higher development
                if self.development_level >= 0.7 and perception_result.get("interpretation"):
                    # Try to extract a concept from the perception interpretation
                    interpretation = perception_result.get("interpretation", {})
                    if "content_type" in interpretation and interpretation["content_type"] in ["factual", "conceptual"]:
                        # Additional semantic memory storage
                        semantic_input = {
                            "operation": "store",
                            "memory_type": "semantic",
                            "content": {
                                "concept_name": interpretation.get("primary_pattern_type", "concept"),
                                "description": text[:100],
                                "attributes": interpretation,
                                "source": "perception"
                            },
                            "process_id": process_id
                        }
                        # Store in semantic memory
                        try:
                            semantic_result = self.memory.process_input(semantic_input)
                            results["semantic_memory"] = semantic_result
                        except Exception as e:
                            logging.error(f"Semantic memory error: {str(e)}")
                            results["semantic_memory_error"] = str(e)
            
            # Execute the memory operation
            memory_result = self.memory.process_input(memory_input)
            
        except Exception as e:
            logging.error(f"Memory error: {str(e)}")
            # Fallback to working memory in case of errors
            memory_result = {
                "operation": "store",
                "memory_type": "working",
                "status": "error",
                "error": str(e),
                "fallback": True
            }
            
            # Try to use working memory as fallback
            try:
                fallback_result = self.memory.process_input({
                    "operation": "store",
                    "memory_type": "working",
                    "content": {
                        "text": text,
                        "perception": perception_result,
                        "attention": attention_result,
                        "emotion": emotion_result if isinstance(emotion_result, dict) else {}
                    },
                    "process_id": process_id
                })
                memory_result.update(fallback_result)
                memory_result["status"] = "fallback_success"
            except Exception as nested_e:
                logging.error(f"Fallback memory error: {str(nested_e)}")
                memory_result["fallback_error"] = str(nested_e)
        
        memory_time = time.time() - memory_start
        results["memory"] = memory_result
        results["memory_time"] = memory_time
        
        # Step 5: Learning from the experience
        if self.development_level >= 0.5:
            learning_start = time.time()
            try:
                # Extract key information for learning
                stimulus = text
                
                # Get response from perception interpretation
                response = ""
                if perception_result.get("interpretation"):
                    interpretation = perception_result.get("interpretation", {})
                    response = json.dumps(interpretation)[:200]  # Limit length
                
                # Associative learning based on perception+attention
                associative_result = self.learning.process_input({
                    "learning_type": "associative",
                    "operation": "learn",
                    "stimulus": stimulus,
                    "response": response,
                    "strength": attention_result.get("salience", 0.5) if isinstance(attention_result, dict) else 0.5,
                    "process_id": process_id
                })
                
                learning_results = {
                    "associative": associative_result
                }
                
                # Add reinforcement learning at higher development
                if self.development_level >= 0.6:
                    # Use emotional response as reward signal
                    reward = 0.5  # Neutral default
                    if isinstance(emotion_result, dict) and "response" in emotion_result:
                        # Calculate reward from valence
                        valence = emotion_result["response"].get("valence", 0)
                        reward = (valence + 1) / 2  # Convert -1:1 to 0:1
                    
                    state = f"processing_{process_id}"
                    action = "analyze_text"
                    
                    reinforcement_result = self.learning.process_input({
                        "learning_type": "reinforcement",
                        "operation": "learn",
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "process_id": process_id
                    })
                    
                    learning_results["reinforcement"] = reinforcement_result
                
                # Add procedural learning at higher development
                if self.development_level >= 0.7:
                    # Learn procedural skill for processing text
                    skill_name = "text_processing"
                    steps = ["perceive", "attend", "emote", "memorize", "learn"]
                    
                    procedural_result = self.learning.process_input({
                        "learning_type": "procedural",
                        "operation": "learn_sequence",
                        "skill": skill_name,
                        "steps": steps,
                        "process_id": process_id
                    })
                    
                    # Practice the skill
                    practice_result = self.learning.process_input({
                        "learning_type": "procedural",
                        "operation": "practice",
                        "skill": skill_name,
                        "quality": attention_result.get("salience", 0.5) if isinstance(attention_result, dict) else 0.5,
                        "duration": 1.0,
                        "process_id": process_id
                    })
                    
                    learning_results["procedural"] = {
                        "learn": procedural_result,
                        "practice": practice_result
                    }
                
                # Add meta learning at higher development
                if self.development_level >= 0.8:
                    # Select learning strategy
                    content_type = perception_result.get("interpretation", {}).get("content_type", "general")
                    
                    meta_result = self.learning.process_input({
                        "learning_type": "meta",
                        "operation": "select_strategy",
                        "domain": "language",
                        "content_type": content_type,
                        "cognitive_resources": attention_result.get("salience", 0.7) if isinstance(attention_result, dict) else 0.7,
                        "process_id": process_id
                    })
                    
                    learning_results["meta"] = meta_result
                
                # Integrate all learning approaches
                if self.development_level >= 0.9:
                    integrate_result = self.learning.process_input({
                        "learning_type": "integrate",
                        "domain": "language",
                        "content_type": "text_processing",
                        "learning_types": ["associative", "reinforcement", "procedural", "meta"],
                        "primary_type": "associative",
                        "stimulus": stimulus,
                        "response": response,
                        "process_id": process_id
                    })
                    
                    learning_results["integrated"] = integrate_result
                
            except Exception as e:
                logging.error(f"Learning error: {str(e)}")
                learning_results = {
                    "status": "error",
                    "error": str(e)
                }
            
            learning_time = time.time() - learning_start
            results["learning"] = learning_results
            results["learning_time"] = learning_time
        else:
            results["learning"] = "Not yet developed"
        
        # Track overall processing time
        total_time = (
            perception_time +
            (attention_time if "attention_time" in results else 0) +
            (emotion_time if "emotion_time" in results else 0) +
            (memory_time if "memory_time" in results else 0) +
            (learning_time if "learning_time" in results else 0)
        )
        results["total_processing_time"] = total_time
        
        # Save result to history
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "process_id": process_id,
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "development_level": self.development_level,
            "total_time": total_time,
            "success": True
        })
        
        logging.info(f"Processing completed in {total_time:.3f} seconds")
        return results
        
    def set_development_level(self, level: float):
        """Set development level for all modules"""
        prev_level = self.development_level
        self.development_level = level
        
        # Update each module's development level
        self.perception.development_level = level
        self.perception._update_submodule_development()
        
        self.attention.development_level = level
        self.attention._adjust_parameters_for_development()
        
        self.memory.development_level = level
        self.memory._adjust_memory_for_development()
        
        # Only update emotion if development level is sufficient
        if level >= 0.3:
            self.emotion.development_level = level
            self.emotion._adjust_parameters_for_development()
        
        # Only update learning if development level is sufficient
        if level >= 0.5:
            self.learning.development_level = level
            if hasattr(self.learning, "update_development"):
                self.learning.update_development(level - prev_level)
        
        logging.info(f"Development level updated: {prev_level:.2f} → {level:.2f}")
        
    def retrieve_relevant_memories(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Retrieve memories relevant to the query
        
        Args:
            query: Search query
            limit: Maximum number of memories to retrieve
            
        Returns:
            Memory retrieval results
        """
        if self.development_level < 0.4:
            return {"status": "error", "message": "Memory retrieval not available at current development level"}
        
        process_id = f"retrieve_{int(time.time())}"
        
        try:
            # Determine memory type based on development level
            if self.development_level >= 0.7:
                # Use integrated memory search at higher levels
                result = self.memory.process_input({
                    "operation": "search",
                    "memory_type": "integrated",
                    "query": query,
                    "limit": limit,
                    "process_id": process_id
                })
            elif self.development_level >= 0.5:
                # Use episodic memory at mid levels
                result = self.memory.process_input({
                    "operation": "search",
                    "memory_type": "episodic",
                    "query": query,
                    "limit": limit,
                    "process_id": process_id
                })
            else:
                # Use working memory at basic levels
                result = self.memory.process_input({
                    "operation": "search",
                    "memory_type": "working",
                    "query": query,
                    "limit": limit,
                    "process_id": process_id
                })
                
            return result
            
        except Exception as e:
            logging.error(f"Memory retrieval error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }
    
    def regulate_emotion(self, target_valence: Optional[float] = None, 
                         target_arousal: Optional[float] = None) -> Dict[str, Any]:
        """
        Regulate emotional state
        
        Args:
            target_valence: Target valence value (-1 to 1)
            target_arousal: Target arousal value (0 to 1)
            
        Returns:
            Emotion regulation results
        """
        if self.development_level < 0.4:
            return {"status": "error", "message": "Emotion regulation not available at current development level"}
        
        process_id = f"regulate_{int(time.time())}"
        
        try:
            # Get current emotional state
            state_result = self.emotion.process_input({
                "operation": "query",
                "process_id": process_id
            })
            
            current_state = state_result.get("current_state", {})
            
            # Prepare regulation input
            regulation_input = {
                "operation": "regulate",
                "current_state": current_state,
                "process_id": process_id
            }
            
            # Add targets if specified
            if target_valence is not None:
                regulation_input["target_valence"] = target_valence
            
            if target_arousal is not None:
                regulation_input["target_arousal"] = target_arousal
            
            # Process regulation request
            result = self.emotion.process_input(regulation_input)
            
            return result
            
        except Exception as e:
            logging.error(f"Emotion regulation error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }
    
    def learn_association(self, stimulus: str, response: str, strength: float = 0.5) -> Dict[str, Any]:
        """
        Learn an association between stimulus and response
        
        Args:
            stimulus: The stimulus input
            response: The associated response
            strength: Association strength (0 to 1)
            
        Returns:
            Learning results
        """
        if self.development_level < 0.5:
            return {"status": "error", "message": "Learning not available at current development level"}
        
        process_id = f"learn_{int(time.time())}"
        
        try:
            result = self.learning.process_input({
                "learning_type": "associative",
                "operation": "learn",
                "stimulus": stimulus,
                "response": response,
                "strength": strength,
                "process_id": process_id
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Learning error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "process_id": process_id
            }
        
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get the current state of all cognitive modules"""
        state = {
            "development_level": self.development_level,
            "perception": self.perception.get_state(),
            "attention": self.attention.get_state(),
            "memory": self.memory.get_state()
        }
        
        # Add emotion state if developed enough
        if self.development_level >= 0.3:
            state["emotion"] = self.emotion.get_state()
            
        # Add learning state if developed enough
        if self.development_level >= 0.5:
            state["learning"] = self.learning.get_state()
            
        return state
    
    def save_state(self, filepath: str) -> bool:
        """
        Save system state to disk
        
        Args:
            filepath: Path to save the state file
            
        Returns:
            Success status
        """
        try:
            # Get full state and history
            full_state = {
                "development_level": self.development_level,
                "timestamp": datetime.now().isoformat(),
                "state_history": self.state_history
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(full_state, f, indent=2)
                
            logging.info(f"System state saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save system state: {str(e)}")
            return False

class EducationalScenarioTester:
    """
    Tests the cognitive system with an educational scenario
    
    This class simulates a learning scenario where the cognitive system
    is exposed to educational content about a particular subject.
    """
    
    def __init__(self, scenario_name: str, scenario_content: List[Dict[str, Any]]):
        """
        Initialize the scenario tester
        
        Args:
            scenario_name: Name of the educational scenario
            scenario_content: List of content items with text and metadata
        """
        self.scenario_name = scenario_name
        self.scenario_content = scenario_content
        self.results = []
        
    def run_test_at_level(self, level: float) -> Dict[str, Any]:
        """
        Run the scenario at a specific development level
        
        Args:
            level: Cognitive system development level (0.0 to 1.0)
            
        Returns:
            Test results
        """
        print_section(f"Testing {self.scenario_name} at Development Level {level:.1f}")
        
        # Initialize cognitive system at the specified level
        system = CognitiveSystem(development_level=level)
        
        # Print initial cognitive state
        print(f"{BOLD}Initial Cognitive State:{RESET}")
        initial_state = system.get_cognitive_state()
        print(f"Development Level: {initial_state['development_level']:.2f}")
        
        # Process each content item
        scenario_results = []
        for i, content_item in enumerate(self.scenario_content):
            item_type = content_item.get("type", "text")
            text = content_item.get("text", "")
            metadata = content_item.get("metadata", {})
            
            print_section(f"Processing Item {i+1}: {item_type}")
            print(f"{CYAN}Content:{RESET} {text[:100]}..." if len(text) > 100 else text)
            
            # Process the content through the cognitive system
            context = {
                "scenario": self.scenario_name,
                "item_number": i + 1,
                "item_type": item_type,
                "metadata": metadata
            }
            
            result = system.process_text(text, context)
            
            # Print key results based on development level
            self._print_result_summary(result, level)
            
            # Store result
            scenario_results.append({
                "item_number": i + 1,
                "item_type": item_type,
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "result": result
            })
            
            # Add a pause between items for clarity
            time.sleep(0.5)
        
        # After processing all items, test memory retrieval
        if level >= 0.4:
            print_section("Testing Memory Retrieval")
            
            # Generate a query based on the scenario
            query = f"information about {self.scenario_name}"
            print(f"Query: {query}")
            
            memory_result = system.retrieve_relevant_memories(query)
            
            if "items" in memory_result and memory_result["items"]:
                print(f"Retrieved {len(memory_result['items'])} memories:")
                for i, item in enumerate(memory_result["items"]):
                    if "text" in item:
                        print(f"  {i+1}. {item['text'][:100]}...")
                    elif "content" in item and "text" in item["content"]:
                        print(f"  {i+1}. {item['content']['text'][:100]}...")
            else:
                print("No memories retrieved")
                
            scenario_results.append({
                "item_type": "memory_retrieval",
                "query": query,
                "result": memory_result
            })
        
        # Test emotion regulation at higher levels
        if level >= 0.5:
            print_section("Testing Emotion Regulation")
            
            print("Regulating toward positive valence:")
            regulation_result = system.regulate_emotion(target_valence=0.5)
            
            if isinstance(regulation_result, dict):
                if "regulation_result" in regulation_result:
                    reg_result = regulation_result["regulation_result"]
                else:
                    reg_result = regulation_result
                    
                print(f"Regulation Strategy: {reg_result.get('regulation_strategy', 'unknown')}")
                print(f"Success Level: {reg_result.get('success_level', 0):.2f}")
                
                # Check for original and regulated states
                if "original_state" in reg_result and "regulated_state" in reg_result:
                    orig = reg_result["original_state"]
                    regulated = reg_result["regulated_state"]
                    
                    # Extract values considering both dict and object formats
                    if hasattr(orig, 'valence'):
                        orig_valence = orig.valence
                        orig_arousal = orig.arousal
                    else:
                        orig_valence = orig.get('valence', 0)
                        orig_arousal = orig.get('arousal', 0)
                        
                    if hasattr(regulated, 'valence'):
                        reg_valence = regulated.valence
                        reg_arousal = regulated.arousal
                    else:
                        reg_valence = regulated.get('valence', 0)
                        reg_arousal = regulated.get('arousal', 0)
                    
                    print(f"Valence: {orig_valence:.2f} → {reg_valence:.2f}")
                    print(f"Arousal: {orig_arousal:.2f} → {reg_arousal:.2f}")
            else:
                print(f"Regulation result: {regulation_result}")
                
            scenario_results.append({
                "item_type": "emotion_regulation",
                "target_valence": 0.5,
                "result": regulation_result
            })
        
        # Test learning capabilities at higher levels
        if level >= 0.6:
            print_section("Testing Learning Capabilities")
            
            # Test associative learning
            stimulus = f"{self.scenario_name} concepts"
            response = "educational information"
            
            print(f"Learning association: '{stimulus}' → '{response}'")
            learning_result = system.learn_association(stimulus, response, strength=0.7)
            
            if isinstance(learning_result, dict):
                print(f"Association ID: {learning_result.get('association_id', 'unknown')}")
                print(f"Strength: {learning_result.get('strength', 0):.2f}")
                print(f"Status: {learning_result.get('status', 'unknown')}")
            else:
                print(f"Learning result: {learning_result}")
                
            scenario_results.append({
                "item_type": "learning_test",
                "stimulus": stimulus,
                "response": response,
                "result": learning_result
            })
        
        # Final cognitive state
        print_section("Final Cognitive State")
        final_state = system.get_cognitive_state()
        
        print(f"Development Level: {final_state['development_level']:.2f}")
        print(f"Memory Usage:")
        
        # Print memory state if available
        if "memory" in final_state:
            memory_state = final_state["memory"]
            if "working_memory" in memory_state:
                working = memory_state["working_memory"]
                print(f"  Working Memory: {working.get('current_usage', 0)}/{working.get('capacity', 0)} items")
            
            if level >= 0.5 and "episodic_memory" in memory_state:
                episodic = memory_state["episodic_memory"]
                print(f"  Episodic Memory: {episodic.get('episode_count', 0)} episodes")
            
            if level >= 0.7 and "semantic_memory" in memory_state:
                semantic = memory_state["semantic_memory"]
                print(f"  Semantic Memory: {semantic.get('concept_count', 0)} concepts")
                
            if level >= 0.8 and "associative_memory" in memory_state:
                associative = memory_state["associative_memory"]
                print(f"  Associative Memory: {associative.get('association_count', 0)} associations")
        
        # Print emotion state if available
        if level >= 0.3 and "emotion" in final_state:
            emotion_state = final_state["emotion"]
            if "current_state" in emotion_state:
                current = emotion_state["current_state"]
                print(f"\nCurrent Emotional State:")
                print(f"  Dominant Emotion: {current.get('dominant_emotion', 'neutral')}")
                print(f"  Valence: {current.get('valence', 0):.2f}")
                print(f"  Arousal: {current.get('arousal', 0):.2f}")
        
        # Print learning state if available
        if level >= 0.5 and "learning" in final_state:
            learning_state = final_state["learning"]
            print(f"\nLearning State:")
            
            if "associative_learning" in learning_state:
                assoc = learning_state["associative_learning"]
                print(f"  Associative Pairs: {assoc.get('association_count', 0)}")
            
            if level >= 0.6 and "reinforcement_learning" in learning_state:
                reinf = learning_state["reinforcement_learning"]
                print(f"  Q-values: {reinf.get('q_value_count', 0)}")
            
            if level >= 0.7 and "procedural_learning" in learning_state:
                proc = learning_state["procedural_learning"]
                print(f"  Procedural Skills: {proc.get('skill_count', 0)}")
            
            if level >= 0.8 and "meta_learning" in learning_state:
                meta = learning_state["meta_learning"]
                print(f"  Learning Strategies: {meta.get('strategy_count', 0)}")
        
        # Save system state
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = os.path.join(results_dir, f"{self.scenario_name}_level_{level:.1f}_{timestamp}.json")
        system.save_state(state_path)
        
        # Compile all results
        test_result = {
            "scenario_name": self.scenario_name,
            "development_level": level,
            "timestamp": datetime.now().isoformat(),
            "scenario_results": scenario_results,
            "initial_state": initial_state,
            "final_state": final_state,
            "state_file": state_path
        }
        
        self.results.append(test_result)
        return test_result
    
    def _print_result_summary(self, result: Dict[str, Any], level: float):
        """Print a summary of processing results"""
        # Print perception results
        print(f"\n{BOLD}{CYAN}Perception Results:{RESET}")
        if "perception" in result and result["perception"] != "Not yet developed":
            patterns = result["perception"].get("patterns", [])
            if patterns:
                print(f"  Recognized {len(patterns)} patterns")
                for pattern in patterns[:2]:  # Show just a couple of patterns
                    print(f"    - {pattern['pattern_type']} (confidence: {pattern['confidence']:.2f})")
                if len(patterns) > 2:
                    print(f"    - ... ({len(patterns) - 2} more patterns)")
            else:
                print("  No patterns recognized")
                
            # Print interpretation if available
            if "interpretation" in result["perception"]:
                interp = result["perception"]["interpretation"]
                print(f"  Interpretation: {interp.get('content_type', 'unknown')} " +
                      f"(complexity: {interp.get('complexity', 'unknown')})")
        else:
            print("  Not yet developed")
        
        # Print attention results
        if level >= 0.1:
            print(f"\n{BOLD}{MAGENTA}Attention Results:{RESET}")
            if "attention" in result and result["attention"] != "Not yet developed":
                print(f"  Captures attention: {result['attention'].get('captures_attention', False)}")
                print(f"  Salience: {result['attention'].get('salience', 0):.2f}")
                
                if "current_focus" in result["attention"]:
                    focus = result["attention"]["current_focus"]
                    print(f"  Current focus: {focus.get('source', 'unknown')}")
                    print(f"  Focus intensity: {focus.get('intensity', 0):.2f}")
            else:
                print("  Not yet developed")
        
        # Print emotion results
        if level >= 0.3:
            print(f"\n{BOLD}{YELLOW}Emotion Results:{RESET}")
            if "emotion" in result and result["emotion"] != "Not yet developed":
                if isinstance(result["emotion"], dict) and "response" in result["emotion"]:
                    emotion = result["emotion"]["response"]
                    print(f"  Dominant emotion: {emotion.get('dominant_emotion', 'neutral')}")
                    print(f"  Valence: {emotion.get('valence', 0):.2f}")
                    print(f"  Arousal: {emotion.get('arousal', 0):.2f}")
                    
                    # Show top emotions if available
                    if "emotion_intensities" in emotion:
                        intensities = emotion["emotion_intensities"]
                        if intensities:
                            top_emotions = sorted(intensities.items(), key=lambda x: x[1], reverse=True)[:2]
                            print("  Top emotions:")
                            for emotion_name, intensity in top_emotions:
                                print(f"    - {emotion_name}: {intensity:.2f}")
                else:
                    print(f"  {result['emotion']}")
            else:
                print("  Not yet developed")
        
        # Print memory results
        if level >= 0.4:
            print(f"\n{BOLD}{GREEN}Memory Results:{RESET}")
            if "memory" in result and result["memory"] != "Not yet developed":
                memory = result["memory"]
                print(f"  Operation: {memory.get('operation', 'unknown')}")
                print(f"  Status: {memory.get('status', 'unknown')}")
                
                # Show storage location
                if "memory_type" in memory:
                    print(f"  Memory type: {memory['memory_type']}")
                
                # Show item or episode ID if available
                if "item_id" in memory:
                    print(f"  Item ID: {memory['item_id']}")
                elif "episode_id" in memory:
                    print(f"  Episode ID: {memory['episode_id']}")
                elif "concept_id" in memory:
                    print(f"  Concept ID: {memory['concept_id']}")
            else:
                print("  Not yet developed")
                
            # Show semantic memory results if available
            if "semantic_memory" in result:
                semantic = result["semantic_memory"]
                print(f"  Semantic memory: {semantic.get('status', 'unknown')}")
                if "concept_id" in semantic:
                    print(f"  Concept ID: {semantic['concept_id']}")
        
        # Print learning results
        if level >= 0.5:
            print(f"\n{BOLD}{BLUE}Learning Results:{RESET}")
            if "learning" in result and result["learning"] != "Not yet developed":
                learning = result["learning"]
                
                # Show associative learning results
                if isinstance(learning, dict) and "associative" in learning:
                    assoc = learning["associative"]
                    print(f"  Associative learning: {assoc.get('status', 'unknown')}")
                    if "association_id" in assoc:
                        print(f"  Association ID: {assoc['association_id']}")
                    if "strength" in assoc:
                        print(f"  Strength: {assoc['strength']:.2f}")
                
                # Show reinforcement learning if available
                if isinstance(learning, dict) and "reinforcement" in learning and level >= 0.6:
                    reinf = learning["reinforcement"]
                    print(f"  Reinforcement learning: {reinf.get('status', 'unknown')}")
                    if "updated_q" in reinf:
                        print(f"  Updated Q-value: {reinf['updated_q']:.2f}")
                
                # Show procedural learning if available
                if isinstance(learning, dict) and "procedural" in learning and level >= 0.7:
                    proc = learning["procedural"]
                    if isinstance(proc, dict) and "practice" in proc:
                        practice = proc["practice"]
                        print(f"  Procedural learning: {practice.get('status', 'unknown')}")
                        if "new_proficiency" in practice:
                            print(f"  New proficiency: {practice['new_proficiency']:.2f}")
                
                # Show meta learning if available
                if isinstance(learning, dict) and "meta" in learning and level >= 0.8:
                    meta = learning["meta"]
                    print(f"  Meta learning: {meta.get('status', 'unknown')}")
                    if "selected_strategy" in meta:
                        strategy = meta["selected_strategy"]
                        print(f"  Selected strategy: {strategy.get('name', 'unknown')}")
                
                # Show integrated learning if available
                if isinstance(learning, dict) and "integrated" in learning and level >= 0.9:
                    integrated = learning["integrated"]
                    print(f"  Integrated learning: {integrated.get('status', 'unknown')}")
                    if "integration_level" in integrated:
                        print(f"  Integration level: {integrated['integration_level']:.2f}")
            else:
                print("  Not yet developed")
                
        # Print processing times
        print(f"\n{BOLD}Processing Times:{RESET}")
        print(f"  Perception: {result.get('perception_time', 0):.3f}s")
        if level >= 0.1:
            print(f"  Attention: {result.get('attention_time', 0):.3f}s")
        if level >= 0.3:
            print(f"  Emotion: {result.get('emotion_time', 0):.3f}s")
        if level >= 0.4:
            print(f"  Memory: {result.get('memory_time', 0):.3f}s")
        if level >= 0.5:
            print(f"  Learning: {result.get('learning_time', 0):.3f}s")
        print(f"  Total: {result.get('total_processing_time', 0):.3f}s")
    
    def run_developmental_progression(self, levels: List[float] = None) -> Dict[str, Any]:
        """
        Run the scenario at multiple development levels to show progression
        
        Args:
            levels: List of development levels to test (default: [0.1, 0.3, 0.6, 0.9])
            
        Returns:
            Test results for all levels
        """
        if levels is None:
            levels = [0.1, 0.3, 0.6, 0.9]
            
        print_section(f"Testing Developmental Progression for {self.scenario_name}")
        print(f"Testing at levels: {', '.join([f'{l:.1f}' for l in levels])}")
        
        progression_results = []
        for level in levels:
            result = self.run_test_at_level(level)
            progression_results.append(result)
            
            # Add a pause between tests for clarity
            if level != levels[-1]:
                print("\nAdvancing to next development level...\n")
                time.sleep(1)
        
        # Summarize progression
        print_section("Developmental Progression Summary")
        print(f"Scenario: {self.scenario_name}")
        print(f"Development levels tested: {', '.join([f'{l:.1f}' for l in levels])}")
        
        # Show how capabilities increased with development
        print("\nCapability Progression:")
        
        for i, level in enumerate(levels):
            print(f"\nLevel {level:.1f}:")
            result = progression_results[i]
            final_state = result["final_state"]
            
            # Count patterns recognized
            total_patterns = 0
            for item_result in result["scenario_results"]:
                if "result" in item_result and "perception" in item_result["result"]:
                    perception = item_result["result"]["perception"]
                    if isinstance(perception, dict) and "patterns" in perception:
                        total_patterns += len(perception["patterns"])
            
            print(f"  - Patterns recognized: {total_patterns}")
            
            # Show memory capacity
            if "memory" in final_state:
                memory = final_state["memory"]
                if "working_memory" in memory:
                    print(f"  - Working memory capacity: {memory['working_memory'].get('capacity', 0)}")
                
                if level >= 0.5 and "episodic_memory" in memory:
                    print(f"  - Episodic memories: {memory['episodic_memory'].get('episode_count', 0)}")
                
                if level >= 0.7 and "semantic_memory" in memory:
                    print(f"  - Semantic concepts: {memory['semantic_memory'].get('concept_count', 0)}")
            
            # Show emotional complexity
            if level >= 0.3 and "emotion" in final_state:
                emotion = final_state["emotion"]
                if "emotional_capacity" in emotion:
                    print(f"  - Emotional complexity: {emotion['emotional_capacity'].get('emotional_complexity', 'basic')}")
            
            # Show learning capabilities
            if level >= 0.5 and "learning" in final_state:
                learning = final_state["learning"]
                if "associative_learning" in learning:
                    print(f"  - Associative pairs: {learning['associative_learning'].get('association_count', 0)}")
                
                if level >= 0.8 and "meta_learning" in learning:
                    print(f"  - Learning strategies: {learning['meta_learning'].get('strategy_count', 0)}")
        
        # Return all results
        return {
            "scenario_name": self.scenario_name,
            "levels_tested": levels,
            "progression_results": progression_results,
            "timestamp": datetime.now().isoformat()
        }

def create_astronomy_scenario():
    """Create an educational scenario about astronomy"""
    return EducationalScenarioTester(
        scenario_name="astronomy",
        scenario_content=[
            {
                "type": "introduction",
                "text": "Astronomy is the study of celestial objects such as stars, planets, comets, and galaxies, as well as phenomena that originate outside Earth's atmosphere.",
                "metadata": {
                    "subject": "astronomy",
                    "complexity": "basic"
                }
            },
            {
                "type": "factual",
                "text": "The Sun is a star at the center of our Solar System. It is approximately 4.6 billion years old and has a diameter of about 1.39 million kilometers.",
                "metadata": {
                    "subject": "astronomy",
                    "topic": "sun",
                    "complexity": "medium"
                }
            },
            {
                "type": "question",
                "text": "What is the difference between a planet and a dwarf planet?",
                "metadata": {
                    "subject": "astronomy",
                    "topic": "planets",
                    "complexity": "medium"
                }
            },
            {
                "type": "factual",
                "text": "Planets orbit the Sun, have sufficient mass to be rounded by their own gravity, and have cleared their neighboring region of other objects. Dwarf planets meet the first two criteria but not the third.",
                "metadata": {
                    "subject": "astronomy",
                    "topic": "planets",
                    "complexity": "high"
                }
            },
            {
                "type": "emotional",
                "text": "The vastness of space can fill us with awe and wonder! When we look at the night sky and contemplate the billions of stars and countless galaxies, we feel both incredibly small and deeply connected to the universe.",
                "metadata": {
                    "subject": "astronomy",
                    "topic": "philosophical",
                    "complexity": "high",
                    "emotional_content": "high"
                }
            },
            {
                "type": "conclusion",
                "text": "Astronomy helps us understand our place in the universe and how cosmic forces have shaped our world. It combines observational data with physics and mathematics to explain the nature and behavior of celestial objects.",
                "metadata": {
                    "subject": "astronomy",
                    "complexity": "medium"
                }
            }
        ]
    )

def create_music_scenario():
    """Create an educational scenario about music theory"""
    return EducationalScenarioTester(
        scenario_name="music_theory",
        scenario_content=[
            {
                "type": "introduction",
                "text": "Music theory is the study of how music works. It examines the language and notation of music and includes the study of elements such as rhythm, harmony, melody, and form.",
                "metadata": {
                    "subject": "music",
                    "complexity": "basic"
                }
            },
            {
                "type": "factual",
                "text": "Notes in Western music are named using the first seven letters of the alphabet: A, B, C, D, E, F, and G. After G, the sequence repeats at a higher pitch, forming an octave.",
                "metadata": {
                    "subject": "music",
                    "topic": "notes",
                    "complexity": "basic"
                }
            },
            {
                "type": "procedural",
                "text": "To build a major scale, follow this pattern of whole and half steps: whole, whole, half, whole, whole, whole, half. Starting from C, this gives us C, D, E, F, G, A, B, C.",
                "metadata": {
                    "subject": "music",
                    "topic": "scales",
                    "complexity": "medium"
                }
            },
            {
                "type": "factual",
                "text": "A chord is three or more notes played simultaneously. The most common chord is the triad, which consists of a root note, a third, and a fifth. Major triads have a major third and perfect fifth above the root.",
                "metadata": {
                    "subject": "music",
                    "topic": "chords",
                    "complexity": "medium"
                }
            },
            {
                "type": "emotional",
                "text": "Music has the remarkable power to evoke strong emotions! A minor chord might make us feel sad or contemplative, while a major chord often sounds happy or uplifting. This emotional connection is what makes music so meaningful across all cultures.",
                "metadata": {
                    "subject": "music",
                    "topic": "emotion",
                    "complexity": "medium",
                    "emotional_content": "high"
                }
            },
            {
                "type": "question",
                "text": "How does rhythm differ from meter in music?",
                "metadata": {
                    "subject": "music",
                    "topic": "rhythm",
                    "complexity": "high"
                }
            },
            {
                "type": "factual",
                "text": "Rhythm refers to the pattern of durations of notes and silences in music, while meter organizes these rhythms into regular groupings, indicated by a time signature. For example, 4/4 meter groups beats in sets of four.",
                "metadata": {
                    "subject": "music",
                    "topic": "rhythm",
                    "complexity": "high"
                }
            }
        ]
    )

def create_programming_scenario():
    """Create an educational scenario about computer programming"""
    return EducationalScenarioTester(
        scenario_name="programming",
        scenario_content=[
            {
                "type": "introduction",
                "text": "Computer programming is the process of designing and building executable computer programs to accomplish specific tasks. It involves analysis, algorithms, coding, testing, and maintenance.",
                "metadata": {
                    "subject": "computer science",
                    "complexity": "basic"
                }
            },
            {
                "type": "factual",
                "text": "Variables are named storage locations that contain data which can be modified during program execution. They are fundamental to almost all programming languages.",
                "metadata": {
                    "subject": "programming",
                    "topic": "variables",
                    "complexity": "basic"
                }
            },
            {
                "type": "procedural",
                "text": "To create a function in Python, use the 'def' keyword followed by a function name and parentheses. For example: def greet(name): return 'Hello, ' + name",
                "metadata": {
                    "subject": "programming",
                    "topic": "functions",
                    "complexity": "medium"
                }
            },
            {
                "type": "factual",
                "text": "Control structures like if-else statements, loops, and switches allow programs to make decisions and repeat actions. They control the flow of execution based on conditions.",
                "metadata": {
                    "subject": "programming",
                    "topic": "control flow",
                    "complexity": "medium"
                }
            },
            {
                "type": "question",
                "text": "What is the difference between a for loop and a while loop?",
                "metadata": {
                    "subject": "programming",
                    "topic": "loops",
                    "complexity": "medium"
                }
            },
            {
                "type": "factual",
                "text": "A for loop iterates over a sequence for a predetermined number of iterations, while a while loop continues as long as a specified condition remains true. Choose a for loop when you know the number of iterations in advance.",
                "metadata": {
                    "subject": "programming",
                    "topic": "loops",
                    "complexity": "medium"
                }
            },
            {
                "type": "factual",
                "text": "Object-Oriented Programming (OOP) is a paradigm based on 'objects' that contain data and code. The four main principles of OOP are encapsulation, abstraction, inheritance, and polymorphism.",
                "metadata": {
                    "subject": "programming",
                    "topic": "OOP",
                    "complexity": "high"
                }
            },
            {
                "type": "emotional",
                "text": "The joy of programming comes when your code finally works after hours of debugging! That moment of triumph when you solve a complex problem is incredibly satisfying and motivates programmers to take on even greater challenges.",
                "metadata": {
                    "subject": "programming",
                    "topic": "experience",
                    "complexity": "medium",
                    "emotional_content": "high"
                }
            }
        ]
    )

def test_cognitive_system_at_level(level: float):
    """Run a comprehensive test of the cognitive system at a specific level"""
    print_section(f"Comprehensive Cognitive System Test at Level {level:.1f}")
    
    # Create educational scenarios
    astronomy = create_astronomy_scenario()
    astronomy.run_test_at_level(level)
    
    # Add a pause between scenarios
    time.sleep(1)
    
    music = create_music_scenario()
    music.run_test_at_level(level)
    
    # Add a pause between scenarios
    time.sleep(1)
    
    programming = create_programming_scenario()
    programming.run_test_at_level(level)

def test_developmental_progression():
    """Test how the cognitive system develops through different levels"""
    print_section("Testing Developmental Progression of the Cognitive System")
    
    # Test with the programming scenario across multiple levels
    programming_scenario = create_programming_scenario()
    programming_scenario.run_developmental_progression([0.1, 0.3, 0.5, 0.7, 0.9])

def main():
    """Main test function"""
    print_section("Integrated Cognitive System Test Suite")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "progression":
            # Test developmental progression
            test_developmental_progression()
        elif sys.argv[1] == "level" and len(sys.argv) > 2:
            # Test at a specific level
            try:
                level = float(sys.argv[2])
                if 0.0 <= level <= 1.0:
                    test_cognitive_system_at_level(level)
                else:
                    print(f"{RED}Level must be between 0.0 and 1.0{RESET}")
            except ValueError:
                print(f"{RED}Invalid level format. Please specify a number between 0.0 and 1.0{RESET}")
        elif sys.argv[1] == "scenario" and len(sys.argv) > 2:
            # Test a specific scenario at default level (0.7)
            scenario_name = sys.argv[2].lower()
            level = 0.7
            if len(sys.argv) > 3:
                try:
                    level = float(sys.argv[3])
                except ValueError:
                    pass
                
            if scenario_name == "astronomy":
                astronomy = create_astronomy_scenario()
                astronomy.run_test_at_level(level)
            elif scenario_name == "music":
                music = create_music_scenario()
                music.run_test_at_level(level)
            elif scenario_name == "programming":
                programming = create_programming_scenario()
                programming.run_test_at_level(level)
            else:
                print(f"{RED}Unknown scenario: {scenario_name}. Available: astronomy, music, programming{RESET}")
        else:
            print(f"{YELLOW}Usage:{RESET}")
            print(f"  python test_cognitive_system.py progression")
            print(f"  python test_cognitive_system.py level <0.0-1.0>")
            print(f"  python test_cognitive_system.py scenario <name> [level]")
    else:
        # Default: test at mid-high level
        test_cognitive_system_at_level(0.7)
    
    print_section("Test Complete")
    
if __name__ == "__main__":
    main()