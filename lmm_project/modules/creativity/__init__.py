# Creativity module for the LMM system
# Responsible for creative generation, novelty detection, and concept combination

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
import uuid

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.creativity.models import CreativityState, CreativeOutput, CreativityMetrics
from lmm_project.modules.creativity.concept_combination import ConceptCombination
from lmm_project.modules.creativity.divergent_thinking import DivergentThinking
from lmm_project.modules.creativity.imagination import Imagination
from lmm_project.modules.creativity.novelty_detection import NoveltyDetection
from lmm_project.core.message import Message

def get_module(
    module_id: str = "creativity",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "CreativityModule":
    """
    Factory function to create and initialize a creativity module
    
    Args:
        module_id: Unique identifier for this module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level (0.0-1.0)
        
    Returns:
        Initialized creativity module
    """
    module = CreativityModule(module_id=module_id, event_bus=event_bus)
    module.set_development_level(development_level)
    return module

class CreativityModule(BaseModule):
    """
    Integrated creativity module responsible for generating novel and useful ideas
    
    This module combines several submodules for different aspects of creativity:
    - Novelty detection: identifying unusual or surprising elements
    - Concept combination: creating new ideas by combining existing concepts
    - Divergent thinking: generating multiple potential solutions
    - Imagination: generating novel mental scenarios and simulations
    
    The creativity module integrates these components and coordinates their
    development, providing unified creative capabilities for the LMM system.
    
    Developmental progression:
    - Basic novelty detection and simple associations in infancy
    - Expanding combinatorial play in early childhood
    - Structured creative problem-solving in later childhood
    - Metaphorical thinking and abstraction in adolescence
    - Integrated creative cognition with metacognitive awareness in adulthood
    """
    
    # Developmental milestones for creativity
    development_milestones = {
        0.0: "novelty_awareness",     # Basic novelty detection
        0.2: "associative_play",      # Simple combinations and associations
        0.4: "divergent_exploration", # Multiple solution generation
        0.6: "imaginative_scenarios", # Mental simulation capabilities  
        0.8: "metaphorical_thinking", # Abstract creative connections
        0.95: "creative_metacognition" # Self-aware creative processes
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the creativity module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="creativity", event_bus=event_bus)
        
        # Initialize state
        self.state = CreativityState()
        
        # Initialize component modules
        self.novelty_detection = NoveltyDetection(
            module_id=f"{module_id}_novelty_detection",
            event_bus=event_bus
        )
        
        self.concept_combination = ConceptCombination(
            module_id=f"{module_id}_concept_combination",
            event_bus=event_bus
        )
        
        self.divergent_thinking = DivergentThinking(
            module_id=f"{module_id}_divergent_thinking",
            event_bus=event_bus
        )
        
        self.imagination = Imagination(
            module_id=f"{module_id}_imagination",
            event_bus=event_bus
        )
        
        # Track component modules
        self.components = {
            "novelty_detection": self.novelty_detection,
            "concept_combination": self.concept_combination,
            "divergent_thinking": self.divergent_thinking,
            "imagination": self.imagination
        }
        
        # Initialize creativity metrics
        self.state.metrics = CreativityMetrics(
            fluency=0.0,
            flexibility=0.0,
            originality=0.0,
            elaboration=0.0,
            coherence=0.0,
            usefulness=0.0,
            last_updated=datetime.now()
        )
        
        # Initialize creative outputs history
        self.state.output_history = []
        self.max_history_size = 50
        
        # Subscribe to relevant events
        if self.event_bus:
            # Listen for creative outputs from components
            self.event_bus.subscribe("creative_output", self._handle_creative_output)
            
            # Listen for direct creativity requests
            self.event_bus.subscribe("creativity_request", self._handle_creativity_request)
            
            # Listen for evaluation requests
            self.event_bus.subscribe("creativity_evaluation", self._handle_evaluation_request)
            
            # Listen for developmental updates
            self.event_bus.subscribe("developmental_update", self._handle_developmental_update)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to generate creative outputs
        
        Args:
            input_data: Dictionary containing input to process creatively
            
        Returns:
            Dictionary with the results of creative processing
        """
        # Extract input information
        input_type = input_data.get("type", "general")
        content = input_data.get("content", {})
        mode = input_data.get("mode", "auto")  # auto, divergent, combination, imagination
        
        # Validate input
        if not content:
            return {
                "status": "error",
                "message": "Input content required",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
        
        # Generate a unique ID for this creative process
        process_id = str(uuid.uuid4())
        
        # Determine which creative processes to apply based on mode and developmental level
        processes = self._select_processes(mode, input_type)
        
        results = {
            "status": "success",
            "module_id": self.module_id,
            "module_type": self.module_type,
            "process_id": process_id,
            "outputs": []
        }
        
        # Apply selected creative processes
        for process_name in processes:
            if process_name in self.components:
                process_result = self.components[process_name].process_input({
                    "type": input_type,
                    "content": content,
                    "process_id": process_id
                })
                
                # Store process result
                results[process_name] = process_result
                
                # If process generated outputs, add them to the outputs list
                if "outputs" in process_result and isinstance(process_result["outputs"], list):
                    results["outputs"].extend(process_result["outputs"])
        
        # If no outputs were generated directly, create a unified output
        if not results["outputs"] and all(proc in results for proc in processes):
            unified_output = self._create_unified_output(process_id, input_type, content, results)
            if unified_output:
                results["outputs"] = [unified_output]
        
        # Update creativity metrics based on outputs
        if results["outputs"]:
            self._update_creativity_metrics(results["outputs"])
            
        return results
    
    def update_development(self, amount: float) -> float:
        """
        Update the module's developmental level
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # Update components with proportional development
        # Focus early development on novelty detection and basic associations
        # Later development focuses more on imagination and divergent thinking
        
        if new_level < 0.3:
            # Early development prioritizes novelty detection and concept combination
            novelty_amt = 0.5 * amount
            concept_amt = 0.3 * amount
            divergent_amt = 0.1 * amount
            imagination_amt = 0.1 * amount
        elif new_level < 0.6:
            # Middle development balances all components with focus on divergent thinking
            novelty_amt = 0.2 * amount
            concept_amt = 0.3 * amount
            divergent_amt = 0.3 * amount
            imagination_amt = 0.2 * amount
        else:
            # Late development focuses on imagination and divergent thinking
            novelty_amt = 0.1 * amount
            concept_amt = 0.2 * amount
            divergent_amt = 0.3 * amount
            imagination_amt = 0.4 * amount
            
        # Update each component
        self.novelty_detection.update_development(novelty_amt)
        self.concept_combination.update_development(concept_amt)
        self.divergent_thinking.update_development(divergent_amt)
        self.imagination.update_development(imagination_amt)
        
        return new_level
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_creativity"
        for level, name in sorted(self.development_milestones.items()):
            if self.development_level >= level:
                milestone = name
        return milestone
    
    def _select_processes(self, mode: str, input_type: str) -> List[str]:
        """
        Select which creative processes to apply based on mode and development
        
        Args:
            mode: Requested processing mode
            input_type: Type of input to process
            
        Returns:
            List of process names to apply
        """
        # If a specific mode is requested, use the corresponding process
        if mode == "divergent":
            return ["divergent_thinking"]
        elif mode == "combination":
            return ["concept_combination"]
        elif mode == "imagination":
            return ["imagination"]
        elif mode == "novelty":
            return ["novelty_detection"]
            
        # For auto mode, select based on development level and input type
        processes = []
        
        # Always include novelty detection for all development levels
        processes.append("novelty_detection")
        
        # Add processes based on developmental level
        if self.development_level >= 0.2:
            processes.append("concept_combination")
        
        if self.development_level >= 0.4:
            processes.append("divergent_thinking")
            
        if self.development_level >= 0.6:
            processes.append("imagination")
            
        # Adjust based on input type
        if input_type == "problem":
            # Prioritize divergent thinking for problems
            if "divergent_thinking" not in processes and self.development_level >= 0.3:
                processes.append("divergent_thinking")
        elif input_type == "concept":
            # Prioritize concept combination for concepts
            if "concept_combination" not in processes and self.development_level >= 0.1:
                processes.append("concept_combination")
        elif input_type == "scenario":
            # Prioritize imagination for scenarios
            if "imagination" not in processes and self.development_level >= 0.3:
                processes.append("imagination")
                
        return processes
    
    def _create_unified_output(
        self, 
        process_id: str, 
        input_type: str, 
        content: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a unified creative output from multiple process results
        
        Args:
            process_id: ID of the creative process
            input_type: Type of input processed
            content: Original input content
            results: Results from individual creative processes
            
        Returns:
            Unified creative output or None if not possible
        """
        # Extract outputs from each process
        novelty_result = results.get("novelty_detection", {})
        combination_result = results.get("concept_combination", {})
        divergent_result = results.get("divergent_thinking", {})
        imagination_result = results.get("imagination", {})
        
        # Create unified content based on available results
        unified_content = {
            "original_input": content,
            "process_id": process_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add process-specific elements to unified content
        if "novelty_score" in novelty_result:
            unified_content["novelty_score"] = novelty_result["novelty_score"]
            
        if "new_concepts" in combination_result:
            unified_content["combined_concepts"] = combination_result.get("new_concepts", [])
            
        if "solutions" in divergent_result:
            unified_content["solutions"] = divergent_result.get("solutions", [])
            
        if "scene" in imagination_result:
            unified_content["imagined_scene"] = imagination_result.get("scene", {})
        
        # Calculate aggregate scores
        novelty_score = novelty_result.get("novelty_score", 0.0)
        
        # Default other scores
        coherence_score = 0.5
        usefulness_score = 0.5
        
        # Adjust scores based on available data
        if "coherence" in combination_result:
            coherence_score = combination_result["coherence"]
        elif "coherence" in imagination_result:
            coherence_score = imagination_result["coherence"]
            
        if "usefulness" in divergent_result:
            usefulness_score = divergent_result["usefulness"]
        
        # Create creative output
        output = {
            "content": unified_content,
            "output_type": f"unified_{input_type}",
            "novelty_score": novelty_score,
            "coherence_score": coherence_score,
            "usefulness_score": usefulness_score,
            "source_components": [comp for comp in results.keys() if comp in self.components]
        }
        
        return output
    
    def _update_creativity_metrics(self, outputs: List[Dict[str, Any]]) -> None:
        """
        Update creativity metrics based on recent outputs
        
        Args:
            outputs: List of creative outputs to evaluate
        """
        if not outputs:
            return
            
        # Calculate metrics across all outputs
        fluency = len(outputs)  # Number of outputs
        
        # Extract scores
        novelty_scores = [output.get("novelty_score", 0.0) for output in outputs]
        coherence_scores = [output.get("coherence_score", 0.0) for output in outputs]
        usefulness_scores = [output.get("usefulness_score", 0.0) for output in outputs]
        
        # Calculate averages
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        avg_usefulness = sum(usefulness_scores) / len(usefulness_scores) if usefulness_scores else 0.0
        
        # Extract output types
        output_types = {output.get("output_type", "unknown") for output in outputs}
        flexibility = len(output_types) / 4.0  # Normalize by typical max types
        
        # Use novelty as originality approximation
        originality = avg_novelty
        
        # Calculate elaboration based on content complexity
        elaboration = 0.0
        for output in outputs:
            content = output.get("content", {})
            
            # Count the number of elements in the content
            if isinstance(content, dict):
                # More keys suggests more elaboration
                elaboration += min(1.0, len(content) / 10.0)  # Normalize to max of 1.0
        
        elaboration = elaboration / len(outputs) if outputs else 0.0
        
        # Update metrics
        self.state.metrics.fluency = min(1.0, fluency / 10.0)  # Normalize to max of 1.0
        self.state.metrics.flexibility = min(1.0, flexibility)
        self.state.metrics.originality = min(1.0, originality)
        self.state.metrics.elaboration = min(1.0, elaboration)
        self.state.metrics.coherence = min(1.0, avg_coherence)
        self.state.metrics.usefulness = min(1.0, avg_usefulness)
        self.state.metrics.last_updated = datetime.now()
        
        # Store outputs in history
        for output in outputs:
            # Add to history
            self.state.output_history.append({
                "timestamp": datetime.now().isoformat(),
                "output": output
            })
        
        # Limit history size
        if len(self.state.output_history) > self.max_history_size:
            self.state.output_history = self.state.output_history[-self.max_history_size:]
    
    def _handle_creative_output(self, message: Message) -> None:
        """
        Handle creative output messages from component modules
        
        Args:
            message: The creative output message
        """
        if isinstance(message.content, dict):
            # Store in history
            self.state.output_history.append({
                "timestamp": datetime.now().isoformat(),
                "output": message.content
            })
            
            # Limit history size
            if len(self.state.output_history) > self.max_history_size:
                self.state.output_history = self.state.output_history[-self.max_history_size:]
            
            # Update metrics (as a single output)
            self._update_creativity_metrics([message.content])
            
            # Forward to other modules if appropriate
            if self.event_bus and self.development_level > 0.5:
                # At higher development levels, we send creative outputs to working memory
                self.event_bus.publish(
                    Message(
                        sender="creativity",
                        message_type="working_memory_update",
                        content={
                            "type": "creative_output",
                            "content": message.content,
                            "priority": message.content.get("novelty_score", 0.5)
                        }
                    )
                )
    
    def _handle_creativity_request(self, message: Message) -> None:
        """
        Handle direct requests for creative processing
        
        Args:
            message: The creativity request message
        """
        if isinstance(message.content, dict):
            # Process the request
            result = self.process_input(message.content)
            
            # Publish the result
            if self.event_bus:
                self.event_bus.publish(
                    Message(
                        sender="creativity",
                        message_type="creativity_result",
                        content=result
                    )
                )
    
    def _handle_evaluation_request(self, message: Message) -> None:
        """
        Handle requests to evaluate the creativity of content
        
        Args:
            message: The evaluation request message
        """
        if isinstance(message.content, dict):
            content = message.content.get("content", {})
            
            if not content:
                # No content to evaluate
                if self.event_bus:
                    self.event_bus.publish(
                        Message(
                            sender="creativity",
                            message_type="creativity_evaluation_result",
                            content={
                                "status": "error",
                                "message": "No content to evaluate"
                            }
                        )
                    )
                return
                
            # Evaluate creativity by processing with novelty detection
            novelty_result = self.novelty_detection.process_input({
                "type": message.content.get("type", "general"),
                "content": content
            })
            
            # Create evaluation result
            evaluation = {
                "status": "success",
                "module_id": self.module_id,
                "module_type": self.module_type,
                "content_id": message.content.get("content_id", str(uuid.uuid4())),
                "novelty_score": novelty_result.get("novelty_score", 0.0),
                "surprise_level": novelty_result.get("surprise_level", 0.0),
                "creativity_score": novelty_result.get("novelty_score", 0.0) * 0.7 + 0.3  # Base creativity is 0.3 + novelty effect
            }
            
            # Apply developmental modulation
            if self.development_level < 0.3:
                # Early development: creativity is mostly novelty
                evaluation["creativity_score"] = novelty_result.get("novelty_score", 0.0) * 0.9 + 0.1
            elif self.development_level < 0.6:
                # Middle development: creativity balances novelty with baseline
                evaluation["creativity_score"] = novelty_result.get("novelty_score", 0.0) * 0.7 + 0.3
            else:
                # Late development: creativity is more sophisticated
                # We would use other factors here, but for simplicity we'll use the formula above
                pass
                
            # Publish the evaluation result
            if self.event_bus:
                self.event_bus.publish(
                    Message(
                        sender="creativity",
                        message_type="creativity_evaluation_result",
                        content=evaluation
                    )
                )
    
    def _handle_developmental_update(self, message: Message) -> None:
        """
        Handle developmental update messages
        
        Args:
            message: The developmental update message
        """
        if isinstance(message.content, dict):
            # Check if this is a global update or specifically for this module
            target_module = message.content.get("target_module")
            
            if target_module is None or target_module == self.module_id or target_module == "all":
                # Update development
                amount = message.content.get("amount", 0.01)
                self.update_development(amount)
