# TODO: Implement the NoveltyDetection class to identify unusual or surprising patterns
# This component should be able to:
# - Detect statistically unusual patterns in inputs
# - Identify violations of expectations
# - Recognize novelty in different domains (perceptual, conceptual, etc.)
# - Distinguish between degrees of novelty

# TODO: Implement developmental progression in novelty detection:
# - Simple statistical outlier detection in early stages
# - Basic expectation violation detection in childhood
# - Complex pattern novelty recognition in adolescence
# - Subtle novelty detection in adulthood

# TODO: Create mechanisms for:
# - Statistical novelty: Detect low-probability patterns
# - Expectation violation: Identify deviations from predictions
# - Conceptual novelty: Recognize unusual concept combinations
# - Contextual novelty: Detect appropriateness for context

# TODO: Implement novelty signals that:
# - Direct attention to novel stimuli
# - Trigger curiosity and exploration
# - Modulate learning rates for novel information
# - Contribute to emotional reactions (surprise, interest)

# TODO: Connect to attention, memory, and learning systems
# Novelty detection should guide attention, enhance memory formation,
# and influence learning rates for novel information

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.creativity.models import NoveltyDetectionState, CreativeOutput
from lmm_project.modules.creativity.neural_net import NoveltyDetector

class NoveltyDetection(BaseModule):
    """
    Identifies novel and surprising inputs
    
    This module detects novelty in inputs by comparing them
    to prior experiences and expectations, enabling the system
    to identify and attend to new and unexpected information.
    
    Developmental progression:
    - Basic feature mismatch detection in early stages
    - Statistical novelty detection in childhood 
    - Contextual surprise detection in adolescence
    - Abstract conceptual novelty recognition in adulthood
    """
    
    # Developmental milestones for novelty detection
    development_milestones = {
        0.0: "feature_mismatch",     # Basic detection of mismatched features
        0.25: "statistical_novelty", # Statistical deviation from baseline
        0.5: "contextual_surprise",  # Surprise in specific contexts
        0.75: "expectation_violation", # Violation of abstract expectations
        0.9: "conceptual_novelty"    # Novelty at the conceptual level
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the novelty detection module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="novelty_detection", event_bus=event_bus)
        
        # Initialize state
        self.state = NoveltyDetectionState()
        
        # Initialize neural network for novelty detection
        self.input_dim = 128  # Default dimension
        self.network = NoveltyDetector(
            input_dim=self.input_dim,
            hidden_dim=256,
            memory_size=100  # Store last 100 experiences
        )
        
        # Initialize novelty thresholds for different input types
        self.state.novelty_thresholds = {
            "perception": 0.7,  # Higher threshold for perceptual inputs
            "concept": 0.6,     # Medium threshold for conceptual inputs
            "emotion": 0.5,     # Lower threshold for emotional inputs
            "memory": 0.6,      # Medium threshold for memory inputs
            "default": 0.65     # Default threshold
        }
        
        # Initialize memory of recently processed inputs
        self.state.recently_processed = []  # Limited by max_memory_size
        self.max_memory_size = 50  # Maximum number of items to store
        
        # Initialize baseline distributions for different input types
        self.baselines = {
            "perception": {"mean": None, "std": None, "samples": []},
            "concept": {"mean": None, "std": None, "samples": []},
            "emotion": {"mean": None, "std": None, "samples": []},
            "memory": {"mean": None, "std": None, "samples": []}
        }
        
        # Maximum samples to store for each baseline
        self.max_baseline_samples = 100
        
        # Subscribe to relevant events
        if self.event_bus:
            self.event_bus.subscribe("perception_input", self._handle_input)
            self.event_bus.subscribe("concept_created", self._handle_input)
            self.event_bus.subscribe("emotion_state", self._handle_input)
            self.event_bus.subscribe("memory_retrieved", self._handle_input)
            self.event_bus.subscribe("novelty_query", self._handle_novelty_query)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to detect novelty
        
        Args:
            input_data: Dictionary containing input to evaluate for novelty
            
        Returns:
            Dictionary with the results of novelty detection
        """
        # Extract input information
        input_type = input_data.get("type", "default")
        content = input_data.get("content", {})
        input_id = input_data.get("input_id", str(uuid.uuid4()))
        update_baseline = input_data.get("update_baseline", True)
        context = input_data.get("context", {})
        
        # Validate input
        if not content:
            return {
                "status": "error",
                "message": "Input content required",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
            
        # Create input embedding (simplified - in a real system you would use actual embeddings)
        input_embedding = torch.randn(1, self.input_dim)
            
        # Detect novelty
        try:
            novelty_result = self._detect_novelty(input_embedding, content, input_type)
            
            # Get appropriate threshold for this input type
            threshold = self.state.novelty_thresholds.get(input_type, self.state.novelty_thresholds["default"])
            
            # Determine if the input is novel based on threshold
            is_novel = novelty_result["novelty_score"] > threshold
            
            # Update novelty scores dictionary
            self.state.novelty_scores[input_id] = novelty_result["novelty_score"]
            
            # Add to recently processed inputs
            self._update_recent_inputs(input_id, content, input_type, novelty_result["novelty_score"], is_novel)
            
            # Update baseline distribution if requested
            if update_baseline:
                self._update_baseline(input_embedding, input_type)
            
            # Create result
            result = {
                "status": "success",
                "module_id": self.module_id,
                "module_type": self.module_type,
                "input_id": input_id,
                "input_type": input_type,
                "novelty_score": novelty_result["novelty_score"],
                "is_novel": is_novel,
                "threshold": threshold,
                "surprise_level": self._calculate_surprise(novelty_result["novelty_score"], threshold)
            }
            
            # If input is novel, create and publish a creative output
            if is_novel and self.event_bus:
                creative_output = CreativeOutput(
                    content={
                        "input_id": input_id,
                        "input_type": input_type,
                        "content": content,
                        "novelty_details": {
                            "novelty_score": novelty_result["novelty_score"],
                            "threshold": threshold,
                            "surprise_level": result["surprise_level"]
                        }
                    },
                    output_type="novel_input",
                    novelty_score=novelty_result["novelty_score"],
                    coherence_score=0.5,  # Default coherence
                    usefulness_score=0.5,  # Default usefulness
                    source_components=[self.module_id]
                )
                
                self.event_bus.publish(
                    msg_type="creative_output",
                    content=creative_output.model_dump()
                )
                
                # Also publish a direct novelty alert
                self.event_bus.publish(
                    msg_type="novelty_alert",
                    content=result
                )
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error detecting novelty: {str(e)}",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
        
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.developmental_level
        new_level = super().update_development(amount)
        
        # Update novelty detection capabilities based on development
        
        # Adjust novelty thresholds (gradually lower thresholds to detect more subtle novelty)
        threshold_reduction = 0.2 * (new_level - previous_level)
        for input_type in self.state.novelty_thresholds:
            self.state.novelty_thresholds[input_type] = max(
                0.3,  # Minimum threshold
                self.state.novelty_thresholds[input_type] - threshold_reduction
            )
        
        # Adjust surprise sensitivity (increases with development)
        self.state.surprise_sensitivity = min(1.0, 0.3 + 0.7 * new_level)
        
        return new_level
    
    def _get_current_milestone(self) -> str:
        """Get the current developmental milestone"""
        milestone = "pre_novelty_detection"
        for level, name in sorted(self.development_milestones.items()):
            if self.developmental_level >= level:
                milestone = name
        return milestone
    
    def _detect_novelty(self, 
                       input_embedding: torch.Tensor, 
                       content: Dict[str, Any],
                       input_type: str) -> Dict[str, Any]:
        """
        Detect novelty in an input
        
        Args:
            input_embedding: Tensor embedding of the input
            content: The input content
            input_type: Type of input (perception, concept, etc.)
            
        Returns:
            Dictionary with novelty detection results
        """
        # Process through neural network to detect novelty
        with torch.no_grad():
            # Get appropriate update_memory setting based on developmental level
            # More developed systems are more selective about what they add to memory
            update_memory = np.random.random() < (1.0 - 0.5 * self.developmental_level)
            
            network_output = self.network(
                input_embedding,
                update_memory=update_memory
            )
            
        # Extract novelty score
        novelty_score = network_output["novelty_score"].item()
        
        # Apply developmental modulation to novelty detection
        if self.developmental_level < 0.25:
            # Basic feature novelty only at early stages
            novelty_score = self._modulate_early_novelty(novelty_score, content)
        elif self.developmental_level < 0.5:
            # Statistical novelty at intermediate stages
            novelty_score = self._apply_statistical_novelty(novelty_score, input_type)
        elif self.developmental_level < 0.75:
            # Contextual surprise at advanced stages
            context_modifier = np.random.uniform(0.8, 1.2)  # Simplified context effect
            novelty_score = min(1.0, novelty_score * context_modifier)
        else:
            # Conceptual novelty at highly developed stages
            # This might even reduce novelty of some inputs that are conceptually expected
            conceptual_modifier = 1.0 + 0.3 * (np.random.random() - 0.5)  # [-0.15, 0.15] adjustment
            novelty_score = max(0.0, min(1.0, novelty_score * conceptual_modifier))
        
        return {
            "novelty_score": novelty_score,
            "encoded": network_output["encoded"].cpu().numpy().tolist()
        }
    
    def _modulate_early_novelty(self, 
                              novelty_score: float, 
                              content: Dict[str, Any]) -> float:
        """Modulate novelty for early development stages focusing on features"""
        # Simplified feature-based novelty - in a real system this would analyze features
        # Here we'll just add some random variation
        feature_modifier = 0.2 * np.random.random()
        return min(1.0, novelty_score + feature_modifier)
    
    def _apply_statistical_novelty(self, 
                                 novelty_score: float, 
                                 input_type: str) -> float:
        """Apply statistical novelty detection based on baseline distributions"""
        baseline = self.baselines.get(input_type, self.baselines["perception"])
        
        # If we don't have enough samples for a baseline, return the raw score
        if baseline["mean"] is None or baseline["std"] is None:
            return novelty_score
            
        # Calculate z-score
        z_score = (novelty_score - baseline["mean"]) / max(0.01, baseline["std"])
        
        # Convert to probability using sigmoid
        probability = 1.0 / (1.0 + np.exp(-z_score))
        
        # Blend raw score with statistical score
        blended_score = 0.3 * novelty_score + 0.7 * probability
        
        return min(1.0, blended_score)
    
    def _update_baseline(self, 
                        input_embedding: torch.Tensor, 
                        input_type: str) -> None:
        """Update baseline distribution for an input type"""
        # Get relevant baseline
        baseline = self.baselines.get(input_type, self.baselines["perception"])
        
        # Extract novelty score from neural network
        with torch.no_grad():
            network_output = self.network(input_embedding, update_memory=False)
            novelty_score = network_output["novelty_score"].item()
            
        # Add to samples
        baseline["samples"].append(novelty_score)
        
        # Limit samples to max size
        if len(baseline["samples"]) > self.max_baseline_samples:
            baseline["samples"] = baseline["samples"][-self.max_baseline_samples:]
            
        # Update statistics
        baseline["mean"] = np.mean(baseline["samples"])
        baseline["std"] = np.std(baseline["samples"]) if len(baseline["samples"]) > 1 else 0.1
    
    def _update_recent_inputs(self, 
                            input_id: str, 
                            content: Dict[str, Any], 
                            input_type: str,
                            novelty_score: float,
                            is_novel: bool) -> None:
        """Update the record of recently processed inputs"""
        # Create input record
        input_record = {
            "input_id": input_id,
            "input_type": input_type,
            "timestamp": datetime.now().isoformat(),
            "novelty_score": novelty_score,
            "is_novel": is_novel,
            "content_summary": self._create_content_summary(content)
        }
        
        # Add to recently processed
        self.state.recently_processed.append(input_record)
        
        # Limit to max size
        if len(self.state.recently_processed) > self.max_memory_size:
            self.state.recently_processed = self.state.recently_processed[-self.max_memory_size:]
    
    def _create_content_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input content (to avoid storing everything)"""
        # In a real system, this would create a compact summary
        # Here we'll just do a simple filtering
        
        if not isinstance(content, dict):
            return {"type": str(type(content))}
            
        # Create a summary with a subset of keys
        summary = {}
        
        # Include a maximum of 5 keys
        for i, (key, value) in enumerate(content.items()):
            if i >= 5:
                break
                
            # Simplify values to strings if they're complex
            if isinstance(value, dict):
                summary[key] = "dict"
            elif isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            else:
                summary[key] = str(value)[:50]  # Truncate long strings
                
        return summary
    
    def _calculate_surprise(self, novelty_score: float, threshold: float) -> float:
        """Calculate surprise level based on novelty score and threshold"""
        # No surprise if below threshold
        if novelty_score <= threshold:
            return 0.0
            
        # Calculate how far above threshold
        surprise_factor = (novelty_score - threshold) / (1.0 - threshold)
        
        # Modulate by surprise sensitivity
        surprise_level = surprise_factor * self.state.surprise_sensitivity
        
        return min(1.0, surprise_level)
    
    def _handle_input(self, message: Message) -> None:
        """Handle various input messages for novelty detection"""
        # Process the input for novelty
        input_type = message.msg_type.split('_')[0]  # Extract first part of message type
        
        self.process_input({
            "type": input_type,
            "content": message.content,
            "input_id": str(uuid.uuid4())
        })
    
    def _handle_novelty_query(self, message: Message) -> None:
        """Handle explicit novelty query messages"""
        if isinstance(message.content, dict):
            # Process the query
            result = self.process_input(message.content)
            
            # Publish result
            if self.event_bus:
                self.event_bus.publish(
                    msg_type="novelty_result",
                    content=result,
                    source=self.module_id,
                    target=message.source
                )
