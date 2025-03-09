"""
Perception Module

This module integrates various components for processing and understanding
sensory input. It serves as the primary interface between the Mind and
external stimuli, converting text input into meaningful patterns and
features for higher cognitive processing.

For this LMM implementation, perception is text-based, as the system does
not have physical sensory organs like eyes or ears.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from dataclasses import dataclass, field
import logging
import uuid
import torch
from datetime import datetime

from lmm_project.core.event_bus import EventBus
from lmm_project.modules.base_module import BaseModule
from lmm_project.core.message import Message
from lmm_project.modules.perception.sensory_input import SensoryInputProcessor
from lmm_project.modules.perception.pattern_recognition import PatternRecognizer
from lmm_project.modules.perception.models import PerceptionResult, Pattern, SensoryInput, PerceptionParameters

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "perception",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "PerceptionSystem":
    """
    Factory function to create and return a perception module
    
    This function initializes and returns a complete perception system,
    with sensory input processing and pattern recognition capabilities.
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication
        development_level: Initial developmental level for the system
        
    Returns:
        Initialized PerceptionSystem
    """
    return PerceptionSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class PerceptionSystem(BaseModule):
    """
    Integrated perception system that processes sensory input and recognizes patterns
    
    The perception system develops progressively from basic sensory processing
    to sophisticated pattern detection and interpretation capabilities.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic sensory awareness",
        0.2: "Simple pattern recognition",
        0.4: "Feature integration",
        0.6: "Context-sensitive perception",
        0.8: "Advanced pattern recognition",
        1.0: "Sophisticated perception"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the perception system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="perception_system",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create sensory input processor
        self.sensory_processor = SensoryInputProcessor(
            module_id=f"{module_id}_sensory",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create pattern recognizer
        self.pattern_recognizer = PatternRecognizer(
            module_id=f"{module_id}_patterns",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Configuration parameters - adjusted for better pattern detection at all levels
        self.parameters = PerceptionParameters(
            token_sensitivity=0.8,        # Increased for better token detection
            ngram_sensitivity=0.7,        # Increased for better n-gram detection
            semantic_sensitivity=0.5,     # Increased for better semantic pattern detection
            novelty_threshold=0.4,        # Decreased to accept more patterns as novel
            pattern_activation_threshold=0.1,  # Significantly lowered to detect more patterns
            developmental_scaling=True
        )
        
        # Apply parameters to submodules
        self._apply_parameters()
        
        # Set developmental levels for submodules
        self._update_submodule_development()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("raw_text_input")
            self.subscribe_to_message("perception_query")
            
        # Try to use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Perception system initialized with device: {self.device}")
    
    def _apply_parameters(self):
        """Apply configuration parameters to submodules"""
        # Apply to pattern recognizer
        if hasattr(self.pattern_recognizer, 'token_sensitivity'):
            self.pattern_recognizer.token_sensitivity = self.parameters.token_sensitivity
        
        if hasattr(self.pattern_recognizer, 'ngram_sensitivity'):
            self.pattern_recognizer.ngram_sensitivity = self.parameters.ngram_sensitivity
            
        if hasattr(self.pattern_recognizer, 'semantic_sensitivity'):
            self.pattern_recognizer.semantic_sensitivity = self.parameters.semantic_sensitivity
            
        if hasattr(self.pattern_recognizer, 'novelty_threshold'):
            self.pattern_recognizer.novelty_threshold = self.parameters.novelty_threshold
            
        if hasattr(self.pattern_recognizer, 'pattern_activation_threshold'):
            self.pattern_recognizer.pattern_activation_threshold = self.parameters.pattern_activation_threshold
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensory input through the full perception pipeline
        
        Args:
            input_data: Raw sensory input data
            
        Returns:
            Dictionary with processed perception results
        """
        # Generate ID for this processing operation
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Log the incoming input processing
        text = input_data.get("text", "")
        if text:
            logger.info(f"Perception processing: '{text[:50]}...' (process {process_id})")
            logger.info(f"Input data before sensory processing: {list(input_data.keys())}")
        
        # Process through sensory input processor
        sensory_result = self.sensory_processor.process_input(input_data)
        
        # Debug logging to see what's happening with the data
        logger.info(f"Sensory result keys: {list(sensory_result.keys())}")
        
        # Ensure the text field is included in the data passed to the pattern recognizer
        # This fixes the issue where text wasn't being passed correctly
        if "text" not in sensory_result and text:
            logger.info(f"Adding missing text field to sensory_result")
            sensory_result["text"] = text
        
        # Double-check text field is set
        logger.info(f"Text field present before pattern recognition: {'text' in sensory_result}")
        if "text" in sensory_result:
            logger.info(f"Text value length: {len(sensory_result['text'])}")
        
        # Process through pattern recognizer
        pattern_result = self.pattern_recognizer.process_input(sensory_result)
        
        # Integrate results based on developmental level
        result = self._integrate_results(process_id, sensory_result, pattern_result)
        
        # Publish integrated results
        if self.event_bus:
            self.publish_message(
                "perception_result",
                {"result": result, "process_id": process_id}
            )
            
        return result
    
    def _integrate_results(
        self, 
        process_id: str,
        sensory_result: Dict[str, Any],
        pattern_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate results from sensory processing and pattern recognition
        
        The integration becomes more sophisticated with development
        
        Args:
            process_id: ID of the processing operation
            sensory_result: Results from sensory processing
            pattern_result: Results from pattern recognition
            
        Returns:
            Integrated perception result
        """
        # Basic integrated result
        result = {
            "process_id": process_id,
            "timestamp": datetime.now().isoformat(),
            "development_level": self.development_level,
            "module_id": self.module_id,
            "text": sensory_result.get("text", ""),
            "patterns": pattern_result.get("patterns", [])
        }
        
        # Add more detailed integration based on development level
        if self.development_level < 0.3:
            # Basic integration - just sensory features and patterns
            result["basic_features"] = sensory_result.get("basic_features", {})
            
        elif self.development_level < 0.6:
            # More integrated result with features
            result["basic_features"] = sensory_result.get("basic_features", {})
            result["features"] = sensory_result.get("features", {})
            result["recognized_pattern_types"] = list(set(
                p.get("pattern_type", "") for p in pattern_result.get("patterns", [])
            ))
            
        else:
            # Sophisticated integration with context and interpretation
            result["basic_features"] = sensory_result.get("basic_features", {})
            result["features"] = sensory_result.get("features", {})
            result["linguistic_features"] = sensory_result.get("linguistic_features", {})
            result["recognized_pattern_types"] = list(set(
                p.get("pattern_type", "") for p in pattern_result.get("patterns", [])
            ))
            
            # Add pattern interpretation
            result["interpretation"] = self._interpret_patterns(
                pattern_result.get("patterns", []),
                sensory_result
            )
            
        return result
    
    def _interpret_patterns(
        self, 
        patterns: List[Dict[str, Any]],
        sensory_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret the meaning of recognized patterns
        
        This is a higher-level function that becomes more sophisticated
        with development, providing meaningful interpretation of patterns.
        
        Args:
            patterns: List of recognized patterns
            sensory_result: Results from sensory processing
            
        Returns:
            Dictionary with interpretation of patterns
        """
        # Start with basic interpretation
        interpretation = {
            "primary_pattern": None,
            "content_type": "unknown",
            "complexity": "simple"
        }
        
        if not patterns:
            return interpretation
            
        # Count pattern types
        pattern_types = {}
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "unknown")
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
            
        # Find the most common pattern type
        primary_pattern_type = max(pattern_types.items(), key=lambda x: x[1])[0] if pattern_types else "unknown"
        interpretation["primary_pattern_type"] = primary_pattern_type
        
        # Find the highest confidence pattern
        highest_confidence_pattern = max(patterns, key=lambda p: p.get("confidence", 0))
        interpretation["primary_pattern"] = highest_confidence_pattern.get("pattern_id")
        
        # Determine complexity based on pattern count and types
        if len(patterns) > 10 and len(pattern_types) > 3:
            interpretation["complexity"] = "complex"
        elif len(patterns) > 5:
            interpretation["complexity"] = "moderate"
        else:
            interpretation["complexity"] = "simple"
            
        # Determine content type based on patterns
        if any(p.get("pattern_type") == "syntactic" and p.get("attributes", {}).get("pattern_type") == "question" for p in patterns):
            interpretation["content_type"] = "question"
        elif any(p.get("pattern_type") == "syntactic" and p.get("attributes", {}).get("pattern_type") == "exclamation" for p in patterns):
            interpretation["content_type"] = "exclamation"
        elif any(p.get("pattern_type") == "contextual" and p.get("attributes", {}).get("pattern_type") == "answer_to_question" for p in patterns):
            interpretation["content_type"] = "answer"
        elif any(p.get("pattern_type") == "neural" and p.get("confidence", 0) > 0.8 for p in patterns):
            interpretation["content_type"] = "familiar"
        elif any(p.get("novelty", 0) > 0.7 for p in patterns):
            interpretation["content_type"] = "novel"
        elif "?" in sensory_result.get("text", ""):
            interpretation["content_type"] = "question"
        elif "!" in sensory_result.get("text", ""):
            interpretation["content_type"] = "exclamation"
        elif len(sensory_result.get("text", "").split()) < 5:
            interpretation["content_type"] = "brief_statement"
        else:
            interpretation["content_type"] = "statement"
            
        # Calculate average novelty
        novelties = [p.get("novelty", 0.5) for p in patterns]
        interpretation["novelty_level"] = sum(novelties) / len(novelties) if novelties else 0.5
        
        # Add text properties from sensory processing
        features = sensory_result.get("features", {})
        if self.development_level >= 0.7 and features:
            # Extract interesting features
            interesting_features = {}
            
            # Token diversity
            if "token_diversity" in features:
                interesting_features["token_diversity"] = features["token_diversity"]
                
            # Unusual tokens
            if "linguistic_features" in sensory_result:
                ling_features = sensory_result["linguistic_features"]
                if "unusual_tokens" in ling_features:
                    interesting_features["unusual_words"] = ling_features["unusual_tokens"]
                    
            # Add to interpretation if we found interesting features
            if interesting_features:
                interpretation["text_features"] = interesting_features
                
        return interpretation
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the perception system
        
        This updates both the system's overall development level and the
        development levels of the subsystems (sensory processor and pattern recognizer)
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update submodule development levels
        self._update_submodule_development()
        
        # Adjust parameters based on development level
        if self.parameters.developmental_scaling:
            # Make the system increasingly sensitive as it develops
            self.parameters.token_sensitivity = min(0.9, 0.6 + new_level * 0.3)
            self.parameters.ngram_sensitivity = min(0.85, 0.5 + new_level * 0.35)
            self.parameters.semantic_sensitivity = min(0.8, 0.4 + new_level * 0.4)
            
            # Make threshold lower as system develops
            self.parameters.pattern_activation_threshold = max(0.05, 0.15 - new_level * 0.1)
            
            # Apply updated parameters
            self._apply_parameters()
        
        return new_level
    
    def _update_submodule_development(self):
        """Update the developmental level of submodules"""
        self.sensory_processor.update_development(self.development_level - self.sensory_processor.development_level)
        self.pattern_recognizer.update_development(self.development_level - self.pattern_recognizer.development_level)
    
    def _handle_message(self, message: Message):
        """
        Handle incoming messages
        
        Args:
            message: The message to handle
        """
        if message.message_type == "raw_text_input":
            # Process the raw text input
            if message.content:
                self.process_input(message.content)
        elif message.message_type == "perception_query":
            # Handle queries about perception
            self._handle_perception_query(message)
    
    def _handle_perception_query(self, message: Message):
        """
        Handle perception queries
        
        Args:
            message: Query message
        """
        if not message.content:
            logger.warning("Received empty perception query")
            return
            
        query_type = message.content.get("query_type", "")
        response = {"query_id": message.content.get("query_id", ""), "result": None}
        
        if query_type == "recent_patterns":
            # Return recent patterns
            count = message.content.get("count", 5)
            response["result"] = self.pattern_recognizer.get_recent_patterns(count)
            
        elif query_type == "recent_inputs":
            # Return recent inputs
            count = message.content.get("count", 5)
            response["result"] = self.sensory_processor.get_recent_inputs(count)
            
        elif query_type == "perception_state":
            # Return the current state of perception
            response["result"] = {
                "system_state": self.get_state(),
                "sensory_state": self.sensory_processor.get_state(),
                "pattern_state": self.pattern_recognizer.get_state()
            }
            
        elif query_type == "process_text":
            # Process a specific text
            text = message.content.get("text", "")
            if text:
                response["result"] = self.process_input({"text": text, "process_id": str(uuid.uuid4())})
                
        else:
            response["error"] = f"Unknown query type: {query_type}"
            
        # Publish the response
        if self.event_bus:
            self.publish_message(
                "perception_query_response",
                response
            )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the perception system
        
        Returns:
            Dictionary containing module state
        """
        # Get base state
        base_state = super().get_state()
        
        # Add perception-specific state
        state = {
            **base_state,
            "parameters": self.parameters.model_dump() if hasattr(self.parameters, "model_dump") else vars(self.parameters),
            "submodules": {
                "sensory_processor": {
                    "id": self.sensory_processor.module_id,
                    "development_level": self.sensory_processor.development_level
                },
                "pattern_recognizer": {
                    "id": self.pattern_recognizer.module_id,
                    "development_level": self.pattern_recognizer.development_level
                }
            },
            "device": str(self.device)
        }
        
        return state 
