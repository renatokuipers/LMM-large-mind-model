"""
Language module

This module is responsible for all aspects of language acquisition,
comprehension, and production within the LMM system.
"""

from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
from lmm_project.modules.language.models import LanguageModel, LanguageNeuralState

# Import all language submodules
from lmm_project.modules.language.phoneme_recognition import PhonemeRecognition
from lmm_project.modules.language.word_learning import WordLearning
from lmm_project.modules.language.grammar_acquisition import GrammarAcquisition
from lmm_project.modules.language.semantic_processing import SemanticProcessing
from lmm_project.modules.language.expression_generator import ExpressionGenerator

class LanguageSystem(BaseModule):
    """
    Integrated language system that brings together all language components
    
    The LanguageSystem coordinates phoneme recognition, word learning, 
    grammar acquisition, semantic processing, and expression generation 
    to create a complete language capability.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic sound recognition",
        0.2: "First words and simple understanding",
        0.4: "Growing vocabulary and basic grammar",
        0.6: "Sentence comprehension and production",
        0.8: "Complex grammar and contextual understanding",
        1.0: "Full language mastery"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the language system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize component modules
        self.phoneme_recognition = PhonemeRecognition(
            module_id=f"{module_id}.phoneme_recognition",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.word_learning = WordLearning(
            module_id=f"{module_id}.word_learning",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.grammar_acquisition = GrammarAcquisition(
            module_id=f"{module_id}.grammar_acquisition",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.semantic_processing = SemanticProcessing(
            module_id=f"{module_id}.semantic_processing",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.expression_generator = ExpressionGenerator(
            module_id=f"{module_id}.expression_generator",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize language model
        self.language_model = LanguageModel(
            phonemes=self.phoneme_recognition.phoneme_model,
            vocabulary=self.word_learning.word_model,
            grammar=self.grammar_acquisition.grammar_model,
            semantics=self.semantic_processing.semantic_model,
            expression=self.expression_generator.expression_model,
            developmental_level=development_level,
            module_id=module_id
        )
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.phoneme_recognition_development = development_level
        self.neural_state.word_learning_development = development_level
        self.neural_state.grammar_acquisition_development = development_level
        self.neural_state.semantic_processing_development = development_level
        self.neural_state.expression_generation_development = development_level
        
        # Register for event subscriptions
        if event_bus:
            event_bus.subscribe(
                sender=f"{module_id}.phoneme_recognition", 
                callback=self._handle_phoneme_event
            )
            event_bus.subscribe(
                sender=f"{module_id}.word_learning", 
                callback=self._handle_word_event
            )
            event_bus.subscribe(
                sender=f"{module_id}.grammar_acquisition", 
                callback=self._handle_grammar_event
            )
            event_bus.subscribe(
                sender=f"{module_id}.semantic_processing", 
                callback=self._handle_semantic_event
            )
            event_bus.subscribe(
                sender=f"{module_id}.expression_generator", 
                callback=self._handle_expression_event
            )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the language system
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dict with processing results
        """
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "status": "error",
                "message": "Input must be a dictionary"
            }
        
        # Extract process ID if provided
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract target component and operation
        component = input_data.get("component", "language")
        operation = input_data.get("operation", "")
        
        # Route to appropriate component or handle at system level
        if component == "phoneme_recognition":
            return self.phoneme_recognition.process_input(input_data)
            
        elif component == "word_learning":
            return self.word_learning.process_input(input_data)
            
        elif component == "grammar_acquisition":
            return self.grammar_acquisition.process_input(input_data)
            
        elif component == "semantic_processing":
            return self.semantic_processing.process_input(input_data)
            
        elif component == "expression_generator":
            return self.expression_generator.process_input(input_data)
            
        elif component == "language":
            # Handle language-level operations
            if operation == "comprehend":
                return self._comprehend_input(input_data, process_id)
                
            elif operation == "produce":
                return self._produce_output(input_data, process_id)
                
            elif operation == "get_state":
                return self._get_language_state(input_data, process_id)
                
            else:
                return {
                    "status": "error",
                    "message": f"Unknown operation for language component: {operation}",
                    "process_id": process_id
                }
                
        else:
            return {
                "status": "error",
                "message": f"Unknown component: {component}",
                "process_id": process_id
            }
    
    def _comprehend_input(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Comprehend language input by coordinating across components
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with comprehension results
        """
        # Process depends on input type
        input_type = input_data.get("input_type", "text")
        
        if input_type == "audio":
            # Handle audio input (speech)
            if "audio_features" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing audio_features for audio input",
                    "process_id": process_id
                }
                
            # First process with phoneme recognition
            phoneme_result = self.phoneme_recognition.process_input({
                "operation": "recognize",
                "audio_features": input_data["audio_features"],
                "process_id": process_id
            })
            
            if phoneme_result["status"] != "success":
                return phoneme_result
                
            # Convert phonemes to words
            word_result = self.word_learning.process_input({
                "operation": "recognize",
                "phoneme_sequence": [p[0] for p in phoneme_result["recognized_phonemes"]],
                "process_id": process_id
            })
            
            if word_result["status"] != "success" or not word_result.get("recognized", False):
                return word_result
                
            # Now we have a word or sequence of words to process
            if "recognized_word" in word_result:
                text = word_result["recognized_word"]
            else:
                # Construct from phoneme sequence as fallback
                text = "".join([p[0] for p in phoneme_result["recognized_phonemes"]])
                
        elif input_type == "text":
            # Handle text input directly
            if "text" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing text for text input",
                    "process_id": process_id
                }
                
            text = input_data["text"]
            
        else:
            return {
                "status": "error",
                "message": f"Unknown input_type: {input_type}",
                "process_id": process_id
            }
        
        # Now process the text through grammar and semantics
        
        # Analyze grammar
        grammar_result = self.grammar_acquisition.process_input({
            "operation": "analyze",
            "sentence": text,
            "process_id": process_id
        })
        
        # Extract meaning
        semantic_result = self.semantic_processing.process_input({
            "operation": "understand",
            "text": text,
            "process_id": process_id
        })
        
        # Integrate results
        comprehension = {
            "status": "success",
            "text": text,
            "grammar": {
                "pattern": grammar_result.get("detected_pattern", ""),
                "grammatical": grammar_result.get("grammatical", False)
            },
            "semantics": {
                "concepts": [c["concept"] for c in semantic_result.get("relevant_concepts", [])],
                "depth": semantic_result.get("understanding_depth", 0.0)
            },
            "development_level": self.development_level,
            "process_id": process_id
        }
        
        return comprehension
    
    def _produce_output(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Produce language output by coordinating across components
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with production results
        """
        # Check for intent
        if "intent" not in input_data:
            return {
                "status": "error",
                "message": "Missing intent for language production",
                "process_id": process_id
            }
            
        intent = input_data["intent"]
        context = input_data.get("context", {})
        concepts = input_data.get("concepts", [])
        
        # If concepts are provided, translate to semantic representation
        if concepts:
            # Enrich context with concept information
            for concept in concepts:
                if isinstance(concept, str) and concept not in context:
                    # Query semantic information
                    concept_info = self.semantic_processing.process_input({
                        "operation": "query_semantics",
                        "query_type": "concept",
                        "concept": concept,
                        "process_id": process_id
                    })
                    
                    # Add to context if found
                    if concept_info["status"] == "success":
                        context[concept_info.get("category", "object")] = concept
        
        # Generate expression
        expression_result = self.expression_generator.process_input({
            "operation": "generate",
            "intent": intent,
            "context": context,
            "process_id": process_id
        })
        
        # Check grammaticality if development level permits
        if self.development_level >= 0.4 and "expression" in expression_result:
            grammar_result = self.grammar_acquisition.process_input({
                "operation": "check_grammar",
                "sentence": expression_result["expression"],
                "process_id": process_id
            })
            
            # Add grammar check results
            if grammar_result["status"] == "success":
                expression_result["grammatical"] = grammar_result.get("grammatical", False)
                expression_result["grammaticality_score"] = grammar_result.get("grammaticality_score", 0.0)
        
        return expression_result
    
    def _get_language_state(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Get the current language state
        
        Args:
            input_data: Input data dictionary
            process_id: Process identifier
            
        Returns:
            Dict with language state information
        """
        # Get component states
        phoneme_state = self.phoneme_recognition.get_state()
        word_state = self.word_learning.get_state()
        grammar_state = self.grammar_acquisition.get_state()
        semantic_state = self.semantic_processing.get_state()
        expression_state = self.expression_generator.get_state()
        
        # Update language model component levels
        self.language_model.component_levels = {
            "phoneme_recognition": self.phoneme_recognition.development_level,
            "word_learning": self.word_learning.development_level,
            "grammar_acquisition": self.grammar_acquisition.development_level,
            "semantic_processing": self.semantic_processing.development_level,
            "expression_generation": self.expression_generator.development_level
        }
        
        # Overall language capability assessment
        # This weighs different components based on development stage
        if self.development_level < 0.3:
            # Early stage - phonemes and words most important
            phoneme_weight = 0.5
            word_weight = 0.3
            grammar_weight = 0.1
            semantic_weight = 0.05
            expression_weight = 0.05
        elif self.development_level < 0.6:
            # Middle stage - words and grammar most important
            phoneme_weight = 0.2
            word_weight = 0.3
            grammar_weight = 0.3
            semantic_weight = 0.1
            expression_weight = 0.1
        else:
            # Advanced stage - semantics and expression most important
            phoneme_weight = 0.1
            word_weight = 0.2
            grammar_weight = 0.2
            semantic_weight = 0.25
            expression_weight = 0.25
            
        # Calculate weighted language capability
        capability = (
            phoneme_weight * self.phoneme_recognition.development_level +
            word_weight * self.word_learning.development_level +
            grammar_weight * self.grammar_acquisition.development_level +
            semantic_weight * self.semantic_processing.development_level +
            expression_weight * self.expression_generator.development_level
        )
        
        # Return language state
        return {
            "status": "success",
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "language_capability": capability,
            "components": {
                "phoneme_recognition": phoneme_state,
                "word_learning": word_state,
                "grammar_acquisition": grammar_state,
                "semantic_processing": semantic_state,
                "expression_generator": expression_state
            },
            "process_id": process_id
        }
    
    def _handle_phoneme_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the phoneme recognition component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in phoneme development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update language development tracking
            self._check_language_milestones()
            
    def _handle_word_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the word learning component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in word learning development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update language development tracking
            self._check_language_milestones()
            
    def _handle_grammar_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the grammar acquisition component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in grammar development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update language development tracking
            self._check_language_milestones()
            
    def _handle_semantic_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the semantic processing component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in semantic development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update language development tracking
            self._check_language_milestones()
            
    def _handle_expression_event(self, event: Dict[str, Any]) -> None:
        """Handle events from the expression generator component"""
        message_type = event.get("message_type", "")
        
        if message_type == "development_milestone":
            # A milestone was reached in expression development
            content = event.get("content", {})
            level = content.get("level", 0.0)
            
            # Update language development tracking
            self._check_language_milestones()
    
    def _check_language_milestones(self):
        """Check if any language-level milestones have been reached"""
        # Language development is influenced by component development
        component_levels = [
            self.phoneme_recognition.development_level,
            self.word_learning.development_level,
            self.grammar_acquisition.development_level,
            self.semantic_processing.development_level,
            self.expression_generator.development_level
        ]
        
        # Average component development
        avg_component_level = sum(component_levels) / len(component_levels)
        
        # Get previous development level
        old_level = self.development_level
        
        # Update overall development level (weighted average of components and existing level)
        self.development_level = (self.development_level * 0.3) + (avg_component_level * 0.7)
        self.development_level = max(0.0, min(1.0, self.development_level))
        
        # Check if crossed a milestone
        for level in sorted(self.development_milestones.keys()):
            if old_level < level <= self.development_level:
                milestone = self.development_milestones[level]
                
                # Publish milestone event if we have an event bus
                if self.event_bus:
                    self.event_bus.publish({
                        "sender": self.module_id,
                        "message_type": "development_milestone",
                        "content": {
                            "module": "language",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                print(f"Language Development Milestone: {milestone} (level {level})")
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module and its components
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        
        # Update component development levels
        self.phoneme_recognition.update_development(amount)
        self.word_learning.update_development(amount)
        self.grammar_acquisition.update_development(amount)
        self.semantic_processing.update_development(amount)
        self.expression_generator.update_development(amount)
        
        # Check language milestones
        self._check_language_milestones()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the language system
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "components": {
                "phoneme_recognition": self.phoneme_recognition.get_state(),
                "word_learning": self.word_learning.get_state(),
                "grammar_acquisition": self.grammar_acquisition.get_state(),
                "semantic_processing": self.semantic_processing.get_state(),
                "expression_generator": self.expression_generator.get_state()
            }
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "language_model": self.language_model.dict(),
            "components": {
                "phoneme_recognition": self.phoneme_recognition.save_state(),
                "word_learning": self.word_learning.save_state(),
                "grammar_acquisition": self.grammar_acquisition.save_state(),
                "semantic_processing": self.semantic_processing.save_state(),
                "expression_generator": self.expression_generator.save_state()
            },
            "neural_state": {
                "phoneme_recognition": {
                    "development": self.neural_state.phoneme_recognition_development,
                    "accuracy": self.neural_state.phoneme_recognition_accuracy
                },
                "word_learning": {
                    "development": self.neural_state.word_learning_development,
                    "accuracy": self.neural_state.word_learning_accuracy
                },
                "grammar_acquisition": {
                    "development": self.neural_state.grammar_acquisition_development,
                    "accuracy": self.neural_state.grammar_acquisition_accuracy
                },
                "semantic_processing": {
                    "development": self.neural_state.semantic_processing_development,
                    "accuracy": self.neural_state.semantic_processing_accuracy
                },
                "expression_generation": {
                    "development": self.neural_state.expression_generation_development,
                    "accuracy": self.neural_state.expression_generation_accuracy
                }
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state to load
        """
        # Load module ID
        self.module_id = state["module_id"]
        
        # Load development level
        self.development_level = state["developmental_level"]
        
        # Load language model
        if "language_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.language_model = parse_obj_as(LanguageModel, state["language_model"])
            except Exception as e:
                print(f"Error loading language model: {e}")
        
        # Load component states
        if "components" in state:
            components = state["components"]
            
            if "phoneme_recognition" in components:
                self.phoneme_recognition.load_state(components["phoneme_recognition"])
                
            if "word_learning" in components:
                self.word_learning.load_state(components["word_learning"])
                
            if "grammar_acquisition" in components:
                self.grammar_acquisition.load_state(components["grammar_acquisition"])
                
            if "semantic_processing" in components:
                self.semantic_processing.load_state(components["semantic_processing"])
                
            if "expression_generator" in components:
                self.expression_generator.load_state(components["expression_generator"])
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            
            if "phoneme_recognition" in ns:
                self.neural_state.phoneme_recognition_development = ns["phoneme_recognition"].get("development", self.development_level)
                self.neural_state.phoneme_recognition_accuracy = ns["phoneme_recognition"].get("accuracy", 0.5)
                
            if "word_learning" in ns:
                self.neural_state.word_learning_development = ns["word_learning"].get("development", self.development_level)
                self.neural_state.word_learning_accuracy = ns["word_learning"].get("accuracy", 0.5)
                
            if "grammar_acquisition" in ns:
                self.neural_state.grammar_acquisition_development = ns["grammar_acquisition"].get("development", self.development_level)
                self.neural_state.grammar_acquisition_accuracy = ns["grammar_acquisition"].get("accuracy", 0.5)
                
            if "semantic_processing" in ns:
                self.neural_state.semantic_processing_development = ns["semantic_processing"].get("development", self.development_level)
                self.neural_state.semantic_processing_accuracy = ns["semantic_processing"].get("accuracy", 0.5)
                
            if "expression_generation" in ns:
                self.neural_state.expression_generation_development = ns["expression_generation"].get("development", self.development_level)
                self.neural_state.expression_generation_accuracy = ns["expression_generation"].get("accuracy", 0.5)


def get_module(module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0) -> Union[LanguageSystem, BaseModule]:
    """
    Factory function to create a language module
    
    This function is responsible for creating a language system that can:
    - Comprehend language input (written or spoken)
    - Produce appropriate language output
    - Acquire new language skills through experience
    - Process semantic meaning from language
    - Connect language to concepts and experiences
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level (0.0 to 1.0)
        
    Returns:
        An instance of the LanguageSystem class or a specific component
    """
    # Check if requesting a specific component or the full system
    module_parts = module_id.split('.')
    
    if len(module_parts) > 1 and module_parts[0] == "language":
        component = module_parts[1]
        
        # Return specific component
        if component == "phoneme_recognition":
            return PhonemeRecognition(module_id, event_bus, development_level)
            
        elif component == "word_learning":
            return WordLearning(module_id, event_bus, development_level)
            
        elif component == "grammar_acquisition":
            return GrammarAcquisition(module_id, event_bus, development_level)
            
        elif component == "semantic_processing":
            return SemanticProcessing(module_id, event_bus, development_level)
            
        elif component == "expression_generator":
            return ExpressionGenerator(module_id, event_bus, development_level)
            
        else:
            # Unknown component, return full system
            print(f"Unknown language component '{component}', returning full language system")
            return LanguageSystem(module_id, event_bus, development_level)
    
    # Return full language system
    return LanguageSystem(module_id, event_bus, development_level)
