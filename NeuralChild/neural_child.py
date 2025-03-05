"""
Neural Child module for the NeuralChild project.

This module defines the NeuralChild class, which is the central component that
coordinates all psychological functions and manages the child's development
through interactions with the mother and environment.
"""

from typing import Dict, List, Any, Optional, Set, Union, Callable
import time
import random
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict

from pydantic import BaseModel, Field

from .components.base import NeuralComponent, ConnectionType
from .components.emotion import EmotionComponent
from .components.language import LanguageComponent
from .components.memory import MemoryComponent, MemoryType
from .components.consciousness import ConsciousnessComponent
from .core.mother import Mother, ChildInput, ChildInputType, MotherResponse
from .llm_module import LLMClient
from .config import DevelopmentalStage, CONFIG


class DevelopmentalMetrics(BaseModel):
    """Metrics tracking developmental progress."""
    age_days: int = 0
    developmental_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL
    component_confidence: Dict[str, float] = Field(default_factory=dict)
    vocabulary_size: int = 0
    emotional_stability: float = 0.0
    memory_capacity: int = 0
    self_awareness: float = 0.0
    social_awareness: float = 0.0
    cognitive_complexity: float = 0.0
    language_complexity: float = 0.0
    interaction_count: int = 0
    training_time_seconds: float = 0
    
    class Config:
        arbitrary_types_allowed = True


class InteractionHistory(BaseModel):
    """Record of interactions between mother and child."""
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 100
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_interaction(self, child_input: ChildInput, mother_response: MotherResponse) -> None:
        """
        Add an interaction to the history.
        
        Args:
            child_input: Input from the child
            mother_response: Response from the mother
        """
        interaction = {
            "timestamp": time.time(),
            "child_input": child_input.dict(),
            "mother_response": mother_response.dict()
        }
        
        self.interactions.append(interaction)
        
        # Trim if needed
        if len(self.interactions) > self.max_history:
            self.interactions = self.interactions[-self.max_history:]


class InternalState(BaseModel):
    """Internal state of the neural child."""
    age_days: int = 0
    developmental_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL
    start_time: float = Field(default_factory=time.time)
    last_interaction_time: Optional[float] = None
    component_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    environment_state: Dict[str, Any] = Field(default_factory=dict)
    stimuli: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class NeuralChild:
    """
    The central class representing the neural child with an integrated mind.
    
    This class coordinates all psychological components, handles interactions
    with the mother, manages development, and maintains the child's state.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mother: Optional[Mother] = None,
        load_state_path: Optional[str] = None
    ):
        """
        Initialize the neural child.
        
        Args:
            config: Optional configuration override
            mother: Optional mother instance (created if not provided)
            load_state_path: Optional path to load state from
        """
        # Initialize configuration
        self.config = config or CONFIG
        
        # Initialize state
        self.internal_state = InternalState()
        
        # Initialize components
        self.components: Dict[str, NeuralComponent] = {}
        self._initialize_components()
        
        # Initialize mother
        self.mother = mother or Mother()
        
        # Initialize developmental metrics
        self.metrics = DevelopmentalMetrics()
        
        # Initialize interaction history
        self.interaction_history = InteractionHistory()
        
        # Initialize development parameters
        self.development_params = {
            "time_acceleration": self.config.time.acceleration_factor,
            "stage_duration_days": {stage: 365 for stage in self.config.development.stage_thresholds.keys()},
            "learning_rate_multiplier": self.config.development.learning_rate_multipliers,  # Note plural form
            "variability_factor": self.config.development.development_variation,
            "stage_transition_threshold": 0.7  # Default threshold
        }
        
        # Load existing state if provided
        if load_state_path:
            self.load_state(load_state_path)
        
        # Initialize connections between components
        self._initialize_connections()
    
    def _initialize_components(self) -> None:
        """Initialize all psychological components."""
        # Create components
        self.components["Emotion"] = EmotionComponent(
            development_stage=self.internal_state.developmental_stage
        )
        
        self.components["Language"] = LanguageComponent(
            development_stage=self.internal_state.developmental_stage
        )
        
        self.components["Memory"] = MemoryComponent(
            development_stage=self.internal_state.developmental_stage
        )
        
        self.components["Consciousness"] = ConsciousnessComponent(
            development_stage=self.internal_state.developmental_stage
        )
        
        # Add additional components here when implemented
        # self.components["Perception"] = PerceptionComponent(
        #     development_stage=self.internal_state.developmental_stage
        # )
        
        # self.components["Social"] = SocialComponent(
        #     development_stage=self.internal_state.developmental_stage
        # )
        
        # Initialize component states
        for name, component in self.components.items():
            self.internal_state.component_states[name] = {
                "activation": component.activation,
                "confidence": component.confidence
            }
    
    def _initialize_connections(self) -> None:
        """Initialize connections between components."""
        # Define connections
        connections = [
            # Emotion receives input from Language
            (self.components["Emotion"], self.components["Language"], ConnectionType.FEEDFORWARD),
            
            # Consciousness receives input from all components
            (self.components["Consciousness"], self.components["Emotion"], ConnectionType.FEEDFORWARD),
            (self.components["Consciousness"], self.components["Language"], ConnectionType.FEEDFORWARD),
            (self.components["Consciousness"], self.components["Memory"], ConnectionType.FEEDFORWARD),
            
            # Memory receives input from Emotion and Language
            (self.components["Memory"], self.components["Emotion"], ConnectionType.FEEDFORWARD),
            (self.components["Memory"], self.components["Language"], ConnectionType.FEEDFORWARD),
            
            # Language receives input from Emotion
            (self.components["Language"], self.components["Emotion"], ConnectionType.FEEDFORWARD),
            
            # Add more connections as needed
        ]
        
        # Establish connections
        for source, target, connection_type in connections:
            source.connect_to(target, connection_type)
    
    def _update_developmental_stage(self) -> None:
        """Update the developmental stage based on age and metrics."""
        # Get current stage and age
        current_stage = self.internal_state.developmental_stage
        age_days = self.internal_state.age_days
        
        # Get duration for current stage
        stage_duration = self.development_params["stage_duration_days"].get(
            current_stage, 
            365  # Default: 1 year
        )
        
        # Get confidence values for all components
        component_confidences = [component.confidence for component in self.components.values()]
        avg_confidence = sum(component_confidences) / len(component_confidences) if component_confidences else 0
        
        # Determine if stage transition should occur
        # Transition occurs when both age threshold and confidence threshold are met
        next_stage_index = current_stage.value + 1
        
        if (age_days >= stage_duration and 
                avg_confidence >= self.development_params["stage_transition_threshold"] and
                next_stage_index < len(DevelopmentalStage)):
            
            # Transition to next stage
            next_stage = DevelopmentalStage(next_stage_index)
            
            # Update internal state
            self.internal_state.developmental_stage = next_stage
            
            # Update all components
            for component in self.components.values():
                component.set_development_stage(next_stage)
            
            # Log transition
            print(f"Development stage transition: {current_stage} -> {next_stage} at age {age_days} days")
    
    def _update_age(self, elapsed_seconds: float) -> None:
        """
        Update the age based on elapsed time.
        
        Args:
            elapsed_seconds: Real seconds elapsed
        """
        # Apply time acceleration
        simulated_seconds = elapsed_seconds * self.development_params["time_acceleration"]
        
        # Convert to days (86400 seconds per day)
        simulated_days = simulated_seconds / 86400
        
        # Update age
        self.internal_state.age_days += simulated_days
        self.metrics.age_days = int(self.internal_state.age_days)
        
        # Update training time
        self.metrics.training_time_seconds += elapsed_seconds
        
        # Check for developmental stage transitions
        self._update_developmental_stage()
    
    def _process_all_components(self) -> None:
        """Process all psychological components based on current state."""
        # Process components in a specific order
        component_order = [
            "Emotion",  # Process emotions first
            "Memory",   # Process memory next
            "Language", # Process language after
            "Consciousness"  # Process consciousness last (integrates others)
        ]
        
        # Add any missing components to the end
        for component_name in self.components.keys():
            if component_name not in component_order:
                component_order.append(component_name)
        
        # Process each component
        for component_name in component_order:
            if component_name in self.components:
                component = self.components[component_name]
                
                # Prepare inputs
                inputs = self._prepare_component_inputs(component_name)
                
                # Process component
                try:
                    result = component.process(inputs)
                    
                    # Update component state
                    self.internal_state.component_states[component_name] = {
                        "activation": component.activation,
                        "confidence": component.confidence,
                        **result
                    }
                except Exception as e:
                    print(f"Error processing component {component_name}: {e}")
        
        # Update metrics from component states
        self._update_metrics_from_components()
    
    def _prepare_component_inputs(self, component_name: str) -> Dict[str, Any]:
        """
        Prepare inputs for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary of inputs for the component
        """
        # Start with common inputs
        common_inputs = {
            "age_days": self.internal_state.age_days,
            "developmental_stage": self.internal_state.developmental_stage,
            "stimuli": self.internal_state.stimuli
        }
        
        # Add component-specific inputs
        if component_name == "Emotion":
            # Emotion needs inputs from sensory processing and language
            inputs = {
                **common_inputs,
                "text": self.internal_state.stimuli.get("text", ""),
                "context": {"source": "mother"} if "text" in self.internal_state.stimuli else {}
            }
            
            # Add emotional trigger if available
            if "emotional_trigger" in self.internal_state.stimuli:
                inputs["triggers"] = [self.internal_state.stimuli["emotional_trigger"]]
            
        elif component_name == "Language":
            # Language needs text input and emotional state
            inputs = {
                **common_inputs,
                "text": self.internal_state.stimuli.get("text", ""),
                "emotional_state": self.internal_state.component_states.get("Emotion", {}).get("emotional_state", {}),
                "context": {"source": "mother"} if "text" in self.internal_state.stimuli else {}
            }
            
        elif component_name == "Memory":
            # Memory needs to know the operation
            inputs = {
                **common_inputs,
                "operation": "encode",  # Default operation
                "data": {
                    "text": self.internal_state.stimuli.get("text", ""),
                    "context": self.internal_state.stimuli.get("context", {})
                }
            }
            
            # If in interaction, add to memory
            if "text" in self.internal_state.stimuli:
                inputs["operation"] = "encode"
                
                # Add emotional valence if available
                if "Emotion" in self.internal_state.component_states:
                    emotion_state = self.internal_state.component_states["Emotion"]
                    if "emotional_state" in emotion_state and "primary_emotion" in emotion_state:
                        inputs["data"]["emotional_context"] = {
                            "emotion": emotion_state["primary_emotion"],
                            "valence": emotion_state["emotional_state"].get("dimensions", {}).get("valence", 0)
                        }
            
        elif component_name == "Consciousness":
            # Consciousness needs all component states and stimuli
            inputs = {
                **common_inputs,
                "component_states": self.internal_state.component_states,
                "current_time": time.time()
            }
            
        # Default case - just pass common inputs
        else:
            inputs = common_inputs
        
        return inputs
    
    def _update_metrics_from_components(self) -> None:
        """Update developmental metrics based on component states."""
        # Update component confidence
        for name, component in self.components.items():
            self.metrics.component_confidence[name] = component.confidence
        
        # Update vocabulary size from language component
        if "Language" in self.internal_state.component_states:
            language_state = self.internal_state.component_states["Language"]
            self.metrics.vocabulary_size = language_state.get("vocabulary_size", 0)
            
            # Update language complexity
            if "metrics" in language_state:
                lang_metrics = language_state["metrics"]
                self.metrics.language_complexity = (
                    lang_metrics.get("grammar_complexity", 0) * 0.6 +
                    (lang_metrics.get("mlu", 0) / 10) * 0.4  # Normalize MLU
                )
        
        # Update emotional stability from emotion component
        if "Emotion" in self.internal_state.component_states:
            emotion_state = self.internal_state.component_states["Emotion"]
            if "emotional_state" in emotion_state:
                self.metrics.emotional_stability = emotion_state["emotional_state"].get("stability", 0.0)
        
        # Update memory capacity from memory component
        if "Memory" in self.internal_state.component_states:
            memory_state = self.internal_state.component_states["Memory"]
            if "stats" in memory_state:
                self.metrics.memory_capacity = memory_state["stats"].get("episodic_memory_count", 0)
        
        # Update awareness metrics from consciousness component
        if "Consciousness" in self.internal_state.component_states:
            consciousness_state = self.internal_state.component_states["Consciousness"]
            if "awareness_state" in consciousness_state:
                awareness = consciousness_state["awareness_state"]
                
                # Map awareness levels to numeric values
                awareness_level_map = {
                    "unconscious": 0.0,
                    "minimal": 0.2,
                    "basic": 0.4,
                    "intermediate": 0.6,
                    "advanced": 0.8,
                    "reflective": 1.0
                }
                
                # Update self-awareness
                self_awareness_level = awareness.get("self_awareness", "unconscious")
                self.metrics.self_awareness = awareness_level_map.get(self_awareness_level, 0.0)
                
                # Update social awareness
                social_awareness_level = awareness.get("social_awareness", "unconscious")
                self.metrics.social_awareness = awareness_level_map.get(social_awareness_level, 0.0)
                
                # Update cognitive complexity based on metacognitive awareness
                metacognitive_level = awareness.get("metacognitive_awareness", "unconscious")
                self.metrics.cognitive_complexity = awareness_level_map.get(metacognitive_level, 0.0)
    
    def _generate_child_response(self) -> ChildInput:
        """
        Generate a response to the mother based on internal state.
        
        Returns:
            ChildInput object representing the child's response
        """
        # Determine developmental stage-appropriate response
        stage = self.internal_state.developmental_stage
        
        # Default response type based on stage
        response_type_by_stage = {
            DevelopmentalStage.PRENATAL: ChildInputType.NON_VERBAL,
            DevelopmentalStage.INFANCY: ChildInputType.BABBLE,
            DevelopmentalStage.EARLY_CHILDHOOD: ChildInputType.SINGLE_WORD,
            DevelopmentalStage.MIDDLE_CHILDHOOD: ChildInputType.SIMPLE_PHRASE,
            DevelopmentalStage.ADOLESCENCE: ChildInputType.COMPLEX_SENTENCE,
            DevelopmentalStage.EARLY_ADULTHOOD: ChildInputType.COMPLEX_SENTENCE,
            DevelopmentalStage.MID_ADULTHOOD: ChildInputType.COMPLEX_SENTENCE
        }
        
        input_type = response_type_by_stage.get(stage, ChildInputType.NON_VERBAL)
        
        # Get language production if available
        content = ""
        if "Language" in self.internal_state.component_states:
            language_state = self.internal_state.component_states["Language"]
            
            if "production" in language_state:
                production = language_state["production"]
                
                if production and "utterance" in production and production["utterance"]:
                    content = production["utterance"]
                    
                    # Update input type based on production
                    if production.get("structure_used") == "babbling":
                        input_type = ChildInputType.BABBLE
                    elif production.get("structure_used") == "single_word":
                        input_type = ChildInputType.SINGLE_WORD
                    elif production.get("structure_used") == "two_word":
                        input_type = ChildInputType.SIMPLE_PHRASE
                    elif production.get("structure_used") in ["simple_sentence", "questions", "negative"]:
                        input_type = ChildInputType.COMPLEX_SENTENCE
        
        # If no language production, generate based on stage
        if not content:
            if stage == DevelopmentalStage.PRENATAL:
                content = ""  # No response
            elif stage == DevelopmentalStage.INFANCY:
                # Simple babbling
                vowels = ['a', 'e', 'i', 'o', 'u']
                consonants = ['b', 'm', 'p', 'd', 't']
                
                # Generate random babbling
                syllables = []
                for _ in range(random.randint(1, 3)):
                    consonant = random.choice(consonants)
                    vowel = random.choice(vowels)
                    syllables.append(f"{consonant}{vowel}")
                
                content = "-".join(syllables)
                input_type = ChildInputType.BABBLE
            else:
                # Simple default responses
                content = "..."  # Default for other stages
        
        # Get emotional state if available
        emotional_state = None
        if "Emotion" in self.internal_state.component_states:
            emotion_state = self.internal_state.component_states["Emotion"]
            if "primary_emotion" in emotion_state:
                emotional_state = emotion_state["primary_emotion"]
        
        # Create context for response
        context = {}
        
        # Add consciousness state if available
        if "Consciousness" in self.internal_state.component_states:
            consciousness_state = self.internal_state.component_states["Consciousness"]
            if "attention_focus" in consciousness_state:
                context["attention_focus"] = consciousness_state["attention_focus"]
        
        # Create child input
        child_input = ChildInput(
            content=content,
            input_type=input_type,
            developmental_stage=stage.value,
            emotional_state=emotional_state,
            context=context
        )
        
        return child_input
    
    def _process_mother_response(self, response: MotherResponse) -> Dict[str, Any]:
        """
        Process a response from the mother.
        
        Args:
            response: MotherResponse object
            
        Returns:
            Dictionary of processed stimuli
        """
        # Extract verbal response
        verbal_response = response.verbal_response
        
        # Basic stimuli from response
        stimuli = {
            "text": verbal_response,
            "source": "mother",
            "context": {
                "emotional_state": response.emotional_state,
                "non_verbal_cues": response.non_verbal_cues
            }
        }
        
        # Create emotional trigger based on mother's state
        if hasattr(self.components.get("Emotion"), "create_emotion_trigger"):
            emotional_trigger = self.components["Emotion"].create_emotion_trigger(
                source="mother",
                input_text=verbal_response,
                intensity=0.7,  # Mother's influence is strong
                context={"relationship": "mother", "tone": response.emotional_state}
            )
            
            stimuli["emotional_trigger"] = emotional_trigger
        
        # Add teaching elements if available
        if response.teaching_elements:
            teaching_concepts = []
            
            for element in response.teaching_elements:
                teaching_concepts.append({
                    "concept": element.concept,
                    "explanation": element.explanation,
                    "examples": element.examples
                })
            
            stimuli["teaching_elements"] = teaching_concepts
        
        return stimuli
    
    def interact_with_mother(self, mother_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Interact with the mother.
        
        Args:
            mother_input: Optional direct input from mother (if None, mother will respond to child)
            
        Returns:
            Dictionary with interaction results
        """
        # Record start time
        start_time = time.time()
        
        # If mother input provided, create response directly
        if mother_input:
            # Create simulated mother response
            mother_response = MotherResponse(
                verbal_response=mother_input,
                emotional_state="neutral",
                non_verbal_cues=["smile"],
                teaching_elements=None
            )
            
            # Process the response
            stimuli = self._process_mother_response(mother_response)
            
            # Update internal state
            self.internal_state.stimuli = stimuli
            
            # Process all components
            self._process_all_components()
            
            # Generate child response
            child_input = self._generate_child_response()
            
            # Add to interaction history
            self.interaction_history.add_interaction(child_input, mother_response)
            
        else:
            # Generate child input based on current state
            child_input = self._generate_child_response()
            
            # Get response from mother
            mother_response = self.mother.respond_to_child(child_input)
            
            # Process the response
            stimuli = self._process_mother_response(mother_response)
            
            # Update internal state
            self.internal_state.stimuli = stimuli
            
            # Process all components
            self._process_all_components()
            
            # Add to interaction history
            self.interaction_history.add_interaction(child_input, mother_response)
        
        # Update last interaction time
        self.internal_state.last_interaction_time = time.time()
        
        # Update age and metrics
        elapsed_seconds = time.time() - start_time
        self._update_age(elapsed_seconds)
        
        # Update interaction count
        self.metrics.interaction_count += 1
        
        # Return interaction results
        return {
            "child_input": child_input.dict(),
            "mother_response": mother_response.dict() if isinstance(mother_response, MotherResponse) else mother_response,
            "updated_metrics": self.metrics.dict(),
            "component_states": {name: state for name, state in self.internal_state.component_states.items()}
        }
    
    def simulate_development(
        self, 
        interactions: int, 
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Simulate development through multiple interactions.
        
        Args:
            interactions: Number of interactions to simulate
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with simulation results
        """
        results = []
        
        for i in range(interactions):
            # Perform interaction
            interaction_result = self.interact_with_mother()
            results.append(interaction_result)
            
            # Call callback if provided
            if callback:
                callback({
                    "interaction": i + 1,
                    "total": interactions,
                    "metrics": self.metrics.dict(),
                    "stage": self.internal_state.developmental_stage.value
                })
        
        return {
            "interactions_completed": interactions,
            "final_metrics": self.metrics.dict(),
            "developmental_stage": self.internal_state.developmental_stage.value,
            "simulation_time": self.metrics.training_time_seconds
        }
    
    def process_external_stimulus(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an external stimulus.
        
        Args:
            stimulus: Dictionary describing the stimulus
            
        Returns:
            Dictionary with processing results
        """
        # Record start time
        start_time = time.time()
        
        # Update stimuli
        self.internal_state.stimuli = stimulus
        
        # Process all components
        self._process_all_components()
        
        # Update age and metrics
        elapsed_seconds = time.time() - start_time
        self._update_age(elapsed_seconds)
        
        # Generate response
        response = self._generate_child_response()
        
        return {
            "stimulus": stimulus,
            "response": response.dict(),
            "updated_metrics": self.metrics.dict(),
            "component_states": {name: state for name, state in self.internal_state.component_states.items()}
        }
    
    def save_state(self, path: str) -> bool:
        """
        Save the current state to a file.
        
        Args:
            path: Path to save state to
            
        Returns:
            Whether save was successful
        """
        try:
            # Create state dictionary
            state = {
                "neural_child": {
                    "age_days": self.internal_state.age_days,
                    "developmental_stage": self.internal_state.developmental_stage.value,
                    "metrics": self.metrics.dict(),
                    "interaction_history": [i for i in self.interaction_history.interactions[-10:]],  # Last 10 only
                    "timestamp": time.time()
                },
                "components": {}
            }
            
            # Save component states
            for name, component in self.components.items():
                state["components"][name] = {
                    "activation": component.activation,
                    "confidence": component.confidence,
                    "state": component.__dict__  # This is a simplification - in real implementation, would need proper serialization
                }
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, path: str) -> bool:
        """
        Load state from a file.
        
        Args:
            path: Path to load state from
            
        Returns:
            Whether load was successful
        """
        try:
            # Load from file
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Update neural child state
            if "neural_child" in state:
                child_state = state["neural_child"]
                self.internal_state.age_days = child_state.get("age_days", 0)
                
                # Set developmental stage
                stage_value = child_state.get("developmental_stage", 0)
                if isinstance(stage_value, int) and 0 <= stage_value < len(DevelopmentalStage):
                    self.internal_state.developmental_stage = DevelopmentalStage(stage_value)
                
                # Update metrics
                if "metrics" in child_state:
                    # This is a simplification - in real implementation, would need proper deserialization
                    for key, value in child_state["metrics"].items():
                        if hasattr(self.metrics, key):
                            setattr(self.metrics, key, value)
            
            # Update component states - this is a simplification
            # In real implementation, would need proper deserialization for each component
            if "components" in state:
                for name, component_state in state["components"].items():
                    if name in self.components:
                        if "activation" in component_state:
                            self.components[name].activation = component_state["activation"]
                        if "confidence" in component_state:
                            self.components[name].confidence = component_state["confidence"]
            
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def get_developmental_metrics(self) -> Dict[str, Any]:
        """
        Get current developmental metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.dict()
    
    def get_component_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all components.
        
        Returns:
            Dictionary mapping component names to states
        """
        return {name: component.__dict__ for name, component in self.components.items()}
    
    def get_cognitive_snapshot(self) -> Dict[str, Any]:
        """
        Get a detailed snapshot of current cognitive state.
        
        Returns:
            Dictionary with cognitive snapshot
        """
        # Build snapshot from component states
        snapshot = {
            "age_days": self.internal_state.age_days,
            "developmental_stage": self.internal_state.developmental_stage.value,
            "components": {}
        }
        
        # Add relevant state from each component
        if "Emotion" in self.internal_state.component_states:
            emotion_state = self.internal_state.component_states["Emotion"]
            snapshot["components"]["emotion"] = {
                "primary_emotion": emotion_state.get("primary_emotion"),
                "emotional_state": emotion_state.get("emotional_state", {})
            }
        
        if "Language" in self.internal_state.component_states:
            language_state = self.internal_state.component_states["Language"]
            snapshot["components"]["language"] = {
                "vocabulary_size": language_state.get("vocabulary_size", 0),
                "recent_production": language_state.get("production", {}).get("utterance"),
                "comprehension_level": language_state.get("comprehension", {}).get("comprehension_level", 0)
            }
        
        if "Memory" in self.internal_state.component_states:
            memory_state = self.internal_state.component_states["Memory"]
            snapshot["components"]["memory"] = {
                "focus": memory_state.get("focus"),
                "working_memory_items": memory_state.get("working_memory_items", []),
                "recent_memories": memory_state.get("recent_memories", [])
            }
        
        if "Consciousness" in self.internal_state.component_states:
            consciousness_state = self.internal_state.component_states["Consciousness"]
            snapshot["components"]["consciousness"] = {
                "consciousness_state": consciousness_state.get("consciousness_state"),
                "attention_focus": consciousness_state.get("attention_focus"),
                "awareness_state": consciousness_state.get("awareness_state", {}),
                "self_concept": consciousness_state.get("self_concept", {})
            }
        
        return snapshot